import os
import logging
from typing import Dict
from datetime import datetime, timezone, date, timedelta

from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros

from object_definitions.Config_def import Config
from object_definitions.SimData_def import (SpaceWeatherDay, SPACE_WEATHER_DATA_FILE_PATH)

MSIS_SW_KEYS: list[str] = [
    "ap_24_0",      # 24 hour ap avg. ending now
    "ap_3_0",       # 3 hour ap avg. ending now
    "ap_3_-3",      # 3 hour ap avg. ended 3 hours ago
    "ap_3_-6",      # 3 hour ap avg. ended 6 hours ago
    "ap_3_-9",      # etc.
    "ap_3_-12",
    "ap_3_-15",
    "ap_3_-18",
    "ap_3_-21",
    "ap_3_-24",
    "ap_3_-27",
    "ap_3_-30",
    "ap_3_-33",
    "ap_3_-36",
    "ap_3_-39",
    "ap_3_-42",
    "ap_3_-45",
    "ap_3_-48",
    "ap_3_-51",
    "ap_3_-54",
    "ap_3_-57",
    "f107_1944_0",   # 81-day avg of f107adj
    "f107_24_-24",   # previous day's f107adj
]


class MsisInputUpdater(sysModel.SysModel):
    """
    =========================================================================================================
    ATTRIBUTES:
        spaceWeatherData    (Dict[date, SpaceWeatherDay]) Contains space weather parameters 
                                from date(cfg.startTime-81days) to date(cfg.startTime + simulationDuration hours)
        _simStartDt         (datetime) Simulation start time helper
        _simEndDt           (datetime) Simulation end time helper
    =========================================================================================================
    """
    def __init__(self, cfg: Config, sw_writers: list[messaging.SwDataMsg]):
        super().__init__()

        # Configure update of MSIS input parameters every XXX hours
        updateIntervalHour = 3
        self.updateIntervalNanos = macros.hour2nano(updateIntervalHour)
        self.nextUpdateNanos = 0 

        # Set simulation start and end datetime objects, and load space weather data 
        self._simStartDt = datetime.strptime(cfg.startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        self._simEndDt = self._simStartDt + timedelta(hours=float(cfg.simulationDuration))
        self.sw_writers = sw_writers
        self.spaceWeatherData = self._load_space_weather_data()


    def UpdateState(self, CurrentSimNanos: int) -> None:
        
        # When it is time to update MSIS input parameters
        # If sim jumps over multiple 3 hour bins, catch up (while)
        while CurrentSimNanos >= self.nextUpdateNanos:

            # Get the MSIS inputs for the current 3 hour bin
            msisInputDict = self._get_msis_inputs(CurrentSimNanos)

            # Apply updated MSIS inputs
            self._apply_msis_inputs(msisInputDict)

            # Calculate when the next MSIS input update should be in nanos
            self.nextUpdateNanos += self.updateIntervalNanos
        

    def _load_space_weather_data(self) -> Dict[date, SpaceWeatherDay]:
        """
        Parse space weather data from SPACE_WEATHER_DATA_FILE_PATH once and store a local database 
        for fast queries during runtime. The method will load data in range:
            from date(cfg.startTime - 81days) to date(cfg.startTime + simulationDuration hours)
        And will raise an error if the data file does not exist OR if the data does not cover the desired range

        Uses:
            self.cfg.startTime: "dd.mm.yyyy hh:mm:ss" (UTC)
            self.cfg.simulationDuration: hours (float/int)

        Creates:
            self.spaceWeatherData: Dict[date, SpaceWeatherDay]
        """
        # Parse sim time window
        start_dt = self._simStartDt
        end_dt = self._simEndDt

        # Need history for F10.7A (81-day average). Load with margin.
        load_start = start_dt.date() - timedelta(days=81) - timedelta(days=1)# -1 day buffer for edge cases
        load_end = end_dt.date() + timedelta(days=1)  # +1 day buffer for edge cases

        # Define path to space weather data file and ensure its existance
        sw_path = SPACE_WEATHER_DATA_FILE_PATH

        if not os.path.isfile(sw_path):
            raise FileNotFoundError(
                f"Space weather file not found at '{sw_path}'."
            )

        # Parse file
        data: Dict[date, SpaceWeatherDay] = {}

        with open(sw_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                # Expecting part indices to correspond to the following fields:
                #   0 y,1 m,2 d, 3 days,4 days_m,5 BSR,6 dB,
                #   7..14 Kp1..Kp8,
                #   15..22 ap1..ap8,
                #   23 Ap, 24 SN, 25 f107obs, 26 f107adj, 27 D
                if len(parts) < 28:
                    continue  # defensively skip malformed lines

                y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
                day_key = date(y, m, d)

                # Filter to the required window only (saves memory and speeds up lookup)
                if day_key < load_start or day_key > load_end:
                    continue

                ap_bins = [int(x) for x in parts[15:23]]
                Ap = int(parts[23])
                f107obs = float(parts[25])
                f107adj = float(parts[26])

                day_data = SpaceWeatherDay(
                    ap_bins,
                    Ap,
                    f107obs,
                    f107adj
                )
                data[day_key] = day_data

        if not data:
            raise ValueError(
                f"No space weather data loaded from {sw_path} within {load_start}..{load_end}."
            )
        
        # Ensure the exact requested coverage has been loaded.
        required_days = (load_end - load_start).days + 1
        missing = [
            load_start + timedelta(days=i)
            for i in range(required_days)
            if (load_start + timedelta(days=i)) not in data
        ]
        if missing:
            raise ValueError(
                "Space weather file does not cover the full required date range. "
                f"Missing {len(missing)} day(s); first missing: {missing[0]}, last missing: {missing[-1]}."
            )

        logging.debug(f"[MSIS] Space weather parameters has been parsed and loaded in range {load_start}..{load_end}")
        return data
    

    def _get_msis_inputs(self, sim_time_ns: int) -> Dict[str, float]:
        """
        Compute the 23 MSIS space-weather inputs for the *current* 3-hour UTC bin.

        Args:
            sim_time_ns (int): Basilisk-style simulation time in nanoseconds since simulation start epoch.

        Returns:
            Dict[str, float] keyed by MSIS_SW_KEYS (23 entries).
        """
        if not hasattr(self, "spaceWeatherData"):
            raise RuntimeError("spaceWeatherData not loaded. Call load_space_weather_data() first.")

        # Convert sim time -> UTC datetime (Basilisk time is typically ns)
        now_dt = self._simStartDt + timedelta(seconds=float(sim_time_ns) * macros.NANO2SEC)

        def ap_at(dt_utc: datetime) -> int:
            """Return ap for the 3-hour bin containing dt_utc."""
            day = dt_utc.date()
            rec = self.spaceWeatherData.get(day)
            if rec is None:
                raise ValueError(f"No space weather data for date {day}. Loaded range {min(self.spaceWeatherData.keys())}..{max((self.spaceWeatherData.keys()))}.")
            bin_idx = int(dt_utc.hour // 3)  # 0..7
            return int(rec.ap[bin_idx])

        # ap history at 3-hour resolution:
        # ap_3_0 is current bin; ap_3_-3 is previous bin; ... ap_3_-57 is 19 bins back.
        ap_hist: list[int] = []
        for k in range(0, 20):  # 0..19 => 20 bins => 0, -3, -6, ..., -57 hours
            ap_hist.append(ap_at(now_dt - timedelta(hours=3 * k)))

        # ap_24_0: average of the last 8 bins (24 hours) including current bin
        ap_24_0 = float(sum(ap_hist[0:8])) / 8.0

        # f107_24_-24: previous day's adjusted F10.7
        prev_day = now_dt.date() - timedelta(days=1)
        prev_rec = self.spaceWeatherData.get(prev_day)
        if prev_rec is None:
            raise ValueError(f"No space weather data for previous day {prev_day} needed for f107_24_-24.")
        f107_24_m24 = float(prev_rec.f107adj)

        # f107_1944_0: last 81 day average adjusted f107
        d0 = now_dt.date()
        window_days = [d0 - timedelta(days=i) for i in range(0, 81)]
        f107_window = [float(self.spaceWeatherData[d].f107adj) for d in window_days]
        f107_81avg = float(sum(f107_window)) / float(len(f107_window))

        # Build output in a stable, explicit way (so ordering never depends on dict insertion)
        out: Dict[str, float] = {}
        out["ap_24_0"] = ap_24_0
        out["ap_3_0"] = float(ap_hist[0])
        out["ap_3_-3"] = float(ap_hist[1])
        out["ap_3_-6"] = float(ap_hist[2])
        out["ap_3_-9"] = float(ap_hist[3])
        out["ap_3_-12"] = float(ap_hist[4])
        out["ap_3_-15"] = float(ap_hist[5])
        out["ap_3_-18"] = float(ap_hist[6])
        out["ap_3_-21"] = float(ap_hist[7])
        out["ap_3_-24"] = float(ap_hist[8])
        out["ap_3_-27"] = float(ap_hist[9])
        out["ap_3_-30"] = float(ap_hist[10])
        out["ap_3_-33"] = float(ap_hist[11])
        out["ap_3_-36"] = float(ap_hist[12])
        out["ap_3_-39"] = float(ap_hist[13])
        out["ap_3_-42"] = float(ap_hist[14])
        out["ap_3_-45"] = float(ap_hist[15])
        out["ap_3_-48"] = float(ap_hist[16])
        out["ap_3_-51"] = float(ap_hist[17])
        out["ap_3_-54"] = float(ap_hist[18])
        out["ap_3_-57"] = float(ap_hist[19])
        out["f107_1944_0"] = f107_81avg
        out["f107_24_-24"] = f107_24_m24

        # Optional sanity check: ensure we return exactly the expected keyset
        if set(out.keys()) != set(MSIS_SW_KEYS):
            missing = [k for k in MSIS_SW_KEYS if k not in out]
            extra = [k for k in out.keys() if k not in MSIS_SW_KEYS]
            raise RuntimeError(f"MSIS inputs key mismatch. Missing={missing}, Extra={extra}")

        ########### DEBUG ###########
        # print(f"""[MsisInputUpdater] All MSIS inputs at offset: {float(sim_time_ns) * macros.NANO2HOUR}, date: ({now_dt})
        #            ap_24_0     = {out["ap_24_0"]},      (old: {self.sw_writers[0].read().dataValue})
        #            ap_3_0      = {out["ap_3_0"]},       (old: {self.sw_writers[1].read().dataValue})
        #            ap_3_-3     = {out["ap_3_-3"]},      (old: {self.sw_writers[2].read().dataValue})
        #            ap_3_-6     = {out["ap_3_-6"]},      (old: {self.sw_writers[3].read().dataValue})
        #            ap_3_-9     = {out["ap_3_-9"]},      (old: {self.sw_writers[4].read().dataValue})
        #            ap_3_-12    = {out["ap_3_-12"]},     (old: {self.sw_writers[5].read().dataValue})
        #            ap_3_-15    = {out["ap_3_-15"]},     (old: {self.sw_writers[6].read().dataValue})
        #            ap_3_-18    = {out["ap_3_-18"]}
        #            ap_3_-21    = {out["ap_3_-21"]}
        #            ap_3_-24    = {out["ap_3_-24"]}
        #            ap_3_-27    = {out["ap_3_-27"]}
        #            ap_3_-30    = {out["ap_3_-30"]}
        #            ap_3_-33    = {out["ap_3_-33"]}
        #            ap_3_-36    = {out["ap_3_-36"]}
        #            ap_3_-39    = {out["ap_3_-39"]}
        #            ap_3_-42    = {out["ap_3_-42"]}
        #            ap_3_-45    = {out["ap_3_-45"]}
        #            ap_3_-48    = {out["ap_3_-48"]}
        #            ap_3_-51    = {out["ap_3_-51"]}
        #            ap_3_-54    = {out["ap_3_-54"]}
        #            ap_3_-57    = {out["ap_3_-57"]}
        #            f107_1944_0 = {out["f107_1944_0"]}
        #            f107_24_-24 = {out["f107_24_-24"]}""")
        return out
    

    def _apply_msis_inputs(self, msis_inputs: Dict[str, float]) -> None:
        """
        Publish updated MSIS inputs to the 23 SwData messages in the correct order

        Args:
            msis_inputs (Dict[str, float]): Updated MSIS model inputs for the current 3 hour bin

        Returns:
            None
        """
        for i, key in enumerate(MSIS_SW_KEYS):
            payload = messaging.SwDataMsgPayload(dataValue=float(msis_inputs[key]))
            self.sw_writers[i].write(payload)