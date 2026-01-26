import os
import shutil # For therminal width
import logging
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import Optional
from datetime import datetime, timedelta, timezone

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimObjData, SimData
from object_definitions.TLE_def import TLE
from plotting.DataLoader_def import DataLoader

from skyfield.timelib import Time
from skyfield.api import EarthSatellite, load


class SkyfieldSimulator():
    """
    =========================================================================================================
    ATTRIBUTES:
        cfg                         (Config)
        times                       (list[Time]) List of UTC Skyfield simulation times
        sim_offset                  (NDArray[np.float64]) Corresponding Matplotlib-compatible time vector
        sat_tle_idx_at_times        (list[int]) Indeces giving which satellite/TLE to use at any simulation time
        all_skf_satellite_series    (list[list[EarthSatellite]]) Contains a series of EarthSatellite objects for each satellite
        satellite_names             (list[str]) Satellite names/identifiers
        sim_data                    (Optional[SimData]) Skyfield simulation output data
    =========================================================================================================
    """

    def __init__(self, cfg: Config) -> None:
        logging.debug("Setting up Skyfield simulation...")

        ###################################
        # Configure simulation parameters #
        ###################################
        startTime, duration, deltaT = self.get_skf_simulation_time(cfg) # TODO: modify this function to get startTime from oldest TLE in shared_input_data/tle_files/HYPSO-1_tle_series.txt


        ########################
        # Populate time-vector #
        ########################
        times: list[Time] = []
        ts = load.timescale()
        for t in range(0, duration, deltaT):
            time = ts.utc(startTime + timedelta(seconds=t))
            times.append(time)

        # Convert to matplotlib-compatible simulation offset
        sim_offset = self.skf_time_to_offset(times, startTime)


        #################################
        # Verify and Create TLE file(s) #
        #################################
        leader_tle_path = Path(cfg.leader_tle_series_path)
        tle_processor = TLE()
        tle_processor.verify_leader_TLE_series_file(leader_tle_path)

        # Generate a TLE file for each follower and get their output paths
        tle_paths = tle_processor.generate_follower_tle_files(cfg)

        # Insert the leader TLE file to get a comprehensive list of all TLE file paths
        tle_paths.insert(0, cfg.leader_tle_series_path)


        #######################################################
        # Load TLE files to create Skyfield satellite objects #
        #######################################################
        # NOTE: Each TLE file will generate a series of 'EarthSatellite' objects.
        #       These describe the orbit of the same satellite, but with different epochs
        all_skf_satellite_series: list[list[EarthSatellite]] = []
        satellite_names: list[str] = []

        for i, tle_path in enumerate(tle_paths):        
            # Generate a series of Skyfield EarthSatellite objects for each satellite
            skf_satellite_series = load.tle_file(tle_path)
            all_skf_satellite_series.append(skf_satellite_series)
            
            # Get satellite names/identifiers
            satellite_name = skf_satellite_series[0].name
            if isinstance(satellite_name, str):
                if satellite_name in satellite_names:
                    raise ValueError(f"The satellite name: {satellite_name} already exists in 'satellite_names'")
                satellite_names.append(satellite_name)
            
        # Raise error if the number of satellite series is not as expected
        if not len(all_skf_satellite_series) == cfg.num_satellites:
            raise ValueError("The number of Skyfield EarthSatellite series does not match the expected number of satellites defined by default.yaml")


        #########################################################################################
        # Pre-determine which TLE in TLE-series to use at any given time in the simulation span #
        #########################################################################################
        tle_epoch_series: list[Time] = []

        # The TLE files are ensured to have the same Epochs through verification loops in 'generate_follower_tle_files(.)', 
        # so the choice of skf satellite series doesn't really matter. The leader is selected beacuse element 0 is always defined.
        leader_skf_satellite_series = all_skf_satellite_series[0]

        # Extract epoch series from a series of EarthSatellite objects
        for i, skf_sat in enumerate(leader_skf_satellite_series):
            tle_epoch_series.append(skf_sat.epoch)

        # Pre-determine TLE indecies at any given time
        sat_tle_idx_at_times = self.generate_sat_tle_idx_at_times(times, tle_epoch_series)


        ####################################
        # Set SkyfieldSimulator attributes #
        ####################################
        self.cfg: Config = cfg
        self.times: list[Time] = times
        self.sim_offset: NDArray[np.float64] = sim_offset
        self.sat_tle_idx_at_times: list[int] = sat_tle_idx_at_times
        self.all_skf_satellite_series: list[list[EarthSatellite]] = all_skf_satellite_series
        self.satellite_names: list[str] = satellite_names
        self.sim_data: Optional[SimData] = None
        # self.skf_satellites: list[EarthSatellite] = skf_satellites
        
        logging.debug("Skyfield simulation setup complete")


    def run(self) -> None:

        logging.debug("Running the Skyfield simulation...")

        #################################################################################
        # Run sgp4 propagation to get position and velocity at all simualtion timesteps #
        #################################################################################
        sim_data: list[SimObjData] = []
        n_t = len(self.times)
        total_steps = len(self.all_skf_satellite_series) * n_t
        current_step = 0

        for i, skf_sat_series in enumerate(self.all_skf_satellite_series):
            # TODO:
            # Add checks to ensure we are iterating on the correct satellite
            # Verify that we are iterating over the satellites in the correct order
            # if not isinstance(skf_sat.name, str):
            #     try:
            #         skf_sat_name = str(skf_sat.name)
            #     except:
            #         raise TypeError(f"skf_sat.name is of type {type(skf_sat.name)}, and couldn't be converted to str")
            # else: 
            #     skf_sat_name = str(skf_sat.name)
            
            # if not skf_sat_name == satellites[i].name:
            #     raise NameError(f"There is a mismatch between the skf_satellites and the cfg.satellites. Satellite nr. {i} in skf_satellites is named {skf_sat.name} while the corresponding satellkite in cfg.satellites is named {satellites[i].name}")
            
            skf_sat_name = self.satellite_names[i]

            # Pre-allocate for state matrices 
            positions_eci = np.empty((len(self.times), 3), dtype=float)
            velocities_eci = np.empty((len(self.times), 3), dtype=float)

            # Getting states in ECI (ICRF) frame
            for j, t in enumerate(self.times):
                # Chose correct satellite in the satellite series
                tle_idx = self.sat_tle_idx_at_times[j]
                skf_sat = skf_sat_series[tle_idx]
                
                # Get states
                positions_eci[j, :] = skf_sat.at(t).position.m
                velocities_eci[j, :] = skf_sat.at(t).velocity.m_per_s

                current_step += 1
                self.print_progress(current_step, total_steps)

            sim_object_data = SimObjData(
                skf_sat_name,
                self.sim_offset,
                positions_eci.T,
                velocities_eci.T,
            )

            sim_data.append(sim_object_data)

        # Set SkyfieldSimulator attribute sim_data
        self.sim_data = SimData(sim_data)

        # Write simulation data to file
        self.output_data()

        logging.debug("Skyfield simulation complete")


    def output_data(self) -> None:
        """
        Write the simulation data to a file named '<cfg.timestamp_str>_skf.h5' stored in output_data/sim_data/
        """
        
        # Check that simulation data has been stored
        if self.sim_data is None:
            raise ValueError("Simulation data not yet generated. Call skf.run() before skf.output_data().")
        
        # Log data to file
        self.sim_data.write_data_to_file(self.cfg.timestamp_str, "skf")


    def extract_initial_states_and_update_satellites(self, cfg: Config) -> None:
        """
        Extracts the initial states for each satellite in self.sim_data, 
        and calls a function to update their corresponding Satellite object in config.satellites
        """

        if cfg.use_old_skf_data:
            logging.debug("Skipping the Skyfield simulation. Old Skyfield simulation data will be loaded instead...")

            # Initialize data loader and processor objects
            data_loader = DataLoader()

            # Get the old skyfield datafile name 
            search_str = cfg.old_skf_data_timestamp + "_skf"
            matching_datafile_names = data_loader.get_datafiles_by_timestamp(search_str)
            if len(matching_datafile_names) > 1:
                raise ValueError(f"Found more than one match from the old skyfield data timestamp. Matches found: {matching_datafile_names}")
            elif len(matching_datafile_names) == 0:
                raise ValueError(f"Found no old sSkyfield datafile name matches for: {search_str}")
            old_skf_data_filename = matching_datafile_names[0]

            # Load old Skyfield data
            old_skf_sim_data = data_loader.load_sim_data_file(old_skf_data_filename)

            # set sim_data attribute 
            self.sim_data = old_skf_sim_data

            # Write simulation data to file
            self.output_data()


        # Verify that simulation data has been stored in self.sim_data
        if self.sim_data == None:
            raise ValueError("No simulation data has been stored in SkyfieldSimulation.sim_data")
        
        # Extract initial state for each of the satellites, and update the initial conditions for
        # their corresponding Satellite object in config.satellites
        for i, sat in enumerate(self.sim_data.sim_data):
            cfg.satellites[i].extract_initial_states_and_update(sat)


    @staticmethod
    def generate_sat_tle_idx_at_times(times: list[Time], tle_epoch_series: list[Time]) -> list[int]:
        """
        Generates a list that provides the TLE index at any time in times_i. \n
        The index at time times_i[i] is chosen to correspond to the TLE with the Epoch closest to the given time
        
        Args:
            tle_epoch_series (list[Time]): A list containing the Epochs extracted from the leader satellite's TLE file found in cfg.leader_tle_series_path

        Returns:
            sat_tle_idx_at_times (list[int]): A list that provides the index of the TLE that should be used at time times_i[i]
        """
        # Calculate midpoints and ensure tle_epoch_series is sorted in ascending order
        num_epochs = len(tle_epoch_series)
        midpoints: list[float] = []
        for i in range(0, num_epochs-2):
            curr_tle_epoch = tle_epoch_series[i]
            next_tle_epoch = tle_epoch_series[i+1]

            if not curr_tle_epoch.tt < next_tle_epoch.tt:
                raise ValueError("'tle_epoch_series' is not sorted in ascending order. Indicates that the leader's TLE file in 'cfg.leader_tle_series_path' is not sorted by epoch.")
            
            midpoint = (curr_tle_epoch.tt + next_tle_epoch.tt) / 2
            midpoints.append(midpoint)

        # Generate list of which TLE index to use at any given time
        sat_tle_idx_at_times: list[int] = []
        sat_tle_idx = 0
        next_midpoint_idx = 0
        next_midpoint = midpoints[next_midpoint_idx]
        for i, t in enumerate(times):
            if t.tt >= next_midpoint:
                next_midpoint_idx += 1
                next_midpoint = midpoints[next_midpoint_idx]
                sat_tle_idx += 1
                sat_tle_idx_at_times.append(sat_tle_idx)
            else:
                sat_tle_idx_at_times.append(sat_tle_idx)

        # Verify length
        if not len(sat_tle_idx_at_times) == len(times):
            raise ValueError("'sat_tle_idx_at_times' not the same length as 'times_i'")
        
        logging.debug("TLE series indecies pre-determined for all times in the simulation duration")

        return sat_tle_idx_at_times


    @staticmethod
    def get_skf_simulation_time(cfg: Config) -> tuple[datetime, int, int]:
        cfg_startTime = cfg.startTime
        cfg_duration = cfg.simulationDuration
        cfg_deltaT = cfg.s_set.deltaT
        """
        Parses and converts simulation time parameters from default config into skyfield-compatible types.

        RETURNS:
            skf_startTime       (datetime) utc time
            skf_duration        (float) seconds
            skf_deltaT          (int) seconds
        """

        # Parse simulation starttime and convert into a UTC datetime object
        try:
            skf_startTime = datetime.strptime(cfg_startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        except:
            raise ValueError("Failed to convert config parameter 'startTime' to a datetime object.")
        
        # Convert cfg_duration float(hours) -> int(seconds)
        skf_duration = int(3600 * cfg_duration) 
        if skf_duration < (3600 * cfg_duration):
            raise ValueError("Type conversion float -> int for 'skf_duration' caused a reduction in its value!")

        # deltaT
        skf_deltaT = int(cfg_deltaT)
        if skf_deltaT < cfg_deltaT:
            raise ValueError("Type conversion float -> int for 'skf_deltaT' caused a reduction in its value!")
        
        return skf_startTime, skf_duration, skf_deltaT
    

    @staticmethod
    def skf_time_to_offset(skf_time_vec: list[Time],
                           simulation_startTime: datetime
                           ) -> NDArray[np.float64]:
        """
        Converts a list of Skyfield Time objects into simulation time offsets [seconds]
        relative to the simulation start time.

        INPUTS:
            skf_time_vec (list[skyfield.timelib.Time]):
            List of Skyfield Time objects representing simulation timestamps.
            simulation_startTime (datetime):
            UTC datetime representing the start of the simulation.

        RETURNS:
            simulation_offset (np.ndarray):
            1D array of time offsets [seconds], where 0 = start time.
        """

        # Ensure start time is timezone-aware in UTC
        if simulation_startTime.tzinfo is None:
            simulation_startTime = simulation_startTime.replace(tzinfo=timezone.utc)

        # Convert Skyfield times to UTC datetimes
        utc_datetimes = [t.utc_datetime() for t in skf_time_vec]

        # Compute offset (in seconds) from simulation start
        simulation_offset = np.array([
            (dt - simulation_startTime).total_seconds() for dt in utc_datetimes
        ])

        return simulation_offset
    

    @staticmethod
    def print_progress(step: int, total: int, bar_len: int = 40) -> None:
        """
        Simple text-based progress bar.

        Parameters
        ----------
        step : int
            Current completed step (1-based).
        total : int
            Total number of steps.
        prefix : str
            Text shown before the progress bar.
        bar_len : int
            Character width of the bar.
        """
        yellow = "\033[33m"
        reset = "\033[0m"
        hide_cursor = "\033[?25l"
        show_cursor = "\033[?25h"
        prefix = "Progress: "
        cursor_hidden = False

        # Hide cursor on first call
        if not cursor_hidden:
            print(hide_cursor, end="", flush=True)
            cursor_hidden = True

        if total <= 0:
            frac = 1.0
        else:
            frac = step / total

        frac = max(0.0, min(1.0, frac))  # clamp to [0, 1]
        percent = int(frac * 100.0)

        # Terminal width
        width = shutil.get_terminal_size().columns

        # Reserve space for: prefix, ':  ', percent text '100.00%', two spaces
        reserved = len(prefix) + 2 + len("100%")  # small safety margin
        bar_len = max(10, width - reserved)  # prevent too-small bar

        # Generate bar
        filled = int(bar_len * frac)
        bar = "█" * filled

        # Color only the bar + percent in yellow
        colored_prefix = f"{yellow}{prefix}{reset}"
        colored_bar = f"{yellow}|{bar}{reset}"
        colored_percent = f"{yellow}{percent:3d}%{reset}"

        print(f"\r{colored_prefix}{colored_percent}{colored_bar} ", end="", flush=True)

        # Show cursor again at 100%
        if percent >= 100:
            print(show_cursor, end="", flush=True)
            cursor_hidden = False
            print()  # newline at end