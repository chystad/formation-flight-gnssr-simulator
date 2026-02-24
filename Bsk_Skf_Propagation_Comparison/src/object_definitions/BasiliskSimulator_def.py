import os
import logging
import numpy as np
import matplotlib.pyplot as plt # Only for debug
import matplotlib.colors as mcolors # only for debug
from typing import Optional, Any, Union, Dict
from numpy.typing import NDArray
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import (spacecraft, radiationPressure, spiceInterface, eclipse,  
                                exponentialAtmosphere, msisAtmosphere, dragDynamicEffector, svIntegrators)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport)

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimData, SimObjData
from plotting.plot import PLT_WIDTH, PLT_HEIGHT


EARTH_RADIUS = 6378136.6 # [m] WGS-84 equatorial radius
VIZARD_SAVE_PATH = "/home/chris/code/formation-flight-gnssr-simulator/Bsk_Skf_Propagation_Comparison/output_data/_VizFiles/bsk_sim.bin"
GRAV_COEFF_FILE_PATH = "shared_input_data/grav_coeff/GGM03S.txt"
SPACE_WEATHER_DATA_FILE_PATH = "shared_input_data/msis_data/Kp_ap_Ap_SN_F107_since_2010.txt"
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


@dataclass_json
@dataclass
class SpaceWeatherDay:
    """One UTC day of space-weather data from Kp_ap_Ap_SN_F107_since_2010.txt."""
    ap: list[int]        # 8x 3-hour ap values: [00-03, 03-06, ..., 21-24]
    Ap: int              # daily Ap
    f107obs: float       # adjusted F10.7
    f107adj: float       # observed F10.7


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

        logging.debug(f"[BSK] Space weather parameters has been parsed and loaded in range {load_start}..{load_end}")
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


class BasiliskSimulator:
    """
    =========================================================================================================
    ATTRIBUTES:
        cfg                 (Config) Global config instance 
        integrators         (list[svIntegrators.Any]) List containing the numerical integrator used
                                to propagate each spacecraft's states 
        simTaskName         (str) Simulation task name 
        scSim               (SimBaseClass) Simulation module container
        dynProcess          (ProcessBaseClass) Simulation process 
        scObjects           (list[Spacecraft]) List containing all simulation objects
        scRecorders         (list[]) List containing all simulation recorders (one for each scObject)
        msisInputUpdater    (MsisInputUpdater(SysModel)) Separate task for updating 
                                MSIS input parameters during simulation execution
        spiceTime           (str) Time string used to initialize the SpiceInterface
        epochMsg            (messaging.EpochMsg) Centralized epoch message used by all models
        spaceWeatherData    (Dict[date, SpaceWeatherDay]) Contains space weather parameters 
                                from date(cfg.startTime-81days) to date(cfg.startTime + simulationDuration hours)
        sim_data            (Optional[SimData]) Object containing the simulaton output data 
        _simStartDt         (datetime) Simulation start time helper
        _simEndDt           (datetime) Simulation end time helper
    =========================================================================================================
    """
    def __init__(self, cfg: Config) -> None:
        logging.debug("[BSK] Setting up Basilisk simulation...")
        
        ###############
        # Load config #
        ###############
        
        self.cfg = cfg     # Assign config to self.cfg attribute
        d_set = cfg        # default config
        b_set = cfg.b_set  # basilisk config
        

        ###################################
        # Configure simulation parameters #
        ###################################

        # Set Simulation time
        self.spiceTime = self.to_spice_utc(self.cfg.startTime)   # Only used to initialize SPICE interface
        self.epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(self.spiceTime)   # Used for time-dependent models (SPICE interface (eclipse model by extension), MSIS)
        
        # Helper simulation time datetime objects used by other methods
        self._simStartDt = datetime.strptime(self.cfg.startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        self._simEndDt = self._simStartDt + timedelta(hours=float(self.cfg.simulationDuration))

        # Set fixed simulation integration time step
        simulationTimeStep = macros.sec2nano(b_set.deltaT)

        # Set simulation duration
        simualtionDuration_sec = d_set.simulationDuration * 60 * 60
        simulationDuration = macros.sec2nano(simualtionDuration_sec)

        # Set number of data points
        numDataPoints = simulationDuration // simulationTimeStep

        # Set sample time (same as 'deltaT' in basilisk simulation config)
        samplingTime = unitTestSupport.samplingTime(simulationDuration, simulationTimeStep, numDataPoints)

        # Initialize integrator list to prevent it being CE'ed
        self.integrators = []

        # Create a stable list of publishers (writers) in the exact MSIS order
        self.msisSwWriters: list[messaging.SwDataMsg] = []
        self.msisSwMsgs: list[messaging.SwDataMsg] = []  # optional: store the published m

        # path to basilisk. Used to fetch predesigned models
        bskPath = __path__[0]
        fileName = os.path.basename(os.path.splitext(__file__)[0])

        
        ######################################
        # Set up simulation task and process #
        ######################################

        # Initialize sim_data attribute
        self.sim_data = None
        
        # Select task and process names
        self.simTaskName = "simTask"
        simProcessName = "simProcess"

        # Create a sim module as an empty container
        self.scSim = SimulationBaseClass.SimBaseClass()

        # Configure the use of simulation progress bar
        self.scSim.SetProgressBar(True)

        # Create the simulation process. Will contain all scheduled tasks
        self.dynProcess = self.scSim.CreateNewProcess(simProcessName)

        # create the simulation task with a simulation time step, and give it 0 priority (will be executed last)
        self.dynProcess.addTask(self.scSim.CreateNewTask(self.simTaskName, simulationTimeStep), 0)
        

        ######################################################################
        # Initialize planets according to config and configure their gravity #
        ######################################################################
        gravFactory, spiceObj = self.conditional_planet_gravity_generation()


        ######################################################
        # Initialize Eclipse Model (Earth eclipsing the Sun) #
        ######################################################
        sunMsg, eclipseObj = self.conditional_eclipse_init(spiceObj)


        ##############################################
        # Initialize Earth Exponential Density Model #
        ##############################################
        # Initialize the exponential density atmosphere model iff b_set.useExponentialDensityDrag == True
        atm = self.conditional_atmosphere_init()


        #################################################################
        # Initialize scObjects and scRecorders, and attach force models #
        #################################################################
        
        # Initialize empty containers for to-be-defined Spacecraft objects and its recorders
        self.scObjects: list[spacecraft.Spacecraft] = []
        self.scRecorders: list = [] # list of what?
        self.atmRecorders: list = []

        # get satellites from config
        satellites = self.cfg.satellites

        # Define all satellite parameters, attach all applied forces, and make it part of the 
        #################################################################################################
        # Define all spacecraft objects, attach all force models, add it and recorders to the simulator #
        #################################################################################################
        for i, sat in enumerate(satellites):
            # Initialize spacecraft object
            scObj = spacecraft.Spacecraft()
            scObj.ModelTag = sat.name
            scObj.hub.mHub = sat.m_s # getattr(sat, "m_s", 6.0)

            # Add spacecraft object to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scObj)

            if b_set.override_skf_initial_state:
                # Get initial conditions corresponding to satellites separated by an arbitrary angle 
                # in the same orbital plane
                separationAng = self.cfg.inplane_separation_ang * macros.D2R
                rN, vN = self.spaced_satellites_on_same_orbital_plane(i, separationAng, gravFactory.gravBodies["earth"].mu)

                # Edit and uncomment this function to use user-defined initial states:
                # rN, vN = self.custom_initial_states(i)
                
            else: # Default case
                # Use initial state calculated by Skyfield SGP4 at simulation offset 0 seconds
                rN = sat.init_pos # [m]   In N frame (inertial = ECI)
                vN = sat.init_vel # [m/s] in N frame (inertial = ECI)

            # Set the initial conditions for the spacecraft object
            scObj.hub.r_CN_NInit = rN  # m   - r_BN_N
            scObj.hub.v_CN_NInit = vN  # m/s - v_BN_N
            
            
            # ---- Main graviational attraction, Spherical Harmonics and 3rd body perturbation ----
            # The gravitational sources and models have already been defined gravFactory in accordance with cfg
            gravFactory.addBodiesTo(scObj)
            

            # ---- Drag effector (exponential density + cannonball) ----
            scObj = self.conditional_drag_effector(sat, scObj, atm)
            
            
            # ---- SRP effector (cannonball) ----
            # Register this spacecraft with the eclipse model to get its own eclipse msg
            scObj = self.conditional_srp_effector(sat, scObj, sunMsg, eclipseObj)

            
            # ---- Set object integration method ----
            scObj = self.conditional_object_integrator(scObj)
            
           
            # ---- Define and append scRecorders and scObjects ----
            # Create object state and force recorders
            scRec = scObj.scStateOutMsg.recorder(samplingTime)
            assert atm is not None
            atmLog = atm.envOutMsgs[i].recorder(samplingTime)
            # srpRec = self.make_srp_recorder(srp, samplingTime)  

            # Add recorder to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scRec)
            self.scSim.AddModelToTask(self.simTaskName, atmLog)
            # self.scSim.AddModelToTask(self.simTaskName, srpRec)
                        
            # Append defined spacecraft object and scRec to scObjects and scRecorders, respectively
            self.scObjects.append(scObj)
            self.scRecorders.append(scRec)
            self.atmRecorders.append(atmLog)
            # self.srpRecorders.append(srpRec)       


        # Output Vizard .bin file
        viz = vizSupport.enableUnityVisualization(self.scSim, self.simTaskName, self.scObjects,
                                                saveFile=VIZARD_SAVE_PATH
                                                # liveStream=True
                                                )


        # initialize Simulation:  This function runs the self_init()
        # and reset() routines on each module.
        self.scSim.InitializeSimulation()

        # Configure a simulation stop time
        self.scSim.ConfigureStopTime(simulationDuration)
        
        

    def run(self) -> None:
        # Execute the simulation
        logging.debug("[BSK] Running Basilisk simulation...")
       
        self.scSim.ExecuteSimulation()
        # Note that this module simulates both the translational and rotational motion of the spacecraft.
        # In this scenario only the translational (i.e. orbital) motion is tracked.  This means the rotational motion
        # remains at a default inertial frame orientation in this scenario.  There is no appreciable speed hit to
        # simulate both the orbital and rotational motion for a single rigid body.

        # Make configs easily accessible
        d_set = self.cfg        # default config
        b_set = self.cfg.b_set  # basilisk config

        satellites = d_set.satellites
        simulationDuration_sec = d_set.simulationDuration * 60 * 60
        timeStep_sec = b_set.deltaT

        # Create time vector and ensure shape
        numSamples = int(simulationDuration_sec // timeStep_sec + 1)
        t = np.asarray(np.linspace(0, simulationDuration_sec, numSamples))
        t = t.reshape(1, -1) # is now shape: (1,n)

        # Get simulation data
        if len(satellites) != len(self.scRecorders):
            raise ValueError(f"Mismatch between the number of satellites in cfg.satellites({len(satellites)})"
                             f"and the number of trajectories in self.scRecorders ({len(self.scRecorders)})")

        sim_data: list[SimObjData] = []
        for i, recorder in enumerate(self.scRecorders):
            sat_name = satellites[i].name
            pos = np.asarray(recorder.r_BN_N)
            vel = np.asarray(recorder.v_BN_N)

            # Ensure correct dimensions for pos and vel arrays
            pos = pos.T if pos.shape[1] == 3 else pos
            vel = vel.T if vel.shape[1] == 3 else vel

            sim_object_data = SimObjData(
                sat_name,
                t,
                pos,
                vel
            )

            sim_data.append(sim_object_data)

        # Set BasiliskSimulator attribute sim_data
        self.sim_data = SimData(sim_data)

        # Write simulation data to file
        self.output_data()

        logging.debug("[BSK] Basilisk simulation complete")

        ############### DEBUG ###############
        # # Plot initial positions of the 1st satellite, the sun, the earth and the moon (if defined)
        # print("||sat1 position|| @0 [m]   =", np.linalg.norm(self.sim_data.sim_data[0].pos[:,0]))
        # print(self.sunRec)
        # sun_pos = np.asarray(self.sunRec.PositionVector)[0]
        # earth_pos = np.asarray(self.earthRec.PositionVector)[0]
        # if self.moonRec is not None: 
        #     moon_pos = np.asarray(self.moonRec.PositionVector)[0] 
        # else: moon_pos = None
        # print("||Earth position|| @0 [m] =", np.linalg.norm(earth_pos))
        # print("||Sun position|| @0 [m]   =", np.linalg.norm(sun_pos))
        # if moon_pos is not None: print("||Moon position|| @0 [m] =", np.linalg.norm(moon_pos))
        #############################################


        ############## MSIS ATM DEBUG ##############  
        # all_atm_data = self.atmRecorders
        # sat_0_dens_data = all_atm_data[0].neutralDensity
        # print(type(sat_0_dens_data))
        # print(len(sat_0_dens_data))
        # print(np.size(sat_0_dens_data))

        # self.DEBUG_plot_msis_atm_density()
        # self.DEBUG_plot_msis_atm_density_against_altitude()
        ############################################


    def output_data(self) -> None:
        """
        Output simulation data. The data will be stored in 2 separate ways and locations:
            * Vizard .bin file in <VIZARD_SAVE_PATH>
            * Simulation data .h5 file named '<cfg.timestamp_str>_bsk.h5' stored in <DATA_SAVE_FOLDER_PATH>
        """

        # Check that simulation data has been stored
        if self.sim_data is None:
            raise ValueError("Simulation data not yet generated. Call skf.run() before skf.output_data().")
        
        # Log data to file
        self.sim_data.write_data_to_file(self.cfg.timestamp_str, "bsk")


    def conditional_planet_gravity_generation(self) -> tuple[simIncludeGravBody.gravBodyFactory, spiceInterface.SpiceInterface]:
        """
        Initialize a gravBodyFactory and SPICE interface. 
        Always generate the Earth and Sun, but disable the Sun's gravity if useSun3rdBody == False. 
        The Moon is generated iff useMoon3rdBody == True. 
        Modify the Earth's gravity body to include spherical harmonics iff useSphericalHarmonics == True. 
        Always initialize SPICE interface for accurate positions for t he gravitational bodies.
        
        :param self: 
        :return: gravBodyFactory instance 'gravFactory'
        :rtype: gravBodyFactory
        :return: SpiceInterface instance 'spiceObj'
        :rtype: SpiceInterface
        """
        # Always generate earth and sun gravitational bodies 
        # (Sun also needed for eclipse model)
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth()
        sun = gravFactory.createSun()
        
        # Disable the Sun's gravity if useSun3rdBody == False
        if not self.cfg.b_set.useSun3rdBody:
            sun.mu = 0
        else:
            logging.debug("[BSK] Sun 3rd body perturbation initialized")

        # Create the Moon only if useMoon3rdBody == True
        if self.cfg.b_set.useMoon3rdBody:
            moon = gravFactory.createMoon()
            logging.debug("[BSK] Moon 3rd body perturbation initialized")
        
        # Set Earth as the central gravitational body
        earth.isCentralBody = True

        # Use spherical harmonics if useSphericalHarmonics == True
        if self.cfg.b_set.useSphericalHarmonics:
            # If extra customization is required, see the createEarth() macro to change additional values.
            earth.useSphericalHarmonicsGravityModel(
                GRAV_COEFF_FILE_PATH, 
                self.cfg.b_set.sphericalHarmonicsDegree
            )

            logging.debug(f"[BSK] Earth created with spherical harmonics gravity model of order and degree {self.cfg.b_set.sphericalHarmonicsDegree}")

            # The value 2 indicates that the first two harmonics, excluding the 0th order harmonic,
            # are included.  This harmonics data file only includes a zeroth order and J2 term.
        
        # Initialize SPICE publisher to get accurate positions of the planets defined within gravFactory. 
        spicePath = os.path.join(__path__[0], "supportData", "EphemerisData") + os.sep
        spiceKernels = ["de430.bsp", "naif0012.tls", "de-403-masses.tpc", "pck00010.tpc"]
        
        # Will always create SPICE objects "earth" and "sun". "moon" is created if useMoon3rdBody == True
        spiceObj = gravFactory.createSpiceInterface(
            path=spicePath,
            time=self.spiceTime,
            spiceKernelFileNames=spiceKernels,
            epochInMsg=True
        )
        spiceObj.zeroBase = "earth"
        spiceObj.epochInMsg.subscribeTo(self.epochMsg)
        
        # Schedule object to simualtion process
        self.scSim.AddModelToTask(self.simTaskName, spiceObj)

        logging.debug("[BSK] Spice interface initialized for all massive bodies")

        return gravFactory, spiceObj


    def conditional_atmosphere_init(self) -> Optional[Union[exponentialAtmosphere.ExponentialAtmosphere, msisAtmosphere.MsisAtmosphere]]:
        """
        Initialize and schedula an atmosphere model if it has been configured to do so by the config file:

        Priority:
            1) Initialize the MSIS model if 'cfg.b_set.useMsisDrag' == True
            2) Initialize the Exponential density model if 'cfg.b_set.useExponentialDensityDrag' == True

        Returns:
            Atmosphere instance, or None if an atmosphere model hasn't been initialized.
        """
        
        use_msis = self.cfg.b_set.useMsisDrag
        use_exp = self.cfg.b_set.useExponentialDensityDrag            

        # Using MSIS atmosphere model (NRLMSISE-00)
        if use_msis:
            # Initialize MsisAtmosphere instance
            atm = msisAtmosphere.MsisAtmosphere()
            atm.ModelTag = "msisAtm"

            # Default MSIS model inputs.
            # (Only actually valid for 01.01.2026, [00:00:00 - 03:00:00])
            sw_msg = {
                "ap_24_0": 7,   # avg of [ap1(01.01.2026),  ap2(31.12.2025)] (last 8 3-hour segments, including current 3-hour window)
                "ap_3_0": 7,    # ap1(01.01.2026)
                "ap_3_-3": 4,   # ap8(31.12.2025)
                "ap_3_-6": 4,   # ap7(31.12.2025)
                "ap_3_-9": 5,   # ap6(31.12.2025)
                "ap_3_-12": 18, # ap5(31.12.2025)
                "ap_3_-15": 6,  # ap4(31.12.2025)
                "ap_3_-18": 7,  # ap3(31.12.2025)
                "ap_3_-21": 5,  # ap2(31.12.2025)
                "ap_3_-24": 7,  # ap1(31.12.2025)
                "ap_3_-27": 4,  # ap8(30.12.2025)
                "ap_3_-30": 7,  # ap7(30.12.2025)
                "ap_3_-33": 12, # ap6(30.12.2025)
                "ap_3_-36": 15, # ap5(30.12.2025)
                "ap_3_-39": 5,  # ap4(30.12.2025)
                "ap_3_-42": 6,  # ap3(30.12.2025)
                "ap_3_-45": 6,  # ap2(30.12.2025)
                "ap_3_-48": 5,  # ap1(30.12.2025)
                "ap_3_-51": 7,  # ap8(29.12.2025)
                "ap_3_-54": 2,  # ap7(29.12.2025)
                "ap_3_-57": 4,  # ap6(29.12.2025)
                "f107_1944_0": 150, # f107adj avg of last 81 days [f107adj(01.01.2026),  f107adj(13.10.2025)] (value guessed here)
                "f107_24_-24": 164.8 # f107adj(31.12.2025) day avg for the previous day 
            } 

            for i, key in enumerate(MSIS_SW_KEYS):
                writer = messaging.SwDataMsg()
                self.msisSwWriters.append(writer)

                # initial payload
                swMsgData = messaging.SwDataMsgPayload(dataValue=float(sw_msg[key]))
                msg_handle = writer.write(swMsgData)
                self.msisSwMsgs.append(msg_handle)

                # connect MSIS input i to this publisher
                atm.swDataInMsgs[i].subscribeTo(msg_handle)

            # Subscribe to epoch message
            atm.epochInMsg.subscribeTo(self.epochMsg)

            # Schedule a new task in the simulation process to update MSIS model inputs at a slow frequency during simulation execution
            updaterTimeStep = macros.min2nano(30)
            self.dynProcess.addTask(self.scSim.CreateNewTask("msisInputUpdater", updaterTimeStep), 10) # High exec priority
            self.msisInputUpdater = MsisInputUpdater(self.cfg, self.msisSwWriters)
            self.scSim.AddModelToTask("msisInputUpdater", self.msisInputUpdater)

            logging.debug("[BSK] MSIS atmosphere model has been initialized")


        # Using Exponential density atmosphere
        elif use_exp:
            # Initialize ExponentialAtmosphere object
            atm = exponentialAtmosphere.ExponentialAtmosphere()
            atm.ModelTag = "expAtm"

            # Exponential atmosphere parameters
            atm.planetRadius = EARTH_RADIUS
            atm.scaleHeight = 15180.0      # [m] typical scale height (7200 before tuning)
            atm.baseDensity = 1.225         # [kg/m^3] density at 0 m
            atm.envMinReach = 0.0           # [m]
            atm.envMaxReach = 1000e3        # [m] cap model above 1000 km

            # simSetPlanetEnvironment.exponentialAtmosphere(atm, "earth") # Will give the same response as scaleHeight = 7200
            logging.debug("[BSK] Exponential atmosphere model has been initialized")

        
        # If the simulation is configured to not use drag, return None
        else:
            logging.debug("[BSK] No atmosphere model has been initialized")
            return None
        
        # Add to task
        self.scSim.AddModelToTask(self.simTaskName, atm)
        return atm
    

    def conditional_drag_effector(self, 
                                  sat: Satellite,
                                  scObj: spacecraft.Spacecraft,
                                  atm: Optional[Union[exponentialAtmosphere.ExponentialAtmosphere, msisAtmosphere.MsisAtmosphere]]
                                 ) -> spacecraft.Spacecraft:
        """
        if the simulation is configured to use exponential density drag, then define the drag effector,
        mount it on the satellite object, and schedule it in the simulation task
        
        :param self: 

        :param sat: The current Satellite object in the loop
        :type sat: Satellite

        :param scObj: The corresponding Basilisk spacecraft object in the cuurent iteration
        :type scObj: spacecraft.Spacecraft

        :param atm: Exponential density atmospheric model
        :type atm: Optional[exponentialAtmosphere.ExponentialAtmosphere]
        
        :return: Unmodified scObj if useExponentialDensityDrag == false.
          scObject with mounted atmospheric drag if  useExponentialDensityDrag == true
        :rtype: Spacecraft
        """

        use_msis = self.cfg.b_set.useMsisDrag
        use_exp = self.cfg.b_set.useExponentialDensityDrag
        
        if ((not use_msis) and (not use_exp)) or (atm is None):
            logging.debug("[BSK] no atmosphere model initialized")
            return scObj
        
        if use_msis and (not isinstance(atm, msisAtmosphere.MsisAtmosphere)):
            raise TypeError("Basilisk is configured to use an MSIS atmosphere model, but atmosphere object 'atm' is not of type 'msisAtmosphere.MsisAtmosphere'")
        
        elif use_exp and (not use_msis) and (not isinstance(atm, exponentialAtmosphere.ExponentialAtmosphere)):
            raise TypeError("Basilisk is configured to use an Exponential atmosphere model, but atmosphere object 'atm' is not of type 'exponentialAtmosphere.ExponentialAtmosphere'")

        # ---- Drag effector (exponential density + cannonball) ----
        # Register this spacecraft with the atmosphere model to get its own atm mesg
        atm.addSpacecraftToModel(scObj.scStateOutMsg)

        # Define drag
        drag = dragDynamicEffector.DragDynamicEffector()
        drag.cannonballDrag()

        # Set core parameters
        core = dragDynamicEffector.DragBaseData()
        core.dragCoeff = sat.C_D # getattr(sat, "C_D", 2.2)
        core.projectedArea = sat.A_D # getattr(sat, "A_D", 0.06)
        drag.coreParams = core

        # Subscribe to density from this spacecraft's atmosphere message
        atmMsg = atm.envOutMsgs[-1]
        drag.atmoDensInMsg.subscribeTo(atmMsg)

        # Mount and schedule
        scObj.addDynamicEffector(drag)
        self.scSim.AddModelToTask(self.simTaskName, drag)

        return scObj
            

    def conditional_eclipse_init(self, 
                             spiceObj: spiceInterface.SpiceInterface
                             ) -> tuple[Optional[Any], Optional[eclipse.Eclipse]]:
        """
        Initializes an eclipse model
        
        :param self: 
        :param spiceObj: SPICE interface giving the accurate position of the Earth (idx 0), Sun (idx 1) and Moon (idx 2, if created)
        :type spiceObj: spiceInterface.SpiceInterface
        :return: Sun message, Eclipse model if useSRP == True. None, None otherwise.
        :rtype: tuple[Any | None, Eclipse | None]
        """

        # Don't set up SPICE or Eclipse model if config defines useSRP == False
        if not self.cfg.b_set.useSRP:
            return None, None

        # Fetch the Earth's and Sun's position from the SPICE publisher.
        # The Earth and Sun will always have index [0] and [1] because gravFactory always creates Earth first, then Sun.
        # See 'conditional_planet_gravity_generation()' func for logic. 
        earthMsg = spiceObj.planetStateOutMsgs[0]
        sunMsg   = spiceObj.planetStateOutMsgs[1]

        # Initialize eclipse mode (when the Earth eclipses the Sun)
        eclipseObj = eclipse.Eclipse()
        eclipseObj.sunInMsg.subscribeTo(sunMsg)
        eclipseObj.addPlanetToModel(earthMsg) # Earth occluder
        
        # Schedule object to simualtion process
        self.scSim.AddModelToTask(self.simTaskName, eclipseObj) 

        logging.debug("[BSK] Eclipse model has been initialized")


        ####### FOR DEBUG ###############################
        # earthMsg = spiceObj.planetStateOutMsgs[0]
        # sunMsg   = spiceObj.planetStateOutMsgs[1]
        # try:
        #     moonMsg = spiceObj.planetStateOutMsgs[2]
        # except:
        #     logging.debug("[BSK] The Moon gravitational entity is not defined in the SPICE interface")
        # self.sunRec = sunMsg.recorder(samplingTime)
        # self.earthRec = earthMsg.recorder(samplingTime)
        # try:
        #     self.moonRec = moonMsg.recorder(samplingTime) # type: ignore
        # except:
        #     self.moonRec = None
        # self.scSim.AddModelToTask(self.simTaskName, self.sunRec)
        # self.scSim.AddModelToTask(self.simTaskName, self.earthRec)
        # if self.moonRec is not None: self.scSim.AddModelToTask(self.simTaskName, self.moonRec)
        #################################################

        return sunMsg, eclipseObj
    

    def conditional_srp_effector(self, 
                                 sat: Satellite,
                                 scObj: spacecraft.Spacecraft,
                                 sunMsg: Optional[Any],
                                 eclipseObj: Optional[eclipse.Eclipse]) -> spacecraft.Spacecraft:
        """
        if the simulation is configured to use SRP, then define the SRP effector,
        mount it on the satellite object, and schedule it in the simulation task
        
        :param self: 
        :param sat: The current Satellite object in the loop
        :type sat: Satellite
        :param scObj: The corresponding Basilisk spacecraft object in the cuurent iteration
        :type scObj: spacecraft.Spacecraft
        :param sunMsg: The Sun's position or None
        :type sunMsg: Optional[Any]
        :param eclipseObj: Eclipse model
        :type eclipseObj: Optional[eclipse.Eclipse]
        :return: Unmodified scObj if useSRP == false.
          scObject with mounted SRP force if useSRP == true
        :rtype: Spacecraft
        """

        # Don't mount SRP effector on the spacecraft object if useSRP == False or any Optional inputs are None
        if (not self.cfg.b_set.useSRP) or (sunMsg is None) or (eclipseObj is None):
            return scObj
        
        # Register this spacecraft with the eclipse model to get its own eclipse msg
        eclipseObj.addSpacecraftToModel(scObj.scStateOutMsg)

        # Define srp
        srp = radiationPressure.RadiationPressure()
        srp.setUseCannonballModel()
        srp.coefficientReflection = sat.C_R # getattr(sat, "C_R", 1.21)
        srp.area = sat.A_srp # getattr(sat, "A_srp", 0.06)  

        # Subscribe to Sun ephemeris + this spacecraft’s eclipse factor
        srp.sunEphmInMsg.subscribeTo(sunMsg)
        srp.sunEclipseInMsg.subscribeTo(eclipseObj.eclipseOutMsgs[-1])  # last added = this SC

        # Mount SRP onto the spacecraft and schedule it
        scObj.addDynamicEffector(srp)
        self.scSim.AddModelToTask(self.simTaskName, srp)

        logging.debug("[BSK] Solar radiation pressure (SRP) model initialized")

        return scObj


    def conditional_object_integrator(self, scObj: spacecraft.Spacecraft) -> spacecraft.Spacecraft:
        
        integration_method = self.cfg.b_set.integrator

        # Select integration method
        match integration_method:
            case "RKF45":
                logging.debug(f"[BSK] Selecting RKF45 numerical integrator for {scObj.ModelTag}")
                integratorObj = svIntegrators.svIntegratorRKF45(scObj)
            case "RKF78":
                logging.debug(f"[BSK] Selecting RKF78 numerical integrator for {scObj.ModelTag}")
                integratorObj = svIntegrators.svIntegratorRKF78(scObj)
            case _:
                logging.debug(f"[BSK] Selecting defualt RK4 numerical integrator for {scObj.ModelTag}")
                return scObj # Use standard integration method RK4
        
        # Set the object's non-default integration method
        scObj.setIntegrator(integratorObj)

        # Keep a reference so it doesn't get CE'ed
        self.integrators.append(integratorObj)

        return scObj


    @staticmethod
    def spaced_satellites_on_same_orbital_plane(satellite_idx: int, 
                                                separation_ang: float, 
                                                mu: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the ECI initial conditions for satellite cfg.satellites[satellite_idx]. They are calculated to achieve even
        satellite spacing defined by 'separation_ang'.
        
        :param satellite_idx: Description
        :type satellite_idx: int
        :param separation_ang: Description
        :type separation_ang: float
        :param mu: Description
        :type mu: float
        :return: Description
        :rtype: tuple[NDArray[float64], NDArray[float64]]
        """
        # setup the orbit using classical orbit elements
        oe = orbitalMotion.ClassicElements()
    

        rLEO = 7000. * 1000      # meters
        rGEO = 42000. * 1000     # meters

        # Missing type stub causes static error -> Ignore
        oe.a = rLEO                                     # type: ignore
        oe.e = 0.001                                    # type: ignore
        oe.i = 33.3 * macros.D2R                        # type: ignore
        oe.Omega = 48.2 * macros.D2R                    # type: ignore
        oe.omega = 347.8 * macros.D2R                   # type: ignore
        oe.f = 85.3 * macros.D2R                        # type: ignore
        oe.f = oe.f - satellite_idx * separation_ang    # type: ignore

        rN, vN = orbitalMotion.elem2rv(mu, oe)

        return rN, vN
    

    @staticmethod
    def custom_initial_states(satellite_idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Edit the parameters in this method to manually output the initial states for each satellite
        
        Args:
            satellite_num (int): The satellite index in cfg.satellites

        Returns:
            rN (NDArray[np.float64]): Initial position vector for satellite cfg.satellites[satellite_idx] in ECI frame\n
            vN (NDArray[np.float64]): Initial velocity vector for satellite cfg.satellites[satellite_idx] in ECI frame
        """

        rN_list: list[NDArray[np.float64]] = []
        vN_list: list[NDArray[np.float64]] = []

        # 1st (chief) satellite initial conditions (ECI):
        rN1 = np.array([10000e3, 0.0, 0.0])     # Position vector [m]
        vN1 = np.array([0.0, 1e3, 0.0])         # Velocity vector [m/s]
        rN_list.append(rN1)
        vN_list.append(vN1)

        # 2nd satellite initial conditions (ECI):
        rN2 = np.array([10000e3, 0.0, 0.0])     # Position vector [m]
        vN2 = np.array([0.0, 1e3, 0.0])         # Velocity vector [m/s]
        rN_list.append(rN2)
        vN_list.append(vN2)

        # 3rd satellite initial conditions (ECI):
        rN3 = np.array([10000e3, 0.0, 0.0])     # Position vector [m]
        vN3 = np.array([0.0, 1e3, 0.0])         # Velocity vector [m/s]
        rN_list.append(rN3)
        vN_list.append(vN3)

        # Output
        rN = rN_list[satellite_idx]
        vN = vN_list[satellite_idx]
        return rN, vN
    

    @staticmethod
    def to_spice_utc(s: str) -> str:
        # s like "02.04.2025 12:00:00" (DD.MM.YYYY HH:MM:SS) in local time (Europe/Oslo)
        dt_local = datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
        # If the string is already UTC, replace with timezone.utc directly.
        dt_utc = dt_local.replace(tzinfo=timezone.utc)

        return dt_utc.strftime("%Y %b %d %H:%M:%S UTC")
    

    def DEBUG_plot_msis_atm_density_against_altitude(self) -> None:
        """
        Print
        """
        
        def _darker_color(color, factor=0.5):
            """
            Return a darker shade of the given color.
            factor < 1 → darker, factor = 1 → same color
            """
            rgb = np.array(mcolors.to_rgb(color))
            return tuple(factor * rgb)
        
        # Standard Matplotlib tab colors (max 4 satellites)
        base_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
        ]

        # Load state data for all spacecrafts
        all_sc_data = self.scRecorders
        n_sc_data_objects = len(all_sc_data)

        # Load atmosphere data for all spacecrafts
        all_atm_data = self.atmRecorders
        n_atm_data_objects = len(all_atm_data)
        
        # Checks
        if not n_sc_data_objects == n_atm_data_objects:
            raise ValueError(f"Not the same number of spacecraft recorders ({n_sc_data_objects}) as atmosphere recorders ({n_atm_data_objects})")            

        if (n_sc_data_objects == 0) or (n_atm_data_objects == 0):
            raise ValueError(f"No Spacecraft state recorders or Atmosphere recorders have been initialized")
        
        if len(all_sc_data[0].r_BN_N) == 0:
            raise ValueError(f"No satellite position data contained in the spacecraft recorders")

        if len(all_atm_data[0].neutralDensity) == 0:
            raise ValueError(f"No atmosphere density data contained in the atmosphere recorders")
        
        n_data_objects = n_sc_data_objects
            
        # Initialize twin-plot
        fig, ax_alt = plt.subplots(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax_den = ax_alt.twinx()

        # Iterate through all satellites to plot
        satellites = self.cfg.satellites
        for i in range(n_data_objects):
            # Temp: only plot atm density against altitude for the first satellite
            if i > 0:
                break

            # Set colors
            alt_color = base_colors[i]
            den_color = _darker_color(alt_color, 0.7)

            # Get current spacecraft name
            sat_name = satellites[i].name

            # Get data for the current spacecraft
            sc_data = all_sc_data[i].r_BN_N
            t_sc = all_sc_data[i].times() * 1e-9 / 3600 # [h]
            dens_data = all_atm_data[i].neutralDensity
            t_atm = all_atm_data[i].times() * 1e-9 / 3600 # [h]

            # Calculate altitude
            sc_alt = (np.linalg.norm(sc_data, axis=1) - EARTH_RADIUS)*1e-3 # [km]

            # --- Altitude (left axis) ---
            ax_alt.plot(
                t_sc,
                sc_alt,
                color = alt_color,
                label=f"Altitude {sat_name}"
            )
            
            # --- Density (right axis) ---
            ax_den.plot(
                t_atm,
                dens_data,
                color = den_color,
                label=f"Density {sat_name}"
            )

        # ax_den.set_yscale("log")

        ax_alt.set_xlabel("Time [h]")
        ax_alt.set_ylabel("Altitude [km]")
        ax_den.set_ylabel("Density [kg/m³]")

        # Legends in bottom corners
        ax_alt.legend(loc="lower left")
        ax_den.legend(loc="lower right")

        ax_alt.grid(True)

        plt.tight_layout()
        plt.show()