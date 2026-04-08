import os
import logging
import numpy as np
import matplotlib.pyplot as plt # Only for debug
import matplotlib.colors as mcolors # only for debug
from typing import Optional, Any, Union
from numpy.typing import NDArray
from datetime import datetime, timezone, timedelta

from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.simulation import (spacecraft, radiationPressure, spiceInterface, eclipse,  
                                exponentialAtmosphere, msisAtmosphere, dragDynamicEffector, 
                                svIntegrators, reactionWheelStateEffector,
                                RWConfigPayload, groundLocation)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, 
                                unitTestSupport, vizSupport, fswSetupRW, simIncludeRW)

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimData, SimObjData
from object_definitions.MsisInputUpdater_def import (MsisInputUpdater, MSIS_SW_KEYS)
from Formation_Flying_Energy_Analysis.src.object_definitions.FswStack_def import FswStack


# from plotting.plot import PLT_WIDTH, PLT_HEIGHT
PLT_HEIGHT = 6.0
PLT_WIDTH = 16.0

EARTH_RADIUS = 6378136.6 # [m] WGS-84 equatorial radius
VIZARD_SAVE_PATH = "/home/chris/code/formation-flight-gnssr-simulator/Formation_Flying_Energy_Analysis/output_data/_VizFiles/bsk_sim.bin"
GRAV_COEFF_FILE_PATH = "shared_input_data/grav_coeff/GGM03S.txt"


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


    Scheduler execution priority
    <SimBaseClass>
    |---<Process>
        |---<Task>
            |---<Model>
    scSim
    |
    |---simProcess
        |
        |---simTask Pri: 0
            |
            |---scObj (All of them)
            |
            |---
            |
            |---
            |
            |---
            |
            |---
        |
        |---fswTask (Task) Pri: 5
        |
        |---msisInputUpdater (Task) Pri: 10


    
    =========================================================================================================
    """
    def __init__(self, cfg: Config) -> None:
        logging.debug("[BSK] Setting up Basilisk simulation...")
    
        
        ###################################
        # Configure simulation parameters #
        ###################################

        self.cfg = cfg

        # Set Simulation time
        self.spiceTime = self._to_spice_utc(self.cfg.startTime)   # Only used to initialize SPICE interface
        self.epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(self.spiceTime)   # Used for time-dependent models (SPICE interface (eclipse model by extension), MSIS)
        
        # Helper simulation time datetime objects used by other methods
        self._simStartDt = datetime.strptime(self.cfg.startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        self._simEndDt = self._simStartDt + timedelta(hours=float(self.cfg.simulationDuration))

        # Set fixed simulation integration time step
        simulationTimeStep = macros.sec2nano(self.cfg.deltaT)

        # Set simulation duration
        simualtionDuration_sec = self.cfg.simulationDuration * 60 * 60
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

        # Create a stable list of flight software stacks
        self.fswStacks: list[FswStack] = []
        # self.rwEffectors: list = []
        self.rwFactories: list[simIncludeRW.rwFactory] = []
        self.fswRwParamMsgs: list[messaging.RWArrayConfigMsg] = []

        # Dictionary to keep track of RW clusters for each satellite
        self.rwClusters: dict[str, list[RWConfigPayload.RWConfigPayload]] = {}

        # Stable list containing all ground stations
        self.groundStations: list[groundLocation.GroundLocation] = []

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


        #########################################
        # Create dedicated Flight Software task #
        #########################################
        self.fswTaskName = "fswTask"
        self.dynProcess.addTask(self.scSim.CreateNewTask(self.fswTaskName, simulationTimeStep), 5)  


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


        ##############################################
        # Initialize Ground Stations #
        ##############################################
        gs_state_msgs: list[messaging.GroundStateMsg] = []
        for i, gs in enumerate(cfg.ground_stations):
            groundStation = groundLocation.GroundLocation()
            groundStation.ModelTag = gs.gs_tag
            groundStation.planetRadius = EARTH_RADIUS
            groundStation.specifyLocation(np.radians(gs.lat), np.radians(gs.long), gs.alt)
            groundStation.planetInMsg.subscribeTo(spiceObj.planetStateOutMsgs[0])
            groundStation.minimumElevation = np.radians(gs.min_elev)
            groundStation.maximumRange = gs.max_range

            # Append to stable list
            self.groundStations.append(groundStation)
            gs_state_msgs.append(groundStation.currentGroundStateOutMsg)

            # Add to task
            self.scSim.AddModelToTask(self.simTaskName, groundStation)




        #################################################################
        # Initialize scObjects and scRecorders, and attach force models #
        #################################################################
        
        # Initialize empty containers for to-be-defined Spacecraft objects and its recorders
        self.scObjects: list[spacecraft.Spacecraft] = []
        self.scRecorders: list = [] # list of what?
        self.atmRecorders: list = []
        self.rwMotorRecorders: list = []
        self.attErrRecorders: list = []
        self.snTransRecorders: list = []
        self.mrpRecorders: list = []
        self.rwRecorders: list = []

        # get satellites from config
        satellites = self.cfg.satellites

        #################################################################################################
        # Loop through all satellites to define all spacecraft objects, attach all force models and FSW #
        #################################################################################################
        for i, sat in enumerate(satellites):

            # Initialize spacecraft object
            scObj = spacecraft.Spacecraft()
            scObj.ModelTag = sat.name
            scObj.hub.mHub = sat.m_s
            scObj.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # [m] position vector of body-fixed point B relative to CM
            scObj.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(sat.I_B) # [kg m^2] Inertia of hub about point Bc in B frame components

            # Add spacecraft object to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scObj, 1)

            # Get initial conditions
            if self.cfg.sat_init_source == "oe":
                # Get initial state vector from orbital elements
                oe = sat.init_OEs
                mu = gravFactory.gravBodies["earth"].mu
                rN, vN = orbitalMotion.elem2rv(mu, oe)

            elif self.cfg.sat_init_source == "vec":
                # Use initial state given directly from config
                rN = sat.init_pos # [m]   In N frame (inertial = ECI)
                vN = sat.init_vel # [m/s] in N frame (inertial = ECI)

            else:
                raise ValueError(f"Unrecognized satellite initial condition source '{self.cfg.sat_init_source}'")

            # Set the initial conditions for the spacecraft object
            scObj.hub.r_CN_NInit = rN  # [m]   r_BN_N
            scObj.hub.v_CN_NInit = vN  # [m/s] v_BN_N
            scObj.hub.sigma_BNInit = sat.init_att  # orientation of Body(B) relative to inertial(N) expressed using MRP
            scObj.hub.omega_BN_BInit = sat.init_angvel  # [rad/s] angular velocity of Body(B) relative to inertial(N) expressed in (B)
            
            # ---- Main graviational attraction, Spherical Harmonics and 3rd body perturbation ----
            # The gravitational sources and models have already been defined gravFactory in accordance with cfg
            gravFactory.addBodiesTo(scObj)
            

            # ---- Drag effector ----
            scObj = self.conditional_drag_effector(sat, scObj, atm)
            
            
            # ---- SRP effector ----
            # Register this spacecraft with the eclipse model to get its own eclipse msg
            scObj = self.conditional_srp_effector(sat, scObj, sunMsg, eclipseObj)
            

            # ---- Reaction Wheel State Effector ---- 
            # Create RWs from config, create a RW effector and attach to the spacecraft
            scObj, rwFactory, rwEffector = self.RW_effector(sat, scObj, i)


             # ---- Attach spacecraft to ground station(s) and prepair access msgs for fsw ---- 
            gs_access_msgs: list[messaging.AccessMsg] = [] # will contain the access msg for this spacecraft against all ground stations
            for j, gs in enumerate(self.groundStations):
                gs.addSpacecraftToModel(scObj.scStateOutMsg)
                gs_access_msgs.append(gs.accessOutMsgs[-1]) # -1 idx refers to the latest added sc (current iteration sat)

            
            # ----  Get the spacecraft-sun eclipse msg for fsw ---- 
            assert eclipseObj is not None
            sun_eclipse_msg: messaging.EclipseMsg = eclipseObj.eclipseOutMsgs[-1] 
            # TODO: Using index '-1' assumes that no other SC have been added to the eclipse model since 'conditional_srp_effector'
            #       This will never actually be a problem, but it might be fragile to assume this. 
            #       Move this line inside 'conditional_srp_effector' or use satellite index instead. 

            # ---- Flight software ----
            assert sunMsg is not None
            fswRwParamMsg = rwFactory.getConfigMessage()
            self.fswRwParamMsgs.append(fswRwParamMsg)
            fsw = FswStack(
                sat = sat,
                sat_idx = i,
                sc_state_out_msg = scObj.scStateOutMsg,
                rw_speed_out_msg = rwEffector.rwSpeedOutMsg,
                rw_config_msg = fswRwParamMsg,
                gs_access_msgs = gs_access_msgs,
                gs_state_msgs = gs_state_msgs,
                sun_eclipse_msg = sun_eclipse_msg,
                sun_state_msg = sunMsg
            )            
            self.scSim.AddModelToTask(self.fswTaskName, fsw)

            # RW effector must know its commanded torque
            rwEffector.rwMotorCmdInMsg.subscribeTo(fsw.rwMotorTorqueOutMsg)


            # ---- Set object integration method ----
            scObj = self.conditional_object_integrator(scObj)

           
            # ---- Define and append persistent objects and recorders ----
            # Create recorders
            scRec = scObj.scStateOutMsg.recorder(samplingTime)
            assert atm is not None
            atmLog = atm.envOutMsgs[i].recorder(samplingTime)
            rwMotorLog = fsw.rwMotorTorqueOutMsg.recorder(samplingTime)
            attErrorLog = fsw.attGuidOutMsg.recorder(samplingTime)
            snTransLog = fsw.navTransOutMsg.recorder(samplingTime)
            mrpLog = rwEffector.rwSpeedOutMsg.recorder(samplingTime)
            rwLogs: list = []
            for j in range(len(self.cfg.spinUVecs)):
                rwLogs.append(rwEffector.rwOutMsgs[j].recorder(samplingTime))
                self.scSim.AddModelToTask(self.simTaskName, rwLogs[j])
            # srpRec = self.make_srp_recorder(srp, samplingTime)  

            # Add recorder to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scRec)
            self.scSim.AddModelToTask(self.simTaskName, atmLog)
            self.scSim.AddModelToTask(self.simTaskName, rwMotorLog)
            self.scSim.AddModelToTask(self.simTaskName, attErrorLog)
            self.scSim.AddModelToTask(self.simTaskName, snTransLog)
            self.scSim.AddModelToTask(self.simTaskName, mrpLog)
            # self.scSim.AddModelToTask(self.simTaskName, srpRec)
                        
            # Append object and recorders to avvoid them getting CE'ed
            self.scObjects.append(scObj)
            self.fswStacks.append(fsw)
            self.rwFactories.append(rwFactory)

            self.scRecorders.append(scRec)
            self.atmRecorders.append(atmLog)
            self.rwMotorRecorders.append(rwMotorLog)
            self.attErrRecorders.append(attErrorLog)
            self.snTransRecorders.append(snTransLog)
            self.mrpRecorders.append(mrpLog)
            self.rwRecorders.append(rwLogs)
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

        satellites = self.cfg.satellites
        simulationDuration_sec = self.cfg.simulationDuration * 60 * 60
        timeStep_sec = self.cfg.deltaT

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
        # self._DEBUG_plot_msis_atm_density_against_altitude()
        ############################################


        ############ RW and Pointing controll debug ############
        sat_idx = 0
        fileName = os.path.basename(os.path.splitext(__file__)[0])
        # num_RWs = len(self.RWs)
        num_RWs = len(self.cfg.spinUVecs)

        dataUsReq = self.rwMotorRecorders[sat_idx].motorTorque
        dataSigmaBR = self.attErrRecorders[sat_idx].sigma_BR
        dataOmegaBR = self.attErrRecorders[sat_idx].omega_BR_B
        dataOmegaRW = self.mrpRecorders[sat_idx].wheelSpeeds

        dataRW = []
        # for i, RW in enumerate(self.RWs):
        for i in range(num_RWs):
            dataRW.append(self.rwRecorders[sat_idx][i].u_current)
        np.set_printoptions(precision=16)

        #
        #   plot the results
        #
        timeData = self.rwMotorRecorders[sat_idx].times() * macros.NANO2MIN
        plt.close("all")  # clears out plots from earlier test runs

        self._DEBUG_plot_attitude_error(timeData, dataSigmaBR)
        figureList = {}
        pltName = fileName + "1"
        figureList[pltName] = plt.figure(1)

        self._DEBUG_plot_rw_motor_torque(timeData, dataUsReq, dataRW, num_RWs)
        pltName = fileName + "2"
        figureList[pltName] = plt.figure(2)

        self._DEBUG_plot_rate_error(timeData, dataOmegaBR)
        self._DEBUG_plot_rw_speeds(timeData, dataOmegaRW, num_RWs)
        pltName = fileName + "3"
        figureList[pltName] = plt.figure(4)

        plt.show()

        # close the plots being saved off to avoid over-writing old and new figures
        plt.close("all")


        ########################################################


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
        if not self.cfg.useSun3rdBody:
            sun.mu = 0
        else:
            logging.debug("[BSK] Sun 3rd body perturbation initialized")

        # Create the Moon only if useMoon3rdBody == True
        if self.cfg.useMoon3rdBody:
            moon = gravFactory.createMoon()
            logging.debug("[BSK] Moon 3rd body perturbation initialized")
        
        # Set Earth as the central gravitational body
        earth.isCentralBody = True

        # Use spherical harmonics if useSphericalHarmonics == True
        if self.cfg.useSphericalHarmonics:
            # If extra customization is required, see the createEarth() macro to change additional values.
            earth.useSphericalHarmonicsGravityModel(
                GRAV_COEFF_FILE_PATH, 
                self.cfg.sphericalHarmonicsDegree
            )

            logging.debug(f"[BSK] Earth created with spherical harmonics gravity model of order and degree {self.cfg.sphericalHarmonicsDegree}")

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
        
        use_msis = self.cfg.useMsisDrag
        use_exp = self.cfg.useExponentialDensityDrag            

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
            logging.debug("[BSK] Exponential atmosphere mgfdgjfodel has been initialized")

        
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

        use_msis = self.cfg.useMsisDrag
        use_exp = self.cfg.useExponentialDensityDrag
        
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
                             ) -> tuple[Optional[messaging.SpicePlanetStateMsg], Optional[eclipse.Eclipse]]:
        """
        Initializes an eclipse model
        
        :param self: 
        :param spiceObj: SPICE interface giving the accurate position of the Earth (idx 0), Sun (idx 1) and Moon (idx 2, if created)
        :type spiceObj: spiceInterface.SpiceInterface
        :return: Sun message, Eclipse model if useSRP == True. None, None otherwise.
        :rtype: tuple[Any | None, Eclipse | None]
        """

        # Don't set up SPICE or Eclipse model if config defines useSRP == False
        if not self.cfg.useSRP:
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
        if (not self.cfg.useSRP) or (sunMsg is None) or (eclipseObj is None):
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
        
        integration_method = self.cfg.integrator

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


    def RW_effector(self, 
                    sat: Satellite, 
                    scObj: spacecraft.Spacecraft, 
                    i: int
        ) -> tuple[spacecraft.Spacecraft, 
                   simIncludeRW.rwFactory, 
                   reactionWheelStateEffector.ReactionWheelStateEffector]:
        """
        Generate reaction wheel object(s) using the config parameters, create the RW state effector, 
        add it to the spacecraft and schedule it in the simulation task. 

        Params:
            sat (Satellite): current satellite object in satellite loop
            scObj (Spacecraft): Current spacecraft object in satellite loop
            i (int): Satellite iteration number

        Returns:
            (Spacecraft): The spacecraft object WITH attached RW effector
            (rwFactory): The factory that created the RWs and keep all their information
            (ReactionWheelStateEffector): The RW effector
        """
        
        rwFactory = simIncludeRW.rwFactory()

        # Select internal RW physics model
        match self.cfg.RW_model:
            case "BalancedWheels":
                varRWModel: int = messaging.BalancedWheels
                logging.debug(f"[BSK] 'BalancedWheels' internal RW model selected")
            case "JitterSimple":
                varRWModel: int = messaging.JitterSimple
                logging.debug(f"[BSK] 'JitterSimple' internal RW model selected")
            case "JitterFullyCoupled":
                varRWModel: int = messaging.JitterFullyCoupled
                logging.debug(f"[BSK] 'JitterFullyCoupled' internal RW model selected")
            case _:
                raise ValueError(f"Unrecognized 'RW_model' received."
                                f"Got '{self.cfg.RW_model}', expected ['BalancedWheels', 'JitterSimple', 'JitterFullyCoupled']")
            
        # Create RWs (one for each vector present in self.cfg.spinUVecs)
        RWs: list[RWConfigPayload.RWConfigPayload] = []
        for j, spinUVec in enumerate(self.cfg.spinUVecs):
            if self.cfg.useFriction:
                RW = rwFactory.create(
                    'custom',
                    spinUVec,
                    label =         f"sc{i}W{j}",
                    RWModel =       varRWModel,
                    rWB_B =         [0., 0., 0.],
                    Omega =         self.cfg.init_rpm,
                    Omega_max =     self.cfg.max_rpm,
                    maxMomentum =   self.cfg.maxMomentum,
                    u_max =         self.cfg.maxTorque,
                    u_min =         self.cfg.minTorque,
                    # Js =            self.cfg.I_RW, # Js calculated using Omega_max and maxMomentum
                    useMinTorque =  self.cfg.useMinTorque,
                    useMaxTorque =  True,
                    useFriction =   True,
                    fCoulomb =      self.cfg.fCoulomb,
                    fStatic =       self.cfg.fStatic,
                    betaStatic =    self.cfg.betaStatic,
                    cViscous =      self.cfg.cViscous
                )
            else:
                RW = rwFactory.create(
                    'custom',
                    spinUVec,
                    label =         f"sc{i}W{j}",
                    RWModel =       varRWModel,
                    rWB_B =         [0., 0., 0.],
                    Omega =         self.cfg.init_rpm,
                    Omega_max =     self.cfg.max_rpm,
                    maxMomentum =   self.cfg.maxMomentum,
                    u_max =         self.cfg.maxTorque,
                    u_min =         self.cfg.minTorque,
                    # Js =            self.cfg.I_RW, # Js calculated using Omega_max and maxMomentum
                    useMinTorque =  self.cfg.useMinTorque,
                    useMaxTorque =  True,
                )
            RWs.append(RW)

        # Keep track of which RWs belong to each sc object
        self.rwClusters[scObj.ModelTag] = RWs
        
        logging.debug(f"[BSK] {len(RWs)} Reaction wheels created for satellite '{sat.name}'")

        # Create RW effector and attach to the spacecraft 
        rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwEffector.ModelTag = f"RW_cluster_{i}"
        rwFactory.addToSpacecraft(scObj.ModelTag, rwEffector, scObj)

        # Add RW effector to the simulation process
        self.scSim.AddModelToTask(self.simTaskName, rwEffector, 2)
        
        return scObj, rwFactory, rwEffector


    @staticmethod
    def _spaced_satellites_on_same_orbital_plane(satellite_idx: int, 
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
    def _to_spice_utc(s: str) -> str:
        # s like "02.04.2025 12:00:00" (DD.MM.YYYY HH:MM:SS) in local time (Europe/Oslo)
        dt_local = datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
        # If the string is already UTC, replace with timezone.utc directly.
        dt_utc = dt_local.replace(tzinfo=timezone.utc)

        return dt_utc.strftime("%Y %b %d %H:%M:%S UTC")
    

    # ---- DEBUG plotting functions using recorder data ---- #
    def _DEBUG_plot_msis_atm_density_against_altitude(self) -> None:
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

    @staticmethod
    def _DEBUG_plot_attitude_error(timeData, dataSigmaBR):
        """Plot the attitude errors."""
        plt.figure(1)
        for idx in range(3):
            plt.plot(timeData, dataSigmaBR[:, idx],
                    color=unitTestSupport.getLineColor(idx, 3),
                    label=r'$\sigma_' + str(idx) + '$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel(r'Attitude Error $\sigma_{B/R}$')

    @staticmethod
    def _DEBUG_plot_rw_cmd_torque(timeData, dataUsReq, numRW):
        """Plot the RW command torques."""
        plt.figure(2)
        for idx in range(3):
            plt.plot(timeData, dataUsReq[:, idx],
                    '--',
                    color=unitTestSupport.getLineColor(idx, numRW),
                    label=r'$\hat u_{s,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('RW Motor Torque (Nm)')

    @staticmethod
    def _DEBUG_plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW):
        """Plot the RW actual motor torques."""
        plt.figure(2)
        for idx in range(3):
            plt.plot(timeData, dataUsReq[:, idx],
                    '--',
                    color=unitTestSupport.getLineColor(idx, numRW),
                    label=r'$\hat u_{s,' + str(idx) + '}$')
            plt.plot(timeData, dataRW[idx],
                    color=unitTestSupport.getLineColor(idx, numRW),
                    label='$u_{s,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('RW Motor Torque (Nm)')

    @staticmethod
    def _DEBUG_plot_rate_error(timeData, dataOmegaBR):
        """Plot the body angular velocity rate tracking errors."""
        plt.figure(3)
        for idx in range(3):
            plt.plot(timeData, dataOmegaBR[:, idx],
                    color=unitTestSupport.getLineColor(idx, 3),
                    label=r'$\omega_{BR,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('Rate Tracking Error (rad/s) ')

    @staticmethod
    def _DEBUG_plot_rw_speeds(timeData, dataOmegaRW, numRW):
        """Plot the RW spin rates."""
        plt.figure(4)
        for idx in range(numRW):
            plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                    color=unitTestSupport.getLineColor(idx, numRW),
                    label=r'$\Omega_{' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('RW Speed (RPM) ')