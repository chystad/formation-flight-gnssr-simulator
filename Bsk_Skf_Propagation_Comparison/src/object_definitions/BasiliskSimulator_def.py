import os
import logging
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray
from datetime import datetime, timezone

# from object_definitions.BaseSimulator_def import BaseSimulator
from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimData, SimObjData

from Basilisk import __path__
from Basilisk.simulation import (spacecraft, radiationPressure, spiceInterface, eclipse,  
                                exponentialAtmosphere, dragDynamicEffector, svIntegrators)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport)

VIZARD_SAVE_PATH = "/home/chris/code/formation-flight-gnssr-simulator/Bsk_Skf_Propagation_Comparison/output_data/_VizFiles/bsk_sim.bin"
GRAV_COEFF_FILE_PATH = "shared_input_data/grav_coeff/GGM03S.txt"

"""
=========================================================================================================
TODO docstring
=========================================================================================================
"""

class BasiliskSimulator:
    """
    =========================================================================================================
    ATTRIBUTES:
        cfg             Config instance
        simTaskName     Simulation task name (str)
        scSim           Simulation module container
        scObjects       List containing all simulation objects (satellites)
        scRecorders   List containing all simulation recorders (one for each scObject)
        sim_data        Object containing the simulaton output data (Optional[SimData])
        sunRec          Sun state recorder
    =========================================================================================================
    """
    def __init__(self, cfg: Config) -> None:
        logging.debug("Setting up Basilisk simulation...")
        
        ###############
        # Load config #
        ###############
        
        self.cfg = cfg     # Assign config to self.cfg attribute
        d_set = cfg        # default config
        b_set = cfg.b_set  # basilisk config
        


        ###################################
        # Configure simulation parameters #
        ###################################

        # Set fixed simulation integration time step
        simulationTimeStep = macros.sec2nano(b_set.deltaT)

        # Set simulation duration
        simualtionDuration_sec = d_set.simulationDuration * 60 * 60
        simulationDuration = macros.sec2nano(simualtionDuration_sec)

        # Set number of data points
        numDataPoints = simulationDuration // simulationTimeStep

        # Set sample time (same as 'deltaT' in basilisk simulation config)
        samplingTime = unitTestSupport.samplingTime(simulationDuration, simulationTimeStep, numDataPoints)

        # Initialize integrators
        self.integrators = []

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

        # Create the simulation process
        dynProcess = self.scSim.CreateNewProcess(simProcessName)

        # create the dynamics task and specify the integration update time
        dynProcess.addTask(self.scSim.CreateNewTask(self.simTaskName, simulationTimeStep))
        

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
            scObj.hub.mHub = getattr(sat, "m_s", 6.0)

            # Add spacecraft object to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scObj)

            if b_set.override_skf_initial_state:
                # Get initial conditions corresponding to satellites separated by an arbitrary angle 
                # in the same orbital plane
                separationAng = 20.0 * macros.D2R
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
            # srpRec = self.make_srp_recorder(srp, samplingTime)  

            # Add recorder to the simulation process
            self.scSim.AddModelToTask(self.simTaskName, scRec)
            # self.scSim.AddModelToTask(self.simTaskName, srpRec)
                        
            # Append defined spacecraft object and scRec to scObjects and scRecorders, respectively
            self.scObjects.append(scObj)
            self.scRecorders.append(scRec)
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
        logging.debug("Basilisk simulation running...")
       
        self.scSim.ExecuteSimulation()
        # Note that this module simulates both the translational and rotational motion of the spacecraft.
        # In this scenario only the translational (i.e. orbital) motion is tracked.  This means the rotational motion
        # remains at a default inertial frame orientation in this scenario.  There is no appreciable speed hit to
        # simulate both the orbital and rotational motion for a single rigid body.  In the later scenarios
        # the rotational motion is engaged by specifying rotational initial conditions, as well as rotation
        # related effectors.  In this simple scenario only translational motion is setup and tracked.
        # Further, the default spacecraft parameters, such as the unit mass and the principle inertia values are
        # just fine for this orbit simulation as they don't impact the orbital dynamics in this case.
        # This is true for all gravity force only orbital simulations. Later
        # tutorials, such as scenarioAttitudeFeedback.py,
        # illustrate how to over-ride default values with desired simulation values.

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

        logging.debug("Basilisk simulation complete")

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
        Always initialize SPICE interface for accurate positions for the gravitational bodies.
        
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

        # Create the Moon only if useMoon3rdBody == True
        if self.cfg.b_set.useMoon3rdBody:
            moon = gravFactory.createMoon()
        
        # Set Earth as the central gravitational body
        earth.isCentralBody = True

        # Use spherical harmonics if useSphericalHarmonics == True
        if self.cfg.b_set.useSphericalHarmonics:
            # If extra customization is required, see the createEarth() macro to change additional values.
            earth.useSphericalHarmonicsGravityModel(
                GRAV_COEFF_FILE_PATH, 
                self.cfg.b_set.sphericalHarmonicsDegree
            )

            # The value 2 indicates that the first two harmonics, excluding the 0th order harmonic,
            # are included.  This harmonics data file only includes a zeroth order and J2 term.
        
        # Initialize SPICE publisher to get accurate positions of the planets defined within gravFactory. 
        spicePath = os.path.join(__path__[0], "supportData", "EphemerisData") + os.sep
        spiceKernels = ["de430.bsp", "naif0012.tls", "de-403-masses.tpc", "pck00010.tpc"]
        spiceTime = self.to_spice_utc(self.cfg.startTime)
        
        # Will always create SPICE objects "earth" and "sun". "moon" is created if useMoon3rdBody == True
        spiceObj = gravFactory.createSpiceInterface(
            path=spicePath,
            time=spiceTime,
            spiceKernelFileNames=spiceKernels,
            epochInMsg=False
        )
        spiceObj.zeroBase = "earth"
        
        # Schedule object to simualtion process
        self.scSim.AddModelToTask(self.simTaskName, spiceObj)


        return gravFactory, spiceObj


    def conditional_atmosphere_init(self) -> Optional[exponentialAtmosphere.ExponentialAtmosphere]:
        
        # If the simulation is configured to not use drag, return None
        if not self.cfg.b_set.useExponentialDensityDrag:
            return None
        
        # Initialize ExponentialAtmosphere object
        atm = exponentialAtmosphere.ExponentialAtmosphere()

        # Atmosphere parameters
        atm.ModelTag = "expAtm"
        atm.planetRadius = 6378136.6            # [m] WGS-84 equatorial radius
        atm.scaleHeight = 7200.0                 # [m] typical scale height
        atm.baseDensity = 1.225                  # [kg/m^3] density at 0 m
        atm.envMinReach = 0.0                    # [m]
        atm.envMaxReach = 1000e3             # [m] cap model above 1000 km
        self.scSim.AddModelToTask(self.simTaskName, atm)

        return atm
    

    def conditional_drag_effector(self, 
                                  sat: Satellite,
                                  scObj: spacecraft.Spacecraft,
                                  atm: Optional[exponentialAtmosphere.ExponentialAtmosphere]
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
        
        if (not self.cfg.b_set.useExponentialDensityDrag) or (atm is None):
            return scObj
        
        if not (isinstance(atm, exponentialAtmosphere.ExponentialAtmosphere)):
            raise TypeError("The atmosphere object 'atm' is not of type exponentialAtmosphere.ExponentialAtmosphere")

        # ---- Drag effector (exponential density + cannonball) ----
        # Register this spacecraft with the atmosphere model to get its own atm mesg
        atm.addSpacecraftToModel(scObj.scStateOutMsg)

        # Define drag
        drag = dragDynamicEffector.DragDynamicEffector()
        drag.cannonballDrag()

        # Set core parameters
        core = dragDynamicEffector.DragBaseData()
        core.dragCoeff = getattr(sat, "C_D", 2.2)
        core.projectedArea = getattr(sat, "A_D", 0.06)
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


        ####### FOR DEBUG ###############################
        # earthMsg = spiceObj.planetStateOutMsgs[0]
        # sunMsg   = spiceObj.planetStateOutMsgs[1]
        # try:
        #     moonMsg = spiceObj.planetStateOutMsgs[2]
        # except:
        #     logging.debug("The Moon gravitational entity is not defined in the SPICE interface")
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
        srp.coefficientReflection = getattr(sat, "C_R", 1.21)
        srp.area = getattr(sat, "A_srp", 0.06)  

        # Subscribe to Sun ephemeris + this spacecraft’s eclipse factor
        srp.sunEphmInMsg.subscribeTo(sunMsg)
        srp.sunEclipseInMsg.subscribeTo(eclipseObj.eclipseOutMsgs[-1])  # last added = this SC

        # Mount SRP onto the spacecraft and schedule it
        scObj.addDynamicEffector(srp)
        self.scSim.AddModelToTask(self.simTaskName, srp)

        return scObj


    def conditional_object_integrator(self, scObj: spacecraft.Spacecraft) -> spacecraft.Spacecraft:
        
        integration_method = self.cfg.b_set.integrator

        # Select integration method
        match integration_method:
            case "RKF45":
                logging.debug("Selecting RKF45 numerical integrator")
                integratorObj = svIntegrators.svIntegratorRKF45(scObj)
            case "RKF78":
                logging.debug("Selecting RKF78 numerical integrator")
                integratorObj = svIntegrators.svIntegratorRKF78(scObj)
            case _:
                logging.debug("Selecting defualt RK4 numerical integrator")
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

        oe.a = rLEO
        oe.e = 0.001
        oe.i = 33.3 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R
        oe.f = oe.f - satellite_idx * separation_ang

        rN, vN = orbitalMotion.elem2rv(mu, oe)

        return rN, vN
    

    @staticmethod
    def custom_initial_states(satellite_idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Edit this function to manually output the initial states for the satellites
        
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
        # s like "02.04.2025 12:00:00" (DD.MM.YYYY HH:MM:SS) in local time (Europe/Oslo)?
        dt_local = datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
        # If the string is *already* UTC, replace with timezone.utc directly.
        dt_utc = dt_local.replace(tzinfo=timezone.utc)
        return dt_utc.strftime("%Y %b %d %H:%M:%S UTC")