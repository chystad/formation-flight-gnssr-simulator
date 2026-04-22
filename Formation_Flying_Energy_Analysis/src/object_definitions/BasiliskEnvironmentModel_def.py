#
#  ISC License
#
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

# Main structure adapted from basilisk/examples/MultiSatBskSim/modelsMultiSat/BSK_EnvironmentEarth.py

from __future__ import annotations
from typing import TYPE_CHECKING

import os
import logging
import numpy as np
from typing import Optional, Any, Union

from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.simulation import (spacecraft, spiceInterface, eclipse,  
                                exponentialAtmosphere, msisAtmosphere, groundLocation)
from Basilisk.utilities import simIncludeGravBody

from object_definitions.Config_def import Config
from object_definitions.MsisInputUpdater_def import (MsisInputUpdater, MSIS_SW_KEYS)
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 


EARTH_RADIUS = 6378136.6 # [m] WGS-84 equatorial radius
GRAV_COEFF_FILE_PATH = "shared_input_data/grav_coeff/GGM03S.txt"


class BasiliskEnvironmentModel:
    """
    Initialize, schedule and store all shared environment-models

    All environment models initialized, and their place in the BasiliskSimulator process/task architecture:
    BasiliskSimulator
    |
    |---EnvironmentProcess
        |
        |---EnvironmentTask
            |
            |---spiceObj
            |---eclipseObj
            |---groundStation(s) 
            |---atmObj           (optional)
        |
        |---MsisInputUpdaterTask (optional)
            |
            |---msisInputUpdater (optional)
    """
    def __init__(self, 
                 sim: BasiliskSimulator,
                 cfg: Config,
                 ) -> None:
    
        self.sim = sim
        self.cfg = cfg

        # Ensure that a dedicated environment process has been created
        assert sim.envProcess is not None

        # Define task names and time-steps
        self.envTaskName = "EnvironmentTask"
        self.msisTaskName = "MsisInputUpdaterTask" # Optional task
        
        # Create task as part of the environment process
        sim.envProcess.addTask(sim.CreateNewTask(self.envTaskName, sim.envRateNanos))

        # Persistent containers for models
        self.gravFactory: Optional[simIncludeGravBody.gravBodyFactory] = None
        self.spiceObj: Optional[spiceInterface.SpiceInterface] = None
        self.eclipseObj: Optional[eclipse.Eclipse] = None
        self.atmObj: Optional[Union[
                exponentialAtmosphere.ExponentialAtmosphere,
                msisAtmosphere.MsisAtmosphere,
            ]] = None
        self.msisInputUpdater: Optional[MsisInputUpdater] = None

        # Persistent containers for other objects
        self.msisSwWriters: list[messaging.SwDataMsg] = []
        self.msis_sw_msgs: list[messaging.SwDataMsg] = []
        self.groundStations: list[groundLocation.GroundLocation] = []
        self.gs_state_msgs: list[messaging.GroundStateMsg] = []
        
        # Initialize all environmental models sequentially
        self._setup_gravity_and_spice()
        self._setup_eclipse()
        self._setup_atmosphere()
        self._setup_ground_locations()
        
        # Schedule all required models
        sim.AddModelToTask(self.envTaskName, self.spiceObj, 20)
        sim.AddModelToTask(self.envTaskName, self.eclipseObj, 20)
        for gs in self.groundStations:
            sim.AddModelToTask(self.envTaskName, gs, 20)

        # Schedule optional models if they have been initialized
        if self.atmObj is not None:
            sim.AddModelToTask(self.envTaskName, self.atmObj, 20)
        if self.msisInputUpdater is not None:
            # Create a dedicated task for the Msis Updater such that it can run on a slower rate
            sim.envProcess.addTask(sim.CreateNewTask(self.msisTaskName, sim.msisRateNanos)) 
            sim.AddModelToTask(self.msisTaskName, self.msisInputUpdater, 20)
        
        


    ###########################
    # Public helper functions #
    ###########################
    
    def add_spacecraft_to_grav_bodies(self, scObj: spacecraft.Spacecraft) -> None:
        """
        Add spacecraft instance to all planet gravities
        """
        # Ensure that the gravity factory has been initialized
        assert self.gravFactory is not None

        self.gravFactory.addBodiesTo(scObj)

    
    def add_spacecraft_to_eclipse(self, scObj: spacecraft.Spacecraft) -> messaging.EclipseMsg:
        """
        Add spacecraft instance with the shared eclipse model
        """
        # Ensure that an eclipse model has been initialized
        assert self.eclipseObj is not None
        
        # Get spacecraft state message
        sc_state_out_msg: messaging.SCStatesMsg = scObj.scStateOutMsg

        # Add to model
        self.eclipseObj.addSpacecraftToModel(sc_state_out_msg)
        eclipse_out_msg : messaging.EclipseMsg = self.eclipseObj.eclipseOutMsgs[-1]

        logging.debug(f"[ENV] '{scObj.ModelTag}' added to the shared eclipse model")
        return eclipse_out_msg

    
    def add_spacecraft_to_atmosphere(self, scObj: spacecraft.Spacecraft) -> Optional[messaging.AtmoPropsMsg]:
        """
        Add spacecraft instance to the shared atmosphere model. 
        """
        # Ensure that an atmosphere model has been initialized
        if self.atmObj is None:
            return None

        # Get spacecraft state message
        sc_state_out_msg: messaging.SCStatesMsg = scObj.scStateOutMsg

        # Add to model
        self.atmObj.addSpacecraftToModel(sc_state_out_msg)
        atm_out_msg: messaging.AtmoPropsMsg = self.atmObj.envOutMsgs[-1]

        logging.debug(f"[ENV] '{scObj.ModelTag}' added to the shared atmosphere model")
        return atm_out_msg

    
    def connect_spacecraft_to_ground_stations(self, scObj: spacecraft.Spacecraft) -> list[messaging.AccessMsg]:
        """
        Connect spacecraft instance to all ground station(s) and 
        prepair their access messages for the flight software

        Args:
            scObj (Spacecraft): Current Basilisk Spacecraft instance

        Returns:
            list[AccessMsg]: A list of access messages, one for each spacecraft-ground-station pair
        """
        # Get spacecraft state message
        sc_state_out_msg: messaging.SCStatesMsg = scObj.scStateOutMsg

        # Attach spacecraft to ground station(s) and prepair access msgs for fsw 
        gs_access_msgs: list[messaging.AccessMsg] = [] # will contain the access msg for this spacecraft against all ground stations
        for j, gs in enumerate(self.groundStations):
            gs.addSpacecraftToModel(sc_state_out_msg)
            gs_access_msgs.append(gs.accessOutMsgs[-1]) # -1 idx refers to the latest added sc (current iteration sat)

        logging.debug(f"[ENV] '{scObj.ModelTag}' connected to {len(self.groundStations)} ground locations")
        return gs_access_msgs
    



    ###############################################
    # Private Environmental model setup functions #
    ###############################################

    def _setup_gravity_and_spice(self) -> None:
        """
        Initialize a gravBodyFactory and SPICE interface. 
        Always generate the Earth and Sun, but disable the Sun's gravity if useSun3rdBody == False. 
        The Moon is generated iff useMoon3rdBody == True. 
        Modify the Earth's gravity body to include spherical harmonics iff useSphericalHarmonics == True. 
        Always initialize SPICE interface for accurate positions for t he gravitational bodies.

        The method assigns the attributes:
            self.gravFactory (simIncludeGravBody.gravBodyFactory):
                Contains information about all the generated planets
            self.spiceObj (spiceInterface.SpiceInterface)
                Contains ephemeris data for all the generated planets
            self.earth_idx (int): Index used to reference the Earth
            self.sun_idx (int): Index used to reference the Sun
            self.moon_idx (Optional[int]): Index used to reference the Moon
        """        
        # Always generate earth and sun gravitational bodies 
        # (Sun also needed for eclipse model)
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = self.gravFactory.createEarth()
        sun = self.gravFactory.createSun()
        
        
        # Disable the Sun's gravity if useSun3rdBody == False
        if not self.cfg.useSun3rdBody:
            sun.mu = 0
        else:
            logging.debug("[ENV] Sun 3rd body perturbation initialized")

        # Create the Moon only if useMoon3rdBody == True
        if self.cfg.useMoon3rdBody:
            moon = self.gravFactory.createMoon()
            
            logging.debug("[ENV] Moon 3rd body perturbation initialized")
        
        # Set Earth as the central gravitational body
        earth.isCentralBody = True

        # Use spherical harmonics if useSphericalHarmonics == True
        if self.cfg.useSphericalHarmonics:
            earth.useSphericalHarmonicsGravityModel(
                GRAV_COEFF_FILE_PATH, 
                self.cfg.sphericalHarmonicsDegree
            )
            logging.debug(f"[ENV] Earth created with spherical harmonics gravity model of order and degree {self.cfg.sphericalHarmonicsDegree}")
        
        # Initialize SPICE publisher to get accurate positions of the planets defined within gravFactory. 
        spicePath = os.path.join(__path__[0], "supportData", "EphemerisData") + os.sep
        spiceKernels = ["de430.bsp", "naif0012.tls", "de-403-masses.tpc", "pck00010.tpc"]
        
        # Will always create SPICE objects "earth" and "sun". "moon" is created if useMoon3rdBody == True
        self.spiceObj = self.gravFactory.createSpiceInterface(
            path=spicePath,
            time=self.sim.spiceTime,
            spiceKernelFileNames=spiceKernels,
            epochInMsg=True
        )

        self.spiceObj.zeroBase = "earth"
        self.spiceObj.epochInMsg.subscribeTo(self.sim.epoch_msg)

        self.earth_idx: int = 0
        self.sun_idx: int   = 1
        self.moon_idx: Optional[int] = None
        if self.cfg.useMoon3rdBody: self.moon_idx = 2

        logging.debug("[ENV] Spice interface initialized for all massive bodies")


    def _setup_eclipse(self) -> None:
        """
        Initializes a model for when the Earth eclipses the Sun

        The method assigns the attribute:
            self.eclipseObj (eclipse.Eclipse): The eclipse model including the Earth and Sun
        """
        # Ensure that the spice object has been initialized
        assert self.spiceObj is not None

        # Fetch the Earth's and Sun's position from the SPICE publisher.
        earth_msg = self.spiceObj.planetStateOutMsgs[self.earth_idx]
        sun_msg   = self.spiceObj.planetStateOutMsgs[self.sun_idx]

        # Initialize eclipse mode (when the Earth eclipses the Sun)
        self.eclipseObj = eclipse.Eclipse()
        self.eclipseObj.sunInMsg.subscribeTo(sun_msg)
        self.eclipseObj.addPlanetToModel(earth_msg) # Earth occluder

        logging.debug("[ENV] Eclipse model has been initialized")


    def _setup_atmosphere(self) -> None:
        """
        Initialize an exponential, MSIS or None atmosphere model, depending on the config settings.
        If an MSIS atmosphere is initialized, also create an Msis Input Updater SysModel

        Priority:
            1) Initialize the MSIS model if 'cfg.useMsisDrag' == True
            2) Initialize the Exponential density model if 'cfg.useExponentialDensityDrag' == True

        The method assigns the attributes:
            self.atmObj (Optional[MsisAtmosphere | ExponentialAtmosphere]): 
                Container for the initialized atmosphere
            self.msisInputUpdater (Optional[MsisInputUpdater]):
                Container for the MsisInputUpdater SysModel
        """
        useMsis = self.cfg.useMsisDrag
        useExp = self.cfg.useExponentialDensityDrag            

        # Using MSIS atmosphere model (NRLMSISE-00)
        if useMsis:
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
                self.msis_sw_msgs.append(msg_handle)

                # connect MSIS input i to this publisher
                atm.swDataInMsgs[i].subscribeTo(msg_handle)

            # Subscribe to epoch message
            atm.epochInMsg.subscribeTo(self.sim.epoch_msg)

            # Schedule a new task in the simulation process to update MSIS model inputs at a slow frequency during simulation execution
            self.msisInputUpdater = MsisInputUpdater(self.cfg, self.msisSwWriters)

            logging.debug("[ENV] MSIS atmosphere model has been initialized")

        # Using Exponential density atmosphere
        elif useExp:
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
            logging.debug("[ENV] Exponential atmosphere mgfdgjfodel has been initialized")

        
        # If the simulation is configured to not use drag, return None
        else:
            logging.debug("[ENV] WARNING! No atmosphere model has been initialized")
            atm = None

        # Assign model as attribute
        self.atmObj = atm


    def _setup_ground_locations(self) -> None:
        """
        Initialize all ground stations/locations from config
        It requires self.spiceObj to be initialized first.

        The method populates the persistent containers:
            self.groundStations (list[GroundLocation]):
                Container for all initialized ground locations
            self.gs_state_msgs (list[messaging.GroundStateMsg])
                Container for the position state for each ground location
        """
        # Ensure that the spice object has been initialized
        assert self.spiceObj is not None
        
        # Iterate through config ground stations, and create a ground location for each
        for i, gs in enumerate(self.cfg.ground_stations):
            gsTag = gs.gs_tag
            gsLatRad = np.radians(gs.lat)
            gsLongRad = np.radians(gs.long)
            gsAlt = gs.alt
            gsMinElev = np.radians(gs.min_elev)
            gsMaxRange = gs.max_range

            groundStation = groundLocation.GroundLocation()
            groundStation.ModelTag = gsTag
            groundStation.planetRadius = EARTH_RADIUS
            groundStation.specifyLocation(
                gsLatRad, 
                gsLongRad, 
                gsAlt )
            groundStation.planetInMsg.subscribeTo(self.spiceObj.planetStateOutMsgs[self.earth_idx])
            groundStation.minimumElevation = gsMinElev
            groundStation.maximumRange = gsMaxRange

            # Append to stable list
            self.groundStations.append(groundStation)
            self.gs_state_msgs.append(groundStation.currentGroundStateOutMsg)

        logging.debug(f"[ENV] {len(self.groundStations)} ground stations have been initialized")

