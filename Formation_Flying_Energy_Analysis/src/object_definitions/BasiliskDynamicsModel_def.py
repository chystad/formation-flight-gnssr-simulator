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

#  Main structure based on basilisk/examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py

from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from enum import Enum
from typing import Optional, Any, TypeAlias

from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.simulation import (spacecraft, radiationPressure, spiceInterface, eclipse,  
                                exponentialAtmosphere, msisAtmosphere, dragDynamicEffector, 
                                svIntegrators, reactionWheelStateEffector,
                                RWConfigPayload, groundLocation, thrusterDynamicEffector)
from Basilisk.simulation import (simplePowerSink, simpleSolarPanel, simpleBattery, ReactionWheelPower, fuelTank)
from Basilisk.utilities import (orbitalMotion, 
                                unitTestSupport, simIncludeRW, simIncludeThruster)

BasiliskRecorder: TypeAlias = Any # To avoid spreading 'Any' type to make intent clearer

from object_definitions.Config_def import Config
from object_definitions.FswStack_def import FswStack
from object_definitions.Satellite_def import Satellite
# from object_definitions.FormationControlStack_def import FormationControlStack
from object_definitions.BasiliskEnvironmentModel_def import BasiliskEnvironmentModel
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 


ACCEPTED_THRUSTER_MODELS = ['MOOG_Monarc_1', 'MOOG_Monarc_5', 'MOOG_Monarc_22_6', 'MOOG_Monarc_90HT']


class BasiliskDynamicsModel:
    """
    Creates a Basilisk Spacecraft instance, adds it to all initialized environment models 
    and attaches all components/effectors. The dynamics model includes includes:
        * The spacecraft itself 
        * Affecting gravity bodies 
        * Drag effector 
        * SRP effector 
        * Reaction wheel effector 
        * Thruster effector 
        * Solar panels 
        * EPS 
        * Ground locations 

    All dynamics models, and theri place in the BasiliskSimulator process/task architecture:
    BasiliskSimulator
    |
    |---DynamicsProcess_<sat_idx>
        |
        |---DynamicsTask_<sat_idx>
            |
            |---scObj
            |---dragEffector [20]   (optional)
            |---srpEffector [20]    (optional)
            |---solarPanel(s) [20]
            |---rwEffector [20]
            |---thrusterEffector [20]
            |---fuelTankEffector [20]
            |---battery [20]
            |---obcPowerSink [20]
            |---rwPower(s) [20]
            |
            |---thrusterStateRecorder [10]
            |---fuelTankStateRecorder [10]
            |---rwStateRecorder(s) [10]
            |---rwPowerRecorder(s) [10]
            |---batteryStateRecorder [10]
            |---obcPowerSinkRecorder [10]
            |---solarPanelPowerRecorders(s) [10]
    """
    def __init__(self, 
                 sim: BasiliskSimulator,
                 cfg: Config,
                 sat: Satellite,
                 sat_idx: int
                 ) -> None:
        
        self.sim = sim
        self.cfg = cfg
        self.sat = sat
        self.sat_idx = sat_idx
        self.numRWs = len(cfg.spinUVecs)
        self.numSPs = len(cfg.solar_panels)
        self.logTag = f"DYN{sat_idx}"
    
        # Ensure that the environment model has been initialized
        assert sim.envModel is not None
        self.envModel: BasiliskEnvironmentModel = sim.envModel

        # Create dynamics task as part of the dynamics process
        assert sim.dynProcesses[sat_idx] is not None
        self.dynTaskName = f"DynamicsTask_{sat_idx}"
        sim.dynProcesses[sat_idx].addTask(sim.CreateNewTask(self.dynTaskName, sim.dynRateNanos)) # type: ignore

        # Initialize spacecraft instance
        self.scObj = spacecraft.Spacecraft()

        # Disturbance effector modules
        self.dragEffector: Optional[dragDynamicEffector.DragDynamicEffector] = None
        self.srpEffector: Optional[radiationPressure.RadiationPressure] = None
        
        # Satellite actuator modules
        self.rwFactory: Optional[simIncludeRW.rwFactory] = None
        self.rwEffector: Optional[reactionWheelStateEffector.ReactionWheelStateEffector] = None
        self.thrusterFactory: Optional[simIncludeThruster.thrusterFactory] = None
        self.thrusterEffector: Optional[thrusterDynamicEffector.ThrusterDynamicEffector] = None

        # Fuel and Electronic power system modules
        self.fuelTankModel: Optional[fuelTank.FuelTankModelUniformBurn] = None
        self.fuelTankEffector: Optional[fuelTank.FuelTank] = None
        self.solarPanels: list[simpleSolarPanel.SimpleSolarPanel] = []
        self.rwPowerList: list[ReactionWheelPower.ReactionWheelPower] = []
        self.battery: Optional[simpleBattery.SimpleBattery] = None
        self.obcPowerSink: Optional[simplePowerSink.SimplePowerSink] = None


        # Persistent message handles exposed to FSW
        self.sc_state_out_msg: Optional[messaging.SCStatesMsg] = None
        self.rw_speed_out_msg: Optional[messaging.RWSpeedMsg] = None
        self.rw_config_msg: Optional[messaging.RWArrayConfigMsg] = None
        self.bat_state_msg: Optional[messaging.PowerStorageStatusMsg] = None
        self.gs_access_msgs: list[messaging.AccessMsg] = []
        self.sun_eclipse_msg: Optional[messaging.EclipseMsg] = None

        # Recorders owned by this class
        self.batteryStateRecorder: BasiliskRecorder    # Battery charge
        self.fuelTankStateRecorder: BasiliskRecorder   # Remaining propellant
        self.thrusterStateRecorder: Optional[BasiliskRecorder] = None   # Thrust, Isp, max thrust, per thruster
        self.rwStateRecorders: list[BasiliskRecorder] = []              # RW configs and speeds, per RW 
        self.rwPowerRecorders: list[BasiliskRecorder] = []              # RW power consumption, per RW
        self.obcPowerSinkRecorder: Optional[BasiliskRecorder] = None    # OBC power consumption
        self.solarPanelPowerRecorders: list[BasiliskRecorder] = []      # Solar panel power generation, per panel
        self.rwSpeedRecorder: Optional[BasiliskRecorder] = None         # (minimal replacement for rwStateRecorders) RW speeds
        
        # TODO: Move to BasiliskEnvironmentModel
        # self.sunEclipseRecorder: Optional[BasiliskRecorder] = None      # Sun illumination / shadow factor
        # self.groundStationAccessRecorder: list[BasiliskRecorder] = []   # Ground station access, per ground station   
        
        # Initialize all dynamics models sequentially
        self._setup_spacecraft_hub()
        self._setup_gravity()
        self._setup_drag_effector()
        self._setup_srp_effector()
        self._setup_ground_station_access()
        self._setup_solar_panels()
        self._setup_rw_effector()
        self._setup_thrusters()
        self._setup_fuel_tank()
        self._setup_eps()
        self._setup_integrator()
        self._setup_dynamics_recorders()

        # Schedule all initialized modules to task
        sim.AddModelToTask(self.dynTaskName, self.scObj, 20)
        if self.dragEffector is not None: 
            sim.AddModelToTask(self.dynTaskName, self.dragEffector, 20)
        if self.srpEffector is not None:
            sim.AddModelToTask(self.dynTaskName, self.srpEffector, 20)
        for solarPanel in self.solarPanels:
            sim.AddModelToTask(self.dynTaskName, solarPanel, 20)
        sim.AddModelToTask(self.dynTaskName, self.rwEffector, 20)
        sim.AddModelToTask(self.dynTaskName, self.thrusterEffector, 20)
        sim.AddModelToTask(self.dynTaskName, self.fuelTankEffector, 20)
        sim.AddModelToTask(self.dynTaskName, self.battery, 20)
        sim.AddModelToTask(self.dynTaskName, self.obcPowerSink, 20)
        for rwPower in self.rwPowerList:
            sim.AddModelToTask(self.dynTaskName, rwPower, 20)

        # Schedule all recorders to task (lower priority => executes after models)
        sim.AddModelToTask(self.dynTaskName, self.batteryStateRecorder, 10)
        sim.AddModelToTask(self.dynTaskName, self.fuelTankStateRecorder, 10)
        if sim.cfg.data_mode == "debug":
            sim.AddModelToTask(self.dynTaskName, self.thrusterStateRecorder, 10)
            for i in range(self.numRWs):
                sim.AddModelToTask(self.dynTaskName, self.rwStateRecorders[i], 10)
                sim.AddModelToTask(self.dynTaskName, self.rwPowerRecorders[i], 10)
            sim.AddModelToTask(self.dynTaskName, self.obcPowerSinkRecorder, 10)
            for i in range(self.numSPs):
                sim.AddModelToTask(self.dynTaskName, self.solarPanelPowerRecorders[i], 10)
            # sim.AddModelToTask(self.dynTaskName, self.rwSpeedRecorder, 10)  
        



    ###########################
    # Public helper functions #
    ###########################

    def connect_fsw_cmd_to_rw_effector(self, fsw: FswStack) -> None:
        """
        Connect the computed FSW RW torque to the RW effector
        TODO: Connect the computed thrust to the thruster effector
        """
        # Subscribe RWs to motor torque commands from the FSW 
        assert self.rwEffector is not None
        self.rwEffector.rwMotorCmdInMsg.subscribeTo(fsw.rwMotorTorqueOutMsg)

    
    # def connect_form_ctrl_cmds_to_thr_effector(self, formationControl: FormationControlStack) -> None:
    #     """
    #     [DEPRECIATED]
    #     """
    #     pass

    #     # TODO: Uncomment once FSW has been expanded to output thruster commands
    #     # # Subscribe thruster to firing commands from the FSW
    #     # assert self.thrusterEffector is not None
    #     # self.thrusterEffector.cmdsInMsg.subscribeTo(fsw.thrOnTimeCmdOutMsg)

    #     assert self.thrusterEffector is not None
    #     self.thrusterEffector.cmdsInMsg.subscribeTo(
    #         formationControl.form_thr_cmd_out_msgs[self.sat_idx]
    #     )


    def connect_fsw_thr_cmd_to_thr_effector(self, fsw: FswStack) -> None:

        assert self.thrusterEffector is not None
        self.thrusterEffector.cmdsInMsg.subscribeTo(fsw.thrOnTimeCmdOutMsg)




    ##########################################
    # Private dynamics model setup functions #
    ########################################## 

    def _setup_spacecraft_hub(self) -> None:
        """
        Initialize the spacecraft hub and set initial conditions.
        TODO: Modify init contition to use deployer case with random satellite ejection vector

        The method sets the attribute:
            self.scObj (Spacecraft): Spacecraft instance containing hub parameters from cfg and initial conditions
            self.sc_state_out_msg (messaging.SCStatesMsg): Spacecraft state msg
        """
        self.scObj.ModelTag = f"spacecraft_{self.sat_idx}"
        self.scObj.hub.mHub = self.sat.m_s
        self.scObj.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # [m] position vector of body-fixed point B relative to CM
        self.scObj.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.sat.I_B) # [kg m^2] Inertia of hub about point Bc in B frame components

        # Assign initial conditions
        if self.cfg.sat_init_source == "oe":
            # Get initial state vector from orbital elements
            assert self.envModel.gravFactory is not None
            oe = self.sat.init_OEs
            mu = self.envModel.gravFactory.gravBodies["earth"].mu
            rN, vN = orbitalMotion.elem2rv(mu, oe)

        elif self.cfg.sat_init_source == "vec":
            # Use initial state given directly from config
            rN = self.sat.init_pos # [m]   In N frame (inertial = ECI)
            vN = self.sat.init_vel # [m/s] in N frame (inertial = ECI)

        else:
            raise ValueError(f"[{self.logTag}] Unrecognized satellite initial condition source '{self.cfg.sat_init_source}'")

        # Set the initial conditions for the spacecraft object
        self.scObj.hub.r_CN_NInit = rN  # [m]   r_BN_N
        self.scObj.hub.v_CN_NInit = vN  # [m/s] v_BN_N
        self.scObj.hub.sigma_BNInit = self.sat.init_att  # orientation of Body(B) relative to inertial(N) expressed using MRP
        self.scObj.hub.omega_BN_BInit = self.sat.init_angvel  # [rad/s] angular velocity of Body(B) relative to inertial(N) expressed in (B)

        # Assign msg attribute
        self.sc_state_out_msg = self.scObj.scStateOutMsg
        
        logging.debug(f"[{self.logTag}] Spacecraft hub initialized with initial conditions")


    def _setup_gravity(self) -> None:
        """
        Add all configured gravity bodies to the spacecraft.
        """
        self.envModel.add_spacecraft_to_grav_bodies(self.scObj)
        logging.debug(f"[{self.logTag}] Gravity bodies added to '{self.scObj.ModelTag}'") 


    def _setup_drag_effector(self) -> None:
        """
        if the simulation is configured to use an atmosphere model, then define the drag effector,
        mount it on the satellite object, and schedule it in the simulation task
        # TODO: Investigate other effectors than cannonball

        The method creates and populates the attribute:
            self.dragEffector (DragDynamicEffector): 
                Drag effector asserting a force on the spacecraft depending on the atmosphere model density output
        """

        assert self.envModel is not None
        atmObj = self.envModel.atmObj
        useMsis = self.cfg.useMsisDrag
        useExp = self.cfg.useExponentialDensityDrag
        
        if ((not useMsis) and (not useExp)) or (atmObj is None):
            logging.debug(f"""[{self.logTag}] no atmosphere model is initialized 
                          -> Drag disabled for '{self.scObj.ModelTag}'""")
            self.dragEffector = None
            return
        
        if useMsis and (not isinstance(atmObj, msisAtmosphere.MsisAtmosphere)):
            raise TypeError(f"""[{self.logTag}] Basilisk is configured to use an MSIS atmosphere model, but atmosphere object 'atmObj' is not of type 'MsisAtmosphere'
                            -> Drag disabled for '{self.scObj.ModelTag}'""")
        
        elif useExp and (not useMsis) and (not isinstance(atmObj, exponentialAtmosphere.ExponentialAtmosphere)):
            raise TypeError(f"""[{self.logTag}] Basilisk is configured to use an Exponential atmosphere model, but atmosphere object 'atmObj' is not of type 'ExponentialAtmosphere'
                            -> Drag disabled for '{self.scObj.ModelTag}'""")
        
        # Register spacecraft with the shared atmosphere model
        atm_out_msg = self.envModel.add_spacecraft_to_atmosphere(self.scObj)
        assert atm_out_msg is not None

        # Create drag effector
        dragEffector = dragDynamicEffector.DragDynamicEffector()
        dragEffector.ModelTag = f"{self.scObj.ModelTag}_dragEff"
        dragEffector.cannonballDrag() # TODO

        # Set core parameters
        core = dragDynamicEffector.DragBaseData()
        core.dragCoeff = self.sat.C_D # getattr(sat, "C_D", 2.2)
        core.projectedArea = self.sat.A_D # getattr(sat, "A_D", 0.06)
        dragEffector.coreParams = core

        # Subscribe to density from this spacecraft's atmosphere message
        dragEffector.atmoDensInMsg.subscribeTo(atm_out_msg)

        self.scObj.addDynamicEffector(dragEffector)
        self.dragEffector = dragEffector
        
        logging.debug(f"[{self.logTag}] Drag effector initialized for '{self.scObj.ModelTag}'")


    def _setup_srp_effector(self) -> None:
        """
        If the simulation is configured to use SRP, then define the SRP effector and
        mount it on the spacecraft instance

        The method sets the following attributes:
            self.srpEffector (RadiationPressure): 
                SRP effector asserting a force on the spacecraft depending on the Sun illumination
            self.sun_eclipse_msg (EclipseMsg): Message containing the illumination factor from the Sun on the sc
        """

        # Don't mount SRP effector on the spacecraft object if useSRP == False or any Optional inputs are None
        if (not self.cfg.useSRP) or (self.envModel.eclipseObj is None):
            logging.debug(f"""[{self.logTag}] SRP disabled for '{self.scObj.ModelTag}'""")
            self.srpEffector = None
            self.sun_eclipse_msg = None
            return
        
        # Register spacecraft with the shared eclipse model
        sun_eclipse_msg = self.envModel.add_spacecraft_to_eclipse(self.scObj)

        # Latest eclipseOutMsg belongs to this spacecraft because it was just added
        assert self.envModel.spiceObj is not None
        sun_msg = self.envModel.spiceObj.planetStateOutMsgs[self.envModel.sun_idx]

        srpEffector = radiationPressure.RadiationPressure()
        srpEffector.setUseCannonballModel()
        srpEffector.coefficientReflection = self.sat.C_R
        srpEffector.area = self.sat.A_srp

        # Subscribe to Sun ephemeris + this spacecraft’s eclipse factor
        srpEffector.sunEphmInMsg.subscribeTo(sun_msg)
        srpEffector.sunEclipseInMsg.subscribeTo(sun_eclipse_msg)

        # Attach SRP effector to spacecraft and set attributes
        self.scObj.addDynamicEffector(srpEffector)
        self.srpEffector = srpEffector
        self.sun_eclipse_msg = sun_eclipse_msg
        
        logging.debug(f"[{self.logTag}] SRP effector initialized for '{self.scObj.ModelTag}'")


    def _setup_ground_station_access(self) -> None:
        """
        Connect spacecraft to all shared ground stations.
        """
        gs_access_msgs = self.envModel.connect_spacecraft_to_ground_stations(self.scObj)

        if gs_access_msgs is None:
            self.gs_access_msgs = []
        else:
            self.gs_access_msgs = gs_access_msgs

        logging.debug(f"""[{self.logTag}] Ground-station access hookup complete for '{self.scObj.ModelTag}' with {len(self.gs_access_msgs)} ground locations""")


    def _setup_solar_panels(self) -> None:
        """
        Create and attach all solar panels.

        The method sets the following attributes:
            self.solarPanels (list[SimpleSolarPanel]): All solar panel instances, one for each solar panel in cfg
        """
        assert self.envModel.spiceObj is not None

        if not self.cfg.useSRP:
            # Without SRP effector, the panels do strictly require SRP force modeling.
            # -> Only require sun state from SPICE
            sun_eclipse_msg = None
        else:
            assert self.sun_eclipse_msg is not None
            sun_eclipse_msg = self.sun_eclipse_msg

        sun_msg = self.envModel.spiceObj.planetStateOutMsgs[self.envModel.sun_idx]

        # Initialize solar panels
        self.solarPanels = []
        for i, sp in enumerate(self.cfg.solar_panels):

            # Generate solar panel module
            solarPanel = simpleSolarPanel.SimpleSolarPanel()
            solarPanel.ModelTag = f"{self.scObj.ModelTag}_sp{i}"

            solarPanel.stateInMsg.subscribeTo(self.scObj.scStateOutMsg)
            solarPanel.sunInMsg.subscribeTo(sun_msg)

            # If eclipse exists, use it. 
            if sun_eclipse_msg is not None:
                solarPanel.sunEclipseInMsg.subscribeTo(sun_eclipse_msg)
            
            solarPanel.setPanelParameters(sp.nHat_B, sp.panel_area, sp.panel_efficiency)

            self.solarPanels.append(solarPanel)

        if len(self.solarPanels) != self.numSPs:
            raise ValueError(f"[{self.logTag}] The number of initalized solar panels ({len(self.solarPanels)}) is not the same as its own attribute self.numSPs ({self.numSPs})")

        logging.debug(f"""[{self.logTag}] {len(self.solarPanels)} solar panel(s) initialized for '{self.scObj.ModelTag}'""")


    def _setup_rw_effector(self) -> None:
        """
        Generate reaction wheel object(s) using the config parameters, create the RW state effector and 
        add it to the spacecraft.

        This method sets the following attributes:
            self.rwFactory (rwFactory): The factory used to generate all RWs for this spacecraft
            self.rwEffector (ReactionWheelStateEffector): RW effector applying a torque on the spacecraft
            self.rw_speed_out_msg (messaging.RWSpeedMsg): Msg containing the individual RW's angular velocity
            self.rw_config_msg (messaging.RWArrayConfigMsg): Msg containing the RW cluster configuration
        """
        
        rwFactory = simIncludeRW.rwFactory()

        # Select internal RW physics model
        match self.cfg.RW_model:
            case "BalancedWheels":
                varRWModel: int = messaging.BalancedWheels
            case "JitterSimple":
                varRWModel: int = messaging.JitterSimple
            case "JitterFullyCoupled":
                varRWModel: int = messaging.JitterFullyCoupled
            case _:
                raise ValueError(f"Unrecognized 'RW_model' received."
                                f"Got '{self.cfg.RW_model}', expected ['BalancedWheels', 'JitterSimple', 'JitterFullyCoupled']")
            
        # Create RWs (one for each vector present in self.cfg.spinUVecs)
        RWs: list[RWConfigPayload.RWConfigPayload] = []
        for i, spinUVec in enumerate(self.cfg.spinUVecs):
            if self.cfg.useFriction:
                RW = rwFactory.create(
                    'custom',
                    spinUVec,
                    label =         f"sc{self.sat_idx}W{i}",
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
                    label =         f"sc{self.sat_idx}W{i}",
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

        if len(RWs) != self.numRWs:
            raise ValueError(f"[{self.logTag}] The number of initalized RWs ({len(RWs)}) is not the same as its own attribute self.numRWs ({self.numRWs})")

        # Create RW effector and attach to the spacecraft 
        rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwEffector.ModelTag = f"{self.scObj.ModelTag}_rwEff"
        rwFactory.addToSpacecraft(self.scObj.ModelTag, rwEffector, self.scObj)        

        # Set attributes
        self.rwFactory = rwFactory
        self.rwEffector = rwEffector
        self.rw_speed_out_msg = rwEffector.rwSpeedOutMsg
        self.rw_config_msg = rwFactory.getConfigMessage()
        
        logging.debug(f"[{self.logTag}] Reaction wheel effector with {len(RWs)} RWs initialized for '{self.scObj.ModelTag}'")


    def _setup_thrusters(self) -> None:
        """
        Create and attach a custom-parameter thruster effector

        This method sets the following attributes:
            self.thrusterFactory (thrusterFactory)
            self.thrusterEffector (ThrusterDynamicEffector)
        """

        # Create fresh factory and dynamics container
        thrusterFactory = simIncludeThruster.thrusterFactory()
        thrusterEffector = thrusterDynamicEffector.ThrusterDynamicEffector()
        thrusterEffector.ModelTag = f"{self.scObj.ModelTag}_thrEff"

        # Override custom parameters with existing Basilisk model
        if self.cfg.thr_model_override in ACCEPTED_THRUSTER_MODELS:
            thr_model = self.cfg.thr_model_override
            thrusterFactory.create(
                thrusterType = thr_model, 
                r_B = self.cfg.thr_pos_B, 
                tHat_B = self.cfg.thr_dir_B,
                useMinPulseTime = False)
        
        # Use custom thruster parameters from config
        else:
            # Create one fully custom thruster using Blank_Thruster
            thrusterFactory.create(
                thrusterType="Blank_Thruster",
                r_B = self.cfg.thr_pos_B,
                tHat_B = self.cfg.thr_dir_B,
                useMinPulseTime = self.cfg.use_min_pulse_time,
                MinOnTime = self.cfg.min_pulse_time,
                MaxThrust = self.cfg.max_thrust,
                thrBlowDownCoeff = self.cfg.thrust_blowdown_coeff,
                steadyIsp = self.cfg.steady_isp,
                ispBlowDownCoeff = self.cfg.isp_blowdown_coeff,
                areaNozzle = self.cfg.area_nozzle,
                thrusterMagDisp = self.cfg.thr_mag_disp
            )

        # Attach thruster set to spacecraft
        thrusterFactory.addToSpacecraft(
            thrusterEffector.ModelTag,
            thrusterEffector,
            self.scObj
        )

        # Assign attributes
        self.thrusterFactory = thrusterFactory
        self.thrusterEffector = thrusterEffector

        logging.debug(f"[{self.logTag}] Thruster effector initialized for '{self.scObj.ModelTag}'")


    def _setup_fuel_tank(self) -> None:
        """
        TODO: Use custom parameters from config. Now it is just static

        The method sets the attributes:
            self.fuelTankModel (FuelTankModelUniformBurn)
            self.fuelTankEffector (FuelTank)
        """
        # Initialize the fuel tank model and effector
        fuelTankModel = fuelTank.FuelTankModelUniformBurn() # Cylindrical tank
        fuelTankEffector = fuelTank.FuelTank()
        fuelTankEffector.ModelTag = f"{self.scObj.ModelTag}_fuelEff"
        
        fuelTankEffector.setTankModel(fuelTankModel)
        fuelTankModel.maxFuelMass = self.scObj.hub.mHub * 0.05 # [kg] fraction of the total satellite mass
        fuelTankModel.propMassInit = fuelTankModel.maxFuelMass * 1.0 # Fraction of max mass
        fuelTankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        fuelTankEffector.r_TB_B = [[0.0], [0.0], [0.0]]
        fuelTankModel.radiusTankInit = 0.05 # [m] The tank kan only have 1/2 side length radius for a 6U sat
        fuelTankModel.lengthTank = 0.2 # [m] The tank occupies ~2U of the satellite with V = 2pi x 0.05m x 0.2m
        
        # Add the tank and connect the thrusters
        self.scObj.addStateEffector(fuelTankEffector)
        fuelTankEffector.addThrusterSet(self.thrusterEffector)

        # Assign attributes
        self.fuelTankModel = fuelTankModel
        self.fuelTankEffector = fuelTankEffector

        logging.debug(f"[{self.logTag}] Fuel tank initialized for '{self.scObj.ModelTag}'")


    def _setup_eps(self) -> None:
        """
        Initialize all power modules: battery, RW power consumption, OBC power consumption, solar panel charging

        This method sets the following attributres:
            self.battery (SimpleBattery): The spacecraft power storage device. 
                All other power consumers/sources affect its remaining charge
            self.rwPowerList (ReactionWheelPower): Modules converting RW actuation into realistic power consumption
            self.obcPowerSink (SimplePowerSink): Constant power draw representing the OBC consumption
            self.bat_state_msg (messaging.PowerStorageStatusMsg): Remaining battery charge 
        """
        
        assert self.rwFactory is not None
        assert self.rwEffector is not None
        
        # Create a simpleBattery
        battery = simpleBattery.SimpleBattery()
        battery.ModelTag = f"{self.scObj.ModelTag}_battery"
        storageCapacity_Wh = self.cfg.bat_storage_capacity # [Wh]
        storageCapacity_Ws = storageCapacity_Wh * 3600.0 # [Ws = Joules]
        battery.storageCapacity = storageCapacity_Ws
        battery.storedCharge_Init = storageCapacity_Ws * self.cfg.init_bat_charge 

        # Create RW power modules
        rwPowerList: list[ReactionWheelPower.ReactionWheelPower] = []
        numRWs = self.rwFactory.getNumOfDevices()
        for i in range(numRWs):
            rwPower = ReactionWheelPower.ReactionWheelPower()
            rwPower.ModelTag = f"{self.scObj.ModelTag}_RW{i}Power"
            rwPower.basePowerNeed = self.cfg.RW_base_draw # [W]
            rwPower.rwStateInMsg.subscribeTo(self.rwEffector.rwOutMsgs[i])
            
            if False: # TODO: Is it realistic for RWs to generate electricity on smallsats?
                rwPower.mechToElecEfficiency = 0.5 # TODO: Realistic value
            
            rwPowerList.append(rwPower)

        # Constant power consumption from OBC
        obcPowerSink = simplePowerSink.SimplePowerSink()
        obcPowerSink.ModelTag = f"{self.scObj.ModelTag}_OBCPower"
        obcPowerSink.nodePowerOut = -1 * self.cfg.OBC_const_draw  # [W]
        
        # Add all power sources/consumers to battery
        for sp in self.solarPanels:
            battery.addPowerNodeToModel(sp.nodePowerOutMsg)

        for rwPow in rwPowerList:
            battery.addPowerNodeToModel(rwPow.nodePowerOutMsg)
        
        battery.addPowerNodeToModel(obcPowerSink.nodePowerOutMsg)
        
        # Assign attributes
        self.battery = battery
        self.rwPowerList = rwPowerList
        self.obcPowerSink = obcPowerSink
        self.bat_state_msg = battery.batPowerOutMsg

        logging.debug(f"[{self.logTag}] EPS initialized for '{self.scObj.ModelTag}'")         


    def _setup_integrator(self) -> None:
        integration_method = self.cfg.integrator

        # Select integration method
        match integration_method:
            case "RKF45":
                integratorObj = svIntegrators.svIntegratorRKF45(self.scObj)
            case "RKF78":
                
                integratorObj = svIntegrators.svIntegratorRKF78(self.scObj)
            case _:
                logging.debug(f"[{self.logTag}] Selecting defualt RK4 numerical integrator for '{self.scObj.ModelTag}'")
                return # Use standard integration method RK4
        
        # Set the object's non-default integration method
        self.scObj.setIntegrator(integratorObj)

        # Keep a reference so it doesn't get CE'ed
        self.sim.integrators.append(integratorObj)

        logging.debug(f"[{self.logTag}] Selecting {integration_method} numerical integrator for '{self.scObj.ModelTag}'")


    def _setup_dynamics_recorders(self) -> None:
        """
        Initialize all dynamics recorders 

        This method sets the attributes:
            self.thrusterStateRecorder:     Logs thrust force, thrust torque, Isp blowdown, Thruster force blowdown
            self.fuelTankStateRecorder:     Logs fuel mass
            self.rwStateRecorders:          Logs RW spin speed and torque
            self.rwPowerRecorders:          Logs RW net power
            self.batteryStateRecorder:      Logs battery storage level and net power 
            self.obcPowerSinkRecorder:      Logs OBC sink net power
            self.solarPanelPowerRecorders:  Logs Solar panel net power

        NOTE: A possible data collection optimization is to replace the broad 'rwStateRecorders' with the narrow 'rwSpeedRecorder'
              If so, the RW torque must be collected from FSW.
        """
        # Relevant Sample rates
        lowSampleRateNanos = self.sim.lowSampleRateNanos
        midSampleRateNanos = self.sim.midSampleRateNanos
        highSampleRateNanos = self.sim.highSampleRateNanos

        # Set recorder sample rates
        batteryStateRate = midSampleRateNanos # NOTE: This should always be 'midSampleRateNanos' for 'midRateTimes' to be correct in SimData._pull_single_spacecraft_data
        fuelTankStateRate = highSampleRateNanos
        thrusterStateRate = highSampleRateNanos
        rwStateRate = highSampleRateNanos
        rwPowerRate = highSampleRateNanos
        obcPowerSinkRate = midSampleRateNanos
        solarPanelPowerRate = midSampleRateNanos
        rwSpeedRate = highSampleRateNanos
        
        # Verify that rates are exact multiples of dynRate
        if lowSampleRateNanos % self.sim.dynRateNanos != 0.0:
            raise ValueError("'lowSampleRateNanos' is not an exact multiple of 'dynRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'LOW_SAMPLE_RATE' and/or 'DYN_RATE' to fix this error")
        if midSampleRateNanos % self.sim.dynRateNanos != 0.0:
            raise ValueError("'midSampleRateNanos' is not an exact multiple of 'dynRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'MID_SAMPLE_RATE' and/or 'DYN_RATE' to fix this error")
        if highSampleRateNanos % self.sim.dynRateNanos != 0.0:
            raise ValueError("'highSampleRateNanos' is not an exact multiple of 'dynRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'HIGH_SAMPLE_RATE' and/or 'DYN_RATE' to fix this error")
        
            
        # Validate that all necessary modules have been initialized
        assert self.thrusterEffector is not None
        assert self.fuelTankEffector is not None
        assert self.rwEffector is not None
        assert self.battery is not None
        assert self.obcPowerSink is not None
        assert self.rwFactory is not None

        # Mandatory recorders (battery + fuel tank)
        self.batteryStateRecorder = self.battery.batPowerOutMsg.recorder(batteryStateRate) # storageLevel [Ws] + currentNetPower [W]
        self.batteryStateRecorder_RateNanos = batteryStateRate
        self.fuelTankStateRecorder = self.fuelTankEffector.fuelTankOutMsg.recorder(fuelTankStateRate) # attribute: fuelMass [kg]
        self.fuelTankStateRecorder_RateNanos = fuelTankStateRate

        # Optional 'debug' recorders
        if self.sim.cfg.data_mode == "debug":
            # Thruster recorder
            self.thrusterStateRecorder = self.thrusterEffector.thrusterOutMsgs[0].recorder(thrusterStateRate) # attributes: thrustForce_B [N] + thrustBlowDownFactor [%] + ispBlowDownFactor [%] + (thrustTorquePntB_B) [Nm]
            self.thrusterStateRecorder_RateNanos = thrusterStateRate

            # RW power recorders
            for i in range(self.numRWs):
                rwStateRec = self.rwEffector.rwOutMsgs[i].recorder(rwStateRate) # Omega [rad/s] + u_current [Nm]
                rwPowRec = self.rwPowerList[i].nodePowerOutMsg.recorder(rwPowerRate) # netPower [W]
                self.rwStateRecorders.append(rwStateRec)
                self.rwPowerRecorders.append(rwPowRec)
            self.rwStateRecorder_RateNanos = rwStateRate
            self.rwPowerRecorder_RateNanos = rwPowerRate

            # Other Power modules recorders
            self.obcPowerSinkRecorder = self.obcPowerSink.nodePowerOutMsg.recorder(obcPowerSinkRate) # netPower [W]
            self.obcPowerSinkRecorder_RateNanos = obcPowerSinkRate
            for i in range(self.numSPs):
                spPowRec = self.solarPanels[i].nodePowerOutMsg.recorder(solarPanelPowerRate) # netPower [W]
                self.solarPanelPowerRecorders.append(spPowRec)
            self.solarPanelPowerRecorder_RateNanos = solarPanelPowerRate

            # RW speed recorder (use instead of rwStateRecorders if only RW speeds are necessary)
            # self.rwSpeedRecorder = self.rwEffector.rwSpeedOutMsg.recorder(rwSpeedRate) # wheelSpeeds [rot/s OR rad/s, not sure] per wheel
            # self.rwSpeedRecorder_RateNanos = rwSpeedRate

        logging.debug(f"[{self.logTag}] Dynamics recorders initialized for '{self.scObj.ModelTag}'")