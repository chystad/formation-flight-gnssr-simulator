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

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass, simulationArchTypes, macros, unitTestSupport, vizSupport
from Basilisk.simulation import spacecraft

from object_definitions.Config_def import Config
from object_definitions.FswStack_def import FswStack
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimData, SimObjData
from object_definitions.BasiliskDynamicsModel_def import BasiliskDynamicsModel
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle
from object_definitions.BasiliskEnvironmentModel_def import BasiliskEnvironmentModel


VIZARD_SAVE_PATH = "/home/chris/code/formation-flight-gnssr-simulator/Formation_Flying_Energy_Analysis/output_data/_VizFiles/bsk_sim.bin"

# Model rates [sec] TODO: Move to Config
ENV_RATE: float = 0.5 # Update rate for environment models
DYN_RATE: float = 0.5 # Update rate for dynamical models
FSW_RATE: float = 0.5 # Update rate for flight software stack
REL_NAV_RATE: float = 0.5 # TODO: Update rate for the formation flight stack 
MSIS_RATE: float = 30. # Update rate for MSIS input parameters

TRANS_SAMPLE_RATE: float = 30. # NOTE: Must be integer multilple of 'DYN_RATE'
ATT_SAMPLE_RATE: float = 30. # NOTE: Must be integer multilple of 'DYN_RATE'
POWER_SAMPLE_RATE: float = 30. # NOTE: Must be integer multilple of 'DYN_RATE'


class BasiliskSimulator(SimulationBaseClass.SimBaseClass):
    """
    =========================================================================================================
    Scenario orchestrator for the multi-satellite Basilisk simulation.

    Ownership model:
        - this class owns simulation time, tasks, processes, initialization, execution,
          scenario-level coordination, and output collection.
        - BasiliskEnvironmentalModels owns shared environmental models/messages.
        - BasiliskDynamicModels owns one spacecraft's physical plant/effectors/recorders.
        - FswStack owns one spacecraft's software/control pipeline.

    ATTRIBUTES:
        cfg                 (Config) Global config instance 
        
        --- Time attributes ---
        envRate             (float) TODO
        dynRate             (float) TODO
        fswRate             (float) TODO
        relNacRate          (float) TODO
        spiceTime           (str) Time string used to initialize the SpiceInterface
        epochMsg            (messaging.EpochMsg) Centralized epoch message used by all models
        simulationTimeStepNanos
        simulationDurationSec
        sampintTime
        _simStartDt         (datetime) Simulation start time helper
        _simEndDt           (datetime) Simulation end time helper

    
        
       
    Process-Task-Structure:

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
    |
    |---DynamicsProcess_<sat_idx>
        |
        |---DynamicsTask_<sat_idx>
            |
            |---scObj
            |---drag      (optional)
            |---srp       (optional)
            |---rwEffector
            |---solarPanel(s)
            |---rw power model(s)
            |---power sink
            |---battery
            |---recorders
    |
    |---FswProcess_<sat_idx>
        |
        |---FswTask_<sat_idx>
            |
            |---FswStack
    |
    |---FormationNavProcess
        |
        |---TODO
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        logging.debug("[BSK] Setting up modular Basilisk simulation...")

        self.cfg = cfg
        self.numSatellites = cfg.num_satellites
        self.sim_data: Optional[SimData] = None # TODO: I think the entire data structure and saving 
        # should be changed to something more manageble and more compatible with the Basilisk arcitecture
        
        # ------------------------------------------------------------------
        # Time configuration
        # ------------------------------------------------------------------
        self.envRateNanos: int =    macros.sec2nano(ENV_RATE)
        self.dynRateNanos: int =    macros.sec2nano(DYN_RATE)
        self.fswRateNanos: int =    macros.sec2nano(FSW_RATE)
        self.relNavRateNanos: int = macros.sec2nano(REL_NAV_RATE)
        self.msisRateNanos: int =   macros.sec2nano(MSIS_RATE)

        self.transSampleRateNanos: int = macros.sec2nano(TRANS_SAMPLE_RATE)
        self.attSampleRateNanos: int   = macros.sec2nano(ATT_SAMPLE_RATE)
        self.powerSampleRateNanos: int = macros.sec2nano(POWER_SAMPLE_RATE)
        
        self.spiceTime = self._to_spice_utc(self.cfg.startTime) # Used to initialize SPICE interface
        self.epoch_msg: messaging.EpochMsg = unitTestSupport.timeStringToGregorianUTCMsg(self.spiceTime) # Used for time-dependent models (SPICE interface (eclipse model by extension), MSIS)

        self._simStartDt = datetime.strptime(self.cfg.startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        self._simEndDt = self._simStartDt + timedelta(hours=float(self.cfg.simulationDuration))

        self.simulationTimeStepNanos = macros.sec2nano(self.cfg.deltaT)
        self.simulationDurationSec = float(self.cfg.simulationDuration) * 60.0 * 60.0
        self.simulationDurationNanos = macros.sec2nano(self.simulationDurationSec)

        # Number of data points and recorder sampling time
        numDataPoints = self.simulationDurationNanos // self.simulationTimeStepNanos
        self.samplingTime = unitTestSupport.samplingTime(
            self.simulationDurationNanos,
            self.simulationTimeStepNanos,
            numDataPoints
        )

        # ------------------------------------------------------------------
        # Stable containers
        # ------------------------------------------------------------------
        self.integrators = []
        self.envModel: BasiliskEnvironmentModel
        self.envProcess: simulationArchTypes.ProcessBaseClass
        
        self.dynProcesses: list[Optional[simulationArchTypes.ProcessBaseClass]] = [None] * self.numSatellites
        self.dynProcessNames: list[Optional[str]] = [None] * self.numSatellites
        self.fswProcesses: list[Optional[simulationArchTypes.ProcessBaseClass]] = [None] * self.numSatellites
        self.fswProcessNames: list[Optional[str]] = [None] * self.numSatellites
        self.scRuntimeBundles: list[Optional[SpacecraftRuntimeBundle]] = [None] * self.numSatellites

        
        
        
        
        # # Convenience containers kept for backward-compatible downstream usage
        # self.scObjects: list[Any] = []
        # self.scRecorders: list[Any] = []
        # self.atmRecorders: list[Any] = []
        # self.rwMotorRecorders: list[Any] = []
        # self.attErrRecorders: list[Any] = []
        # self.snTransRecorders: list[Any] = []
        # self.mrpRecorders: list[Any] = []
        # self.rwRecorders: list[list[Any]] = []
        # self.allSpRecorders: list[list[Any]] = []
        # self.allRwPowRecorders: list[list[Any]] = []
        # self.psRecorders: list[Any] = []
        # self.batRecorders: list[Any] = []
        # self.fswStacks: list[FswStack] = []
        # self.dynModels: list[BasiliskDynamicModels] = []

        # # Model and coresponding process containers
        # self.DynModels = []
        # self.FSWModels = []
        # self.envProcessName: Optional[str] = None
        # self.DynamicsProcessName = []
        # self.FSWProcessName = []
        # self.envProcess = None
        # self.dynProcess = []
        # self.fswProcess = []

        # ------------------------------------------------------------------
        # 1) Shared environmental models
        # ------------------------------------------------------------------
        self.envModel = self._build_environment_model()


        # ------------------------------------------------------------------
        # 2) Per-satellite dynamics + FSW
        # ------------------------------------------------------------------
        for sat_idx, sat in enumerate(self.cfg.satellites):

            # Build per-satellite components, dynamics and FSW, then bundle
            dynModel =        self._build_spacecraft_dynamics_model(sat_idx, sat)
            fsw =             self._build_spacecraft_fsw(sat_idx, sat, dynModel)
            scRuntimeBundle = self._build_spacecraft_runtime_bundle(sat_idx, sat, dynModel, fsw)

            # Add bundle to stable list
            self.scRuntimeBundles[sat_idx] = scRuntimeBundle

            # Stable backward-compatible containers
            # self.dynModels.append(runtime.dyn)
            # self.fswStacks.append(runtime.fsw)
            # self.scObjects.append(runtime.sc_obj)
            # self.scRecorders.append(runtime.sc_state_recorder)
            # self.atmRecorders.append(runtime.atm_recorder)
            # self.rwMotorRecorders.append(runtime.rw_motor_recorder)
            # self.attErrRecorders.append(runtime.att_err_recorder)
            # self.snTransRecorders.append(runtime.nav_trans_recorder)
            # self.mrpRecorders.append(runtime.rw_speed_recorder)
            # self.rwRecorders.append(runtime.rw_recorders)
            # self.allSpRecorders.append(runtime.solar_panel_recorders)
            # self.allRwPowRecorders.append(runtime.rw_power_recorders)
            # self.psRecorders.append(runtime.power_sink_recorder)
            # self.batRecorders.append(runtime.battery_recorder)

        # ------------------------------------------------------------------
        # 3) Visualization
        # ------------------------------------------------------------------
        if len(self.scRuntimeBundles) > 0:
            self._configure_vizard()
            

        # ------------------------------------------------------------------
        # 4) Initialize and configure stop time
        # ------------------------------------------------------------------
        self.SetProgressBar(True)
        self.InitializeSimulation()
        self.ConfigureStopTime(self.simulationDurationNanos)

        logging.debug("[BSK] Modular Basilisk simulation setup complete")

    # ======================================================================
    # Public functions
    # ======================================================================

    def run(self) -> None:
        """
        Execute the simulation and collect position/velocity results
        into self.sim_data in the same overall format as before.
        """
        logging.debug("[BSK] Running Basilisk simulation...")
        self.ExecuteSimulation()

        logging.debug("[BSK] Basilisk simulation complete")




    # ======================================================================
    # Private scenario construction functions
    # ======================================================================

    def _build_environment_model(self) -> BasiliskEnvironmentModel:
        """
        Create the shared environmental model object and register its models
        on the 'self.envProcessName' process

        The method assigns the sim attributes:
            self.envProcessName (str): Name of the environment model process
            self.envProcess (simulationArchTypes.ProcessBaseClass): The environment process instance
            self.envModels (BasiliskEnvironmentalModels): The initialized environmental models
        """
        self.envProcessName = "EnvironmentProcess"
        self.envProcess = self.CreateNewProcess(self.envProcessName, 100)

        envModel = BasiliskEnvironmentModel(
            sim=self,
            cfg=self.cfg,
        )

        return envModel
    

    def _build_spacecraft_dynamics_model(self, sat_idx: int, sat: Satellite) -> BasiliskDynamicsModel:
        """
        Build the dynamics model for the current spacecraft
        NOTE: Method is expected to be called from within a cfg.satellites loop
        """
        # Create dynamics process for the spacecraft and its components/effectors. Assign to persistent lists
        dynProcessName = f"DynamicsProcess_{sat_idx}"
        self.dynProcessNames[sat_idx] = dynProcessName
        self.dynProcesses[sat_idx] = self.CreateNewProcess(dynProcessName)

        dynModel = BasiliskDynamicsModel(
            sim=self,
            cfg=self.cfg,
            sat=sat,
            sat_idx=sat_idx,
        )

        return dynModel


    def _build_spacecraft_fsw(self, sat_idx: int, sat: Satellite, dynModel: BasiliskDynamicsModel) -> FswStack:
        """
        Build the FSW stack for the current spacecraft, and connect it to the attitude effector
        NOTE: Method is expected to be called from within a cfg.satellites loop
        """
        
        # Make sure the required message handles have been initialized
        assert dynModel.sc_state_out_msg is not None
        assert dynModel.rw_speed_out_msg is not None
        assert dynModel.rw_config_msg is not None
        assert dynModel.bat_state_msg is not None
        assert dynModel.sun_eclipse_msg is not None
        assert self.envModel.spiceObj is not None

        # Create FSW process for the spacecraft local GNC system. Assign to persistent lists
        fswProcessName = f"FswProcess_{sat_idx}"
        self.fswProcessNames[sat_idx] = fswProcessName
        self.fswProcesses[sat_idx] = self.CreateNewProcess(fswProcessName)

        fsw = FswStack(
            sim = self,
            sat = sat,
            sat_idx = sat_idx,
            sc_state_out_msg = dynModel.sc_state_out_msg,
            rw_speed_out_msg = dynModel.rw_speed_out_msg,
            rw_config_msg = dynModel.rw_config_msg,
            bat_state_msg = dynModel.bat_state_msg,
            gs_access_msgs = dynModel.gs_access_msgs,
            gs_state_msgs = self.envModel.gs_state_msgs,
            sun_eclipse_msg = dynModel.sun_eclipse_msg,
            sun_state_msg = self.envModel.spiceObj.planetStateOutMsgs[self.envModel.sun_idx], # TODO: Make attribute of env
            log_timestamp = self.cfg.timestamp_str
        )
        dynModel.connect_fsw(fsw)

        return fsw

    
    def _build_spacecraft_runtime_bundle(self, sat_idx: int, sat: Satellite, 
                                         dynModel: BasiliskDynamicsModel, fsw: FswStack
                                         ) -> SpacecraftRuntimeBundle:
        """
        Bundle per-satellite models together into a SpacecraftRuntimeBundle instance 
        NOTE: Method is expected to be called from within a cfg.satellites loop
        """

        # Construct spacecraft runtime bundle
        scRuntimeBundle = SpacecraftRuntimeBundle(
            sat_idx = sat_idx,
            sat = sat,
            scObj = dynModel.scObj,
            dynModel = dynModel,
            fsw = fsw
        )

        return scRuntimeBundle


    def _extract_scObjs_from_scRuntimeBundles(self) -> list[spacecraft.Spacecraft]:
        """
        Extract a list of Spacecraft instances from 'self.scRuntimeBundles'.
        """
        
        if (len(self.scRuntimeBundles) == 0) or (None in self.scRuntimeBundles):
            raise AttributeError(f"""[BSK] 'self.scRuntimeBundles' does not contain enough initialized SpacecraftRuntimeBundle-s. 
                                 -> Cannot extract scObj list""")
        
        if len(self.scRuntimeBundles) != self.numSatellites:
            raise AttributeError(f"""[BSK] self.scRuntimeBundles' contains {len(self.scRuntimeBundles)} instances, 
                                 expected the same as number of satellites ({self.numSatellites}). 
                                 -> Cannot extract scObj list""")
        
        # Get spacecraft objects
        scObjs: list[spacecraft.Spacecraft] = []
        for i, sc in enumerate(self.scRuntimeBundles):
            assert sc is not None # actually obsolete, but included to make pylance happy
            scObjs.append(sc.scObj)

        return scObjs


    # ======================================================================
    # Private helper
    # ======================================================================

    def _configure_vizard(self):
        """
        Configure Vizard for the current multi-spacecraft simulation.

        Features:
            - one stacked storage panel per spacecraft
                * battery charge
                * fuel level
            - RW panels shown by default
            - spacecraft and planet coordinate axes visible by default
            - orbit/trajectory lines shown
        """
        if len(self.scRuntimeBundles) == 0 or None in self.scRuntimeBundles:
            raise RuntimeError("[BSK] Cannot configure Vizard before all spacecraft runtime bundles exist.")

        if not vizSupport.vizFound:
            logging.warning("[BSK] Vizard support not available. Skipping Vizard configuration.")
            self.viz = None
            self.vizGenericStorageList = []
            return None

        sc_bundles = [bundle for bundle in self.scRuntimeBundles if bundle is not None]
        sc_objs = [bundle.scObj for bundle in sc_bundles]

        rw_effector_list = []
        generic_storage_list = []

        # Keep Python-side references alive for the whole simulation
        self.vizGenericStorageList = generic_storage_list

        # Only include thruster list if all spacecraft actually have one
        use_thrusters = all(bundle.dynModel.thrusterEffector is not None for bundle in sc_bundles)
        thr_effector_list = []

        for bundle in sc_bundles:
            dyn = bundle.dynModel

            rw_effector_list.append(dyn.rwEffector)

            if use_thrusters:
                # Vizard expects a list-of-lists, one inner list per spacecraft
                thr_effector_list.append([dyn.thrusterEffector])

            sc_storage_panels = []

            # -------------------------------------------------
            # Battery storage bar
            # -------------------------------------------------
            if dyn.battery is not None:
                battery_panel = vizSupport.vizInterface.GenericStorage()
                battery_panel.label = "Battery"
                battery_panel.units = "W-s"
                battery_panel.color = vizSupport.vizInterface.IntVector(
                    vizSupport.toRGBA255("red") +
                    vizSupport.toRGBA255("orange") +
                    vizSupport.toRGBA255("green")
                )
                battery_panel.thresholds = vizSupport.vizInterface.IntVector([20, 60])

                battery_in_msg = messaging.PowerStorageStatusMsgReader()
                battery_in_msg.subscribeTo(dyn.battery.batPowerOutMsg)
                battery_panel.batteryStateInMsg = battery_in_msg

                battery_panel.this.disown() # type: ignore
                sc_storage_panels.append(battery_panel)

            # -------------------------------------------------
            # Fuel tank storage bar
            # -------------------------------------------------
            if dyn.fuelTankEffector is not None:
                fuel_panel = vizSupport.vizInterface.GenericStorage()
                fuel_panel.label = "Fuel"
                fuel_panel.units = "kg"
                fuel_panel.color = vizSupport.vizInterface.IntVector(
                    vizSupport.toRGBA255("red") +
                    vizSupport.toRGBA255("orange") +
                    vizSupport.toRGBA255("cyan")
                )
                fuel_panel.thresholds = vizSupport.vizInterface.IntVector([20, 60])

                fuel_in_msg = messaging.FuelTankMsgReader()
                fuel_in_msg.subscribeTo(dyn.fuelTankEffector.fuelTankOutMsg)
                fuel_panel.fuelTankStateInMsg = fuel_in_msg

                fuel_panel.this.disown() # type: ignore
                sc_storage_panels.append(fuel_panel)

            generic_storage_list.append(sc_storage_panels)

        first_dyn_task_name = sc_bundles[0].dynModel.dynTaskName

        if use_thrusters:
            self.viz = vizSupport.enableUnityVisualization(
                self,
                first_dyn_task_name,
                sc_objs,
                saveFile=VIZARD_SAVE_PATH,
                rwEffectorList=rw_effector_list,
                thrEffectorList=thr_effector_list,
                genericStorageList=generic_storage_list,
            )
        else:
            self.viz = vizSupport.enableUnityVisualization(
                self,
                first_dyn_task_name,
                sc_objs,
                saveFile=VIZARD_SAVE_PATH,
                rwEffectorList=rw_effector_list,
                genericStorageList=generic_storage_list,
            )

        # -------------------------------------------------
        # General Vizard display defaults
        # -------------------------------------------------
        self.viz.settings.showSpacecraftLabels = True

        # Show Earth-centered orbit/trajectory traces
        self.viz.settings.orbitLinesOn = 1
        self.viz.settings.trueTrajectoryLinesOn = -1

        # Show spacecraft and planet coordinate axes, but hide axis labels
        self.viz.settings.spacecraftCSon = 1
        self.viz.settings.planetCSon = 1
        self.viz.settings.showCSLabels = -1

        if len(sc_objs) > 0:
            self.viz.settings.mainCameraTarget = sc_objs[0].ModelTag
            self.viz.liveSettings.relativeOrbitChief = sc_objs[0].ModelTag

        for bundle in sc_bundles:
            vizSupport.setInstrumentGuiSetting(
                self.viz,
                spacecraftName=bundle.scObj.ModelTag,
                showGenericStoragePanel=True
            )

            vizSupport.setActuatorGuiSetting(
                self.viz,
                spacecraftName=bundle.scObj.ModelTag,
                viewRWPanel=True,
                viewRWHUD=False,
                showRWLabels=False
            )

        logging.debug("[BSK] Vizard configured")
        return self.viz
    

    def _output_data(self) -> None:
        """
        Output simulation data.
        """
        if self.sim_data is None:
            raise ValueError("Simulation data not yet generated. Call run() before _output_data().")

        self.sim_data.write_data_to_file(self.cfg.timestamp_str, "bsk")

    @staticmethod
    def _to_spice_utc(s: str) -> str:
        """
        Convert a config time string into the SPICE time format used by the current codebase.
        """
        dt_local = datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
        dt_utc = dt_local.replace(tzinfo=timezone.utc)
        return dt_utc.strftime("%Y %b %d %H:%M:%S UTC")