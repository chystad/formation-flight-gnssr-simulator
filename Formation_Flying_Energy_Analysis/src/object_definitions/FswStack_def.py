from __future__ import annotations
from typing import TYPE_CHECKING

import os
import csv
import logging
import itertools
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeAlias

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, fswSetupThrusters
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.fswAlgorithms import mrpFeedback, attTrackingError, inertial3D, rwMotorTorque, spacecraftReconfig
from Basilisk.simulation import simpleNav

BasiliskRecorder: TypeAlias = Any # To avoid spreading 'Any' type to make intent clearer

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 

LOG_DATA_SAVE_DIR = Path('Formation_Flying_Energy_Analysis/output_data/logs')

MRP_K: float = 0.05 # MRP pointing controller: Gain on MRP attitude error 
MRP_P: float = 0.035 # MRP pointing controller: Gain on Rate error
MRP_KI: float = -1  # MRP pointing controller: Integral gain (-1 -> disable)
BURN_ATT_ADJUSTMENT_TIME_SEC = 15.0 # [s] Fixed time from the burn is requested until it is executed. 

SHADOWFAC_ENTER_THRESHOLD = 0.6 # The minimum illumination required to enter CHARGE state (0, 1)
SHADOWFAC_EXIT_THRESHOLD = 0.4 # The maximum illumination requred to exit CHARGE state (0, 1)
EMERGENCY_BATTERY_EXIT_THRESHOLD = 0.7 # The lower limit for when the battery is considered to have enough charge to exit EMERGENCY mode (0, 1)
# CAPTURE_BATTERY_THRESHOLD = 0.4 # The minimum battery percentage (inclusive) required for entering CAPTURE mode (0, 1)
CAPTURE_BATTERY_ENTER_THRESHOLD = 0.45
CAPTURE_BATTERY_EXIT_THRESHOLD = 0.38
# COMMS_BATTERY_THRESHOLD = 0.3 # The minimum battery percentage (inclusive) required for entering COMMS mode (0, 1)
COMMS_BATTERY_ENTER_THRESHOLD = 0.35
COMMS_BATTERY_EXIT_THRESHOLD = 0.28
CRITICAL_BATTERY_THRESHOLD = 0.2 # Upper limit (exclusive) for when the battery is considered to have critially low charge left (0, 1)
LOW_BATTERY_THRESHOLD = 0.3 # Upper limit (exclusive) for when the battery is considere to have low charge left (0.1)
MAX_HOURS_SINCE_LAST_COM_THRESHOLD = 12 # Limit (incluse) for when the maximum time has passed since last com. 
                                        # After this, communication will be prioritized over payload capturing.
MIN_MINUTES_COM_TIME = 10. # Minimum time a comunication event should last, if comunication is feasible 


class PointingMode(str, Enum):
    COAST = "coast"
    COMMS = "comms"
    CHARGE = "charge"
    CAPTURE = "capture"
    BURN_TRANSIT = "burn_transit"
    BURN = "burn"
    EMERGENCY = "emergency"
    ERROR = "error"


POINTING_MODE_TO_INT: dict[PointingMode, int] = {
    PointingMode.COAST: 0,
    PointingMode.COMMS: 1,
    PointingMode.CHARGE: 2,
    PointingMode.CAPTURE: 3,
    PointingMode.BURN_TRANSIT: 4,
    PointingMode.BURN: 5,
    PointingMode.EMERGENCY: 6,
    PointingMode.ERROR: 7,
}

INT_TO_POINTING_MODE: dict[int, PointingMode] = {
    value: key for key, value in POINTING_MODE_TO_INT.items()
}


class PointingModeRecorder:
    """
    Lightweight Python recorder for FswStack.pointingMode.

    This is not a Basilisk message recorder. It is a small Python-side recorder
    designed to work with SimData and RecorderFlusher.
    """

    def __init__(self) -> None:
        self.modeCode: list[int] = []
        self.timeNanos: list[int] = []

    def record(self, CurrentSimNanos: int, pointing_mode: PointingMode) -> None:
        self.timeNanos.append(int(CurrentSimNanos))
        self.modeCode.append(POINTING_MODE_TO_INT[pointing_mode])

    def clear(self) -> None:
        self.modeCode.clear()
        self.timeNanos.clear()


class _FswStackScheduler(sysModel.SysModel):
    """
    Small Basilisk-scheduled adapter.

    FswStack itself is a plain owner class, like BasiliskDynamicsModel.
    This object is the scheduled SysModel that calls back into FswStack.
    """
    def __init__(self, owner: "FswStack", sat_idx: int):
        super().__init__()
        self.owner = owner
        self.ModelTag = f"FswStackScheduler_{sat_idx}"

    def UpdateState(self, CurrentSimNanos: int) -> None:
        self.owner._update_state(CurrentSimNanos)

    def SelfInit(self):
        self.owner._self_init()

    def CrossInit(self):
        self.owner._cross_init()

    def Reset(self, CurrentSimNanos: int):
        self.owner._reset(CurrentSimNanos)


class FswStack():
    """
    Plain owner/manager class for one spacecraft's FSW stack.

    Owns:
        - SimpleNav
        - inertial3D guidance
        - attTrackingError
        - mrpFeedback
        - rwMotorTorque
        - mode-switching logic

    BasiliskSimulator
    |
    |---FswProcess_<sat_idx>
        |
        |---FswTask_<sat_idx>
            |
            |---scheduler [20]
            |
            |---navTransRecorder [10]
            |---navAttRecorder [10]
            |---attRefRecorder [10]
            |---attErrRecorder [10]
            |---cmdTorqueRecorder [10]
            |---rwMotorTorqueRecorder [10]
            
    """

    def __init__(
        self,
        sim: BasiliskSimulator,
        sat: Satellite,
        sat_idx: int,
        scModelTag: str,
        sc_state_out_msg: messaging.SCStatesMsg,
        mass_vehicle_config_out_msg: messaging.VehicleConfigMsg,
        rw_speed_out_msg: messaging.RWSpeedMsg,
        rw_config_msg: messaging.RWArrayConfigMsg,
        bat_state_msg: messaging.PowerStorageStatusMsg,
        gs_access_msgs: list[messaging.AccessMsg],
        gs_state_msgs: list[messaging.GroundStateMsg],
        sun_eclipse_msg: messaging.EclipseMsg,
        sun_state_msg: messaging.SpicePlanetStateMsg,
        thr_config_array_msg: messaging.THRArrayConfigMsg,
        fuel_tank_msg: messaging.FuelTankMsg,
        log_timestamp: str,
        DEBUG_sc_I : Any,
    ):
        self.scheduler = _FswStackScheduler(self, sat_idx)
        self.sim = sim
        self.sat = sat
        self.sat_idx = sat_idx
        # self.dynModel = dynModel
        self.massVehicleConfigOutMsg = mass_vehicle_config_out_msg
        self.scModelTag = scModelTag
        self.formEnabled = sim.cfg.form_enabled
        self.batStateMsg = bat_state_msg
        self.gsAccessMsgs = gs_access_msgs
        self.gsStateMsgs = gs_state_msgs
        self.selectedGsIdx: Optional[int] = None
        self.sunEclipseMsg = sun_eclipse_msg
        self.sunStateMsg = sun_state_msg
        self.thrConfigArrayMsg = thr_config_array_msg
        self.fuelTankMsg = fuel_tank_msg
        self.logTimestamp = log_timestamp
        self.lastCommsNanos = 0
        self.currentCommsStartNanos = 0
        self.lowBatIsUnresolved = False # True if the battery has been charged enough to exit the low state
        self.ModelTag = f"RwFswStack{sat_idx}"
        self.logTag = f"FSW{sat_idx}"
        self.pointingMode = PointingMode.COAST

        # Burn request detection
        self.activeBurnRequest = False
        self.activeBurnEventStartNanos: Optional[int] = None
        self.activeBurnDurationS = 0.0
        self.prevFormationOnTimeMax = 0.0

        # Burn attitude request detection
        self.activeBurnAttRequest = False
        self.activeBurnAttRequestStartNanos: Optional[int] = None
        self.prevFormationAttRefSignature = np.zeros(6)
        self.burnDetectedDuringActiveAttRequest = False


        self.DEBUG_sc_I = DEBUG_sc_I

        # Create FSW task as part of the FSW process
        assert sim.fswProcesses[sat_idx] is not None
        self.fswTaskName = f"FswTask_{sat_idx}"
        sim.fswProcesses[sat_idx].addTask(sim.CreateNewTask(self.fswTaskName, sim.fswRateNanos)) # type: ignore

        # Message definition
        self.attRefMsg = None
        self.attGuidMsg = None
        self.com_status_msg: Optional[messaging.DeviceStatusMsg] = None
        self.pay_status_msg: Optional[messaging.DeviceStatusMsg] = None
        self.prop_idle_status_msg: Optional[messaging.DeviceStatusMsg] = None
        self.prop_heat_status_msg: Optional[messaging.DeviceStatusMsg] = None
        self.prop_thr_status_msg: Optional[messaging.DeviceStatusMsg] = None
        
        # Read burn attitude and thrust request from FormationControlStack
        self.formAttRefInMsg: Optional[messaging.AttRefMsg] = None
        self.formThrCmdInMsg: Optional[messaging.THRArrayOnTimeCmdMsg] = None

        # Initialize output thruster command
        self.thrOnTimeCmdOutMsg = messaging.THRArrayOnTimeCmdMsg()
        self.fuelSafeThrCmdOutMsg = messaging.THRArrayOnTimeCmdMsg() # 'Safe' to respect fuel limitations
        init_thr_payload = messaging.THRArrayOnTimeCmdMsgPayload()
        init_thr_payload.OnTimeRequest = [0.0]
        self.thrOnTimeCmdOutMsg.write(init_thr_payload)
        self.fuelSafeThrCmdOutMsg.write(init_thr_payload)

        # Recorders owned by this class
        self.navTransRecorder: BasiliskRecorder          # Position, velocity
        self.navAttRecorder: Optional[BasiliskRecorder] = None            # Attitude, angular rate
        self.attRefRecorder: Optional[BasiliskRecorder] = None            # Desired attitude, desired angular rate
        self.attErrRecorder: Optional[BasiliskRecorder] = None            # Attitude tracking error, angular-rate tracking error
        self.cmdTorqueRecorder: Optional[BasiliskRecorder] = None         # Commanded body torque
        self.rwMotorTorqueRecorder: Optional[BasiliskRecorder] = None     # RW motor torques
        self.pointingModeRecorder = PointingModeRecorder()                # PointingMode, where each mode corresponds to an int
        self.pointingModeRecorder_RateNanos = sim.fswRateNanos


        # Initialize mode switching log file
        self._log_mode_switching_logic(write_header_only=True)

        # -------------------------------------------------
        # Internal FSW modules
        # -------------------------------------------------
        self.nav = simpleNav.SimpleNav()
        self.nav.ModelTag = f"SimpleNavigation_{sat_idx}"

        self.guid = inertial3D.inertial3D()
        self.guid.ModelTag = f"inertial3D_{sat_idx}"

        self.att_err = attTrackingError.attTrackingError()
        self.att_err.ModelTag = f"attErrorInertial3D_{sat_idx}"

        self.ctrl = mrpFeedback.mrpFeedback()
        self.ctrl.ModelTag = f"mrpFeedback_{sat_idx}"

        self.rw_map = rwMotorTorque.rwMotorTorque()
        self.rw_map.ModelTag = f"rwMotorTorque_{sat_idx}"

        self.form_ctrl = spacecraftReconfig.spacecraftReconfig()
        self.form_ctrl.ModelTag = f"formationControl_{sat_idx}"

        # RW mapping configuration (controllable axes)
        self.rw_map.controlAxes_B = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]



        # Vehicle config msg
        vehicle_config_out = messaging.VehicleConfigMsgPayload(ISCPntB_B=sat.I_B)
        self._vc_msg = messaging.VehicleConfigMsg().write(vehicle_config_out)

        # Controller gains
        self.ctrl.K = MRP_K
        self.ctrl.P = MRP_P
        self.ctrl.Ki = MRP_KI
        if self.ctrl.Ki > 0:
            self.ctrl.integralLimit = 2.0 / self.ctrl.Ki * 0.1

        ############### From example scenario 
        # self.decayTime = 50
        # self.xi = 0.9
        # self.ctrl.Ki = -1  # make value negative to turn off integral feedback
        # self.ctrl.P = 2 * np.max(DEBUG_sc_I) / self.decayTime
        # self.ctrl.K = (self.ctrl.P / self.xi) * \
        #                             (self.ctrl.P / self.xi) / np.max(
        #     DEBUG_sc_I)
        ############################

        # Setup models with correct parameters
        self._setup_gateway_msgs()
        self._setup_formation_control()
        self._setup_desired_OE_difference()
        self._setup_eps_components()
        self._setup_fsw_recorders()

        # Message wiring
        self.nav.scStateInMsg.subscribeTo(sc_state_out_msg)
        self.att_err.attNavInMsg.subscribeTo(self.nav.attOutMsg)
        self.att_err.attRefInMsg.subscribeTo(self.attRefMsg) # OLD: self.guid.attRefOutMsg
        self.ctrl.guidInMsg.subscribeTo(self.attGuidMsg) # OLD: self.att_err.attGuidOutMsg
        self.ctrl.vehConfigInMsg.subscribeTo(self._vc_msg)
        self.ctrl.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.ctrl.rwSpeedsInMsg.subscribeTo(rw_speed_out_msg)
        self.rw_map.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.rw_map.vehControlInMsg.subscribeTo(self.ctrl.cmdTorqueOutMsg)

        # Add scheduler and recorders to task (Low priority => Executes last)
        sim.AddModelToTask(self.fswTaskName, self.scheduler, 20)
        sim.AddModelToTask(self.fswTaskName, self.navTransRecorder, 10)
        if sim.cfg.data_mode == "debug":
            sim.AddModelToTask(self.fswTaskName, self.navAttRecorder, 10)
            sim.AddModelToTask(self.fswTaskName, self.attRefRecorder, 10)
            sim.AddModelToTask(self.fswTaskName, self.attErrRecorder, 10)
            sim.AddModelToTask(self.fswTaskName, self.cmdTorqueRecorder, 10)
            sim.AddModelToTask(self.fswTaskName, self.rwMotorTorqueRecorder, 10)

        logging.debug(f"[{self.logTag}] Created FSW stack for '{self.scModelTag}'")


    ###########################
    # Public helper functions #
    ###########################
    
    def connect_chief_trans_to_form_ctrl(self, fswChief: FswStack) -> None:
            """
            Connect the chief translational states to the spacecraftReconfig model

            Args:
                fswChief (FswStack): The chief's FSW stack
            """
            self.form_ctrl.chiefTransInMsg.subscribeTo(fswChief.nav.transOutMsg)



    ##############################
    # SysModel Scheduler methods #
    ##############################

    def _modules(self):
        return [self.nav, self.guid, self.form_ctrl, self.att_err, self.ctrl, self.rw_map]


    def _update_state(self, CurrentSimNanos: int) -> None:
        """
        Run all modules
        """
        self.nav.UpdateState(CurrentSimNanos)
        self._eval_pointing_mode(CurrentSimNanos)
        self._guidance(CurrentSimNanos)
        self.guid.UpdateState(CurrentSimNanos)
        self.form_ctrl.UpdateState(CurrentSimNanos)
        # self._limit_thrust_cmd_by_fuel(CurrentSimNanos)
        self.att_err.UpdateState(CurrentSimNanos)
        self.ctrl.UpdateState(CurrentSimNanos)
        self.rw_map.UpdateState(CurrentSimNanos)
        self.pointingModeRecorder.record(CurrentSimNanos, self.pointingMode)


    def _self_init(self):
        for m in self._modules():
            if hasattr(m, "SelfInit"):
                m.SelfInit()


    def _cross_init(self):
        for m in self._modules():
            if hasattr(m, "CrossInit"):
                m.CrossInit()


    def _reset(self, CurrentSimNanos: int):
        for m in self._modules():
            if hasattr(m, "Reset"):
                m.Reset(CurrentSimNanos)



    ################################
    # Private pointing GNC methods #
    ################################
    
    def _coast_desired_att(self) -> NDArray[np.float64]:
        """
        Point solar panels towards Sun (even though the Earth eclipses the Sun), and try to achieve NADIR antenna pointing
        """
        # TODO: Right now, it is hard-coded that solar panals are at the Z+ face, and the antenna at Y+ face
        #       Make use of 'r_BP_B' and 'r_BA_B' in config to assign this dynamically. 

        # Get spacecraft position relative to Earth in inertial frame 
        r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N)
        r_NB_N = - r_BN_N
        r_NB_N_hat = r_NB_N / np.linalg.norm(r_NB_N)

        # Get Sun position vector relative to the Earth in inertial frame
        r_SN_N = np.array(self.sunStateMsg.read().PositionVector)

        # Unit vector from spacecraft Body to the Sun (desired solar panel direction)
        r_SB_N = r_SN_N - r_BN_N
        r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Want antenna to point nadir as much as possible
        a = r_NB_N_hat - np.dot(r_NB_N_hat, r_SB_N_hat)/np.linalg.norm(r_SB_N_hat)**2 * r_SB_N_hat
        a_hat = a / np.linalg.norm(a)
        
        z_hat = r_SB_N_hat
        y_hat = a_hat
        x_hat = np.cross(y_hat, z_hat)
        z_hat = np.cross(x_hat, y_hat)

        # Direction cosine matrix for the desired attitude
        C_DN_N = np.vstack((x_hat, y_hat, z_hat))

        # Convert into desired Modified Rodrigues Parameters
        mrp_D = rbk.C2MRP(C_DN_N)

        return mrp_D
    

    def _comms_desired_att(self) -> NDArray[np.float64]:
        """
        Point antenna towards available ground station, and try to point solar panels towards the sun
        """
        # TODO: Right now, it is hard-coded that solar panals are at the Z+ face, and the antenna at Y+ face
        #       Make use of 'r_BP_B' and 'r_BA_B' in config to assign this dynamically. 

        # Get the available GS position relative to Earth in inertial frame
        assert self.selectedGsIdx is not None
        r_LN_N = np.array(self.gsStateMsgs[self.selectedGsIdx].read().r_LN_N)

        # Get spacecraft position relative to Earth in inertial frame 
        r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N)
        
        # Unit vector from spacecraft Body to selected ground station (desired antenna direction)
        r_LB_N = r_LN_N - r_BN_N
        r_LB_N_hat = r_LB_N / np.linalg.norm(r_LB_N)

        # Get Sun position vector relative to the Earth in inertial frame
        r_SN_N = np.array(self.sunStateMsg.read().PositionVector)

        # Unit vector from spacecraft Body to the Sun (sun vector)
        r_SB_N = r_SN_N - r_BN_N
        r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Project the sun vector into the plane normal to the desired antenna direction vector)
        # https://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps/ProjectionOfVectorOntoPlane
        s = r_SB_N_hat - np.dot(r_SB_N_hat, r_LB_N_hat)/np.linalg.norm(r_LB_N_hat)**2 * r_LB_N_hat
        s_hat = s / np.linalg.norm(s)

        # TODO: Make flexible for other solar panel / antenna face configurations
        y_hat = r_LB_N_hat
        z_hat = s_hat
        x_hat = np.cross(y_hat, z_hat)
        z_hat = np.cross(x_hat, y_hat)

        # Direction cosine matrix for the desired attitude
        C_DN_N = np.vstack((x_hat, y_hat, z_hat))

        # Convert into desired Modified Rodrigues Parameters
        mrp_D = rbk.C2MRP(C_DN_N)

        return mrp_D
    

    def _charge_desired_att(self) -> NDArray[np.float64]:
        """
        Point solar panels towards Sun, and try to achieve NADIR antenna pointing
        """
        # TODO: Right now, it is hard-coded that solar panals are at the Z+ face, and the antenna at Y+ face
        #       Make use of 'r_BP_B' and 'r_BA_B' in config to assign this dynamically. 

        # Get spacecraft position relative to Earth in inertial frame 
        r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N)
        r_NB_N = - r_BN_N
        r_NB_N_hat = r_NB_N / np.linalg.norm(r_NB_N)

        # Get Sun position vector relative to the Earth in inertial frame
        r_SN_N = np.array(self.sunStateMsg.read().PositionVector)

        # Unit vector from spacecraft Body to the Sun (desired solar panel direction)
        r_SB_N = r_SN_N - r_BN_N
        r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Want antenna to point nadir as much as possible
        a = r_NB_N_hat - np.dot(r_NB_N_hat, r_SB_N_hat)/np.linalg.norm(r_SB_N_hat)**2 * r_SB_N_hat
        a_hat = a / np.linalg.norm(a)
        
        z_hat = r_SB_N_hat
        y_hat = a_hat
        x_hat = np.cross(y_hat, z_hat)
        z_hat = np.cross(x_hat, y_hat)

        # Direction cosine matrix for the desired attitude
        C_DN_N = np.vstack((x_hat, y_hat, z_hat))

        # Convert into desired Modified Rodrigues Parameters
        mrp_D = rbk.C2MRP(C_DN_N)

        return mrp_D


    def _capture_desired_att(self) -> NDArray[np.float64]:
        """
        Point the GNSS-R payload antenna toward nadir (and GNSS receiver zenith) 
        while keeping the solar panels as aligned with the Sun as possible.

        Because the payload is mounted on the same Body-axis as the largest solar panel area, 
        the second largest solar panel area mounted on +X face is directed towards the sun aswell. 

        Desired body-frame alignment:
            - Body -Z axis points toward Earth center
            - Body +X axis points along the Sun vector projected into the plane
            normal to the nadir direction
        """

        # Get spacecraft position relative to Earth in inertial frame
        r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N)

        # Vector from spacecraft body origin B to Earth center E, expressed in N.
        # Here E is the Earth-centered inertial origin, so r_EN_N = 0 and:
        #   r_EB_N = r_EN_N - r_BN_N = -r_BN_N
        r_EB_N = -r_BN_N
        r_EB_N_hat = r_EB_N / np.linalg.norm(r_EB_N)

        # Get Sun position vector relative to Earth in inertial frame
        r_SN_N = np.array(self.sunStateMsg.read().PositionVector)

        # Unit vector from spacecraft body origin B to the Sun, expressed in N
        r_SB_N = r_SN_N - r_BN_N
        r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Project the Sun vector into the plane normal to r_EB_N.
        # This gives the desired +X body-axis direction as close to the Sun as possible,
        # while preserving the nadir-pointing -Z constraint.
        s = r_SB_N_hat - (
            np.dot(r_SB_N_hat, r_EB_N_hat) / np.linalg.norm(r_EB_N_hat) ** 2
        ) * r_EB_N_hat
        s_hat = s / np.linalg.norm(s)

        # Desired body axes expressed in inertial frame.
        # Body -Z points toward Earth center:
        #   -z_hat = r_EB_N_hat  =>  z_hat = -r_EB_N_hat
        x_hat = s_hat
        z_hat = -r_EB_N_hat
        y_hat = np.cross(z_hat, x_hat)

        # Recompute x_hat to enforce orthogonality numerically.
        x_hat = np.cross(y_hat, z_hat)

        # Direction cosine matrix for the desired attitude
        C_DN_N = np.vstack((x_hat, y_hat, z_hat))

        # Convert into desired Modified Rodrigues Parameters
        mrp_D = rbk.C2MRP(C_DN_N)

        return mrp_D


    def _emergency_desired_att(self) -> NDArray[np.float64]:
        # TODO: Turn off power consumption from RWs 
        # NOTE: Right now, same as CHARGE

        # Get spacecraft position relative to Earth in inertial frame 
        r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N)
        r_NB_N = - r_BN_N
        r_NB_N_hat = r_NB_N / np.linalg.norm(r_NB_N)

        # Get Sun position vector relative to the Earth in inertial frame
        r_SN_N = np.array(self.sunStateMsg.read().PositionVector)

        # Unit vector from spacecraft Body to the Sun (desired solar panel direction)
        r_SB_N = r_SN_N - r_BN_N
        r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Want antenna to point nadir as much as possible
        a = r_NB_N_hat - np.dot(r_NB_N_hat, r_SB_N_hat)/np.linalg.norm(r_SB_N_hat)**2 * r_SB_N_hat
        a_hat = a / np.linalg.norm(a)
        
        z_hat = r_SB_N_hat
        y_hat = a_hat
        x_hat = np.cross(y_hat, z_hat)
        z_hat = np.cross(x_hat, y_hat)

        # Direction cosine matrix for the desired attitude
        C_DN_N = np.vstack((x_hat, y_hat, z_hat))

        # Convert into desired Modified Rodrigues Parameters
        mrp_D = rbk.C2MRP(C_DN_N)
        
        return mrp_D
        
    
    def _eval_pointing_mode(self, CurrentSimNanos: int) -> None:
        """
        Updates self.pointingMode based on the objects of intrest within the spacecraft's LOS 
        in accordance with the designed finite state machine.
        """
        
        old_pointing_mode = self.pointingMode
        
        # Initialize logical parameters dependent on burn request from FormationControlStack
        burnRequested = False # True if the thrusters are required to run more than XXX seconds to maintain formation
        burnAttRequested = False # True if a burn attitude is requested
        
        # NOTE: Is this even needed if all parameters are assigned during runtime anyway? 
        # Initialize position-dependent logical parameters 
        canCap = False # TODO
        canCom = False # True if a ground station is within the spacecraft's LOS
        canChar = False # True if the spacecraft is illuminated by the sun with enough intensity

        # Initialize battery-dependent logical parameters
        capBat = False # True if there is enough remaining battery to enter CAPTURE
        comBat = False  # True if there is enough remaining battery to enter COMMS 
        critBat = False  # True if the battery is low enough to enter EMERGENCY
        exitEmergencyFlag = False # True if in EMERGENCY and the battery has been sufficiently charged

        # Initialize time-dependent logical parameters
        maxNoCom = False # True if the duration since exiting COMMS last time exceeds a max threshold
        comEventComplete = True # True if comunication event duration has extended a min threshold
        
        
        # ---- Decide if the spacecraft can capture scientific data (set canCap) ---- 
        canCap = True # This is a major assumption stating that the 
                      # GNSS-constellation-GNSS-R-satellite geometry always allows observations
        # NOTE: This will potentially override other feasible states, so measures must be taken to avoid this


        # ---- Decide if the spacecraft can charge (set canChar) ---- 
        # Fraction of illumination due to eclipse. 0 = fully shadowed, 1 = fully illuminated.
        shadowFac = self.sunEclipseMsg.read().shadowFactor 
        if (old_pointing_mode == PointingMode.CHARGE) and (shadowFac >= SHADOWFAC_EXIT_THRESHOLD):
            canChar = True
        elif (old_pointing_mode == PointingMode.CHARGE) and (shadowFac < SHADOWFAC_EXIT_THRESHOLD):
            canChar = False
        elif (old_pointing_mode != PointingMode.CHARGE) and (shadowFac >= SHADOWFAC_ENTER_THRESHOLD):
            canChar = True
        else:
            canChar = False

        
        # ---- Decide if the spacecraft can communicate (set canCom) ---- 
        for i, msg in enumerate(self.gsAccessMsgs):
            p = msg.read()
            
            # NOTE: First-come-first-serve logic.
            # TODO: Add some logic to select the "best" ground station if multiple are available
            if int(p.hasAccess) == 1:
                canCom = True
                self.selectedGsIdx = i
                break
        if not canCom:
            self.selectedGsIdx = None


        # ---- Determine battery level (set battery logical params) ---- #
        batStorageLevel = float(self.batStateMsg.read().storageLevel)
        batStorageCapacity = float(self.batStateMsg.read().storageCapacity)
        if batStorageCapacity == 0.0:
            batStorageFrac = -1
        else:
            batStorageFrac = (batStorageLevel / batStorageCapacity)

        # If 'batStorageFrac' is defined, set 'capBat', 'camBat', 'critBat', 'exitEmergencyFlag'
        if batStorageFrac >= 0:
            # Capture battery hysteresis:
            # - If already in CAPTURE, remain allowed until battery drops below exit threshold.
            # - If not in CAPTURE, only enter CAPTURE after charging above enter threshold.
            if old_pointing_mode == PointingMode.CAPTURE:
                capBat = batStorageFrac >= CAPTURE_BATTERY_EXIT_THRESHOLD
            else:
                capBat = batStorageFrac >= CAPTURE_BATTERY_ENTER_THRESHOLD
            
            if old_pointing_mode == PointingMode.COMMS:
                comBat = batStorageFrac >= COMMS_BATTERY_EXIT_THRESHOLD
            else:
                comBat = batStorageFrac >= COMMS_BATTERY_ENTER_THRESHOLD

            if batStorageFrac < CRITICAL_BATTERY_THRESHOLD:
                critBat = True
            else:
                critBat = False

            if (self.pointingMode == PointingMode.EMERGENCY) and batStorageFrac >= EMERGENCY_BATTERY_EXIT_THRESHOLD:
                exitEmergencyFlag = True
            else:
                exitEmergencyFlag = False

            # set self.lowBatIsUnresolved
            if self.pointingMode != PointingMode.EMERGENCY:
                if (not self.lowBatIsUnresolved) and (batStorageFrac < LOW_BATTERY_THRESHOLD):
                    self.lowBatIsUnresolved = True

                elif self.lowBatIsUnresolved and (batStorageFrac >= EMERGENCY_BATTERY_EXIT_THRESHOLD):
                    self.lowBatIsUnresolved = False
            else:
                # Emergency mode overrides mode switching logic instead
                self.lowBatIsUnresolved = False

            
        # ---- Evaluate the time since last communication (set maxNoCom) ---- #
        hoursSinceLastComms = (CurrentSimNanos - self.lastCommsNanos) * macros.NANO2HOUR
        if hoursSinceLastComms >= MAX_HOURS_SINCE_LAST_COM_THRESHOLD:
            maxNoCom = True
        else:
            maxNoCom = False


        # ---- Evaluate if time spent in comunication mode (set comEventComplete) ---- #
        if old_pointing_mode == PointingMode.COMMS:
            minutesSinceComStart = (CurrentSimNanos - self.currentCommsStartNanos) * macros.NANO2MIN
            
            if minutesSinceComStart > MIN_MINUTES_COM_TIME:
                comEventComplete = True
            else:
                comEventComplete = False


        # ---- Decide if spacecraft burn is requested, and if so, is the requested attitude reached (set burnRequested, burnAttitudeReached) ---- #
        if self.formEnabled and self.sat_idx != 0:
            burnAttRequested = self._formation_burn_attitude_requested(CurrentSimNanos)
            burnRequested = self._formation_burn_requested(CurrentSimNanos)
            
            if burnAttRequested:
                logging.debug(f"[{self.logTag}] burn attitude requested @ t={CurrentSimNanos * macros.NANO2MIN} min")

            # if burnRequested:
                # logging.debug(f"[{self.logTag}] burn requested @ t={CurrentSimNanos * macros.NANO2MIN} min")
        else:
            burnAttRequested = False
            burnRequested = False


        # ---- Set helper logical parameters for readability and easier easier debugging ---- # 
        if canCom and comBat:
            comPossible = True
        else:
            comPossible = False

        if canCap and capBat:
            capPossible = True
        else:
            capPossible = False

        if (old_pointing_mode == PointingMode.COMMS) and (not comEventComplete) and canCom:
            continueComEvent = True
        else:
            continueComEvent = False

        
        # ---- Mode switching logic ---- #
        if critBat or (self.pointingMode == PointingMode.EMERGENCY and not exitEmergencyFlag):
                nextMode = PointingMode.EMERGENCY

        # Override normal pointing operations with burn pointing
        elif burnAttRequested or burnRequested:
            if burnRequested:
                nextMode = PointingMode.BURN
            elif burnAttRequested:
                nextMode = PointingMode.BURN_TRANSIT
            else:
                nextMode = PointingMode.ERROR

        # Normal formation-independent operations
        elif not maxNoCom:
            if continueComEvent:                        # 1. Finishing current comunication event
                nextMode = PointingMode.COMMS
            elif self.lowBatIsUnresolved and canChar:   # 2. Prevent further decreasing battery level if already low by charging
                nextMode = PointingMode.CHARGE
            elif capPossible:                           # 3. Capture scientific GNSS-R data
                nextMode = PointingMode.CAPTURE
            elif comPossible:                           # 4. Ground station communication 
                nextMode = PointingMode.COMMS
            elif canChar:                               # 5. Charge
                nextMode = PointingMode.CHARGE
            else:                                       # 6. Coast
                nextMode = PointingMode.COAST
        elif maxNoCom:
            if comPossible or continueComEvent:         # 1. Ground station communication
                nextMode = PointingMode.COMMS
            elif self.lowBatIsUnresolved and canChar:   # 2. Prevent further decreasing battery level if already low by charging
                nextMode = PointingMode.CHARGE
            elif capPossible:                           # 3. Capture scientific data
                nextMode = PointingMode.CAPTURE
            elif canChar:                               # 4. Charge
                nextMode = PointingMode.CHARGE
            else:                                       # 5. Coast
                nextMode = PointingMode.COAST
        else:
            nextMode = PointingMode.ERROR

        self.pointingMode = nextMode

        ########################### DEBUG ###########################
        # self.pointingMode = PointingMode.COAST
        #############################################################
        
        # Create log entry if pointing mode changes
        if old_pointing_mode != self.pointingMode:
            # Update self.lastCommsNanos
            if old_pointing_mode == PointingMode.COMMS:
                self.lastCommsNanos = CurrentSimNanos
                hoursSinceLastComms = 0.

            # Update self.currentCommsStartNanos
            if self.pointingMode == PointingMode.COMMS:
                self.currentCommsStartNanos = CurrentSimNanos

            self._evaluate_active_eps_components(old_pointing_mode, self.pointingMode, CurrentSimNanos)

            currentSimMins = CurrentSimNanos * macros.NANO2MIN
            if self.formEnabled:
                # cmd = self._read_form_thr_cmd()

                self._log_mode_switching_logic(
                    currentSimMins=currentSimMins,
                    old_pointing_mode=old_pointing_mode,
                    new_pointing_mode=self.pointingMode,
                    hoursSinceLastComms=hoursSinceLastComms,
                    batStorageFrac=batStorageFrac,
                    canChar=canChar,
                    canCom=canCom,
                    comBat=comBat,
                    canCap=canCap,
                    capBat=capBat,
                    critBat=critBat,
                    maxNoCom=maxNoCom,
                    emergencyExitFlag=exitEmergencyFlag,
                    # formAttRefInMsg=self.formAttRefInMsg.read().sigma_RN, # type: ignore
                    # formThrCmdInMsg=list(cmd.OnTimeRequest) if cmd is not None else None,
                    thrOnTimeCmdOutMsg=None,
                    burnRequested=burnRequested
            )
            else:
                self._log_mode_switching_logic(
                    currentSimMins=currentSimMins,
                    old_pointing_mode=old_pointing_mode,
                    new_pointing_mode=self.pointingMode,
                    hoursSinceLastComms=hoursSinceLastComms,
                    batStorageFrac=batStorageFrac,
                    canChar=canChar,
                    canCom=canCom,
                    comBat=comBat,
                    canCap=canCap,
                    capBat=capBat,
                    critBat=critBat,
                    maxNoCom=maxNoCom,
                    emergencyExitFlag=exitEmergencyFlag,
                    formAttRefInMsg=None,
                    formThrCmdInMsg=None,
                    thrOnTimeCmdOutMsg=None,
                    burnRequested=burnRequested
            )
    
    
    def _guidance(self, CurrentSimNanos: int) -> None:
        """
        Updates the desired MRP oerientation 'self.guid.sigma_R0N' based on the current pointing mode
        """

        match self.pointingMode:
            case PointingMode.COAST:
                self.guid.sigma_R0N = self._coast_desired_att()
  
            case PointingMode.COMMS:
                self.guid.sigma_R0N = self._comms_desired_att()

            case PointingMode.CHARGE:
                self.guid.sigma_R0N = self._charge_desired_att()

            case PointingMode.CAPTURE:
                self.guid.sigma_R0N = self._capture_desired_att()

            case PointingMode.BURN_TRANSIT:
                pass # The attitude is overridden by formation controller

            case PointingMode.BURN:
                pass # The attitude is overridden by formation controller
            
            case PointingMode.EMERGENCY:
                self.guid.sigma_R0N = self._emergency_desired_att()

            case _:
                logging.debug(f"[{self.logTag}] Undefined pointing mode '{self.pointingMode}' reached for '{self.scModelTag}'")
                self.guid.sigma_R0N = [0.0, 0.0, 1.0]
                raise ValueError("")
            



    ################################
    # Private setup helper methods #
    ################################

    def _setup_formation_control(self) -> None:
        """
        
        """
        assert self.sim.envModel.gravFactory is not None
        self.form_ctrl.deputyTransInMsg.subscribeTo(self.nav.transOutMsg)
        self.form_ctrl.attRefInMsg.subscribeTo(self.attRefMsg)
        self.form_ctrl.thrustConfigInMsg.subscribeTo(self.thrConfigArrayMsg)
        self.form_ctrl.vehicleConfigInMsg.subscribeTo(self.massVehicleConfigOutMsg)
        self.form_ctrl.mu = self.sim.envModel.gravFactory.gravBodies["earth"].mu 
        self.form_ctrl.attControlTime = BURN_ATT_ADJUSTMENT_TIME_SEC  # [s] "Padding" time from burn is requested until executed. Time used to adjust attitue 

        # connect a blank chief message
        chiefData = messaging.NavTransMsgPayload()
        chiefMsg = messaging.NavTransMsg().write(chiefData)
        self.form_ctrl.chiefTransInMsg.subscribeTo(chiefMsg)



    def _setup_desired_OE_difference(self) -> None:
        """
        Calculates and sets the desired classic orbital element difference dependinng on the selected formation type

        The spacecraftReconfig module expects the desired classic orbital element difference to be on the following format:
            [da, de, di, dOmega, domega, dM], 
        Where 'da' is normalized to become  dimentionless 
        """
        
        if self.sim.cfg.form_type == "cat":
            # Don't assign  desired OED for chief spacecraft
            if self.sat_idx == 0:
                return

            desiredSeparation = self.sim.cfg.cat_const_separation

            # Set up the station keeping requirements
            rho = 1000.0                 # [m]
            a_ref = 6878137.0            # [m], approximate 500 km Earth orbit
            eps = rho / a_ref            # 1.4539e-4

            # TODO: Calculate the desired OED to get the desired separation given circular cheif orbit

            if self.sat_idx == 1:
                self.form_ctrl.targetClassicOED = [0.0000, eps, eps, 0.0000, 0.0000, -0.003]
            if self.sat_idx == 2:
                self.form_ctrl.targetClassicOED = [0.0000, 2*eps, 2*eps, 0.0000, 0.0000, 0.003]


        elif self.sim.cfg.form_type == "cc":

            # Set up the station keeping requirements
            rho = 400.0                  # [m]
            a_ref = 6878137.0            # [m], approximate 500 km Earth orbit
            eps = rho / a_ref            # 1.4539e-4

            if self.sat_idx == 1:
                self.form_ctrl.targetClassicOED = [
                    0., # da/a
                    eps, # de
                    eps, #di
                    0., # dOmega
                    0., #domega
                    0.  # dM
                ]
            if self.sat_idx == 2:
                self.form_ctrl.targetClassicOED = [
                    0., 
                    2*eps, 
                    2*eps, 
                    0., 
                    0., 
                    0.
                ]

            if self.sat_idx > 2:
                raise ValueError(f"Desired orbital element difference has not been implemented for more than 2 follower spacecraft")


        else: 
            raise ValueError(f"Formation types other than 'constant along-track separation has not yet been implemented")
        

    def _setup_gateway_msgs(self):
        """
        Create C-wrapped gateway messages such that different modules can write to this message
        and provide a common input msg for down-stream modules.
        """
        self.attRefMsg = messaging.AttRefMsg_C()
        self.attGuidMsg = messaging.AttGuidMsg_C()

        self._zero_gateway_msgs()

        # Add both the guidance and formation control modules as writers of the attitude reference message 
        messaging.AttRefMsg_C_addAuthor(self.form_ctrl.attRefOutMsg, self.attRefMsg)
        messaging.AttRefMsg_C_addAuthor(self.guid.attRefOutMsg, self.attRefMsg)

        # Add the attitude erro rmodule as writer of the attitude guidance message
        messaging.AttGuidMsg_C_addAuthor(self.att_err.attGuidOutMsg, self.attGuidMsg)

        # connect gateway FSW effector command msgs with the dynamics
        # assert self.dynModel.rwEffector is not None
        # assert self.dynModel.thrusterEffector is not None

        # Connected in dynModel's public methods
        # self.dynModel.rwEffector.rwMotorCmdInMsg.subscribeTo(self.rw_map.rwMotorTorqueOutMsg)
        # self.dynModel.thrusterEffector.cmdsInMsg.subscribeTo(self.form_ctrl.onTimeOutMsg)


    
    def _setup_eps_components(self) -> None:
        """
        Create persistent status command messages for switchable EPS loads.

        The eps modules are initialized OFF. The dynamics-side
        components subscribes to this message.
        """
        def _new_device_status_msg(status: int) -> messaging.DeviceStatusMsg:
            payload = messaging.DeviceStatusMsgPayload()
            payload.deviceStatus = status
            return messaging.DeviceStatusMsg().write(payload)
        
        self.com_status_msg = _new_device_status_msg(0)
        self.pay_status_msg = _new_device_status_msg(0)
        self.prop_idle_status_msg = _new_device_status_msg(1)
        self.prop_heat_status_msg = _new_device_status_msg(0)
        self.prop_thr_status_msg = _new_device_status_msg(0)
        

        logging.debug(f"[{self.logTag}] EPS component status messages initialized")



    def _setup_fsw_recorders(self):
        """
        Initialize all fsw recorders

        This method sets the attributes:
            self.navTransRecorder:      Logs position and velocity
            self.navAttRecorder:        Logs attitude and anfular rate
            self.attRefRecorder:        Logs desired attitude and angular rate
            self.attErrRecorder:        Logs attitude and angular rate error
            self.cmdTorqueRecorder:     Logs commanded body torque
            self.rwMotorTorqueRecorder: Logs actual RW torque
        """

        # Relevant Sample rates
        lowSampleRateNanos = self.sim.lowSampleRateNanos
        midSampleRateNanos = self.sim.midSampleRateNanos
        highSampleRateNanos = self.sim.highSampleRateNanos

        # Set recorder sample rates
        navTransRate = lowSampleRateNanos # NOTE: This should always be 'lowSampleRateNanos' for 'lowRateTimes' to be correct in SimData._pull_single_spacecraft_data
        navAttRate = highSampleRateNanos  # NOTE: This should always be 'highSampleRateNanos' for 'highRateTimes' to be correct in SimData._pull_single_spacecraft_data
        attRefRate = highSampleRateNanos
        attErrRate = highSampleRateNanos
        cmdTorqueRate = highSampleRateNanos
        rwMotorTorqueRate = highSampleRateNanos
        
        # Verify that rates are exact multiples of dynRate
        if lowSampleRateNanos % self.sim.fswRateNanos != 0.0:
            raise ValueError("'lowSampleRateNanos' is not an exact multiple of 'fswRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'LOW_SAMPLE_RATE' and/or 'FSW_RATE' to fix this error")
        if midSampleRateNanos % self.sim.fswRateNanos != 0.0:
            raise ValueError("'midSampleRateNanos' is not an exact multiple of 'fswRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'MID_SAMPLE_RATE' and/or 'FSW_RATE' to fix this error")
        if highSampleRateNanos % self.sim.fswRateNanos != 0.0:
            raise ValueError("'highSampleRateNanos' is not an exact multiple of 'fswRateNanos'. "
                             "This would have caused inconsistent sampling intervals. "
                             "Change 'HIGH_SAMPLE_RATE' and/or 'FSW_RATE' to fix this error")

        # Mandetory translational state recorder
        self.navTransRecorder = self.nav.transOutMsg.recorder(navTransRate) # r_BN_N [m] + v_BN_N [m/s]
        self.navTransRecorder_RateNanos = navTransRate
        
        # Optional 'debug' recorders
        if self.sim.cfg.data_mode == "debug":
            # Attitude and angular rate recorder
            self.navAttRecorder = self.nav.attOutMsg.recorder(navAttRate) # sigma_BN [MRP] + omega_BN_B [rad/s]
            self.navAttRecorder_RateNanos =  navAttRate

            # Desired orientational states and corresponding error
            assert self.attRefMsg is not None
            assert self.attGuidMsg is not None
            self.attRefRecorder = self.attRefMsg.recorder(attRefRate) # sigma_RN [MRP] + omega_RN_N [rad/s]
            self.attRefRecorder_RateNanos = attRefRate
            self.attErrRecorder = self.attGuidMsg.recorder(attErrRate) # sigma_BR [MRP] + omega_BR_B [rad/s]
            self.attErrRecorder_RateNanos = attErrRate

            # RW commanded and outputted torque
            self.cmdTorqueRecorder = self.ctrl.cmdTorqueOutMsg.recorder(cmdTorqueRate) # torqueRequestBody [Nm]
            self.cmdTorqueRecorder_RateNanos = cmdTorqueRate
            self.rwMotorTorqueRecorder = self.rw_map.rwMotorTorqueOutMsg.recorder(rwMotorTorqueRate) # motorTorque [Nm]
            self.rwMotorTorqueRecorder_RateNanos = rwMotorTorqueRate

        logging.debug(f"[{self.logTag}] FSW recorders initialized for '{self.scModelTag}'")

        
        

    ##########################
    # Private helper methods #
    ##########################  

    def _evaluate_active_eps_components(
        self,
        oldPM: PointingMode,
        newPM: PointingMode,
        CurrentSimNanos: int,
    ) -> None:
        if self.com_status_msg is None:
            raise RuntimeError("com_status_msg has not been initialized.")
        if self.pay_status_msg is None:
            raise RuntimeError("pay_status_msg has not been initialized.")
        if self.prop_idle_status_msg is None:
            raise RuntimeError("prop_idle_status_msg has not been initialized.")
        if self.prop_heat_status_msg is None:
            raise RuntimeError("prop_heat_status_msg has not been initialized.")
        if self.prop_thr_status_msg is None:
            raise RuntimeError("prop_thr_status_msg has not been initialized.")

        def write_status(msg: messaging.DeviceStatusMsg, enabled: bool) -> None:
            payload = messaging.DeviceStatusMsgPayload()
            payload.deviceStatus = 1 if enabled else 0
            msg.write(payload, CurrentSimNanos)

        # Only change active components if there is an actual mode transition
        if not oldPM == newPM:
        
            # Communication sink
            if newPM == PointingMode.COMMS:
                write_status(self.com_status_msg, True)
            if oldPM == PointingMode.COMMS:
                write_status(self.com_status_msg, False)

            # Payload sink
            if newPM == PointingMode.CAPTURE:
                write_status(self.pay_status_msg, True)
            if oldPM == PointingMode.CAPTURE:
                write_status(self.pay_status_msg, False)

            # Propulsion system sinks
            if newPM == PointingMode.BURN_TRANSIT:
                write_status(self.prop_idle_status_msg, False)
                write_status(self.prop_heat_status_msg, True)
                write_status(self.prop_thr_status_msg, False)
            if oldPM == PointingMode.BURN_TRANSIT:
                if newPM == PointingMode.BURN:
                    write_status(self.prop_idle_status_msg, False)
                    write_status(self.prop_heat_status_msg, False)
                    write_status(self.prop_thr_status_msg, True)
                else: 
                    write_status(self.prop_idle_status_msg, True)
                    write_status(self.prop_heat_status_msg, False)
                    write_status(self.prop_thr_status_msg, False)
            
            if newPM == PointingMode.BURN:
                write_status(self.prop_idle_status_msg, False)
                write_status(self.prop_heat_status_msg, False)
                write_status(self.prop_thr_status_msg, True)
            if oldPM == PointingMode.BURN:
                if newPM == PointingMode.BURN_TRANSIT:
                    write_status(self.prop_idle_status_msg, False)
                    write_status(self.prop_heat_status_msg, True)
                    write_status(self.prop_thr_status_msg, False)
                else:
                    write_status(self.prop_idle_status_msg, True)
                    write_status(self.prop_heat_status_msg, False)
                    write_status(self.prop_thr_status_msg, False)

            # Propulsion heater sink
            if newPM == PointingMode.BURN_TRANSIT:
                write_status(self.prop_heat_status_msg, True)
            if oldPM == PointingMode.BURN_TRANSIT:
                write_status(self.prop_heat_status_msg, False)

            # Propulsion thrusting sink
            if newPM == PointingMode.BURN:
                write_status(self.prop_thr_status_msg, True)
            if oldPM == PointingMode.BURN:
                write_status(self.prop_thr_status_msg, False)


    
    def _limit_thrust_cmd_by_fuel(self, CurrentSimNanos: int) -> None:
        raw_cmd = self.form_ctrl.onTimeOutMsg.read()
        fuel = self.fuelTankMsg.read()

        remaining_fuel = max(float(fuel.fuelMass), 0.0)
        reserve = 1e-6

        payload = messaging.THRArrayOnTimeCmdMsgPayload()

        logging.debug(f"[{self.logTag}] Remaining fuel @t={CurrentSimNanos * macros.NANO2MIN:.2f} min: {remaining_fuel:.2f}")
        if remaining_fuel <= reserve:
            payload.OnTimeRequest = [0.0 for _ in raw_cmd.OnTimeRequest]
            logging.debug(f"[{self.logTag}] Limited thrust cmd to 0.0 due to empty fueltank")
        else:
            payload.OnTimeRequest = [float(t) for t in raw_cmd.OnTimeRequest]

        self.fuelSafeThrCmdOutMsg.write(payload, CurrentSimNanos)

    
    
    def _zero_gateway_msgs(self):
        """Zero all FSW gateway message payloads"""
        assert self.attRefMsg is not None
        assert self.attGuidMsg is not None        
        self.attRefMsg.write(messaging.AttRefMsgPayload())
        self.attGuidMsg.write(messaging.AttGuidMsgPayload())

        # Zero all actuator commands
        self.rw_map.rwMotorTorqueOutMsg.write(messaging.ArrayMotorTorqueMsgPayload())
        self.form_ctrl.onTimeOutMsg.write(messaging.THRArrayOnTimeCmdMsgPayload())


    def _formation_burn_attitude_requested(self, CurrentSimNanos: int) -> bool:
        """
        Return True while a logging-only formation burn attitude request is active.

        A burn attitude request is detected from a new/change in the
        spacecraftReconfig attitude reference output. The request remains active
        while waiting for the corresponding burn command, and then remains active
        during the burn itself.

        This method only affects logging mode selection:
            burnAttRequested and not burnRequested -> BURN_TRANSIT
            burnAttRequested and burnRequested     -> BURN

        Attitude and thrust are still handled by spacecraftReconfig.
        """

        ############
        # This approach does not work because 'spacecraftReconfig' publishes attitude requests 
        # periodically, and not in the approximate time around a burn execution. 
        ############
        return False

        # if (not self.formEnabled) or (self.sat_idx == 0):
        #     self.activeBurnAttRequest = False
        #     self.activeBurnAttRequestStartNanos = None
        #     self.prevFormationAttRefSignature = np.zeros(6)
        #     self.burnDetectedDuringActiveAttRequest = False
        #     return False

        # att_ref_change_tol = 1e-8
        # max_wait_for_burn_s = BURN_ATT_ADJUSTMENT_TIME_SEC + 5.0

        # def _read_current_att_ref_signature() -> Optional[NDArray[np.float64]]:
        #     try:
        #         att_ref = self.form_ctrl.attRefOutMsg.read()
        #     except Exception as e:
        #         logging.debug(
        #             f"[{self.logTag}] Could not read formation-control attitude reference: {repr(e)}"
        #         )
        #         return None

        #     sigma_RN = np.array(att_ref.sigma_RN, dtype=float)
        #     omega_RN_N = np.array(att_ref.omega_RN_N, dtype=float)

        #     return np.concatenate((sigma_RN, omega_RN_N))

        # # --------------------------------------------------
        # # 1. If an attitude request is already active, keep it
        # #    active until the associated burn has completed.
        # # --------------------------------------------------
        # if self.activeBurnAttRequest:
        #     if self.activeBurnRequest:
        #         # A burn has now been detected during this attitude request.
        #         # Stay in burn-pointing override while the burn event is active.
        #         self.burnDetectedDuringActiveAttRequest = True
        #         return True

        #     if self.burnDetectedDuringActiveAttRequest:
        #         # The corresponding burn was active before, but is no longer active.
        #         # Therefore the burn-attitude request has served its purpose.
        #         current_signature = _read_current_att_ref_signature()
        #         if current_signature is not None:
        #             self.prevFormationAttRefSignature = current_signature

        #         self.activeBurnAttRequest = False
        #         self.activeBurnAttRequestStartNanos = None
        #         self.burnDetectedDuringActiveAttRequest = False
        #         return False

        #     # No burn has started yet. Keep BURN_TRANSIT active while waiting
        #     # for spacecraftReconfig to issue the corresponding burn command.
        #     if self.activeBurnAttRequestStartNanos is None:
        #         self.activeBurnAttRequestStartNanos = CurrentSimNanos

        #     elapsed_wait_s = (
        #         CurrentSimNanos - self.activeBurnAttRequestStartNanos
        #     ) * macros.NANO2SEC

        #     if elapsed_wait_s <= max_wait_for_burn_s:
        #         return True

        #     # Timeout: attitude request was detected, but no burn followed.
        #     # Clear the request to avoid getting stuck in BURN_TRANSIT forever.
        #     current_signature = _read_current_att_ref_signature()
        #     if current_signature is not None:
        #         self.prevFormationAttRefSignature = current_signature

        #     self.activeBurnAttRequest = False
        #     self.activeBurnAttRequestStartNanos = None
        #     self.burnDetectedDuringActiveAttRequest = False

        #     logging.debug(
        #         f"[{self.logTag}] Burn attitude request timed out at "
        #         f"t={CurrentSimNanos * macros.NANO2MIN:.3f} min"
        #     )

        #     return False

        # --------------------------------------------------
        # 2. No active attitude request: detect a new or changed
        #    spacecraftReconfig attitude reference.
        # --------------------------------------------------
        # current_signature = _read_current_att_ref_signature()
        # if current_signature is None:
        #     return False

        # signature_change = np.linalg.norm(
        #     current_signature - self.prevFormationAttRefSignature
        # )

        # current_signature_norm = np.linalg.norm(current_signature)

        # new_attitude_request = (
        #     current_signature_norm > att_ref_change_tol
        #     and signature_change > att_ref_change_tol
        # )

        # self.prevFormationAttRefSignature = current_signature

        # if not new_attitude_request:
        #     return False

        # self.activeBurnAttRequest = True
        # self.activeBurnAttRequestStartNanos = CurrentSimNanos
        # self.burnDetectedDuringActiveAttRequest = False

        # logging.debug(
        #     f"[{self.logTag}] New logging-only burn attitude request detected at "
        #     f"t={CurrentSimNanos * macros.NANO2MIN:.3f} min"
        # )

        # return True


    def _formation_burn_requested(self, CurrentSimNanos: int) -> bool:
        """
        Return True while a logging-only formation burn event is active.

        A burn event is triggered when spacecraftReconfig produces a new nonzero
        on-time command. Once detected, the burn request remains active for the
        estimated burn duration.

        This function only affects logging modes. Attitude and thrust are still
        handled by spacecraftReconfig.
        """

        if (not self.formEnabled) or (self.sat_idx == 0):
            self.activeBurnRequest = False
            self.activeBurnEventStartNanos = None
            self.activeBurnDurationS = 0.0
            self.prevFormationOnTimeMax = 0.0
            return False

        on_time_tol = 1e-9

        # --------------------------------------------------
        # 1. If a burn event is already active, keep it active
        #    until the estimated burn duration has elapsed.
        # --------------------------------------------------
        if self.activeBurnRequest:
            if self.activeBurnEventStartNanos is None:
                self.activeBurnEventStartNanos = CurrentSimNanos

            elapsed_event_s = (
                CurrentSimNanos - self.activeBurnEventStartNanos
            ) * macros.NANO2SEC

            if elapsed_event_s < self.activeBurnDurationS:
                return True

            # Burn logging window completed.
            self.activeBurnRequest = False
            self.activeBurnEventStartNanos = None
            self.activeBurnDurationS = 0.0

            return False

        # --------------------------------------------------
        # 2. No active event: look for a new spacecraftReconfig
        #    burn command.
        # --------------------------------------------------
        try:
            thr_cmd = self.form_ctrl.onTimeOutMsg.read()
        except Exception as e:
            logging.debug(
                f"[{self.logTag}] Could not read formation-control thrust command: {repr(e)}"
            )
            return False

        on_time_request = np.array(thr_cmd.OnTimeRequest, dtype=float)

        if on_time_request.size == 0:
            current_on_time_max = 0.0
        else:
            current_on_time_max = float(np.max(on_time_request))

        new_burn_command = (
            current_on_time_max > on_time_tol
            and (
                self.prevFormationOnTimeMax <= on_time_tol
                or abs(current_on_time_max - self.prevFormationOnTimeMax) > on_time_tol
            )
        )

        self.prevFormationOnTimeMax = current_on_time_max

        if not new_burn_command:
            return False

        # New logging-only burn event detected.
        self.activeBurnRequest = True
        self.activeBurnEventStartNanos = CurrentSimNanos
        self.activeBurnDurationS = current_on_time_max

        # logging.debug(
        #     f"[{self.logTag}] New logging-only burn event detected at "
        #     f"t={CurrentSimNanos * macros.NANO2MIN:.3f} min: "
        #     f"estimated burn duration={self.activeBurnDurationS:.6f} s"
        # )

        return True
    
    
    def _burn_attitude_reached(self, CurrentSimNanos: int) -> bool:
        """
        Return True after the fixed BURN_TRANSIT duration has elapsed.

        This does not control attitude or thrust. It only determines whether the
        logging mode should be BURN_TRANSIT or BURN.
        """

        if not self.activeBurnRequest:
            return False

        if self.activeBurnEventStartNanos is None:
            return False

        elapsed_event_s = (
            CurrentSimNanos - self.activeBurnEventStartNanos
        ) * macros.NANO2SEC

        return elapsed_event_s >= BURN_ATT_ADJUSTMENT_TIME_SEC
        
    
    def _log_mode_switching_logic(
        self,
        currentSimMins: Optional[float] = None,
        old_pointing_mode: Optional[PointingMode] = None,
        new_pointing_mode: Optional[PointingMode] = None,
        hoursSinceLastComms: Optional[float] = None,
        batStorageFrac: Optional[float] = None,
        canChar: Optional[bool] = None,
        canCom: Optional[bool] = None,
        comBat: Optional[bool] = None,
        canCap: Optional[bool] = None,
        capBat: Optional[bool] = None,
        critBat: Optional[bool] = None,
        maxNoCom: Optional[bool] = None,
        emergencyExitFlag: Optional[bool] = None,
        formAttRefInMsg: Optional[list] = None,
        formThrCmdInMsg: Optional[Any] = None,
        thrOnTimeCmdOutMsg: Optional[list[float]] = None,
        burnRequested: Optional[bool] = None,
        burnAttitudeReached: Optional[bool] = None,
        write_header_only: bool = False,
    ) -> None:
        """
        Initialize log or append one row of mode-switching logic data to a CSV log file.
        If write_header_only=True, only ensure the file exists and write the header if missing.
        """

        # Ensure the log directory exists
        self.sim.cfg.output_data_save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.logTag}_mode_switching.csv"

        filepath = self.sim.cfg.output_data_save_dir / filename

        header = [
            "ModelTag",
            "currentSimMins",
            "oldPointingMode",
            "newPointingMode",
            "hoursSinceLastComms",
            "batStorageFrac",
            "canChar",
            "canCom",
            "comBat",
            "canCap",
            "capBat",
            "critBat",
            "maxNoCom",
            "emergencyExitFlag",
            "formAttRefInMsg",
            "formThrCmdInMsg",
            "thrOnTimeCmdOutMsg",
            "burnRequested",
            "burnAttitudeReached"

        ]

        row = [
            self.ModelTag,
            currentSimMins,
            str(old_pointing_mode),
            str(new_pointing_mode),
            hoursSinceLastComms,
            batStorageFrac,
            canChar,
            canCom,
            comBat,
            canCap,
            capBat,
            critBat,
            maxNoCom,
            emergencyExitFlag,
            formAttRefInMsg,
            formThrCmdInMsg,
            thrOnTimeCmdOutMsg,
            burnRequested,
            burnAttitudeReached,
        ]

        file_exists = filepath.exists()

        with open(filepath, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(header)
                logging.debug(f"[{self.logTag}] Mode switching log created for '{self.scModelTag}'")

            if write_header_only:
                return

            row = [
                self.ModelTag,
                currentSimMins,
                str(old_pointing_mode) if old_pointing_mode is not None else None,
                str(new_pointing_mode) if new_pointing_mode is not None else None,
                hoursSinceLastComms,
                batStorageFrac,
                canChar,
                canCom,
                comBat,
                canCap,
                capBat,
                critBat,
                maxNoCom,
                emergencyExitFlag,
                formAttRefInMsg,
                formThrCmdInMsg,
                thrOnTimeCmdOutMsg,
                burnRequested,
                burnAttitudeReached,
            ]

            writer.writerow(row)
    
    
    