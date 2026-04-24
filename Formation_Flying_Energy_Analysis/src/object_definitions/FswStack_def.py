from __future__ import annotations
from typing import TYPE_CHECKING

import os
import csv
import logging
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.fswAlgorithms import mrpFeedback, attTrackingError, inertial3D, rwMotorTorque
from Basilisk.simulation import simpleNav

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 

LOG_DATA_SAVE_DIR = Path('Formation_Flying_Energy_Analysis/output_data/logs')
MRP_K: float = 0.01 # MRP pointing controller: Gain on MRP attitude error 
MRP_P: float = 0.02 # MRP pointing controller: Gain on Rate error
MRP_KI: float = -1  # MRP pointing controller: Integral gain (-1 -> disable)
SHADOWFAC_ENTER_THRESHOLD = 0.6 # The minimum illumination required to enter CHARGE state (0, 1)
SHADOWFAC_EXIT_THRESHOLD = 0.4 # The maximum illumination requred to exit CHARGE state (0, 1)
EMERGENCY_BATTERY_EXIT_THRESHOLD = 0.6 # The lower limit for when the battery is considered to have enough charge to exit EMERGENCY mode (0, 1)
CAPTURE_BATTERY_THRESHOLD = 0.4 # The minimum battery percentage (inclusive) required for entering CAPTURE mode (0, 1)
COMMS_BATTERY_THRESHOLD = 0.3 # The minimum battery percentage (inclusive) required for entering COMMS mode (0, 1)
CRITICAL_BATTERY_THRESHOLD = 0.2 # Upper limit (exclusive) for when the battery is considered to have critially low charge left (0, 1)
MAX_HOURS_SINCE_LAST_COM_THRESHOLD = 48 # Limit (incluse) for when the maximum time has passed since last com. 
                                        # After this, communication will be prioritized over payload capturing.


class PointingMode(str, Enum):
    COAST = "coast"
    COMMS = "comms"
    CHARGE = "charge"
    CAPTURE = "capture"
    FORM_CAPTURE = "form_capture" # TODO: Placeholder mode for when
    EMERGENCY = "emergency"
    ERROR = "error"


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

    Schedules:
        - one internal _FswStackScheduler SysModel on FswTask_<sat_idx>
    """

    def __init__(
        self,
        sim: BasiliskSimulator,
        sat: Satellite,
        sat_idx: int,
        sc_state_out_msg: messaging.SCStatesMsg,
        rw_speed_out_msg: messaging.RWSpeedMsg,
        rw_config_msg: messaging.RWArrayConfigMsg,
        bat_state_msg: messaging.PowerStorageStatusMsg,
        gs_access_msgs: list[messaging.AccessMsg],
        gs_state_msgs: list[messaging.GroundStateMsg],
        sun_eclipse_msg: messaging.EclipseMsg,
        sun_state_msg: messaging.SpicePlanetStateMsg,
        log_timestamp: str,
    ):
        self.sim = sim
        self.sat = sat
        self.sat_idx = sat_idx

        self.ModelTag = f"RwFswStack{sat_idx}"
        self.logTag = f"FSW{sat_idx}"

        # -------------------------------------------------
        # Create FSW task inside the owner, like DynamicsModel
        # -------------------------------------------------
        assert sim.fswProcesses[sat_idx] is not None
        self.fswTaskName = f"FswTask_{sat_idx}"
        sim.fswProcesses[sat_idx].addTask(sim.CreateNewTask(self.fswTaskName, sim.fswRateNanos)) # type: ignore

        # -------------------------------------------------
        # Existing persistent state
        # -------------------------------------------------
        self.pointingMode = PointingMode.COAST
        self.batStateMsg = bat_state_msg
        self.gsAccessMsgs = gs_access_msgs
        self.gsStateMsgs = gs_state_msgs
        self.selectedGsIdx: Optional[int] = None
        self.sunEclipseMsg = sun_eclipse_msg
        self.sunStateMsg = sun_state_msg
        self.logTimestamp = log_timestamp

        self.lastCommsNanos = 0

        self.oldSunEclipseMsgShadowFactor = None
        self.prevGsPosPrintHours = 0.0

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

        # -------------------------------------------------
        # Guidance configuration
        # -------------------------------------------------
        self.guid.sigma_R0N = [0.0, 0.0, 0.0]

        # -------------------------------------------------
        # RW mapping configuration
        # -------------------------------------------------
        self.rw_map.controlAxes_B = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]

        # -------------------------------------------------
        # Vehicle config msg
        # -------------------------------------------------
        vehicle_config_out = messaging.VehicleConfigMsgPayload(ISCPntB_B=sat.I_B)
        self._vc_msg = messaging.VehicleConfigMsg().write(vehicle_config_out)

        # -------------------------------------------------
        # Controller gains
        # -------------------------------------------------
        self.ctrl.K = MRP_K
        self.ctrl.P = MRP_P
        self.ctrl.Ki = MRP_KI

        if self.ctrl.Ki > 0:
            self.ctrl.integralLimit = 2.0 / self.ctrl.Ki * 0.1

        # -------------------------------------------------
        # Message wiring
        # -------------------------------------------------
        self.nav.scStateInMsg.subscribeTo(sc_state_out_msg)

        self.att_err.attNavInMsg.subscribeTo(self.nav.attOutMsg)
        self.att_err.attRefInMsg.subscribeTo(self.guid.attRefOutMsg)

        self.ctrl.guidInMsg.subscribeTo(self.att_err.attGuidOutMsg)
        self.ctrl.vehConfigInMsg.subscribeTo(self._vc_msg)
        self.ctrl.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.ctrl.rwSpeedsInMsg.subscribeTo(rw_speed_out_msg)

        self.rw_map.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.rw_map.vehControlInMsg.subscribeTo(self.ctrl.cmdTorqueOutMsg)

        # -------------------------------------------------
        # Exposed outputs
        # -------------------------------------------------
        self.rwMotorTorqueOutMsg = self.rw_map.rwMotorTorqueOutMsg
        self.attGuidOutMsg = self.att_err.attGuidOutMsg
        self.cmdTorqueOutMsg = self.ctrl.cmdTorqueOutMsg
        self.navAttOutMsg = self.nav.attOutMsg
        self.navTransOutMsg = self.nav.transOutMsg

        self.changed_pointing_obj = False

        # -------------------------------------------------
        # Scheduled adapter
        # -------------------------------------------------
        self.scheduler = _FswStackScheduler(self, sat_idx)
        sim.AddModelToTask(self.fswTaskName, self.scheduler, 20)

        logging.debug(f"[{self.logTag}] Created FSW stack for satellite {sat_idx}")


    def _modules(self):
        return [self.nav, self.guid, self.att_err, self.ctrl, self.rw_map]


    def _update_state(self, CurrentSimNanos: int) -> None:
        """
        Same functionality as old UpdateState(), but FswStack is no longer
        itself the scheduled SysModel.
        """
        self.nav.UpdateState(CurrentSimNanos)
        self._eval_pointing_mode(CurrentSimNanos)
        self._guidance(CurrentSimNanos)
        self.guid.UpdateState(CurrentSimNanos)
        self.att_err.UpdateState(CurrentSimNanos)
        self.ctrl.UpdateState(CurrentSimNanos)
        self.rw_map.UpdateState(CurrentSimNanos)


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
        TODO
        """
        return np.array([0., 0., 1.])


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
            if batStorageFrac >= CAPTURE_BATTERY_THRESHOLD:
                capBat = True
            else:
                capBat = False
            
            if batStorageFrac >= COMMS_BATTERY_THRESHOLD:
                comBat = True
            else:
                comBat = False

            if batStorageFrac < CRITICAL_BATTERY_THRESHOLD:
                critBat = True
            else:
                critBat = False

            if (self.pointingMode == PointingMode.EMERGENCY) and batStorageFrac >= EMERGENCY_BATTERY_EXIT_THRESHOLD:
                exitEmergencyFlag = True
            else:
                exitEmergencyFlag = False


        # ---- Evaluate the time since last communication (set maxNoCom) ---- #
        hoursSinceLastComms = (CurrentSimNanos - self.lastCommsNanos) * macros.NANO2HOUR
        if hoursSinceLastComms >= MAX_HOURS_SINCE_LAST_COM_THRESHOLD:
            maxNoCom = True
        else:
            maxNoCom = False


        # ---- Set helper logical parameters for readability and easier easier debugging ---- # 
        if canCom and comBat:
            comPossible = True
        else:
            comPossible = False

        if canCap and capBat:
            capPossible = True
        else:
            capPossible = False

        
        # ---- Mode switching logic ---- #
        if critBat or (self.pointingMode == PointingMode.EMERGENCY and not exitEmergencyFlag):
                nextMode = PointingMode.EMERGENCY
        elif not maxNoCom:
            if capPossible:
                nextMode = PointingMode.CAPTURE
            elif comPossible:
                nextMode = PointingMode.COMMS
            elif canChar:
                nextMode = PointingMode.CHARGE
            else:
                nextMode = PointingMode.COAST
        elif maxNoCom:
            if comPossible:
                nextMode = PointingMode.COMMS
            elif capPossible:
                nextMode = PointingMode.CAPTURE
            elif canChar:
                nextMode = PointingMode.CHARGE
            else:
                nextMode = PointingMode.COAST
        else:
            nextMode = PointingMode.ERROR

        self.pointingMode = nextMode
        
        # Create log entry if pointing mode changes
        if old_pointing_mode != self.pointingMode:
            # Update self.lastCommsNanos
            if old_pointing_mode == PointingMode.COMMS:
                self.lastCommsNanos = CurrentSimNanos
                hoursSinceLastComms = 0.
            
            
            currentSimMins = CurrentSimNanos * macros.NANO2MIN
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
            )
    
    
    def _guidance(self, CurrentSimNanos: int) -> None:
        """
        Updates the desired MRP oerientation 'self.guid.sigma_R0N' based on the current pointing mode
        """
        self.guid.sigma_R0N = [0.0, 0.0, 1.0]

        # TODO: Compute and apply the actual correct pointing orientation based on the current pointingMode
        match self.pointingMode:
            case PointingMode.COAST:
                self.guid.sigma_R0N = self._coast_desired_att()
  
            case PointingMode.COMMS:
                self.guid.sigma_R0N = self._comms_desired_att()

            case PointingMode.CHARGE:
                self.guid.sigma_R0N = self._charge_desired_att()

            case PointingMode.CAPTURE:
                self.guid.sigma_R0N = self._capture_desired_att()

            case PointingMode.EMERGENCY:
                self.guid.sigma_R0N = self._emergency_desired_att()

            case _:
                logging.debug(f"[{self.logTag}] Undefined pointing mode '{self.pointingMode}' reached in {self.ModelTag}")
                raise ValueError("")
        
    
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
        write_header_only: bool = False,
    ) -> None:
        """
        Initialize log or append one row of mode-switching logic data to a CSV log file.
        If write_header_only=True, only ensure the file exists and write the header if missing.
        """

        # Ensure the log directory exists
        LOG_DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.logTimestamp}_{self.logTag}_mode_switching_logic.csv"

        filepath = LOG_DATA_SAVE_DIR / filename

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
        ]

        file_exists = filepath.exists()

        with open(filepath, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(header)
                logging.debug(f"[{self.logTag}] Mode switching log created for {self.ModelTag}")

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
            ]

            writer.writerow(row)