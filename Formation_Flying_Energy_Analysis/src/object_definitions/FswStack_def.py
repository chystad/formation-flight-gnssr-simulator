import os
import csv
import logging
import numpy as np
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
    EMERGENCY = "emergency"
    ERROR = "error"


class FswStack(sysModel.SysModel):
    """
    One-per-spacecraft RW flight-software "stack" wrapped in a single SysModel so it can be scheduled
    as one model inside a dedicated FSW task.

    Internal order each UpdateState:
        1) SimpleNav
        2) Guidance (inertial3D)
        3) Tracking error (attTrackingError)
        4) Control law (mrpFeedback)
        5) Torque mapping (rwMotorTorque)

    Exposed outputs (message handles):
        - rwMotorTorqueOutMsg 
        - attGuidOutMsg         
        - cmdTorqueOutMsg  
        - navAttOutMsg / navTransOutMsg 

    =========================================================================================================
    ATTRIBUTES:
        modelTag            (str) Name of FswStack instance
        pointingMode        (PointingMode) State describing the current pointing objective
        gsAccessMsgs        (list[AccessMsg]) Access msgs for the satellite against all ground stations
        gsPosMsgs           (list[GroundStateMsg]) Contain 'r_LN_N' describing the GS position vector relative to inertial frame origin in inertial coordinates.
        sunEclipseMsg       (EclipseMsgPayload) Contains 'shadowFactor' property used to evaluate Sun illumination on the SC
        nav                 (SimpleNav) 
        guid                (inertial3D)
        att_err             (attTrackingErr)
        ctrl                (mrpFeedback) 
        rw_map              (rwMotorTorque)
        rwMotorTorqueOutMsg (rwMotorTorqueOutMsg)
        attGuidOutMsg       (attGuidOutMsg)
        cmdTorqueOutMsg     (cmdTorqueOutMsg)
        navAttOutMsg        (navAttOutMsg)
        navTransOutMsg      (navTransOutMsg)

        _vc_msg         (VehicleConfigMsg) Vehicle configuration message. Hub inertia.
    =========================================================================================================

    """

    def __init__(
        self,
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
        log_timestamp: str
    ):
        """
        Args:
            sat (Satellite): current Satellite instance
            sat_idx (int): Satellite iteration number
            sc_state_out_msg (messaging.SCStatesMsg): Sc position and velpcity in inerital frame
            rw_speed_out_msg (messaging.RWSpeedMsg): The angular speeds of all RWs in the SC's RW cluster
            rw_config_msg (messaging.RWArrayConfigMsg): Desciption of how the SC's RW cluster is defined
            gs_access_msgs (list[messaging.AccessMsg]): List containing the SC's access to all GSs.
                Ex: gs_access_msgs[0] will contain the boolean value describing if the spacecraft is within GS[0]'s LOS.
    
        """
        super().__init__()

        # self.sim = sim
        self.ModelTag = f"RwFswStack{sat_idx}"
        self.LogTag = f"FSW{sat_idx}"
        self.pointingMode = PointingMode.COAST
        self.batStateMsg = bat_state_msg
        self.gsAccessMsgs = gs_access_msgs
        self.gsStateMsgs = gs_state_msgs
        self.selectedGsIdx: Optional[int] = None 
        self.sunEclipseMsg = sun_eclipse_msg
        self.sunStateMsg = sun_state_msg
        self.logTimestamp = log_timestamp

        # Keep track of the last time COMMS was exited (last commuication attempt was completed)
        self.lastCommsNanos = 0

        ## temp
        self.oldSunEclipseMsgShadowFactor = None
        self.prevGsPosPrintHours = 0.
        ##

        # Initialize mode switching log
        self._log_mode_switching_logic(write_header_only=True)
        

        # ----------------------------
        # Internal FSW modules
        # ----------------------------
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

        # ----------------------------
        # Guidance configuration
        # ----------------------------
        # set the desired inertial orientation using Modified Rodrigues Parameterization (MRP)
        self.guid.sigma_R0N = [0.0, 0.0, 0.0]

        # ----------------------------
        # RW mapping configuration
        # ----------------------------
        # Identity (control 3 body axes)
        control_axes_B: list[int] = [1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1]
        self.rw_map.controlAxes_B = control_axes_B

        # ----------------------------
        # Vehicle config msg (inertia)
        # ----------------------------
        # create the FSW vehicle configuration message
        # use the same inertia in the FSW algorithm as in the simulation
        vehicle_config_out = messaging.VehicleConfigMsgPayload(ISCPntB_B=sat.I_B)
        self._vc_msg = messaging.VehicleConfigMsg().write(vehicle_config_out)

        # ----------------------------
        # Controller gains
        # ----------------------------
        # Defaults match your BasiliskSimulator_def.py constants; override via args if desired.
        self.ctrl.K = MRP_K
        self.ctrl.P = MRP_P
        self.ctrl.Ki = MRP_KI

        if self.ctrl.Ki > 0:
            self.ctrl.integralLimit = 2.0 / self.ctrl.Ki * 0.1

        # ----------------------------
        # Message wiring (subscriptions)
        # ----------------------------
        # Nav reads truth state
        self.nav.scStateInMsg.subscribeTo(sc_state_out_msg)

        # Tracking error compares nav vs reference
        self.att_err.attNavInMsg.subscribeTo(self.nav.attOutMsg)
        self.att_err.attRefInMsg.subscribeTo(self.guid.attRefOutMsg)

        # Controller reads guidance error + vehicle inertia + RW params + RW speeds
        self.ctrl.guidInMsg.subscribeTo(self.att_err.attGuidOutMsg)
        self.ctrl.vehConfigInMsg.subscribeTo(self._vc_msg)
        self.ctrl.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.ctrl.rwSpeedsInMsg.subscribeTo(rw_speed_out_msg)

        # RW mapping reads RW params + commanded body torque
        self.rw_map.rwParamsInMsg.subscribeTo(rw_config_msg)
        self.rw_map.vehControlInMsg.subscribeTo(self.ctrl.cmdTorqueOutMsg)

        # ----------------------------
        # Exposed outputs (for wiring/logging in BasiliskSimulator_def)
        # ----------------------------
        self.rwMotorTorqueOutMsg = self.rw_map.rwMotorTorqueOutMsg
        self.attGuidOutMsg = self.att_err.attGuidOutMsg
        self.cmdTorqueOutMsg = self.ctrl.cmdTorqueOutMsg
        self.navAttOutMsg = self.nav.attOutMsg
        self.navTransOutMsg = self.nav.transOutMsg

        # ----------------------------
        # Flags used for debugging and testing
        # ---------------------------
        self.changed_pointing_obj: bool = False

        logging.debug(f"[{self.LogTag}] Created RW FSW stack for satellite {sat_idx}")


    def UpdateState(self, CurrentSimNanos: int) -> None:
        """
        Update all states
        """
        # Order matters: nav -> guidance -> error -> control -> mapping
        
        # # [TEST] Perform a 180 deg flip maneuver after 2.5 minutes
        # if not self.changed_pointing_obj and (CurrentSimNanos*macros.NANO2MIN > 2.5):
        #     self.guid.sigma_R0N = [0.0, 0.0, 1.0]
        #     self.changed_pointing_obj = True


        # ========== [TEST] ========== #
        # shadowFac = self.sunEclipseMsg.read().shadowFactor

        # if shadowFac != self.oldSunEclipseMsgShadowFactor:
        #     print(f"new eclipse state: {shadowFac} " 
        #           f"@: {CurrentSimNanos*macros.NANO2MIN}")
            
        # self.oldSunEclipseMsgShadowFactor = shadowFac
        # ============================ #


        # ========== [GroundLocation position in N] ========== #
        # timeBetweenGsPosPrintsHours = 0.8
        # if (self.prevGsPosPrintHours + timeBetweenGsPosPrintsHours) <= CurrentSimNanos*macros.NANO2HOUR or (CurrentSimNanos == 0):
            
        #     for i, gs in enumerate(self.gsStateMsgs):
        #         gsPos = gs.read().r_LN_N

        #         print(gsPos)
        #         print(np.linalg.norm(gsPos))

        #         self.prevGsPosPrintHours = CurrentSimNanos*macros.NANO2HOUR

        # Conclusion from this little experiment: The Basilisk inerital frame (N) is an ECI frame. 
        # The vector describing the ground locations change with time -> Not ECEF!!

        # ==================================================== #
        
        self.nav.UpdateState(CurrentSimNanos)
        self._eval_pointing_mode(CurrentSimNanos)
        self._guidance(CurrentSimNanos)
        self.guid.UpdateState(CurrentSimNanos)
        self.att_err.UpdateState(CurrentSimNanos)
        self.ctrl.UpdateState(CurrentSimNanos)
        self.rw_map.UpdateState(CurrentSimNanos)

    def SelfInit(self):
        # called by Basilisk during InitializeSimulation()
        for m in self._modules():
            if hasattr(m, "SelfInit"):
                m.SelfInit()

    def CrossInit(self):
        # some modules use cross-init to resolve message interfaces
        for m in self._modules():
            if hasattr(m, "CrossInit"):
                m.CrossInit()

    def Reset(self, CurrentSimNanos: int):
        # called by Basilisk during InitializeSimulation()
        for m in self._modules():
            if hasattr(m, "Reset"):
                m.Reset(CurrentSimNanos)

    def _modules(self):
        return [self.nav, self.guid, self.att_err, self.ctrl, self.rw_map]


    def _guidance(self, CurrentSimNanos: int) -> None:
        """
        Updates the desired MRP oerientation 'self.guid.sigma_R0N' based on the current pointing mode
        """
        self.guid.sigma_R0N = [0.0, 0.0, 1.0]

        # TODO: Compute and apply the actual correct pointing orientation based on the current pointingMode
        match self.pointingMode:
            case PointingMode.COAST:
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
                
                # Publish the desired attitude
                self.guid.sigma_R0N = mrp_D

            
            case PointingMode.COMMS:
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
                
                # Publish the desired attitude
                self.guid.sigma_R0N = mrp_D


            case PointingMode.CHARGE:
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
                
                # Publish the desired attitude
                self.guid.sigma_R0N = mrp_D


            case PointingMode.EMERGENCY:
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
                
                # Publish the desired attitude
                self.guid.sigma_R0N = mrp_D


            case _:
                logging.debug(f"[{self.LogTag}] Undefined pointing mode '{self.pointingMode}' reached in {self.ModelTag}")
                raise ValueError("")
        
    
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
        
        



        # # OLD MODE SWITCHING LOGIC
        # if canCom:
        #     self.pointingMode = PointingMode.COMMS
        # elif canChar:
        #     self.pointingMode = PointingMode.CHARGE
        # else:
        #     self.pointingMode = PointingMode.COAST
        
        if old_pointing_mode != self.pointingMode:
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
        
        filename = f"{self.logTimestamp}_{self.LogTag}_mode_switching_logic.csv"

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
                logging.debug(f"[{self.LogTag}] Mode switching log created for {self.ModelTag}")

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