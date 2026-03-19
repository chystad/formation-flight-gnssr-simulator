import logging
from typing import Any, Optional, Sequence
from enum import Enum

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, SimulationBaseClass
from Basilisk.fswAlgorithms import mrpFeedback, attTrackingError, inertial3D, rwMotorTorque
from Basilisk.simulation import simpleNav

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite

MRP_K: float = 0.01 # MRP pointing controller: Gain on MRP attitude error 
MRP_P: float = 0.02 # MRP pointing controller: Gain on Rate error
MRP_KI: float = -1  # MRP pointing controller: Integral gain (-1 -> disable)


class PointingMode(str, Enum):
    INIT = "init"
    COAST = "coast"
    COMMS = "comms"
    CHARGE = "charge"
    CAPTURE = "capture"
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
        gs_access_msgs: list[messaging.AccessMsg]
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

        self.ModelTag = f"RwFswStack{sat_idx}"
        self.pointingMode = PointingMode.INIT
        self.gsAccessMsgs = gs_access_msgs
        

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

        logging.debug(f"[FSW] Created RW FSW stack for satellite {sat_idx}")


    def UpdateState(self, CurrentSimNanos: int) -> None:
        """
        Update all states
        """
        # Order matters: nav -> guidance -> error -> control -> mapping
        
        # # [TEST] Perform a 180 deg flip maneuver after 2.5 minutes
        # if not self.changed_pointing_obj and (CurrentSimNanos*macros.NANO2MIN > 2.5):
        #     self.guid.sigma_R0N = [0.0, 0.0, 1.0]
        #     self.changed_pointing_obj = True

        self._eval_pointing_mode()
        
        self._guidance()

        self.nav.UpdateState(CurrentSimNanos)
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


    def _guidance(self) -> None:
        """
        Updates the desired MRP oerientation 'self.guid.sigma_R0N' based on the current pointing mode
        """
        self.guid.sigma_R0N = [0.0, 0.0, 1.0]

        match self.pointingMode:
            case PointingMode.COAST:
                self.guid.sigma_R0N = [1., 0. , 0.]
            case PointingMode.COMMS:
                self.guid.sigma_R0N = [0., 1., 0.]
            case _:
                raise ValueError(f"Undefined pointing mode reached in {self.ModelTag}")
        
    
    def _eval_pointing_mode(self) -> None:
        """
        Updates self.pointingMode based on the objects of intrest within the spacecraft's LOS 
        in accordance with the designed finite state machine.
        """
        old_pointing_mode = self.pointingMode

        any_gs_avail = False
        avail_gs_idx = -1
        for i, msg in enumerate(self.gsAccessMsgs):
            p = msg.read()
            
            # NOTE: First-come-first-serve logic.
            # TODO: Add battery condition
            # TODO: Add conditional logic with other modes
            if p.hasAccess:
                any_gs_avail = True
                avail_gs_idx = i
                break
        
        if any_gs_avail:
            self.pointingMode = PointingMode.COMMS
        else:
            self.pointingMode = PointingMode.COAST
        
        if old_pointing_mode != self.pointingMode:
            logging.debug(f"[FSW] {self.ModelTag} changed its pointing mode to {self.pointingMode}")
        
        