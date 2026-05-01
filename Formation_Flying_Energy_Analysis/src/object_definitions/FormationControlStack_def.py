from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, TypeAlias

import logging
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, RigidBodyKinematics as rbk

from object_definitions.Config_def import Config
from object_definitions.SimData_def import FormationFollowerStatus
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle

if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 


BasiliskRecorder: TypeAlias = Any


class _FormationControlScheduler(sysModel.SysModel):
    def __init__(self, owner: "FormationControlStack"):
        super().__init__()
        self.owner = owner
        self.ModelTag = "FormationControlScheduler"

    def UpdateState(self, CurrentSimNanos: int) -> None:
        self.owner._update_state(CurrentSimNanos)

    def SelfInit(self):
        self.owner._self_init()

    def CrossInit(self):
        self.owner._cross_init()

    def Reset(self, CurrentSimNanos: int):
        self.owner._reset(CurrentSimNanos)



class FormationControlStack:
    """
    Formation-level control stack.

    Current implemented formation type:
        * constant_along_track

    Convention:
        * spacecraft index 0 is the chief / leader
        * followers target fixed Formation-relative RTN offsets
        * RTN basis is built from the leader Earth-relative ECI state:
            R: radial outward
            T: along-track / velocity-like direction
            N: orbit-normal
    """
    def __init__(self,
                 sim: BasiliskSimulator,
                 cfg: Config,
                 scRuntimeBundles: list[SpacecraftRuntimeBundle]) -> None:
        
        self.scheduler = _FormationControlScheduler(self)
        self.sim = sim
        self.cfg = cfg
        self.scRuntimeBundles = scRuntimeBundles
        self.numSatellites = len(scRuntimeBundles)
        self.logTag = "FORM"

        # Check correct assembly of runtime bundles
        if self.numSatellites != cfg.num_satellites:
            raise ValueError(f"""[{self.logTag}] The number of elements in scRuntimeBundles ({self.numSatellites}) """
                             f"""does not match the number of satellites from config ({cfg.num_satellites})""")
        
        # Raise error if too few satellites
        if self.numSatellites < 2:
            raise ValueError(f"[{self.logTag}] Formation control requires at least 2 spacecraft.")
        
        # TODO: Temp until I have implemented CPO formation
        if self.cfg.form_type != "constant_along_track":
            raise NotImplementedError(
                f"[{self.logTag}] Formation type '{self.cfg.form_type}' is not implemented yet. "
                "Currently supported: 'constant_along_track'."
            )
        
        # Create Formation control task and add it to the formation control process
        self.formationControlTaskName = f"FormationControlTask"
        sim.formationControlProcess.addTask(sim.CreateNewTask(self.formationControlTaskName, sim.formCtrlRateNanos))

        # Internal state
        self.lastUpdateNanos: Optional[int] = None
        self.dwellTimes: list[float] = [0.0] * self.numSatellites
        self.statusBySatIdx: dict[int, FormationFollowerStatus] = {}

        # Exposed formation-level status
        self.formationAchieved: bool = False
        self.maxPositionErrorM: float = cfg.form_pos_tolerance # TODO Use these parameters
        self.maxVelocityErrorMps: float = cfg.form_vel_tolerance # TODO Use these parameters

        # Per-spacecraft exposed commands. Index 0 leader is always no-burn.
        self.burnRequired: list[bool] = [False] * self.numSatellites
        self.burnAttitudeMrp: list[NDArray[np.float64]] = [
            np.zeros(3) for _ in range(self.numSatellites)
        ]
        self.thrustOnTimeS: list[float] = [0.0] * self.numSatellites

        # Output messages for Fsw and thruster effector
        self.form_att_ref_out_msgs: list[messaging.AttRefMsg] = []
        self.form_thr_cmd_out_msgs: list[messaging.THRArrayOnTimeCmdMsg] = []

        # Populate initial output messages
        for sat_idx in range(self.numSatellites):
            att_msg = messaging.AttRefMsg()
            att_payload = messaging.AttRefMsgPayload()
            att_payload.sigma_RN = [0.0, 0.0, 0.0]
            att_payload.omega_RN_N = [0.0, 0.0, 0.0]
            att_payload.domega_RN_N = [0.0, 0.0, 0.0]
            att_msg.write(att_payload)

            thr_msg = messaging.THRArrayOnTimeCmdMsg()
            thr_payload = messaging.THRArrayOnTimeCmdMsgPayload()
            thr_payload.OnTimeRequest = [0.0]
            thr_msg.write(thr_payload)

            self.form_att_ref_out_msgs.append(att_msg)
            self.form_thr_cmd_out_msgs.append(thr_msg)
        
        # TODO: Recorders can be added later after converting status/commands to messages.
        self.formationStatusRecorder: Optional[BasiliskRecorder] = None

        # Add scheduler (TODO: and recorders) to task
        sim.AddModelToTask(self.formationControlTaskName, self.scheduler, 20)

        logging.debug(
            f"[{self.logTag}] FormationControlStack initialized with "
            f"{self.numSatellites} spacecrafts, with formation type '{self.cfg.form_type}'"
        )
    
    
    
    
    ###########################
    # Public helper functions #
    ###########################
    
    def connect_form_ctrl_cmds_to_fsw(self) -> None:
        """
        
        """

            



    ##############################
    # SysModel Scheduler methods #
    ##############################

    def _modules(self):
        return []
    
    def _update_state(self, CurrentSimNanos: int) -> None:
        """
        Read all spacecraft states, evaluate formation error, decide if burns are needed,
        and expose formation-achieved status and per-follower burn commands.
        """

        dt = self._compute_dt(CurrentSimNanos)

        states = self._read_all_sc_states()
        chief_r_N, chief_v_N = states[0]

        C_RTN_N = self._eci_to_rtn_dcm(chief_r_N, chief_v_N)

        pos_errors: list[float] = []
        vel_errors: list[float] = []
        all_followers_achieved = True

        # Leader does not perform formation-control burns in this first implementation
        self.burnRequired[0] = False
        self.thrustOnTimeS[0] = 0.0
        self.burnAttitudeMrp[0] = np.zeros(3)

        # Compute formation-control attitude and burns for remainding follower satellites
        for sat_idx in range(1, self.numSatellites):
            follower_r_N, follower_v_N = states[sat_idx]

            rel_r_N = follower_r_N - chief_r_N
            rel_v_N = follower_v_N - chief_v_N

            rel_r_RTN = C_RTN_N @ rel_r_N
            rel_v_RTN = C_RTN_N @ rel_v_N

            desired_r_RTN = self._desired_constant_along_track_rtn(sat_idx, chief_r_N)
            desired_v_RTN = np.zeros(3)

            err_r_RTN = rel_r_RTN - desired_r_RTN
            err_v_RTN = rel_v_RTN - desired_v_RTN

            pos_err = float(np.linalg.norm(err_r_RTN))
            vel_err = float(np.linalg.norm(err_v_RTN))

            inside_tol = (
                pos_err <= self.cfg.form_pos_tolerance
                and vel_err <= self.cfg.form_vel_tolerance
            )

            if inside_tol:
                self.dwellTimes[sat_idx] += dt
            else:
                self.dwellTimes[sat_idx] = 0.0

            achieved = self.dwellTimes[sat_idx] >= self.cfg.dwell_time
            all_followers_achieved = all_followers_achieved and achieved

            burn_required, burn_dir_RTN, thrust_on_time_s = self._constant_along_track_control(
                err_r_RTN=err_r_RTN,
                err_v_RTN=err_v_RTN,
                inside_tol=inside_tol,
            )

            burn_att_mrp = self._burn_attitude_from_rtn_direction(
                burn_dir_RTN=burn_dir_RTN,
                C_RTN_N=C_RTN_N,
                sat_idx=sat_idx,
            )

            self.burnRequired[sat_idx] = burn_required
            self.thrustOnTimeS[sat_idx] = thrust_on_time_s
            self.burnAttitudeMrp[sat_idx] = burn_att_mrp

            status = FormationFollowerStatus(
                sat_idx=sat_idx,
                desired_along_track_m=float(desired_r_RTN[1]),
                radial_error_m=float(err_r_RTN[0]),
                along_track_error_m=float(err_r_RTN[1]),
                cross_track_error_m=float(err_r_RTN[2]),
                pos_error_norm_m=pos_err,
                vel_error_norm_mps=vel_err,
                inside_tolerance=inside_tol,
                dwell_time_s=self.dwellTimes[sat_idx],
                achieved=achieved,
                burn_required=burn_required,
                burn_direction_rtn=burn_dir_RTN,
                burn_attitude_mrp=burn_att_mrp,
                thrust_on_time_s=thrust_on_time_s,
            )
            self.statusBySatIdx[sat_idx] = status

            pos_errors.append(pos_err)
            vel_errors.append(vel_err)

        self.maxPositionErrorM = max(pos_errors) if pos_errors else 0.0
        self.maxVelocityErrorMps = max(vel_errors) if vel_errors else 0.0
        self.formationAchieved = all_followers_achieved

        self._write_output_messages()

        self.lastUpdateNanos = CurrentSimNanos


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


    #################################
    # Private Formation GNC methods #
    #################################

    def _desired_constant_along_track_rtn(self, 
                                          sat_idx: int, 
                                          chief_r_N: NDArray[np.float64]
                                          ) -> NDArray[np.float64]:
        """
        Follower-1 targets +1*cat_const_separation along-track from leader,
        Follower-2 targets +2*cat_const_separation, etc.
        +sign: in front of leader, 
        -sign: behind leader

        TODO: if relative position/velocity is already on one side, assign sign to reduce control effort needed
        """

        # Calculate desired constant along track difference
        # NOTE: Assuming leader is in a circular orbit
        r_chief_norm = np.linalg.norm(chief_r_N)
        desired_separation = sat_idx * self.cfg.cat_const_separation

        # Angle between the leader position vector and the relative position vector from leader to follower
        psi = np.acos((desired_separation**2) / (2*r_chief_norm*desired_separation))

        # Angle between normal plane defined by the leaders RTN 'r' vector (radial direction) and the relative position vector from leader to follower
        theta = np.pi/2 - psi

        x_cat_d = -1 * desired_separation * np.sin(theta) # Always negative in the RTN frame if on the same trajector
        y_cat_d = desired_separation * np.cos(theta)

        return np.array([x_cat_d, y_cat_d, 0.0], dtype=float)
    

    def _constant_along_track_control(
        self,
        err_r_RTN: NDArray[np.float64],
        err_v_RTN: NDArray[np.float64],
        inside_tol: bool,
    ) -> tuple[bool, NDArray[np.float64], float]:
        """
        First-pass impulsive-style proportional controller.

        This is intentionally conservative and simple:
            - no burn if inside tolerance
            - burn mostly along +/-T to reduce along-track error
            - include small radial/cross-track terms so deployment errors are not ignored

        Later, this method can be replaced by Basilisk's spacecraftReconfig /
        stationKeeping pipeline once the exact message interface is wired.
        """
        if inside_tol:
            return False, np.zeros(3), 0.0

        # Position and velocity feedback in RTN.
        # Gains are deliberately small because this maps directly to thruster on-time.
        k_pos = 2.0e-4   # [1/s]
        k_vel = 2.0      # [-]

        dv_cmd_RTN = -k_pos * err_r_RTN - k_vel * err_v_RTN

        dv_norm = float(np.linalg.norm(dv_cmd_RTN))
        if dv_norm < 1e-9:
            return False, np.zeros(3), 0.0

        burn_dir_RTN = dv_cmd_RTN / dv_norm

        # Convert commanded delta-v into a crude on-time estimate.
        # Assumes single thruster, max thrust from config, and current spacecraft mass approximated
        # by the satellite dry mass. This should be upgraded to include fuel mass.
        representative_mass = float(self.scRuntimeBundles[1].sat.m_s)
        max_thrust = max(float(self.cfg.max_thrust), 1e-12)

        thrust_on_time_s = representative_mass * dv_norm / max_thrust

        # Safety clamp: avoid huge burns from a bad initial condition.
        thrust_on_time_s = float(np.clip(thrust_on_time_s, 0.0, self.sim.formCtrlRateNanos * macros.NANO2SEC))

        burn_required = thrust_on_time_s > 0.0
        return burn_required, burn_dir_RTN, thrust_on_time_s
    

    def _burn_attitude_from_rtn_direction(
        self,
        burn_dir_RTN: NDArray[np.float64],
        C_RTN_N: NDArray[np.float64],
        sat_idx: int,
    ) -> NDArray[np.float64]:
        """
        Compute an attitude reference that points the configured thruster direction
        along the desired inertial burn direction.

        Assumption:
            cfg.thr_dir_B is the direction of produced thrust in body coordinates.
        """
        if np.linalg.norm(burn_dir_RTN) <= 0.0:
            return np.zeros(3)

        burn_dir_N = C_RTN_N.T @ burn_dir_RTN
        burn_dir_N = burn_dir_N / np.linalg.norm(burn_dir_N)

        thr_dir_B = np.array(self.cfg.thr_dir_B, dtype=float)
        thr_dir_B = thr_dir_B / np.linalg.norm(thr_dir_B)

        # Build a desired body frame in inertial coordinates where body thrust axis
        # aligns with burn_dir_N. For the common case thr_dir_B = +/-X/Y/Z,
        # this constructs a complete DCM robustly enough for first implementation.
        b1_N = burn_dir_N

        # Pick a helper vector not parallel to the burn direction.
        helper_N = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(helper_N, b1_N)) > 0.9:
            helper_N = np.array([0.0, 1.0, 0.0])

        b2_N = np.cross(helper_N, b1_N)
        b2_N = b2_N / np.linalg.norm(b2_N)
        b3_N = np.cross(b1_N, b2_N)
        b3_N = b3_N / np.linalg.norm(b3_N)

        # C_BN maps inertial components into desired body components.
        # First build a body frame assuming +X is thrust axis.
        C_Xthrust_N = np.vstack((b1_N, b2_N, b3_N))

        # If your thruster is not +X_B, rotate the body frame so cfg.thr_dir_B
        # becomes the aligned axis. For axis-aligned thrusters this is enough.
        C_BT = self._body_axis_alignment_dcm(thr_dir_B)
        C_BN = C_BT @ C_Xthrust_N

        sigma_BN = np.array(rbk.C2MRP(C_BN), dtype=float)
        return sigma_BN

    
    ##########################
    # Private helper methods #
    ##########################

    def _write_output_messages(self) -> None:
        """
        Publish the latest formation-control burn attitude and thruster on-time
        commands for each spacecraft.

        Leader index 0 receives no burn command.
        """

        for sat_idx in range(self.numSatellites):

            # -----------------------------
            # Attitude reference output
            # -----------------------------
            att_payload = messaging.AttRefMsgPayload()

            if sat_idx == 0:
                att_payload.sigma_RN = [0.0, 0.0, 0.0]
            else:
                att_payload.sigma_RN = self.burnAttitudeMrp[sat_idx].tolist()

            att_payload.omega_RN_N = [0.0, 0.0, 0.0]
            att_payload.domega_RN_N = [0.0, 0.0, 0.0]

            self.form_att_ref_out_msgs[sat_idx].write(att_payload)

            # -----------------------------
            # Thruster command output
            # -----------------------------
            thr_payload = messaging.THRArrayOnTimeCmdMsgPayload()

            if sat_idx == 0 or not self.burnRequired[sat_idx]:
                thr_payload.OnTimeRequest = [0.0]
            else:
                thr_payload.OnTimeRequest = [float(self.thrustOnTimeS[sat_idx])]

            self.form_thr_cmd_out_msgs[sat_idx].write(thr_payload)


    def _compute_dt(self, CurrentSimNanos: int) -> float:
        """
        Return the time difference in seconds between _update_state calls.
        This should always be 'formCtrlRateNanos * macros.NANO2SEC', but method is included as a precausion
        """
        if self.lastUpdateNanos is None:
            return float(self.sim.formCtrlRateNanos) * macros.NANO2SEC

        dt = (CurrentSimNanos - self.lastUpdateNanos) * macros.NANO2SEC
        if dt <= 0.0:
            dt = float(self.sim.formCtrlRateNanos) * macros.NANO2SEC
        return float(dt)
    
    
    def _read_all_sc_states(self) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Reads spacecraft position and velocity in ECI (N) frame from each scObj in scRuntimeBundles 
        and returns a list containing (pos, vel) tuples for each scObj
        """
        states: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

        for bundle in self.scRuntimeBundles:
            payload = bundle.scObj.scStateOutMsg.read()
            r_BN_N = np.array(payload.r_BN_N, dtype=float)
            v_BN_N = np.array(payload.v_BN_N, dtype=float)
            states.append((r_BN_N, v_BN_N))

        return states


    @staticmethod
    def _eci_to_rtn_dcm(
        r_N: NDArray[np.float64],
        v_N: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Returns Direction Cosine Matrix 'C_RTN_N' such that x_RTN = C_RTN_N @ x_N.
        """
        r_norm = np.linalg.norm(r_N)
        if r_norm <= 0.0:
            raise ValueError("[FORM] Chief position norm is zero; cannot define RTN frame.")

        r_hat = r_N / r_norm

        h_N = np.cross(r_N, v_N)
        h_norm = np.linalg.norm(h_N)
        if h_norm <= 0.0:
            raise ValueError("[FORM] Chief angular momentum norm is zero; cannot define RTN frame.")

        n_hat = h_N / h_norm
        t_hat = np.cross(n_hat, r_hat)
        t_hat = t_hat / np.linalg.norm(t_hat)

        return np.vstack((r_hat, t_hat, n_hat))
    

    @staticmethod
    def _body_axis_alignment_dcm(thr_dir_B: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns C_BT such that the actual body thrust axis is treated as the +X axis
        used internally in _burn_attitude_from_rtn_direction().
        """
        axis = thr_dir_B / np.linalg.norm(thr_dir_B)

        candidates = {
            (1.0, 0.0, 0.0): np.eye(3),
            (-1.0, 0.0, 0.0): np.diag([-1.0, -1.0, 1.0]),
            (0.0, 1.0, 0.0): np.array([[0.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0]]),
            (0.0, -1.0, 0.0): np.array([[0.0, -1.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0]]),
            (0.0, 0.0, 1.0): np.array([[0.0, 0.0, 1.0],
                                       [0.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0]]),
            (0.0, 0.0, -1.0): np.array([[0.0, 0.0, -1.0],
                                        [0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0]]),
        }

        rounded = tuple(np.round(axis, decimals=6))
        if rounded not in candidates:
            raise NotImplementedError(
                "[FORM] Non-axis-aligned thruster directions are not supported "
                "by this first burn-attitude helper."
            )

        return candidates[rounded]