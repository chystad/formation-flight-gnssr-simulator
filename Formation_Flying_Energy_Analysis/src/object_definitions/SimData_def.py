import h5py
import logging
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from numpy.typing import NDArray
from dataclasses import dataclass

from Basilisk.utilities import macros

from object_definitions.Config_def import Config
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle

# Global definition of data save folder path
OUTPUT_DATA_SAVE_DIR = Path('Formation_Flying_Energy_Analysis/output_data/sim_data')


@dataclass
class SampledData:
    data: NDArray[Any] # Data numpy array
    dt_s: float        # [s] Constant time between samples
    n_samples: int     # Number of samples

# TODO
@dataclass 
class MissionSimData:
    TODO: bool

@dataclass 
class SpacecraftSimData:
    # Mandetory data
    r_BN_N: SampledData # (n, 3) [m] B position relative to N, exporessed in N frame (float64)
    v_BN_N: SampledData # (n, 3) [m/s] B velocity relative to N, expressed in N frame (float64)
    fuelMass: SampledData # (n,) [kg] Fuel mass (float32)
    storageLevel: SampledData    # (n,) [Ws] Battery stored charge (float32)
    currentNetPower: SampledData # (n,) [W] Net power received/drained from the battery (float32)

    # Post-processed data
    r_scB_leaderB_RTN: Optional[SampledData] = None # (n, 3) [m] This sc position relative to leader, expressed in RTN frame (float64)
    v_scB_leaderB_RTN: Optional[SampledData] = None # (n, 3) [m/s] This sc velocity relative to leader, expressed in RTN frame (float64)

    # Optional 'debug' FSW-owned data
    sigma_BN: Optional[SampledData] = None   # (n, 3) [MRP] Attitude of B relative to N (float32)
    omega_BN_B: Optional[SampledData] = None # (n, 3) [rad/s] Angular rate of B relative to N, expressed in B frame (float32)

    sigma_RN: Optional[SampledData] = None   # (n, 3) [MRP] Desired attitude, R relative to N (float32)
    omega_RN_N: Optional[SampledData] = None # (n, 3) [rad/s] Desired angular rate, R relative to N, expressed in N frame (float32)

    sigma_BR: Optional[SampledData] = None   # (n, 3) [MRP] Attitude tracking error of B relative to R (float32)
    omega_BR_B: Optional[SampledData] = None # (n, 3) [rad/s] Angular rate tracking error of B relative to R, expressed in B frame (float32)

    cmdTorqueBody: Optional[SampledData] = None  # (n,      3) [Nm] Torque command from FSW, expressed in Body (float32)
    cmdMotorTorque: Optional[SampledData] = None # (n, numRWs) [Nm] Torque command from FSW, expressed in RW frame (float32)

    # Optional 'debug' Dynamics-owned data
    thrustForce_B: Optional[SampledData] = None        # (n, 3) [N] Thrust force vector, expressed in B frame (float32)
    thrustTorquePntB_B: Optional[SampledData] = None   # (n, 3) [Nm] Thrust torque about point B, expressed in B frame (float32)
    thrustBlowDownFactor: Optional[SampledData] = None # (n,) [frac/%???] Current thrust percentage due to tank blow down (float32)
    ispBlowDownFactor: Optional[SampledData] = None    # (n,) [frac/%???] Current Isp percentage due to tank blow down (float32)

    rwOmega: Optional[SampledData] = None    # (n, numRWs) [rad/s] Wheel speed (float32)
    rwUCurrent: Optional[SampledData] = None # (n, numRWs) [Nm] Motor torque (float32)
    rwNetPower: Optional[SampledData] = None # (n, numRWs) [W] RW power used/generated ( < 0 => Consume power) (float32)

    obcNetPower: Optional[SampledData] = None        # (n,) [W] Net power used/generated (float32)
    solarPanelNetPower: Optional[SampledData] = None # (n, numSPs) [W] Net power used/generated (float32)

@dataclass
class FormationFollowerStatus:
    sat_idx: int
    desired_along_track_m: float
    radial_error_m: float
    along_track_error_m: float
    cross_track_error_m: float
    pos_error_norm_m: float
    vel_error_norm_mps: float
    inside_tolerance: bool
    dwell_time_s: float
    achieved: bool
    burn_required: bool
    burn_direction_rtn: NDArray[np.float64]
    burn_attitude_mrp: NDArray[np.float64]
    thrust_on_time_s: float



class SimData:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg


    #########################
    # Public Helper methods #
    #########################
    
    def pull_every_spacecraft_data(self, 
                                   scRuntimeBundles: list[SpacecraftRuntimeBundle | None]
                                   ) -> list[SpacecraftSimData]:
        """
        Pull data from each spacecraft

        Returns:
            list[SpacecraftSimData]: data from list index #i corresponds to spacecraft #sat_idx data
        """

        # Raise error if scRuntimeBundles isn't initialized
        assert scRuntimeBundles is not None
        
        # Local data containers for spacecraft and mission data
        scSimDataList: list[SpacecraftSimData] = []
        
        # Extract per-spacecraft data from recorders
        for i in range(len(scRuntimeBundles)):
            scRuntimeBundle = scRuntimeBundles[i]

            # Validate satellite order
            assert scRuntimeBundle is not None
            if i != scRuntimeBundle.sat_idx:
                raise ValueError(f"Index mismatch between element #{i} in scRuntimeBundles and it satellite index #{scRuntimeBundle.sat_idx}")

            scSimData = self._pull_single_spacecraft_data(scRuntimeBundle, self.cfg.data_mode)
            scSimDataList.append(scSimData)

            # self._DEBUG_print_spacecraft_sim_data_field_sizes(scRuntimeBundle.sat_idx, scSimData)
            # self._DEBUG_print_spacecraft_sim_data_field_dtypes(scRuntimeBundle.sat_idx, scSimData)

        # Compute and add the RTN relative states for the follower satellites
        self._compute_RTN_leader_relative_states(scSimDataList)

        return scSimDataList
    


    ##########################
    # Private Helper methods #
    ##########################

    def _pull_single_spacecraft_data(self, 
                                     scRuntimeBundle: SpacecraftRuntimeBundle, 
                                     data_mode: str
                                     ) -> SpacecraftSimData:
        """
        Pull all relevant data fields from the recorders in SpacecraftRuntimeBundle.

        Returns:
            SpacecraftSimData: Data container for one spacecraft
        """
        
        dynModel = scRuntimeBundle.dynModel
        fsw = scRuntimeBundle.fsw

        # ---------------------------------------------------------
        # Mandatory data extraction as tuple (data, dt, n)
        # ---------------------------------------------------------

        # Translational states
        r_BN_N_data = SampledData(
            np.asarray(fsw.navTransRecorder.r_BN_N, dtype=np.float64),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navTransRecorder.r_BN_N))
        v_BN_N_data = SampledData(
            np.asarray(fsw.navTransRecorder.v_BN_N, dtype=np.float64),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navTransRecorder.v_BN_N))
        lowRateTimes = fsw.navTransRecorder.times() # NOTE This recorder is used to fetch LOW sample rate time vector
        fsw.navTransRecorder.clear() # clear buffer

        # Fuel tank state
        fuelMass_data = SampledData(
            np.asarray(dynModel.fuelTankStateRecorder.fuelMass, dtype=np.float32),
            dynModel.fuelTankStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.fuelTankStateRecorder.fuelMass))
        dynModel.fuelTankStateRecorder.clear()

        # Battery state
        storageLevel_data = SampledData(
            np.asarray(dynModel.batteryStateRecorder.storageLevel, dtype=np.float32),
            dynModel.batteryStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.batteryStateRecorder.storageLevel))
        currentNetPower_data = SampledData(
            np.asarray(dynModel.batteryStateRecorder.currentNetPower, dtype=np.float32),
            dynModel.batteryStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.batteryStateRecorder.currentNetPower))
        midRateTimes = dynModel.batteryStateRecorder.times() # NOTE This recorder is used to fetch MID sample rate time vector
        dynModel.batteryStateRecorder.clear()

        # Construct scSimData with only mandatory fields to minimize buffer
        if data_mode == "optimized":
            scSimData = SpacecraftSimData(
                r_BN_N=r_BN_N_data,
                v_BN_N=v_BN_N_data,
                fuelMass=fuelMass_data,
                storageLevel=storageLevel_data,
                currentNetPower=currentNetPower_data,
            )
        
        # Construct scSimData with all relevant fields for debugging 
        # NOTE: Will cause memory overload for longer time horizons!! NOT well suited for time horizons > 1 week
        elif data_mode == "debug":

            # ---------------------------------------------------------
            # FSW-owned 'debug' data extraction as tuple (data, dt, n)
            # ---------------------------------------------------------

            # Ensure that all FSW recorders described in FswStack._setup_fsw_recorders()
            # are available before extracting data.
            
            assert fsw.navAttRecorder is not None
            assert fsw.attRefRecorder is not None
            assert fsw.attErrRecorder is not None
            assert fsw.cmdTorqueRecorder is not None
            assert fsw.rwMotorTorqueRecorder is not None
            
            # Spacecraft attitude states
            sigma_BN_data = SampledData(
                np.asarray(fsw.navAttRecorder.sigma_BN, dtype=np.float32),
                fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.navAttRecorder.sigma_BN))
            omega_BN_B_data = SampledData(
                np.asarray(fsw.navAttRecorder.omega_BN_B, dtype=np.float32),
                fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.navAttRecorder.omega_BN_B))
            highRateTimes = dynModel.batteryStateRecorder.times() # NOTE This recorder is used to fetch HIGH sample rate time vector
            fsw.navAttRecorder.clear()

            # Desired attitude states
            sigma_RN_data = SampledData(
                np.asarray(fsw.attRefRecorder.sigma_RN, dtype=np.float32),
                fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.attRefRecorder.sigma_RN))
            omega_RN_N_data = SampledData(
                np.asarray(fsw.attRefRecorder.omega_RN_N, dtype=np.float32),
                fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.attRefRecorder.omega_RN_N))
            fsw.attRefRecorder.clear()

            # Attitude tracking errors
            sigma_BR_data = SampledData(
                np.asarray(fsw.attErrRecorder.sigma_BR, dtype=np.float32),
                fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.attErrRecorder.sigma_BR))
            omega_BR_B_data = SampledData(
                np.asarray(fsw.attErrRecorder.omega_BR_B, dtype=np.float32),
                fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.attErrRecorder.omega_BR_B))
            fsw.attErrRecorder.clear()

            # Control outputs
            cmdTorqueBody_data = SampledData(
                np.asarray(fsw.cmdTorqueRecorder.torqueRequestBody, dtype=np.float32),
                fsw.cmdTorqueRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.cmdTorqueRecorder.torqueRequestBody))
            cmdMotorTorque_data = SampledData(
                np.asarray(fsw.rwMotorTorqueRecorder.motorTorque, dtype=np.float32)[:, :dynModel.numRWs], # default size: (n, 36), reduce to (n, numRWs)
                fsw.rwMotorTorqueRecorder_RateNanos * macros.NANO2SEC,
                len(fsw.rwMotorTorqueRecorder.motorTorque))
            fsw.cmdTorqueRecorder.clear()
            fsw.rwMotorTorqueRecorder.clear()
            



            # ---------------------------------------------------------
            # Dynamics-owned data extraction as tuple (data, dt, n)
            # ---------------------------------------------------------

            assert dynModel.thrusterStateRecorder is not None
            assert dynModel.obcPowerSinkRecorder is not None
            assert len(dynModel.rwStateRecorders) == dynModel.numRWs
            assert len(dynModel.rwPowerRecorders) == dynModel.numRWs
            assert len(dynModel.solarPanelPowerRecorders) == dynModel.numSPs

            # Thruster state
            thrustForce_B_data = SampledData(
                np.asarray(dynModel.thrusterStateRecorder.thrustForce_B, dtype=np.float32),
                dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.thrusterStateRecorder.thrustForce_B))
            thrustTorquePntB_B_data = SampledData(
                np.asarray(dynModel.thrusterStateRecorder.thrustTorquePntB_B, dtype=np.float32),
                dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.thrusterStateRecorder.thrustTorquePntB_B))
            thrustBlowDownFactor_data = SampledData(
                np.asarray(dynModel.thrusterStateRecorder.thrustBlowDownFactor, dtype=np.float32),
                dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.thrusterStateRecorder.thrustBlowDownFactor))
            ispBlowDownFactor_data = SampledData(
                np.asarray(dynModel.thrusterStateRecorder.ispBlowDownFactor, dtype=np.float32),
                dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.thrusterStateRecorder.ispBlowDownFactor))
            dynModel.thrusterStateRecorder.clear()
            
            # Reaction wheel states, one array per RW
            rwOmega_data = SampledData(
                np.asarray([rec.Omega for rec in dynModel.rwStateRecorders], dtype=np.float32).T,
                dynModel.rwStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.rwStateRecorders[0].Omega) if dynModel.numRWs > 0 else 0)
            rwUCurrent_data = SampledData(
                np.asarray([rec.u_current for rec in dynModel.rwStateRecorders], dtype=np.float32).T,
                dynModel.rwStateRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.rwStateRecorders[0].u_current) if dynModel.numRWs > 0 else 0)
            for rec in dynModel.rwStateRecorders:
                rec.clear()

            # Reaction wheel power consumption, one array per RW
            rwNetPower_data = SampledData(
                np.asarray([rec.netPower for rec in dynModel.rwPowerRecorders], dtype=np.float32).T,
                dynModel.rwPowerRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.rwPowerRecorders[0].netPower) if dynModel.numRWs > 0 else 0)
            for rec in dynModel.rwPowerRecorders:
                rec.clear()

            # OBC power sink
            obcNetPower_data = SampledData(
                np.asarray(dynModel.obcPowerSinkRecorder.netPower, dtype=np.float32),
                dynModel.obcPowerSinkRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.obcPowerSinkRecorder.netPower))
            dynModel.obcPowerSinkRecorder.clear()

            # Solar panel power generation, one array per solar panel
            solarPanelNetPower_data = SampledData(
                np.asarray([rec.netPower for rec in dynModel.solarPanelPowerRecorders], dtype=np.float32).T,
                dynModel.solarPanelPowerRecorder_RateNanos * macros.NANO2SEC,
                len(dynModel.solarPanelPowerRecorders[0].netPower) if dynModel.numSPs > 0 else 0)
            for rec in dynModel.solarPanelPowerRecorders:
                rec.clear()
            
            scSimData = SpacecraftSimData(
                r_BN_N=r_BN_N_data,
                v_BN_N=v_BN_N_data,
                fuelMass=fuelMass_data,
                storageLevel=storageLevel_data,
                currentNetPower=currentNetPower_data,
                sigma_BN=sigma_BN_data,
                omega_BN_B=omega_BN_B_data,
                sigma_RN=sigma_RN_data,
                omega_RN_N=omega_RN_N_data,
                sigma_BR=sigma_BR_data,
                omega_BR_B=omega_BR_B_data,
                cmdTorqueBody=cmdTorqueBody_data,
                cmdMotorTorque=cmdMotorTorque_data,
                thrustForce_B=thrustForce_B_data,
                thrustTorquePntB_B=thrustTorquePntB_B_data,
                thrustBlowDownFactor=thrustBlowDownFactor_data,
                ispBlowDownFactor=ispBlowDownFactor_data,
                rwOmega=rwOmega_data,
                rwUCurrent=rwUCurrent_data,
                rwNetPower=rwNetPower_data,
                obcNetPower=obcNetPower_data,
                solarPanelNetPower=solarPanelNetPower_data,
            )
            
        else:
            raise AttributeError(f"Unrecognized config 'data_mode'. Got {self.cfg.data_mode}, expected ['debug', 'optimized']")
  
        logging.debug(f"[DATA] Extracted data from spacecraft #{scRuntimeBundle.sat_idx}")
        return scSimData
    

    def _compute_RTN_leader_relative_states(self, scSimDataList: list[SpacecraftSimData]) -> None:
        """
        Compute follower spacecraft translational states relative to the leader spacecraft,
        expressed in the leader RTN frame.

        Assumes leader is index 0

        For each follower:
            r_rel_N = r_follower_N - r_leader_N
            v_rel_N = v_follower_N - v_leader_N

        Then:
            r_rel_RTN = C_RTN_N @ r_rel_N
            v_rel_RTN = C_RTN_N @ v_rel_N

        The computed RTN states are stored in:
            follower.r_scB_leaderB_N
            follower.v_scB_leaderB_N
        """
        logging.debug(f"[DATA] Computing follower relative states and transforming into RTN frame")

        if len(scSimDataList) == 0:
            return

        leader = scSimDataList[0]

        r_leader_N = leader.r_BN_N.data
        v_leader_N = leader.v_BN_N.data

        if r_leader_N.shape != v_leader_N.shape:
            raise ValueError(
                f"Leader position and velocity shapes do not match. "
                f"Got r_BN_N {r_leader_N.shape}, v_BN_N {v_leader_N.shape}"
            )

        if r_leader_N.ndim != 2 or r_leader_N.shape[1] != 3:
            raise ValueError(
                f"Expected leader r_BN_N shape (n, 3), got {r_leader_N.shape}"
            )

        # Leader itself has zero relative state in its own RTN frame.
        leader.r_scB_leaderB_RTN = SampledData(
            data=np.zeros_like(r_leader_N),
            dt_s=leader.r_BN_N.dt_s,
            n_samples=leader.r_BN_N.n_samples,
        )
        leader.v_scB_leaderB_RTN = SampledData(
            data=np.zeros_like(v_leader_N),
            dt_s=leader.v_BN_N.dt_s,
            n_samples=leader.v_BN_N.n_samples,
        )

        for sat_idx in range(1, len(scSimDataList)):
            follower = scSimDataList[sat_idx]

            r_follower_N = follower.r_BN_N.data
            v_follower_N = follower.v_BN_N.data

            if r_follower_N.shape != r_leader_N.shape:
                raise ValueError(
                    f"Position shape mismatch for follower #{sat_idx}. "
                    f"Leader r_BN_N shape {r_leader_N.shape}, "
                    f"follower r_BN_N shape {r_follower_N.shape}"
                )

            if v_follower_N.shape != v_leader_N.shape:
                raise ValueError(
                    f"Velocity shape mismatch for follower #{sat_idx}. "
                    f"Leader v_BN_N shape {v_leader_N.shape}, "
                    f"follower v_BN_N shape {v_follower_N.shape}"
                )

            if follower.r_BN_N.dt_s != leader.r_BN_N.dt_s:
                raise ValueError(
                    f"Position sample time mismatch for follower #{sat_idx}. "
                    f"Leader dt {leader.r_BN_N.dt_s}, follower dt {follower.r_BN_N.dt_s}"
                )

            if follower.v_BN_N.dt_s != leader.v_BN_N.dt_s:
                raise ValueError(
                    f"Velocity sample time mismatch for follower #{sat_idx}. "
                    f"Leader dt {leader.v_BN_N.dt_s}, follower dt {follower.v_BN_N.dt_s}"
                )

            r_rel_N = r_follower_N - r_leader_N
            v_rel_N = v_follower_N - v_leader_N

            r_rel_RTN = self._transform_N_vectors_to_leader_RTN(
                vectors_N=r_rel_N,
                r_leader_N=r_leader_N,
                v_leader_N=v_leader_N,
            )

            v_rel_RTN = self._transform_N_vectors_to_leader_RTN(
                vectors_N=v_rel_N,
                r_leader_N=r_leader_N,
                v_leader_N=v_leader_N,
            )

            follower.r_scB_leaderB_RTN = SampledData(
                data=r_rel_RTN,
                dt_s=follower.r_BN_N.dt_s,
                n_samples=follower.r_BN_N.n_samples,
            )

            follower.v_scB_leaderB_RTN = SampledData(
                data=v_rel_RTN,
                dt_s=follower.v_BN_N.dt_s,
                n_samples=follower.v_BN_N.n_samples,
            )

            logging.debug(f"[DATA] Computed RTN leader-relative states for spacecraft #{sat_idx}")


    @staticmethod
    def _leader_dcm_N_to_RTN_for_all_times(
        r_leader_N: NDArray[np.float64],
        v_leader_N: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Build one DCM per sample that maps inertial N-frame vectors into the
        leader RTN frame.

        RTN basis:
            R-hat: radial, along leader position
            N-hat: orbit-normal, along r x v
            T-hat: transverse/in-track, N-hat x R-hat

        Returns:
            C_RTN_N with shape (n, 3, 3), such that:
                x_RTN = C_RTN_N @ x_N
        """

        r_norm = np.linalg.norm(r_leader_N, axis=1)
        h_leader_N = np.cross(r_leader_N, v_leader_N)
        h_norm = np.linalg.norm(h_leader_N, axis=1)

        if np.any(r_norm == 0.0):
            raise ValueError("Cannot construct RTN frame because at least one leader position norm is zero.")

        if np.any(h_norm == 0.0):
            raise ValueError("Cannot construct RTN frame because at least one leader angular momentum norm is zero.")

        R_hat_N = r_leader_N / r_norm[:, None]
        N_hat_N = h_leader_N / h_norm[:, None]
        T_hat_N = np.cross(N_hat_N, R_hat_N)

        C_RTN_N = np.stack((R_hat_N, T_hat_N, N_hat_N), axis=1)

        return C_RTN_N


    @classmethod
    def _transform_N_vectors_to_leader_RTN(
        cls,
        vectors_N: NDArray[np.float64],
        r_leader_N: NDArray[np.float64],
        v_leader_N: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Transform a time history of vectors from inertial N into leader RTN.

        Args:
            vectors_N:    shape (n, 3)
            r_leader_N:   shape (n, 3)
            v_leader_N:   shape (n, 3)

        Returns:
            vectors_RTN:  shape (n, 3)
        """

        C_RTN_N = cls._leader_dcm_N_to_RTN_for_all_times(r_leader_N, v_leader_N)

        vectors_RTN = np.einsum("nij,nj->ni", C_RTN_N, vectors_N)

        return vectors_RTN













    ##########################
    # Private DEEBUG methods #
    ##########################

    @staticmethod
    def _DEBUG_print_spacecraft_sim_data_field_sizes(sat_idx: int, 
                                                     scSimData: SpacecraftSimData
                                                     ) -> None:
        
        assert scSimData.sigma_BN is not None
        assert scSimData.omega_BN_B is not None
        assert scSimData.sigma_RN is not None
        assert scSimData.omega_RN_N is not None
        assert scSimData.sigma_BR is not None
        assert scSimData.omega_BR_B is not None
        assert scSimData.cmdTorqueBody is not None
        assert scSimData.cmdMotorTorque is not None
        assert scSimData.thrustTorquePntB_B is not None
        assert scSimData.thrustBlowDownFactor is not None
        assert scSimData.ispBlowDownFactor is not None
        assert scSimData.rwOmega is not None
        assert scSimData.rwUCurrent is not None
        assert scSimData.rwNetPower is not None
        assert scSimData.obcNetPower is not None
        assert scSimData.solarPanelNetPower is not None
        assert scSimData.thrustForce_B is not None
        logging.debug(f"""
[DATA] Spacecraft #{sat_idx} data shapes

FSW-owned data:
  r_BN_N:              {np.shape(scSimData.r_BN_N.data)}
  v_BN_N:              {np.shape(scSimData.v_BN_N.data)}
  sigma_BN:            {np.shape(scSimData.sigma_BN.data)}
  omega_BN_B:          {np.shape(scSimData.omega_BN_B.data)}
  sigma_RN:            {np.shape(scSimData.sigma_RN.data)}
  omega_RN_N:          {np.shape(scSimData.omega_RN_N.data)}
  sigma_BR:            {np.shape(scSimData.sigma_BR.data)}
  omega_BR_B:          {np.shape(scSimData.omega_BR_B.data)}
  cmdTorqueBody:       {np.shape(scSimData.cmdTorqueBody.data)}
  cmdMotorTorque:      {np.shape(scSimData.cmdMotorTorque.data)}

Dynamics-owned data:
  thrustForce_B:       {np.shape(scSimData.thrustForce_B.data)}
  thrustBlowDownFactor:{np.shape(scSimData.thrustBlowDownFactor.data)}
  ispBlowDownFactor:   {np.shape(scSimData.ispBlowDownFactor.data)}
  thrustTorquePntB_B:  {np.shape(scSimData.thrustTorquePntB_B.data)}
  fuelMass:            {np.shape(scSimData.fuelMass.data)}
  rwOmega:             {np.shape(scSimData.rwOmega.data)}
  rwUCurrent:          {np.shape(scSimData.rwUCurrent.data)}
  rwNetPower:          {np.shape(scSimData.rwNetPower.data)}
  storageLevel:        {np.shape(scSimData.storageLevel.data)}
  currentNetPower:     {np.shape(scSimData.currentNetPower.data)}
  obcNetPower:         {np.shape(scSimData.obcNetPower.data)}
  solarPanelNetPower:  {np.shape(scSimData.solarPanelNetPower.data)}
""")
        
    @staticmethod
    def _DEBUG_print_spacecraft_sim_data_field_dtypes(
        sat_idx: int,
        scSimData: SpacecraftSimData,
    ) -> None:
        """
        Print dtype information for all populated SpacecraftSimData fields.

        Optional fields are only accessed after explicit None checks/asserts,
        so Pylance can recognize that they are populated.
        """

        logging.debug(f"""
[DATA] Spacecraft #{sat_idx} data dtypes

Mandatory data:
  r_BN_N:              {scSimData.r_BN_N.data.dtype}
  v_BN_N:              {scSimData.v_BN_N.data.dtype}
  fuelMass:            {scSimData.fuelMass.data.dtype}
  storageLevel:        {scSimData.storageLevel.data.dtype}
  currentNetPower:     {scSimData.currentNetPower.data.dtype}
""")

        # -------------------------------------------------
        # Post-processed RTN data
        # -------------------------------------------------
        if scSimData.r_scB_leaderB_RTN is not None:
            assert scSimData.v_scB_leaderB_RTN is not None

            logging.debug(f"""
Post-processed RTN data:
  r_scB_leaderB_RTN:   {scSimData.r_scB_leaderB_RTN.data.dtype}
  v_scB_leaderB_RTN:   {scSimData.v_scB_leaderB_RTN.data.dtype}
""")

        # -------------------------------------------------
        # Optional FSW-owned debug data
        # -------------------------------------------------
        if scSimData.sigma_BN is not None:
            assert scSimData.omega_BN_B is not None
            assert scSimData.sigma_RN is not None
            assert scSimData.omega_RN_N is not None
            assert scSimData.sigma_BR is not None
            assert scSimData.omega_BR_B is not None
            assert scSimData.cmdTorqueBody is not None
            assert scSimData.cmdMotorTorque is not None

            logging.debug(f"""
FSW-owned debug data:
  sigma_BN:            {scSimData.sigma_BN.data.dtype}
  omega_BN_B:          {scSimData.omega_BN_B.data.dtype}
  sigma_RN:            {scSimData.sigma_RN.data.dtype}
  omega_RN_N:          {scSimData.omega_RN_N.data.dtype}
  sigma_BR:            {scSimData.sigma_BR.data.dtype}
  omega_BR_B:          {scSimData.omega_BR_B.data.dtype}
  cmdTorqueBody:       {scSimData.cmdTorqueBody.data.dtype}
  cmdMotorTorque:      {scSimData.cmdMotorTorque.data.dtype}
""")

        # -------------------------------------------------
        # Optional dynamics-owned debug data
        # -------------------------------------------------
        if scSimData.thrustForce_B is not None:
            assert scSimData.thrustTorquePntB_B is not None
            assert scSimData.thrustBlowDownFactor is not None
            assert scSimData.ispBlowDownFactor is not None
            assert scSimData.rwOmega is not None
            assert scSimData.rwUCurrent is not None
            assert scSimData.rwNetPower is not None
            assert scSimData.obcNetPower is not None
            assert scSimData.solarPanelNetPower is not None

            logging.debug(f"""
Dynamics-owned debug data:
  thrustForce_B:        {scSimData.thrustForce_B.data.dtype}
  thrustTorquePntB_B:   {scSimData.thrustTorquePntB_B.data.dtype}
  thrustBlowDownFactor: {scSimData.thrustBlowDownFactor.data.dtype}
  ispBlowDownFactor:    {scSimData.ispBlowDownFactor.data.dtype}
  rwOmega:              {scSimData.rwOmega.data.dtype}
  rwUCurrent:           {scSimData.rwUCurrent.data.dtype}
  rwNetPower:           {scSimData.rwNetPower.data.dtype}
  obcNetPower:          {scSimData.obcNetPower.data.dtype}
  solarPanelNetPower:   {scSimData.solarPanelNetPower.data.dtype}
""")