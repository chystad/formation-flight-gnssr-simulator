# object_definitions/RecorderFlushSysModel_def.py

from __future__ import annotations

import logging
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from Basilisk.architecture import sysModel
from Basilisk.utilities import macros

from object_definitions.Config_def import Config
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle

from constants import(
    FLUSH_RATE,
    HIGH_SAMPLE_RATE,
    MID_SAMPLE_RATE,
    LOW_SAMPLE_RATE,
)


class RecorderFlusher(sysModel.SysModel):
    """
    Periodically flush Basilisk recorder buffers to per-satellite HDF5 files.

    File structure matches SimDataWriter:
        sat_X.h5
        |
        |---field_name
            |---data
            |---dt_s
            |---n_samples

    Important:
        - This should run less frequently than the recorders.
        - This clears recorder buffers after successful append.
        - RTN leader-relative states are computed chunk-wise before clearing.
    """

    def __init__(
        self,
        cfg: Config,
        sc_runtime_bundles: list[SpacecraftRuntimeBundle | None],
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.sc_runtime_bundles = sc_runtime_bundles
        self.output_data_save_dir = cfg.output_data_save_dir
        self.full_debug = (not cfg.mc_enabled) and cfg.data_mode == "debug"
        self.logTag = "REC_FLUSH"

        self.ModelTag = "RecorderFlushSysModel"

        # Check if the task rate is compatible with all recorder sample rates
        rates = {
            "HIGH_SAMPLE_RATE": macros.sec2nano(HIGH_SAMPLE_RATE),
            "MID_SAMPLE_RATE": macros.sec2nano(MID_SAMPLE_RATE),
            "LOW_SAMPLE_RATE": macros.sec2nano(LOW_SAMPLE_RATE),
        }
        for name, rate in rates.items():
            divisible = macros.hour2nano(FLUSH_RATE) % rate == 0.0
            if not divisible:
                raise ValueError(f"FLUSH_RATE is not divisible by {name}")
            
        logging.debug(f"[{self.logTag}] Recorder flusher setup complete")

        
    def UpdateState(self, CurrentSimNanos: int) -> None:
        # logging.debug(f"[{self.logTag}] Did nothing @ {CurrentSimNanos * macros.NANO2HOUR:.2f}")#
        self.flush(CurrentSimNanos)

    def flush(self, CurrentSimNanos: int | None = None) -> None:
        
        bundles = [b for b in self.sc_runtime_bundles if b is not None]
        if len(bundles) == 0:
            logging.debug(f"[{self.logTag}] No spacecraft runtime bundles found in 'sc_runtime_bundles'")
            return

        # Check if the output directory exists. Create if not. 
        self.output_data_save_dir.mkdir(parents=True, exist_ok=True)

        # First collect all chunks before clearing any recorder.
        # This is required because follower RTN states need leader data.
        chunks: list[dict[str, tuple[NDArray[Any], float]]] = []

        # Collect all data chunks from all spacecrafts depending on 'full_debug' flag
        for bundle in bundles:
            chunks.append(self._collect_spacecraft_chunk(bundle))

        if len(chunks[0]["r_BN_N"][0]) == 0:
            logging.debug(f"[{self.logTag}] No data in loaded chunks")
            return

        # Add leader-relative RTN states to chunk. zeros for leader.
        self._add_chunk_rtn_states(chunks)

        for bundle, chunk in zip(bundles, chunks):
            sat_idx = bundle.sat_idx
            out_path = self.output_data_save_dir / f"sat_{sat_idx}.h5"

            with h5py.File(out_path, "a") as h5:
                for field_name, (data, dt_s) in chunk.items():
                    self._append_sampled_data_group(h5, field_name, data, dt_s)

        # Only clear after all files were successfully written.
        for bundle in bundles:
            self._clear_spacecraft_recorders(bundle)

        if CurrentSimNanos is not None:
            logging.debug(
                f"[{self.logTag}] Flushed recorder data at "
                f"{CurrentSimNanos * macros.NANO2HOUR:.2f} hours"
            )

        del chunks

    def _collect_spacecraft_chunk(
        self,
        bundle: SpacecraftRuntimeBundle,
    ) -> dict[str, tuple[NDArray[Any], float]]:
        dyn = bundle.dynModel
        fsw = bundle.fsw

        chunk: dict[str, tuple[NDArray[Any], float]] = {}


        # -------------------------------------------------
        # Debug fields
        # -------------------------------------------------
        chunk["lowRateTimes"] = (
            np.asarray(fsw.navTransRecorder.times(), dtype=np.uint64),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
        )

        # -------------------------------------------------
        # Optimized / mandatory fields
        # -------------------------------------------------
        chunk["r_BN_N"] = (
            np.asarray(fsw.navTransRecorder.r_BN_N, dtype=np.float64),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["v_BN_N"] = (
            np.asarray(fsw.navTransRecorder.v_BN_N, dtype=np.float64),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["fuelMass"] = (
            np.asarray(dyn.fuelTankStateRecorder.fuelMass, dtype=np.float32),
            dyn.fuelTankStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["storageLevel"] = (
            np.asarray(dyn.batteryStateRecorder.storageLevel, dtype=np.float32),
            dyn.batteryStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["currentNetPower"] = (
            np.asarray(dyn.batteryStateRecorder.currentNetPower, dtype=np.float32),
            dyn.batteryStateRecorder_RateNanos * macros.NANO2SEC,
        )

        if not self.full_debug:
            return chunk

        # -------------------------------------------------
        # Full debug fields
        # -------------------------------------------------
        assert fsw.navAttRecorder is not None
        assert fsw.attRefRecorder is not None
        assert fsw.attErrRecorder is not None
        assert fsw.cmdTorqueRecorder is not None
        assert fsw.rwMotorTorqueRecorder is not None

        chunk["sigma_BN"] = (
            np.asarray(fsw.navAttRecorder.sigma_BN, dtype=np.float32),
            fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["omega_BN_B"] = (
            np.asarray(fsw.navAttRecorder.omega_BN_B, dtype=np.float32),
            fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["sigma_RN"] = (
            np.asarray(fsw.attRefRecorder.sigma_RN, dtype=np.float32),
            fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["omega_RN_N"] = (
            np.asarray(fsw.attRefRecorder.omega_RN_N, dtype=np.float32),
            fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["sigma_BR"] = (
            np.asarray(fsw.attErrRecorder.sigma_BR, dtype=np.float32),
            fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["omega_BR_B"] = (
            np.asarray(fsw.attErrRecorder.omega_BR_B, dtype=np.float32),
            fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["cmdTorqueBody"] = (
            np.asarray(fsw.cmdTorqueRecorder.torqueRequestBody, dtype=np.float32),
            fsw.cmdTorqueRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["cmdMotorTorque"] = (
            np.asarray(fsw.rwMotorTorqueRecorder.motorTorque, dtype=np.float32)[:, :dyn.numRWs],
            fsw.rwMotorTorqueRecorder_RateNanos * macros.NANO2SEC,
        )

        assert dyn.thrusterStateRecorder is not None
        assert dyn.obcPowerSinkRecorder is not None

        chunk["thrustForce_B"] = (
            np.asarray(dyn.thrusterStateRecorder.thrustForce_B, dtype=np.float32),
            dyn.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["thrustTorquePntB_B"] = (
            np.asarray(dyn.thrusterStateRecorder.thrustTorquePntB_B, dtype=np.float32),
            dyn.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["thrustBlowDownFactor"] = (
            np.asarray(dyn.thrusterStateRecorder.thrustBlowDownFactor, dtype=np.float32),
            dyn.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["ispBlowDownFactor"] = (
            np.asarray(dyn.thrusterStateRecorder.ispBlowDownFactor, dtype=np.float32),
            dyn.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
        )

        chunk["rwOmega"] = (
            np.asarray([rec.Omega for rec in dyn.rwStateRecorders], dtype=np.float32).T,
            dyn.rwStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["rwUCurrent"] = (
            np.asarray([rec.u_current for rec in dyn.rwStateRecorders], dtype=np.float32).T,
            dyn.rwStateRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["rwNetPower"] = (
            np.asarray([rec.netPower for rec in dyn.rwPowerRecorders], dtype=np.float32).T,
            dyn.rwPowerRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["obcNetPower"] = (
            np.asarray(dyn.obcPowerSinkRecorder.netPower, dtype=np.float32),
            dyn.obcPowerSinkRecorder_RateNanos * macros.NANO2SEC,
        )
        chunk["solarPanelNetPower"] = (
            np.asarray([rec.netPower for rec in dyn.solarPanelPowerRecorders], dtype=np.float32).T,
            dyn.solarPanelPowerRecorder_RateNanos * macros.NANO2SEC,
        )

        return chunk

    def _add_chunk_rtn_states(
        self,
        chunks: list[dict[str, tuple[NDArray[Any], float]]],
    ) -> None:
        """
        Compute follower spacecrafts translational states relative to the leader spacecraft,
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
        leader_r = chunks[0]["r_BN_N"][0]
        leader_v = chunks[0]["v_BN_N"][0]
        dt_s = chunks[0]["r_BN_N"][1]

        if leader_r.shape[0] == 0:
            return

        chunks[0]["r_scB_leaderB_RTN"] = (np.zeros_like(leader_r), dt_s)
        chunks[0]["v_scB_leaderB_RTN"] = (np.zeros_like(leader_v), dt_s)

        C_RTN_N = self._leader_dcm_N_to_RTN_for_all_times(leader_r, leader_v)

        for sat_idx in range(1, len(chunks)):
            follower_r = chunks[sat_idx]["r_BN_N"][0]
            follower_v = chunks[sat_idx]["v_BN_N"][0]

            if follower_r.shape != leader_r.shape:
                raise ValueError(
                    f"Chunk position shape mismatch for sat {sat_idx}: "
                    f"leader {leader_r.shape}, follower {follower_r.shape}"
                )

            if follower_v.shape != leader_v.shape:
                raise ValueError(
                    f"Chunk velocity shape mismatch for sat {sat_idx}: "
                    f"leader {leader_v.shape}, follower {follower_v.shape}"
                )

            r_rel_N = follower_r - leader_r
            v_rel_N = follower_v - leader_v

            r_rel_RTN = np.einsum("nij,nj->ni", C_RTN_N, r_rel_N)
            v_rel_RTN = np.einsum("nij,nj->ni", C_RTN_N, v_rel_N)

            chunks[sat_idx]["r_scB_leaderB_RTN"] = (r_rel_RTN, dt_s)
            chunks[sat_idx]["v_scB_leaderB_RTN"] = (v_rel_RTN, dt_s)

    @staticmethod
    def _leader_dcm_N_to_RTN_for_all_times(
        r_leader_N: NDArray[np.float64],
        v_leader_N: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r_norm = np.linalg.norm(r_leader_N, axis=1)
        h_leader_N = np.cross(r_leader_N, v_leader_N)
        h_norm = np.linalg.norm(h_leader_N, axis=1)

        if np.any(r_norm == 0.0):
            raise ValueError("Cannot construct RTN frame: leader position norm is zero.")

        if np.any(h_norm == 0.0):
            raise ValueError("Cannot construct RTN frame: leader angular momentum norm is zero.")

        R_hat_N = r_leader_N / r_norm[:, None]
        N_hat_N = h_leader_N / h_norm[:, None]
        T_hat_N = np.cross(N_hat_N, R_hat_N)

        return np.stack((R_hat_N, T_hat_N, N_hat_N), axis=1)

    def _append_sampled_data_group(
        self,
        h5: h5py.File,
        group_name: str,
        data: NDArray[Any],
        dt_s: float,
    ) -> None:
        if data.shape[0] == 0:
            return

        if data.ndim not in (1, 2):
            raise ValueError(
                f"Unsupported data dimension for '{group_name}'. "
                f"Expected 1D or 2D, got shape {data.shape}."
            )

        if group_name not in h5:
            grp = h5.create_group(group_name)

            maxshape = (None,) if data.ndim == 1 else (None, data.shape[1])
            chunk_shape = (
                min(100_000, data.shape[0]),
            ) if data.ndim == 1 else (
                min(100_000, data.shape[0]),
                data.shape[1],
            )

            grp.create_dataset(
                "data",
                data=data,
                maxshape=maxshape,
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            grp.create_dataset("dt_s", data=float(dt_s))
            grp.create_dataset("n_samples", data=int(data.shape[0]))
            return

        grp = h5[group_name]
        dset = grp["data"]

        old_n = int(dset.shape[0])
        new_n = old_n + int(data.shape[0])

        if dset.ndim != data.ndim:
            raise ValueError(
                f"Dimension mismatch for '{group_name}': "
                f"existing {dset.shape}, new {data.shape}"
            )

        if data.ndim == 2 and dset.shape[1] != data.shape[1]:
            raise ValueError(
                f"Column mismatch for '{group_name}': "
                f"existing {dset.shape}, new {data.shape}"
            )

        dset.resize((new_n,) if data.ndim == 1 else (new_n, data.shape[1]))
        dset[old_n:new_n] = data

        grp["dt_s"][...] = float(dt_s)
        grp["n_samples"][...] = int(new_n)

    def _clear_spacecraft_recorders(self, bundle: SpacecraftRuntimeBundle) -> None:
        dyn = bundle.dynModel
        fsw = bundle.fsw

        fsw.navTransRecorder.clear()
        dyn.fuelTankStateRecorder.clear()
        dyn.batteryStateRecorder.clear()

        if not self.full_debug:
            return

        assert fsw.navAttRecorder is not None
        assert fsw.attRefRecorder is not None
        assert fsw.attErrRecorder is not None
        assert fsw.cmdTorqueRecorder is not None
        assert fsw.rwMotorTorqueRecorder is not None

        fsw.navAttRecorder.clear()
        fsw.attRefRecorder.clear()
        fsw.attErrRecorder.clear()
        fsw.cmdTorqueRecorder.clear()
        fsw.rwMotorTorqueRecorder.clear()

        if dyn.thrusterStateRecorder is not None:
            dyn.thrusterStateRecorder.clear()

        for rec in dyn.rwStateRecorders:
            rec.clear()

        for rec in dyn.rwPowerRecorders:
            rec.clear()

        if dyn.obcPowerSinkRecorder is not None:
            dyn.obcPowerSinkRecorder.clear()

        for rec in dyn.solarPanelPowerRecorders:
            rec.clear()