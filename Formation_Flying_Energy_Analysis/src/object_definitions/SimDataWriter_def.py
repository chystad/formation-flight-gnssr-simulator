import h5py
import logging
import numpy as np

from dataclasses import dataclass
from object_definitions.Config_def import Config
# from object_definitions.SimData_def import SimData
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle
from object_definitions.SimData_def import (SpacecraftSimData, MissionSimData)

class SimDataWriter:
    def __init__(self, cfg: Config, scSimDataList: list[SpacecraftSimData], missionSimData = None) -> None:
        
        self.output_data_save_dir = cfg.output_data_save_dir # All output data will be stored in this folder
        self.mc_enabled = cfg.mc_enabled
        self.data_mode = cfg.data_mode
        self.run_idx = cfg.run_idx # Needed for naming data file
        self.scSimDataList = scSimDataList
        self.missionSimData = missionSimData
        self.numSc = len(scSimDataList)
        self.output_data_every_24h = cfg.output_data_every_24h
        if cfg.output_data_every_24h:
            logging.debug(f"WARNING: SimDataWriter initialized even though 'output_data_every_24h' == True. RecorderFlusher should then be responible for data output. Therefore, the writer will always output the optimized data subset if 'write_data_to_files' is called.")

    
    def write_data_to_files(self) -> None:
        if self.data_mode == "optimized" or self.output_data_every_24h:
            self._write_reduced_data_to_files()
        else:
            self._write_full_data_to_files()

    
    def _write_reduced_data_to_files(self) -> None:
        """
        Write optimized per-spacecraft data to one HDF5 file per satellite.

        File structure:
            sat_X.h5
            |
            |---r_BN_N
                |---data
                |---dt_s
                |---n_samples
            |
            |---...
        """

        self.output_data_save_dir.mkdir(parents=True, exist_ok=True)

        for sat_idx, sc_data in enumerate(self.scSimDataList):
            out_path = self.output_data_save_dir / f"sat_{sat_idx}.h5"

            with h5py.File(out_path, "w") as h5: 
                # Mandatory optimized fields
                self._write_sampled_data_group(h5, "r_BN_N", sc_data.r_BN_N)
                self._write_sampled_data_group(h5, "v_BN_N", sc_data.v_BN_N)
                self._write_sampled_data_group(h5, "fuelMass", sc_data.fuelMass)
                self._write_sampled_data_group(h5, "storageLevel", sc_data.storageLevel)
                self._write_sampled_data_group(h5, "currentNetPower", sc_data.currentNetPower)
                self._write_sampled_data_group(h5, "pointingModeCode", sc_data.pointingModeCode)

                # Optional post-processed RTN fields
                if sc_data.r_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "r_scB_leaderB_RTN", sc_data.r_scB_leaderB_RTN)

                if sc_data.v_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "v_scB_leaderB_RTN", sc_data.v_scB_leaderB_RTN)

            logging.debug(f"[DATA_WRITER] Wrote optimized satellite data to {out_path}")

    
    def _write_full_data_to_files(self) -> None:
        """
        Write full per-spacecraft data to one HDF5 file per satellite.

        File structure:
            sat_X.h5
            |
            |---r_BN_N
                |---data
                |---dt_s
                |---n_samples
            |
            |---...
        """

        self.output_data_save_dir.mkdir(parents=True, exist_ok=True)

        for sat_idx, sc_data in enumerate(self.scSimDataList):
            out_path = self.output_data_save_dir / f"sat_{sat_idx}.h5"

            with h5py.File(out_path, "w") as h5: 
                # Mandatory fields
                self._write_sampled_data_group(h5, "r_BN_N", sc_data.r_BN_N)
                self._write_sampled_data_group(h5, "v_BN_N", sc_data.v_BN_N)
                self._write_sampled_data_group(h5, "fuelMass", sc_data.fuelMass)
                self._write_sampled_data_group(h5, "storageLevel", sc_data.storageLevel)
                self._write_sampled_data_group(h5, "currentNetPower", sc_data.currentNetPower)
                self._write_sampled_data_group(h5, "pointingModeCode", sc_data.pointingModeCode)

                # Optional debug fields
                if sc_data.r_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "r_scB_leaderB_RTN", sc_data.r_scB_leaderB_RTN)

                if sc_data.v_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "v_scB_leaderB_RTN", sc_data.v_scB_leaderB_RTN)

                if sc_data.sigma_BN is not None:
                    self._write_sampled_data_group(h5, "sigma_BN", sc_data.sigma_BN)
                
                if sc_data.omega_BN_B is not None:
                    self._write_sampled_data_group(h5, "omega_BN_B", sc_data.omega_BN_B)

                if sc_data.sigma_RN is not None:
                    self._write_sampled_data_group(h5, "sigma_RN", sc_data.sigma_RN)

                if sc_data.omega_RN_N is not None:
                    self._write_sampled_data_group(h5, "omega_RN_N", sc_data.omega_RN_N)

                if sc_data.sigma_BR is not None:
                    self._write_sampled_data_group(h5, "sigma_BR", sc_data.sigma_BR)

                if sc_data.omega_BR_B is not None:
                    self._write_sampled_data_group(h5, "omega_BR_B", sc_data.omega_BR_B)
                
                if sc_data.cmdTorqueBody is not None:
                    self._write_sampled_data_group(h5, "cmdTorqueBody", sc_data.cmdTorqueBody)
                
                if sc_data.cmdMotorTorque is not None:
                    self._write_sampled_data_group(h5, "cmdMotorTorque", sc_data.cmdMotorTorque)

                if sc_data.thrustForce_B is not None:
                    self._write_sampled_data_group(h5, "thrustForce_B", sc_data.thrustForce_B)

                if sc_data.thrustTorquePntB_B is not None:
                    self._write_sampled_data_group(h5, "thrustTorquePntB_B", sc_data.thrustTorquePntB_B)

                if sc_data.thrustBlowDownFactor is not None:
                    self._write_sampled_data_group(h5, "thrustBlowDownFactor", sc_data.thrustBlowDownFactor)
                
                if sc_data.ispBlowDownFactor is not None:
                    self._write_sampled_data_group(h5, "ispBlowDownFactor", sc_data.ispBlowDownFactor)
                
                if sc_data.rwOmega is not None:
                    self._write_sampled_data_group(h5, "rwOmega", sc_data.rwOmega)
                
                if sc_data.rwUCurrent is not None:
                    self._write_sampled_data_group(h5, "rwUCurrent", sc_data.rwUCurrent)

                if sc_data.rwNetPower is not None:
                    self._write_sampled_data_group(h5, "rwNetPower", sc_data.rwNetPower)

                if sc_data.obcNetPower is not None:
                    self._write_sampled_data_group(h5, "obcNetPower", sc_data.obcNetPower)

                if sc_data.comNetPower is not None:
                    self._write_sampled_data_group(h5, "comNetPower", sc_data.comNetPower)

                if sc_data.batHeatNetPower is not None:
                    self._write_sampled_data_group(h5, "batHeatNetPower", sc_data.batHeatNetPower)

                if sc_data.payNetPower is not None:
                    self._write_sampled_data_group(h5, "payNetPower", sc_data.payNetPower)

                if sc_data.propIdleNetPower is not None:
                    self._write_sampled_data_group(h5, "propIdleNetPower", sc_data.propIdleNetPower)

                if sc_data.propHeatNetPower is not None:
                    self._write_sampled_data_group(h5, "propHeatNetPower", sc_data.propHeatNetPower)

                if sc_data.propThrNetPower is not None:
                    self._write_sampled_data_group(h5, "propThrNetPower", sc_data.propThrNetPower)

                if sc_data.solarPanelNetPower is not None:
                    self._write_sampled_data_group(h5, "solarPanelNetPower", sc_data.solarPanelNetPower)
                

            logging.debug(f"[DATA_WRITER] Wrote optimized satellite data to {out_path}")



    ##########################
    # Private helper methods #
    ##########################

    def _write_sampled_data_group(self, h5: h5py.File, group_name: str, sampled_data) -> None:
        """
        Write one SampledData object to an HDF5 group.

        The array is written in 100,000 sample chunks to avoid unnecessary memory copies.
        """

        grp = h5.create_group(group_name)

        data = sampled_data.data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        shape = data.shape
        dtype = data.dtype

        # Choose chunking along the sample dimension.
        # Keeps chunks reasonably sized while supporting both (n,) and (n, m) arrays.
        if data.ndim == 1:
            chunk_shape = (min(100_000, shape[0]),)
        elif data.ndim == 2:
            chunk_shape = (min(100_000, shape[0]), shape[1])
        else:
            raise ValueError(
                f"Unsupported data dimension for '{group_name}'. "
                f"Expected 1D or 2D array, got shape {shape}."
            )

        dset = grp.create_dataset(
            "data",
            shape=shape,
            dtype=dtype,
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # Write in chunks along first axis to reduce peak memory use.
        chunk_n = chunk_shape[0]
        for start in range(0, shape[0], chunk_n):
            stop = min(start + chunk_n, shape[0])
            dset[start:stop] = data[start:stop]

        grp.create_dataset("dt_s", data=float(sampled_data.dt_s))
        grp.create_dataset("n_samples", data=int(sampled_data.n_samples))