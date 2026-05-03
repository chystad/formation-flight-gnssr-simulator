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

    
    def write_data_to_files(self) -> None:
        if self.mc_enabled or (self.data_mode == "optimized"):
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

                # Optional post-processed RTN fields
                if sc_data.r_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "r_scB_leaderB_RTN", sc_data.r_scB_leaderB_RTN)

                if sc_data.v_scB_leaderB_RTN is not None:
                    self._write_sampled_data_group(h5, "v_scB_leaderB_RTN", sc_data.v_scB_leaderB_RTN)

            logging.debug(f"[DATA_WRITER] Wrote optimized satellite data to {out_path}")

    
    def _write_full_data_to_files(self) -> None:
        pass



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