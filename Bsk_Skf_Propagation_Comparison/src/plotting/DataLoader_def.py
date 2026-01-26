import h5py
import logging
import numpy as np
from pathlib import Path

from object_definitions.Config_def import Config
from object_definitions.SimData_def import SimData, SimObjData, OUTPUT_DATA_SAVE_DIR


class DataLoader:
    def __init__(self) -> None:
        pass


    def load_sim_data_file(self, filename: str) -> SimData:
        """
        Read an HDF5 file at output_data/sim_out/<filename>[.h5] and return a SimData instance.

        Expected HDF5 layout:
        /time           (1,n)
        /objects/
            <satellite_name>/
                pos     (3,n)
                vel     (3,n)
        """
        logging.debug(f"Loading simulation data from {filename}")

        # normalize filename and build path
        fname = filename if filename.endswith(".h5") else f"{filename}.h5"
        file_path = OUTPUT_DATA_SAVE_DIR / fname
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        sim_obj_list: list[SimObjData] = []

        with h5py.File(file_path, "r") as f:
            # --- validate structure ---
            if "time" not in f:
                raise KeyError("Missing dataset '/time' in HDF5 file.")
            if "objects" not in f:
                raise KeyError("Missing group '/objects' in HDF5 file.")

            # load shared time vector (n,)
            time_dataset = self.get_dataset(f, "time")
            time = np.asarray(time_dataset[:], dtype=np.float64)

            # logging.debug(f"Data dimensions read from {filename}/time\n"
            #         f"                  time shape: {time.shape}")

            # iterate satellites
            objects_group = f["objects"]
            if not isinstance(objects_group, h5py.Group):
                raise KeyError("'/objects' must be a group in the HDF5 file.")
            
            if len(objects_group.keys()) == 0:
                raise ValueError("No objects found under '/objects'.")

            for sat_name in objects_group.keys():
                g = objects_group[sat_name]
                if not isinstance(g, h5py.Group):
                    raise KeyError("'/objects/sat_name' must be a group in the HDF5 file.")

                # required datasets
                if "pos" not in g or "vel" not in g:
                    raise KeyError(f"Object '{sat_name}' missing 'pos' or 'vel' dataset.")

                pos_dataset = self.get_dataset(g, "pos")
                vel_dataset = self.get_dataset(g, "vel")
                pos = np.asarray(pos_dataset[:], dtype=np.float64)
                vel = np.asarray(vel_dataset[:], dtype=np.float64)

                # logging.debug(f"Data dimensions read from from {filename}/objects/{sat_name}\n"
                #     f"                  pos shape:  {pos.shape}\n"
                #     f"                  vel shape:  {vel.shape}")

                # Ensure pos and vel are matrices
                if pos.ndim != 2 or vel.ndim != 2:
                    raise ValueError(f"'pos'/'vel' must be 2-D for '{sat_name}'. "
                                    f"Got pos {pos.shape}, vel {vel.shape}")
                
                # Expect stored as (3, n). Convert to (3, n) for SimObjData.
                if pos.shape[0] != 3 or vel.shape[0] != 3:
                    # If they were stored as (n, 3), transpose to (3, n) first for consistency
                    if pos.shape[1] == 3:
                        pos = pos.T
                    if vel.shape[1] == 3:
                        vel = vel.T
                    if pos.shape[1] != 3 or vel.shape[1] != 3:
                        raise ValueError(f"Expected shape (3, n) or (n, 3) for '{sat_name}'/pos and '{sat_name}'/vel. "
                                        f"Got pos {pos.shape}, vel {vel.shape}")


                # Verify that number of columns in time and pos/vel 
                if pos.shape[1] != time.shape[1] or vel.shape[1] != time.shape[1]:
                    raise ValueError(f"Time length {time.shape[1]} does not match pos/vel for '{sat_name}': "
                                    f"pos {pos.shape}, vel {vel.shape}")
                
                
                # logging.debug(f"Data dimensions loaded into SimData object from {sat_name}\n"
                #     f"                  time shape: {time.shape} \n"
                #     f"                  pos shape:  {pos.shape}\n"
                #     f"                  vel shape:  {vel.shape}")
                

                sim_obj = SimObjData(
                    satellite_name=sat_name,
                    time=time,
                    pos=pos,
                    vel=vel,
                )
                sim_obj_list.append(sim_obj)

        return SimData(sim_obj_list)
    

    def load_and_separate_data(self, datafiles_to_plot: list[str]) -> tuple[SimData, SimData]:
        """
        Read the 2 datafiles listed in 'datafiles_to_plot' and return one SimData object for each simulation framework. 
        The function expects 'datafiles_to_plot' to contain 2 strings, one containing the Skyfield identifyer 'skf', 
        and the other containing the Basilisk identifyer 'bsk'.

        Args:
            cfg (Config): simulation config object
            datafiles_to_plot (list[str]): list holding 2 filenames of data contained in 'output_data/sim_data'. 
                One filename must contain the identifyer 'skf', and the other 'bsk'.
        """
        assert len(datafiles_to_plot) == 2
        
        # Check which simulator generated the data
        skf_sim_data = None
        bsk_sim_data = None
        for datafile in datafiles_to_plot:
            sim_data = self.load_sim_data_file(datafile)
            if "skf" in datafile:
                skf_sim_data = sim_data
            elif "bsk" in datafile:
                bsk_sim_data = sim_data
            else:
                raise ValueError(f"Neither 'skf' or 'bsk' is present in datafile name: {datafile}"
                                f"The datafile cannot be categorized as skyfield or basilisk simulation output.")
            
        if skf_sim_data is None or bsk_sim_data is None:
            raise ValueError(f"one or both strings: 'skf', 'bsk' with was not found in a datafile starting with <cfg.timestamp_str>") 
        
        return skf_sim_data, bsk_sim_data
    

    @staticmethod
    def get_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
        """
        Load dataset from a parent group. Also ensure that the output is of type h5py.Dataset
        """
        obj = parent.get(name)
        if not isinstance(obj, h5py.Dataset):
            raise KeyError(f"Expected dataset at '{parent.name}/{name}'")
        return obj


    @ staticmethod
    def get_datafiles_by_timestamp(timestamp: str) -> list[str]:
        """
        Return the full filenames (str) of all .h5 files in OUTPUT_DATA_SAVE_DIR
        whose filename starts with the given timestamp.

        Args:
            timestamp (str): The timestamp string, e.g. "20251103_134512".

        Returns:
            list[str]: A list of matching full filenames.
        """
        if not isinstance(OUTPUT_DATA_SAVE_DIR, Path):
            raise TypeError("OUTPUT_DATA_SAVE_DIR must be a pathlib.Path object.")

        # Collect all matching .h5 files and return their names as strings
        matching_files = [
            file.name for file in OUTPUT_DATA_SAVE_DIR.glob(f"{timestamp}*.h5")
        ]

        if matching_files == []:
            raise FileNotFoundError(f"Datafile with timestamp: '{timestamp}' was not found.")

        return matching_files