import h5py
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from numpy.typing import NDArray
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# Global definition of data save folder path
OUTPUT_DATA_SAVE_DIR = Path('Bsk_Skf_Propagation_Comparison/output_data/sim_data')


@dataclass_json
@dataclass
class SimObjData:
    satellite_name: str
    time: NDArray[np.float64] # (1,n)
    pos: NDArray[np.float64] # (3,n)
    vel: NDArray[np.float64] # (3,n)

@dataclass_json
@dataclass
class RelObjData:
    satellite_name: str
    time: NDArray[np.float64] # (1,n)
    rel_pos: NDArray[np.float64] # (3,n)
    rel_vel: NDArray[np.float64] # (3,n)



class SimData:
    def __init__(self, all_sim_obj_data: list[SimObjData]) -> None:
        """
        =========================================================================================================
        INPUTS:
            all_sim_obj_data (list[SimObjData])
        
        ATTRIBUTES:
            sim_data (list[SimObjData])
            rel_data (Optional[list[RelObjData]])

        METHODS:
            extract_time_vec
            write_data_to_file
            create_data_filename
        =========================================================================================================
        """
        
        self.sim_data: list[SimObjData] = all_sim_obj_data
        self.rel_data: Optional[list[RelObjData]] = None
        return


    def extract_time_vec(self) -> NDArray[np.float64]:
        """
        Verify that all elements in all the object's time vector are the same. If they are, return the time vector
        """

        # Verify that all elements in all the object's time vector are the same
        prev_obj_t = self.sim_data[0].time
        for i in range(1, len(self.sim_data)):
            curr_obj_t = self.sim_data[i].time

            if not np.array_equal(curr_obj_t, prev_obj_t):
                raise ValueError("Time vectors within SimData object are not the same")
            
            prev_obj_t = curr_obj_t

        # If all time vectors are the same, return the last one
        return prev_obj_t
    
    
    def write_data_to_file(self, timestamp_str: str, sim_type: str) -> None:
        """
        Saves the simulation data to OUTPUT_DATA_SAVE_DIR/<filename>.h5
        The output data file will have the following structure:

        /time           (1,n)
        /objects/
            <satellite_name>/
                pos     (3,n)
                vel     (3,n)
        """
        # Ensure target directory exists
        OUTPUT_DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Generate time dependant filename 
        filename = self.create_data_filename(timestamp_str, sim_type)

        # Construct full file path
        file_path = OUTPUT_DATA_SAVE_DIR / f"{filename}.h5"
        
        # Get simulation time vector
        time = self.extract_time_vec()

        with h5py.File(file_path, "w") as f:
            # Ensure the time vector has the correct shape
            if time.ndim == 1:
                time = time.reshape(1,-1)
            elif time.shape[1] == 1:
                time = time.reshape(1,-1)
            elif time.shape[0] == 1 and len(time) > 0:
                time = time
            else:
                raise ValueError("Could not convert time vector into (1,n) dimension")
            
            f.create_dataset("time", data=time, compression="gzip", compression_opts=4, shuffle=True)
            g_objs = f.create_group("objects")

            for obj_data in self.sim_data:
                g = g_objs.create_group(obj_data.satellite_name)

                # Ensure shape (3,n): transpose the arrays are (n,3)
                pos = obj_data.pos.T if obj_data.pos.shape[1] == 3 else obj_data.pos
                vel = obj_data.vel.T if obj_data.vel.shape[1] == 3 else obj_data.vel

                g.create_dataset("pos", data=pos, compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("vel", data=vel, compression="gzip", compression_opts=4, shuffle=True)

                g["pos"].attrs["units"] = "m"
                g["vel"].attrs["units"] = "m/s"
                g.attrs["description"] = "Satellite trajectory data" # Specify frame here

            logging.debug(f"Data written to: {file_path}")
        return


    def create_data_filename(self, timestamp_str: str, sim_type: str) -> str:
        """
        Generating time-dependant filename.
        """
        # Check that the simulation type str is 1 of 2 accepted values
        if sim_type != "skf" and sim_type != "bsk":
            raise ValueError("Invalid input argument 'sim_type'. Only exepted values are 'skf' or 'bsk'.")

        return f"{timestamp_str}_{sim_type}"