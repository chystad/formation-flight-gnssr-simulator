import logging
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.SimData_def import SimData, SimObjData, RelObjData






class DataProcessor:
    def __init__(self) -> None:
        pass

    def calculate_pos_diff(self):
        pass

    @staticmethod
    def calculate_eci2rtn_rotmat(chief_pos_eci: NDArray[np.float64], chief_vel_eci: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the ECI-to-RTN rotation matrix for one timestep

        Args:
            chief_pos_eci (NDArray[np.float64]): The chief satellite's position vector (3,1) in ECI frame
            chief_vel_eci (NDArray[np.float64]): The chief satellite's velocity vector (3,1) in ECI frame
            
        Returns:
            R_eci2rtn (NDArray[np.float64]): Rotation matrix for transformations ECI -> RTN. Valid for 1 timestep.
        """

        # Ensure that the position and velocity vectors are of shape (3,1) 
        assert chief_pos_eci.shape[0] == 3 and chief_pos_eci.shape[1] == 1
        assert chief_vel_eci.shape[0] == 3 and chief_vel_eci.shape[1] == 1

        # Intermediate calculation
        r_cross_v = np.linalg.cross(chief_pos_eci.ravel(), chief_vel_eci.ravel())
        
        # Caulate the RTN basis vectors
        Ri = chief_pos_eci.ravel() / np.linalg.norm(chief_pos_eci)
        Ni = r_cross_v / np.linalg.norm(r_cross_v)
        Ti = np.linalg.cross(Ni.ravel(), Ri.ravel())

        # Ensure correct shape
        assert Ri.ndim == 1 and Ri.shape[0] == 3
        assert Ti.ndim == 1 and Ti.shape[0] == 3
        assert Ni.ndim == 1 and Ni.shape[0] == 3

        # Assemble rotation matrix
        R_eci2rtn = np.array([Ri,
                              Ti, 
                              Ni], np.float64)
        
        return R_eci2rtn
    


    def calculate_relative_formation_movement_rtc(
        self,
        sim_data: SimData,
        data_reduction_factor: int = 1,
    ) -> None:
        """
        Calculate the relative position and velocity between the chief and followers
        expressed in RTN, with optional downsampling.

        Args:
            sim_data (SimData): Simulation data to process.
            data_reduction_factor (int): Compute RTN transform every n samples.

        Returns:
            None — results stored in sim_data.rel_data
        """
        logging.debug(
            f"Calculating RTN relative motion (downsample factor = {data_reduction_factor})..."
        )

        if sim_data.rel_data is not None:
            raise ValueError(
                f"sim_data.rel_data has already been initialized. Type: {type(sim_data.rel_data)}"
            )

        # Chief satellite is always index 0
        chief_data = sim_data.sim_data[0]

        n_satellites = len(sim_data.sim_data)
        n_samples = chief_data.pos.shape[1]

        # -------------------------------
        # Create downsample index vector
        # -------------------------------
        idx = np.arange(0, n_samples, data_reduction_factor)
        n_red = len(idx)

        # -------------------------------
        # Chief rel-data (always zero)
        # -------------------------------
        chief_time = chief_data.time[..., idx]
        chief_rel_pos = np.zeros((3, n_red), dtype=np.float64)
        chief_rel_vel = np.zeros((3, n_red), dtype=np.float64)

        rel_chief_data = RelObjData(
            chief_data.satellite_name,
            chief_time,
            chief_rel_pos,
            chief_rel_vel,
        )

        rel_data: list[RelObjData] = [rel_chief_data]

        # -------------------------------
        # Process each follower
        # -------------------------------
        for i in range(1, n_satellites):

            sat_data = sim_data.sim_data[i]

            # Full ECI relative vectors
            rel_pos_eci_full = chief_data.pos - sat_data.pos
            rel_vel_eci_full = chief_data.vel - sat_data.vel

            # Downsample these BEFORE transforming
            rel_pos_eci = rel_pos_eci_full[:, idx]
            rel_vel_eci = rel_vel_eci_full[:, idx]

            chief_pos_ds = chief_data.pos[:, idx]
            chief_vel_ds = chief_data.vel[:, idx]

            # Allocate RTN arrays
            rel_pos_rtn = np.zeros((3, n_red), dtype=np.float64)
            rel_vel_rtn = np.zeros((3, n_red), dtype=np.float64)

            # -------------------------------
            # Transform each downsampled point
            # -------------------------------
            for k in range(n_red):

                cp = chief_pos_ds[:, k:k+1]
                cv = chief_vel_ds[:, k:k+1]

                rel_p = rel_pos_eci[:, k:k+1]
                rel_v = rel_vel_eci[:, k:k+1]

                # Compute ECI→RTN rotation matrix
                R = self.calculate_eci2rtn_rotmat(cp, cv)

                rel_pos_rtn[:, k:k+1] = R @ rel_p
                rel_vel_rtn[:, k:k+1] = R @ rel_v

            # Store follower result
            rel_obj = RelObjData(
                sat_data.satellite_name,
                sat_data.time[..., idx],  # downsampled time
                rel_pos_rtn,
                rel_vel_rtn,
            )

            rel_data.append(rel_obj)

        # Save into SimData
        sim_data.rel_data = rel_data




    # def calculate_relative_formation_movement_rtc(self, sim_data: SimData, data_reduction_factor: int = 1) -> None:
    #     """
    #     Calculate the relative position and velocity between the chief formation satellite and all other satellites expressed in RTC frame.
    #     Add this data to the input's sim_data.rel_data attribute.

    #     Args:
    #         sim_data (SimData): Skyfield or Basilisk simualtion data. 
    #             Its attribute .rel_data must not be initialized with its default value: None

    #     Returns:
    #         None
    #     """
    #     logging.debug("Calculating the relative position between chief and follower and transforming into RTN frame...")   
        
    #     if sim_data.rel_data is not None:
    #         raise ValueError(f"sim_data.rel_data has already been initialized. Type: {type(sim_data.rel_data)}")
        
    #     # The chief satellite will always be the 1st satellite in SimData
    #     chief_data = sim_data.sim_data[0]

    #     # Get the dimensions
    #     n_satellites = len(sim_data.sim_data)
    #     n_samples = chief_data.pos.shape[1]

    #     # Create a releative object data with zeros for relative position and velocity for the chief satellite
    #     chief_time = chief_data.time
    #     chief_rel_pos = np.zeros((3, n_samples), np.float64)
    #     chief_rel_vel = np.zeros((3, n_samples), np.float64)

    #     rel_chief_data = RelObjData(
    #         chief_data.satellite_name,
    #         chief_time,
    #         chief_rel_pos,
    #         chief_rel_vel
    #     )
        
    #     rel_data: list[RelObjData] = [rel_chief_data]
    #     for i in range(1, n_satellites):
    #         sat_data = sim_data.sim_data[i]
    #         sat_name = sat_data.satellite_name

    #         # Position and velocity difference in ECI between chief and current satellite
    #         sat_rel_pos_eci = chief_data.pos - sat_data.pos
    #         sat_rel_vel_eci = chief_data.vel - sat_data.vel

    #         # Allocate relative position and velocity data
    #         rel_pos_rtn = np.zeros((3, n_samples), np.float64)
    #         rel_vel_rtn = np.zeros((3, n_samples), np.float64)

    #         # Rotate the relative position and velocity into RTN frame and store fore every timestep.
    #         for j in range(0, n_samples):

    #             curr_sat_rel_pos_eci = sat_rel_pos_eci[:,j:(j+1)]
    #             curr_sat_rel_vel_eci = sat_rel_vel_eci[:,j:(j+1)]

    #             # Get rotation matrix ECI -> RTN
    #             R_eci2rtn = self.calculate_eci2rtn_rotmat(chief_data.pos[:,j:(j+1)], chief_data.vel[:,j:(j+1)])

    #             # Rotate relative position and velocity vectors for this timestep
    #             curr_sat_rel_pos_rtn = R_eci2rtn @ curr_sat_rel_pos_eci
    #             curr_sat_rel_vel_rtn = R_eci2rtn @ curr_sat_rel_vel_eci

    #             # Insert results into pre-allocated array
    #             rel_pos_rtn[:,j:(j+1)] = curr_sat_rel_pos_rtn
    #             rel_vel_rtn[:,j:(j+1)] = curr_sat_rel_vel_rtn

    #         assert rel_pos_rtn.shape[0] == 3 and rel_pos_rtn.shape[1] == n_samples
    #         assert rel_vel_rtn.shape[0] == 3 and rel_vel_rtn.shape[1] == n_samples            
                
    #         rel_obj_data = RelObjData(
    #             sat_name,
    #             sat_data.time,
    #             rel_pos_rtn,
    #             rel_vel_rtn
    #         )

    #         rel_data.append(rel_obj_data)
                
    #     # Set input SimData attribute 'rel_data'
    #     sim_data.rel_data = rel_data

    @staticmethod
    def ensure_increasing(t, arr3xn):
        """If time is not strictly increasing, sort time and associated (3, n) array accordingly."""
        idx = np.argsort(t)
        if not np.all(idx == np.arange(t.size)):
            t = t[idx]
            arr3xn = arr3xn[:, idx]
        return t, arr3xn

    @staticmethod
    def interp_3xn(src_t, src_3xn, dst_t):
        """Interpolate a (3, n) array defined on src_t onto dst_t (per row)."""
        out = np.empty((3, dst_t.size), dtype=src_3xn.dtype)
        for c in range(3):
            out[c] = np.interp(dst_t, src_t, src_3xn[c])
        return out