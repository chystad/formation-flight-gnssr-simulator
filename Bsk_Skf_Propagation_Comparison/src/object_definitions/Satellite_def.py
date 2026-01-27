import logging
import numpy as np
from typing import Optional
from numpy.typing import NDArray

from object_definitions.SimData_def import SimObjData


class Satellite:
    def __init__(
            self,
            name: str,
            m_s: float, # [kg] Satellite mass
            C_D: float, # Drag coefficient
            A_D: float, # [m^2] Cross-section area perpendicular to the velocity
            C_R: float, # Radiation pressure coefficient (0 reflecting, 1 absorbing)
            A_srp: float, # [m^2] Cross-section area perpendicular to the Sun-vector 
            init_pos: Optional[NDArray[np.float64]],
            init_vel: Optional[NDArray[np.float64]]
        ):
        """
        ==========================================================================================================
        NOTE: Satellite attribute types will always be inherited from inputs. Necessay parsing and type 
          conversions will be performed by the 'Config' class functions
        
        ATTRIBUTES:
            name:               Satellite name 'str'
        ==========================================================================================================
        """
            
        # Assign attribute values 
        self.name: str = name
        self.m_s: float = m_s
        self.C_D: float = C_D
        self.A_D: float = A_D
        self.C_R: float= C_R
        self.A_srp: float = A_srp
        self.init_pos: Optional[NDArray[np.float64]] = init_pos
        self.init_vel: Optional[NDArray[np.float64]] = init_vel


    def extract_initial_states_and_update(self, sim_object_data: SimObjData) -> None:

        logging.debug(f"Extracting initial states for {sim_object_data.satellite_name}")

        # Normalize time array to always be 1D: shape (n,)
        time = np.asarray(sim_object_data.time).ravel()

        # Verify that the simulation data is connected to the Satellite object
        if not sim_object_data.satellite_name == self.name:
            raise ValueError(f"Mismatch between sim_object_data satellite name ({sim_object_data.satellite_name}) and self.name ({self.name})")
        
        # Verify that the first states are evaluated at t = 0.0 second
        if not time[0] == 0:
            raise ValueError(f"The first element in sim_object_data.time is nonzero: {sim_object_data.time[0]}")
        
        # Extract initial states
        init_pos = sim_object_data.pos[:,0]
        init_vel = sim_object_data.vel[:,0]

        # Update attributes
        self.init_pos = init_pos
        self.init_vel = init_vel

