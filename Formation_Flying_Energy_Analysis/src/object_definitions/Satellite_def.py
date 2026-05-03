import logging
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray

from Basilisk.utilities import orbitalMotion




class Satellite:
    def __init__(
            self,
            name: str,
            m_s: float, # [kg] Satellite mass
            C_D: float, # Drag coefficient
            A_D: float, # [m^2] Cross-section area perpendicular to the velocity
            C_R: float, # Radiation pressure coefficient (0 reflecting, 1 absorbing)
            A_srp: float, # [m^2] Cross-section area perpendicular to the Sun-vector 
            I_B: list[float], # [kg m^2] Inertia of hub about point Bc in B frame components
            r_BP_B: list[int], # Unit vector pointing towards the satellite face with the largest solar panel area expressed in B
            r_BA_B: list[int], # Unit vector pointing towards the satellite face with the communication antennas expressed in B
            deployment_vel: NDArray[np.float64], # [m/s] satellite deployment velocity from shared deployer
            init_att: list[list[float]], # Orientation of Body relative to Inertial expressed using MRP
            init_angvel: list[list[float]] # Angular velocity og Body relative to Inertial expressed in Body
        ) -> None:
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
        self.I_B: list[float] = I_B
        self.r_BP_B: list[int] = r_BP_B
        self.r_BA_B: list[int] = r_BA_B
        self.deployment_vel: NDArray[np.float64] = deployment_vel
        self.init_att: list[list[float]] = init_att
        self.init_angvel: list[list[float]] = init_angvel