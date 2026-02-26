import yaml
import logging
import numpy as np
from typing import Any, Optional
from pathlib import Path
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# from object_definitions.TLE_def import TLE
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import OUTPUT_DATA_SAVE_DIR

from Basilisk.utilities import (orbitalMotion, macros)



class Config:
    def __init__(self, config_file_path: str) -> None:
        """
        =========================================================================================================
        [WORK IN PROGRESS]
        Initialize Config instance with attributes from the config file

        INPUTS:
           config_file_path                    
        
        ATTRIBUTES:
            startTime (str):                TODO: Replace with functionality that automatically uses the Epoch from the oldest TLE file as startDate
            simulationDuration (float):     Simulation duration in hours
            use_old_skf_data (bool):        If true: skip the Skyfield simulation and instead use the data from a previous run.
                                                Used when you want to compare the same SGP4 baseline against multiple Basilisk runs.
            old_skf_data_timestamp (str):   Timestamp str for the old Skyfield data
            inplane_separation_ang (float): The in-plane orbital separation angle in degrees
            num_satellites (int):           The total number of satellites included in the simulation (leader + #follower(s))
            all_sat_params
            timestamp_str (str):            Used in the naming of data files. str holding the real-world simulation start time.
            satellites (list[Satellite]):   One Satellite instance for each satellite described in the default config.
            b_set (BasiliskSettings):       BasiliskSettings instance describing the Basilisk simulation settings
            s_set (SkyfieldSettings):       SkyfieldSettings instance describing the Skyfield simulation settings     
        =========================================================================================================
        """
        ####################
        # Load cofig files #
        ####################
        d_cfg = self.read(config_file_path)                 # default config
        
        ##################################################
        # Fetch global simulation parameters config file #
        ##################################################
        startTime_str =             str(    d_cfg['SIMULATION']['startTime'])
        simulationDuration =        float(  d_cfg['SIMULATION']['simulationDuration'])  
        deltaT =                    float(  d_cfg['SIMULATION']['deltaT'])  
        integrator =                str(    d_cfg['SIMULATION']['integrator'])
        num_satellites =            int(    d_cfg['SIMULATION']['num_satellites'])
        sat_init_source =           str(    d_cfg['SIMULATION']['sat_init_source'])
        all_sat_params =                    d_cfg['SATELLITES'] # dict[str, dict[str, Any]]

        # Reaction wheel parameters (same for all satellites)
        Omega =                      float(   d_cfg['RW_PARAMETERS']['Omega'])

        # Thruster parameters (same for all satellites)
        temp =                      bool(   d_cfg['THRUSTER_PARAMETERS']['temp'])

        # Magnetorquer parameters (same for all satellites)
        temp =                      bool(   d_cfg['MTQ_PARAMETERS']['temp'])
        
        # Disturbance torque settings
        temp =                      bool(   d_cfg['DISTURBANCE_TORQUE']['temp'])
        
        # Disturbance force settings
        sphericalHarmonicsDegree =      int(    d_cfg['DISTURBANCE_FORCE']['sphericalHarmonicsDegree'])
        useSphericalHarmonics =         bool(   d_cfg['DISTURBANCE_FORCE']['useSphericalHarmonics'])
        useMsisDrag =                   bool(   d_cfg['DISTURBANCE_FORCE']['useMsisDrag'])
        useExponentialDensityDrag =     bool(   d_cfg['DISTURBANCE_FORCE']['useExponentialDensityDrag'])
        useSRP =                        bool(   d_cfg['DISTURBANCE_FORCE']['useSRP'])
        useSun3rdBody =                 bool(   d_cfg['DISTURBANCE_FORCE']['useSun3rdBody'])
        useMoon3rdBody =                bool(   d_cfg['DISTURBANCE_FORCE']['useMoon3rdBody'])

        # Create Satellite intstances
        satellites = self.generate_satellite_instances_from_config(
            all_sat_params, 
            num_satellites,
            sat_init_source
        )
        
        ##############################
        # Assign instance attributes #
        ##############################
        # Simulation
        self.timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.startTime: str = startTime_str
        self.simulationDuration: float = simulationDuration
        self.deltaT: float = deltaT
        self.integrator: str = integrator
        self.num_satellites: int = num_satellites
        self.sat_init_source: str = sat_init_source

        # Satellites
        self.satellites: list[Satellite] = satellites

        # TODO: RW parameters
        self.Omega: float = Omega # [RPM]

        # TODO: Thruster parameters
        self.temp: bool = temp

        # TODO: MTQ parameters
        self.temo: bool = temp

        # TODO: Disturbance torque
        self.temp: bool = temp

        # Disturbance force
        self.sphericalHarmonicsDegree: int = sphericalHarmonicsDegree
        self.useSphericalHarmonics: bool = useSphericalHarmonics
        self.useMsisDrag: bool = useMsisDrag
        self.useExponentialDensityDrag: bool = useExponentialDensityDrag
        self.useSRP: bool = useSRP
        self.useSun3rdBody: bool = useSun3rdBody
        self.useMoon3rdBody: bool = useMoon3rdBody

        # Save a combined config under OUTPUT_DATA_SAVE_DIR
        self.save_config(config_file_path)


    def read(self, config_file_path: str):
        # Get full path to the target config file
        config_path = Path(config_file_path)
        
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.full_load(f)

        return config
    

    def save_config(self, config_file_path: str) -> None:
        """
        Save the config file as:
            <repo_root>/Bsk_Skf_Propagation_Comparison/output_data/sim_data/<timestamp_str>_cfg.yaml
        """
        # Ensure output directory exists
        OUTPUT_DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Build output path using timestamp_str from this Config instance
        out_path = OUTPUT_DATA_SAVE_DIR / f"{self.timestamp_str}_cfg.yaml"

        # Config paths
        default_cfg_path = Path(config_file_path)
        
        # Read raw text from each config file in the specified order
        with open(default_cfg_path, "r") as f_default:
            default_text = f_default.read()

        # Combine texts: default, then skyfield, then basilisk
        # Add blank lines between sections for readability
        out_text = (
            default_text.rstrip() + "\n")

        # Write combined config snapshot
        with open(out_path, "w") as f_out:
            f_out.write(out_text)

        logging.info(f"[CFG] Config snapshot written to: {out_path}")
    

    def generate_satellite_instances_from_config(self, 
                                                 all_sat_params: dict[str, dict[str, float]],
                                                 num_satellites: int,
                                                 sat_init_source) -> list[Satellite]:
        """
        Generates a list of Satellite objects that acts like a common reference for both the Skyfield and Basilisk simulations.
        The number of satellites are defined by 'num_satellites' in default.yaml, 
        while the individual physical satellite parameters comes from the 'shared_input_data' folder

        Returns:
            (list[Satellite]): A list of num_satellites Satellite instances
        """

        # Check if parameters have been defined for enough satellites
        if len(all_sat_params) < num_satellites:
            raise ValueError(f"There has only been defined parameters for ({len(all_sat_params)}) satellites, while default.yaml specifies ({num_satellites}) satellites total.")

        # Assign satellite names
        sat_it: int = 0
        satellites: list[Satellite] = []
        for sat_role, sat_param in all_sat_params.items():
            if isinstance(sat_role, str):
                # Extract/Generate a satellite name
                if sat_role == "leader":
                    sat_name = "Leader"

                elif (sat_role.startswith("follower-")) and (sat_it > 0):
                    sat_name = f"Follower-{sat_it}"

                else: 
                    raise ValueError(f"Received satellite role: ({sat_role}), but expected: (leader) or (follower-X)")
            else:
                raise ValueError("Satellite parameter keys are not strings")
            
            # Initialize some variables
            init_OEs = sat_param['init_OEs']
            init_state_vec = sat_param['init_state_vec']
            sat_init_OEs: Optional[orbitalMotion.ClassicElements] = None
            sat_init_pos: Optional[NDArray[np.float64]] = None
            sat_init_vel: Optional[NDArray[np.float64]] = None

            # If sat_init_source == "oe", check if orbital elements are provided by the config and extract to sat_init_OEs
            required_elements = ["a", "e", "i", "Omega", "omega", "f"]
            oe = orbitalMotion.ClassicElements()
            if sat_init_source == "oe":
                if isinstance(init_OEs, dict):
                    for key in required_elements:
                        if not key in init_OEs:
                            raise ValueError(f"Orbital element '{key}' for satellite '{sat_name}' is not defined")
                        
                        elif not isinstance(init_OEs[key], float): # TODO: if downstream functions work with int, expand accepted types here. 
                            raise ValueError(f"Orbital element '{key}' for satellite '{sat_name}' is defined, but contains a type {type(init_OEs[key])} value")
                        
                    oe.a = init_OEs["a"] * 1000                 # [m]
                    oe.e = init_OEs["e"]                        # [-]
                    oe.i = init_OEs["i"] * macros.D2R           # [Rad]
                    oe.Omega = init_OEs["omega"] * macros.D2R   # [Rad]
                    oe.omega = init_OEs["omega"] * macros.D2R   # [Rad]
                    oe.f = init_OEs["f"] * macros.D2R           # [Rad]
                    
                    sat_init_OEs = oe
                else:
                    raise ValueError(f"'init_OEs' parameter for satellite '{sat_name}' is not of type dict")
                
            # If sat_init_source == "vec", check if state vector is provided by the config and extract to sat_init_state_vec
            elif sat_init_source == "vec":
                if isinstance(init_state_vec, list):
                    if len(init_state_vec) != 6:
                        raise ValueError(f"Initial state vector for satellite '{sat_name}' does not contain 6 elements")
                    
                    for i, elem in enumerate(init_state_vec):
                        if not (isinstance(elem, int) or isinstance(elem, float)):
                            raise ValueError(f"'init_state_vec' for satellite {sat_name} does not contain elements of the correct type. "
                                             f"Element nr. {i} was of type {type(elem)}, expected int or float")

                    np_state_arr = np.array(init_state_vec, dtype=np.float64)
                    sat_init_pos = np_state_arr[:3] # ECI Position
                    sat_init_vel = np_state_arr[3:] # ECI Velocity

            else:
                raise ValueError(f"Unrecognized satellite initial condition source '{sat_init_source}'")
                    
            # Create Satellite instance form current satellite name and parameters
            satellite = Satellite(
                sat_name,
                m_s = sat_param['m_s'],
                C_D = sat_param['C_D'],
                A_D = sat_param['A_D'],
                C_R = sat_param['C_R'],
                A_srp = sat_param['A_srp'],
                init_OEs = sat_init_OEs,
                init_pos = sat_init_pos,
                init_vel = sat_init_vel
            )

            logging.debug(f"[CFG] Appending {sat_name} to 'satellites'")
            satellites.append(satellite)

            # Check exit condition
            sat_it += 1
            if sat_it >= num_satellites:
                break

        return satellites
    

    # def save_combined_config(self, config_file_path: str, loaded_default_cfg) -> None:
    #     """
    #     [DEPRECIATED] Combine default.yaml, skyfield.yaml, and basilisk.yaml into one file and save as:
    #         <repo_root>/Bsk_Skf_Propagation_Comparison/output_data/sim_data/<timestamp_str>_cfg.yaml

    #     Order: default, then skyfield, then basilisk.
    #     """
    #     # Ensure output directory exists
    #     OUTPUT_DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    #     # Build output path using timestamp_str from this Config instance
    #     out_path = OUTPUT_DATA_SAVE_DIR / f"{self.timestamp_str}_cfg.yaml"

    #     # Config paths
    #     default_cfg_path = Path(config_file_path)
    #     skyfield_cfg_path = Path(loaded_default_cfg['SKYFIELD']['config_path'])
    #     basilisk_cfg_path = Path(loaded_default_cfg['BASILISK']['config_path'])
        
    #     # Read raw text from each config file in the specified order
    #     with open(default_cfg_path, "r") as f_default:
    #         default_text = f_default.read()

    #     with open(skyfield_cfg_path, "r") as f_skf:
    #         skyfield_text = f_skf.read()

    #     with open(basilisk_cfg_path, "r") as f_bsk:
    #         basilisk_text = f_bsk.read()

    #     # Combine texts: default, then skyfield, then basilisk
    #     # Add blank lines between sections for readability
    #     combined_text = (
    #         default_text.rstrip() + "\n\n"
    #         + skyfield_text.rstrip() + "\n\n"
    #         + basilisk_text.rstrip() + "\n"
    #     )

    #     # Write combined config snapshot
    #     with open(out_path, "w") as f_out:
    #         f_out.write(combined_text)

    #     logging.info(f"[CFG] Combined config written to: {out_path}")