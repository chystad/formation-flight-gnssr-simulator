import yaml
import logging
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.Satellite_def import Satellite
from object_definitions.TwoLineElement_def import TLE
from object_definitions.SimData_def import DATA_SAVE_FOLDER_PATH

COMBINED_CFG_SAVE_FOLDER = Path('Bsk_Skf_Propagation_Comparison/output_data/sim_data')

"""
=========================================================================================================
Overview of classes and their methods with high-level functionality

Config
    __init__:       Initializes the Config instance with global config parameters that 
                      apply to all simulations
    read:           Reads the config file using yaml.full_load

BasiliskConfig
    __init__:       Initializes the BasiliskConfig instance with config parameters that 
                      only apply to th basilisk simualtion framework

SkyfiledConfig:
    __init__:
=========================================================================================================
"""


@dataclass_json
@dataclass
class BasiliskSettings:
    deltaT: float
    integrator: str
    sphericalHarmonicsDegree: int
    useSphericalHarmonics: bool
    useExponentialDensityDrag: bool
    useSRP: bool
    useSun3rdBody: bool
    useMoon3rdBody: bool
    override_skf_initial_state: bool

@dataclass_json
@dataclass
class SkyfieldSettigns:
    deltaT: float


class Config:
    def __init__(self, config_file_path: str) -> None:
        """
        =========================================================================================================
        [WORK IN PROGRESS]
        Initialize Config instance with attributes from the config file

        INPUTS:
           config_file_path                    
        
        ATTRIBUTES:
            startTime (str):
            simulationDuration (float):
            tle_export_path (str):
            use_old_skf_data (bool):        If true: skip the Skyfield simulation and instead use the data from a previous run.
                                                Used when you want to compare the same SGP4 baseline against multiple Basilisk runs.
            old_skf_data_timestamp (str):   Timestamp str for the old Skyfield data
            show_plots (bool):              Show plots if true
            save_plots (bool):              Save plots if true
            bypass_sim_to_plot (bool):      If true: Skip the simulation to plot old data
            data_timestamp_to_plot (str):   Timestamp str for the data to plot. Only plot this if 'bypass_sim_to_plot' == true
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
        b_cfg = self.read(d_cfg['BASILISK']['config_path']) # basilisk config
        s_cfg = self.read(d_cfg['SKYFIELD']['config_path']) # skyfield condig
        
        ##################################################
        # Fetch parameters from the various config files #
        ##################################################
        # Fetch from default.yaml
        startTime_str = d_cfg['SIMULATION']['startTime'] # str
        simulationDuration = d_cfg['SIMULATION']['simulationDuration'] # float  
        tle_export_path = d_cfg['SIMULATION']['tle_export_path'] # str
        use_old_skf_data = d_cfg['SIMULATION']['use_old_skf_data'] # bool
        old_skf_data_timestamp = d_cfg['SIMULATION']['old_skf_data_timestamp'] # str
        show_plots = d_cfg['PLOTTING']['show_plots'] # bool
        save_plots = d_cfg['PLOTTING']['save_plots'] # bool
        bypass_sim_to_plot =  d_cfg['PLOTTING']['bypass_sim_to_plot'] # str
        data_timestamp_to_plot = d_cfg['PLOTTING']['data_timestamp_to_plot'] # str

        # Fetch from basilisk.yaml
        bsk_deltaT = b_cfg['BASILISK_SIMULATION']['deltaT']
        integrator = b_cfg['BASILISK_SIMULATION']['integrator']
        sphericalHarmonicsDegree = b_cfg['BASILISK_SIMULATION']['sphericalHarmonicsDegree']
        useSphericalHarmonics = b_cfg['BASILISK_SIMULATION']['useSphericalHarmonics']
        useExponentialDensityDrag = b_cfg['BASILISK_SIMULATION']['useExponentialDensityDrag']
        useSRP = b_cfg['BASILISK_SIMULATION']['useSRP']
        useSun3rdBody = b_cfg['BASILISK_SIMULATION']['useSun3rdBody']
        useMoon3rdBody = b_cfg['BASILISK_SIMULATION']['useMoon3rdBody']
        override_skf_initial_state = b_cfg['BASILISK_SIMULATION']['override_skf_initial_state']

        # Fetch from skyfield.yaml
        skf_deltaT = s_cfg['SKYFIELD_SIMULATION']['deltaT']

        # Create Satellite intstances
        satellites = self.generate_satellite_instances_from_config(d_cfg)
        
        ##############################
        # Assign instance attributes #
        ##############################
        self.startTime = startTime_str
        self.simulationDuration = simulationDuration
        self.tle_export_path = tle_export_path
        self.use_old_skf_data = use_old_skf_data
        self.old_skf_data_timestamp = old_skf_data_timestamp
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.bypass_sim_to_plot = bypass_sim_to_plot
        self.data_timestamp_to_plot = data_timestamp_to_plot
        self.timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.satellites = satellites

        # Assign BasiliskSettings instance to b_set attribute
        self.b_set = BasiliskSettings(
            bsk_deltaT,
            integrator,
            sphericalHarmonicsDegree,
            useSphericalHarmonics,
            useExponentialDensityDrag,
            useSRP,
            useSun3rdBody,
            useMoon3rdBody,
            override_skf_initial_state
        )

        # Assign SkyfieldSettings instance to s_set attribute
        self.s_set = SkyfieldSettigns(
            skf_deltaT
        )

        # Save a combined config under COMBINED_CFG_SAVE_FOLDER if the simulation is not bypassed
        if not bypass_sim_to_plot:
            self.save_combined_config(config_file_path, d_cfg)
        
        else:
            logging.debug(f"Bypassing simulation to plot data with the timestamp: {data_timestamp_to_plot}")

            # Check if there are any datafiles with the timestamp 'data_timestamp_to_plot'
            # Collect all matching .h5 files and return their names as strings
            matching_files = [
                file.name for file in DATA_SAVE_FOLDER_PATH.glob(f"{data_timestamp_to_plot}*.h5")
            ]
            if matching_files == []:
                raise FileNotFoundError(f"Datafile with timestamp: '{data_timestamp_to_plot}' was not found.")


    def read(self, config_file_path: str):
        # Get full path to the target config file
        config_path = Path(config_file_path)
        
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.full_load(f)

        return config



    def save_combined_config(self, config_file_path: str, loaded_default_cfg) -> None:
        """
        Combine default.yaml, skyfield.yaml, and basilisk.yaml into one file and save as:
            <repo_root>/Bsk_Skf_Propagation_Comparison/output_data/sim_data/<timestamp_str>_cfg.yaml

        Order: default, then skyfield, then basilisk.
        """
        # Ensure output directory exists
        COMBINED_CFG_SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

        # Build output path using timestamp_str from this Config instance
        out_path = COMBINED_CFG_SAVE_FOLDER / f"{self.timestamp_str}_cfg.yaml"

        # Config paths
        default_cfg_path = Path(config_file_path)
        skyfield_cfg_path = Path(loaded_default_cfg['SKYFIELD']['config_path'])
        basilisk_cfg_path = Path(loaded_default_cfg['BASILISK']['config_path'])
        
        # Read raw text from each config file in the specified order
        with open(default_cfg_path, "r") as f_default:
            default_text = f_default.read()

        with open(skyfield_cfg_path, "r") as f_skf:
            skyfield_text = f_skf.read()

        with open(basilisk_cfg_path, "r") as f_bsk:
            basilisk_text = f_bsk.read()

        # Combine texts: default, then skyfield, then basilisk
        # Add blank lines between sections for readability
        combined_text = (
            default_text.rstrip() + "\n\n"
            + skyfield_text.rstrip() + "\n\n"
            + basilisk_text.rstrip() + "\n"
        )

        # Write combined config snapshot
        with open(out_path, "w") as f_out:
            f_out.write(combined_text)

        logging.info(f"Combined config written to: {out_path}")
    
        
    
    @staticmethod
    def generate_satellite_instances_from_config(loaded_default_cfg) -> list[Satellite]:
        """
        TODO: Write docstring...

        TODO: Add functionality to handle the case where initial pos/vel is not provided in default.yaml:
                -> Okay if 'use_custom_initial_state' == false (from basilisk.yaml)
                    -> Assign None for satellite atributes related to the initial custom pos/vel
                -> Error otherwise
        """

        def _parse_inital_states_from_config(initial_pos: str, initial_vel: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            # [DEPRECIATED AS OF 23.10]
            """
            INPUT:
                initial_pos             3 element initial satellite position list. Elements can be str, int or float
                initial_vel             3 element initial satellite velocity list. Elements can be str, int or float

            OUTPUT:
                parsed_ndarray_pos      3 element np.array position vector 
                parsed_ndarray_vel      3 element np.array velocity vector 
            """
            

            if len(initial_pos) != 3 or len(initial_vel) != 3:
                raise ValueError("Wrong number of elements in 'initial_pos' or 'initial_vel' from default.yaml")
            
            parsed_pos = [_parse_element(x) for x in initial_pos]
            parsed_vel = [_parse_element(v) for v in initial_vel]

            parsed_ndarray_pos = np.array(parsed_pos, dtype=float)
            parsed_ndarray_vel = np.array(parsed_vel, dtype=float)
            
            #parsed_initial_state = np.array(parsed_pos + parsed_vel, dtype=float)

            return parsed_ndarray_pos, parsed_ndarray_vel
        

        def _parse_element(val) -> float:
            """
            Parse input value from type str or int to type float that can later be used in np.array([], dtype=float)
            """
            if isinstance(val, str):
                try:
                    return float(val.replace("E", "e"))
                except:
                    raise ValueError(f"Invalid numeric string: {val}")
            elif isinstance(val, (float, int)):
                return float(val)
            else:
                raise TypeError(f"Unsupported type {type(val)} for element {val}")



        # Use raw config for satellite information
        all_sat_info = loaded_default_cfg['SATELLITES']
        
        satellite: Satellite
        satellites: list[Satellite] = []
        for sat, sat_info in all_sat_info.items():
            # Check if all satellite attributes are compatible with the Satellite object
            allowed = {'name', 'tle_line1', 'tle_line2', 'm_s', 'C_D', 'A_D', 'C_R', 'A_srp'} # NOTE: This must be updated of the config-format changes!
            unknown = set(sat_info) - allowed
            if unknown:
                raise ValueError(f"{sat}: unknown keys for {unknown}")
            
            # Extract tle strings
            tle_line1 = sat_info['tle_line1']
            tle_line2 = sat_info['tle_line2']
            
            # Generate TLE instance from tle lines
            tle = TLE(
                tle_line1,
                tle_line2
            )
            
            # Create Satellite instance form current config satellite
            satellite = Satellite(
                sat_info['name'],
                tle_line1,
                tle_line2,
                tle,
                m_s = sat_info['m_s'],
                C_D = sat_info['C_D'],
                A_D = sat_info['A_D'],
                C_R = sat_info['C_R'],
                A_srp = sat_info['A_srp'],
                init_pos = None, # This field will be populated by data from skyfield later
                init_vel = None  # This field will be populated by data from skyfield later
            )
            
            logging.debug(f"Appending {sat_info['name']} to 'satellites'")
            satellites.append(satellite)

        return satellites
        


        
