import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.TLE_def import TLE
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import OUTPUT_DATA_SAVE_DIR


@dataclass_json
@dataclass
class BasiliskSettings:
    deltaT: float
    integrator: str
    sphericalHarmonicsDegree: int
    useSphericalHarmonics: bool
    useMsisDrag: bool
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
            startTime (str):                TODO: Replace with functionality that automatically uses the Epoch from the oldest TLE file as startDate
            simulationDuration (float):     Simulation duration in hours
            use_old_skf_data (bool):        If true: skip the Skyfield simulation and instead use the data from a previous run.
                                                Used when you want to compare the same SGP4 baseline against multiple Basilisk runs.
            old_skf_data_timestamp (str):   Timestamp str for the old Skyfield data
            show_plots (bool):              Show plots if true
            save_plots (bool):              Save plots if true
            bypass_sim_to_plot (bool):      If true: Skip the simulation to plot old data
            data_timestamp_to_plot (str):   Timestamp str for the data to plot. Only plot this if 'bypass_sim_to_plot' == true
            leader_tle_series_path (str):   Path to the .txt file containing a series of TLEs from oldest to newest for the leader satellite
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
        b_cfg = self.read(d_cfg['BASILISK']['config_path']) # basilisk config
        s_cfg = self.read(d_cfg['SKYFIELD']['config_path']) # skyfield condig
        
        ##################################################
        # Fetch parameters from the various config files #
        ##################################################
        # Fetch from default.yaml
        startTime_str =             d_cfg['SIMULATION']['startTime'] # str
        simulationDuration =        d_cfg['SIMULATION']['simulationDuration'] # float  
        use_old_skf_data =          d_cfg['SIMULATION']['use_old_skf_data'] # bool
        old_skf_data_timestamp =    d_cfg['SIMULATION']['old_skf_data_timestamp'] # str
        show_plots =                d_cfg['PLOTTING']['show_plots'] # bool
        save_plots =                d_cfg['PLOTTING']['save_plots'] # bool
        bypass_sim_to_plot =        d_cfg['PLOTTING']['bypass_sim_to_plot'] # str
        data_timestamp_to_plot =    d_cfg['PLOTTING']['data_timestamp_to_plot'] # str
        leader_tle_series_path =    d_cfg['SATELLITES']['leader_tle_series_path'] # str
        inplane_separation_ang =    d_cfg['SATELLITES']['inplane_separation_ang'] # float
        num_satellites =            d_cfg['SATELLITES']['num_satellites'] # int
        all_sat_params =            d_cfg['SATELLITES']['SATELLITE_PARAMETERS'] # ??

        # Fetch from basilisk.yaml
        bsk_deltaT =                    b_cfg['BASILISK_SIMULATION']['deltaT']
        integrator =                    b_cfg['BASILISK_SIMULATION']['integrator']
        sphericalHarmonicsDegree =      b_cfg['BASILISK_SIMULATION']['sphericalHarmonicsDegree']
        useSphericalHarmonics =         b_cfg['BASILISK_SIMULATION']['useSphericalHarmonics']
        useMsisDrag =                   b_cfg['BASILISK_SIMULATION']['useMsisDrag']
        useExponentialDensityDrag =     b_cfg['BASILISK_SIMULATION']['useExponentialDensityDrag']
        useSRP =                        b_cfg['BASILISK_SIMULATION']['useSRP']
        useSun3rdBody =                 b_cfg['BASILISK_SIMULATION']['useSun3rdBody']
        useMoon3rdBody =                b_cfg['BASILISK_SIMULATION']['useMoon3rdBody']
        override_skf_initial_state =    b_cfg['BASILISK_SIMULATION']['override_skf_initial_state']

        # Fetch from skyfield.yaml
        skf_deltaT = s_cfg['SKYFIELD_SIMULATION']['deltaT']

        # Create Satellite intstances
        satellites = self.generate_satellite_instances_from_config(
            all_sat_params, 
            leader_tle_series_path, 
            inplane_separation_ang, 
            num_satellites
        )
        
        ##############################
        # Assign instance attributes #
        ##############################
        self.startTime: str = startTime_str
        self.simulationDuration: float = simulationDuration
        self.use_old_skf_data: bool = use_old_skf_data
        self.old_skf_data_timestamp: str = old_skf_data_timestamp
        self.show_plots: bool = show_plots
        self.save_plots: bool = save_plots
        self.bypass_sim_to_plot: bool = bypass_sim_to_plot
        self.data_timestamp_to_plot: str = data_timestamp_to_plot
        self.leader_tle_series_path: str = leader_tle_series_path
        self.inplane_separation_ang: float = inplane_separation_ang
        self.num_satellites: int = num_satellites
        self.timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.satellites: list[Satellite] = satellites

        # Assign BasiliskSettings instance to b_set attribute
        self.b_set = BasiliskSettings(
            bsk_deltaT,
            integrator,
            sphericalHarmonicsDegree,
            useSphericalHarmonics,
            useMsisDrag,
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

        # Save a combined config under OUTPUT_DATA_SAVE_DIR if the simulation is not bypassed
        if not bypass_sim_to_plot:
            self.save_combined_config(config_file_path, d_cfg)
        
        else:
            logging.debug(f"Bypassing simulation to plot data with the timestamp: {data_timestamp_to_plot}")

            # Check if there are any datafiles with the timestamp 'data_timestamp_to_plot'
            # Collect all matching .h5 files and return their names as strings
            matching_files = [
                file.name for file in OUTPUT_DATA_SAVE_DIR.glob(f"{data_timestamp_to_plot}*.h5")
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
        OUTPUT_DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Build output path using timestamp_str from this Config instance
        out_path = OUTPUT_DATA_SAVE_DIR / f"{self.timestamp_str}_cfg.yaml"

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


    def generate_satellite_instances_from_config(self, 
                                                 all_sat_params: Any,
                                                 leader_tle_series_path: str,
                                                 inplane_separation_ang: float,
                                                 num_satellites: int) -> list[Satellite]:
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

        leader_path = Path(leader_tle_series_path)
        sat_it: int = 0
        satellites: list[Satellite] = []
        tle_processor = TLE(
            leader_tle_series_path,
            inplane_separation_ang,
            num_satellites
        )
        for sat_role, sat_param in all_sat_params.items():
            if isinstance(sat_role, str):
                # Extract/Generate a satellite name
                if sat_role == "leader":
                    sat_name = tle_processor.extract_satellite_name_from_TLE(leader_path)

                elif (sat_role.startswith("follower-")) and (sat_it > 0):
                    sat_name = f"follower-{sat_it}"

                else: 
                    raise ValueError(f"Received satellite role: ({sat_role}), but expected: (leader) or (follower-X)")
            else:
                raise ValueError("Satellite parameter keys are not strings")

            # Create Satellite instance form current satellite name and parameters
            satellite = Satellite(
                sat_name,
                m_s = sat_param['m_s'],
                C_D = sat_param['C_D'],
                A_D = sat_param['A_D'],
                C_R = sat_param['C_R'],
                A_srp = sat_param['A_srp'],
                init_pos = None, # This field will be populated by data from skyfield later
                init_vel = None  # This field will be populated by data from skyfield later
            )

            logging.debug(f"Appending {sat_name} to 'satellites'")
            satellites.append(satellite)

            # Check exit condition
            sat_it += 1
            if sat_it >= num_satellites:
                break

        return satellites