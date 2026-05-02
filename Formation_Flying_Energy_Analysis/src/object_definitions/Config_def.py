import yaml
import logging
import numpy as np
from typing import Any, Optional
from pathlib import Path
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime

from Basilisk.utilities import (orbitalMotion, macros)

from object_definitions.Satellite_def import Satellite
from object_definitions.SolarPanel_def import SolarPanel
from object_definitions.GroundStation_def import GroundStation
from object_definitions.MonteCarloConfig_def import MonteCarloConfig
from constants import (OVERRIDE_CONFIG_DIR, OUTPUT_DATA_ROOT_DIR, 
                       SINGLE_OUTPUT_DATA_DIR_NAME, BATCH_OUTPUT_DATA_DIR_NAME)


class Config:
    def __init__(self, base_config_path: str, mc_cfg: MonteCarloConfig, run_idx: int) -> None:
        """
        =========================================================================================================
        [WORK IN PROGRESS]
        Initialize Config instance with attributes from the config file. 
        Perform checks to ensure all parameters are received as expected. 

        INPUTS:
           base_config_path                    
        
        ATTRIBUTES:
            mc_enabled (bool):                  If Monte Carlo simulation is enabled/disabled (TODO: This will impact 
                                                how logging is configured and how data and plots are outputted) 
            run_idx (int):                  The current bsk simulator run number. 0 if mc_enabled == False or if this is the first run in MC
            mc_output_data_dir (Path):      The path to the folder where data from each sim in the Monte Carlo run is saved  
            single_output_data_dir (Path):  The path to the folder where data from a single simulation without Monte Carlo is saved
            startTime (str):                Simulation start time epoch in UTC
            simulationDuration (float):     Simulation duration in hours
            use_old_skf_data (bool):        If true: skip the Skyfield simulation and instead use the data from a previous run.
                                                Used when you want to compare the same SGP4 baseline against multiple Basilisk runs.
            old_skf_data_timestamp (str):   Timestamp str for the old Skyfield data
            inplane_separation_ang (float): The in-plane orbital separation angle in degrees
            num_satellites (int):           The total number of satellites included in the simulation (leader + #follower(s))
            all_sat_params
            timestamp_str (str):            Used in the naming of data files. str holding the real-world simulation start time.
            satellites (list[Satellite]):   One Satellite instance for each satellite described in the default config.
            ground_stations (list[GroundStation]): 
            solar_panels (list[SolarPanel]):
            b_set (BasiliskSettings):       BasiliskSettings instance describing the Basilisk simulation settings
            s_set (SkyfieldSettings):       SkyfieldSettings instance describing the Skyfield simulation settings     
        =========================================================================================================
        """
        ###################
        # Load cofig file #
        ###################
        if (not mc_cfg.mc_enabled) or (run_idx == 0):
            cfg = self.read(base_config_path)
        else:
            cfg = self._resolve_base_override(base_config_path, mc_cfg, run_idx)
        
        ##################################################
        # Fetch global simulation parameters config file #
        ##################################################
        startTime_str =         str(    cfg['SIMULATION']['startTime'])
        simulationDuration =    float(  cfg['SIMULATION']['simulationDuration'])  
        deltaT =                float(  cfg['SIMULATION']['deltaT'])  
        integrator =            str(    cfg['SIMULATION']['integrator'])
        num_satellites =        int(    cfg['SIMULATION']['num_satellites'])
        sat_init_source =       str(    cfg['SIMULATION']['sat_init_source'])
        data_mode =             str(    cfg['SIMULATION']['data_mode'])
        all_sat_params =                cfg['SATELLITES'] # dict[str, dict[str, Any]]
        all_gs_params =                 cfg['GROUND_STATIONS'] # dict[str, dict[str, Any]]     

        # Formation control parameters
        form_enabled =                  cfg['FORMATION_CONTROL']['form_enabled']
        form_type =             str(    cfg['FORMATION_CONTROL']['form_type'])
        form_pos_tolerance =    float(  cfg['FORMATION_CONTROL']['form_pos_tolerance'])
        form_vel_tolerance =    float(  cfg['FORMATION_CONTROL']['form_vel_tolerance'])
        dwell_time =            float(  cfg['FORMATION_CONTROL']['dwell_time'])
        cat_const_separation =  float(  cfg['FORMATION_CONTROL']['constant_along_track']['cat_const_separation'])
        cpo_radial_amp =        float(  cfg['FORMATION_CONTROL']['circular_projected_orbit']['cpo_radial_amp'])
        cpo_cross_track_amp =        float(  cfg['FORMATION_CONTROL']['circular_projected_orbit']['cpo_cross_track_amp'])
        cpo_phase_deg =        float(  cfg['FORMATION_CONTROL']['circular_projected_orbit']['cpo_phase_deg'])        

        # Electrical power system parameters (same for all satellites)
        bat_storage_capacity =  float(  cfg['EPS_PARAMETERS']['bat_storage_capacity'])
        init_bat_charge =       float(  cfg['EPS_PARAMETERS']['init_bat_charge'])
        RW_base_draw =          float(  cfg['EPS_PARAMETERS']['RW_base_draw'])
        OBC_const_draw =        float(  cfg['EPS_PARAMETERS']['OBC_const_draw'])
        all_sp_params =                 cfg['EPS_PARAMETERS']['solar_panels'] # dict[str, dict[str, Any]]  

        # Reaction wheel parameters (same for all satellites)
        RW_model =              str(    cfg['RW_PARAMETERS']['RW_model'])
        spinUVecs =                     cfg['RW_PARAMETERS']['spinUVecs'] # list[list[float]]
        init_rpm =              float(  cfg['RW_PARAMETERS']['init_rpm'])
        max_rpm =               float(  cfg['RW_PARAMETERS']['max_rpm'])
        maxMomentum =           float(  cfg['RW_PARAMETERS']['maxMomentum'])
        maxTorque =             float(  cfg['RW_PARAMETERS']['maxTorque'])
        minTorque =             float(  cfg['RW_PARAMETERS']['minTorque'])
        # I_RW =                  float(  cfg['RW_PARAMETERS']['I_RW'])
        useMinTorque =          bool(   cfg['RW_PARAMETERS']['useMinTorque'])
        useFriction =           bool(   cfg['RW_PARAMETERS']['useFriction'])
        fCoulomb =              float(  cfg['RW_PARAMETERS']['fCoulomb'])
        fStatic =               float(  cfg['RW_PARAMETERS']['fStatic'])
        betaStatic =            float(  cfg['RW_PARAMETERS']['betaStatic'])
        cViscous =              float(  cfg['RW_PARAMETERS']['cViscous'])
        
        # Thruster parameters (same for all satellites)
        thr_pos_B =                     cfg['THRUSTER_PARAMETERS']['thr_pos_B'] # list[float]
        thr_dir_B =                     cfg['THRUSTER_PARAMETERS']['thr_dir_B'] # list[float]
        thr_model_override =    str(    cfg['THRUSTER_PARAMETERS']['thr_model_override'])
        use_min_pulse_time =    bool(   cfg['THRUSTER_PARAMETERS']['use_min_pulse_time'])
        min_pulse_time =        float(  cfg['THRUSTER_PARAMETERS']['min_pulse_time'])
        max_thrust =            float(  cfg['THRUSTER_PARAMETERS']['max_thrust'])
        thrust_blowdown_coeff =         cfg['THRUSTER_PARAMETERS']['thrust_blowdown_coeff'] # list[float]
        steady_isp =            float(  cfg['THRUSTER_PARAMETERS']['steady_isp'])
        isp_blowdown_coeff =            cfg['THRUSTER_PARAMETERS']['isp_blowdown_coeff'] # list[float]
        area_nozzle =           float(  cfg['THRUSTER_PARAMETERS']['area_nozzle'])
        thr_mag_disp =          float(  cfg['THRUSTER_PARAMETERS']['thr_mag_disp'])
        
        # Magnetorquer parameters (same for all satellites)
        temp =                  bool(   cfg['MTQ_PARAMETERS']['temp'])
        
        # Disturbance torque settings
        temp =                  bool(   cfg['DISTURBANCE_TORQUE']['temp'])
        
        # Disturbance force settings
        sphericalHarmonicsDegree =      int(    cfg['DISTURBANCE_FORCE']['sphericalHarmonicsDegree'])
        useSphericalHarmonics =         bool(   cfg['DISTURBANCE_FORCE']['useSphericalHarmonics'])
        useMsisDrag =                   bool(   cfg['DISTURBANCE_FORCE']['useMsisDrag'])
        useExponentialDensityDrag =     bool(   cfg['DISTURBANCE_FORCE']['useExponentialDensityDrag'])
        useSRP =                        bool(   cfg['DISTURBANCE_FORCE']['useSRP'])
        useSun3rdBody =                 bool(   cfg['DISTURBANCE_FORCE']['useSun3rdBody'])
        useMoon3rdBody =                bool(   cfg['DISTURBANCE_FORCE']['useMoon3rdBody'])
        

        ################################################################
        # Perform checks to ensure parameters are received as expected #
        ################################################################      

        # Validate simulation parameters
        self.validate_sim_parameters(
            data_mode
        )

        # TODO: Validate formation control parameters
        
        # Validate RW parameters
        self.validate_rw_parameters(
            RW_model, 
            spinUVecs
        )

        # TODO Validate Thruster parameters
        self.validate_thruster_parameters(
            thr_pos_B,
            thr_dir_B,
            thr_model_override,
            use_min_pulse_time,
            min_pulse_time,
            max_thrust,
            thrust_blowdown_coeff,
            steady_isp,
            isp_blowdown_coeff,
            area_nozzle,
            thr_mag_disp
        )

        # Validate EPS parameters (except solar panel parameters. These are validated in 'generate_solar_panel_instances_from_config')
        self.validate_eps_parameters(
            bat_storage_capacity,
            init_bat_charge,
            RW_base_draw,
            OBC_const_draw
        )
        
        # Validate satellite parameters and create 'Satellite' intstances
        satellites = self.generate_satellite_instances_from_config(
            all_sat_params, 
            num_satellites,
            sat_init_source
        )

        # Validate solar panel parameters and create 'SolarPanel' instances
        solar_panels = self.generate_solar_panel_instances_from_config(
            all_sp_params
        )

        # Validate ground station parameters and create 'GroundStation' instances
        ground_stations = self.generate_ground_station_instances_from_config(
            all_gs_params
        )
        
        ##############################
        # Assign instance attributes #
        ##############################
        # Monte Carlo
        self.mc_enabled: bool = mc_cfg.mc_enabled
        self.mc_dir_name: str = mc_cfg.mc_dir_name
        self.run_idx: int = run_idx
        
        # Simulation
        self.output_data_save_dir: Path = self._build_output_data_save_dir(mc_cfg, run_idx)
        self.timestamp_str: str = mc_cfg.timestamp_str
        self.startTime: str = startTime_str
        self.simulationDuration: float = simulationDuration
        self.deltaT: float = deltaT
        self.integrator: str = integrator
        self.num_satellites: int = num_satellites
        self.sat_init_source: str = sat_init_source
        self.data_mode: str = data_mode

        # Satellites
        self.satellites: list[Satellite] = satellites

        # Ground stattions
        self.ground_stations: list[GroundStation] = ground_stations

        # Formation control parameters
        self.form_enabled: bool = form_enabled
        self.form_type: str = form_type
        self.form_pos_tolerance: float = form_pos_tolerance
        self.form_vel_tolerance: float = form_vel_tolerance
        self.dwell_time: float = dwell_time
        self.cat_const_separation: float = cat_const_separation
        self.cpo_radial_amp: float = cpo_radial_amp
        self.cpo_cross_track_amp: float = cpo_cross_track_amp
        self.cpo_phase_deg: float = cpo_phase_deg

        # EPS parameters
        self.bat_storage_capacity: float = bat_storage_capacity
        self.init_bat_charge: float = init_bat_charge
        self.RW_base_draw: float = RW_base_draw
        self.OBC_const_draw: float = OBC_const_draw
        self.solar_panels: list[SolarPanel] = solar_panels

        # RW parameters
        self.RW_model: str = RW_model
        self.spinUVecs: list[list[float]] = spinUVecs
        self.init_rpm: float = init_rpm 
        self.max_rpm: float = max_rpm
        self.maxMomentum: float = maxMomentum 
        self.maxTorque: float = maxTorque
        self.minTorque: float = minTorque
        # self.I_RW: float = I_RW
        self.useMinTorque: bool = useMinTorque
        self.useFriction: bool = useFriction
        self.fCoulomb: float = fCoulomb
        self.fStatic: float = fStatic
        self.betaStatic: float = betaStatic
        self.cViscous: float = cViscous

        # Thruster parameters
        self.thr_pos_B: list[float] = thr_pos_B
        self.thr_dir_B: list[float] = thr_dir_B
        self.thr_model_override: str = thr_model_override
        self.use_min_pulse_time: bool = use_min_pulse_time
        self.min_pulse_time: float = min_pulse_time
        self.max_thrust: float = max_thrust
        self.thrust_blowdown_coeff: list[float] = thrust_blowdown_coeff
        self.steady_isp: float = steady_isp
        self.isp_blowdown_coeff: list[float] = isp_blowdown_coeff
        self.area_nozzle: float = area_nozzle
        self.thr_mag_disp: float = thr_mag_disp

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
        self.save_config(base_config_path)


    def read(self, config_file_path: str) -> dict:
        # Get full path to the target config file
        config_path = Path(config_file_path)
        
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.full_load(f)

        return config
    

    def _resolve_base_override(self,
                               base_config_path: str,
                               mc_cfg: MonteCarloConfig,
                               run_idx: int
                               ) -> dict:
        """
        Load base.yaml and, if Monte Carlo is enabled, recursively merge the run-specific
        override file into it.

        The override only needs to contain the fields that should change.
        Unspecified fields remain equal to the base config.
        """

        base_cfg = self.read(base_config_path)

        # return base config is single run is enabled or this is the first Bsk run in MC
        if (not mc_cfg.mc_enabled) or (run_idx == 0):
            return base_cfg
        
        run_name = self._get_run_name(mc_cfg.mc_enabled, run_idx)
        override_filename = f"{run_name}.yaml"
        override_path = OVERRIDE_CONFIG_DIR / override_filename

        if not override_path.is_file():
            raise FileNotFoundError(
                f"Monte Carlo override file not found for run {run_idx}: {override_path}"
            )

        override_cfg = self.read(str(override_path))

        resolved_cfg = self._deep_merge_dicts(base_cfg, override_cfg)

        return resolved_cfg
    

    def _deep_merge_dicts(self, base: dict, override: dict) -> dict:
        """
        Recursively merge override into base.

        If both base[key] and override[key] are dicts, merge them recursively.
        Otherwise, override[key] replaces base[key].
        """
        merged = dict(base)

        for key, override_value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(override_value, dict)
            ):
                merged[key] = self._deep_merge_dicts(merged[key], override_value)
            else:
                merged[key] = override_value

        return merged
    

    def _get_run_name(self, mc_enabled: bool, run_idx: int) -> str:
        """
        Get override config name for this Bsk run inside Monte Carlo
        """
        if mc_enabled:
            return f"run_{run_idx:03d}"
        else:
            return "error"
    
    
    def save_config(self, base_config_path: str) -> None:
        """
        Write config files to output directory. 
        
        The file output from this method follows the following logic:

        if Monte Carlo is enabled:            
            if first run, write:
                copy of base to:         'output_data/batch_runs/Monte_Carlo_<timestamp>/base.yaml'
                empty override to:       'output_data/batch_runs/Monte_Carlo_<timestamp>/run_000/run_000_override.yaml'
            else, write:
                copy of run override to: 'output_data/batch_runs/Monte_Carlo_<timestamp>/run_XXX/run_XXX_override.yaml'
        else (single run is enabled):
            write:
                copy of base to:         'output_data/single_runs/<timestamp>_cfg.yaml'
                
        """

        def _write_raw_text_to_path(raw_text: str, out_path: Path) -> None:
            # Add blank lines between sections for readability. Also helps standard formatting
            out_text = (raw_text.rstrip() + "\n")

            # Write the out text to out path
            with open(out_path, "w") as f_out:
                f_out.write(out_text)


        # Ensure output directory exists
        self.output_data_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Build config output file name for single run configuration
        out_single_base_file_name = f"{self.timestamp_str}_cfg.yaml"
        
        # Build base/override input/output names for Monte Carlo configuration
        run_name = self._get_run_name(self.mc_enabled, self.run_idx)
        read_mc_override_file_name = f"{run_name}.yaml"
        out_mc_override_file_name = f"{run_name}_override.yaml" # override output filename
        out_mc_base_file_name = "base.yaml"

        
        # Read config file to copy, and write to appropriate location depending on run idx and if MC is enabled
        if self.mc_enabled:

            override_mc_out_path = self.output_data_save_dir / out_mc_override_file_name

            # Write empty override file in 'Monte_Carlo_<timestamp>/run_000/' 
            # AND write base.yaml to current MC root 'Monte_Carlo_<timestamp>/'
            if self.run_idx == 0:
                override_raw_text = ""
                _write_raw_text_to_path(override_raw_text, override_mc_out_path)

                # Build path to ouput file in MC root 'Monte_Carlo_<timestamp>/'
                base_mc_out_path     = OUTPUT_DATA_ROOT_DIR / BATCH_OUTPUT_DATA_DIR_NAME / self.mc_dir_name / out_mc_base_file_name

                # Read raw text from base config
                with open(Path(base_config_path), "r") as cfg:
                    base_raw_text = cfg.read()

                _write_raw_text_to_path(base_raw_text, base_mc_out_path)

                logging.debug(f"""[CFG] Empty override written to:    '{override_mc_out_path}' and\n"""
                              f"""      copy of base.yaml written to: '{base_mc_out_path}""")
                print(f"""[CFG] Empty override written to:    '{override_mc_out_path}' and\n"""
                              f"""      copy of base.yaml written to: '{base_mc_out_path}""")


            # Copy current override file from 'configs/' to current run output folder
            else:
                override_read_path = OVERRIDE_CONFIG_DIR / read_mc_override_file_name
                with open(override_read_path, "r") as cfg:
                    override_raw_text = cfg.read()

                _write_raw_text_to_path(override_raw_text, override_mc_out_path)

                logging.debug(f"[CFG] Copy of {read_mc_override_file_name} written to: '{override_mc_out_path}'")
                print(f"[CFG] Copy of {read_mc_override_file_name} written to: '{override_mc_out_path}'")


        # Write a copy of base.yaml to 'single_runs/'
        else:
            base_single_out_path = self.output_data_save_dir / out_single_base_file_name
            
            # Read raw text from base config
            with open(Path(base_config_path), "r") as cfg:
                base_raw_text = cfg.read()

            _write_raw_text_to_path(base_raw_text, base_single_out_path)
            
            logging.debug(f"[CFG] Copy of override written to: '{base_single_out_path}'")
            print(f"[CFG] Copy of override written to: '{base_single_out_path}'")        
    

    def _build_output_data_save_dir(self,
                                    mc_cfg: MonteCarloConfig,
                                    run_idx: int
                                    ) -> Path:
        """
        If this basilisk run is a part of a Monte Carlo run, build:
            output_data_save_dir = OUTPUT_DATA_ROOT_DIR / BATCH_OUTPUT_DATA_DIR_NAME / mc_dir_name / mc_run_dir_name
        Else, build:
            output_data_save_dir = OUTPUT_DATA_ROOT_DIR / SINGLE_OUTPUT_DATA_DIR_NAME
        """
        
        # Define the output data save folder name for single runs outside of Monte Carlo
        mc_dir_name = mc_cfg.mc_dir_name
        mc_run_dir_name = f"run_{run_idx:03d}" # TODO: Assign to attribute

        # Ensure output data root folder exists
        OUTPUT_DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)

        # Build the data output dir path depending on if it's a Monte Carlo run or not
        if mc_cfg.mc_enabled:
            output_data_save_dir = OUTPUT_DATA_ROOT_DIR / BATCH_OUTPUT_DATA_DIR_NAME / mc_dir_name / mc_run_dir_name
        else:
            output_data_save_dir = OUTPUT_DATA_ROOT_DIR / SINGLE_OUTPUT_DATA_DIR_NAME

        # Ensure full output data folder exists
        output_data_save_dir.mkdir(parents=True, exist_ok=True)

        return output_data_save_dir


    def generate_satellite_instances_from_config(self, 
                                                 all_sat_params: dict[str, dict[str, float]],
                                                 num_satellites: int,
                                                 sat_init_source: str) -> list[Satellite]:
        """
        Validate all satellite parameters and generate a list of Satellite objects that store all the satellite parameters.
        The number of satellites are defined by 'num_satellites' in base.yaml, 
        and the individual satellite parameters are defined in the fields under 'leader', 'follower-1', 'follower-2', etc.

        Args:
            all_sat_params (dict[str, dict[str, float]]): Loaded dictionaty of all satellites' parameters
            num_satellites (int): The number of satellites to include from 'all_sat_params'
            sat_init_source (str): Chooce which method to use for defining the satellite initial state 
        
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
                        raise ValueError(f"Initial state vector for satellite '{sat_name}' contains {len(init_state_vec)} elements (expected 6)")
                    
                    for i, elem in enumerate(init_state_vec):
                        if not (isinstance(elem, int) or isinstance(elem, float)):
                            raise ValueError(f"'init_state_vec' for satellite {sat_name} does not contain elements of the correct type. "
                                             f"Element nr. {i} was of type {type(elem)} (expected int or float)")

                    np_state_arr = np.array(init_state_vec, dtype=np.float64)
                    sat_init_pos = np_state_arr[:3] # ECI Position
                    sat_init_vel = np_state_arr[3:] # ECI Velocity

                else: 
                    raise ValueError(f"'init_state_vec' parameter for satellite '{sat_name}' is not of type list")

            else:
                raise ValueError(f"Unrecognized satellite initial condition source '{sat_init_source}'")

            
            # Check that I_B from config is correct, and transform it into the type expected by Basilisk
            I_B = sat_param['I_B']
            if isinstance(I_B, list):
                if len(I_B) != 9:
                    raise ValueError(f"Inertia matrix list for satellite '{sat_name}' contains {len(I_B)} elements (expected 9)")

                for i, elem in enumerate(I_B):
                    if not (isinstance(elem, int) or isinstance(elem, float)):
                        raise ValueError(f"'I_B' for satellite {sat_name} does not contain elements of the correct type. "
                                         f"Element nr. {i} was of type {type(elem)} (expected int or float)")
            else:
                raise ValueError(f"'I_B' parameter for satellite '{sat_name}' is of type {type(I_B)} (expected list)")
            

            # Check that r_BP_B from config has elements of type int, has 3 elements and has length == 1
            r_BP_B = sat_param['r_BP_B']
            if isinstance(r_BP_B, list):
                if len(r_BP_B) != 3:
                    raise ValueError(f"Solar panel face vector parameter 'r_BP_B' for satellite {sat_name} contains {len(r_BP_B)} elements (expected 3)")
                for i, comp in enumerate(r_BP_B):
                    if not isinstance(comp, int):
                        try:
                            comp = int(comp)
                            r_BP_B[i] = comp
                        except:
                            raise ValueError(f"Component nr {i} in 'r_BP_B' for satellite {sat_name} failed to convert into type 'int'")
                if not np.isclose(np.linalg.norm(r_BP_B), 1): 
                    raise ValueError(f"The norm of 'r_BP_B' for satellite {sat_name} is not sufficiently close to '1' (norm: {np.linalg.norm(r_BP_B)})")
            else:
                raise ValueError(f"Solar panel face vector parameter 'r_BP_B' for satellite {sat_name} is of type: {type(r_BP_B)} (expected 'list')")


            # Check that r_BA_B from config has elements of type int, has 3 elements and has length == 1
            r_BA_B = sat_param['r_BA_B']
            if isinstance(r_BA_B, list):
                if len(r_BA_B) != 3:
                    raise ValueError(f"Antenna face vector parameter 'r_BA_B' for satellite {sat_name} contains {len(r_BA_B)} elements (expected 3)")
                for i, comp in enumerate(r_BA_B):
                    if not isinstance(comp, int):
                        try:
                            comp = int(comp)
                            r_BA_B[i] = comp
                        except:
                            raise ValueError(f"Component nr {i} in 'r_BA_B' for satellite {sat_name} failed to convert into type 'int'")
                if not np.isclose(np.linalg.norm(r_BA_B), 1): 
                    raise ValueError(f"The norm of 'r_BA_B' for satellite {sat_name} is not sufficiently close to '1' (norm: {np.linalg.norm(r_BA_B)})")
            else:
                raise ValueError(f"Antenna face vector parameter 'r_BA_B' for satellite {sat_name} is of type: {type(r_BA_B)} (expected 'list')")

            # Check that init_att from config is correct
            init_att = sat_param['init_att']
            if isinstance(init_att, list):
                if not len(init_att) == 3:
                    raise ValueError(f"'init_att' for satellite {sat_name} contained {len(init_att)} elements (expected 3)")
                init_att_arr = np.array(init_att)
                init_att_norm = np.linalg.norm(init_att_arr)
                if init_att_norm > 1:
                    logging.warning(f"[WARNING] The eucledian norm of 'init_att' for satellite {sat_name} is over 1 ({init_att_norm}). "
                                    f"The initial attitude is outside the principal MRP set. Consider selecting smaller components")
                for i, elem in enumerate(init_att):
                    if not len(elem) == 1:
                        raise ValueError(f"MRP parameter nr. {i} in 'init_att' for satellite {sat_name} contained {len(elem)} elements (expected 1)")
                    if not (isinstance(elem[0], int) or isinstance(elem[0], float)):
                        raise ValueError(f"'init_att' for satellite {sat_name} does not contain elements of the correct type. "
                                         f"Element nr. {i} was of type {type(elem)} (expected int or float)")
            else:
                raise ValueError(f"'init_att' parameter for satellite {sat_name} is of type {type(init_att)} (expected list[list[float]])")
            

            # Check that init_angvel from config is correct
            init_angvel = sat_param['init_angvel']
            if isinstance(init_angvel, list):
                if not len(init_angvel) == 3:
                    raise ValueError(f"'init_angvel' for satellite {sat_name} contained {len(init_angvel)} elements (expected 3)")
                for i, elem in enumerate(init_angvel):
                    if not len(elem) == 1:
                        raise ValueError(f"MRP parameter nr. {i} in 'init_angvel' for satellite {sat_name} contained {len(elem)} elements (expected 1)")
                    if not (isinstance(elem[0], int) or isinstance(elem[0], float)):
                        raise ValueError(f"'init_angvel' for satellite {sat_name} does not contain elements of the correct type. "
                                         f"Element nr. {i} was of type {type(elem)} (expected int or float)")
            else:
                raise ValueError(f"'init_angvel' parameter for satellite {sat_name} is of type {type(init_angvel)} (expected list[list[float]])")
            
            # Create Satellite instance form current satellite name and parameters
            satellite = Satellite(
                sat_name,
                m_s = sat_param['m_s'],
                C_D = sat_param['C_D'],
                A_D = sat_param['A_D'],
                C_R = sat_param['C_R'],
                A_srp = sat_param['A_srp'],
                I_B = I_B,
                r_BP_B = r_BP_B,
                r_BA_B = r_BA_B,
                init_OEs = sat_init_OEs,
                init_pos = sat_init_pos,
                init_vel = sat_init_vel,
                init_att = init_att,
                init_angvel = init_angvel
            )

            logging.debug(f"[CFG] Appending {sat_name} to 'satellites'")
            satellites.append(satellite)

            # Check exit condition
            sat_it += 1
            if sat_it >= num_satellites:
                break

        return satellites
    

    def generate_ground_station_instances_from_config(self, 
                                                      all_gs_params: dict[str, dict[str, float]]
                                                      ) -> list[GroundStation]:
        """
        For each ground station in GROUND_STATIONS, validate parameters and create a GroundStation instance
        Append all GroundStation instances to a list and return. 
        Raise ValueError if incorrect type/value is detected.

        Args:
            all_gs_params (dict[str, dict[str, float]]): A dictionary loaded from config containing parameters for all ground stations

        Returns:
            list[GroundStation]: A list containing one GroundStation instance for each ground station described in config
        """

        # Loop through all ground stations in all_gs_params
        ground_stations: list[GroundStation] = []
        gs_tags: list[str] = []
        for gs_key, gs_param in all_gs_params.items():

            # Load all ground station parameters for 'gs_key' of arbitrary/un-verified type
            gs_tag =    gs_param['gs_tag']
            lat =       gs_param['lat']
            long =      gs_param['long']
            alt =       gs_param['alt']
            min_elev =  gs_param['min_elev']
            max_range = gs_param['max_range'] # Can be either int or float
            
            # ============ Check 'gs_tag' is of type str and is unique
            if not isinstance(gs_tag, str):
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'gs_tag' of type 'str'. Got: {type(gs_tag)}")
            if gs_tag not in gs_tags:
                gs_tags.append(gs_tag)
            else:
                raise ValueError(f"Ground station '{gs_key}' parameter 'gs_tag' is not unique. "
                                 f"Existing tags prior to this ground station: {gs_tags}")
            
            # ============ Check 'lat' is of type float and has value in range [-90S, 90N]
            if not isinstance(lat, float):
                try:
                    lat = float(lat)
                except:
                    raise ValueError(f"Expected ground station '{gs_key}' parameter 'lat' of type 'float'. Got: {type(lat)}")
            if (lat < -90) or (lat > 90):
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'lat' to be in range [-90, 90]. Got: {lat}")
            
            # ============ Check 'long' is of type float and has value in range [-180W, 180E]
            if not isinstance(long, float):
                try: 
                    long = float(long)
                except:
                    raise ValueError(f"Expected ground station '{gs_key}' parameter 'long' of type 'float'. Got: {type(long)}")
            if (long < -180) or (long > 180):
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'long' to be in range [-180, 180]. Got: {long}")
            
            # ============ Check 'alt' is of type float and has value in range [0, 10'000]
            if not isinstance(alt, float):
                try:
                    alt = float(alt)
                except:
                    raise ValueError(f"Expected ground station '{gs_key}' parameter 'alt' of type 'float'. Got: {type(alt)}")
            if (alt < 0) or (alt > 10000):
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'alt' to be in range [0, 10'000]. Got: {alt}")
            
            # ============ Check 'min_elev' is of type float and has value in range [0, 90]
            if not isinstance(min_elev, float):
                try:
                    min_elev = float(min_elev)
                except:
                    raise ValueError(f"Expected ground station '{gs_key}' parameter 'min_elev' of type 'float'. Got: {type(min_elev)}")
            if (min_elev < 0) or (min_elev > 90):
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'min_elev' to be in range [0, 90]. Got: {min_elev}")
            
            # ============ Check 'max_range' is int if value = -1 and float otherwise with positive value
            if isinstance(max_range, int):
                if not max_range == -1:
                    max_range = float(max_range)
            elif not isinstance(max_range, float):
                try:
                    max_range = float(max_range)
                except:
                    raise ValueError(f"Expected ground station '{gs_key}' parameter 'max_range' of type 'float | int'. Got: {type(max_range)}")
            if isinstance(max_range, float) and max_range <= 0:
                raise ValueError(f"Expected ground station '{gs_key}' parameter 'max_range' to be -1 or greater than zero. Got: {max_range}")

            # Initialize GroundStation instance
            gs = GroundStation(
                gs_tag=gs_tag,
                latitude=lat,
                longitude=long,
                altitude=alt,
                min_elev=min_elev,
                max_range=max_range
            )

            # Append to list
            logging.debug(f"[CFG] Appending {gs_tag} to 'ground_stations'")
            ground_stations.append(gs)
    
        return ground_stations


    def generate_solar_panel_instances_from_config(self, 
                                                   all_sp_params: dict[str, dict[str, float]]
                                                   ) -> list[SolarPanel]:
        """
        Validate all solar panel parameters and generate one SolarPanel instance for each solar panel
        defined in 'EPS_PARAMETERS/solar_panels'

        Args:
            all_sp_params (dict[str, dict[str, float]]): Dictionary loaded from config 
                containing panel parameters for each solar panel
        
        Returns:
            list[SolarPanel]: A list containing one SolarPanel instance 
                for each solar panel defined in 'EPS_PARAMETERS/solar_panels'
        """
        
        # Loop through all solar panels in all_sp_params
        solar_panels: list[SolarPanel] = []
        sat_panel_faces: list[list[int]] = []
        for sp_key, sp_param in all_sp_params.items():
            
            # Load all ground station parameters for 'gs_key' of arbitrary/un-verified type
            nHat_B =           sp_param['nHat_B']
            panel_area =       sp_param['panel_area']
            panel_efficiency = sp_param['panel_efficiency']

            # ============ Check 'nHat_B' is a list of 3 int, with a vector norm of 1, and mounted on a unique sat face
            if not isinstance(nHat_B, list):
                raise ValueError(f"Expected solar panel '{sp_key}' parameter 'nHat_B' of type 'list'. Got: {type(nHat_B)}")
            else:
                if not len(nHat_B) == 3:
                    raise ValueError(f"Expected solar panel '{sp_key}' parameter 'nHat_B' to have 3 elements. Got {len(nHat_B)}")
                for i, elem in enumerate(nHat_B):
                    if not isinstance(elem, int):
                        try:
                            elem = int(elem)
                            nHat_B[i] = elem
                        except:
                            raise ValueError(f"Expected solar panel '{sp_key}' parameter 'nHat_B' element nr {i} to be of type 'int'. Got: {type(elem)}")
                norm = np.linalg.norm(np.array(nHat_B))
                if not norm == 1:
                    raise ValueError(f"Expected solar panel '{sp_key}' parameter 'nHat_B' to be of unit length. Got vector length {norm}")
            
                if nHat_B in sat_panel_faces:
                    raise ValueError(f"There has already been generated a solar panel on the satellite face {nHat_B}")
                else:
                    sat_panel_faces.append(nHat_B)

            # ============ Check 'panel_area' is a non-negative float
            if not isinstance(panel_area, float):
                try:
                    panel_area = float(panel_area)
                except:
                    raise ValueError(f"Expected solar panel '{sp_key}' parameter 'panel_area' of type 'float'. Got: {type(panel_area)}")
            if (panel_area <= 0):
                raise ValueError(f"Expected solar panel '{sp_key}' parameter 'panel_area' to be bigger than 0m^2. Got: {panel_area}")
            if (panel_area > 100):
                raise ValueError(f"Unrealistic solar panel area 'panel_area' detected for panel '{sp_key}'. Got: {panel_area}")
            
            # ============ Check 'panel_efficiency' is a float in range [0,1]
            if not isinstance(panel_efficiency, float):
                try:
                    panel_efficiency = float(panel_efficiency)
                except:
                    raise ValueError(f"Expected solar panel '{sp_key}' parameter 'panel_efficiency' of type 'float'. Got: {type(panel_efficiency)}")
            if (panel_efficiency < 0) or (panel_efficiency > 1):
                raise ValueError(f"Expected solar panel '{sp_key}' parameter 'panel_area' to be in range [0, 1]. Got: {panel_efficiency}")
            
            # Initialize SolarPanel instance
            solar_panel = SolarPanel(
                nHat_B = nHat_B,
                panel_area = panel_area,
                panel_efficiency = panel_efficiency
            )
            
            # Add to list
            logging.debug(f"[CFG] Appending {sp_key} to 'solar_panels'")
            solar_panels.append(solar_panel)

        return solar_panels               


    def validate_sim_parameters(self,
                                data_mode: str) -> None:
        """
        
        """

        # ============ Check 'data_mode' is type str and is an acceptable string
        if isinstance(data_mode, str):
            if not ((data_mode == "debug") or (data_mode == "optimized")):
                raise ValueError(f"Unexpected value given for 'data_mode'. "
                                 f"Got '{data_mode}', expected ['debug', 'optimized'])")
        else: 
            raise ValueError(f"Unexpected type given for 'data_mode'. "
                             f"Got '{type(data_mode)}', expected 'str'")



    def validate_rw_parameters(self,
                               RW_model,
                               spinUVecs,
                               ) -> None:
        """
        Validate all RW parameter inputs to ensure correct type and value.
        Raise ValueError if incorrect type or value is detected
        TODO: init_rpm
        TODO: max_rpm
        TODO: maxMomentum
        TODO: maxTorque
        TODO: minTorque
        TODO: useMinTorque
        TODO: useFriction
        TODO: fCoulomb
        TODO: fStatic
        TODO: betaStatic
        TODO: cViscous
        """
        # ============ Check 'RW_model' is type str and is an acceptable string
        if isinstance(RW_model, str):
            if not ((RW_model == "BalancedWheels") or (RW_model == "JitterSimple") or (RW_model == "JitterFullyCoupled")):
                raise ValueError(f"Unexpected value given for 'RW_model'. "
                                 f"Got '{RW_model}', expected ['BalancedWheels', 'JitterSimple', 'JitterFullyCoupled'])")
        else: 
            raise ValueError(f"Unexpected type given for 'RW_model'. "
                             f"Got '{type(RW_model)}', expected 'str'")
        
        # ============ Check 'spinUVecs' is a list of >0 lists, each internal list has 3 elements with norm==1
        if isinstance(spinUVecs, list):
            num_RWs = len(spinUVecs)
            if num_RWs == 0:
                raise ValueError(f"No RW unit vectors defined in 'spinUVecs'")
            
            for i, spin_uvec in enumerate(spinUVecs):
                if len(spin_uvec) != 3:
                    raise ValueError(f"RW spin unit vector nr. {i} has {len(spin_uvec)} elements, expected 3")
                v = np.array(spin_uvec)
                norm = np.linalg.norm(v)
                if not np.isclose(norm, 1.0):
                    raise ValueError(f"RW spin vector nr. {i} is not of unit length. (given length: {norm})")
        else:
            raise ValueError(f"Unexpected type given for 'spinUVecs'. "
                             f"Got '{type(spinUVecs)}', expected 'list[list[float]]'")
        
    
    def validate_thruster_parameters(self,
                                     thr_pos_B: list,
                                     thr_dir_B: list,
                                     thr_model_override: str,
                                     use_min_pulse_time: bool,
                                     min_pulse_time: float,
                                     max_thrust: float,
                                     thrust_blowdown_coeff: list[float],
                                     steady_isp: float,
                                     isp_blowdown_coeff: list[float],
                                     area_nozzle: float,
                                     thr_mag_disp: float
                                     ) -> None:
        """
        TODO
        Validate all thruster parameter inputs to ensure correct type and value. 
        Raise ValueError if incorrect type or value is detected
        """
        accepted_model_overrides = ['MOOG_Monarc_1', 'MOOG_Monarc_5', 'MOOG_Monarc_22_6', 'MOOG_Monarc_90HT']
        pass


    def validate_eps_parameters(self,
                                bat_storage_capacity: float,
                                init_bat_charge: float,
                                RW_base_draw: float,
                                OBC_const_draw: float) -> None:
        """
        Validate all EPS parameter inputs to ensure correct type and value. 
        Raise ValueError if incorrect type or value is detected
        """
        
        # ============ Check if 'bat_storage_capacity' is of type float and has value >= 0
        if not isinstance(bat_storage_capacity, float):
            raise ValueError(f"Expected EPS parameter 'bat_storage_capacity' of type 'float'. Got: {type(bat_storage_capacity)}")
        if bat_storage_capacity < 0:
            raise ValueError(f"Expected EPS parameter 'bat_storage_capacity' to be bigger or equal to 0. Got: {bat_storage_capacity}")
        
        # ============ Check if 'init_bat_charge' is of type float and has value in range (0, 1)
        if not isinstance(init_bat_charge, float):
            raise ValueError(f"Expected EPS parameter 'init_bat_charge' of type 'float'. Got: {type(init_bat_charge)}")
        if (init_bat_charge < 0) or (init_bat_charge > 1):
            raise ValueError(f"Expected EPS parameter 'init_bat_charge' to be in range [0, 1]. Got: {init_bat_charge}")
        
        # ============ Check if 'RW_base_draw' is of type float and has value >= 0
        if not isinstance(RW_base_draw, float):
            raise ValueError(f"Expected EPS parameter 'RW_base_draw' of type 'float'. Got: {type(RW_base_draw)}")
        if bat_storage_capacity < 0:
            raise ValueError(f"Expected EPS parameter 'RW_base_draw' to be bigger or equal to 0. Got: {RW_base_draw}")
        
        # ============ Check if 'OBC_const_draw' is of type float and has value >= 0
        if not isinstance(OBC_const_draw, float):
            raise ValueError(f"Expected EPS parameter 'OBC_const_draw' of type 'float'. Got: {type(OBC_const_draw)}")
        if OBC_const_draw < 0:
            raise ValueError(f"Expected EPS parameter 'OBC_const_draw' to be bigger or equal to 0. Got: {OBC_const_draw}")