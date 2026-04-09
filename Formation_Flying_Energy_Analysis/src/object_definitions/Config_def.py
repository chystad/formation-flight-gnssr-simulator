import yaml
import logging
import numpy as np
from typing import Any, Optional
from pathlib import Path
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime

from object_definitions.Satellite_def import Satellite
from object_definitions.GroundStation_def import GroundStation
from object_definitions.SolarPanel_def import SolarPanel
from object_definitions.SimData_def import OUTPUT_DATA_SAVE_DIR

from Basilisk.utilities import (orbitalMotion, macros, unitTestSupport)



class Config:
    def __init__(self, config_file_path: str) -> None:
        """
        =========================================================================================================
        [WORK IN PROGRESS]
        Initialize Config instance with attributes from the config file. 
        Perform checks to ensure all parameters are received as expected. 

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
            ground_stations (list[GroundStation]): 
            solar_panels (list[SolarPanel]):
            b_set (BasiliskSettings):       BasiliskSettings instance describing the Basilisk simulation settings
            s_set (SkyfieldSettings):       SkyfieldSettings instance describing the Skyfield simulation settings     
        =========================================================================================================
        """
        ###################
        # Load cofig file #
        ###################
        d_cfg = self.read(config_file_path) # default config
        
        ##################################################
        # Fetch global simulation parameters config file #
        ##################################################
        startTime_str =         str(    d_cfg['SIMULATION']['startTime'])
        simulationDuration =    float(  d_cfg['SIMULATION']['simulationDuration'])  
        deltaT =                float(  d_cfg['SIMULATION']['deltaT'])  
        integrator =            str(    d_cfg['SIMULATION']['integrator'])
        num_satellites =        int(    d_cfg['SIMULATION']['num_satellites'])
        sat_init_source =       str(    d_cfg['SIMULATION']['sat_init_source'])
        all_sat_params =                d_cfg['SATELLITES'] # dict[str, dict[str, Any]]
        all_gs_params =                 d_cfg['GROUND_STATIONS'] # dict[str, dict[str, Any]]     

        # Electrical power system parameters (same for all satellites)
        bat_storage_capacity =  float(  d_cfg['EPS_PARAMETERS']['bat_storage_capacity'])
        init_bat_charge =       float(  d_cfg['EPS_PARAMETERS']['init_bat_charge'])
        RW_base_draw =          float(  d_cfg['EPS_PARAMETERS']['RW_base_draw'])
        OBC_const_draw =        float(  d_cfg['EPS_PARAMETERS']['OBC_const_draw'])
        all_sp_params =                 d_cfg['EPS_PARAMETERS']['solar_panels'] # dict[str, dict[str, Any]]  

        # Reaction wheel parameters (same for all satellites)
        RW_model =              str(    d_cfg['RW_PARAMETERS']['RW_model'])
        spinUVecs =                     d_cfg['RW_PARAMETERS']['spinUVecs'] # list[list[float]]
        init_rpm =              float(  d_cfg['RW_PARAMETERS']['init_rpm'])
        max_rpm =               float(  d_cfg['RW_PARAMETERS']['max_rpm'])
        maxMomentum =           float(  d_cfg['RW_PARAMETERS']['maxMomentum'])
        maxTorque =             float(  d_cfg['RW_PARAMETERS']['maxTorque'])
        minTorque =             float(  d_cfg['RW_PARAMETERS']['minTorque'])
        # I_RW =                  float(  d_cfg['RW_PARAMETERS']['I_RW'])
        useMinTorque =          bool(   d_cfg['RW_PARAMETERS']['useMinTorque'])
        useFriction =           bool(   d_cfg['RW_PARAMETERS']['useFriction'])
        fCoulomb =              float(  d_cfg['RW_PARAMETERS']['fCoulomb'])
        fStatic =               float(  d_cfg['RW_PARAMETERS']['fStatic'])
        betaStatic =            float(  d_cfg['RW_PARAMETERS']['betaStatic'])
        cViscous =              float(  d_cfg['RW_PARAMETERS']['cViscous'])
        
        

        # Thruster parameters (same for all satellites)
        temp =                  bool(   d_cfg['THRUSTER_PARAMETERS']['temp'])

        # Magnetorquer parameters (same for all satellites)
        temp =                  bool(   d_cfg['MTQ_PARAMETERS']['temp'])
        
        # Disturbance torque settings
        temp =                  bool(   d_cfg['DISTURBANCE_TORQUE']['temp'])
        
        # Disturbance force settings
        sphericalHarmonicsDegree =      int(    d_cfg['DISTURBANCE_FORCE']['sphericalHarmonicsDegree'])
        useSphericalHarmonics =         bool(   d_cfg['DISTURBANCE_FORCE']['useSphericalHarmonics'])
        useMsisDrag =                   bool(   d_cfg['DISTURBANCE_FORCE']['useMsisDrag'])
        useExponentialDensityDrag =     bool(   d_cfg['DISTURBANCE_FORCE']['useExponentialDensityDrag'])
        useSRP =                        bool(   d_cfg['DISTURBANCE_FORCE']['useSRP'])
        useSun3rdBody =                 bool(   d_cfg['DISTURBANCE_FORCE']['useSun3rdBody'])
        useMoon3rdBody =                bool(   d_cfg['DISTURBANCE_FORCE']['useMoon3rdBody'])

        
        ################################################################
        # Perform checks to ensure parameters are received as expected #
        ################################################################
       
        # Validate RW parameters
        self.validate_rw_parameters(
            RW_model, 
            spinUVecs
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

        # Ground stattions
        self.ground_stations: list[GroundStation] = ground_stations

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