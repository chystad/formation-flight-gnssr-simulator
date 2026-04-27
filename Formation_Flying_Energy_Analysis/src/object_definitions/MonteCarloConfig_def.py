import yaml
from pathlib import Path
from datetime import datetime

from constants import (OUTPUT_DATA_ROOT_DIR, BATCH_OUTPUT_DATA_DIR_NAME)

class MonteCarloConfig:
    def __init__(self, mc_config_file_path: str) -> None:
    
        # -------------------------------
        # Load cofig file
        # -------------------------------
        mc_cfg = self._read(mc_config_file_path)
        
        # -------------------------------
        # Fetch global simulation parameters config file #
        # -------------------------------
        # Monte Carlo parameters
        mc_enabled =        mc_cfg['MONTE_CARLO']['mc_enabled']
        num_bsk_sims =  mc_cfg['MONTE_CARLO']['num_bsk_sims']

        # Monte Carlo run description
        desc_gnssr_formation_type = str(    mc_cfg['MONTE_CARLO']['desc_gnssr_formation_type'])
        desc_varied_parameters =    str(    mc_cfg['MONTE_CARLO']['desc_varied_parameters'])
        desc_goal =                 str(    mc_cfg['MONTE_CARLO']['desc_goal'])
        
        # -------------------------------
        # Perform checks to ensure parameters are received as expected #
        # ------------------------------- 
        self._validate_monte_carlo_parameters(
            mc_enabled=mc_enabled,
            num_bsk_sims=num_bsk_sims
        )

        # -------------------------------
        # Assign instance attributes #
        # -------------------------------
        # Monte Carlo parameters
        self.mc_enabled: bool = mc_enabled
        self.num_bsk_sims: int = num_bsk_sims

        # Helper parameters
        self.timestamp_str: str = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.mc_dir_name: str = f"Monte_Carlo_{self.timestamp_str}"
        self.mc_run_dir_name_base: str = "run_" # Data from a single bsk run inside MC will be stored inside the f"{mc_run_name_base}{run_idx:03d}" folder

        # Monte Carlo run description
        self.desc_gnssr_formation_type: str = desc_gnssr_formation_type
        self.desc_varied_parameters: str = desc_varied_parameters
        self.desc_goal: str = desc_goal

        

    
    ####################
    # Public functions #
    ####################

    def generate_mc_data_output_folders(self) -> None:
        """
        Generate empty Monte Carlo data folder hierarchy:
        OUTPUT_DATA_ROOT_DIR/
        |
        |---BATCH_OUTPUT_DATA_DIR_NAME/
            |
            |---<self.mc_dir_name>/
                |
                |---run_000/
                |---run_001/
                |      .
                |      .
                |      .
                |---run_<num_bsk_sims - 1>/
        """
        pass
        

    def generate_config_overrides(self) -> None:
        """
        Generate <num_bsk_sims - 1> config override files with distributed satellite deployment 
        velocity and angular rate from a shared deployer orbit.
        Override files will be generated in 'configs/run_overrides/'
        """
        pass
    



    ############################
    # Private helper functions #
    ############################

    def _read(self, config_file_path: str):
        # Get full path to the target config file
        config_path = Path(config_file_path)
        
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.full_load(f)

        return config
    

    def _validate_monte_carlo_parameters(self,
                                         mc_enabled: bool,
                                         num_bsk_sims: int
                                         ) -> None:
        """
        Validate all monte carlo input parameters from config file.
        Raise value error if any parameter is received not as expected
        """

        if not isinstance(mc_enabled, bool):
            raise ValueError(f"'mc_enabled' parameter is of type {type(mc_enabled)}, expected bool")
        
        if not isinstance(num_bsk_sims, int):
            raise ValueError(f"'num_bsk_sims' parameter is of type {type(num_bsk_sims)}, expected int")
        else:
            if mc_enabled and (num_bsk_sims <= 1):
                raise ValueError(f"The Monte Carlo run must be configured to run more than 1 Basilisk sim, got {num_bsk_sims}")