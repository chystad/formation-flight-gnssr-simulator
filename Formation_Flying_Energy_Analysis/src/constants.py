from pathlib import Path

# File/dir paths and names
BASE_CONFIG_PATH = 'Formation_Flying_Energy_Analysis/configs/base.yaml'
OVERRIDE_CONFIG_DIR = Path('Formation_Flying_Energy_Analysis/configs/run_overrides')
MC_CONFIG_PATH = 'Formation_Flying_Energy_Analysis/configs/monte_carlo.yaml'
OUTPUT_DATA_ROOT_DIR = Path('Formation_Flying_Energy_Analysis/output_data')
SINGLE_OUTPUT_DATA_DIR_NAME = "single_runs"
BATCH_OUTPUT_DATA_DIR_NAME = "batch_runs"
SPACE_WEATHER_DATA_FILE_PATH = "shared_input_data/msis_data/Kp_ap_Ap_SN_F107_since_2010.txt"