from pathlib import Path

# File/dir paths and names
BASE_CONFIG_PATH = 'Formation_Flying_Energy_Analysis/configs/base.yaml'
OVERRIDE_CONFIG_DIR = Path('Formation_Flying_Energy_Analysis/configs/run_overrides')
MC_CONFIG_PATH = 'Formation_Flying_Energy_Analysis/configs/monte_carlo.yaml'
OUTPUT_DATA_ROOT_DIR = Path('Formation_Flying_Energy_Analysis/output_data')
SINGLE_OUTPUT_DATA_DIR_NAME = "single_runs"
BATCH_OUTPUT_DATA_DIR_NAME = "batch_runs"
SPACE_WEATHER_DATA_FILE_PATH = "shared_input_data/msis_data/Kp_ap_Ap_SN_F107_since_2010.txt"
VIZARD_SAVE_PATH = "/home/chris/code/formation-flight-gnssr-simulator/Formation_Flying_Energy_Analysis/output_data/_VizFiles/bsk_sim.bin"

# Model rates [<time>/update] TODO: Move to Config
ENV_RATE: float = 0.5 # [s/update] Update rate for environment models
DYN_RATE: float = 0.5 # [s/update] Update rate for dynamical models
FSW_RATE: float = 0.5 # [s/update] Update rate for flight software stack
FORM_CTRL_RATE: float = 0.5 # [s/update] TODO: Update rate for the formation flight stack 
MSIS_RATE: float = 30. # [s/update] Update rate for MSIS input parameters
FLUSH_RATE: float = 3. # [hour/update]

# Recorder sample rates [s/sample]
HIGH_SAMPLE_RATE: float = 0.5 # [s/sample] NOTE: Must be integer multilple of 'DYN_RATE'
MID_SAMPLE_RATE: float = 5. # [s/sample] NOTE: Must be integer multilple of 'DYN_RATE'
LOW_SAMPLE_RATE: float = 30. # [s/sample] NOTE: Must be integer multilple of 'DYN_RATE'