import h5py
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from numpy.typing import NDArray
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# Global definition of data save folder path
OUTPUT_DATA_SAVE_DIR = Path('Formation_Flying_Energy_Analysis/output_data/sim_data')
SPACE_WEATHER_DATA_FILE_PATH = "shared_input_data/msis_data/Kp_ap_Ap_SN_F107_since_2010.txt"


@dataclass_json
@dataclass
class SpaceWeatherDay:
    """One UTC day of space-weather data from Kp_ap_Ap_SN_F107_since_2010.txt."""
    ap: list[int]        # 8x 3-hour ap values: [00-03, 03-06, ..., 21-24]
    Ap: int              # daily Ap
    f107obs: float       # adjusted F10.7
    f107adj: float       # observed F10.7

# TODO
@dataclass 
class MissionSimData:
    TODO: bool

# TODO
@dataclass 
class SpacecraftSimData:
    TODO: bool