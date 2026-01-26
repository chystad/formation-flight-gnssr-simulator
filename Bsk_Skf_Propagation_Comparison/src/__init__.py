import logging

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.Config_def import Config


def initialize(config_file_path) -> Config:
    """
    ==========================================================================================================
    1. Configure logging format
    2. Initialize global config instance
    ==========================================================================================================
    """
    # Configure debug logging format
    logging.basicConfig(
        format="%(asctime)s    %(message)s",
        datefmt="[%H:%M:%S]",
        level=logging.DEBUG,
    )
    
    # Get Config instance forom config file
    cfg = Config(config_file_path)

    return cfg