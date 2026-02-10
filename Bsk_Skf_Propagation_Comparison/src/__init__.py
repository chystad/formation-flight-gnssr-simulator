import logging
import matplotlib as mpl

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

    # Only show warnings and errors globally
    logging.basicConfig(level=logging.WARNING)

    # Matplotlib: silence backend + font-manager chatter
    mpl.set_loglevel("warning")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Pillow (PIL): silence PNG chunk debug like "STREAM b'IHDR'"
    #PngImagePlugin.debug = False
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    
    # Get Config instance forom config file
    cfg = Config(config_file_path)

    return cfg