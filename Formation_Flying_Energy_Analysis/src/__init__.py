import logging
import matplotlib as mpl

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from object_definitions.Config_def import Config
from object_definitions.MonteCarloConfig_def import MonteCarloConfig


def initialize(mc_config_file_path) -> MonteCarloConfig:
    """
    ==========================================================================================================
    1. Initialize global Monte Carlo instance
    2. Configure global logging format
    ==========================================================================================================
    """
    mc_cfg = MonteCarloConfig(mc_config_file_path)

    if mc_cfg.mc_enabled:
        # -------------------------------------------------------------
        # Configure reduced per-sim logging
        # -------------------------------------------------------------
        
        # TODO
        pass

    else:
        # -------------------------------------------------------------
        # Configure full debug logging
        # -------------------------------------------------------------

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

    return mc_cfg