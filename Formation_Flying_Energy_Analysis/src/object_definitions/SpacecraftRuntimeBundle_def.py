from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft

from object_definitions.Satellite_def import Satellite
from object_definitions.BasiliskDynamicsModel_def import BasiliskDynamicsModel
from object_definitions.FswStack_def import FswStack




@dataclass
class SpacecraftRuntimeBundle:
    """
    Stable runtime bundle for one spacecraft.

    This keeps all per-satellite objects alive and provides a single place for the
    scenario orchestrator to find the dynamics object, the FSW object, and all logs.
    """
    sat_idx: int # 0-indexed
    sat: Satellite
    scObj: spacecraft.Spacecraft

    # Per-satellite model objects (environment models are omitted bc. they are same for all)
    dynModel: BasiliskDynamicsModel
    fsw: FswStack

    # Frequently used handles cached at orchestration level
    # sc_state_recorder: Any

    # Optional/common recorders
    # atm_recorder: Any = None
    # rw_motor_recorder: Any = None
    # att_err_recorder: Any = None
    # nav_trans_recorder: Any = None
    # rw_speed_recorder: Any = None
    # battery_recorder: Any = None
    # power_sink_recorder: Any = None

    # Collections
    # rw_recorders: list[Any] = field(default_factory=list)
    # solar_panel_recorders: list[Any] = field(default_factory=list)
    # rw_power_recorders: list[Any] = field(default_factory=list)

    # Escape hatch for future additions
    # extra: dict[str, Any] = field(default_factory=dict)