from __future__ import annotations

import h5py

"""
Optimized plotting from a saved single-run HDF5 output folder.

Assumed output file structure:
~/Formation_Flying_Energy_Analysis/output_data/single_runs/<timestamp>/
  * <timestamp>_cfg.yaml
  * FSW0_mode_switching
  * FSW1_mode_switching
  * sat_0.h5
  * sat_1.h5

The intent of this script is to load only the HDF5 fields needed for one
analysis/plot at a time, then release them before moving on to the next plot.
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as mpl
import numpy as np
from numpy.typing import NDArray


# ========================= USER INPUT ========================= #
SINGLE_RUN_TIMESTAMP_TO_LOAD = "20260528_103611"  # timestamp of the single run folder
PLT_SAT_IDX = 1                                  # 0-indexed spacecraft used for spacecraft-specific plots

# Base folder containing all single-run folders.
SINGLE_RUNS_BASE_DIR = (
    Path.home()
    / "Formation_Flying_Energy_Analysis"
    / "output_data"
    / "single_runs"
)

# Downsampling settings for ordinary continuous-valued fields.
# If DOWNSAMPLE_STRIDE is None, the loader chooses a stride so that each loaded
# field has at most MAX_PLOT_POINTS samples. Set MAX_PLOT_POINTS=None to disable
# automatic downsampling.
DOWNSAMPLE_STRIDE: int | None = None
MAX_PLOT_POINTS: int | None = 300_000

# Pointing mode is a discrete step signal. For this field, preserve all mode
# transitions instead of simple stride-based downsampling. This avoids missing
# short modes such as BURN or BURN_TRANSIT.
PRESERVE_POINTING_MODE_TRANSITIONS = True

# Figure behavior
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_OUTPUT_DIR_NAME = "plots"
# ============================================================== #



# ========================= PATH SETUP ========================= #
THIS_FILE = Path(__file__).resolve()

# .../Formation_Flying_Energy_Analysis/src/plotting/single_run_analysis_and_plotting.py
SRC_DIR = THIS_FILE.parents[1]
PROJECT_DIR = SRC_DIR.parent

SINGLE_RUNS_BASE_DIR = PROJECT_DIR / "output_data" / "single_runs"
# ============================================================= #


# Battery storage capacity used for plotting battery energy percentage.
# If None, the script normalizes by max(storageLevel) from the loaded data.
# Recommended: set this to the same value as cfg.bat_storage_capacity [Wh].
BAT_STORAGE_CAPACITY_WH: float | None = 100.0


# Keep this mapping local so this file can be used as a lightweight
# post-processing script without importing the full simulator stack.
INT_TO_POINTING_MODE: dict[int, str] = {
    0: "coast",
    1: "comms",
    2: "charge",
    3: "capture",
    4: "burn_transit",
    5: "burn",
    6: "emergency",
    7: "error",
}


@dataclass
class LoadedH5Field:
    """Container for one loaded HDF5 field."""

    data: NDArray[Any]
    dt_s: float
    n_samples: int
    source_n_samples: int
    stride: int
    source_indices: NDArray[np.int64] | None = None



#########################################
# Data loading /saving helper functions #
#########################################

def get_single_run_dir(timestamp: str) -> Path:
    run_dir = SINGLE_RUNS_BASE_DIR / timestamp

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Single-run folder does not exist: {run_dir}\n"
            f"Current SINGLE_RUNS_BASE_DIR is: {SINGLE_RUNS_BASE_DIR}\n"
            "Check that SINGLE_RUN_TIMESTAMP_TO_LOAD matches an existing folder."
        )

    return run_dir


def get_satellite_h5_path(timestamp: str, sat_idx: int) -> Path:
    """Return the HDF5 file path for one spacecraft."""

    run_dir = get_single_run_dir(timestamp)
    sat_path = run_dir / f"sat_{sat_idx}.h5"

    if not sat_path.exists():
        raise FileNotFoundError(f"Satellite HDF5 file does not exist: {sat_path}")

    return sat_path


def get_available_satellite_indices(timestamp: str) -> list[int]:
    """Return sorted spacecraft indices for available sat_*.h5 files."""

    run_dir = get_single_run_dir(timestamp)
    sat_indices: list[int] = []

    for path in run_dir.glob("sat_*.h5"):
        try:
            sat_idx = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        sat_indices.append(sat_idx)

    sat_indices = sorted(set(sat_indices))

    if len(sat_indices) == 0:
        raise FileNotFoundError(f"No sat_*.h5 files found in {run_dir}")

    return sat_indices


def _choose_stride(n_samples: int) -> int:
    """
    Choose a downsampling stride for ordinary continuous fields.

    Priority:
      1. Use DOWNSAMPLE_STRIDE if explicitly set.
      2. Otherwise, use MAX_PLOT_POINTS to choose a stride.
      3. If both are None, load every sample.
    """

    if DOWNSAMPLE_STRIDE is not None:
        if DOWNSAMPLE_STRIDE < 1:
            raise ValueError(f"DOWNSAMPLE_STRIDE must be >= 1, got {DOWNSAMPLE_STRIDE}.")
        return int(DOWNSAMPLE_STRIDE)

    if MAX_PLOT_POINTS is None:
        return 1

    if MAX_PLOT_POINTS < 1:
        raise ValueError(f"MAX_PLOT_POINTS must be >= 1, got {MAX_PLOT_POINTS}.")

    return max(1, int(np.ceil(n_samples / MAX_PLOT_POINTS)))


def _mode_transition_indices(mode_code: NDArray[Any]) -> NDArray[np.int64]:
    """
    Return indices needed to preserve every discrete mode transition.

    This compresses a full sampled mode time history to one point per constant
    segment start, plus the final sample. It is therefore much smaller than the
    full time history while still preserving all transitions for a step plot.
    """

    if mode_code.ndim != 1:
        raise ValueError(
            f"pointingModeCode must be a 1D signal, got shape {mode_code.shape}."
        )

    if mode_code.size == 0:
        return np.array([], dtype=np.int64)

    change_indices = np.nonzero(np.diff(mode_code) != 0)[0] + 1

    # Include first sample and final sample. np.unique handles the case n=1.
    indices = np.unique(
        np.concatenate(
            (
                np.array([0], dtype=np.int64),
                change_indices.astype(np.int64),
                np.array([mode_code.size - 1], dtype=np.int64),
            )
        )
    )

    return indices


def load_satellite_fields(
    timestamp: str,
    sat_idx: int,
    field_names: list[str],
    preserve_change_fields: set[str] | None = None,
) -> dict[str, LoadedH5Field]:
    """
    Load selected fields from one spacecraft HDF5 file.

    Args:
        timestamp:
            Single-run timestamp identifying the run folder.

        sat_idx:
            0-indexed spacecraft index.

        field_names:
            HDF5 field names to load, for example:
                ["currentNetPower", "fuelMass"]

        preserve_change_fields:
            Fields to compress by preserving discrete value changes instead of
            applying stride-based downsampling. Currently intended for
            "pointingModeCode".

    Returns:
        Dictionary mapping field name to LoadedH5Field.

    Expected HDF5 field format:
        field_name/
          data
          dt_s
          n_samples
    """

    if preserve_change_fields is None:
        preserve_change_fields = set()

    if len(field_names) == 0:
        raise ValueError("field_names must contain at least one field name.")

    sat_path = get_satellite_h5_path(timestamp, sat_idx)
    loaded: dict[str, LoadedH5Field] = {}

    with h5py.File(sat_path, "r") as h5:
        for field_name in field_names:
            if field_name not in h5:
                available = sorted(list(h5.keys()))
                raise KeyError(
                    f"Field '{field_name}' not found in {sat_path}.\n"
                    f"Available fields are: {available}"
                )

            grp = h5[field_name]
            if "data" not in grp or "dt_s" not in grp or "n_samples" not in grp:
                raise KeyError(
                    f"Field '{field_name}' in {sat_path} does not have the expected "
                    "subgroups/datasets: data, dt_s, n_samples."
                )

            dset = grp["data"]
            dt_s = float(grp["dt_s"][()])
            source_n_samples = int(grp["n_samples"][()])

            if source_n_samples != int(dset.shape[0]):
                raise ValueError(
                    f"n_samples mismatch for '{field_name}' in {sat_path}: "
                    f"metadata says {source_n_samples}, dataset shape is {dset.shape}."
                )

            if field_name in preserve_change_fields:
                # Load the full discrete signal, compress to transition points,
                # and return only the transition-preserving subset.
                full_data = np.asarray(dset[()])
                transition_idx = _mode_transition_indices(full_data)
                data = full_data[transition_idx]
                stride = 1
                source_indices = transition_idx
                del full_data
            else:
                stride = _choose_stride(source_n_samples)
                data = np.asarray(dset[::stride])
                source_indices = None

            loaded[field_name] = LoadedH5Field(
                data=data,
                dt_s=dt_s,
                n_samples=int(data.shape[0]),
                source_n_samples=source_n_samples,
                stride=stride,
                source_indices=source_indices,
            )

    return loaded


def save_or_show_plot(fig: mpl.Figure, run_dir: Path, file_stem: str) -> None:
    """Save and/or show a figure depending on the user settings."""

    if SAVE_PLOTS:
        out_dir = run_dir / PLOT_OUTPUT_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{file_stem}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {out_path}")

    # if SHOW_PLOTS:
    #     mpl.show()





###################################################################################################################################################################################
############################################################### S I N G L E    S A T E L L I T E    P L O T T I N G ###############################################################
###################################################################################################################################################################################

def plot_all_single_pointing_mode_plots() -> None:
    plot_single_satellite_pointing_mode_from_h5(
        timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD,
        sat_idx=PLT_SAT_IDX,
    )

def plot_all_single_satellite_fuel_plots() -> None:
    plot_single_satellite_fuel_mass_from_h5(
        timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD,
        sat_idx=PLT_SAT_IDX,
    )

def plot_all_single_satellite_eps_plots() -> None:
    plot_single_satellite_simple_eps_overview(
        timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD,
        sat_idx=PLT_SAT_IDX,
        bat_storage_capacity_Wh=BAT_STORAGE_CAPACITY_WH
    )
    plot_single_satellite_simple_eps_overview_with_pointing_mode(
        timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD,
        sat_idx=PLT_SAT_IDX,
        bat_storage_capacity_Wh=BAT_STORAGE_CAPACITY_WH
    )


#######################################
# Fuel consumption over time plotting #
#######################################

def plot_single_satellite_fuel_mass_from_h5(timestamp: str, sat_idx: int) -> None:
    """
    Load and plot fuel mass over time for one spacecraft.

    This mirrors the fuel-mass subplot style used in debug_plotting.py:
        - time axis in minutes
        - fuel mass in kg
        - plain y-axis formatting
        - grid and legend
    """

    loaded = load_satellite_fields(
        timestamp=timestamp,
        sat_idx=sat_idx,
        field_names=["fuelMass"],
    )

    fuel_data = loaded["fuelMass"]
    fuel_mass = fuel_data.data

    if fuel_mass.ndim != 1:
        raise ValueError(
            f"Expected fuelMass data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {fuel_mass.shape}."
        )

    t_min = np.arange(fuel_data.n_samples) * fuel_data.dt_s * fuel_data.stride / 60.0
    t_day = t_min / 60.0 / 24.0

    fig, ax = mpl.subplots(figsize=(10, 4))

    label = "Leader" if sat_idx == 0 else f"Follower {sat_idx}"

    ax.plot(t_day, fuel_mass, label=label)

    ax.set_title(f"Fuel mass for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Fuel mass [kg]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()

    print(
        f"Loaded fuelMass for sat_{sat_idx}: "
        f"{fuel_data.n_samples} plotted samples from "
        f"{fuel_data.source_n_samples} stored samples "
        f"(stride={fuel_data.stride})."
    )

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, f"sat_{sat_idx}_fuel_mass")

    del loaded, fuel_data, fuel_mass, t_min
    gc.collect()


##################################
# Simple EPS overview plotting   #
##################################

def _compute_battery_energy_percent(
    storage_level_Ws: NDArray[Any],
    bat_storage_capacity_Wh: float | None,
) -> NDArray[Any]:
    """
    Convert battery storage level [Ws] to battery energy percentage.

    If bat_storage_capacity_Wh is None, use max(storage_level_Ws) as a fallback
    normalization value. This is useful when the config value is not available,
    but it means the plotted percentage is relative to the maximum observed
    stored energy, not necessarily the true battery capacity.
    """

    if storage_level_Ws.ndim != 1:
        raise ValueError(
            f"Expected storageLevel to be 1D, got shape {storage_level_Ws.shape}."
        )

    if bat_storage_capacity_Wh is not None:
        if bat_storage_capacity_Wh <= 0.0:
            raise ValueError(
                f"BAT_STORAGE_CAPACITY_WH must be positive or None, "
                f"got {bat_storage_capacity_Wh}."
            )

        bat_storage_capacity_Ws = bat_storage_capacity_Wh * 3600.0
    else:
        bat_storage_capacity_Ws = float(np.max(storage_level_Ws))

        if bat_storage_capacity_Ws <= 0.0:
            raise ValueError(
                "Cannot normalize battery energy percentage because "
                "max(storageLevel) <= 0."
            )

        print(
            "BAT_STORAGE_CAPACITY_WH is None. "
            "Battery percentage is normalized by max(storageLevel) in the loaded data."
        )

    return 100.0 * storage_level_Ws / bat_storage_capacity_Ws


def _plot_pointing_mode_on_axis(
    ax: mpl.Axes,
    timestamp: str,
    sat_idx: int,
) -> None:
    """
    Plot pointingModeCode on an existing axis.

    This is the subplot-equivalent of plot_single_satellite_pointing_mode_from_h5().
    """

    preserve_fields = {"pointingModeCode"} if PRESERVE_POINTING_MODE_TRANSITIONS else set()

    loaded = load_satellite_fields(
        timestamp=timestamp,
        sat_idx=sat_idx,
        field_names=["pointingModeCode"],
        preserve_change_fields=preserve_fields,
    )

    mode_data = loaded["pointingModeCode"]
    mode_code = mode_data.data.astype(int)

    if mode_data.source_indices is not None:
        t_h = mode_data.source_indices * mode_data.dt_s / 3600.0
    else:
        t_h = (
            np.arange(mode_data.n_samples)
            * mode_data.dt_s
            * mode_data.stride
            / 3600.0
        )

    ax.step(t_h, mode_code, where="post")
    ax.set_title("Pointing mode")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Mode [-]")

    tick_values = sorted(INT_TO_POINTING_MODE.keys())
    tick_labels = [INT_TO_POINTING_MODE[value] for value in tick_values]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

    ax.grid(True)

    print(
        f"Loaded pointingModeCode for sat_{sat_idx}: "
        f"{mode_data.n_samples} plotted samples from "
        f"{mode_data.source_n_samples} stored samples."
    )

    del loaded, mode_data, mode_code, t_h
    gc.collect()


def plot_single_satellite_simple_eps_overview(
    timestamp: str,
    sat_idx: int,
    bat_storage_capacity_Wh: float | None = BAT_STORAGE_CAPACITY_WH,
) -> None:
    """
    Plot a simple EPS overview for one spacecraft.

    Top subplot:
        Battery energy [%]

    Bottom subplot:
        Battery current net power [W]
    """

    loaded = load_satellite_fields(
        timestamp=timestamp,
        sat_idx=sat_idx,
        field_names=["storageLevel", "currentNetPower"],
    )

    storage_data = loaded["storageLevel"]
    net_power_data = loaded["currentNetPower"]

    storage_level_Ws = storage_data.data
    current_net_power_W = net_power_data.data

    if storage_level_Ws.ndim != 1:
        raise ValueError(
            f"Expected storageLevel data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {storage_level_Ws.shape}."
        )

    if current_net_power_W.ndim != 1:
        raise ValueError(
            f"Expected currentNetPower data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {current_net_power_W.shape}."
        )

    battery_energy_percent = _compute_battery_energy_percent(
        storage_level_Ws=storage_level_Ws,
        bat_storage_capacity_Wh=bat_storage_capacity_Wh,
    )

    t_storage_h = (
        np.arange(storage_data.n_samples)
        * storage_data.dt_s
        * storage_data.stride
        / 3600.0
    )
    t_power_h = (
        np.arange(net_power_data.n_samples)
        * net_power_data.dt_s
        * net_power_data.stride
        / 3600.0
    )

    fig, axs = mpl.subplots(2, 1, sharex=False, figsize=(11, 6))
    fig.suptitle(f"Simple EPS overview for spacecraft #{sat_idx}")

    axs[0].plot(t_storage_h, battery_energy_percent, label="Battery energy")
    axs[0].set_title("Battery energy")
    axs[0].set_ylabel("Energy [%]")
    axs[0].set_ylim(-5.0, 105.0)

    axs[1].plot(t_power_h, current_net_power_W, label="Current net power")
    axs[1].set_title("Current net power")
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Power [W]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()

    print(
        f"Loaded EPS fields for sat_{sat_idx}: "
        f"storageLevel {storage_data.n_samples}/{storage_data.source_n_samples} samples, "
        f"currentNetPower {net_power_data.n_samples}/{net_power_data.source_n_samples} samples."
    )

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, f"sat_{sat_idx}_simple_eps_overview")

    del (
        loaded,
        storage_data,
        net_power_data,
        storage_level_Ws,
        current_net_power_W,
        battery_energy_percent,
        t_storage_h,
        t_power_h,
    )
    gc.collect()


def plot_single_satellite_simple_eps_overview_with_pointing_mode(
    timestamp: str,
    sat_idx: int,
    bat_storage_capacity_Wh: float | None = BAT_STORAGE_CAPACITY_WH,
) -> None:
    """
    Plot a simple EPS overview together with pointing mode.

    Top subplot:
        Battery energy [%]

    Middle subplot:
        Battery current net power [W]

    Bottom subplot:
        Pointing mode. This uses the same plotting logic as
        plot_single_satellite_pointing_mode_from_h5(), but draws into the
        bottom subplot. The bottom subplot is allocated approximately 45%
        of the vertical plotting space.
    """

    loaded = load_satellite_fields(
        timestamp=timestamp,
        sat_idx=sat_idx,
        field_names=["storageLevel", "currentNetPower"],
    )

    storage_data = loaded["storageLevel"]
    net_power_data = loaded["currentNetPower"]

    storage_level_Ws = storage_data.data
    current_net_power_W = net_power_data.data

    if storage_level_Ws.ndim != 1:
        raise ValueError(
            f"Expected storageLevel data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {storage_level_Ws.shape}."
        )

    if current_net_power_W.ndim != 1:
        raise ValueError(
            f"Expected currentNetPower data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {current_net_power_W.shape}."
        )

    battery_energy_percent = _compute_battery_energy_percent(
        storage_level_Ws=storage_level_Ws,
        bat_storage_capacity_Wh=bat_storage_capacity_Wh,
    )

    t_storage_h = (
        np.arange(storage_data.n_samples)
        * storage_data.dt_s
        * storage_data.stride
        / 3600.0
    )
    t_power_h = (
        np.arange(net_power_data.n_samples)
        * net_power_data.dt_s
        * net_power_data.stride
        / 3600.0
    )

    # Height ratios sum to 1 conceptually. Bottom gets 4.5 / 10 = 45%.
    fig, axs = mpl.subplots(
        3,
        1,
        sharex=False,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [2.75, 2.75, 4.5]},
    )
    fig.suptitle(f"Simple EPS overview with pointing mode for spacecraft #{sat_idx}")

    axs[0].plot(t_storage_h, battery_energy_percent, label="Battery energy")
    axs[0].set_title("Battery energy")
    axs[0].set_ylabel("Energy [%]")
    axs[0].set_ylim(-5.0, 105.0)

    axs[1].plot(t_power_h, current_net_power_W, label="Current net power")
    axs[1].set_title("Current net power")
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Power [W]")

    for ax in axs[:2]:
        ax.grid(True)
        ax.legend()

    # Load and plot pointing mode after plotting EPS fields. This helper loads
    # only pointingModeCode and releases it immediately after plotting.
    _plot_pointing_mode_on_axis(
        ax=axs[2],
        timestamp=timestamp,
        sat_idx=sat_idx,
    )

    mpl.tight_layout()

    print(
        f"Loaded EPS fields for sat_{sat_idx}: "
        f"storageLevel {storage_data.n_samples}/{storage_data.source_n_samples} samples, "
        f"currentNetPower {net_power_data.n_samples}/{net_power_data.source_n_samples} samples."
    )

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, f"sat_{sat_idx}_simple_eps_overview_with_pointing_mode")

    del (
        loaded,
        storage_data,
        net_power_data,
        storage_level_Ws,
        current_net_power_W,
        battery_energy_percent,
        t_storage_h,
        t_power_h,
    )
    gc.collect()


#############################
# Operational Mode Plotting #
#############################

def plot_single_satellite_pointing_mode_from_h5(timestamp: str, sat_idx: int) -> None:
    """
    Load and plot the integer-coded pointing mode for one spacecraft.

    This intentionally mirrors debug_plotting.plot_single_satellite_pointing_mode(),
    but reads directly from HDF5 and avoids constructing a full SpacecraftSimData
    object.
    """

    preserve_fields = {"pointingModeCode"} if PRESERVE_POINTING_MODE_TRANSITIONS else set()

    loaded = load_satellite_fields(
        timestamp=timestamp,
        sat_idx=sat_idx,
        field_names=["pointingModeCode"],
        preserve_change_fields=preserve_fields,
    )

    mode_data = loaded["pointingModeCode"]
    mode_code = mode_data.data.astype(int)

    if mode_data.source_indices is not None:
        # Transition-preserving compressed data. The source indices define the
        # true sample numbers in the original full signal.
        t_h = mode_data.source_indices * mode_data.dt_s / 3600.0
    else:
        # Ordinary stride-based loaded data.
        t_h = np.arange(mode_data.n_samples) * mode_data.dt_s * mode_data.stride / 3600.0

    fig, ax = mpl.subplots(figsize=(11, 4))
    ax.step(t_h, mode_code, where="post")

    ax.set_title(f"Pointing mode for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Pointing mode [-]")

    tick_values = sorted(INT_TO_POINTING_MODE.keys())
    tick_labels = [INT_TO_POINTING_MODE[value] for value in tick_values]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

    ax.grid(True)
    mpl.tight_layout()

    print(
        f"Loaded pointingModeCode for sat_{sat_idx}: "
        f"{mode_data.n_samples} plotted samples from "
        f"{mode_data.source_n_samples} stored samples."
    )

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, f"sat_{sat_idx}_pointing_mode")

    del loaded, mode_data, mode_code, t_h
    gc.collect()









#######################################################################################################################################################################################
############################################################### M U L T I P L E    S A T E L L I T E    P L O T T I N G ###############################################################
####################################################################################################################################################################################### 

def plot_all_formation_plots() -> None:
    plot_3D_RTN_leader_relative_pos_for_all_followers_from_h5(timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD)

def plot_all_multi_satellite_fuel_mass_plots() -> None:
    plot_multiple_satellites_fuel_mass_from_h5(timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD)

def plot_all_multi_satellites_simple_eps_overview() -> None:
    plot_multiple_satellites_simple_eps_overview_with_pointing_mode(
        timestamp=SINGLE_RUN_TIMESTAMP_TO_LOAD,
        bat_storage_capacity_Wh=BAT_STORAGE_CAPACITY_WH)


################################
# All follower formation plots #
################################

def plot_3D_RTN_leader_relative_pos_for_all_followers_from_h5(
    timestamp: str,
) -> None:
    """
    Load and plot leader-relative follower position trajectories expressed in
    the leader RTN frame.

    This mirrors debug_plotting.plot_3D_RTN_leader_relative_pos_for_all_followers(),
    but reads directly from HDF5 and loads one follower field at a time.

    Required HDF5 field for each follower:
        r_scB_leaderB_RTN/data with shape (n_samples, 3)
    """

    sat_indices = get_available_satellite_indices(timestamp)
    follower_indices = [sat_idx for sat_idx in sat_indices if sat_idx != 0]

    if len(follower_indices) == 0:
        raise ValueError(
            f"No follower satellite files found for timestamp {timestamp}. "
            f"Available satellite indices: {sat_indices}"
        )

    follower_colors = {
        1: "C1",  # standard mpl orange
        2: "C2",  # standard mpl green
    }

    fig = mpl.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Leader is located at origin in its own RTN frame.
    ax.scatter(
        0.0,
        0.0,
        0.0,
        marker="o",
        s=60,  # type: ignore[arg-type]
        label="Leader",
    )

    for sat_idx in follower_indices:
        loaded = load_satellite_fields(
            timestamp=timestamp,
            sat_idx=sat_idx,
            field_names=["r_scB_leaderB_RTN"],
        )

        rel_pos_data = loaded["r_scB_leaderB_RTN"]
        r_RTN = rel_pos_data.data

        if r_RTN.ndim != 2 or r_RTN.shape[1] != 3:
            raise ValueError(
                f"Expected r_scB_leaderB_RTN data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {r_RTN.shape}."
            )

        ax.plot(
            r_RTN[:, 0],
            r_RTN[:, 1],
            r_RTN[:, 2],
            color=follower_colors.get(sat_idx),
            label=f"Follower {sat_idx}",
        )

        print(
            f"Loaded r_scB_leaderB_RTN for sat_{sat_idx}: "
            f"{rel_pos_data.n_samples} plotted samples from "
            f"{rel_pos_data.source_n_samples} stored samples "
            f"(stride={rel_pos_data.stride})."
        )

        del loaded, rel_pos_data, r_RTN
        gc.collect()

    ax.set_title("Leader-relative position, expressed in RTN")
    ax.set_xlabel("Radial [m]")
    ax.set_ylabel("Along-track [m]")
    ax.set_zlabel("Cross-track [m]")  # type: ignore[attr-defined]
    ax.legend()
    ax.grid(True)

    mpl.tight_layout()

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, "rtn_leader_relative_position_3d")



#######################################
# Fuel consumption over time plotting #
#######################################

def plot_multiple_satellites_fuel_mass_from_h5(timestamp: str) -> None:
    """
    Load and plot fuel mass over time for every spacecraft in one single-run folder.

    This mirrors plot_single_satellite_fuel_mass_from_h5(), but loops over all
    available sat_*.h5 files in the run directory and plots one fuel-mass curve
    per spacecraft.
    """

    sat_indices = get_available_satellite_indices(timestamp)

    if len(sat_indices) == 0:
        raise ValueError(f"No satellite HDF5 files found for timestamp {timestamp}.")

    fig, ax = mpl.subplots(figsize=(10, 4))

    for sat_idx in sat_indices:
        loaded = load_satellite_fields(
            timestamp=timestamp,
            sat_idx=sat_idx,
            field_names=["fuelMass"],
        )

        fuel_data = loaded["fuelMass"]
        fuel_mass = fuel_data.data

        if fuel_mass.ndim != 1:
            raise ValueError(
                f"Expected fuelMass data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {fuel_mass.shape}."
            )

        t_min = (
            np.arange(fuel_data.n_samples)
            * fuel_data.dt_s
            * fuel_data.stride
            / 60.0
        )

        label = "Leader" if sat_idx == 0 else f"Follower {sat_idx}"

        ax.plot(
            t_min,
            fuel_mass,
            label=label,
        )

        print(
            f"Loaded fuelMass for sat_{sat_idx}: "
            f"{fuel_data.n_samples} plotted samples from "
            f"{fuel_data.source_n_samples} stored samples "
            f"(stride={fuel_data.stride})."
        )

        del loaded, fuel_data, fuel_mass, t_min
        gc.collect()

    ax.set_title("Fuel mass for all spacecraft")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Fuel mass [kg]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(fig, run_dir, "all_satellites_fuel_mass")






def plot_multiple_satellites_simple_eps_overview_with_pointing_mode(
    timestamp: str,
    bat_storage_capacity_Wh: float | None = BAT_STORAGE_CAPACITY_WH,
) -> None:
    """
    Plot simple EPS overview and pointing mode for every spacecraft.

    Top subplot:
        Battery energy [%] for each spacecraft.

    Middle subplot:
        Battery current net power [W] for each spacecraft.

    Bottom subplot:
        Pointing mode for each spacecraft. Uses the same transition-preserving
        logic as plot_single_satellite_pointing_mode_from_h5(), but draws all
        spacecraft into the same axis.

    The pointing-mode subplot is allocated approximately 45% of the vertical
    plotting space.
    """

    sat_indices = get_available_satellite_indices(timestamp)

    if len(sat_indices) == 0:
        raise ValueError(f"No satellite HDF5 files found for timestamp {timestamp}.")

    fig, axs = mpl.subplots(
        3,
        1,
        sharex=False,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2.75, 2.75, 4.5]},
    )
    fig.suptitle("Simple EPS overview with pointing mode for all spacecraft")

    for sat_idx in sat_indices:
        loaded = load_satellite_fields(
            timestamp=timestamp,
            sat_idx=sat_idx,
            field_names=["storageLevel", "currentNetPower"],
        )

        storage_data = loaded["storageLevel"]
        net_power_data = loaded["currentNetPower"]

        storage_level_Ws = storage_data.data
        current_net_power_W = net_power_data.data

        if storage_level_Ws.ndim != 1:
            raise ValueError(
                f"Expected storageLevel data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {storage_level_Ws.shape}."
            )

        if current_net_power_W.ndim != 1:
            raise ValueError(
                f"Expected currentNetPower data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {current_net_power_W.shape}."
            )

        battery_energy_percent = _compute_battery_energy_percent(
            storage_level_Ws=storage_level_Ws,
            bat_storage_capacity_Wh=bat_storage_capacity_Wh,
        )

        t_storage_h = (
            np.arange(storage_data.n_samples)
            * storage_data.dt_s
            * storage_data.stride
            / 3600.0
        )

        t_power_h = (
            np.arange(net_power_data.n_samples)
            * net_power_data.dt_s
            * net_power_data.stride
            / 3600.0
        )

        label = "Leader" if sat_idx == 0 else f"Follower {sat_idx}"

        axs[0].plot(
            t_storage_h,
            battery_energy_percent,
            label=label,
        )

        axs[1].plot(
            t_power_h,
            current_net_power_W,
            label=label,
        )

        print(
            f"Loaded EPS fields for sat_{sat_idx}: "
            f"storageLevel {storage_data.n_samples}/{storage_data.source_n_samples} samples, "
            f"currentNetPower {net_power_data.n_samples}/{net_power_data.source_n_samples} samples."
        )

        del (
            loaded,
            storage_data,
            net_power_data,
            storage_level_Ws,
            current_net_power_W,
            battery_energy_percent,
            t_storage_h,
            t_power_h,
        )
        gc.collect()

        # -------------------------
        # Pointing mode
        # -------------------------
        preserve_fields = {"pointingModeCode"} if PRESERVE_POINTING_MODE_TRANSITIONS else set()

        mode_loaded = load_satellite_fields(
            timestamp=timestamp,
            sat_idx=sat_idx,
            field_names=["pointingModeCode"],
            preserve_change_fields=preserve_fields,
        )

        mode_data = mode_loaded["pointingModeCode"]
        mode_code = mode_data.data.astype(int)

        if mode_data.source_indices is not None:
            t_mode_h = mode_data.source_indices * mode_data.dt_s / 3600.0
        else:
            t_mode_h = (
                np.arange(mode_data.n_samples)
                * mode_data.dt_s
                * mode_data.stride
                / 3600.0
            )

        axs[2].step(
            t_mode_h,
            mode_code,
            where="post",
            label=label,
        )

        print(
            f"Loaded pointingModeCode for sat_{sat_idx}: "
            f"{mode_data.n_samples} plotted samples from "
            f"{mode_data.source_n_samples} stored samples."
        )

        del mode_loaded, mode_data, mode_code, t_mode_h
        gc.collect()

    axs[0].set_title("Battery energy")
    axs[0].set_ylabel("Energy [%]")
    axs[0].set_ylim(-5.0, 105.0)

    axs[1].set_title("Current net power")
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Power [W]")

    axs[2].set_title("Pointing mode")
    axs[2].set_xlabel("Time [h]")
    axs[2].set_ylabel("Mode [-]")

    tick_values = sorted(INT_TO_POINTING_MODE.keys())
    tick_labels = [INT_TO_POINTING_MODE[value] for value in tick_values]
    axs[2].set_yticks(tick_values)
    axs[2].set_yticklabels(tick_labels)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()

    run_dir = get_single_run_dir(timestamp)
    save_or_show_plot(
        fig,
        run_dir,
        "all_satellites_simple_eps_overview_with_pointing_mode",
    )





########
# main #
########

def main() -> None:
    """Run selected post-processing plots."""

    print("hello from single_run_analysis_and_plotting")

    run_dir = get_single_run_dir(SINGLE_RUN_TIMESTAMP_TO_LOAD)
    print(f"Loading single-run data from: {run_dir}")

    
    
    # ============ Single satellite plots ============ # 
    plot_all_single_pointing_mode_plots()   
    # plot_all_single_satellite_fuel_plots()
    # plot_all_single_satellite_eps_plots()

    # ============ Multiple satellite plots ============ # 
    # plot_all_formation_plots()
    # plot_all_multi_satellite_fuel_mass_plots()
    # plot_all_multi_satellites_simple_eps_overview()

    if SHOW_PLOTS:
        mpl.show()

if __name__ == "__main__":
    main()
