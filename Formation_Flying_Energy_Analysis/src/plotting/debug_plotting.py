import math
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np

from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport

from object_definitions.SimData_def import SpacecraftSimData
from object_definitions.FswStack_def import INT_TO_POINTING_MODE


# Global plot sizing, time-axis, and save settings
PLOT_WIDTH_IN = 16.0
PLOT_HEIGHT_PER_SUBPLOT_IN = 2.6
PLOT_MIN_HEIGHT_IN = 3.6
TIME_AXIS_LABEL = "Time [h]"
PLOT_SAVE_DPI = 300
PLOT_SAVE_BBOX_INCHES = "tight"


def _get_timeseries_figsize(n_subplots: int) -> tuple[float, float]:
    """Return the standard figure size for time-series debug plots."""
    if n_subplots <= 0:
        raise ValueError("n_subplots must be positive.")

    return (PLOT_WIDTH_IN, max(PLOT_MIN_HEIGHT_IN, PLOT_HEIGHT_PER_SUBPLOT_IN * n_subplots))


def _sample_times_h(sample_data) -> np.ndarray:
    """Return the sample time vector in hours for a SimData time series."""
    return np.arange(sample_data.n_samples) * sample_data.dt_s / 3600.0


def _save_figure_if_requested(
    fig: mpl.Figure, # type: ignore
    save_plt: bool,
    plt_out_dir: Path,
    filename: str,
) -> None:
    """Save a figure to plt_out_dir if requested."""
    if not save_plt:
        return

    plt_out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plt_out_dir / filename,
        dpi=PLOT_SAVE_DPI,
        bbox_inches=PLOT_SAVE_BBOX_INCHES,
    )



# Orbital element difference settings
MU_EARTH_M3_S2 = 3.986004418e14
OE_EPS = 1e-12
OE_COMPACT_HEIGHT_PER_SUBPLOT_IN = 1.65

# Hard-coded formation-control OEd setpoint parameters.
# WARNING: These values are intentionally duplicated from the current FswStack
# station-keeping setup for debug-plot visualization only. Update these constants
# if the formation-control targetClassicOED definition is changed.
OE_TARGET_RHO_M = 400.0
OE_TARGET_A_REF_M = 6878137.0
OE_TARGET_EPS = OE_TARGET_RHO_M / OE_TARGET_A_REF_M


def _get_compact_timeseries_figsize(n_subplots: int) -> tuple[float, float]:
    """Return a slightly shorter figure size for dense comparison plots."""
    if n_subplots <= 0:
        raise ValueError("n_subplots must be positive.")

    return (PLOT_WIDTH_IN, max(PLOT_MIN_HEIGHT_IN, OE_COMPACT_HEIGHT_PER_SUBPLOT_IN * n_subplots))


def _wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _continuous_angular_difference(eval_angle_rad: np.ndarray, base_angle_rad: np.ndarray) -> np.ndarray:
    """Return a continuous angular difference eval - base in radians."""
    return np.unwrap(_wrap_to_pi(eval_angle_rad - base_angle_rad))


def _rv_to_classical_oe_series(
    r_N: np.ndarray,
    v_N: np.ndarray,
    mu: float = MU_EARTH_M3_S2,
) -> dict[str, np.ndarray]:
    """
    Convert an inertial position/velocity time series to classical orbital elements.

    Args:
        r_N: Position array with shape (n_samples, 3) in meters.
        v_N: Velocity array with shape (n_samples, 3) in meters per second.
        mu: Gravitational parameter in m^3/s^2.

    Returns:
        Dictionary containing a [m], e [-], i [rad], Omega [rad], omega [rad],
        and M [rad].
    """
    if r_N.ndim != 2 or r_N.shape[1] != 3:
        raise ValueError(f"Expected position data to have shape (n_samples, 3), got {r_N.shape}.")
    if v_N.ndim != 2 or v_N.shape[1] != 3:
        raise ValueError(f"Expected velocity data to have shape (n_samples, 3), got {v_N.shape}.")
    if r_N.shape[0] != v_N.shape[0]:
        raise ValueError(
            f"Position and velocity sample counts do not match: "
            f"r_N has {r_N.shape[0]} samples, v_N has {v_N.shape[0]} samples."
        )

    n_samples = r_N.shape[0]
    a_arr = np.zeros(n_samples, dtype=np.float64)
    e_arr = np.zeros(n_samples, dtype=np.float64)
    i_arr = np.zeros(n_samples, dtype=np.float64)
    Omega_arr = np.zeros(n_samples, dtype=np.float64)
    omega_arr = np.zeros(n_samples, dtype=np.float64)
    M_arr = np.zeros(n_samples, dtype=np.float64)

    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for sample_idx in range(n_samples):
        r_vec = r_N[sample_idx, :].astype(np.float64)
        v_vec = v_N[sample_idx, :].astype(np.float64)

        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)

        if r_norm < OE_EPS:
            raise ValueError(f"Position norm is too small at sample {sample_idx}.")

        h_vec = np.cross(r_vec, v_vec)
        h_norm = np.linalg.norm(h_vec)

        if h_norm < OE_EPS:
            raise ValueError(f"Angular momentum norm is too small at sample {sample_idx}.")

        n_vec = np.cross(k_hat, h_vec)
        n_norm = np.linalg.norm(n_vec)

        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r_norm
        e_norm = np.linalg.norm(e_vec)

        specific_energy = 0.5 * v_norm**2 - mu / r_norm
        if abs(specific_energy) < OE_EPS:
            a = np.inf
        else:
            a = -mu / (2.0 * specific_energy)

        inc = np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0))

        if n_norm > OE_EPS:
            Omega = np.arctan2(n_vec[1], n_vec[0]) % (2.0 * np.pi)
        else:
            Omega = 0.0

        if n_norm > OE_EPS and e_norm > OE_EPS:
            omega = np.arctan2(
                np.dot(np.cross(n_vec, e_vec), h_vec) / (n_norm * e_norm * h_norm),
                np.dot(n_vec, e_vec) / (n_norm * e_norm),
            ) % (2.0 * np.pi)
        else:
            omega = 0.0

        if e_norm > OE_EPS:
            true_anomaly = np.arctan2(
                np.dot(np.cross(e_vec, r_vec), h_vec) / (e_norm * r_norm * h_norm),
                np.dot(e_vec, r_vec) / (e_norm * r_norm),
            ) % (2.0 * np.pi)
        elif n_norm > OE_EPS:
            true_anomaly = np.arctan2(
                np.dot(np.cross(n_vec, r_vec), h_vec) / (n_norm * r_norm * h_norm),
                np.dot(n_vec, r_vec) / (n_norm * r_norm),
            ) % (2.0 * np.pi)
        else:
            true_anomaly = np.arctan2(r_vec[1], r_vec[0]) % (2.0 * np.pi)

        if e_norm < 1.0 - OE_EPS:
            eccentric_anomaly = 2.0 * np.arctan2(
                np.sqrt(1.0 - e_norm) * np.sin(true_anomaly / 2.0),
                np.sqrt(1.0 + e_norm) * np.cos(true_anomaly / 2.0),
            )
            eccentric_anomaly = eccentric_anomaly % (2.0 * np.pi)
            mean_anomaly = (eccentric_anomaly - e_norm * np.sin(eccentric_anomaly)) % (2.0 * np.pi)
        else:
            mean_anomaly = np.nan

        a_arr[sample_idx] = a
        e_arr[sample_idx] = e_norm
        i_arr[sample_idx] = inc
        Omega_arr[sample_idx] = Omega
        omega_arr[sample_idx] = omega
        M_arr[sample_idx] = mean_anomaly

    return {
        "a": a_arr,
        "e": e_arr,
        "i": i_arr,
        "Omega": Omega_arr,
        "omega": omega_arr,
        "M": M_arr,
    }


def _get_spacecraft_classical_oe(scSimData: SpacecraftSimData, sat_idx: int) -> dict[str, np.ndarray]:
    """Return classical orbital elements for one spacecraft from r_BN_N and v_BN_N."""
    r_N = scSimData.r_BN_N.data
    v_N = scSimData.v_BN_N.data

    if r_N.ndim != 2 or r_N.shape[1] != 3:
        raise ValueError(f"Expected r_BN_N.data for spacecraft #{sat_idx} to have shape (n_samples, 3), got {r_N.shape}.")
    if v_N.ndim != 2 or v_N.shape[1] != 3:
        raise ValueError(f"Expected v_BN_N.data for spacecraft #{sat_idx} to have shape (n_samples, 3), got {v_N.shape}.")
    if r_N.shape != v_N.shape:
        raise ValueError(
            f"Position and velocity shape mismatch for spacecraft #{sat_idx}: "
            f"r_BN_N {r_N.shape}, v_BN_N {v_N.shape}."
        )

    return _rv_to_classical_oe_series(r_N, v_N)


def _compute_follower_leader_oe_difference(
    leader_oe: dict[str, np.ndarray],
    follower_oe: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute OE differences using the same order as plot_leader_oe_diff: eval - base.

    Here the follower is the evaluated trajectory and the leader is the baseline,
    so the plotted difference is follower - leader.
    """
    return {
        "da": (follower_oe["a"] - leader_oe["a"]) / 1000.0,
        "de": follower_oe["e"] - leader_oe["e"],
        "di": np.rad2deg(_continuous_angular_difference(follower_oe["i"], leader_oe["i"])),
        "dOmega": np.rad2deg(_continuous_angular_difference(follower_oe["Omega"], leader_oe["Omega"])),
        "domega": np.rad2deg(_continuous_angular_difference(follower_oe["omega"], leader_oe["omega"])),
        "dM": np.rad2deg(_continuous_angular_difference(follower_oe["M"], leader_oe["M"])),
    }


def _get_hard_coded_oe_target_for_follower(follower_idx: int) -> dict[str, float]:
    """Return the hard-coded formation-control target OEd for one follower."""
    target_eps = follower_idx * OE_TARGET_EPS

    return {
        "da": 0.0,
        "de": target_eps,
        "di": np.rad2deg(target_eps),
        "dOmega": 0.0,
        "domega": 0.0,
        "dM": 0.0,
    }

# Global X-component vector colors
SIGMA_1_COLOR = "#1f77b4"  # blue
SIGMA_2_COLOR = "#2ca02c"  # green
SIGMA_3_COLOR = "#b59b3b"  # yellow-green / ochre
SIGMA_4_COLOR = "#9467bd"  # purple


def plot_all_formation_plots(save_plt: bool, plt_out_dir: Path, scSimDataList: list[SpacecraftSimData]) -> None:
    plot_3D_RTN_leader_relative_pos_for_all_followers(save_plt, plt_out_dir, scSimDataList)
    plot_RTN_component_leader_relative_pos_for_all_followers(save_plt, plt_out_dir, scSimDataList)
    plot_orbital_element_differences_for_all_followers(save_plt, plt_out_dir, scSimDataList)


def plot_all_thruster_fuel_plots(save_plt: bool, plt_out_dir: Path, scSimDataList: list[SpacecraftSimData]) -> None:
    plot_propulsion_sys_for_all_satellites(save_plt, plt_out_dir, scSimDataList)
    plot_executed_burns_and_fuel_mass_for_all_satellites(save_plt, plt_out_dir, scSimDataList)


def plot_all_per_satellite_GNC_plots(save_plt: bool, plt_out_dir: Path, scSimDataList: list[SpacecraftSimData], sat_idx: int) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_attitude_error(save_plt, plt_out_dir, scSimData, sat_idx)
    plot_single_satellite_attitude_reference(save_plt, plt_out_dir, scSimData, sat_idx)
    plot_single_satellite_rw_torques_and_speeds(save_plt, plt_out_dir, scSimData, sat_idx)


def plot_all_eps_plots(save_plt: bool, plt_out_dir: Path, scSimDataList: list[SpacecraftSimData], sat_idx: int, bat_storage_capacity_Wh: float) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_eps_power(save_plt, plt_out_dir, scSimData, sat_idx)
    plot_single_satellite_battery_energy_fraction(save_plt, plt_out_dir, scSimData, sat_idx, bat_storage_capacity_Wh)
    plot_single_satellite_eps_overview(save_plt, plt_out_dir, scSimData, sat_idx, bat_storage_capacity_Wh)


def plot_all_pointing_mode_plots(save_plt: bool, plt_out_dir: Path, scSimDataList: list[SpacecraftSimData], sat_idx: int) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_pointing_mode(save_plt, plt_out_dir, scSimData, sat_idx)

################################
# All follower formation plots #
################################

def plot_3D_RTN_leader_relative_pos_for_all_followers(
    save_plt: bool,
    plt_out_dir: Path,
    scSimDataList: list[SpacecraftSimData],
) -> None:
    """
    Plot leader-relative follower position trajectories expressed in the
    leader RTN frame.

    Assumes:
        - scSimDataList[0] is the leader
        - follower.r_scB_leaderB_RTN is populated
        - r_scB_leaderB_RTN.data has shape (n_samples, 3)
        - RTN component order is [Radial, Along-track, Cross-track]
    """

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig = mpl.figure(figsize=_get_timeseries_figsize(1))
    ax = fig.add_subplot(111, projection="3d")

    # Leader is located at origin in its own RTN frame
    ax.scatter(
        0.0,
        0.0,
        0.0,
        marker="o",
        s=60, # type: ignore
        label="Leader",
    )

    # Plot followers
    for sat_idx, scSimData in enumerate(scSimDataList[1:], start=1):
        rel_pos = scSimData.r_scB_leaderB_RTN

        if rel_pos is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no r_scB_leaderB_RTN data. "
                "Make sure _compute_RTN_leader_relative_states() has been called."
            )

        r_RTN = rel_pos.data

        if r_RTN.ndim != 2 or r_RTN.shape[1] != 3:
            raise ValueError(
                f"Expected r_scB_leaderB_RTN.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {r_RTN.shape}."
            )

        ax.plot(
            r_RTN[:, 0],
            r_RTN[:, 1],
            r_RTN[:, 2],
            label=f"Follower {sat_idx}",
        )

    ax.set_title("Leader-relative position, expressed in RTN")
    ax.set_xlabel("Radial [m]")
    ax.set_ylabel("Along-track [m]")
    ax.set_zlabel("Cross-track [m]") # type: ignore
    ax.legend()
    ax.grid(True)

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        "leader_relative_position_RTN_3D.png",
    )


def plot_RTN_component_leader_relative_pos_for_all_followers(
    save_plt: bool,
    plt_out_dir: Path,
    scSimDataList: list[SpacecraftSimData],
) -> None:
    """
    Plot component-wise leader-relative follower positions expressed in
    the leader RTN frame.

    Assumes:
        - scSimDataList[0] is the leader
        - follower.r_scB_leaderB_RTN is populated
        - r_scB_leaderB_RTN.data has shape (n_samples, 3)
        - RTN component order is [Radial, Along-track, Cross-track]
    """

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=_get_timeseries_figsize(3))
    fig.suptitle("Component-wise RTN relative position (r_follower - r_leader)")

    for sat_idx, scSimData in enumerate(scSimDataList[1:], start=1):
        rel_pos = scSimData.r_scB_leaderB_RTN

        if rel_pos is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no r_scB_leaderB_RTN data. "
                "Make sure _compute_RTN_leader_relative_states() has been called."
            )

        r_RTN = rel_pos.data

        if r_RTN.ndim != 2 or r_RTN.shape[1] != 3:
            raise ValueError(
                f"Expected r_scB_leaderB_RTN.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {r_RTN.shape}."
            )

        t_h = _sample_times_h(rel_pos)

        if len(t_h) != r_RTN.shape[0]:
            raise ValueError(
                f"Time vector length mismatch for spacecraft #{sat_idx}. "
                f"Got len(t_h)={len(t_h)}, but position data has "
                f"{r_RTN.shape[0]} samples."
            )

        color = f"C{sat_idx - 1}"

        axs[0].plot(t_h, r_RTN[:, 0], color=color, label=f"Follower {sat_idx}")
        axs[1].plot(t_h, r_RTN[:, 1], color=color, label=f"Follower {sat_idx}")
        axs[2].plot(t_h, r_RTN[:, 2], color=color, label=f"Follower {sat_idx}")

    axs[0].set_title("Radial (R)")
    axs[1].set_title("Along-track (T)")
    axs[2].set_title("Cross-track (N)")

    axs[0].set_ylabel("d_pos [m]")
    axs[1].set_ylabel("d_pos [m]")
    axs[2].set_ylabel("d_pos [m]")
    axs[2].set_xlabel(TIME_AXIS_LABEL)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        "leader_relative_position_RTN_components.png",
    )



def plot_orbital_element_differences_for_all_followers(
    save_plt: bool,
    plt_out_dir: Path,
    scSimDataList: list[SpacecraftSimData],
    leader_idx: int = 0,
    follower_indices: list[int] | None = None,
) -> None:
    """
    Plot classical orbital-element differences for one or more leader-follower pairs.

    This function does not use the Basilisk orbital-element conversion utilities.
    Instead, it computes classical orbital elements directly from r_BN_N and v_BN_N,
    using the same subtraction order as plot_leader_oe_diff in plot.py:

        evaluated - baseline

    For this leader-follower comparison, the follower is treated as the evaluated
    spacecraft and the leader is treated as the baseline spacecraft. Therefore,
    each plotted difference is follower - leader.

    Generated plots:
        1. One compact 6-subplot comparison figure with all follower-leader pairs.
        2. One single-axis figure per follower-leader pair containing all six OEd components.
    """
    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")
    if leader_idx < 0 or leader_idx >= len(scSimDataList):
        raise ValueError(f"leader_idx={leader_idx} is outside scSimDataList with {len(scSimDataList)} spacecraft.")

    if follower_indices is None:
        follower_indices = [idx for idx in range(len(scSimDataList)) if idx != leader_idx]

    if len(follower_indices) == 0:
        raise ValueError("No follower spacecraft were selected for orbital-element difference plotting.")

    leader = scSimDataList[leader_idx]
    t_h = _sample_times_h(leader.r_BN_N)
    leader_oe = _get_spacecraft_classical_oe(leader, leader_idx)

    oe_diffs_by_follower: dict[int, dict[str, np.ndarray]] = {}
    oe_targets_by_follower: dict[int, dict[str, float]] = {}

    for follower_idx in follower_indices:
        if follower_idx < 0 or follower_idx >= len(scSimDataList):
            raise ValueError(
                f"follower_idx={follower_idx} is outside scSimDataList with "
                f"{len(scSimDataList)} spacecraft."
            )
        if follower_idx == leader_idx:
            raise ValueError("follower_indices must not include leader_idx.")

        follower = scSimDataList[follower_idx]

        if follower.r_BN_N.dt_s != leader.r_BN_N.dt_s:
            raise ValueError(
                f"Position sample time mismatch for follower #{follower_idx}: "
                f"leader dt_s={leader.r_BN_N.dt_s}, follower dt_s={follower.r_BN_N.dt_s}."
            )
        if follower.r_BN_N.n_samples != leader.r_BN_N.n_samples:
            raise ValueError(
                f"Position sample count mismatch for follower #{follower_idx}: "
                f"leader n_samples={leader.r_BN_N.n_samples}, "
                f"follower n_samples={follower.r_BN_N.n_samples}."
            )

        follower_oe = _get_spacecraft_classical_oe(follower, follower_idx)
        oe_diffs_by_follower[follower_idx] = _compute_follower_leader_oe_difference(
            leader_oe,
            follower_oe,
        )
        oe_targets_by_follower[follower_idx] = _get_hard_coded_oe_target_for_follower(follower_idx)

    element_order = ["da", "de", "di", "dOmega", "domega", "dM"]
    axis_labels = {
        "da": r"$\Delta a$ [km]",
        "de": r"$\Delta e$ [-]",
        "di": r"$\Delta i$ [deg]",
        "dOmega": r"$\Delta \Omega$ [deg]",
        "domega": r"$\Delta \omega$ [deg]",
        "dM": r"$\Delta M$ [deg]",
    }
    line_labels = {
        "da": r"$\Delta a$ [km]",
        "de": r"$\Delta e$ [-]",
        "di": r"$\Delta i$ [deg]",
        "dOmega": r"$\Delta \Omega$ [deg]",
        "domega": r"$\Delta \omega$ [deg]",
        "dM": r"$\Delta M$ [deg]",
    }

    # ------------------------------------------------------------------
    # Plot 1: all follower-leader pairs, one OEd component per subplot
    # ------------------------------------------------------------------
    fig, axs = mpl.subplots(6, 1, sharex=True, figsize=_get_compact_timeseries_figsize(6))
    fig.suptitle("Orbital Element Difference Comparison for Leader and Followers")

    for follower_idx, oe_diffs in oe_diffs_by_follower.items():
        label = f"Follower {follower_idx} - Leader {leader_idx}"
        for ax_idx, element_name in enumerate(element_order):
            axs[ax_idx].plot(t_h, oe_diffs[element_name], label=label)

    # WARNING: Desired OEd lines are hard-coded to match the current FswStack
    # targetClassicOED setup: de = sat_idx*eps, di = sat_idx*eps, others = 0.
    zero_target_elements = {"da", "dOmega", "domega", "dM"}
    zero_target_label_used = False

    for ax_idx, element_name in enumerate(element_order):
        if element_name in zero_target_elements:
            axs[ax_idx].axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Desired zero OEd" if not zero_target_label_used else None,
            )
            zero_target_label_used = True
        elif element_name in {"de", "di"}:
            for follower_idx, oe_targets in oe_targets_by_follower.items():
                axs[ax_idx].axhline(
                    oe_targets[element_name],
                    color=f"C{follower_idx - 1}",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Follower {follower_idx} desired {axis_labels[element_name]}",
                )

    for ax_idx, element_name in enumerate(element_order):
        axs[ax_idx].set_ylabel(axis_labels[element_name])
        axs[ax_idx].grid(True)

    axs[-1].set_xlabel(TIME_AXIS_LABEL)
    axs[0].legend(loc="best")

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        "orbital_element_differences_all_followers_components.png",
    )

    # ------------------------------------------------------------------
    # Plot 2: one single-axis all-component figure per follower-leader pair
    # ------------------------------------------------------------------
    for follower_idx, oe_diffs in oe_diffs_by_follower.items():
        fig_pair, ax_pair = mpl.subplots(figsize=_get_timeseries_figsize(1))

        for element_name in element_order:
            ax_pair.plot(t_h, oe_diffs[element_name], label=line_labels[element_name])

        # WARNING: Desired OEd lines are hard-coded to match the current FswStack
        # targetClassicOED setup: de = sat_idx*eps, di = sat_idx*eps, others = 0.
        oe_targets = oe_targets_by_follower[follower_idx]
        ax_pair.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Desired zero OEd",
        )
        ax_pair.axhline(
            oe_targets["de"],
            color="C1",
            linestyle="--",
            linewidth=1.0,
            label=r"Desired $\Delta e$",
        )
        ax_pair.axhline(
            oe_targets["di"],
            color="C2",
            linestyle="--",
            linewidth=1.0,
            label=r"Desired $\Delta i$ [deg]",
        )

        ax_pair.set_title(
            f"Orbital element differences: Follower {follower_idx} - Leader {leader_idx}"
        )
        ax_pair.set_xlabel(TIME_AXIS_LABEL)
        ax_pair.set_ylabel("OE difference [mixed units]")
        ax_pair.grid(True)
        ax_pair.legend(loc="best")

        mpl.tight_layout()
        _save_figure_if_requested(
            fig_pair,
            save_plt,
            plt_out_dir,
            f"orbital_element_differences_follower_{follower_idx}_leader_{leader_idx}_combined.png",
        )


def plot_orbital_element_difference(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: list[SpacecraftSimData],
    sat_idx: int,
) -> None:
    print("NEW OE DIFF PLOT")
    """
    Plot osculating orbital-element differences between the leader and one spacecraft.

    The leader is assumed to be ``scSimData[0]``. The selected spacecraft is
    ``scSimData[sat_idx]``. The plotted element-difference vector follows the
    Basilisk station-keeping example convention:

        [da/a, de, di, dOmega, domega, dM]

    where the angular differences are wrapped to [-pi, pi].
    """

    if len(scSimData) == 0:
        raise ValueError("scSimData is empty.")

    if sat_idx <= 0:
        raise ValueError(
            "sat_idx must refer to a follower spacecraft. "
            "The leader is assumed to be at index 0."
        )

    if sat_idx >= len(scSimData):
        raise ValueError(
            f"sat_idx={sat_idx} is outside scSimData with "
            f"{len(scSimData)} spacecraft."
        )

    leader = scSimData[0]
    follower = scSimData[sat_idx]

    r_leader_N = leader.r_BN_N.data
    v_leader_N = leader.v_BN_N.data
    r_follower_N = follower.r_BN_N.data
    v_follower_N = follower.v_BN_N.data

    if r_leader_N.ndim != 2 or r_leader_N.shape[1] != 3:
        raise ValueError(
            f"Expected leader r_BN_N.data to have shape (n_samples, 3), "
            f"got {r_leader_N.shape}."
        )

    if v_leader_N.shape != r_leader_N.shape:
        raise ValueError(
            f"Leader position and velocity shapes do not match. "
            f"Got r_BN_N {r_leader_N.shape}, v_BN_N {v_leader_N.shape}."
        )

    if r_follower_N.shape != r_leader_N.shape:
        raise ValueError(
            f"Position shape mismatch for spacecraft #{sat_idx}. "
            f"Leader r_BN_N shape {r_leader_N.shape}, "
            f"follower r_BN_N shape {r_follower_N.shape}."
        )

    if v_follower_N.shape != v_leader_N.shape:
        raise ValueError(
            f"Velocity shape mismatch for spacecraft #{sat_idx}. "
            f"Leader v_BN_N shape {v_leader_N.shape}, "
            f"follower v_BN_N shape {v_follower_N.shape}."
        )

    if leader.r_BN_N.dt_s != follower.r_BN_N.dt_s:
        raise ValueError(
            f"Position sample time mismatch for spacecraft #{sat_idx}. "
            f"Leader dt_s={leader.r_BN_N.dt_s}, "
            f"follower dt_s={follower.r_BN_N.dt_s}."
        )

    if leader.v_BN_N.dt_s != follower.v_BN_N.dt_s:
        raise ValueError(
            f"Velocity sample time mismatch for spacecraft #{sat_idx}. "
            f"Leader dt_s={leader.v_BN_N.dt_s}, "
            f"follower dt_s={follower.v_BN_N.dt_s}."
        )

    sim_length = r_leader_N.shape[0]
    oed = np.empty((sim_length, 6), dtype=np.float64)
    mu = getattr(orbitalMotion, "MU_EARTH", 3.986004418e5)

    for i in range(sim_length):
        oe_leader = orbitalMotion.rv2elem(mu, r_leader_N[i], v_leader_N[i])
        oe_follower = orbitalMotion.rv2elem(mu, r_follower_N[i], v_follower_N[i])

        oed[i, 0] = (oe_follower.a - oe_leader.a) / oe_leader.a # type: ignore
        oed[i, 1] = oe_follower.e - oe_leader.e # type: ignore
        oed[i, 2] = oe_follower.i - oe_leader.i # type: ignore
        oed[i, 3] = oe_follower.Omega - oe_leader.Omega # type: ignore
        oed[i, 4] = oe_follower.omega - oe_leader.omega # type: ignore

        E_leader = orbitalMotion.f2E(oe_leader.f, oe_leader.e) # type: ignore
        E_follower = orbitalMotion.f2E(oe_follower.f, oe_follower.e) # type: ignore
        M_leader = orbitalMotion.E2M(E_leader, oe_leader.e) # type: ignore
        M_follower = orbitalMotion.E2M(E_follower, oe_follower.e) # type: ignore
        oed[i, 5] = M_follower - M_leader

        for j in range(3, 6):
            if oed[i, j] > math.pi:
                oed[i, j] -= 2.0 * math.pi
            if oed[i, j] < -math.pi:
                oed[i, j] += 2.0 * math.pi

    t_h = _sample_times_h(leader.r_BN_N)

    fig, axs = mpl.subplots(6, 1, sharex=True, figsize=_get_timeseries_figsize(6))
    fig.suptitle(f"Orbital-element differences: spacecraft #{sat_idx} relative to leader")

    labels = [
        r"$\Delta a / a$ [-]",
        r"$\Delta e$ [-]",
        r"$\Delta i$ [rad]",
        r"$\Delta \Omega$ [rad]",
        r"$\Delta \omega$ [rad]",
        r"$\Delta M$ [rad]",
    ]

    for i, ax in enumerate(axs):
        ax.plot(t_h, oed[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axs[-1].set_xlabel(TIME_AXIS_LABEL)

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"orbital_element_difference_sat_{sat_idx}.png",
    )






#####################################
# All satellite thruster fuel plots #
#####################################

def plot_propulsion_sys_for_all_satellites(
    save_plt: bool,
    plt_out_dir: Path,
    scSimDataList: list[SpacecraftSimData],
) -> None:
    """
    Plot thrust force magnitude and fuel mass for all spacecraft.

    Top subplot:
        - thrust force magnitude from thrustForce_B

    Bottom subplot:
        - fuel mass from fuelMass
    """

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=_get_timeseries_figsize(3))
    fig.suptitle("Thrust force magnitude and fuel mass for all spacecraft")

    for sat_idx, scSimData in enumerate(scSimDataList):
        if sat_idx == 0:
            color = f"C{len(scSimDataList)}"
            label = "Leader"
        else:
            color = f"C{sat_idx - 1}"
            label = f"Follower {sat_idx}"

        # -------------------------
        # Thrust magnitude subplot
        # -------------------------
        thrust_data = scSimData.thrustForce_B

        if thrust_data is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no thrustForce_B data. "
                "This plot requires data_mode='debug'."
            )

        thrust_B = thrust_data.data

        if thrust_B.ndim != 2 or thrust_B.shape[1] != 3:
            raise ValueError(
                f"Expected thrustForce_B.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {thrust_B.shape}."
            )

        t_thrust_h = _sample_times_h(thrust_data)
        thrust_mag = np.linalg.norm(thrust_B, axis=1)

        axs[0].plot(
            t_thrust_h,
            thrust_mag,
            color=color,
            label=label,
        )

        # -------------------------
        # Thrust torque magnitude subplot
        # -------------------------
        torque_data = scSimData.thrustTorquePntB_B

        if torque_data is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no thrustTorquePntB_B data. "
                "This plot requires data_mode='debug'."
            )

        torque_B = torque_data.data

        if torque_B.ndim != 2 or torque_B.shape[1] != 3:
            raise ValueError(
                f"Expected thrustTorquePntB_B.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {torque_B.shape}."
            )

        t_torque_h = _sample_times_h(torque_data)
        torque_mag = np.linalg.norm(torque_B, axis=1)

        axs[1].plot(
            t_torque_h,
            torque_mag,
            color=color,
            label=label,
        )


        # -------------------------
        # Fuel mass subplot
        # -------------------------
        fuel_data = scSimData.fuelMass
        fuel_mass = fuel_data.data

        if fuel_mass.ndim != 1:
            raise ValueError(
                f"Expected fuelMass.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {fuel_mass.shape}."
            )

        t_fuel_h = _sample_times_h(fuel_data)

        axs[2].plot(
            t_fuel_h,
            fuel_mass,
            color=color,
            label=label,
        )

    axs[0].set_title("Thrust force magnitude")
    axs[0].set_xlabel(TIME_AXIS_LABEL)
    axs[0].set_ylabel("Thrust [N]")

    axs[1].set_title("Thrust torque magnitude")
    axs[1].set_xlabel(TIME_AXIS_LABEL)
    axs[1].set_ylabel("Torque [Nm]")

    axs[2].set_title("Fuel mass")
    axs[2].set_xlabel(TIME_AXIS_LABEL)
    axs[2].set_ylabel("Fuel mass [kg]")
    axs[2].ticklabel_format(axis="y", style="plain", useOffset=False)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        "propulsion_system_all_spacecraft.png",
    )








def plot_executed_burns_and_fuel_mass_for_all_satellites(
    save_plt: bool,
    plt_out_dir: Path,
    scSimDataList: list[SpacecraftSimData],
) -> None:
    """
    Plot executed burns and fuel mass for all spacecraft.

    This is the two-subplot version of plot_propulsion_sys_for_all_satellites():
        - thrust force magnitude from thrustForce_B
        - fuel mass from fuelMass

    The thrust torque subplot is intentionally omitted.
    """

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig, axs = mpl.subplots(2, 1, sharex=True, figsize=_get_timeseries_figsize(2))
    fig.suptitle("Thrust force magnitude and fuel mass for all spacecraft")

    for sat_idx, scSimData in enumerate(scSimDataList):
        if sat_idx == 0:
            color = f"C{len(scSimDataList)}"
            label = "Leader"
        else:
            color = f"C{sat_idx - 1}"
            label = f"Follower {sat_idx}"

        # -------------------------
        # Thrust magnitude subplot
        # -------------------------
        thrust_data = scSimData.thrustForce_B

        if thrust_data is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no thrustForce_B data. "
                "This plot requires data_mode='debug'."
            )

        thrust_B = thrust_data.data

        if thrust_B.ndim != 2 or thrust_B.shape[1] != 3:
            raise ValueError(
                f"Expected thrustForce_B.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples, 3), got {thrust_B.shape}."
            )

        t_thrust_h = _sample_times_h(thrust_data)
        thrust_mag = np.linalg.norm(thrust_B, axis=1)

        axs[0].plot(
            t_thrust_h,
            thrust_mag,
            color=color,
            label=label,
        )

        # -------------------------
        # Fuel mass subplot
        # -------------------------
        fuel_data = scSimData.fuelMass
        fuel_mass = fuel_data.data

        if fuel_mass.ndim != 1:
            raise ValueError(
                f"Expected fuelMass.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {fuel_mass.shape}."
            )

        t_fuel_h = _sample_times_h(fuel_data)

        axs[1].plot(
            t_fuel_h,
            fuel_mass,
            color=color,
            label=label,
        )

    axs[0].set_title("Thrust force magnitude")
    axs[0].set_ylabel("Thrust [N]")

    axs[1].set_title("Fuel mass")
    axs[1].set_xlabel(TIME_AXIS_LABEL)
    axs[1].set_ylabel("Fuel mass [kg]")
    axs[1].ticklabel_format(axis="y", style="plain", useOffset=False)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        "executed_burns_and_fuel_mass_all_spacecraft.png",
    )



###########################
# Per satellite GNC plots #
###########################

def plot_single_satellite_attitude_error(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    """
    Plot attitude tracking error sigma_BR for a single spacecraft.
    """

    sigma_data = scSimData.sigma_BR

    if sigma_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no sigma_BR data. "
            "This plot requires data_mode='debug'."
        )

    sigma_BR = sigma_data.data

    if sigma_BR.ndim != 2 or sigma_BR.shape[1] != 3:
        raise ValueError(
            f"Expected sigma_BR.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples, 3), got {sigma_BR.shape}."
        )

    t_h = _sample_times_h(sigma_data)

    fig, ax = mpl.subplots(figsize=_get_timeseries_figsize(1))

    ax.plot(t_h, sigma_BR[:, 0], color=SIGMA_1_COLOR, label=r"$\sigma_1$")
    ax.plot(t_h, sigma_BR[:, 1], color=SIGMA_2_COLOR, label=r"$\sigma_2$")
    ax.plot(t_h, sigma_BR[:, 2], color=SIGMA_3_COLOR, label=r"$\sigma_3$")

    ax.set_title(f"Attitude tracking error for spacecraft #{sat_idx}")
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel(r"Error $\sigma_{B/R}$")
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"attitude_error_sat_{sat_idx}.png",
    )


def plot_single_satellite_attitude_reference(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    """
    Plot attitude reference sigma_RN for a single spacecraft.
    """

    sigma_data = scSimData.sigma_RN

    if sigma_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no sigma_RN data. "
            "This plot requires data_mode='debug'."
        )

    sigma_RN = sigma_data.data

    if sigma_RN.ndim != 2 or sigma_RN.shape[1] != 3:
        raise ValueError(
            f"Expected sigma_RN.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples, 3), got {sigma_RN.shape}."
        )

    t_h = _sample_times_h(sigma_data)

    fig, ax = mpl.subplots(figsize=_get_timeseries_figsize(1))

    ax.plot(t_h, sigma_RN[:, 0], color=SIGMA_1_COLOR, label=r"$\sigma_1$")
    ax.plot(t_h, sigma_RN[:, 1], color=SIGMA_2_COLOR, label=r"$\sigma_2$")
    ax.plot(t_h, sigma_RN[:, 2], color=SIGMA_3_COLOR, label=r"$\sigma_3$")

    ax.set_title(f"Attitude reference for spacecraft #{sat_idx}")
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel(r"$\sigma_{R/N}$")
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"attitude_reference_sat_{sat_idx}.png",
    )


def plot_single_satellite_rw_torques_and_speeds(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    """
    Plot reaction wheel speeds and torques for a single spacecraft.
    """

    rw_speed_data = scSimData.rwOmega
    rw_actual_torque_data = scSimData.rwUCurrent
    rw_cmd_torque_data = scSimData.cmdMotorTorque

    if rw_speed_data is None:
        raise ValueError(f"Spacecraft #{sat_idx} has no rwOmega data. This plot requires data_mode='debug'.")
    if rw_actual_torque_data is None:
        raise ValueError(f"Spacecraft #{sat_idx} has no rwUCurrent data. This plot requires data_mode='debug'.")
    if rw_cmd_torque_data is None:
        raise ValueError(f"Spacecraft #{sat_idx} has no cmdMotorTorque data. This plot requires data_mode='debug'.")

    rwOmega_rads = rw_speed_data.data
    rwUCurrent = rw_actual_torque_data.data
    cmdMotorTorque = rw_cmd_torque_data.data

    if rwOmega_rads.ndim != 2:
        raise ValueError(f"Expected rwOmega.data to have shape (n_samples, numRWs), got {rwOmega_rads.shape}.")
    if rwUCurrent.ndim != 2:
        raise ValueError(f"Expected rwUCurrent.data to have shape (n_samples, numRWs), got {rwUCurrent.shape}.")
    if cmdMotorTorque.ndim != 2:
        raise ValueError(f"Expected cmdMotorTorque.data to have shape (n_samples, numRWs), got {cmdMotorTorque.shape}.")

    numRWs = rwOmega_rads.shape[1]

    if rwUCurrent.shape[1] != numRWs:
        raise ValueError(
            f"RW count mismatch: rwOmega has {numRWs} wheels, "
            f"but rwUCurrent has {rwUCurrent.shape[1]}."
        )
    if cmdMotorTorque.shape[1] != numRWs:
        raise ValueError(
            f"RW count mismatch: rwOmega has {numRWs} wheels, "
            f"but cmdMotorTorque has {cmdMotorTorque.shape[1]}."
        )

    rw_colors = [SIGMA_1_COLOR, SIGMA_2_COLOR, SIGMA_3_COLOR, SIGMA_4_COLOR]

    if numRWs > len(rw_colors):
        raise ValueError(
            f"This plotting function currently supports up to {len(rw_colors)} RWs, "
            f"but spacecraft #{sat_idx} has {numRWs}."
        )

    t_speed_h = _sample_times_h(rw_speed_data)
    t_actual_torque_h = _sample_times_h(rw_actual_torque_data)
    t_cmd_torque_h = _sample_times_h(rw_cmd_torque_data)

    rwOmega_rpm = rwOmega_rads * 60.0 / (2.0 * np.pi)

    fig, axs = mpl.subplots(2, 1, sharex=True, figsize=_get_timeseries_figsize(2))
    fig.suptitle(f"Reaction wheel speed and torque for spacecraft #{sat_idx}")

    for rw_idx in range(numRWs):
        color = rw_colors[rw_idx]

        axs[0].plot(
            t_speed_h,
            rwOmega_rpm[:, rw_idx],
            color=color,
            label=fr"RW {rw_idx + 1}",
        )

        axs[1].plot(
            t_actual_torque_h,
            rwUCurrent[:, rw_idx],
            color=color,
            linestyle="-",
            label=fr"RW {rw_idx + 1} actual",
        )

        axs[1].plot(
            t_cmd_torque_h,
            cmdMotorTorque[:, rw_idx],
            color=color,
            linestyle=":",
            label=fr"RW {rw_idx + 1} cmd",
        )

    axs[0].set_ylabel("Speed [RPM]")
    axs[1].set_ylabel("Torque [Nm]")
    axs[1].set_xlabel(TIME_AXIS_LABEL)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"reaction_wheel_speed_torque_sat_{sat_idx}.png",
    )






###########################
# Per satellite EPS plots #
###########################

def plot_single_satellite_eps_power(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    """
    Plot EPS power terms for a single spacecraft.

    This is mainly intended to verify switchable power sinks, such as
    the communication system, payload, and propulsion-related power sinks.
    """

    battery_power_data = scSimData.currentNetPower
    obc_power_data = scSimData.obcNetPower
    bat_heat_power_data = scSimData.batHeatNetPower
    com_power_data = scSimData.comNetPower
    pay_power_data = scSimData.payNetPower
    prop_idle_power_data = scSimData.propIdleNetPower
    prop_heat_power_data = scSimData.propHeatNetPower
    prop_thr_power_data = scSimData.propThrNetPower
    solar_power_data = scSimData.solarPanelNetPower

    if obc_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no obcNetPower data. "
            "This plot requires data_mode='debug'."
        )

    if bat_heat_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no batHeatNetPower data. "
            "Make sure dynModel.batHeatPowerSinkRecorder is extracted in SimData_def.py."
        )

    if com_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no comNetPower data. "
            "Make sure dynModel.comPowerSinkRecorder is extracted in SimData_def.py."
        )

    if pay_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no payNetPower data. "
            "Make sure dynModel.payPowerSinkRecorder is extracted in SimData_def.py."
        )

    if prop_idle_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no propIdleNetPower data. "
            "Make sure dynModel.propIdlePowerSinkRecorder is extracted in SimData_def.py."
        )

    if prop_heat_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no propHeatNetPower data. "
            "Make sure dynModel.propHeatPowerSinkRecorder is extracted in SimData_def.py."
        )

    if prop_thr_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no propThrNetPower data. "
            "Make sure dynModel.propThrPowerSinkRecorder is extracted in SimData_def.py."
        )

    if solar_power_data is None:
        raise ValueError(
            f"Spacecraft #{sat_idx} has no solarPanelNetPower data. "
            "This plot requires data_mode='debug'."
        )

    battery_net_power = battery_power_data.data
    obc_net_power = obc_power_data.data
    bat_heat_net_power = bat_heat_power_data.data
    com_net_power = com_power_data.data
    pay_net_power = pay_power_data.data
    prop_idle_net_power = prop_idle_power_data.data
    prop_heat_net_power = prop_heat_power_data.data
    prop_thr_net_power = prop_thr_power_data.data
    solar_panel_net_power = solar_power_data.data

    sink_series = [
        ("currentNetPower", battery_net_power),
        ("obcNetPower", obc_net_power),
        ("batHeatNetPower", bat_heat_net_power),
        ("comNetPower", com_net_power),
        ("payNetPower", pay_net_power),
        ("propIdleNetPower", prop_idle_net_power),
        ("propHeatNetPower", prop_heat_net_power),
        ("propThrNetPower", prop_thr_net_power),
    ]

    for name, data in sink_series:
        if data.ndim != 1:
            raise ValueError(
                f"Expected {name}.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {data.shape}."
            )

    if solar_panel_net_power.ndim != 2:
        raise ValueError(
            f"Expected solarPanelNetPower.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples, numSPs), got {solar_panel_net_power.shape}."
        )

    t_bat_h = _sample_times_h(battery_power_data)
    t_obc_h = _sample_times_h(obc_power_data)
    t_bat_heat_h = _sample_times_h(bat_heat_power_data)
    t_com_h = _sample_times_h(com_power_data)
    t_pay_h = _sample_times_h(pay_power_data)
    t_prop_idle_h = _sample_times_h(prop_idle_power_data)
    t_prop_heat_h = _sample_times_h(prop_heat_power_data)
    t_prop_thr_h = _sample_times_h(prop_thr_power_data)
    t_solar_h = _sample_times_h(solar_power_data)

    total_solar_power = np.sum(solar_panel_net_power, axis=1)

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=_get_timeseries_figsize(3))
    fig.suptitle(f"EPS power for spacecraft #{sat_idx}")

    # -------------------------
    # Battery net power
    # -------------------------
    axs[0].plot(t_bat_h, battery_net_power, label="Battery net power")
    axs[0].set_title("Battery net power")
    axs[0].set_ylabel("Power [W]")

    # -------------------------
    # Power sinks
    # -------------------------
    prop_color = "C4"

    axs[1].plot(t_obc_h, obc_net_power, label="OBC")
    axs[1].plot(t_bat_heat_h, bat_heat_net_power, label="Battery heater")
    axs[1].plot(t_com_h, com_net_power, label="Comms")
    axs[1].plot(t_pay_h, pay_net_power, label="Payload")

    axs[1].plot(
        t_prop_idle_h,
        prop_idle_net_power,
        color=prop_color,
        linestyle="-",
        label="Propulsion idle",
    )
    axs[1].plot(
        t_prop_heat_h,
        prop_heat_net_power,
        color=prop_color,
        linestyle="--",
        label="Propulsion heating",
    )
    axs[1].plot(
        t_prop_thr_h,
        prop_thr_net_power,
        color=prop_color,
        linestyle=":",
        label="Propulsion thrusting",
    )

    axs[1].set_title("Power sinks")
    axs[1].set_ylabel("Power [W]")

    # -------------------------
    # Solar generation
    # -------------------------
    axs[2].plot(t_solar_h, total_solar_power, label="Total solar generation")
    axs[2].set_title("Solar power generation")
    axs[2].set_xlabel(TIME_AXIS_LABEL)
    axs[2].set_ylabel("Power [W]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"eps_power_sat_{sat_idx}.png",
    )


def plot_single_satellite_battery_energy_fraction(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
    bat_storage_capacity_Wh: float,
) -> None:
    """
    Plot stored battery energy as a fraction of maximum stored energy
    for a single spacecraft.
    """

    storage_data = scSimData.storageLevel

    storage_level_Ws = storage_data.data

    if storage_level_Ws.ndim != 1:
        raise ValueError(
            f"Expected storageLevel.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples,), got {storage_level_Ws.shape}."
        )

    if bat_storage_capacity_Wh <= 0.0:
        raise ValueError(
            f"Battery storage capacity must be positive, got "
            f"{bat_storage_capacity_Wh} Wh."
        )

    bat_storage_capacity_Ws = bat_storage_capacity_Wh * 3600.0
    storage_fraction = storage_level_Ws / bat_storage_capacity_Ws

    t_h = _sample_times_h(storage_data)

    fig, ax = mpl.subplots(figsize=_get_timeseries_figsize(1))

    ax.plot(t_h, storage_fraction, label="Battery energy fraction")

    ax.set_title(f"Battery stored energy fraction for spacecraft #{sat_idx}")
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Stored energy fraction [-]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"battery_energy_fraction_sat_{sat_idx}.png",
    )


def plot_single_satellite_eps_overview(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
    bat_storage_capacity_Wh: float,
) -> None:
    """
    Plot combined EPS overview for a single spacecraft.

    Top:
        Battery stored energy percentage.

    2nd:
        Total solar generation, total power consumption, and net battery power.

    3rd:
        Individual solar panel generation and total solar generation.

    4th:
        Individual power sinks and total consumption.
    """

    # -------------------------
    # Fetch mandatory data
    # -------------------------
    storage_data = scSimData.storageLevel
    battery_power_data = scSimData.currentNetPower

    # -------------------------
    # Fetch debug EPS data
    # -------------------------
    obc_power_data = scSimData.obcNetPower
    bat_heat_power_data = scSimData.batHeatNetPower
    com_power_data = scSimData.comNetPower
    pay_power_data = scSimData.payNetPower
    prop_idle_power_data = scSimData.propIdleNetPower
    prop_heat_power_data = scSimData.propHeatNetPower
    prop_thr_power_data = scSimData.propThrNetPower
    rw_power_data = scSimData.rwNetPower
    solar_power_data = scSimData.solarPanelNetPower

    required_debug_data = {
        "obcNetPower": obc_power_data,
        "batHeatNetPower": bat_heat_power_data,
        "comNetPower": com_power_data,
        "payNetPower": pay_power_data,
        "propIdleNetPower": prop_idle_power_data,
        "propHeatNetPower": prop_heat_power_data,
        "propThrNetPower": prop_thr_power_data,
        "rwNetPower": rw_power_data,
        "solarPanelNetPower": solar_power_data,
    }

    for name, data in required_debug_data.items():
        if data is None:
            raise ValueError(
                f"Spacecraft #{sat_idx} has no {name} data. "
                "This plot requires data_mode='debug' and corresponding recorder extraction."
            )

    assert obc_power_data is not None
    assert bat_heat_power_data is not None
    assert com_power_data is not None
    assert pay_power_data is not None
    assert prop_idle_power_data is not None
    assert prop_heat_power_data is not None
    assert prop_thr_power_data is not None
    assert rw_power_data is not None
    assert solar_power_data is not None

    # -------------------------
    # Extract arrays
    # -------------------------
    storage_level_Ws = storage_data.data
    battery_net_power = battery_power_data.data

    obc_net_power = obc_power_data.data
    bat_heat_net_power = bat_heat_power_data.data
    com_net_power = com_power_data.data
    pay_net_power = pay_power_data.data
    prop_idle_net_power = prop_idle_power_data.data
    prop_heat_net_power = prop_heat_power_data.data
    prop_thr_net_power = prop_thr_power_data.data
    rw_net_power = rw_power_data.data
    solar_panel_net_power = solar_power_data.data

    # -------------------------
    # Validate shapes
    # -------------------------
    one_dim_series = {
        "storageLevel": storage_level_Ws,
        "currentNetPower": battery_net_power,
        "obcNetPower": obc_net_power,
        "batHeatNetPower": bat_heat_net_power,
        "comNetPower": com_net_power,
        "payNetPower": pay_net_power,
        "propIdleNetPower": prop_idle_net_power,
        "propHeatNetPower": prop_heat_net_power,
        "propThrNetPower": prop_thr_net_power,
    }

    for name, data in one_dim_series.items():
        if data.ndim != 1:
            raise ValueError(
                f"Expected {name}.data for spacecraft #{sat_idx} "
                f"to have shape (n_samples,), got {data.shape}."
            )

    if rw_net_power.ndim != 2:
        raise ValueError(
            f"Expected rwNetPower.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples, numRWs), got {rw_net_power.shape}."
        )

    if solar_panel_net_power.ndim != 2:
        raise ValueError(
            f"Expected solarPanelNetPower.data for spacecraft #{sat_idx} "
            f"to have shape (n_samples, numSPs), got {solar_panel_net_power.shape}."
        )

    if bat_storage_capacity_Wh <= 0.0:
        raise ValueError(
            f"Battery storage capacity must be positive, got "
            f"{bat_storage_capacity_Wh} Wh."
        )

    # -------------------------
    # Derived quantities
    # -------------------------
    bat_storage_capacity_Ws = bat_storage_capacity_Wh * 3600.0
    storage_percent = 100.0 * storage_level_Ws / bat_storage_capacity_Ws

    total_generation = np.sum(solar_panel_net_power, axis=1)
    total_rw_power = np.sum(rw_net_power, axis=1)

    # Sinks and RW powers are negative in the raw Basilisk convention.
    # Plot total consumption as a positive magnitude.
    total_consumption = -1.0 * (
        obc_net_power
        + bat_heat_net_power
        + com_net_power
        + pay_net_power
        + prop_idle_net_power
        + prop_heat_net_power
        + prop_thr_net_power
        + total_rw_power
    )

    # -------------------------
    # Time vectors
    # -------------------------
    t_storage_h = _sample_times_h(storage_data)
    t_battery_power_h = _sample_times_h(battery_power_data)

    t_obc_h = _sample_times_h(obc_power_data)
    t_bat_heat_h = _sample_times_h(bat_heat_power_data)
    t_com_h = _sample_times_h(com_power_data)
    t_pay_h = _sample_times_h(pay_power_data)
    t_prop_idle_h = _sample_times_h(prop_idle_power_data)
    t_prop_heat_h = _sample_times_h(prop_heat_power_data)
    t_prop_thr_h = _sample_times_h(prop_thr_power_data)
    t_rw_h = _sample_times_h(rw_power_data)
    t_solar_h = _sample_times_h(solar_power_data)

    # -------------------------
    # Plot style constants
    # -------------------------
    total_generation_color = "green"
    total_consumption_color = "red"
    net_power_color = "blue"
    propulsion_color = "C4"

    # -------------------------
    # Plot
    # -------------------------
    fig, axs = mpl.subplots(4, 1, sharex=True, figsize=_get_timeseries_figsize(4))
    fig.suptitle(f"EPS overview for spacecraft #{sat_idx}")

    # -------------------------
    # Top: battery percentage
    # -------------------------
    axs[0].plot(t_storage_h, storage_percent, label="Battery energy")
    axs[0].set_title("Remaining battery energy")
    axs[0].set_ylabel("Energy [%]")
    axs[0].set_ylim(-5.0, 105.0)

    # -------------------------
    # 2nd: aggregate EPS balance
    # -------------------------
    axs[1].plot(
        t_solar_h,
        total_generation,
        color=total_generation_color,
        linestyle="--",
        label="Total generation",
    )
    axs[1].plot(
        t_obc_h,
        total_consumption,
        color=total_consumption_color,
        linestyle="--",
        label="Total consumption",
    )
    axs[1].plot(
        t_battery_power_h,
        battery_net_power,
        color=net_power_color,
        linestyle="-",
        label="Net power",
    )
    axs[1].set_title("EPS power balance")
    axs[1].set_ylabel("Power [W]")

    # -------------------------
    # 3rd: solar generation breakdown
    # -------------------------
    num_solar_panels = solar_panel_net_power.shape[1]

    # Matplotlib's default cycle usually gives C0, C1, C2, ... .
    # Skip C2 because it is commonly green, so individual panels are not green.
    non_green_solar_colors = [
        "C0", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
    ]

    for sp_idx in range(num_solar_panels):
        color = non_green_solar_colors[sp_idx % len(non_green_solar_colors)]
        axs[2].plot(
            t_solar_h,
            solar_panel_net_power[:, sp_idx],
            color=color,
            linestyle="-",
            label=f"Solar panel {sp_idx + 1}",
        )

    axs[2].plot(
        t_solar_h,
        total_generation,
        color=total_generation_color,
        linestyle="-",
        linewidth=2.0,
        label="Total generation",
    )
    axs[2].set_title("Solar power generation")
    axs[2].set_ylabel("Power [W]")

    # -------------------------
    # 4th: sink breakdown
    # -------------------------
    axs[3].plot(
        t_obc_h,
        total_consumption,
        color=total_consumption_color,
        linestyle="-",
        linewidth=2.0,
        label="Total consumption",
    )

    axs[3].plot(t_obc_h, obc_net_power, label="OBC")
    axs[3].plot(t_bat_heat_h, bat_heat_net_power, label="Battery heater")
    axs[3].plot(t_com_h, com_net_power, label="Communication system")
    axs[3].plot(t_pay_h, pay_net_power, label="Payload")

    axs[3].plot(
        t_prop_idle_h,
        prop_idle_net_power,
        color=propulsion_color,
        linestyle="-",
        label="Propulsion idle",
    )
    axs[3].plot(
        t_prop_heat_h,
        prop_heat_net_power,
        color=propulsion_color,
        linestyle="-",
        label="Propulsion heating",
    )
    axs[3].plot(
        t_prop_thr_h,
        prop_thr_net_power,
        color=propulsion_color,
        linestyle="-",
        label="Propulsion thrusting",
    )

    axs[3].plot(t_rw_h, total_rw_power, label="Reaction wheels total")

    axs[3].set_title("Power sink breakdown")
    axs[3].set_xlabel(TIME_AXIS_LABEL)
    axs[3].set_ylabel("Power [W]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"eps_overview_sat_{sat_idx}.png",
    )




######################
# Pointing Mode Plot #
######################

def plot_single_satellite_pointing_mode(
    save_plt: bool,
    plt_out_dir: Path,
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    mode_data = scSimData.pointingModeCode
    mode_code = mode_data.data

    t_h = _sample_times_h(mode_data)

    fig, ax = mpl.subplots(figsize=_get_timeseries_figsize(1))
    ax.step(t_h, mode_code, where="post")

    ax.set_title(f"Operational mode for spacecraft #{sat_idx}")
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("Pointing mode [-]")

    tick_values = list(INT_TO_POINTING_MODE.keys())
    tick_labels = [mode.value for mode in INT_TO_POINTING_MODE.values()]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

    ax.grid(True)
    mpl.tight_layout()
    _save_figure_if_requested(
        fig,
        save_plt,
        plt_out_dir,
        f"pointing_mode_sat_{sat_idx}.png",
    )