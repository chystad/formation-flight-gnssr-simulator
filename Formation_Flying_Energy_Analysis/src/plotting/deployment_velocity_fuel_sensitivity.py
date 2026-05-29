"""
Deployment velocity formation acquisition fuel sensitivity processing.

Place this file in:
    Formation_Flying_Energy_Analysis/src/plotting/deployment_velocity_fuel_sensitivity.py

The script only reads the HDF5 data files. It does not import Basilisk.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import matplotlib.pyplot as mpl
import numpy as np


# =============================================================================
# Hard-coded formation-acquisition definition
# =============================================================================

# WARNING:
# These formation target constants are intentionally hard-coded to match the
# current concentric-circle formation-control setup in FswStack._setup_desired_OE_difference().
# If the formation-control target is changed in FswStack, these values must be
# updated here as well.
RHO_M = 400.0
A_REF_M = 6878137.0
EPS = RHO_M / A_REF_M

# The first OEd component follows the spacecraftReconfig convention da/a [-].
DELTA_A_OVER_A_REFERENCE = 0.0
DELTA_A_TOLERANCE = 0.00001

# The eccentricity-difference component is dimensionless.
DELTA_E_TOLERANCE = 0.0002

# The inclination-difference component is in radians.
DELTA_I_TOLERANCE_RAD = 0.00002

# Formation must remain inside tolerance for this duration before acquisition is accepted.
ACQUISITION_DWELL_TIME_H = 0.5

# Earth gravitational parameter [m^3/s^2].
MU_EARTH = 3.986004418e14

# Plot settings.
PLOT_WIDTH_IN = 16.0
PLOT_HEIGHT_IN = 6.0
PLOT_DPI = 300
LEGEND_LOC = "upper right"

# Inclusive end time for all time-history plots.
# Set this to a value less than or equal to the simulated time horizon.
PLOT_END_TIME_H = 15.0

# Battery maximum stored energy used for EPS percentage plots.
# The storageLevel time history is assumed to be stored in Wh for this post-processing script.
BATTERY_MAX_ENERGY = 100.0  # [Wh] Update to match the simulated spacecraft battery capacity.


@dataclass(frozen=True)
class SampledArray:
    data: np.ndarray
    dt_s: float
    n_samples: int

    @property
    def time_h(self) -> np.ndarray:
        return np.arange(self.n_samples, dtype=np.float64) * self.dt_s / 3600.0


@dataclass(frozen=True)
class SatelliteCoreData:
    r_BN_N: SampledArray
    v_BN_N: SampledArray
    fuelMass: SampledArray
    storageLevel: SampledArray
    currentNetPower: SampledArray
    pointingModeCode: SampledArray


@dataclass(frozen=True)
class BurnAcquisitionMetrics:
    number_of_burns_until_acq: int
    average_thruster_on_time_until_acq_s: float
    average_attitude_tracking_error_during_burns_deg: float


@dataclass(frozen=True)
class LinearAcquisitionModelParameters:
    """Least-squares linear models for acquisition sensitivity metrics."""
    m_acq_slope_g_per_mps: float
    m_acq_intercept_g: float
    t_acq_slope_h_per_mps: float
    t_acq_intercept_h: float


@dataclass
class RunProcessingResult:
    run_dir: Path
    run_name: str
    case_label: str
    plot_label: str
    differential_deployment_speed_mps: float
    follower_sat_idx: int

    acquired: bool
    t_acq_h: float
    t_dwell_pass_h: float
    m_acq_g: float

    number_of_burns_until_acq: int
    average_thruster_on_time_until_acq_s: float
    average_attitude_tracking_error_during_burns_deg: float

    total_fuel_consumed_g: float
    initial_fuel_mass_kg: float
    final_fuel_mass_kg: float

    min_abs_da_over_a_error: float
    min_abs_de_error: float
    min_abs_di_error_rad: float
    target_de: float
    target_di_rad: float

    time_state_h: np.ndarray
    da_over_a: np.ndarray
    de: np.ndarray
    di_rad: np.ndarray
    dOmega_rad: np.ndarray
    domega_rad: np.ndarray
    dM_rad: np.ndarray
    r_rel_RTN_m: np.ndarray

    time_fuel_h: np.ndarray
    cumulative_fuel_consumed_g: np.ndarray

    time_eps_h: np.ndarray
    battery_energy_percent: np.ndarray
    current_net_power_w: np.ndarray

    time_mode_h: np.ndarray
    pointing_mode_code: np.ndarray

    @property
    def m_acq_kg(self) -> float:
        return self.m_acq_g / 1000.0 if np.isfinite(self.m_acq_g) else float("nan")

    @property
    def total_fuel_consumed_kg(self) -> float:
        return self.total_fuel_consumed_g / 1000.0


def _read_sampled_array(h5_path: Path, group_name: str) -> SampledArray:
    """Read one SimDataWriter SampledData group from a satellite HDF5 file."""
    with h5py.File(h5_path, "r") as h5:
        if group_name not in h5:
            raise KeyError(f"Group '{group_name}' was not found in '{h5_path}'.")

        grp = h5[group_name]
        data = np.asarray(grp["data"])
        dt_s = float(np.asarray(grp["dt_s"]))
        n_samples = int(np.asarray(grp["n_samples"]))

    if data.shape[0] != n_samples:
        raise ValueError(
            f"Sample-count mismatch for '{group_name}' in '{h5_path}'. "
            f"data.shape[0]={data.shape[0]}, n_samples={n_samples}."
        )

    return SampledArray(data=data, dt_s=dt_s, n_samples=n_samples)



def _try_read_sampled_array(h5_path: Path, group_name: str) -> Optional[SampledArray]:
    """Read one sampled HDF5 group if it exists; otherwise return None."""
    with h5py.File(h5_path, "r") as h5:
        if group_name not in h5:
            return None

    return _read_sampled_array(h5_path, group_name)


def _pointing_mode_is_burn(mode_value: object) -> bool:
    """Return True if a mode-switching CSV value represents PointingMode.BURN."""
    mode_text = str(mode_value).strip()
    return mode_text == "PointingMode.BURN" or mode_text.endswith(".BURN") or mode_text == "BURN"


def _read_burn_intervals_h(run_dir: Path, follower_sat_idx: int) -> list[tuple[float, float]]:
    """
    Read burn intervals from FSW<sat_idx>_mode_switching.csv.

    A burn starts when newPointingMode enters PointingMode.BURN and ends when
    oldPointingMode leaves PointingMode.BURN.
    """
    csv_path = run_dir / f"FSW{follower_sat_idx}_mode_switching.csv"

    if not csv_path.exists():
        return []

    intervals: list[tuple[float, float]] = []
    active_start_h: Optional[float] = None

    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {"currentSimMins", "oldPointingMode", "newPointingMode"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Mode-switching CSV '{csv_path}' is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        for row in reader:
            raw_time_min = row.get("currentSimMins")
            if raw_time_min is None or raw_time_min == "":
                continue

            try:
                time_h = float(raw_time_min) / 60.0
            except ValueError:
                continue

            old_is_burn = _pointing_mode_is_burn(row.get("oldPointingMode"))
            new_is_burn = _pointing_mode_is_burn(row.get("newPointingMode"))

            if new_is_burn and not old_is_burn and active_start_h is None:
                active_start_h = time_h

            if old_is_burn and not new_is_burn and active_start_h is not None:
                stop_h = time_h
                if stop_h >= active_start_h:
                    intervals.append((active_start_h, stop_h))
                active_start_h = None

    return intervals


def _read_sigma_br_sampled_array(run_dir: Path, follower_sat_idx: int) -> Optional[SampledArray]:
    """Read sigma_BR from the follower HDF5 file if it is available."""
    h5_path = run_dir / f"sat_{follower_sat_idx}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find expected satellite file: '{h5_path}'.")

    sigma_BR = _try_read_sampled_array(h5_path, "sigma_BR")

    if sigma_BR is not None:
        if sigma_BR.data.ndim != 2 or sigma_BR.data.shape[1] != 3:
            raise ValueError(
                f"Expected sigma_BR in '{h5_path}' to have shape (n, 3), "
                f"got {sigma_BR.data.shape}."
            )

    return sigma_BR


def _mrp_error_angle_deg(sigma_br: np.ndarray) -> np.ndarray:
    """
    Convert MRP attitude-error vectors to principal rotation-angle magnitudes [deg].

    The result is the absolute attitude tracking error angle corresponding to
    each sigma_BR sample.
    """
    sigma_norm = np.linalg.norm(sigma_br.astype(np.float64), axis=1)
    angle_rad = 4.0 * np.arctan(sigma_norm)
    return np.rad2deg(np.abs(angle_rad))


def compute_burn_metrics_until_acquisition(
    run_dir: Path,
    follower_sat_idx: int,
    t_acq_h: float,
) -> BurnAcquisitionMetrics:
    """
    Compute burn-count, mean burn duration, and mean attitude tracking error up to t_acq.

    Burn intervals are selected by start time: every burn with start time <= t_acq
    is included. Therefore, if t_acq occurs during a burn, the full burn interval
    is still included when its stop time is present in the mode-switching log.
    """
    if not np.isfinite(t_acq_h):
        return BurnAcquisitionMetrics(
            number_of_burns_until_acq=0,
            average_thruster_on_time_until_acq_s=float("nan"),
            average_attitude_tracking_error_during_burns_deg=float("nan"),
        )

    burn_intervals_h = _read_burn_intervals_h(run_dir, follower_sat_idx)
    included_intervals_h = [
        (start_h, stop_h)
        for start_h, stop_h in burn_intervals_h
        if start_h <= t_acq_h
    ]

    if len(included_intervals_h) == 0:
        return BurnAcquisitionMetrics(
            number_of_burns_until_acq=0,
            average_thruster_on_time_until_acq_s=float("nan"),
            average_attitude_tracking_error_during_burns_deg=float("nan"),
        )

    durations_s = np.array(
        [(stop_h - start_h) * 3600.0 for start_h, stop_h in included_intervals_h],
        dtype=np.float64,
    )

    average_thruster_on_time_s = float(np.nanmean(durations_s))

    sigma_BR = _read_sigma_br_sampled_array(run_dir, follower_sat_idx)
    if sigma_BR is None:
        average_attitude_error_deg = float("nan")
    else:
        sigma_time_h = sigma_BR.time_h
        attitude_error_deg = _mrp_error_angle_deg(sigma_BR.data)

        burn_error_samples: list[np.ndarray] = []
        for start_h, stop_h in included_intervals_h:
            mask = (sigma_time_h >= start_h) & (sigma_time_h <= stop_h)
            if np.any(mask):
                burn_error_samples.append(attitude_error_deg[mask])

        if len(burn_error_samples) == 0:
            average_attitude_error_deg = float("nan")
        else:
            average_attitude_error_deg = float(np.nanmean(np.concatenate(burn_error_samples)))

    return BurnAcquisitionMetrics(
        number_of_burns_until_acq=len(included_intervals_h),
        average_thruster_on_time_until_acq_s=average_thruster_on_time_s,
        average_attitude_tracking_error_during_burns_deg=average_attitude_error_deg,
    )

def load_satellite_core_data(run_dir: Path, sat_idx: int) -> SatelliteCoreData:
    """Load the minimum per-satellite data needed for this analysis."""
    h5_path = run_dir / f"sat_{sat_idx}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find expected satellite file: '{h5_path}'.")

    r_BN_N = _read_sampled_array(h5_path, "r_BN_N")
    v_BN_N = _read_sampled_array(h5_path, "v_BN_N")
    fuelMass = _read_sampled_array(h5_path, "fuelMass")
    storageLevel = _read_sampled_array(h5_path, "storageLevel")
    currentNetPower = _read_sampled_array(h5_path, "currentNetPower")
    pointingModeCode = _read_sampled_array(h5_path, "pointingModeCode")

    _validate_state_shapes(h5_path, r_BN_N, v_BN_N)

    return SatelliteCoreData(
        r_BN_N=r_BN_N,
        v_BN_N=v_BN_N,
        fuelMass=fuelMass,
        storageLevel=storageLevel,
        currentNetPower=currentNetPower,
        pointingModeCode=pointingModeCode,
    )


def _validate_state_shapes(h5_path: Path, r_BN_N: SampledArray, v_BN_N: SampledArray) -> None:
    if r_BN_N.data.ndim != 2 or r_BN_N.data.shape[1] != 3:
        raise ValueError(
            f"Expected r_BN_N in '{h5_path}' to have shape (n, 3), "
            f"got {r_BN_N.data.shape}."
        )

    if v_BN_N.data.ndim != 2 or v_BN_N.data.shape[1] != 3:
        raise ValueError(
            f"Expected v_BN_N in '{h5_path}' to have shape (n, 3), "
            f"got {v_BN_N.data.shape}."
        )

    if r_BN_N.data.shape != v_BN_N.data.shape:
        raise ValueError(
            f"Position/velocity shape mismatch in '{h5_path}': "
            f"r_BN_N={r_BN_N.data.shape}, v_BN_N={v_BN_N.data.shape}."
        )

    if r_BN_N.n_samples != v_BN_N.n_samples:
        raise ValueError(
            f"Position/velocity sample-count mismatch in '{h5_path}': "
            f"r_BN_N={r_BN_N.n_samples}, v_BN_N={v_BN_N.n_samples}."
        )


def wrap_to_pi(angle_rad: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def angular_difference_rad(eval_angle_rad: np.ndarray, base_angle_rad: np.ndarray) -> np.ndarray:
    """Compute a continuous angular difference eval - base."""
    return np.unwrap(wrap_to_pi(eval_angle_rad - base_angle_rad))


def rv_to_classical_oe_series(
    r_BN_N_m: np.ndarray,
    v_BN_N_mps: np.ndarray,
    mu_m3_s2: float = MU_EARTH,
) -> dict[str, np.ndarray]:
    """Convert inertial position/velocity histories to classical orbital elements."""
    if r_BN_N_m.shape != v_BN_N_mps.shape:
        raise ValueError(
            f"Position and velocity shapes do not match: "
            f"r={r_BN_N_m.shape}, v={v_BN_N_mps.shape}."
        )

    if r_BN_N_m.ndim != 2 or r_BN_N_m.shape[1] != 3:
        raise ValueError(f"Expected state arrays with shape (n, 3), got {r_BN_N_m.shape}.")

    n = r_BN_N_m.shape[0]
    eps_num = 1e-12

    a_arr = np.empty(n, dtype=np.float64)
    e_arr = np.empty(n, dtype=np.float64)
    i_arr = np.empty(n, dtype=np.float64)
    Omega_arr = np.empty(n, dtype=np.float64)
    omega_arr = np.empty(n, dtype=np.float64)
    M_arr = np.empty(n, dtype=np.float64)

    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for idx in range(n):
        r_vec = np.asarray(r_BN_N_m[idx], dtype=np.float64)
        v_vec = np.asarray(v_BN_N_mps[idx], dtype=np.float64)

        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)

        if r_norm < eps_num:
            raise ValueError(f"Position norm is too small at sample {idx}.")

        h_vec = np.cross(r_vec, v_vec)
        h_norm = np.linalg.norm(h_vec)

        if h_norm < eps_num:
            raise ValueError(f"Angular-momentum norm is too small at sample {idx}.")

        n_vec = np.cross(k_hat, h_vec)
        n_norm = np.linalg.norm(n_vec)

        e_vec = np.cross(v_vec, h_vec) / mu_m3_s2 - r_vec / r_norm
        e_norm = np.linalg.norm(e_vec)

        specific_energy = 0.5 * v_norm**2 - mu_m3_s2 / r_norm
        a = np.inf if abs(specific_energy) < eps_num else -mu_m3_s2 / (2.0 * specific_energy)

        inc = np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0))

        if n_norm > eps_num:
            Omega = np.arctan2(n_vec[1], n_vec[0]) % (2.0 * np.pi)
        else:
            Omega = 0.0

        if n_norm > eps_num and e_norm > eps_num:
            omega = np.arctan2(
                np.dot(np.cross(n_vec, e_vec), h_vec) / (n_norm * e_norm * h_norm),
                np.dot(n_vec, e_vec) / (n_norm * e_norm),
            ) % (2.0 * np.pi)
        else:
            omega = 0.0

        if e_norm > eps_num:
            true_anomaly = np.arctan2(
                np.dot(np.cross(e_vec, r_vec), h_vec) / (e_norm * r_norm * h_norm),
                np.dot(e_vec, r_vec) / (e_norm * r_norm),
            ) % (2.0 * np.pi)
        else:
            # Circular fallback: use argument of latitude where possible.
            if n_norm > eps_num:
                true_anomaly = np.arctan2(
                    np.dot(np.cross(n_vec, r_vec), h_vec) / (n_norm * r_norm * h_norm),
                    np.dot(n_vec, r_vec) / (n_norm * r_norm),
                ) % (2.0 * np.pi)
            else:
                true_anomaly = np.arctan2(r_vec[1], r_vec[0]) % (2.0 * np.pi)

        if e_norm < 1.0 - eps_num:
            eccentric_anomaly = 2.0 * np.arctan2(
                np.sqrt(1.0 - e_norm) * np.sin(true_anomaly / 2.0),
                np.sqrt(1.0 + e_norm) * np.cos(true_anomaly / 2.0),
            )
            eccentric_anomaly = eccentric_anomaly % (2.0 * np.pi)
            mean_anomaly = (eccentric_anomaly - e_norm * np.sin(eccentric_anomaly)) % (2.0 * np.pi)
        else:
            mean_anomaly = np.nan

        a_arr[idx] = a
        e_arr[idx] = e_norm
        i_arr[idx] = inc
        Omega_arr[idx] = Omega
        omega_arr[idx] = omega
        M_arr[idx] = mean_anomaly

    return {
        "a": a_arr,
        "e": e_arr,
        "i": i_arr,
        "Omega": Omega_arr,
        "omega": omega_arr,
        "M": M_arr,
    }


def compute_follower_minus_leader_oed(
    leader: SatelliteCoreData,
    follower: SatelliteCoreData,
) -> dict[str, np.ndarray]:
    """
    Compute follower-leader orbital-element difference.

    Subtraction order:
        follower - leader

    The first OEd component follows the spacecraftReconfig convention:
        da_over_a = (a_follower - a_leader) / a_leader
    """
    if leader.r_BN_N.n_samples != follower.r_BN_N.n_samples:
        raise ValueError(
            "Leader and follower state sample counts do not match: "
            f"leader={leader.r_BN_N.n_samples}, follower={follower.r_BN_N.n_samples}."
        )

    if not math.isclose(leader.r_BN_N.dt_s, follower.r_BN_N.dt_s):
        raise ValueError(
            "Leader and follower state sample times do not match: "
            f"leader={leader.r_BN_N.dt_s}, follower={follower.r_BN_N.dt_s}."
        )

    leader_oe = rv_to_classical_oe_series(leader.r_BN_N.data, leader.v_BN_N.data)
    follower_oe = rv_to_classical_oe_series(follower.r_BN_N.data, follower.v_BN_N.data)

    return {
        "da_over_a": (follower_oe["a"] - leader_oe["a"]) / leader_oe["a"],
        "de": follower_oe["e"] - leader_oe["e"],
        "di": angular_difference_rad(follower_oe["i"], leader_oe["i"]),
        "dOmega": angular_difference_rad(follower_oe["Omega"], leader_oe["Omega"]),
        "domega": angular_difference_rad(follower_oe["omega"], leader_oe["omega"]),
        "dM": angular_difference_rad(follower_oe["M"], leader_oe["M"]),
    }


def compute_follower_minus_leader_rtn_position(
    leader: SatelliteCoreData,
    follower: SatelliteCoreData,
) -> np.ndarray:
    """
    Compute follower position relative to leader, expressed in leader RTN frame.

    RTN component order is [Radial, Along-track, Cross-track].
    """
    r_leader_N = leader.r_BN_N.data.astype(np.float64)
    v_leader_N = leader.v_BN_N.data.astype(np.float64)
    r_follower_N = follower.r_BN_N.data.astype(np.float64)

    if r_leader_N.shape != r_follower_N.shape:
        raise ValueError("Leader and follower position arrays must have the same shape.")

    r_norm = np.linalg.norm(r_leader_N, axis=1)
    h_leader_N = np.cross(r_leader_N, v_leader_N)
    h_norm = np.linalg.norm(h_leader_N, axis=1)

    if np.any(r_norm == 0.0):
        raise ValueError("Cannot construct RTN frame because at least one leader position norm is zero.")
    if np.any(h_norm == 0.0):
        raise ValueError("Cannot construct RTN frame because at least one leader angular momentum norm is zero.")

    R_hat_N = r_leader_N / r_norm[:, None]
    N_hat_N = h_leader_N / h_norm[:, None]
    T_hat_N = np.cross(N_hat_N, R_hat_N)

    C_RTN_N = np.stack((R_hat_N, T_hat_N, N_hat_N), axis=1)
    r_rel_N = r_follower_N - r_leader_N
    return np.einsum("nij,nj->ni", C_RTN_N, r_rel_N)


def find_acquisition_time_h(
    time_h: np.ndarray,
    oed: dict[str, np.ndarray],
    follower_sat_idx: int,
) -> tuple[bool, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the first acquisition time using the hard-coded tolerance and dwell definition.

    Instantaneous in-formation definition:
        abs(da/a - 0) <= DELTA_A_TOLERANCE
        abs(de - follower_sat_idx * EPS) <= DELTA_E_TOLERANCE
        abs(di - follower_sat_idx * EPS) <= DELTA_I_TOLERANCE_RAD

    Dwell definition:
        The instantaneous condition must remain true continuously for
        ACQUISITION_DWELL_TIME_H. If the condition first passes the dwell timer
        at t_dwell_pass_h, then t_acq_h = t_dwell_pass_h - ACQUISITION_DWELL_TIME_H.
    """
    target_da_over_a = DELTA_A_OVER_A_REFERENCE
    target_de = follower_sat_idx * EPS
    target_di_rad = follower_sat_idx * EPS

    da_error = oed["da_over_a"] - target_da_over_a
    de_error = oed["de"] - target_de
    di_error = oed["di"] - target_di_rad

    acquired_mask = (
        np.abs(da_error) <= DELTA_A_TOLERANCE
    ) & (
        np.abs(de_error) <= DELTA_E_TOLERANCE
    ) & (
        np.abs(di_error) <= DELTA_I_TOLERANCE_RAD
    )

    if len(time_h) == 0 or not np.any(acquired_mask):
        return False, float("nan"), float("nan"), da_error, de_error, di_error

    dwell_start_h: Optional[float] = None
    last_inside_idx: Optional[int] = None

    for idx, inside in enumerate(acquired_mask):
        if inside:
            if dwell_start_h is None:
                dwell_start_h = float(time_h[idx])
            last_inside_idx = idx

            elapsed_h = float(time_h[idx]) - dwell_start_h
            if elapsed_h >= ACQUISITION_DWELL_TIME_H:
                t_dwell_pass_h = float(time_h[idx])
                t_acq_h = t_dwell_pass_h - ACQUISITION_DWELL_TIME_H
                return True, t_acq_h, t_dwell_pass_h, da_error, de_error, di_error
        else:
            dwell_start_h = None
            last_inside_idx = None

    # If all samples inside tolerance, but total dwell time is not long enough.
    _ = last_inside_idx
    return False, float("nan"), float("nan"), da_error, de_error, di_error


def compute_cumulative_fuel_consumed_g(fuel_mass_kg: np.ndarray) -> np.ndarray:
    """Convert fuel-mass time history into cumulative consumed propellant mass [g]."""
    if fuel_mass_kg.ndim != 1:
        raise ValueError(f"Expected fuel_mass_kg to be 1D, got shape {fuel_mass_kg.shape}.")

    cumulative_g = (float(fuel_mass_kg[0]) - fuel_mass_kg.astype(np.float64)) * 1000.0
    return np.maximum(cumulative_g, 0.0)


def interpolate_fuel_consumed_at_time_g(
    time_fuel_h: np.ndarray,
    cumulative_fuel_consumed_g: np.ndarray,
    t_acq_h: float,
) -> float:
    """Interpolate cumulative fuel consumption at the acquisition time."""
    if not np.isfinite(t_acq_h):
        return float("nan")

    if t_acq_h < time_fuel_h[0] or t_acq_h > time_fuel_h[-1]:
        return float("nan")

    return float(np.interp(t_acq_h, time_fuel_h, cumulative_fuel_consumed_g))


def infer_differential_deployment_speed_mps(case_label: Optional[str], run_index: int) -> float:
    """
    Infer |v_leader - v_follower| from a case label.

    Expected labels include examples such as:
        'Leader +0.5 m/s, follower -0.5 m/s'
        'Leader 0.0 m/s, follower 0.0 m/s'

    If parsing fails, the function falls back to the campaign convention where
    run index 0..4 corresponds to differential speeds 0..4 m/s.
    """
    if case_label is not None:
        leader_match = re.search(r"leader\s*([+-]?\d+(?:\.\d+)?)\s*m/s", case_label, flags=re.IGNORECASE)
        follower_match = re.search(r"follower\s*([+-]?\d+(?:\.\d+)?)\s*m/s", case_label, flags=re.IGNORECASE)

        if leader_match and follower_match:
            leader_v = float(leader_match.group(1))
            follower_v = float(follower_match.group(1))
            return abs(leader_v - follower_v)

        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", case_label)
        if len(numbers) >= 2:
            return abs(float(numbers[0]) - float(numbers[1]))
        if len(numbers) == 1:
            return abs(float(numbers[0]))

    return float(run_index)


def differential_speed_label(speed_mps: float) -> str:
    """Return a compact legend label for a differential deployment speed."""
    if np.isclose(speed_mps, round(speed_mps)):
        return rf"$\Delta v_{{dep}}={speed_mps:.0f}$ m/s"
    return rf"$\Delta v_{{dep}}={speed_mps:.2f}$ m/s"


def process_single_run(
    run_dir: Path,
    follower_sat_idx: int,
    case_label: Optional[str] = None,
    run_index: int = 0,
    differential_deployment_speed_mps: Optional[float] = None,
) -> RunProcessingResult:
    """Process one run_XXX folder."""
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: '{run_dir}'.")

    if differential_deployment_speed_mps is None:
        differential_deployment_speed_mps = infer_differential_deployment_speed_mps(case_label, run_index)

    leader = load_satellite_core_data(run_dir, sat_idx=0)
    follower = load_satellite_core_data(run_dir, sat_idx=follower_sat_idx)

    t_state_h = leader.r_BN_N.time_h
    oed = compute_follower_minus_leader_oed(leader, follower)
    r_rel_RTN_m = compute_follower_minus_leader_rtn_position(leader, follower)

    acquired, t_acq_h, t_dwell_pass_h, da_error, de_error, di_error = find_acquisition_time_h(
        time_h=t_state_h,
        oed=oed,
        follower_sat_idx=follower_sat_idx,
    )

    fuel_mass = follower.fuelMass.data.astype(np.float64)
    time_fuel_h = follower.fuelMass.time_h
    cumulative_fuel_consumed_g = compute_cumulative_fuel_consumed_g(fuel_mass)

    m_acq_g = interpolate_fuel_consumed_at_time_g(
        time_fuel_h=time_fuel_h,
        cumulative_fuel_consumed_g=cumulative_fuel_consumed_g,
        t_acq_h=t_acq_h,
    )

    burn_metrics = compute_burn_metrics_until_acquisition(
        run_dir=run_dir,
        follower_sat_idx=follower_sat_idx,
        t_acq_h=t_acq_h,
    )

    total_fuel_consumed_g = float(cumulative_fuel_consumed_g[-1])

    if BATTERY_MAX_ENERGY <= 0.0:
        raise ValueError(f"BATTERY_MAX_ENERGY must be positive, got {BATTERY_MAX_ENERGY}.")

    storage_level_Wh = follower.storageLevel.data.astype(np.float64)
    battery_energy_percent = 100.0 * storage_level_Wh / BATTERY_MAX_ENERGY
    time_eps_h = follower.storageLevel.time_h

    current_net_power_w = follower.currentNetPower.data.astype(np.float64)
    if current_net_power_w.ndim != 1:
        raise ValueError(f"Expected currentNetPower.data to be 1D, got shape {current_net_power_w.shape}.")

    pointing_mode_code = follower.pointingModeCode.data.astype(np.int16)
    time_mode_h = follower.pointingModeCode.time_h

    return RunProcessingResult(
        run_dir=run_dir,
        run_name=run_dir.name,
        case_label=case_label or run_dir.name,
        plot_label=differential_speed_label(differential_deployment_speed_mps),
        differential_deployment_speed_mps=float(differential_deployment_speed_mps),
        follower_sat_idx=follower_sat_idx,
        acquired=bool(acquired),
        t_acq_h=t_acq_h,
        t_dwell_pass_h=t_dwell_pass_h,
        m_acq_g=m_acq_g,
        number_of_burns_until_acq=burn_metrics.number_of_burns_until_acq,
        average_thruster_on_time_until_acq_s=burn_metrics.average_thruster_on_time_until_acq_s,
        average_attitude_tracking_error_during_burns_deg=burn_metrics.average_attitude_tracking_error_during_burns_deg,
        total_fuel_consumed_g=total_fuel_consumed_g,
        initial_fuel_mass_kg=float(fuel_mass[0]),
        final_fuel_mass_kg=float(fuel_mass[-1]),
        min_abs_da_over_a_error=float(np.nanmin(np.abs(da_error))),
        min_abs_de_error=float(np.nanmin(np.abs(de_error))),
        min_abs_di_error_rad=float(np.nanmin(np.abs(di_error))),
        target_de=float(follower_sat_idx * EPS),
        target_di_rad=float(follower_sat_idx * EPS),
        time_state_h=t_state_h,
        da_over_a=oed["da_over_a"],
        de=oed["de"],
        di_rad=oed["di"],
        dOmega_rad=oed["dOmega"],
        domega_rad=oed["domega"],
        dM_rad=oed["dM"],
        r_rel_RTN_m=r_rel_RTN_m,
        time_fuel_h=time_fuel_h,
        cumulative_fuel_consumed_g=cumulative_fuel_consumed_g,
        time_eps_h=time_eps_h,
        battery_energy_percent=battery_energy_percent,
        current_net_power_w=current_net_power_w,
        time_mode_h=time_mode_h,
        pointing_mode_code=pointing_mode_code,
    )


def infer_common_monte_carlo_dir(run_dirs: Iterable[Path]) -> Path:
    """Return the common Monte Carlo parent folder shared by run_XXX folders."""
    run_dirs = [Path(p).resolve() for p in run_dirs]

    if len(run_dirs) == 0:
        raise ValueError("No run directories were provided.")

    parents = {p.parent for p in run_dirs}

    if len(parents) != 1:
        parent_list = "\n".join(str(p) for p in sorted(parents))
        raise ValueError(
            "All run directories must share the same Monte Carlo parent folder. "
            f"Got:\n{parent_list}"
        )

    return next(iter(parents))


def _sorted_by_differential_speed(results: Sequence[RunProcessingResult]) -> list[RunProcessingResult]:
    return sorted(results, key=lambda result: result.differential_deployment_speed_mps)


def fit_linear_acquisition_models(
    results: Sequence[RunProcessingResult],
) -> LinearAcquisitionModelParameters:
    """
    Estimate least-squares linear acquisition sensitivity models.

    The fitted models are:
        m_acq(Delta v_dep) = a_m * Delta v_dep + b_m
        t_acq(Delta v_dep) = a_t * Delta v_dep + b_t

    Only acquired runs with finite metric values are used. If fewer than two
    valid samples are available for a metric, its slope and intercept are NaN.
    """
    sorted_results = _sorted_by_differential_speed(results)

    speeds = np.array(
        [result.differential_deployment_speed_mps for result in sorted_results],
        dtype=np.float64,
    )
    m_acq_g = np.array(
        [result.m_acq_g if result.acquired else np.nan for result in sorted_results],
        dtype=np.float64,
    )
    t_acq_h = np.array(
        [result.t_acq_h if result.acquired else np.nan for result in sorted_results],
        dtype=np.float64,
    )

    m_mask = np.isfinite(speeds) & np.isfinite(m_acq_g)
    t_mask = np.isfinite(speeds) & np.isfinite(t_acq_h)

    if np.count_nonzero(m_mask) >= 2:
        a_m, b_m = np.polyfit(speeds[m_mask], m_acq_g[m_mask], deg=1)
    else:
        a_m, b_m = float("nan"), float("nan")

    if np.count_nonzero(t_mask) >= 2:
        a_t, b_t = np.polyfit(speeds[t_mask], t_acq_h[t_mask], deg=1)
    else:
        a_t, b_t = float("nan"), float("nan")

    return LinearAcquisitionModelParameters(
        m_acq_slope_g_per_mps=float(a_m),
        m_acq_intercept_g=float(b_m),
        t_acq_slope_h_per_mps=float(a_t),
        t_acq_intercept_h=float(b_t),
    )


def write_metrics_csv(
    results: list[RunProcessingResult],
    out_csv_path: Path,
    run_dir_reference: Optional[Path] = None,
) -> None:
    """Write one metrics row per processed run."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if run_dir_reference is None:
        run_dir_reference = out_csv_path.parent

    linear_model_params = fit_linear_acquisition_models(results)

    fieldnames = [
        "run_name",
        "run_dir",
        "case_label",
        "differential_deployment_speed_mps",
        "follower_sat_idx",
        "acquired",
        "acquisition_dwell_time_h",
        "t_acq_h",
        "t_dwell_pass_h",
        "m_acq_g",
        "number_of_burns_until_acq",
        "average_thruster_on_time_until_acq_s",
        "average_attitude_tracking_error_during_burns_deg",
        "total_fuel_consumed_g",
        "initial_fuel_mass_kg",
        "final_fuel_mass_kg",
        "target_da_over_a",
        "target_de",
        "target_di_rad",
        "delta_a_tolerance",
        "delta_e_tolerance",
        "delta_i_tolerance_rad",
        "min_abs_da_over_a_error",
        "min_abs_de_error",
        "min_abs_di_error_rad",
        "linear_fit_m_acq_slope_g_per_mps",
        "linear_fit_m_acq_intercept_g",
        "linear_fit_t_acq_slope_h_per_mps",
        "linear_fit_t_acq_intercept_h",
    ]

    with out_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "run_name": result.run_name,
                    "run_dir": os.path.relpath(
                        result.run_dir.resolve(),
                        start=Path(run_dir_reference).resolve(),
                    ),
                    "case_label": result.case_label,
                    "differential_deployment_speed_mps": result.differential_deployment_speed_mps,
                    "follower_sat_idx": result.follower_sat_idx,
                    "acquired": int(result.acquired),
                    "acquisition_dwell_time_h": ACQUISITION_DWELL_TIME_H,
                    "t_acq_h": result.t_acq_h,
                    "t_dwell_pass_h": result.t_dwell_pass_h,
                    "m_acq_g": result.m_acq_g,
                    "number_of_burns_until_acq": result.number_of_burns_until_acq,
                    "average_thruster_on_time_until_acq_s": result.average_thruster_on_time_until_acq_s,
                    "average_attitude_tracking_error_during_burns_deg": result.average_attitude_tracking_error_during_burns_deg,
                    "total_fuel_consumed_g": result.total_fuel_consumed_g,
                    "initial_fuel_mass_kg": result.initial_fuel_mass_kg,
                    "final_fuel_mass_kg": result.final_fuel_mass_kg,
                    "target_da_over_a": DELTA_A_OVER_A_REFERENCE,
                    "target_de": result.target_de,
                    "target_di_rad": result.target_di_rad,
                    "delta_a_tolerance": DELTA_A_TOLERANCE,
                    "delta_e_tolerance": DELTA_E_TOLERANCE,
                    "delta_i_tolerance_rad": DELTA_I_TOLERANCE_RAD,
                    "min_abs_da_over_a_error": result.min_abs_da_over_a_error,
                    "min_abs_de_error": result.min_abs_de_error,
                    "min_abs_di_error_rad": result.min_abs_di_error_rad,
                    "linear_fit_m_acq_slope_g_per_mps": linear_model_params.m_acq_slope_g_per_mps,
                    "linear_fit_m_acq_intercept_g": linear_model_params.m_acq_intercept_g,
                    "linear_fit_t_acq_slope_h_per_mps": linear_model_params.t_acq_slope_h_per_mps,
                    "linear_fit_t_acq_intercept_h": linear_model_params.t_acq_intercept_h,
                }
            )


def write_cumulative_fuel_timeseries_csv(
    results: list[RunProcessingResult],
    out_csv_path: Path,
) -> None:
    """Write cumulative fuel time histories in a long-table CSV format."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "run_name",
                "case_label",
                "differential_deployment_speed_mps",
                "follower_sat_idx",
                "time_h",
                "cumulative_follower_fuel_consumed_g",
            ]
        )

        for result in results:
            for time_h, fuel_g in zip(result.time_fuel_h, result.cumulative_fuel_consumed_g):
                writer.writerow(
                    [
                        result.run_name,
                        result.case_label,
                        result.differential_deployment_speed_mps,
                        result.follower_sat_idx,
                        float(time_h),
                        float(fuel_g),
                    ]
                )


def _time_window_mask(time_h: np.ndarray) -> np.ndarray:
    """Return a mask for the inclusive plotting time window [0, PLOT_END_TIME_H]."""
    if PLOT_END_TIME_H <= 0.0:
        raise ValueError(f"PLOT_END_TIME_H must be positive, got {PLOT_END_TIME_H}.")

    if time_h.size == 0:
        raise ValueError("Cannot plot an empty time vector.")

    if PLOT_END_TIME_H > float(np.nanmax(time_h)) + 1e-12:
        raise ValueError(
            f"PLOT_END_TIME_H={PLOT_END_TIME_H} h exceeds the available data horizon "
            f"({float(np.nanmax(time_h)):.6f} h)."
        )

    return (time_h >= 0.0) & (time_h <= PLOT_END_TIME_H)


def _clip_to_plot_time_window(time_h: np.ndarray, *series: np.ndarray) -> tuple[np.ndarray, ...]:
    """Clip one time vector and matching time histories to the plotting window."""
    mask = _time_window_mask(time_h)
    clipped: list[np.ndarray] = [time_h[mask]]
    for values in series:
        clipped.append(values[mask])
    return tuple(clipped)


def _is_time_in_plot_window(time_h: float) -> bool:
    """Return True if a scalar time should be shown in the current plot window."""
    return np.isfinite(time_h) and (0.0 <= time_h <= PLOT_END_TIME_H)


def _apply_time_axis_limit(axes) -> None:
    """Apply the common inclusive time-axis range to one or more Matplotlib axes."""
    axes_arr = np.atleast_1d(axes)
    for ax in axes_arr:
        ax.set_xlim(0.0, PLOT_END_TIME_H)


def plot_cumulative_follower_fuel_consumption(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """Plot follower cumulative fuel consumption for all runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))

    for result in _sorted_by_differential_speed(results):
        time_fuel_h, cumulative_fuel_consumed_g = _clip_to_plot_time_window(
            result.time_fuel_h,
            result.cumulative_fuel_consumed_g,
        )

        line, = ax.plot(
            time_fuel_h,
            cumulative_fuel_consumed_g,
            linewidth=1.8,
            label=result.plot_label,
        )

        if result.acquired and _is_time_in_plot_window(result.t_acq_h):
            ax.axvline(
                result.t_acq_h,
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

            ax.plot(
                result.t_acq_h,
                result.m_acq_g,
                marker="o",
                color=line.get_color(),
                markersize=4,
            )

    ax.set_title(f"Follower cumulative fuel consumption for deployment velocity cases, t ∈ [0, {PLOT_END_TIME_H:g}] h")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Cumulative follower fuel consumed [g]")
    ax.grid(True, alpha=0.3)
    _apply_time_axis_limit(ax)
    ax.legend(loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_acquisition_fuel_vs_differential_speed(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """Plot acquired fuel consumption m_acq [g] versus deployment-speed difference."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = _sorted_by_differential_speed(results)
    speeds = np.array([result.differential_deployment_speed_mps for result in sorted_results], dtype=float)
    m_acq_g = np.array([result.m_acq_g if result.acquired else np.nan for result in sorted_results], dtype=float)

    fig, ax = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    ax.plot(speeds, m_acq_g, marker="o", linewidth=1.8, label=r"$m_{acq}$")

    ax.set_title("Formation acquisition fuel consumption sensitivity")
    ax.set_xlabel(r"Differential deployment speed $|v_L-v_F|$ [m/s]")
    ax.set_ylabel(r"Acquisition fuel consumption $m_{acq}$ [g]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_acquisition_time_vs_differential_speed(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """Plot acquisition time t_acq [h] versus deployment-speed difference."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = _sorted_by_differential_speed(results)
    speeds = np.array([result.differential_deployment_speed_mps for result in sorted_results], dtype=float)
    t_acq_h = np.array([result.t_acq_h if result.acquired else np.nan for result in sorted_results], dtype=float)

    fig, ax = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    ax.plot(speeds, t_acq_h, marker="o", linewidth=1.8, label=r"$t_{acq}$")

    ax.set_title("Formation acquisition time sensitivity")
    ax.set_xlabel(r"Differential deployment speed $|v_L-v_F|$ [m/s]")
    ax.set_ylabel(r"Acquisition time $t_{acq}$ [h]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_acquisition_time_and_fuel_vs_differential_speed(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """Plot acquisition time and acquisition fuel consumption versus deployment-speed difference."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = _sorted_by_differential_speed(results)
    speeds = np.array([result.differential_deployment_speed_mps for result in sorted_results], dtype=float)
    t_acq_h = np.array([result.t_acq_h if result.acquired else np.nan for result in sorted_results], dtype=float)
    m_acq_g = np.array([result.m_acq_g if result.acquired else np.nan for result in sorted_results], dtype=float)

    linear_model_params = fit_linear_acquisition_models(sorted_results)

    fig, ax_time = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    ax_fuel = ax_time.twinx()

    time_line, = ax_time.plot(
        speeds,
        t_acq_h,
        marker="o",
        linewidth=1.8,
        color="C0",
        label=r"Acquisition time $t_{acq}$",
    )
    fuel_line, = ax_fuel.plot(
        speeds,
        m_acq_g,
        marker="s",
        linewidth=1.8,
        color="C1",
        label=r"Acquisition fuel $m_{acq}$",
    )

    finite_speeds = speeds[np.isfinite(speeds)]
    time_fit_line = None
    fuel_fit_line = None

    if finite_speeds.size >= 2:
        speed_model = np.linspace(
            float(np.nanmin(finite_speeds)),
            float(np.nanmax(finite_speeds)),
            200,
        )

        if np.isfinite(linear_model_params.t_acq_slope_h_per_mps) and np.isfinite(linear_model_params.t_acq_intercept_h):
            t_model_h = (
                linear_model_params.t_acq_slope_h_per_mps * speed_model
                + linear_model_params.t_acq_intercept_h
            )
            time_fit_line, = ax_time.plot(
                speed_model,
                t_model_h,
                linestyle="--",
                linewidth=1.6,
                color=time_line.get_color(),
                label=r"Linear fit $\tilde{t}_{acq}$",
            )

        if np.isfinite(linear_model_params.m_acq_slope_g_per_mps) and np.isfinite(linear_model_params.m_acq_intercept_g):
            m_model_g = (
                linear_model_params.m_acq_slope_g_per_mps * speed_model
                + linear_model_params.m_acq_intercept_g
            )
            fuel_fit_line, = ax_fuel.plot(
                speed_model,
                m_model_g,
                linestyle="--",
                linewidth=1.6,
                color=fuel_line.get_color(),
                label=r"Linear fit $\tilde{m}_{acq}$",
            )

    ax_time.set_title("Formation acquisition sensitivity")
    ax_time.set_xlabel(r"Differential deployment speed $|v_L-v_F|$ [m/s]")
    ax_fuel.set_ylabel(r"Acquisition fuel consumption $m_{acq}$ [g]")
    ax_time.set_ylabel(r"Acquisition time $t_{acq}$ [h]")

    ax_time.tick_params(axis="y")
    ax_fuel.tick_params(axis="y")
    ax_time.grid(True, alpha=0.3)

    legend_lines = [time_line, fuel_line]
    if time_fit_line is not None:
        legend_lines.append(time_fit_line)
    if fuel_fit_line is not None:
        legend_lines.append(fuel_fit_line)

    ax_time.legend(
        legend_lines,
        [line.get_label() for line in legend_lines],
        loc=LEGEND_LOC,
    )

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_da_di_oed_for_all_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """
    Plot the acquisition-relevant orbital-element differences for all runs.

    The plotted OEd quantities are computed using the same follower-minus-leader
    orbital-element difference implementation used for the acquisition metric:
        - da/a [-]
        - di [rad]
    """
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = mpl.subplots(2, 1, sharex=True, figsize=(PLOT_WIDTH_IN, 0.85 * PLOT_HEIGHT_IN))
    fig.suptitle(f"Formation acquisition OEd comparison for leader-follower pair, t ∈ [0, {PLOT_END_TIME_H:g}] h")

    target_da_over_a = DELTA_A_OVER_A_REFERENCE
    target_de_by_sat_idx = {result.follower_sat_idx: result.target_de for result in results}
    target_di_by_sat_idx = {result.follower_sat_idx: result.target_di_rad for result in results}

    for result in _sorted_by_differential_speed(results):
        time_state_h, da_over_a, di_rad = _clip_to_plot_time_window(
            result.time_state_h,
            result.da_over_a,
            result.di_rad,
        )

        line, = axs[0].plot(
            time_state_h,
            da_over_a,
            linewidth=1.4,
            label=result.plot_label,
        )

        color = line.get_color()
        axs[1].plot(
            time_state_h,
            di_rad,
            linewidth=1.4,
            color=color,
            label=result.plot_label,
        )

        if result.acquired and _is_time_in_plot_window(result.t_acq_h):
            for ax in axs:
                ax.axvline(
                    result.t_acq_h,
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.75,
                )

    axs[0].axhline(target_da_over_a, color="black", linestyle="--", linewidth=1.0, label="Desired da/a")

    for follower_sat_idx, target_di_rad in sorted(target_di_by_sat_idx.items()):
        axs[1].axhline(
            target_di_rad,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Desired di, sat {follower_sat_idx}",
        )

    axs[0].set_ylabel(r"$\Delta a/a$ [-]")
    axs[1].set_ylabel(r"$\Delta i$ [rad]")
    axs[1].set_xlabel("Time [h]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
    _apply_time_axis_limit(axs)

    handles, labels = axs[0].get_legend_handles_labels()
    handles_1, labels_1 = axs[1].get_legend_handles_labels()
    unique_handles, unique_labels = _unique_legend_entries(handles + handles_1, labels + labels_1)
    axs[0].legend(unique_handles, unique_labels, loc=LEGEND_LOC, ncol=1)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)



def plot_da_de_di_oed_for_all_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """
    Plot the acquisition-relevant orbital-element differences for all runs.

    The plotted OEd quantities are computed using the same follower-minus-leader
    orbital-element difference implementation used for the acquisition metric:
        - da/a [-]
        - de [-]
        - di [rad]

    Vertical dashed lines mark t_acq for each acquired run.
    """
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=(PLOT_WIDTH_IN, 1.15 * PLOT_HEIGHT_IN))
    fig.suptitle(f"Formation acquisition OEd comparison for leader-follower pair, t ∈ [0, {PLOT_END_TIME_H:g}] h")

    target_da_over_a = DELTA_A_OVER_A_REFERENCE
    target_de_by_sat_idx = {result.follower_sat_idx: result.target_de for result in results}
    target_di_by_sat_idx = {result.follower_sat_idx: result.target_di_rad for result in results}

    for result in _sorted_by_differential_speed(results):
        time_state_h, da_over_a, de, di_rad = _clip_to_plot_time_window(
            result.time_state_h,
            result.da_over_a,
            result.de,
            result.di_rad,
        )

        line, = axs[0].plot(
            time_state_h,
            da_over_a,
            linewidth=1.4,
            label=result.plot_label,
        )

        color = line.get_color()
        axs[1].plot(
            time_state_h,
            de,
            linewidth=1.4,
            color=color,
            label=result.plot_label,
        )
        axs[2].plot(
            time_state_h,
            di_rad,
            linewidth=1.4,
            color=color,
            label=result.plot_label,
        )

        if result.acquired and _is_time_in_plot_window(result.t_acq_h):
            for ax in axs:
                ax.axvline(
                    result.t_acq_h,
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.75,
                )

    axs[0].axhline(target_da_over_a, color="black", linestyle="--", linewidth=1.0, label="Desired da/a")

    for follower_sat_idx, target_de in sorted(target_de_by_sat_idx.items()):
        axs[1].axhline(
            target_de,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Desired de, sat {follower_sat_idx}",
        )

    for follower_sat_idx, target_di_rad in sorted(target_di_by_sat_idx.items()):
        axs[2].axhline(
            target_di_rad,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Desired di, sat {follower_sat_idx}",
        )

    axs[0].set_ylabel(r"$\Delta a/a$ [-]")
    axs[1].set_ylabel(r"$\Delta e$ [-]")
    axs[2].set_ylabel(r"$\Delta i$ [rad]")
    axs[2].set_xlabel("Time [h]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
    _apply_time_axis_limit(axs)

    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique_handles, unique_labels = _unique_legend_entries(handles, labels)
    axs[0].legend(unique_handles, unique_labels, loc=LEGEND_LOC, ncol=1)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)

def plot_all_oed_components_for_all_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """Plot all six follower-minus-leader orbital-element differences for all runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = mpl.subplots(6, 1, sharex=True, figsize=(PLOT_WIDTH_IN, 1.55 * PLOT_HEIGHT_IN))
    fig.suptitle(f"Orbital-element difference comparison for leader-follower pair, t ∈ [0, {PLOT_END_TIME_H:g}] h")

    component_specs = [
        ("da_over_a", r"$\Delta a/a$ [-]", DELTA_A_OVER_A_REFERENCE),
        ("de", r"$\Delta e$ [-]", None),
        ("di_rad", r"$\Delta i$ [rad]", None),
        ("dOmega_rad", r"$\Delta \Omega$ [rad]", 0.0),
        ("domega_rad", r"$\Delta \omega$ [rad]", 0.0),
        ("dM_rad", r"$\Delta M$ [rad]", 0.0),
    ]

    target_de_by_sat_idx = {result.follower_sat_idx: result.target_de for result in results}
    target_di_by_sat_idx = {result.follower_sat_idx: result.target_di_rad for result in results}

    for result in _sorted_by_differential_speed(results):
        time_state_h = result.time_state_h
        mask = _time_window_mask(time_state_h)
        time_state_h = time_state_h[mask]

        color = None
        for ax, (attr_name, _, _) in zip(axs, component_specs):
            data = getattr(result, attr_name)[mask]
            if color is None:
                line, = ax.plot(
                    time_state_h,
                    data,
                    linewidth=1.2,
                    label=result.plot_label,
                )
                color = line.get_color()
            else:
                ax.plot(
                    time_state_h,
                    data,
                    linewidth=1.2,
                    color=color,
                    label=result.plot_label,
                )

            if result.acquired and _is_time_in_plot_window(result.t_acq_h):
                ax.axvline(
                    result.t_acq_h,
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.65,
                )

    for ax, (attr_name, ylabel, target_value) in zip(axs, component_specs):
        if attr_name == "de":
            for follower_sat_idx, target_de in sorted(target_de_by_sat_idx.items()):
                ax.axhline(
                    target_de,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Desired de, sat {follower_sat_idx}",
                )
        elif attr_name == "di_rad":
            for follower_sat_idx, target_di_rad in sorted(target_di_by_sat_idx.items()):
                ax.axhline(
                    target_di_rad,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Desired di, sat {follower_sat_idx}",
                )
        elif target_value is not None:
            label = "Desired zero OEd" if target_value == 0.0 else "Desired da/a"
            ax.axhline(
                target_value,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=label,
            )

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Time [h]")
    _apply_time_axis_limit(axs)

    handles, labels = axs[0].get_legend_handles_labels()
    unique_handles, unique_labels = _unique_legend_entries(handles, labels)
    axs[0].legend(unique_handles, unique_labels, loc=LEGEND_LOC, ncol=1)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def _unique_legend_entries(handles: Sequence, labels: Sequence[str]) -> tuple[list, list[str]]:
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def _select_results_for_indices(results: list[RunProcessingResult], selected_indices: Sequence[int]) -> list[RunProcessingResult]:
    selected: list[RunProcessingResult] = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(results):
            raise IndexError(
                f"Selected RTN run index {idx} is outside the available result index range 0..{len(results)-1}."
            )
        selected.append(results[idx])
    return selected


def plot_rtn_relative_position_3d_for_selected_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
    selected_indices: Sequence[int] = (0, 2, 4),
) -> None:
    """Plot 3D leader-follower RTN relative-position trajectories for selected runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    selected_results = _select_results_for_indices(results, selected_indices)

    fig = mpl.figure(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(0.0, 0.0, 0.0, marker="o", s=60, label="Leader")  # type: ignore[arg-type]

    for result in selected_results:
        _, r_RTN = _clip_to_plot_time_window(
            result.time_state_h,
            result.r_rel_RTN_m,
        )
        ax.plot(
            r_RTN[:, 0],
            r_RTN[:, 1],
            r_RTN[:, 2],
            linewidth=1.5,
            label=result.plot_label,
        )

    ax.set_title(f"Leader-follower RTN relative-position trajectories, t ∈ [0, {PLOT_END_TIME_H:g}] h")
    ax.set_xlabel("Radial [m]")
    ax.set_ylabel("Along-track [m]")
    ax.set_zlabel("Cross-track [m]")  # type: ignore[attr-defined]
    ax.grid(True)
    ax.legend(loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_rtn_relative_position_components_for_selected_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
    selected_indices: Sequence[int] = (0, 2, 4),
) -> None:
    """Plot component-wise leader-follower RTN relative positions for selected runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    selected_results = _select_results_for_indices(results, selected_indices)

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=(PLOT_WIDTH_IN, 1.15 * PLOT_HEIGHT_IN))
    fig.suptitle(f"Leader-follower RTN relative-position components, t ∈ [0, {PLOT_END_TIME_H:g}] h")

    component_indices = [0, 1, 2]
    component_titles = ["Radial (R)", "Along-track (T)", "Cross-track (N)"]

    for result in selected_results:
        time_state_h, r_rel_RTN_m = _clip_to_plot_time_window(
            result.time_state_h,
            result.r_rel_RTN_m,
        )

        color = None
        for ax, comp_idx in zip(axs, component_indices):
            if color is None:
                line, = ax.plot(
                    time_state_h,
                    r_rel_RTN_m[:, comp_idx],
                    linewidth=1.5,
                    label=result.plot_label,
                )
                color = line.get_color()
            else:
                ax.plot(
                    time_state_h,
                    r_rel_RTN_m[:, comp_idx],
                    linewidth=1.5,
                    color=color,
                    label=result.plot_label,
                )

            if result.acquired and _is_time_in_plot_window(result.t_acq_h):
                ax.axvline(
                    result.t_acq_h,
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.75,
                )

    for ax, title in zip(axs, component_titles):
        ax.set_title(title)
        ax.set_ylabel("Relative position [m]")
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Time [h]")
    _apply_time_axis_limit(axs)

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)



def plot_acquisition_burn_metrics_vs_differential_speed(
    results: list[RunProcessingResult],
    out_png_path: Path,
) -> None:
    """
    Plot acquisition burn metrics versus differential deployment speed.

    For each deployment-speed case, two grouped bars are shown:
        - left bar: average thruster on-time during acquisition [s]
        - right bar: number of burns during acquisition [-]

    The left y-axis corresponds to average thruster on-time, while the right
    y-axis corresponds to number of burns.
    """
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = _sorted_by_differential_speed(results)
    x = np.arange(len(sorted_results), dtype=float)
    bar_width = 0.36

    avg_on_time_s = np.array(
        [
            result.average_thruster_on_time_until_acq_s if result.acquired else np.nan
            for result in sorted_results
        ],
        dtype=float,
    )
    burn_counts = np.array(
        [
            result.number_of_burns_until_acq if result.acquired else np.nan
            for result in sorted_results
        ],
        dtype=float,
    )

    x_labels = [f"{result.differential_deployment_speed_mps:g}" for result in sorted_results]

    fig, ax_time = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    ax_count = ax_time.twinx()

    on_time_bars = ax_time.bar(
        x - bar_width / 2.0,
        avg_on_time_s,
        width=bar_width,
        color="C0",
        label="Avg. thruster on-time",
    )
    burn_count_bars = ax_count.bar(
        x + bar_width / 2.0,
        burn_counts,
        width=bar_width,
        color="C1",
        label="Number of burns",
    )

    ax_time.set_title("Formation acquisition burn metrics")
    ax_time.set_xlabel(r"Differential deployment speed $|v_L-v_F|$ [m/s]")
    ax_time.set_ylabel("Average thruster on-time during acquisition [s]")
    ax_count.set_ylabel("Number of burns during acquisition [-]")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(x_labels)

    # Align the horizontal grid with the integer-valued burn-count axis.
    # Since this is a dual-axis plot, the grid is intentionally drawn from
    # the right y-axis instead of the left y-axis.
    ax_time.grid(False)
    finite_counts = burn_counts[np.isfinite(burn_counts)]
    if finite_counts.size > 0:
        max_count = int(np.ceil(float(np.nanmax(finite_counts))))
    else:
        max_count = 1

    # Add integer headroom above the tallest burn-count bar. This keeps the
    # tallest bar from colliding with the upper plot boundary and gives the
    # upper-right legend enough room while preserving integer-aligned grid lines.
    count_axis_padding = max(2, int(np.ceil(0.15 * max_count)))
    count_axis_upper = max(1, max_count + count_axis_padding)
    ax_count.set_ylim(0.0, float(count_axis_upper))
    ax_count.set_yticks(np.arange(0, count_axis_upper + 1, 1))
    ax_count.grid(True, axis="y", alpha=0.3)

    finite_on_times = avg_on_time_s[np.isfinite(avg_on_time_s)]
    if finite_on_times.size > 0:
        max_on_time = float(np.nanmax(finite_on_times))
        on_time_axis_upper = max(1.0, 1.18 * max_on_time)
    else:
        on_time_axis_upper = 1.0
    ax_time.set_ylim(0.0, on_time_axis_upper)

    ax_time.legend(
        [on_time_bars, burn_count_bars],
        [on_time_bars.get_label(), burn_count_bars.get_label()],
        loc=LEGEND_LOC,
    )

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


POINTING_MODE_LABELS = {
    0: "COAST",
    1: "COMMS",
    2: "CHARGE",
    3: "CAPTURE",
    4: "BURN_TRANSIT",
    5: "BURN",
    6: "EMERGENCY",
    7: "ERROR",
}


def plot_eps_overview_for_all_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
    plot_height: float = PLOT_HEIGHT_IN,
) -> None:
    """Plot battery energy percentage and net battery power for all runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = mpl.subplots(2, 1, sharex=True, figsize=(PLOT_WIDTH_IN, 1.15 * PLOT_HEIGHT_IN))
    fig.suptitle(f"Follower EPS overview, t ∈ [0, {PLOT_END_TIME_H:g}] h")

    for result in _sorted_by_differential_speed(results):
        time_eps_h, battery_energy_percent, current_net_power_w = _clip_to_plot_time_window(
            result.time_eps_h,
            result.battery_energy_percent,
            result.current_net_power_w,
        )

        line, = axs[0].plot(
            time_eps_h,
            battery_energy_percent,
            linewidth=1.4,
            label=result.plot_label,
        )
        axs[1].plot(
            time_eps_h,
            current_net_power_w,
            linewidth=1.2,
            color=line.get_color(),
            label=result.plot_label,
        )

    axs[0].set_ylabel("Battery energy [%]")
    axs[1].set_ylabel("Net power [W]")
    axs[1].set_xlabel("Time [h]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
    _apply_time_axis_limit(axs)

    handles, labels = axs[0].get_legend_handles_labels()
    unique_handles, unique_labels = _unique_legend_entries(handles, labels)
    axs[0].legend(unique_handles, unique_labels, loc=LEGEND_LOC, ncol=1)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)


def plot_operational_modes_for_all_runs(
    results: list[RunProcessingResult],
    out_png_path: Path,
    plot_height: float = PLOT_HEIGHT_IN,
) -> None:
    """Plot operational pointing modes over time for all runs."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = mpl.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))

    plotted_mode_values: set[int] = set()
    for result in _sorted_by_differential_speed(results):
        time_mode_h, pointing_mode_code = _clip_to_plot_time_window(
            result.time_mode_h,
            result.pointing_mode_code,
        )
        mode_values = pointing_mode_code.astype(int)
        plotted_mode_values.update(int(v) for v in np.unique(mode_values))

        ax.step(
            time_mode_h,
            mode_values,
            where="post",
            linewidth=1.3,
            label=result.plot_label,
        )

    ax.set_title(f"Follower operational modes, t ∈ [0, {PLOT_END_TIME_H:g}] h")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Operational mode")

    if plotted_mode_values:
        mode_ticks = sorted(plotted_mode_values)
    else:
        mode_ticks = sorted(POINTING_MODE_LABELS)
    ax.set_yticks(mode_ticks)
    ax.set_yticklabels([POINTING_MODE_LABELS.get(mode, str(mode)) for mode in mode_ticks])

    ax.grid(True, alpha=0.3)
    _apply_time_axis_limit(ax)
    ax.legend(loc=LEGEND_LOC)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI, bbox_inches="tight")
    mpl.close(fig)

def run_deployment_velocity_fuel_sensitivity_analysis(
    run_dirs: list[Path],
    follower_sat_idx: int = 1,
    case_labels: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    differential_deployment_speeds_mps: Optional[list[float]] = None,
    selected_rtn_run_indices: Sequence[int] = (0, 1, 2, 3, 4),
) -> list[RunProcessingResult]:
    """
    Process selected Monte Carlo run folders and save metrics/plots.

    Outputs:
        deployment_velocity_formation_acquisition_metrics.csv
        deployment_velocity_cumulative_fuel_timeseries.csv
        deployment_velocity_follower_cumulative_fuel_consumption.png
        deployment_velocity_acquisition_fuel_vs_deployment_speed.png
        deployment_velocity_acquisition_time_vs_deployment_speed.png
        deployment_velocity_acquisition_time_and_fuel_vs_deployment_speed.png
        deployment_velocity_oed_da_di_comparison.png
        deployment_velocity_oed_da_de_di_comparison.png
        deployment_velocity_oed_all_components_comparison.png
        deployment_velocity_rtn_relative_position_3d_selected_runs.png
        deployment_velocity_rtn_relative_position_components_selected_runs.png
        deployment_velocity_eps_overview_all_runs.png
        deployment_velocity_operational_modes_all_runs.png
        deployment_velocity_acquisition_burn_metrics_vs_deployment_speed.png
    """
    if len(run_dirs) == 0:
        raise ValueError("run_dirs cannot be empty.")

    run_dirs = [Path(p) for p in run_dirs]

    if case_labels is not None and len(case_labels) != len(run_dirs):
        raise ValueError(
            f"case_labels length ({len(case_labels)}) must match run_dirs length ({len(run_dirs)})."
        )

    if differential_deployment_speeds_mps is not None and len(differential_deployment_speeds_mps) != len(run_dirs):
        raise ValueError(
            "differential_deployment_speeds_mps length "
            f"({len(differential_deployment_speeds_mps)}) must match run_dirs length ({len(run_dirs)})."
        )

    if output_dir is None:
        output_dir = infer_common_monte_carlo_dir(run_dirs)
    else:
        output_dir = Path(output_dir)

    results: list[RunProcessingResult] = []

    for idx, run_dir in enumerate(run_dirs):
        label = case_labels[idx] if case_labels is not None else None
        speed = differential_deployment_speeds_mps[idx] if differential_deployment_speeds_mps is not None else None
        result = process_single_run(
            run_dir=run_dir,
            follower_sat_idx=follower_sat_idx,
            case_label=label,
            run_index=idx,
            differential_deployment_speed_mps=speed,
        )
        results.append(result)

    metrics_csv = output_dir / "deployment_velocity_formation_acquisition_metrics.csv"
    fuel_timeseries_csv = output_dir / "deployment_velocity_cumulative_fuel_timeseries.csv"
    fuel_plot_png = output_dir / "deployment_velocity_follower_cumulative_fuel_consumption.png"
    acq_fuel_plot_png = output_dir / "deployment_velocity_acquisition_fuel_vs_deployment_speed.png"
    acq_time_plot_png = output_dir / "deployment_velocity_acquisition_time_vs_deployment_speed.png"
    acq_time_fuel_plot_png = output_dir / "deployment_velocity_acquisition_time_and_fuel_vs_deployment_speed.png"
    oed_plot_png = output_dir / "deployment_velocity_oed_da_di_comparison.png"
    oed_acq_plot_png = output_dir / "deployment_velocity_oed_da_de_di_comparison.png"
    oed_all_plot_png = output_dir / "deployment_velocity_oed_all_components_comparison.png"
    rtn_3d_plot_png = output_dir / "deployment_velocity_rtn_relative_position_3d_selected_runs.png"
    rtn_comp_plot_png = output_dir / "deployment_velocity_rtn_relative_position_components_selected_runs.png"
    eps_overview_plot_png = output_dir / "deployment_velocity_eps_overview_all_runs.png"
    operational_modes_plot_png = output_dir / "deployment_velocity_operational_modes_all_runs.png"
    burn_metrics_plot_png = output_dir / "deployment_velocity_acquisition_burn_metrics_vs_deployment_speed.png"

    write_metrics_csv(results, metrics_csv, run_dir_reference=output_dir)
    write_cumulative_fuel_timeseries_csv(results, fuel_timeseries_csv)
    plot_cumulative_follower_fuel_consumption(results, fuel_plot_png)
    plot_acquisition_fuel_vs_differential_speed(results, acq_fuel_plot_png)
    plot_acquisition_time_vs_differential_speed(results, acq_time_plot_png)
    plot_acquisition_time_and_fuel_vs_differential_speed(results, acq_time_fuel_plot_png)
    plot_da_di_oed_for_all_runs(results, oed_plot_png)
    plot_da_de_di_oed_for_all_runs(results, oed_acq_plot_png)
    plot_all_oed_components_for_all_runs(results, oed_all_plot_png)
    plot_rtn_relative_position_3d_for_selected_runs(results, rtn_3d_plot_png, selected_rtn_run_indices)
    plot_rtn_relative_position_components_for_selected_runs(results, rtn_comp_plot_png, selected_rtn_run_indices)
    plot_eps_overview_for_all_runs(results, eps_overview_plot_png)
    plot_operational_modes_for_all_runs(results, operational_modes_plot_png)
    plot_acquisition_burn_metrics_vs_differential_speed(results, burn_metrics_plot_png)

    print(f"[OK] Wrote metrics CSV: {metrics_csv}")
    print(f"[OK] Wrote cumulative fuel time-series CSV: {fuel_timeseries_csv}")
    print(f"[OK] Wrote cumulative fuel plot: {fuel_plot_png}")
    print(f"[OK] Wrote acquisition fuel sensitivity plot: {acq_fuel_plot_png}")
    print(f"[OK] Wrote acquisition time sensitivity plot: {acq_time_plot_png}")
    print(f"[OK] Wrote combined acquisition time/fuel sensitivity plot: {acq_time_fuel_plot_png}")
    print(f"[OK] Wrote OEd da/di comparison plot: {oed_plot_png}")
    print(f"[OK] Wrote OEd da/de/di comparison plot: {oed_acq_plot_png}")
    print(f"[OK] Wrote all-component OEd comparison plot: {oed_all_plot_png}")
    print(f"[OK] Wrote selected-run RTN 3D plot: {rtn_3d_plot_png}")
    print(f"[OK] Wrote selected-run RTN component plot: {rtn_comp_plot_png}")
    print(f"[OK] Wrote EPS overview plot: {eps_overview_plot_png}")
    print(f"[OK] Wrote operational modes plot: {operational_modes_plot_png}")
    print(f"[OK] Wrote acquisition burn metrics plot: {burn_metrics_plot_png}")

    return results


def _parse_comma_separated_labels(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",")]


def _parse_comma_separated_floats(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    return [float(part.strip()) for part in value.split(",")]


def _parse_comma_separated_ints(value: Optional[str]) -> tuple[int, ...]:
    if value is None:
        return (0, 2, 4)
    return tuple(int(part.strip()) for part in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process formation-acquisition fuel sensitivity Monte Carlo runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Explicit run_XXX folders to process.",
    )
    parser.add_argument(
        "--follower-sat-idx",
        type=int,
        default=1,
        help="Follower satellite index. Default: 1.",
    )
    parser.add_argument(
        "--case-labels",
        type=str,
        default=None,
        help=(
            "Optional comma-separated labels, one per run folder. "
            "Example: 'Leader 0 m/s, follower 0 m/s,Leader +0.5 m/s, follower -0.5 m/s'"
        ),
    )
    parser.add_argument(
        "--differential-deployment-speeds-mps",
        type=str,
        default=None,
        help="Optional comma-separated differential deployment speeds [m/s], one per run. Example: '0,1,2,3,4'",
    )
    parser.add_argument(
        "--selected-rtn-run-indices",
        type=str,
        default=None,
        help="Optional comma-separated zero-based run indices for RTN plots. Default: '0,2,4'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder. Default: shared Monte_Carlo_<timestamp> parent folder.",
    )

    args = parser.parse_args()

    run_deployment_velocity_fuel_sensitivity_analysis(
        run_dirs=args.run_dirs,
        follower_sat_idx=args.follower_sat_idx,
        case_labels=_parse_comma_separated_labels(args.case_labels),
        output_dir=args.output_dir,
        differential_deployment_speeds_mps=_parse_comma_separated_floats(args.differential_deployment_speeds_mps),
        selected_rtn_run_indices=_parse_comma_separated_ints(args.selected_rtn_run_indices),
    )


if __name__ == "__main__":
    main()
