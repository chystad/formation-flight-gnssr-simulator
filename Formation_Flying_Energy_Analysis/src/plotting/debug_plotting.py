import math

import matplotlib.pyplot as mpl
import numpy as np

from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport

from object_definitions.SimData_def import SpacecraftSimData
from object_definitions.FswStack_def import INT_TO_POINTING_MODE


# Global X-component vector colors
SIGMA_1_COLOR = "#1f77b4"  # blue
SIGMA_2_COLOR = "#2ca02c"  # green
SIGMA_3_COLOR = "#b59b3b"  # yellow-green / ochre
SIGMA_4_COLOR = "#9467bd"  # purple


def plot_all_formation_plots(scSimDataList: list[SpacecraftSimData]) -> None:
    plot_3D_RTN_leader_relative_pos_for_all_followers(scSimDataList)
    plot_RTN_component_leader_relative_pos_for_all_followers(scSimDataList)
    # for sat_idx in range(1, len(scSimDataList)):
    #     plot_orbital_element_difference(scSimDataList, sat_idx) Currently does not work


def plot_all_thruster_fuel_plots(scSimDataList: list[SpacecraftSimData]) -> None:
    plot_propulsion_sys_for_all_satellites(scSimDataList)


def plot_all_per_satellite_GNC_plots(scSimDataList: list[SpacecraftSimData], sat_idx: int) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_attitude_error(scSimData, sat_idx)
    plot_single_satellite_attitude_reference(scSimData, sat_idx)
    plot_single_satellite_rw_torques_and_speeds(scSimData, sat_idx)


def plot_all_eps_plots(scSimDataList: list[SpacecraftSimData], sat_idx: int, bat_storage_capacity_Wh: float) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_eps_power(scSimData, sat_idx)
    plot_single_satellite_battery_energy_fraction(scSimData, sat_idx, bat_storage_capacity_Wh)
    plot_single_satellite_eps_overview(scSimData, sat_idx, bat_storage_capacity_Wh)


def plot_all_pointing_mode_plots(scSimDataList: list[SpacecraftSimData], sat_idx: int) -> None:
    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_pointing_mode(scSimData, sat_idx)

################################
# All follower formation plots #
################################

def plot_3D_RTN_leader_relative_pos_for_all_followers(
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

    fig = mpl.figure()
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


def plot_RTN_component_leader_relative_pos_for_all_followers(
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

    fig, axs = mpl.subplots(3, 1, sharex=True, figsize=(10, 8))
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

        t_s = np.arange(rel_pos.n_samples) * rel_pos.dt_s

        if len(t_s) != r_RTN.shape[0]:
            raise ValueError(
                f"Time vector length mismatch for spacecraft #{sat_idx}. "
                f"Got len(t_s)={len(t_s)}, but position data has "
                f"{r_RTN.shape[0]} samples."
            )

        color = f"C{sat_idx - 1}"

        axs[0].plot(t_s, r_RTN[:, 0], color=color, label=f"Follower {sat_idx}")
        axs[1].plot(t_s, r_RTN[:, 1], color=color, label=f"Follower {sat_idx}")
        axs[2].plot(t_s, r_RTN[:, 2], color=color, label=f"Follower {sat_idx}")

    axs[0].set_title("Radial (R)")
    axs[1].set_title("Along-track (T)")
    axs[2].set_title("Cross-track (N)")

    axs[0].set_ylabel("d_pos [m]")
    axs[1].set_ylabel("d_pos [m]")
    axs[2].set_ylabel("d_pos [m]")
    axs[2].set_xlabel("Time [s]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()


def plot_orbital_element_difference(
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

        oed[i, 0] = (oe_follower.a - oe_leader.a) / oe_leader.a
        oed[i, 1] = oe_follower.e - oe_leader.e
        oed[i, 2] = oe_follower.i - oe_leader.i
        oed[i, 3] = oe_follower.Omega - oe_leader.Omega
        oed[i, 4] = oe_follower.omega - oe_leader.omega

        E_leader = orbitalMotion.f2E(oe_leader.f, oe_leader.e)
        E_follower = orbitalMotion.f2E(oe_follower.f, oe_follower.e)
        M_leader = orbitalMotion.E2M(E_leader, oe_leader.e)
        M_follower = orbitalMotion.E2M(E_follower, oe_follower.e)
        oed[i, 5] = M_follower - M_leader

        for j in range(3, 6):
            if oed[i, j] > math.pi:
                oed[i, j] -= 2.0 * math.pi
            if oed[i, j] < -math.pi:
                oed[i, j] += 2.0 * math.pi

    t_h = np.arange(sim_length) * leader.r_BN_N.dt_s / 3600.0

    fig, axs = mpl.subplots(6, 1, sharex=True, figsize=(10, 10))
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

    axs[-1].set_xlabel("Time [h]")

    mpl.tight_layout()






#####################################
# All satellite thruster fuel plots #
#####################################

def plot_propulsion_sys_for_all_satellites(
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

    fig, axs = mpl.subplots(3, 1, sharex=False, figsize=(10, 6))
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

        t_thrust_h = (np.arange(thrust_data.n_samples) * thrust_data.dt_s) / 60
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

        t_torque_h = (np.arange(torque_data.n_samples) * torque_data.dt_s) / 60
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

        t_fuel_h = (np.arange(fuel_data.n_samples) * fuel_data.dt_s) / 60

        axs[2].plot(
            t_fuel_h,
            fuel_mass,
            color=color,
            label=label,
        )

    axs[0].set_title("Thrust force magnitude")
    axs[0].set_xlabel("Time [min]")
    axs[0].set_ylabel("Thrust [N]")

    axs[1].set_title("Thrust torque magnitude")
    axs[1].set_xlabel("Time [min]")
    axs[1].set_ylabel("Torque [Nm]")

    axs[2].set_title("Fuel mass")
    axs[2].set_xlabel("Time [min]")
    axs[2].set_ylabel("Fuel mass [kg]")
    axs[2].ticklabel_format(axis="y", style="plain", useOffset=False)

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()







###########################
# Per satellite GNC plots #
###########################

def plot_single_satellite_attitude_error(
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

    t_s = np.arange(sigma_data.n_samples) * sigma_data.dt_s

    fig, ax = mpl.subplots(figsize=(10, 4))

    ax.plot(t_s, sigma_BR[:, 0], color=SIGMA_1_COLOR, label=r"$\sigma_1$")
    ax.plot(t_s, sigma_BR[:, 1], color=SIGMA_2_COLOR, label=r"$\sigma_2$")
    ax.plot(t_s, sigma_BR[:, 2], color=SIGMA_3_COLOR, label=r"$\sigma_3$")

    ax.set_title(f"Attitude tracking error for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel(r"Error $\sigma_{B/R}$")
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()


def plot_single_satellite_attitude_reference(
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

    t_s = np.arange(sigma_data.n_samples) * sigma_data.dt_s

    fig, ax = mpl.subplots(figsize=(10, 4))

    ax.plot(t_s, sigma_RN[:, 0], color=SIGMA_1_COLOR, label=r"$\sigma_1$")
    ax.plot(t_s, sigma_RN[:, 1], color=SIGMA_2_COLOR, label=r"$\sigma_2$")
    ax.plot(t_s, sigma_RN[:, 2], color=SIGMA_3_COLOR, label=r"$\sigma_3$")

    ax.set_title(f"Attitude reference for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\sigma_{R/N}$")
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()


def plot_single_satellite_rw_torques_and_speeds(
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

    t_speed_s = np.arange(rw_speed_data.n_samples) * rw_speed_data.dt_s
    t_actual_torque_s = np.arange(rw_actual_torque_data.n_samples) * rw_actual_torque_data.dt_s
    t_cmd_torque_s = np.arange(rw_cmd_torque_data.n_samples) * rw_cmd_torque_data.dt_s

    rwOmega_rpm = rwOmega_rads * 60.0 / (2.0 * np.pi)

    fig, axs = mpl.subplots(2, 1, sharex=False, figsize=(10, 6))
    fig.suptitle(f"Reaction wheel speed and torque for spacecraft #{sat_idx}")

    for rw_idx in range(numRWs):
        color = rw_colors[rw_idx]

        axs[0].plot(
            t_speed_s,
            rwOmega_rpm[:, rw_idx],
            color=color,
            label=fr"RW {rw_idx + 1}",
        )

        axs[1].plot(
            t_actual_torque_s,
            rwUCurrent[:, rw_idx],
            color=color,
            linestyle="-",
            label=fr"RW {rw_idx + 1} actual",
        )

        axs[1].plot(
            t_cmd_torque_s,
            cmdMotorTorque[:, rw_idx],
            color=color,
            linestyle=":",
            label=fr"RW {rw_idx + 1} cmd",
        )

    axs[0].set_ylabel("Speed [RPM]")
    axs[1].set_ylabel("Torque [Nm]")
    axs[1].set_xlabel("Time [s]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()






###########################
# Per satellite EPS plots #
###########################

def plot_single_satellite_eps_power(
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

    t_bat_h = np.arange(battery_power_data.n_samples) * battery_power_data.dt_s / 3600.0
    t_obc_h = np.arange(obc_power_data.n_samples) * obc_power_data.dt_s / 3600.0
    t_bat_heat_h = np.arange(bat_heat_power_data.n_samples) * bat_heat_power_data.dt_s / 3600.0
    t_com_h = np.arange(com_power_data.n_samples) * com_power_data.dt_s / 3600.0
    t_pay_h = np.arange(pay_power_data.n_samples) * pay_power_data.dt_s / 3600.0
    t_prop_idle_h = np.arange(prop_idle_power_data.n_samples) * prop_idle_power_data.dt_s / 3600.0
    t_prop_heat_h = np.arange(prop_heat_power_data.n_samples) * prop_heat_power_data.dt_s / 3600.0
    t_prop_thr_h = np.arange(prop_thr_power_data.n_samples) * prop_thr_power_data.dt_s / 3600.0
    t_solar_h = np.arange(solar_power_data.n_samples) * solar_power_data.dt_s / 3600.0

    total_solar_power = np.sum(solar_panel_net_power, axis=1)

    fig, axs = mpl.subplots(3, 1, sharex=False, figsize=(10, 8))
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
    axs[2].set_xlabel("Time [h]")
    axs[2].set_ylabel("Power [W]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()


def plot_single_satellite_battery_energy_fraction(
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

    t_h = np.arange(storage_data.n_samples) * storage_data.dt_s / 3600.0

    fig, ax = mpl.subplots(figsize=(10, 4))

    ax.plot(t_h, storage_fraction, label="Battery energy fraction")

    ax.set_title(f"Battery stored energy fraction for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Stored energy fraction [-]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    mpl.tight_layout()


def plot_single_satellite_eps_overview(
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
    t_storage_h = np.arange(storage_data.n_samples) * storage_data.dt_s / 3600.0
    t_battery_power_h = np.arange(battery_power_data.n_samples) * battery_power_data.dt_s / 3600.0

    t_obc_h = np.arange(obc_power_data.n_samples) * obc_power_data.dt_s / 3600.0
    t_bat_heat_h = np.arange(bat_heat_power_data.n_samples) * bat_heat_power_data.dt_s / 3600.0
    t_com_h = np.arange(com_power_data.n_samples) * com_power_data.dt_s / 3600.0
    t_pay_h = np.arange(pay_power_data.n_samples) * pay_power_data.dt_s / 3600.0
    t_prop_idle_h = np.arange(prop_idle_power_data.n_samples) * prop_idle_power_data.dt_s / 3600.0
    t_prop_heat_h = np.arange(prop_heat_power_data.n_samples) * prop_heat_power_data.dt_s / 3600.0
    t_prop_thr_h = np.arange(prop_thr_power_data.n_samples) * prop_thr_power_data.dt_s / 3600.0
    t_rw_h = np.arange(rw_power_data.n_samples) * rw_power_data.dt_s / 3600.0
    t_solar_h = np.arange(solar_power_data.n_samples) * solar_power_data.dt_s / 3600.0

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
    fig, axs = mpl.subplots(4, 1, sharex=False, figsize=(12, 11))
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
    axs[3].set_xlabel("Time [h]")
    axs[3].set_ylabel("Power [W]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    mpl.tight_layout()




######################
# Pointing Mode Plot #
######################

def plot_single_satellite_pointing_mode(
    scSimData: SpacecraftSimData,
    sat_idx: int,
) -> None:
    mode_data = scSimData.pointingModeCode
    mode_code = mode_data.data

    t_h = np.arange(mode_data.n_samples) * mode_data.dt_s / 3600.0

    fig, ax = mpl.subplots(figsize=(10, 4))
    ax.step(t_h, mode_code, where="post")

    ax.set_title(f"Pointing mode for spacecraft #{sat_idx}")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Pointing mode [-]")

    tick_values = list(INT_TO_POINTING_MODE.keys())
    tick_labels = [mode.value for mode in INT_TO_POINTING_MODE.values()]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

    ax.grid(True)
    mpl.tight_layout()