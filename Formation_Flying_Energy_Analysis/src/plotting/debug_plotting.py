import matplotlib.pyplot as mpl
import numpy as np

from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

from object_definitions.SimData_def import SpacecraftSimData


# Global X-component vector colors
SIGMA_1_COLOR = "#1f77b4"  # blue
SIGMA_2_COLOR = "#2ca02c"  # green
SIGMA_3_COLOR = "#b59b3b"  # yellow-green / ochre
SIGMA_4_COLOR = "#9467bd"  # purple


def plot_all_formation_plots(scSimDataList: list[SpacecraftSimData]) -> None:
    plot_3D_RTN_leader_relative_pos_for_all_followers(scSimDataList)
    plot_RTN_component_leader_relative_pos_for_all_followers(scSimDataList)


def plot_all_thruster_fuel_plots(scSimDataList: list[SpacecraftSimData]) -> None:
    plot_propulsion_sys_for_all_satellites(scSimDataList)


def plot_all_per_satellite_GNC_plots(scSimDataList: list[SpacecraftSimData], sat_idx: int) -> None:

    scSimData = scSimDataList[sat_idx]
    plot_single_satellite_attitude_error(scSimData, sat_idx)
    plot_single_satellite_attitude_reference(scSimData, sat_idx)
    plot_single_satellite_rw_torques_and_speeds(scSimData, sat_idx)


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

        t_thrust_h = (np.arange(thrust_data.n_samples) * thrust_data.dt_s)
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

        t_torque_h = (np.arange(torque_data.n_samples) * torque_data.dt_s)
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

        t_fuel_h = (np.arange(fuel_data.n_samples) * fuel_data.dt_s)

        axs[2].plot(
            t_fuel_h,
            fuel_mass,
            color=color,
            label=label,
        )

    axs[0].set_title("Thrust force magnitude")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Thrust [N]")

    axs[1].set_title("Thrust torque magnitude")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Torque [Nm]")

    axs[2].set_title("Fuel mass")
    axs[2].set_xlabel("Time [s]")
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