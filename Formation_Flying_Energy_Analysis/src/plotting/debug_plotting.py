import matplotlib.pyplot as plt
import numpy as np

from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

from object_definitions.SimData_def import SpacecraftSimData


def plot_all_formation_plots(scSimDataList: list[SpacecraftSimData]) -> None:
    
    plot_3D_RTN_leader_relative_pos_for_all_followers(scSimDataList)
    plot_RTN_component_leader_relative_pos_for_all_followers(scSimDataList)
    plt.show()



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

    import numpy as np
    import matplotlib.pyplot as plt

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig = plt.figure()
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

    plt.tight_layout()


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

    import numpy as np
    import matplotlib.pyplot as plt

    if len(scSimDataList) == 0:
        raise ValueError("scSimDataList is empty.")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
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

    plt.tight_layout()