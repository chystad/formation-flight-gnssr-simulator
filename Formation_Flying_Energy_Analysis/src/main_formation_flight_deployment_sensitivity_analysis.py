from pathlib import Path

from plotting.deployment_velocity_fuel_sensitivity import (
    run_deployment_velocity_fuel_sensitivity_analysis,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    mc_dir = (
        repo_root
        / "Formation_Flying_Energy_Analysis"
        / "output_data"
        / "batch_runs"
        / "Monte_Carlo_20260528_224530"
    )

    run_dirs = [
        mc_dir / "run_000",
        mc_dir / "run_001",
        mc_dir / "run_002",
        mc_dir / "run_003",
        mc_dir / "run_004",
    ]

    case_labels = [
        "Leader 0.0 m/s, follower 0.0 m/s",
        "Leader +0.5 m/s, follower -0.5 m/s",
        "Leader +1.0 m/s, follower -1.0 m/s",
        "Leader +1.5 m/s, follower -1.5 m/s",
        "Leader +2.0 m/s, follower -2.0 m/s",
    ]

    run_deployment_velocity_fuel_sensitivity_analysis(
        run_dirs=run_dirs,
        follower_sat_idx=1,
        case_labels=case_labels,
    )


if __name__ == "__main__":
    main()