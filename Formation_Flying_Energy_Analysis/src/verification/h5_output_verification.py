import h5py
import numpy as np
from pathlib import Path


def check_low_rate_times(h5_path, chunk_size=1_000_000):
    """
    Check that all time samples in lowRateTimes are evenly spaced
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as h5:
        dset = h5["lowRateTimes"]["data"]

        n = dset.shape[0]
        if n < 3:
            print("Not enough samples to check spacing.")
            return

        expected_dt = dset[1] - dset[0]
        prev = dset[0]

        uneven_count = 0

        for start in range(1, n, chunk_size):
            stop = min(start + chunk_size, n)

            times = dset[start:stop]
            diffs = times - np.concatenate(([prev], times[:-1]))

            bad_local = np.where(diffs != expected_dt)[0]

            for idx in bad_local:
                global_idx = start + idx
                print(
                    f"Uneven sample at index {global_idx}: "
                    f"t_prev={dset[global_idx - 1]}, "
                    f"t={dset[global_idx]}, "
                    f"dt={diffs[idx]}, "
                    f"expected_dt={expected_dt}"
                )

            uneven_count += len(bad_local)
            prev = times[-1]

        if uneven_count == 0:
            print(f"All {n} samples are evenly spaced. dt = {expected_dt}")
        else:
            print(f"Found {uneven_count} uneven sample(s).")


if __name__ == "__main__":
    check_low_rate_times("sat_0.h5")