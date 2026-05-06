import glob

import h5py

from wamrvit.dataloader.utils import group_and_sort_files

class single_traj_parser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, glob_pattern: str):
        return sorted(glob.glob(glob_pattern))

class multi_traj_under_single_dir_parser:
    def __init__(self, filename_pattern: str, *args, **kwargs):
        self.filename_pattern = filename_pattern

    def __call__(self, glob_pattern: str):
        return group_and_sort_files(glob.glob(glob_pattern), filename_pattern=self.filename_pattern)


class hdf5_multi_traj_parser:
    """
    File parser for HDF5 files that contain multiple trajectories and timesteps.

    Each HDF5 file stores data with shape (n_trajectories, n_timesteps, ...).
    This parser generates virtual paths of the form 'filepath:::traj_idx:::frame_idx'
    and groups them into List[List[str]] (one inner list per trajectory).
    """
    def __init__(self, field_group: str = "t0_fields", shape_field: str = "density", *args, **kwargs):
        self.field_group = field_group
        self.shape_field = shape_field

    def __call__(self, glob_pattern: str):
        hdf5_files = sorted(glob.glob(glob_pattern))
        all_trajectories = []

        for filepath in hdf5_files:
            with h5py.File(filepath, "r") as f:
                shape = f[self.field_group][self.shape_field].shape
                n_traj, n_timesteps = shape[0], shape[1]

            for traj_idx in range(n_traj):
                frames = [
                    f"{filepath}:::{traj_idx}:::{frame_idx}"
                    for frame_idx in range(n_timesteps)
                ]
                all_trajectories.append(frames)

        print(f"Found {len(hdf5_files)} HDF5 files, {len(all_trajectories)} trajectories total.")
        return all_trajectories