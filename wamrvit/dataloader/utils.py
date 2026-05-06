import numpy as np

import re
from dataclasses import dataclass
from collections import defaultdict
import warnings

from typing import Dict, List, Optional, Tuple, Union

def parse_hdf5_virtual_path(virtual_path: str) -> Tuple[str, int, int]:
    """Parse 'filepath:::traj_idx:::frame_idx' into (filepath, traj_idx, frame_idx)."""
    parts = virtual_path.rsplit(":::", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid HDF5 virtual path: {virtual_path!r}. Expected 'filepath:::traj_idx:::frame_idx'.")
    return parts[0], int(parts[1]), int(parts[2])

import torch


@dataclass
class AmReXFileOutput():
    """
    Output of AmReX file extraction.

    Args:
        cells (np.ndarray): Cell-centered data at different refinement levels. Shape: (num_cells, C, H_cell, W_cell).
        centers (np.ndarray): Cell center characteristics corresponding to the extracted data, e.g., (num_cells, 3) for 2D and cell characteristics.
    """
    cells: np.ndarray
    centers: np.ndarray


def get_amrex_field_names(yt_obj) -> List[str]:
    return [i[1] for i in yt_obj.field_list]

    
def precompute_mean_std(normalization_param_dict: Dict, variable_names: Optional[List[str]]=None, return_tensor:bool = True):
    """
    Precompute the mean and std tensors for the given variables. variable_names shall match the order of the input.
    Return shape: (C,)
    """
    mean_list = []
    std_list = []

    if variable_names is None:
        variable_names = list(normalization_param_dict.keys())
        warnings.warn(f"Variable names not provided. Using all variables from normalization dict: {variable_names}")

    for var_name in variable_names:
        if var_name in normalization_param_dict:
            norm_params = normalization_param_dict[var_name]

            if isinstance(norm_params["mean"], dict):
                # If the variable has level-based mean and std, create one tensor per level
                for level in norm_params["mean"].keys():
                    mean_list.append(norm_params["mean"][level])
                    std_list.append(norm_params["std"][level])
            else:
                # For regular variables, add mean and std directly
                mean_list.append(norm_params["mean"])
                std_list.append(norm_params["std"])
        else:
            # If no normalization info is found, use 0 mean and 1 std (no normalization)
            raise ValueError(
                f"No normalization parameters found for variable {var_name}."
            )
            # mean_list.append(0)
            # std_list.append(1)
            # Warning(f"No normalization parameters found for variable {var_name}. Using 0 mean and 1 std.")

    # Convert lists to PyTorch tensors
    if return_tensor:
        return torch.tensor(mean_list, dtype=torch.float32), torch.tensor(std_list, dtype=torch.float32)
    else: # np.arrays
        return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)


def group_and_sort_files(
    file_list: List[str], 
    filename_pattern: str = r".*id(\d+).*idx(\d+).*\.npz"
) -> List[List[str]]:
    """
    Parses a flat list of filenames, groups them by Simulation ID, 
    and sorts them by Timestamp ID.

    Args:
        file_list: A flat list of file paths (strings).
        filename_pattern: Regex to extract (sim_id, timestamp_id). 
                          Default matches: ...idXXXX...idxXXXX.npz
    
    Returns:
        A list of lists. Each inner list represents one distinct trajectory (sim_id),
        sorted chronologically by timestamp_id. The outer list is sorted by sim_id.
    """
    pattern = re.compile(filename_pattern)
    
    # Dictionary to group files: { sim_id: [(time_idx, filepath), ...] }
    grouped_files = defaultdict(list)
    
    print(f"Grouping {len(file_list)} files...")

    for filepath in file_list:
        match = pattern.search(filepath)
        if match:
            sim_id = int(match.group(1))
            time_idx = int(match.group(2))
            # Store tuple to avoid re-parsing during sort
            grouped_files[sim_id].append((time_idx, filepath))
        else:
            # Optional: Log warning for files that don't match pattern
            # print(f"Warning: Skipping file {filepath} (no regex match)")
            pass

    # Sort the Group IDs (Simulation IDs)
    sorted_sim_ids = sorted(grouped_files.keys())
    
    final_sorted_list = []
    
    for sim_id in sorted_sim_ids:
        # Get the list of (time, path) tuples for this simulation
        trajectories = grouped_files[sim_id]
        
        # Sort by time_idx (the first element of the tuple)
        trajectories.sort(key=lambda x: x[0])
        
        # Strip the time_idx back out, keeping only the filepath
        sorted_paths = [path for _, path in trajectories]
        
        final_sorted_list.append(sorted_paths)

    print(f"Found {len(final_sorted_list)} unique trajectories.")
    return final_sorted_list


