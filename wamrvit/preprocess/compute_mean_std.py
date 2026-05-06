import numpy as np
import ray
import yt
import json
from typing import List, Dict, Tuple

state_vars = [
    'HRR', 'T', 'mach', 'pressure', 'rho', 'rhoE', 'rhoUx', 'rhoUy', 
    'rhoY_H', 'rhoY_H2', 'rhoY_H2O', 'rhoY_H2O2', 'rhoY_HO2',
    'rhoY_N2', 'rhoY_O', 'rhoY_O2', 'rhoY_OH'
]

@ray.remote
def _stats_task(path: str, fields: List[str]) -> Dict[str, np.ndarray]:
    """Computes stats for a single snapshot efficiently using yt."""
    try:
        ds = yt.load(path)
        C = len(fields)
        counts = np.zeros(C, dtype=np.int64)
        sums = np.zeros(C, dtype=np.float64)
        sumsqs = np.zeros(C, dtype=np.float64)
        
        # Load all data into memory for fast iteration
        ad = ds.all_data()
        
        for ci, fname in enumerate(fields):
            try:
                # Access the field (triggers IO)
                arr = ad[fname]
                vals = np.asarray(arr.d, dtype=np.float64)
                
                if vals.size == 0: 
                    continue
                
                # Filter NaNs/Infs
                m = np.isfinite(vals)
                if not np.any(m): 
                    continue
                
                v = vals[m]
                
                counts[ci] += v.size
                sums[ci] += float(v.sum())
                sumsqs[ci] += float((v * v).sum())
            except Exception:
                # Handle missing fields in specific snapshots gracefully
                continue
                
        return {"counts": counts, "sums": sums, "sumsqs": sumsqs, "success": True}
        
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return {"success": False}

def compute_and_save_stats(paths: List[str], max_concurrency: int = 16):
    """
    Computes global mean/std and saves to a JSON file matching the target format.
    """
    print(f"Computing statistics on {len(paths)} files...", flush=True)
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Dispatch tasks
    futures = [_stats_task.remote(p, state_vars) for p in paths]
    
    C = len(state_vars)
    total_counts = np.zeros(C, dtype=np.int64)
    total_sums = np.zeros(C, dtype=np.float64)
    total_sumsqs = np.zeros(C, dtype=np.float64)
    
    # Process results as they complete
    while futures:
        done, futures = ray.wait(futures, num_returns=min(len(futures), max_concurrency))
        for res in ray.get(done):
            if res.get("success", False):
                total_counts += res["counts"]
                total_sums += res["sums"]
                total_sumsqs += res["sumsqs"]
            
    # Compute final statistics
    results = {}
    print("Aggregating results...", flush=True)

    for ci, var_name in enumerate(state_vars):
        if total_counts[ci] > 0:
            mean_val = total_sums[ci] / total_counts[ci]
            # Variance = E[X^2] - (E[X])^2
            var_val = max(0.0, (total_sumsqs[ci] / total_counts[ci]) - mean_val ** 2)
            std_val = np.sqrt(var_val)
            
            # Avoid division by zero issues in normalization later
            if std_val < 1e-12:
                std_val = 1.0
        else:
            mean_val = 0.0
            std_val = 1.0
            print(f"Warning: No valid data found for variable {var_name}")

        # Construct dictionary entry
        results[var_name] = {
            "mean": float(mean_val),
            "std": float(std_val)
        }

    # Save to JSON
    output_filename = "dataset_normal.json"
    try:
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Stats saved to {output_filename}")
        
        # Optional: Print preview
        print("Preview:", json.dumps({k: results[k] for k in list(results)[:2]}, indent=2))
        
    except Exception as e:
        print(f"Error writing to {output_filename}: {e}")

if __name__ == "__main__":
    # Example usage:
    # file_paths = ["/path/to/data/plt00001", "/path/to/data/plt00002"]
    # compute_and_save_stats(file_paths)
    pass