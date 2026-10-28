---
language:
- en
license: mit
library_name: pytorch
tags:
- scientific-machine-learning
- pde
- pli
- adaptive-mesh-refinement
- vision-transformer
- conv3d
- wamrvit
---

# PLI WAMRViT-MS — first-1k native multiscale Conv3D, 2→1

An adaptive transformer for two-dimensional PLI forecasting. Two consecutive frames predict the next state. Each quadtree leaf becomes one token, with native-resolution samples retained before learned shrink/projection.

**Model:** `tonyzyl/Oceans11-PLI-collection/pli_1k_conv3D_normalized_wamrvit_ms`  
**Authors:** Yilin Zhuang, Noah Zambrano and Karthik Duraisamy  
**Code:** [WAMRViT](https://github.com/tonyzyl/wamrvit)

## Architecture

`QuadTreeTransformer` with Conv3D history length 2, six transformer layers, 12 heads, embedding width 2304, four native levels, level-conditioned normalization and `(x, y, level_idx)` RoPE. **339,013,944 parameters**; token count varies with topology.

## Inputs and outputs

The trained source grid is **`1120×400`**. Channel order:

```text
Uvelocity, Wvelocity, av_density, energy_case,
vofm_case, vofm_cushion, vofm_maincharge, vofm_outside_air,
vofm_striker, vofm_throw
```

The model's `energy_case` sums all six raw material-energy channels. The PLI transform aggregates 15 raw NPZ fields into these ten fields. Use the supplied `pli_normalization.json`: standardized velocity/density, asinh plus standardized summed energy, and unchanged volume fractions.

| Native input | Layout |
|---|---|
| `values_by_level[L]` | Normalized `(N_L, 10, 2, H_L, W_L)` histories |
| `leaf_to_bucket` | Integer `(N, 2)` mapping to `(level, bucket_row)` |
| `centers` | `(N, 3)`: normalized `(x, y)` and level index |
| Output per level | Normalized `(N_L, 10, 1, H_L, W_L)` absolute-state predictions |

Levels 0–3 retain `80×40`, `40×20`, `20×10` and `10×5` patches. Both history frames share the current topology; `cell_scale_mode` is `level_idx`.

Inverse normalization restores dataset source units. Volume fractions are dimensionless; time steps count consecutive source frames. Physical unit labels and frame-to-seconds conversion are not included in the artifact.

## Reference inference settings

- Root tiles: width 40, height 80; four levels and native storage.
- Initial refinement: tolerance `0.01` on velocity/density, buffer 2.
- Regridding: every four forecast frames, tolerances `[0.1, 0.1, 0.3]`, buffer 2; array backend with `cpu_eager` payload handling.
- Remap retained history in source representation, then normalize for the next call.

Training refinement also used the six raw material-energy channels; the reference inference detector uses velocity/density. Use the public quadtree, regridding and scattering routines to preserve the geometry/value mapping.

## Training

| Setting | Value |
|---|---|
| Data | [LANL Oceans11/HEAT PLI](https://oceans11.lanl.gov/heat/data/pli/pli240420_fp16_full/), first 1,000 ordered trajectories |
| Split | 0.8 fixed training-window prefix |
| History / output / objective | 2→1, PF1 |
| Epochs / seed / updates | 10 / 42 / 98,310 |
| Optimizer | AdamW, LR `1e-4`, betas `(0.9, 0.999)`, weight decay `0.01` |
| Schedule | Cosine; 1,000 warmup updates; minimum LR `1e-6` |
| Precision / averaging | BF16; final EMA weights |
| Batch / GPUs | 1 per GPU, eight GPUs, gradient accumulation 1 |
| Loss | Full-field absolute L2 on normalized fields |

Normalization uses frozen first-1k training-prefix statistics fitted on 80,615 frames. For new training runs, fit statistics on the selected training partition.

## Usage

Install the public WAMRViT package. Native inference has been exercised with Python 3.12, PyTorch 2.9.1 and Diffusers 0.37.1. This helper takes physical-field buckets and matching topology prepared with the public preprocessing routines:

```python
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from wamrvit.checkpoint import load_model
from wamrvit.dataloader.payload.transforms import PLINormalizedTransform

subfolder = "pli_1k_conv3D_normalized_wamrvit_ms"
root = Path(snapshot_download(
    "tonyzyl/Oceans11-PLI-collection", allow_patterns=[f"{subfolder}/*"]
)) / subfolder
stats = json.loads((root / "pli_normalization.json").read_text())
transform = PLINormalizedTransform(stats["input_fields"], str(root / "pli_normalization.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(root, {}).to(device).eval()

def predict_one_native_step(physical_buckets, leaf_to_bucket, centers):
    values = {
        level: torch.as_tensor(
            transform.normalize_fields(np.asarray(value, dtype=np.float32)),
            dtype=torch.float32, device=device,
        )
        for level, value in physical_buckets.items()
    }
    mapping = torch.as_tensor(leaf_to_bucket, dtype=torch.long, device=device)
    positions = torch.as_tensor(centers, dtype=torch.float32, device=device)
    with torch.inference_mode():
        prediction = model.forward_multi_scale(values, mapping, positions)
    return {
        level: transform.inverse_transform(value.cpu().numpy())
        for level, value in prediction.items()
    }
```

Scatter the returned physical buckets using the same leaf mapping and geometry. Multi-step forecasting additionally updates history and regrids according to the settings above.

## Evaluation and limitations

The normalized MS/ViT pair was evaluated over 20 steps using physical-space RMSE and VRMSE, and exercised in a 50-step H100 demo. These are separate checkpoints from the historical raw-field manuscript models.

Intended for PLI surrogate-model research. Autoregressive errors can accumulate; conservation, positivity and volume-fraction bounds are not explicitly enforced. Predictions are deterministic, without calibrated uncertainty estimates.

## License and citation

MIT. See the [project](https://github.com/tonyzyl/wamrvit) for code, acknowledgments and contact.

```bibtex
@misc{zhuang2026dynamicquadtreetokenizationtransformer,
  title={Dynamic Quadtree Tokenization and Transformer for Adaptive Mesh PDE Forecasting},
  author={Yilin Zhuang and Noah Zambrano and Karthik Duraisamy},
  year={2026},
  eprint={2610.04044},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.04044},
}
```
