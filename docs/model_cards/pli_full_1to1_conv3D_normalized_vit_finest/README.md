---
language:
- en
license: mit
library_name: pytorch
tags:
- scientific-machine-learning
- pde
- pli
- vision-transformer
- conv3d
- wamrvit
---

# PLI ViT-finest — full-cohort Conv3D, 1→1

A regular-grid transformer for two-dimensional PLI forecasting. It predicts the next frame from one input frame and supports autoregressive rollout by feeding predictions back as inputs.

**Model:** `tonyzyl/Oceans11-PLI-collection/pli_full_1to1_conv3D_normalized_vit_finest`  
**Authors:** Yilin Zhuang, Noah Zambrano and Karthik Duraisamy  
**Code:** [WAMRViT](https://github.com/tonyzyl/wamrvit)

## Architecture

`QuadTreeTransformer` with Conv3D temporal depth 1, six transformer layers, 12 heads, embedding width 2304 and spatial RoPE. The `1120×400` grid uses fixed `10×5` patches: **8,960 tokens** and **289,018,868 parameters**.

## Inputs and outputs

Input layout: `(B, C, T, H, W)`, with reference shape **`(1, 10, 1, 1120, 400)`**. Each call predicts one absolute state, shape `(1, 10, 1, 1120, 400)`.

Channel order:

```text
Uvelocity, Wvelocity, av_density, energy_case,
vofm_case, vofm_cushion, vofm_maincharge, vofm_outside_air,
vofm_striker, vofm_throw
```

The model's `energy_case` is the sum of all six raw material-energy channels. The PLI transform aggregates 15 raw NPZ fields into these ten fields.

Use the supplied `pli_normalization.json`: standardize velocity/density, apply asinh plus standardization to summed energy, and leave volume fractions unchanged. Inverse-transform predictions to recover dataset source units. Volume fractions are dimensionless; time steps count consecutive source frames. Physical unit labels and frame-to-seconds conversion are not included in the artifact.

## Training

| Setting | Value |
|---|---|
| Data | [LANL Oceans11/HEAT PLI](https://oceans11.lanl.gov/heat/data/pli/pli240420_fp16_full/), all 5,300 trajectories / 534,663 source frames |
| Split | 100% training |
| History / output / objective | 1→1, PF4 autoregressive training |
| Epochs / seed / updates | 10 / 42 / 638,830 |
| Optimizer | AdamW, LR `1e-4`, betas `(0.9, 0.999)`, weight decay `0.01` |
| Schedule | Cosine; 1,000 warmup updates; minimum LR `1e-6` |
| Precision / averaging | BF16; final EMA weights |
| Batch / GPUs | 1 per GPU, eight GPUs, gradient accumulation 1 |
| Loss | Absolute L2 on normalized fields |

Trained from scratch using the public WAMRViT implementation. PF4 means four training rollout steps; each model call remains 1→1. Normalization reuses the frozen first-1k training-prefix statistics, fitted on 80,615 frames. For new training runs, fit statistics on the selected training partition.

Training allocation: eight 80 GB H100s, 60 CPUs and 950 GiB host RAM.

## Usage

Install the public WAMRViT package. The example loads an aggregated, unnormalized `physical_history.npy` with the input shape above. Native inference has been exercised with Python 3.12, PyTorch 2.9.1 and Diffusers 0.37.1.

```python
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from wamrvit.checkpoint import load_model
from wamrvit.dataloader.payload.transforms import PLINormalizedTransform
from wamrvit.quad.quad_utils import make_regular_centers

subfolder = "pli_full_1to1_conv3D_normalized_vit_finest"
root = Path(snapshot_download(
    "tonyzyl/Oceans11-PLI-collection", allow_patterns=[f"{subfolder}/*"]
)) / subfolder
stats = json.loads((root / "pli_normalization.json").read_text())
transform = PLINormalizedTransform(stats["input_fields"], str(root / "pli_normalization.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(root, {}).to(device).eval()
history = np.load("physical_history.npy", allow_pickle=False)
x = torch.as_tensor(transform.normalize_fields(history), dtype=torch.float32, device=device)
centers = make_regular_centers(1120, 400, p=[10, 5], device=device)
with torch.inference_mode():
    prediction = model(x, centers)
physical_prediction = transform.inverse_transform(prediction.cpu().numpy())
```

For longer forecasts, use each normalized prediction as the next input.

## Evaluation and limitations

A 50-step H100 demo used simulation 00951, input frame 21 and targets 22–71. This is a training example: the full-cohort model has no internal held-out split. The first-1k 2→1 models use different training cohorts and objectives.

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
