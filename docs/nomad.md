# Nomad integration

The optional connector serves all three normalized PLI checkpoints in `nomad.yml` through `wamrvit.integrations.nomad.PLIForecastTool`. It uses the public regular/native-MS rollout engines; serving does not start Ray or require future ground truth.

## Install and serve

In an environment where dependency installation is permitted:

```bash
uv sync --extra nomad
uv run --extra nomad nomad serve --transport http --host 127.0.0.1 --port 8000 nomad.yml
```

Alternatively, install the package with `pip install '.[nomad]'`. The extra targets Nomad 0.2.1 and Diffusers 0.37.1. Plain WAMRViT imports do not require Nomad. Weights load in `from_pretrained`; Nomad controls each instance's device and offloading.

| Tool | Model | Input frames |
|---|---|---:|
| `pli-ms-2to1-1k` | First-1k native-MS | 2 |
| `pli-vit-2to1-1k` | First-1k ViT-finest | 2 |
| `pli-vit-1to1-full` | Full-5,300 ViT-finest | 1 |

Each entry has a unique tool name and an external HF subfolder source. The config pins the published weights by default. `WAMRVIT_HF_REVISION` can select a later reviewed revision without modifying weights.

**Model cards:** all three checkpoint-folder READMEs are published and mirrored in `docs/model_cards/`. The config pins their documentation revision; weights, configs and normalization are unchanged.

## Scientific interface

`PLIForecastInput` extends Nomad's `AutoRegressiveInput`:

- `initial_state`: a Cartesian 2D `WellFormat` with exactly ten `t0_fields`, each `(history, 1120, 400)`, in dataset source units.
- Field names: `Uvelocity`, `Wvelocity`, `av_density`, `energy_case`, `vofm_case`, `vofm_cushion`, `vofm_maincharge`, `vofm_outside_air`, `vofm_striker`, `vofm_throw`. Here `energy_case` is the sum of the six raw material energies. Raw 15-channel input must be aggregated first.
- `duration`: 1–100 future source-frame steps, not seconds. The serving cap bounds rollout buffering/response size; it is not a model accuracy claim.
- `return_trajectory`: false by default, returning the last observation and endpoint. True returns the last observation plus every predicted frame. Full trajectories can be large: 50 steps of all ten fields occupy about 0.9 GiB before transport compression.

Fields have `(time, y, x)` axes. Spatial dimension names may be omitted or set to `['y', 'x']`; supplied coordinates and scalars are preserved as metadata, not controls that rescale or parameterize the trained physics. History times, if provided, must be one source-frame step apart. Output times start at the latest supplied history time, or zero when absent.

The connector normalizes internally using the checkpoint's statistics and returns de-normalized values. It does not clip predictions, change timestep spacing, pad missing history, or apply independent boundary-condition controls. Separate vector/tensor fields and boundary-condition requests are rejected rather than silently ignored.

For native-MS, the internal mesh uses four levels, width-40/height-80 roots, initial tolerance 0.01 on velocity/density, buffer 2, and regridding every four steps with tolerances `[0.1, 0.1, 0.3]`. Regridding remaps history in source representation before renormalizing. Request batching is processed independently so different horizons/topologies are never padded together.

## Typed client and review artifacts

Supply already-downloaded PLI NPZ files. The client aggregates raw fields, creates typed inputs, decodes the response, saves HDF5 and optionally plots density input/prediction/delta:

```bash
uv run --extra nomad python examples/nomad/forecast_pli.py \
  --tool pli-vit-1to1-full --inputs /path/to/pli240420_id00951_pvi_idx00021.npz \
  --steps 5 --trajectory --output /path/to/forecast.h5 --plot /path/to/forecast.png
```

For either 2→1 tool, provide both frame-20 and frame-21 NPZ files to `--inputs`. The client reads source-frame indices from canonical PLI filenames and rejects gaps, reversed order and mixed trajectories. The client matches the published loader's energy-reduction precision: source dtype for regular checkpoints, FP32 native exports for MS. Direct typed inputs are already aggregated. No serialized tensor payload needs to be constructed by hand.

The full-cohort 1→1 model saw simulation 00951 during training. Use a separately declared held-out cohort for accuracy comparisons; the first-1k and full-cohort models are not a controlled history-length ablation.

## Validation status

Private qualification under ignored `tmp/` passed 23 core/adapter checks, 140 existing regression tests and three real-checkpoint GPU parity cases. All three models exactly matched five cached reference steps, including native-MS regridding; device offload/reassignment and the optional-extra wheel were checked. Adapter tests use official Nomad 0.2.1 class source with transport doubles.

The shared environment lacks Nomad/FastMCP, and installation was not authorized. Real `nomad serve`, MCP transport and dependency-installation tests therefore remain pending; the source-contract checks are not server validation.
