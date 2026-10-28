# Model cards

Local copies of the cards published in the [Hugging Face collection](https://huggingface.co/tonyzyl/Oceans11-PLI-collection):

| Model | History → output | Training | Card |
|---|---|---|---|
| Full-cohort ViT-finest | 1→1 | 5,300 trajectories, PF4 | [README](pli_full_1to1_conv3D_normalized_vit_finest/README.md) |
| First-1k ViT-finest | 2→1 | 0.8 training-window prefix, PF1 | [README](pli_1k_conv3D_normalized_vit_finest/README.md) |
| First-1k WAMRViT-MS | 2→1 | 0.8 training-window prefix, PF1 | [README](pli_1k_conv3D_normalized_wamrvit_ms/README.md) |

Each card covers architecture, fields, normalization, training, inference and evaluation. Publication changed only the three folder READMEs. Connector usage is documented in [the integration guide](../nomad.md).
