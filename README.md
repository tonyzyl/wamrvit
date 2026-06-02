# WAMRViT

Adaptive Vision Transformer with quadtree-based adaptive patching for scientific spatiotemporal forecasting. Refines high-gradient regions, coarsens smooth ones.

## Visualization

### AMR

<table>
  <tr>
    <th></th>
    <th align="center">Model prediction</th>
    <th align="center">Ground truth</th>
  </tr>
  <tr>
    <th>HRR</th>
    <td><img src="assets/animations/AMR/comb_avit_2to1_target_pf4_lv5_lr1e-4_HRR_traj0_frame330_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/AMR/GT_HRR_traj0_frame330.gif" width="100%"/></td>
  </tr>
  <tr>
    <th>Pressure</th>
    <td><img src="assets/animations/AMR/comb_avit_2to1_target_pf4_lv5_lr1e-4_pressure_traj0_frame330_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/AMR/GT_pressure_traj0_frame330.gif" width="100%"/></td>
  </tr>
  <tr>
    <th>Density (ρ)</th>
    <td><img src="assets/animations/AMR/comb_avit_2to1_target_pf4_lv5_lr1e-4_rho_traj0_frame330_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/AMR/GT_rho_traj0_frame330.gif" width="100%"/></td>
  </tr>
  <tr>
    <th>Temperature</th>
    <td><img src="assets/animations/AMR/comb_avit_2to1_target_pf4_lv5_lr1e-4_T_traj0_frame330_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/AMR/GT_T_traj0_frame330.gif" width="100%"/></td>
  </tr>
</table>

Combustion trajectory, frame 330. Each row pairs the model's prediction with the ground truth at matching resolution.

### PLI

<table width="100%">
  <tr>
    <th width="33.33%" align="center">Ground truth</th>
    <th width="33.33%" align="center">WAMRViT (adaptive)</th>
    <th width="33.33%" align="center">WAMRViT (multi-scale)</th>
  </tr>
  <tr>
    <td><img src="assets/animations/PLI/GT_av_density_traj950_frame20.gif" width="100%"/></td>
    <td><img src="assets/animations/PLI/pli_avit_2to1_target_pf1_g1_a2_10_epoch_80224_lr1e-4_av_density_traj950_frame20_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/PLI/pli_avit_2to1_target_pf1_g1_a2_10_epoch_80224_lr1e-4_multi_adaln_fullfield_av_density_traj950_frame20_Pred.gif" width="100%"/></td>
  </tr>
  <tr>
    <th width="33.33%" align="center">ViT (finest)</th>
    <th width="33.33%" align="center">ViT (mid)</th>
    <th width="33.33%" align="center">SwinV2</th>
  </tr>
  <tr>
    <td><img src="assets/animations/PLI/pli_reg_finest_target_pf1_g1_10_epoch_80224_lr1e-4_av_density_traj950_frame20_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/PLI/pli_reg_mid_target_pf1_g1_10_epoch_p16_8_rope50_140_lr1e-4_av_density_traj950_frame20_Pred.gif" width="100%"/></td>
    <td><img src="assets/animations/PLI/pli_swin_finest_target_pf1_g1_10_epoch_lr1e-4_av_density_traj950_frame20_Pred.gif" width="100%"/></td>
  </tr>
</table>

av_density, PLI trajectory 951, starting frame 20. 1120 × 400. Top row: ground truth and the two adaptive WAMRViT variants; bottom row: the three regular-grid baselines.

### TRL

<table width="100%">
  <tr><th align="center">Ground truth</th></tr>
  <tr><td><img src="assets/animations/TRL/GT_density_traj0_frame20.gif" width="100%"/></td></tr>
  <tr><th align="center">WAMRViT (adaptive)</th></tr>
  <tr><td><img src="assets/animations/TRL/trl_avit_a2_200ep_uniform_density_traj0_frame20_Pred.gif" width="100%"/></td></tr>
  <tr><th align="center">WAMRViT (multi-scale)</th></tr>
  <tr><td><img src="assets/animations/TRL/trl_avit_a2_200ep_multi_adaln_fullfield_density_traj0_frame20_Pred.gif" width="100%"/></td></tr>
  <tr><th align="center">ViT (finest)</th></tr>
  <tr><td><img src="assets/animations/TRL/trl_vit_finest_200ep_bs8_default_cosine_lr1e-4_ps44_density_traj0_frame20_Pred.gif" width="100%"/></td></tr>
  <tr><th align="center">ViT (mid)</th></tr>
  <tr><td><img src="assets/animations/TRL/trl_vit_mid_200ep_bs8_lr1e-4_ps48_rope96_32_density_traj0_frame20_Pred.gif" width="100%"/></td></tr>
  <tr><th align="center">SwinV2</th></tr>
  <tr><td><img src="assets/animations/TRL/trl_swin_finest_200ep_bs8_lr1e-4_density_traj0_frame20_Pred.gif" width="100%"/></td></tr>
</table>

Density field, TRL (turbulent radiative layer) trajectory 0, frame 20. 128 × 384. The adaptive variants overlay the quadtree mesh. 

---

## Install

```bash
uv sync
uv pip install -e .
```

Python ≥ 3.12.

## Status

Code-only release: configuration files, normalization data, and example
commands are not included in this revision.

## Layout

```
train.py                    Training entry point (Ray Tune + DDP)
wamrvit/
├── quadtree_transformer.py Adaptive quadtree ViT
├── swin_transformer.py     Regular-grid SwinV2 baseline
├── quad/                   Quadtree primitives and adaptation
├── dataloader/             Data loading, topology caching
├── rollout/                Autoregressive rollout, metrics
└── visualization/          Plotting utilities
```
