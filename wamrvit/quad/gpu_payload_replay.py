"""Plain PyTorch payload replay for a CPU-produced adaptive lineage plan."""
from __future__ import annotations

import numpy as np
import torch

from wamrvit.quad.regrid_lineage import COARSEN_OP, REFINE_OP, LineagePlan


def _operation_generations(plan: LineagePlan) -> np.ndarray:
    generations = np.zeros(plan.kind.shape[0], dtype=np.int32)
    for slot in range(plan.initial_slots, plan.kind.shape[0]):
        if plan.kind[slot] == REFINE_OP:
            generations[slot] = generations[int(plan.parent[slot])] + 1
        elif plan.kind[slot] == COARSEN_OP:
            generations[slot] = int(np.max(generations[plan.children[slot]])) + 1
        else:  # LineagePlan validation already rejects this; retain a loud boundary.
            raise ValueError(f"Unsupported generated operation at slot {slot}.")
    return generations


def _gather_slots(
    source: torch.Tensor,
    workspace: torch.Tensor,
    slots: np.ndarray | torch.Tensor,
    initial_slots: int,
) -> torch.Tensor:
    device = source.device
    slots_tensor = torch.as_tensor(slots, dtype=torch.long, device=device)
    result = torch.empty(
        (len(slots),) + tuple(source.shape[1:]),
        dtype=source.dtype,
        device=device,
    )
    source_mask = slots_tensor < int(initial_slots)
    if bool(torch.any(source_mask)):
        result[source_mask] = torch.index_select(source, 0, slots_tensor[source_mask])
    generated_mask = ~source_mask
    if bool(torch.any(generated_mask)):
        workspace_slots = slots_tensor[generated_mask] - int(initial_slots)
        result[generated_mask] = torch.index_select(workspace, 0, workspace_slots)
    return result


def _resize_bilinear_float64(
    patches: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Mirror ``resize_patch_bilinear_jit`` coordinates and float64 writes."""
    input_h, input_w = int(patches.shape[-2]), int(patches.shape[-1])
    device = patches.device
    y_scale = (input_h - 1.0) / (target_h - 1.0) if target_h > 1 else 0.0
    x_scale = (input_w - 1.0) / (target_w - 1.0) if target_w > 1 else 0.0
    src_y = torch.arange(target_h, dtype=torch.float64, device=device) * y_scale
    src_x = torch.arange(target_w, dtype=torch.float64, device=device) * x_scale
    y0 = src_y.to(torch.long)
    x0 = src_x.to(torch.long)
    if input_h > 1:
        y0 = torch.clamp(y0, max=input_h - 2)
    else:
        y0 = torch.zeros_like(y0)
    if input_w > 1:
        x0 = torch.clamp(x0, max=input_w - 2)
    else:
        x0 = torch.zeros_like(x0)
    y1 = y0 + (1 if input_h > 1 else 0)
    x1 = x0 + (1 if input_w > 1 else 0)
    dy = src_y - y0.to(torch.float64)
    dx = src_x - x0.to(torch.float64)

    values = patches.to(torch.float64)
    v00 = values[..., y0[:, None], x0[None, :]]
    v10 = values[..., y1[:, None], x0[None, :]]
    v01 = values[..., y0[:, None], x1[None, :]]
    v11 = values[..., y1[:, None], x1[None, :]]
    wy0 = (1.0 - dy)[:, None]
    wy1 = dy[:, None]
    wx0 = (1.0 - dx)[None, :]
    wx1 = dx[None, :]
    output = (
        v00 * wy0 * wx0
        + v10 * wy1 * wx0
        + v01 * wy0 * wx1
        + v11 * wy1 * wx1
    )
    return output.to(dtype=patches.dtype)


def _replay_refines(
    parents: torch.Tensor,
    quadrants: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    outputs = torch.empty_like(parents)
    h_mid, w_mid = patch_h // 2, patch_w // 2
    for quadrant in range(4):
        selected = torch.nonzero(quadrants == quadrant, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        parent_batch = torch.index_select(parents, 0, selected)
        y0, y1 = (0, h_mid) if quadrant < 2 else (h_mid, patch_h)
        x0, x1 = (0, w_mid) if quadrant in (0, 2) else (w_mid, patch_w)
        flattened = parent_batch.reshape(
            -1, int(parent_batch.shape[-2]), int(parent_batch.shape[-1])
        )
        quadrant_values = flattened[:, y0:y1, x0:x1]
        resized = _resize_bilinear_float64(quadrant_values, patch_h, patch_w)
        outputs[selected] = resized.reshape_as(parent_batch)
    return outputs


def _replay_coarsens(children: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    # children: (B,4,C,T,H,W), quadrants SW,SE,NW,NE.
    south = torch.cat((children[:, 0], children[:, 1]), dim=-1)
    north = torch.cat((children[:, 2], children[:, 3]), dim=-1)
    stitched = torch.cat((south, north), dim=-2).to(torch.float64)
    total = stitched[..., 0::2, 0::2]
    total = total + stitched[..., 1::2, 0::2]
    total = total + stitched[..., 0::2, 1::2]
    total = total + stitched[..., 1::2, 1::2]
    return (total * 0.25).to(dtype=children.dtype)[..., :patch_h, :patch_w]


def replay_payload(
    source: torch.Tensor,
    plan: LineagePlan,
    *,
    return_status: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Replay refine/coarsen lineage against a resident ``(N,C,T,H,W)`` tensor."""
    if source.ndim != 5:
        raise ValueError(f"Expected source shape (N,C,T,H,W), got {tuple(source.shape)}.")
    if int(source.shape[0]) != int(plan.initial_slots):
        raise ValueError(
            f"Source leaf count {source.shape[0]} does not match lineage "
            f"initial_slots {plan.initial_slots}."
        )
    if tuple(source.shape[-2:]) != tuple(plan.patch_shape):
        raise ValueError(
            f"Source patch shape {tuple(source.shape[-2:])} does not match "
            f"lineage {plan.patch_shape}."
        )

    generated_count = int(plan.kind.shape[0]) - int(plan.initial_slots)
    workspace = torch.empty(
        (generated_count,) + tuple(source.shape[1:]),
        dtype=source.dtype,
        device=source.device,
    )
    generations = _operation_generations(plan)
    max_generation = int(generations.max(initial=0))
    patch_h, patch_w = plan.patch_shape
    cuda_timing = return_status and source.is_cuda
    component_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "lineage_h2d": [], "gpu_refine": [], "gpu_coarsen": [], "gpu_gather": []
    }

    def _event_start(name: str):
        if not cuda_timing:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        component_events[name].append((start, end))
        return end

    total_start = torch.cuda.Event(enable_timing=True) if cuda_timing else None
    total_end = torch.cuda.Event(enable_timing=True) if cuda_timing else None
    if total_start is not None:
        total_start.record()
    lineage_end = _event_start("lineage_h2d")
    device_kind = torch.as_tensor(plan.kind, device=source.device)
    device_parent = torch.as_tensor(plan.parent, dtype=torch.long, device=source.device)
    device_children = torch.as_tensor(plan.children, dtype=torch.long, device=source.device)
    device_quadrant = torch.as_tensor(plan.quadrant, dtype=torch.long, device=source.device)
    device_final_slots = torch.as_tensor(
        plan.final_slots, dtype=torch.long, device=source.device
    )
    device_generations = torch.as_tensor(
        generations, dtype=torch.long, device=source.device
    )
    if lineage_end is not None:
        lineage_end.record()

    with torch.no_grad():
        for generation in range(1, max_generation + 1):
            generation_slots = torch.nonzero(
                device_generations == generation, as_tuple=False
            ).flatten()
            generation_kinds = device_kind[generation_slots]
            refine_slots = generation_slots[generation_kinds == REFINE_OP]
            if refine_slots.numel():
                refine_end = _event_start("gpu_refine")
                parent_slots = device_parent[refine_slots]
                parents = _gather_slots(
                    source, workspace, parent_slots, plan.initial_slots
                )
                refined = _replay_refines(
                    parents, device_quadrant[refine_slots], patch_h, patch_w
                )
                workspace.index_copy_(
                    0, refine_slots - plan.initial_slots, refined
                )
                if refine_end is not None:
                    refine_end.record()

            coarsen_slots = generation_slots[generation_kinds == COARSEN_OP]
            if coarsen_slots.numel():
                coarsen_end = _event_start("gpu_coarsen")
                flat_children = device_children[coarsen_slots].reshape(-1)
                gathered = _gather_slots(
                    source, workspace, flat_children, plan.initial_slots
                )
                children = gathered.reshape(
                    coarsen_slots.numel(), 4, *source.shape[1:]
                )
                coarsened = _replay_coarsens(children, patch_h, patch_w)
                workspace.index_copy_(
                    0, coarsen_slots - plan.initial_slots, coarsened
                )
                if coarsen_end is not None:
                    coarsen_end.record()

        gather_end = _event_start("gpu_gather")
        output = _gather_slots(
            source, workspace, device_final_slots, plan.initial_slots
        )
        if gather_end is not None:
            gather_end.record()

    if total_end is not None:
        total_end.record()
    if not return_status:
        return output
    if cuda_timing:
        torch.cuda.synchronize(source.device)
        timings = {
            name: sum(start.elapsed_time(end) for start, end in events) / 1000.0
            for name, events in component_events.items()
        }
        timings["gpu_replay"] = total_start.elapsed_time(total_end) / 1000.0
    else:
        timings = {name: None for name in component_events}
        timings["gpu_replay"] = None
    return output, {
        "timings": timings,
        "lineage_bytes": int(
            plan.kind.nbytes + plan.parent.nbytes + plan.children.nbytes
            + plan.quadrant.nbytes + plan.final_slots.nbytes + generations.nbytes
        ),
        "operation_nodes": int(plan.kind.shape[0]),
        "final_slots": int(plan.final_slots.shape[0]),
    }
