"""Consumer-side push-forward windows for loaded sequence batches.

Used by uniform and multi-scale training, rollout, and animation. Prediction
feedback is detached before entering the next input window. This helper depends
only on Torch, not models, losses, source loaders, or transport.
"""

import torch


def advance_window(
    curr_input: torch.Tensor | dict[int, torch.Tensor],
    pred: torch.Tensor | dict[int, torch.Tensor],
    input_seq_len: int,
    return_seq_len: int,
    pred_mode: str,
) -> torch.Tensor | dict[int, torch.Tensor]:
    """Roll a push-forward input window forward by one step.

    Shared by the uniform-adaptive path (tensor state ``(N, C, T, H, W)``) and
    the multi-scale path (per-level dict ``{lvl: (N_l, C, T, H_l, W_l)}``).
    Returns the next input window (detached, ``T == input_seq_len``), same
    representation as ``curr_input``. Detaching matches the uniform path: no
    gradient crosses a push-forward step. The tensor branch is bit-identical to
    the inline roll the uniform loop used previously.

    Args:
        curr_input: current input window (tensor or per-level dict).
        pred: model prediction for this step (same representation as curr_input,
            ``T == return_seq_len``).
        input_seq_len: number of input frames to carry into the next step.
        return_seq_len: number of frames predicted per step.
        pred_mode: ``"residual"`` reconstructs the absolute frame
            (``pred + last_input_frame``) before feeding back; ``"target"`` feeds
            ``pred`` directly. For ``return_seq_len > 1`` in residual mode, the
            same last input frame is the reconstruction baseline for all predicted
            frames (broadcast over the time axis).
    """
    if isinstance(curr_input, dict):
        assert curr_input.keys() == pred.keys(), "advance_window: level keys must match"
        return {
            lvl: advance_window(
                curr_input[lvl], pred[lvl], input_seq_len, return_seq_len, pred_mode
            )
            for lvl in curr_input
        }
    if pred_mode == "residual":
        next_pred = pred.detach() + curr_input[:, :, -1].unsqueeze(2)
    elif pred_mode == "target":
        next_pred = pred.detach()
    else:
        raise ValueError(f"advance_window: unknown pred_mode {pred_mode!r}")
    num_from_input = max(input_seq_len - return_seq_len, 0)
    if num_from_input > 0:
        return torch.cat((curr_input[:, :, -num_from_input:], next_pred), dim=2)
    return next_pred[:, :, -input_seq_len:]
