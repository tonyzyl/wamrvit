"""Load local forecasting checkpoints without coupling callers to their format."""

import json
import logging
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _model_class(name: str):
    if name == "SwinV2Transformer":
        from wamrvit.swin_transformer import SwinV2Transformer

        return SwinV2Transformer
    # Preserve the forecasting consumers' historical QuadTreeTransformer fallback.
    from wamrvit.quadtree_transformer import QuadTreeTransformer

    return QuadTreeTransformer


def load_model(path: str | os.PathLike[str], model_cfg: Mapping[str, Any]):
    """Load a Diffusers directory or a raw state dict constructed from ``model_cfg``.

    Directory checkpoints own their architecture configuration; saved class metadata
    takes precedence over the configured class when present. Without that metadata,
    use the configured class (QuadTreeTransformer by default), as legacy callers did.
    Raw state dicts may have one leading DDP ``module.`` prefix. Device placement and
    evaluation mode remain the caller's responsibility.
    """
    path = Path(path)
    model_kwargs = dict(model_cfg)
    name = model_kwargs.pop("_class_name", "QuadTreeTransformer")
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                name = json.load(handle).get("_class_name", name)
        logger.info("Loading %s via diffusers from: %s", name, path)
        return _model_class(name).from_pretrained(path)

    model_cls = _model_class(name)
    if path.name == "model.pt" and path.with_name("ema.pt").is_file():
        warnings.warn(
            f"Loading raw training weights from {path}; a sibling ema.pt exists. "
            "EMA weights are not applied. Use the final Diffusers model/ export "
            "for EMA weights when training used EMA.",
            UserWarning,
            stacklevel=2,
        )
    logger.info("Loading %s state dict from: %s", name, path)
    model = model_cls(**model_kwargs)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model
