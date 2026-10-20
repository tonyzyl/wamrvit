"""Spatial transformer attention with full-head spatial and level RoPE."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import logging
from torch.nn.attention.flex_attention import flex_attention

logger = logging.get_logger(__name__)


class RopeAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. "
                "Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,  # (B, N, C)
        encoder_hidden_states: torch.Tensor = None,  # (B, N_enc, C)
        image_rotary_emb: torch.Tensor | None = None,
        cond_image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        level_idx: torch.Tensor | None = None,  # (N,) long, or None
    ) -> torch.Tensor:

        # QKV projections → (B, N, inner_dim)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # dispatch_attention_fn expects (B, N, H, D_head).
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)  # (B, N_enc, inner_dim)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

        query = attn.norm_q(query)  # (B, N, H, D_head)
        key = attn.norm_k(key)

        # Apply Rotary Embeddings if provided (Note seq dim=1 for B,N,H,D).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        _use_scale_bias = (
            hasattr(attn, "use_scale_bias") and attn.use_scale_bias and level_idx is not None
        )

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))  # (B, N_enc, H, D_head)
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            if cond_image_rotary_emb is not None:
                encoder_query = apply_rotary_emb(
                    encoder_query,
                    cond_image_rotary_emb,
                    sequence_dim=1,
                )
                encoder_key = apply_rotary_emb(
                    encoder_key,
                    cond_image_rotary_emb,
                    sequence_dim=1,
                )

            # Concat along seq axis → (B, N+N_enc, H, D_head).
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if _use_scale_bias:
            # flex_attention with inline score_mod — bias computation fused
            # into the kernel (no N×N materialization).
            _scale_bias_table = attn.scale_bias_table  # (2*max_level_idx+1, H)
            _max_level_idx = attn.max_level_idx
            _level_idx = level_idx  # (N,) long

            def _scale_bias_score_mod(score, b, h, q_idx, kv_idx):
                delta = _level_idx[kv_idx] - _level_idx[q_idx]
                delta = delta.clamp(-_max_level_idx, _max_level_idx)
                return score + _scale_bias_table[delta + _max_level_idx, h]

            # flex_attention expects (B, H, N, D_head).
            hidden_states = flex_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                score_mod=_scale_bias_score_mod,
            ).transpose(1, 2)  # back to (B, N, H, D_head)
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )  # (B, N, H, D_head)

        # Merge heads → (B, N, inner_dim).
        hidden_states = hidden_states.flatten(2)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is None:
            return hidden_states
        else:
            return hidden_states, encoder_hidden_states


class RopeAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = RopeAttnProcessor
    _available_processors = [RopeAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
        use_scale_bias: bool = False,
        max_level_idx: int = 2,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        # Q/K/V projections
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm (head_dim unchanged — applies per-head)
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.use_scale_bias = use_scale_bias
        if use_scale_bias:
            self.max_level_idx = max_level_idx
            self.scale_bias_table = nn.Parameter(torch.zeros(2 * max_level_idx + 1, self.heads))

        # to_out: inner_dim → out_dim
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor | None = None,
        cond_image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        level_idx: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs["level_idx"] = level_idx
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            cond_image_rotary_emb,
            attention_mask=attention_mask,
            **kwargs,
        )
