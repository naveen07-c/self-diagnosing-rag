"""Asymmetric min-max fake quantization: quantize then dequantize in one step."""
from __future__ import annotations

import torch


def qdq(x: torch.Tensor, bits: int, dim: int, group: int) -> torch.Tensor:
    """Group-wise asymmetric fake quantization along `dim`.

    Groups of `group` values along `dim`; the final group may be smaller. The
    arithmetic is done in fp32 for stability and cast back to the input dtype.
    A constant group is returned unchanged (scale guard).
    """
    if bits >= 16:
        return x
    if group <= 0:
        raise ValueError("group must be positive")
    orig_dtype = x.dtype
    xf = x.float()
    dim = dim % xf.dim()
    n = xf.shape[dim]
    pad = (-n) % group
    if pad:
        shape = list(xf.shape)
        shape[dim] = pad
        xf = torch.cat([xf, torch.zeros(shape, dtype=xf.dtype, device=xf.device)], dim=dim)
    n_padded = xf.shape[dim]
    n_groups = n_padded // group
    shape = list(xf.shape)
    grouped = xf.reshape(shape[:dim] + [n_groups, group] + shape[dim + 1:])
    gdim = dim + 1
    mn = grouped.amin(dim=gdim, keepdim=True)
    mx = grouped.amax(dim=gdim, keepdim=True)
    levels = float(2**bits - 1)
    span = mx - mn
    scale = torch.where(span > 0, span / levels, torch.ones_like(span))
    q = torch.round((grouped - mn) / scale).clamp_(0, levels)
    dq = q * scale + mn
    dq = torch.where(span > 0, dq, mn.expand_as(dq))
    out = dq.reshape(shape)
    if pad:
        out = out.narrow(dim, 0, n)
    return out.to(orig_dtype)


def qdq_channel(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Per-token groups along the channel dimension (used by V and uniform K)."""
    return qdq(x, bits, dim=-1, group=group)


def qdq_token(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Per-channel groups along the token dimension (KIVI-style keys)."""
    return qdq(x, bits, dim=-2, group=group)


def quantize_block(x: torch.Tensor, bits: int, layout: str, group: int) -> torch.Tensor:
    if bits >= 16:
        return x
    if layout == "channel":
        return qdq_channel(x, bits, group)
    if layout == "token":
        return qdq_token(x, bits, group)
    raise ValueError(f"unknown layout: {layout}")


def group_counts(n: int, group: int, d: int, layout: str) -> int:
    """Number of quantizer groups for a block of `n` tokens and `d` channels."""
    if layout == "channel":
        return n * ((d + group - 1) // group)
    return d * ((n + group - 1) // group)
