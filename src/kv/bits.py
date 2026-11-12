"""Effective KV bit accounting including fp16 scale/zero overhead."""
from __future__ import annotations

from src.kv.quant import group_counts


def _side_bits(blocks: list[tuple[int, int]], header_tokens: int, d: int, group: int, layout: str) -> tuple[float, float]:
    total_values = (header_tokens + sum(n for n, _ in blocks)) * d
    if total_values == 0:
        return 0.0, 0.0
    nominal_total = header_tokens * d * 16
    overhead = 0
    for n, bits in blocks:
        if n <= 0:
            continue
        nominal_total += bits * n * d
        if bits < 16:
            overhead += 32 * group_counts(n, group, d, layout)
    return (nominal_total + overhead) / total_values, nominal_total / total_values


def bits_accounting(
    blocks: list[tuple[int, int]],
    header_tokens: int,
    d: int,
    group: int,
    layout: str = "channel",
) -> dict:
    """Effective/nominal bits for K and V over the whole prefix.

    `layout` is `channel`, `token`, or `kivi` (K token layout, V channel layout).
    Non-block prefix tokens (header, wrappers) are charged at fp16.
    `nominal` is the bit-width average over all prefix values without the
    scale/zero overhead; it is 16.0 for an all-fp prefix.
    """
    if layout == "kivi":
        k_eff, k_nom = _side_bits(blocks, header_tokens, d, group, "token")
        v_eff, v_nom = _side_bits(blocks, header_tokens, d, group, "channel")
        return {
            "effective_bits": (k_eff + v_eff) / 2,
            "nominal_bits": (k_nom + v_nom) / 2,
            "key_effective_bits": k_eff,
            "value_effective_bits": v_eff,
        }
    eff, nom = _side_bits(blocks, header_tokens, d, group, layout)
    return {"effective_bits": eff, "nominal_bits": nom}


def chunk_bits_accounting(
    blocks: list[tuple[int, int]],
    d: int,
    group: int,
    layout: str = "channel",
) -> dict:
    """Bit accounting over chunk values only (used by the phase-3 toy test)."""
    return bits_accounting(blocks, 0, d, group, layout)


def fp_bits() -> dict:
    return {"effective_bits": 16.0, "nominal_bits": 16.0}
