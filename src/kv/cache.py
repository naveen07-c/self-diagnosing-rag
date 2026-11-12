"""Per-chunk KV precision maps and in-place fake quantization of a DynamicCache."""
from __future__ import annotations

import random

import torch

from src.kv.bits import bits_accounting

VALID_MODES = {"fp", "uniform", "kivi", "guided", "random_guided", "oracle_guided"}


def resolve_kv(cfg: dict | None) -> dict:
    cfg = cfg or {}
    mode = cfg.get("mode", "fp")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown kv mode: {mode}")
    layout = cfg.get("layout")
    if mode == "kivi":
        layout = "kivi"
    elif layout is None:
        layout = "channel"
    return {
        "mode": mode,
        "bits": int(cfg.get("bits", 16)),
        "bits_high": int(cfg.get("bits_high", 8)),
        "bits_low": int(cfg.get("bits_low", 2)),
        "group": int(cfg.get("group", 32)),
        "layout": layout,
        "protected_p": int(cfg.get("protected_p", 0)),
        "seed": int(cfg.get("seed", 1234)),
        "sink_tokens": int(cfg.get("sink_tokens", 4)),
    }


def select_protected(
    chunks: list[dict],
    mode: str,
    p: int,
    seed: int = 1234,
    gold_titles: list[str] | None = None,
) -> set[str]:
    """Choose which chunks keep high precision. Ties break on chunk id."""
    ids = [c["chunk_id"] for c in chunks]
    if mode == "guided":
        ordered = sorted(
            chunks, key=lambda c: (-float(c.get("reranker_score", 0.0)), c["chunk_id"])
        )
        return {c["chunk_id"] for c in ordered[: max(0, min(p, len(ordered)))]}
    if mode == "random_guided":
        rng = random.Random(seed)
        return set(rng.sample(ids, max(0, min(p, len(ids)))))
    if mode == "oracle_guided":
        gold = set(gold_titles or [])
        return {c["chunk_id"] for c in chunks if c.get("title") in gold}
    return set()


def build_bits_map(
    chunks: list[dict],
    kv: dict,
    gold_titles: list[str] | None = None,
) -> dict[str, int]:
    mode = kv["mode"]
    if mode == "fp":
        return {c["chunk_id"]: 16 for c in chunks}
    if mode in ("uniform", "kivi"):
        return {c["chunk_id"]: kv["bits"] for c in chunks}
    protected = select_protected(
        chunks, mode, kv["protected_p"], seed=kv["seed"], gold_titles=gold_titles
    )
    return {
        c["chunk_id"]: (kv["bits_high"] if c["chunk_id"] in protected else kv["bits_low"])
        for c in chunks
    }


def _qdq_side(x: torch.Tensor, bits: int, layout: str, group: int) -> torch.Tensor:
    from src.kv.quant import quantize_block

    return quantize_block(x, bits, layout, group)


def quantize_prefix_cache(
    cache,
    chunk_records: list[dict],
    bits_map: dict[str, int],
    layout: str,
    group: int,
    sink_tokens: int = 4,
) -> None:
    """Quantize-dequantize every chunk block in place; never across block boundaries."""
    if not chunk_records:
        return
    for layer in cache.layers:
        keys = layer.keys
        values = layer.values
        if keys is None:
            continue
        for rec in chunk_records:
            chunk_id = rec["chunk_id"]
            bits = int(bits_map.get(chunk_id, 16))
            if bits >= 16:
                continue
            s, e = rec["block_span"]
            s = max(s, sink_tokens)
            if e <= s:
                continue
            if layout == "kivi":
                keys[:, :, s:e, :] = _qdq_side(keys[:, :, s:e, :], bits, "token", group)
                values[:, :, s:e, :] = _qdq_side(values[:, :, s:e, :], bits, "channel", group)
            else:
                keys[:, :, s:e, :] = _qdq_side(keys[:, :, s:e, :], bits, layout, group)
                values[:, :, s:e, :] = _qdq_side(values[:, :, s:e, :], bits, layout, group)


def prefix_bit_stats(
    chunk_records: list[dict],
    bits_map: dict[str, int],
    head_dim: int,
    group: int,
    layout: str,
) -> dict:
    if not chunk_records:
        return {"effective_bits": 16.0, "nominal_bits": 16.0}
    header_tokens = min(rec["block_span"][0] for rec in chunk_records)
    blocks = []
    for rec in chunk_records:
        s, e = rec["block_span"]
        blocks.append((e - s, int(bits_map.get(rec["chunk_id"], 16))))
    stats = bits_accounting(blocks, header_tokens, head_dim, group, layout)
    stats["header_tokens"] = header_tokens
    return stats
