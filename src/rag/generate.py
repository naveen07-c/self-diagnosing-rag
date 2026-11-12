"""Two-stage prefill generation with optional fake KV quantization of the prefix."""
from __future__ import annotations

import time

import torch
from transformers.cache_utils import DynamicCache

from src.kv.cache import (
    build_bits_map,
    prefix_bit_stats,
    quantize_prefix_cache,
    resolve_kv,
)


def _stop_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(x) for x in eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok in ("assistant", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return ids


def _prefill(model, prefix_ids: torch.Tensor, cache) -> torch.Tensor:
    pos = torch.arange(prefix_ids.shape[1], device=prefix_ids.device).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=prefix_ids,
            past_key_values=cache,
            use_cache=True,
            position_ids=pos,
        )
    return out.logits[:, -1, :]


def generate_with_kv(
    model,
    tokenizer,
    prompt: dict,
    kv_cfg: dict | None = None,
    gold_titles: list[str] | None = None,
    max_new_tokens: int = 64,
    return_details: bool = False,
    ignore_eos: bool = False,
) -> dict:
    kv = resolve_kv(kv_cfg)
    device = next(model.parameters()).device
    prefix_ids = torch.tensor([prompt["prefix_ids"]], device=device)
    suffix_ids = torch.tensor([prompt["suffix_ids"]], device=device)

    t0 = time.perf_counter()
    cache = DynamicCache()
    prefix_logits = _prefill(model, prefix_ids, cache)

    bits_map = build_bits_map(prompt["chunks"], kv, gold_titles=gold_titles)
    if kv["mode"] != "fp":
        quantize_prefix_cache(
            cache,
            prompt["chunks"],
            bits_map,
            kv["layout"],
            kv["group"],
            sink_tokens=kv["sink_tokens"],
        )

    pos = torch.arange(
        prefix_ids.shape[1],
        prefix_ids.shape[1] + suffix_ids.shape[1],
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=suffix_ids,
            past_key_values=cache,
            use_cache=True,
            position_ids=pos,
        )
    logits = out.logits[:, -1, :]

    stop = _stop_ids(model, tokenizer)
    generated: list[int] = []
    cur_len = prefix_ids.shape[1] + suffix_ids.shape[1]
    first_logits = logits.detach().float().clone()
    for _ in range(int(max_new_tokens)):
        next_id = int(torch.argmax(logits, dim=-1).item())
        if not ignore_eos and next_id in stop:
            break
        generated.append(next_id)
        cur_len += 1
        next_ids = torch.tensor([[next_id]], device=device)
        pos = torch.tensor([[cur_len - 1]], device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_ids,
                past_key_values=cache,
                use_cache=True,
                position_ids=pos,
            )
        logits = out.logits[:, -1, :]

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    bit_stats = prefix_bit_stats(
        prompt["chunks"], bits_map, head_dim, kv["group"], kv["layout"]
    )
    result = {
        "answer": answer,
        "generated_ids": generated,
        "generated_tokens": len(generated),
        "prompt_tokens": prompt["prompt_tokens"],
        "prefix_tokens": prompt["prefix_tokens"],
        "suffix_tokens": prompt["suffix_tokens"],
        "latency_s": time.perf_counter() - t0,
        "kv": kv | {"bits_map": bits_map},
        "bit_stats": bit_stats,
        "first_step_logits": first_logits if return_details else None,
        "prefix_last_logits": prefix_logits.detach().float().clone() if return_details else None,
    }
    return result
