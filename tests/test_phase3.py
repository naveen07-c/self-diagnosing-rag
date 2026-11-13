"""Phase 3 gate: quantizer, bit accounting, split equivalence, output change, memory."""
from __future__ import annotations

import json

import torch
from transformers.cache_utils import DynamicCache

from src.data.chunking import chunks_by_id
from src.data.hotpot import load_questions
from src.kv.bits import chunk_bits_accounting
from src.kv.cache import resolve_kv
from src.kv.quant import qdq
from src.models import load_generator
from src.rag.generate import generate_with_kv
from src.rag.prompt import build_prompt
from src.rag.retrieve import build_retrieval, top_k
from src.util import deep_merge, load_experiment, read_jsonl, set_seed


def _prompt(k=5):
    set_seed(1234)
    q = load_questions("dev")[0]
    cfg = load_experiment("configs/baseline_k10.yaml")
    ret = build_retrieval([q], cfg["retrieval"], device="cuda")[q["id"]]
    cmap = chunks_by_id("paragraph")
    chunks = []
    for row in top_k(ret, k):
        c = dict(cmap[row["chunk_id"]])
        c["reranker_score"] = row["reranker_score"]
        chunks.append(c)
    return q, chunks


def _stop_ids(model, tokenizer):
    ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(eos)
    ids.add(tokenizer.eos_token_id)
    ids.add(tokenizer.convert_tokens_to_ids("assistant"))
    return ids


def _single_pass(model, tokenizer, prompt, max_new_tokens=20):
    full = torch.tensor([prompt["full_ids"]], device="cuda")
    cache = DynamicCache()
    with torch.no_grad():
        out = model(
            input_ids=full,
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.arange(full.shape[1], device="cuda").unsqueeze(0),
        )
    prefix_logits = out.logits[:, prompt["prefix_tokens"] - 1, :].float()
    logits = out.logits[:, -1, :]
    gen = []
    cur = full.shape[1]
    stop = _stop_ids(model, tokenizer)
    for _ in range(max_new_tokens):
        nid = int(torch.argmax(logits, dim=-1).item())
        if nid in stop:
            break
        gen.append(nid)
        cur += 1
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[nid]], device="cuda"),
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.tensor([[cur - 1]], device="cuda"),
            )
        logits = out.logits[:, -1, :]
    return gen, prefix_logits


def test_qdq_roundtrip_idempotence_constant():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 32, 128, dtype=torch.float16)
    for bits in (8, 2):
        y = qdq(x, bits, -1, 32)
        group_range = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)).max().item()
        err = (x - y).abs().max().item()
        assert err < 0.01 * group_range if bits == 8 else err < group_range / 3
        assert torch.equal(qdq(y, bits, -1, 32), y)
    const = torch.full((2, 2, 8, 32), 3.14, dtype=torch.float16)
    assert torch.equal(qdq(const, 2, -1, 16), const)


def test_kivi_mode_implies_kivi_layout():
    assert resolve_kv({"mode": "kivi", "layout": "channel"})["layout"] == "kivi"
    assert resolve_kv({"mode": "uniform"})["layout"] == "channel"


def test_bit_accounting_toy_and_mixed():
    toy = chunk_bits_accounting([(64, 2)], 128, 32, "channel")
    assert toy["effective_bits"] == 3.0
    assert toy["nominal_bits"] == 2.0
    mixed = chunk_bits_accounting([(64, 2), (32, 8)], 128, 32, "channel")
    assert mixed["nominal_bits"] == 4.0
    assert mixed["effective_bits"] == 5.0


def test_split_equivalence_and_quantization_changes_output():
    q, chunks = _prompt(k=5)
    model, tokenizer = load_generator("Qwen/Qwen2.5-1.5B-Instruct")
    try:
        prompt = build_prompt(q["question"], chunks, tokenizer)
        joined = tokenizer(
            prompt["prefix_text"] + prompt["suffix_text"], add_special_tokens=False
        )["input_ids"]
        assert joined == prompt["full_ids"]
        fp = generate_with_kv(
            model, tokenizer, prompt, kv_cfg={"mode": "fp"}, max_new_tokens=20, return_details=True
        )
        single, single_prefix_logits = _single_pass(model, tokenizer, prompt, 20)
        assert fp["generated_ids"] == single
        diff = (fp["prefix_last_logits"] - single_prefix_logits).abs().max().item()
        assert diff < 1e-2, diff
        q2 = generate_with_kv(
            model,
            tokenizer,
            prompt,
            kv_cfg={"mode": "uniform", "bits": 2, "group": 32},
            max_new_tokens=20,
            return_details=True,
        )
        assert q2["generated_ids"] != fp["generated_ids"]
        assert (q2["first_step_logits"] - fp["first_step_logits"]).abs().max().item() > 1e-3
    finally:
        del model
        torch.cuda.empty_cache()


def test_8bit_matches_fp_on_30_dev():
    base = load_experiment("configs/kv_phase3.yaml")
    base.update({"split": "dev", "limit": 30, "k": 10})
    from scripts.run_baseline import run

    fp = run(deep_merge(base, {"name": "fp", "kv": {"mode": "fp"}}))
    eight = run(
        deep_merge(
            base,
            {"name": "uniform8", "kv": {"mode": "uniform", "bits": 8, "group": 32}},
        )
    )
    a = {r["question_id"]: r["generated_ids"] for r in fp}
    same = sum(1 for r in eight if a[r["question_id"]] == r["generated_ids"])
    assert same / len(eight) >= 0.9, (same, len(eight))


def test_4k_prefill_memory():
    model, tokenizer = load_generator("Qwen/Qwen2.5-1.5B-Instruct")
    try:
        sentence = (
            "The Amazon rainforest is a large tropical rainforest in South America "
            "that covers much of the Amazon basin and hosts enormous biodiversity. "
        )
        chunks = [
            {"chunk_id": f"c{i}", "title": f"Doc{i}", "text": sentence * 55}
            for i in range(3)
        ]
        prompt = build_prompt("What is the Amazon rainforest?", chunks, tokenizer)
        assert prompt["prompt_tokens"] >= 3800, prompt["prompt_tokens"]
        torch.cuda.reset_peak_memory_stats()
        out = generate_with_kv(
            model, tokenizer, prompt, kv_cfg={"mode": "fp"}, max_new_tokens=64, ignore_eos=True
        )
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"4k prefill + 64 decode peak {peak:.3f} GB (prompt {prompt['prompt_tokens']} tokens)")
        assert out["generated_tokens"] == 64
        assert peak < 3.4, peak
    finally:
        del model
        torch.cuda.empty_cache()
