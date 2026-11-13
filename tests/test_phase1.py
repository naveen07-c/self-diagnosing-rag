"""Phase 1 gate: splits, chunk spans, answer metrics, toy retrieval, determinism."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.chunking import chunks_by_id, load_chunks
from src.data.hotpot import load_questions, load_splits
from src.judge.em import exact_match, retrieval_metrics, token_f1
from src.models import EMBEDDER_ID, load_sentence_encoder
from src.rag.prompt import build_prompt
from src.rag.retrieve import build_retrieval
from src.util import load_experiment, set_seed

TOKENIZER = None


def tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer

        TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return TOKENIZER


def test_splits_fixed_and_disjoint():
    splits = load_splits()
    assert len(splits["dev"]) == 100
    assert len(splits["test"]) == 300
    assert not set(splits["dev"]) & set(splits["test"])
    assert load_splits() == splits
    questions = load_questions()
    assert {q["id"] for q in questions} == set(splits["dev"]) | set(splits["test"])


def test_chunk_spans_decode_to_chunk_text():
    questions = load_questions("dev")
    cmap = chunks_by_id("paragraph")
    chunks = [cmap[cid] for cid in [q["gold_para_ids"][0] for q in questions[:3]]]
    prompt = build_prompt(questions[0]["question"], chunks, tokenizer())
    for rec in prompt["chunks"]:
        decoded = tokenizer().decode(
            prompt["prefix_ids"][rec["text_span"][0] : rec["text_span"][1]],
            skip_special_tokens=False,
        )
        assert " ".join(decoded.split()) == " ".join(rec["text"].split())


def test_chunk_spans_ordered_and_nonoverlapping():
    q = load_questions("dev")[0]
    chunks = [chunks_by_id("paragraph")[pid] for pid in q["gold_para_ids"][:2]]
    prompt = build_prompt(q["question"], chunks, tokenizer())
    assert len(prompt["chunks"]) == 2
    first, second = prompt["chunks"]
    assert first["block_span"][0] > 0
    assert second["block_span"][0] == first["block_span"][1]
    assert first["text_span"][1] <= second["text_span"][0]
    assert first["text_span"][0] < first["text_span"][1] < second["text_span"][1]


def test_em_f1_known_values():
    assert exact_match("The cat.", "cat") == 1.0
    assert exact_match("a dog", "dog") == 1.0
    assert exact_match("Paris, France", "paris france") == 1.0
    assert exact_match("1861", "1862") == 0.0
    assert exact_match("New York City", "New York") == 0.0
    assert token_f1("New York City", "New York") == pytest.approx(0.8)
    assert token_f1("the cat sat", "cat sat down") == pytest.approx(0.8)
    assert token_f1("", "") == 1.0
    assert token_f1("cat", "") == 0.0


def test_retrieval_toy_corpus_rank1():
    set_seed(1234)
    docs = [
        "Paris is the capital and largest city of France.",
        "The mitochondria is the powerhouse of the cell.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "William Shakespeare wrote Hamlet and Macbeth.",
        "The RTX 3050 is a graphics card made by Nvidia.",
    ]
    model = load_sentence_encoder(EMBEDDER_ID, device="cpu")
    emb = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    q = model.encode(
        ["What is the capital of France?"], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    order = np.argsort(-(emb @ q), kind="stable")
    assert order[0] == 0


def test_retrieval_metrics_toy():
    q = {"gold_titles": ["A", "B"]}
    retrieved = [
        {"title": "A", "chunk_id": "a1"},
        {"title": "X", "chunk_id": "x1"},
        {"title": "B", "chunk_id": "b1"},
    ]
    m = retrieval_metrics(retrieved, q, 3)
    assert m["recall_at_k"] == 1.0
    assert m["precision_at_k"] == pytest.approx(2 / 3)
    assert m["mrr"] == 1.0
    assert m["all_gold_retrieved"] == 1.0


def test_retrieval_deterministic():
    cfg = load_experiment("configs/baseline_k10.yaml")
    questions = load_questions("dev")[:2]
    a = build_retrieval(questions, cfg["retrieval"], device="cuda" if _cuda() else "cpu")
    b = build_retrieval(questions, cfg["retrieval"], device="cuda" if _cuda() else "cpu")
    for q in questions:
        assert [r["chunk_id"] for r in a[q["id"]]["pool"]] == [
            r["chunk_id"] for r in b[q["id"]]["pool"]
        ]


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()
