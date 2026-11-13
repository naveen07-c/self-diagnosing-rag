"""Phase 2 gate: NLI ranking, long premises, counterfactual builder/scorer, memory."""
from __future__ import annotations

import torch
import pytest

from src.data.counterfactual import (
    contains_phrase,
    replace_phrase,
    score_counterfactual_answer,
)
from src.data.hotpot import corpus_by_id, load_corpus
from src.judge.nli import NLIScorer, build_premises
from src.util import load_yaml, read_jsonl


@pytest.fixture(scope="module")
def scorer():
    torch.cuda.reset_peak_memory_stats()
    cfg = load_yaml("configs/judge.yaml")
    s = NLIScorer(cfg["model"], device="cuda", max_length=cfg["max_length"], batch_size=cfg["batch_size"])
    yield s
    s.free()


def test_nli_sanity_ranking(scorer):
    entailed = [
        ("Paris is the capital of France.", "Paris is the capital of France."),
        ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris."),
        ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 100 degrees Celsius."),
    ]
    contradicted = [
        ("Paris is the capital of France.", "Berlin is the capital of France."),
        ("The Eiffel Tower is in Paris.", "The Eiffel Tower is in London."),
        ("Water boils at 100 degrees Celsius.", "Water freezes at 100 degrees Celsius."),
    ]
    e_scores = [scorer.score_answer(h, [p])["score"] for p, h in entailed]
    c_scores = [scorer.score_answer(h, [p])["score"] for p, h in contradicted]
    assert min(e_scores) > max(c_scores), (e_scores, c_scores)


def test_long_premise_windowing(scorer):
    sentences = [
        "The Amazon rainforest is a large tropical rainforest in South America.",
        "It covers much of the Amazon basin.",
        "The river has many tributaries.",
    ] * 100
    para = {"sentences": sentences, "text": " ".join(sentences)}
    chunk = {"para_id": "p0", "text": para["text"], "sent_start": 0, "sent_end": len(sentences) - 1}
    premises = build_premises(chunk, {"p0": para}, scorer.tokenizer, max_sentences=3, max_tokens=450)
    assert len(premises) > 1
    out = scorer.score_answer("The Amazon rainforest is in South America.", premises)
    assert 0.0 <= out["score"] <= 1.0
    assert out["n_pairs"] >= len(premises)


def test_counterfactual_builder_20_samples():
    rows = read_jsonl("data/counterfactual.jsonl")[:20]
    cmap = corpus_by_id(load_corpus())
    assert len(rows) == 20
    for r in rows:
        edited_any = False
        for pid in r["gold_para_ids"]:
            text = cmap[pid]["text"]
            new, n = replace_phrase(text, r["original_answer"], r["substitute_answer"])
            if n:
                edited_any = True
                assert not contains_phrase(new, r["original_answer"])
                assert contains_phrase(new, r["substitute_answer"])
        assert edited_any


def test_counterfactual_scorer_three_labels():
    cases = [
        ("The answer is Leonard Cohen.", "Alfred Balk", "Leonard Cohen", "followed_context"),
        ("The answer is Alfred Balk.", "Alfred Balk", "Leonard Cohen", "reverted_to_memory"),
        ("The answer is not stated.", "Alfred Balk", "Leonard Cohen", "other"),
        ("Leonard Cohen, not Alfred Balk.", "Alfred Balk", "Leonard Cohen", "followed_context"),
        ("ALFRED BALK.", "Alfred Balk", "Leonard Cohen", "reverted_to_memory"),
        ("Nothing relevant here.", "Alfred Balk", "Leonard Cohen", "other"),
    ]
    for answer, original, substitute, expected in cases:
        out = score_counterfactual_answer(answer, original, substitute)
        assert out["label"] == expected, (answer, out)
    both = score_counterfactual_answer("Leonard Cohen and Alfred Balk.", "Alfred Balk", "Leonard Cohen")
    assert both["both_present"] == 1.0


def test_judge_gpu_memory(scorer):
    torch.cuda.reset_peak_memory_stats()
    scorer.score_answer("Paris is the capital of France.", ["Paris is the capital of France."])
    peak = torch.cuda.max_memory_allocated() / 1024**3
    assert peak < 3.4, peak
