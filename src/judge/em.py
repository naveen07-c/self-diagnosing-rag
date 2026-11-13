"""SQuAD-style answer normalization, exact match and token F1; retrieval metrics."""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(pred: str, gold: str) -> dict:
    return {"em": exact_match(pred, gold), "f1": token_f1(pred, gold)}


def retrieval_metrics(retrieved: list[dict], q: dict, k: int) -> dict:
    top = retrieved[: int(k)]
    gold = set(q.get("gold_titles") or [])
    titles = [c.get("title") for c in top]
    found = gold & set(titles)
    hits = sum(1 for t in titles if t in gold)
    mrr = 0.0
    for rank, t in enumerate(titles, start=1):
        if t in gold:
            mrr = 1.0 / rank
            break
    return {
        "precision_at_k": hits / k if k else 0.0,
        "recall_at_k": len(found) / len(gold) if gold else 0.0,
        "mrr": mrr,
        "all_gold_retrieved": float(len(found) == len(gold) and bool(gold)),
        "hits_at_k": hits,
    }


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows) for k in keys}


def answer_metrics_multi(pred: str, golds: list[str]) -> dict:
    """Best EM/F1 over answer aliases (used by RGB, which stores alias lists)."""
    best = {"em": 0.0, "f1": 0.0}
    for gold in golds:
        m = answer_metrics(pred, gold)
        if (m["em"], m["f1"]) > (best["em"], best["f1"]):
            best = m
    return best
