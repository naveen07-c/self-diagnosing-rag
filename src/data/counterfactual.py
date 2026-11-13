"""Counterfactual answer substitution set (same coarse answer type).

We replace the gold answer in the gold paragraphs with an answer of
same coarse type drawn from another question. Scoring distinguishes
`followed_context`, `reverted_to_memory` and `other`.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

from src.util import DATA, SEED, read_jsonl, write_jsonl

CF_PATH = DATA / "counterfactual.jsonl"
CF_VERSION = 1

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
NUMBER_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
SKIP_ANSWERS = {"yes", "no", "true", "false", "n/a", "none"}
KEEP_TYPES = {"year", "number", "PERSON", "GPE"}


def _norm_phrase(s: str) -> str:
    return " ".join(s.lower().split())


def contains_phrase(text: str, phrase: str) -> bool:
    if not phrase.strip():
        return False
    pat = r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)"
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def replace_phrase(text: str, old: str, new: str) -> tuple[str, int]:
    if not old.strip():
        return text, 0
    pat = r"(?<!\w)" + re.escape(old.strip()) + r"(?!\w)"
    return re.subn(pat, new, text, flags=re.IGNORECASE)


def _snippet(text: str, phrase: str, width: int = 500) -> str:
    m = re.search(r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)", text, flags=re.IGNORECASE)
    if not m:
        return text[: 2 * width]
    start = max(0, m.start() - width // 2)
    end = min(len(text), m.end() + width)
    return text[start:end]

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def coarse_type(answer: str) -> str:
    a = answer.strip()
    low = a.lower()
    if low in SKIP_ANSWERS or not a:
        return "skip"
    if YEAR_RE.fullmatch(a):
        return "year"
    if NUMBER_RE.fullmatch(a):
        return "number"
    doc = _nlp()(a)
    ents = [e.label_ for e in doc.ents]
    if "PERSON" in ents:
        return "PERSON"
    if "GPE" in ents or "LOC" in ents:
        return "GPE"
    return "other"


def _candidate_pool(questions: list[dict]) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        ans = q["answer"].strip()
        if not ans:
            continue
        t = coarse_type(ans)
        if t == "skip":
            continue
        key = _norm_phrase(ans)
        if key not in seen[t]:
            seen[t].add(key)
            pool[t].append(ans)
    return pool


def _pick_substitute(
    original: str,
    type_: str,
    pool: dict[str, list[str]],
    q: dict,
    context_texts: list[str],
    rng: random.Random,
) -> str | None:
    orig_norm = _norm_phrase(original)
    candidates = [
        a
        for a in pool.get(type_, [])
        if _norm_phrase(a) != orig_norm
        and not contains_phrase(original, a)
        and not contains_phrase(a, original)
        and not contains_phrase(q["question"], a)
    ]
    if not candidates:
        return None
    if type_ == "other":
        candidates = [c for c in candidates if len(c.split()) <= 3] or candidates
    rng.shuffle(candidates)
    for cand in candidates:
        if not any(contains_phrase(t, cand) for t in context_texts):
            return cand
    return None


def build_counterfactual(questions: list[dict], corpus_map: dict[str, dict], force: bool = False) -> list[dict]:
    if CF_PATH.exists() and not force:
        return read_jsonl(CF_PATH)
    pool = _candidate_pool(questions)
    rng = random.Random(SEED + 17)
    title_to_pids: dict[str, list[str]] = defaultdict(list)
    for pid, para in corpus_map.items():
        title_to_pids[para["title"]].append(pid)
    records = []
    for q in questions:
        ans = q["answer"].strip()
        type_ = coarse_type(ans)
        if type_ == "skip" or type_ not in KEEP_TYPES:
            continue
        gold_texts = [corpus_map[pid]["text"] for pid in q.get("gold_para_ids", []) if pid in corpus_map]
        if not any(contains_phrase(t, ans) for t in gold_texts):
            continue
        own_pids: list[str] = []
        for title in q.get("context_titles", []):
            own_pids.extend(title_to_pids.get(title, []))
        own_texts = [corpus_map[pid]["text"] for pid in own_pids if pid in corpus_map]
        substitute = _pick_substitute(ans, type_, pool, q, own_texts, rng)
        if substitute is None:
            continue
        edited = {}
        n_repl = 0
        replaced_pid = None
        for pid in q.get("gold_para_ids", []):
            if pid not in corpus_map:
                continue
            new_text, n = replace_phrase(corpus_map[pid]["text"], ans, substitute)
            edited[pid] = new_text
            n_repl += n
            if n > 0 and replaced_pid is None:
                replaced_pid = pid
        if n_repl == 0 or replaced_pid is None:
            continue
        records.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "split": q.get("split"),
                "original_answer": ans,
                "substitute_answer": substitute,
                "answer_type": type_,
                "gold_titles": q.get("gold_titles", []),
                "gold_para_ids": q.get("gold_para_ids", []),
                "n_replacements": n_repl,
                "example_before": _snippet(corpus_map[replaced_pid]["text"], ans),
                "example_after": _snippet(edited[replaced_pid], substitute),
                "cf_version": CF_VERSION,
            }
        )
    write_jsonl(CF_PATH, records)
    return records


def score_counterfactual_answer(answer: str, original: str, substitute: str) -> dict:
    follows = contains_phrase(answer, substitute)
    reverts = contains_phrase(answer, original)
    if follows:
        label = "followed_context"
    elif reverts:
        label = "reverted_to_memory"
    else:
        label = "other"
    return {
        "label": label,
        "followed_context": float(follows),
        "reverted_to_memory": float(reverts),
        "both_present": float(follows and reverts),
    }
