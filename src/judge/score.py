"""Score generation records against the context they were shown."""
from __future__ import annotations

import re

from src.data.chunking import chunks_by_id
from src.judge.nli import build_premises

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def premises_from_text(text: str, tokenizer, max_sentences: int = 3, max_tokens: int = 450) -> list[str]:
    sentences = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text]
    whole = " ".join(sentences)
    if len(tokenizer(whole, add_special_tokens=False)["input_ids"]) <= max_tokens:
        return [whole]
    out = []
    for i in range(0, len(sentences), max_sentences):
        group = " ".join(sentences[i : i + max_sentences])
        if len(tokenizer(group, add_special_tokens=False)["input_ids"]) > max_tokens:
            out.extend(sentences[i : i + max_sentences])
        else:
            out.append(group)
    return out or [text]


def contexts_for_record(record: dict, chunk_mode: str = "paragraph") -> list[dict]:
    if record.get("context_texts"):
        return record["context_texts"]
    cmap = chunks_by_id(chunk_mode)
    return [cmap[r["chunk_id"]] for r in record.get("retrieved", []) if r["chunk_id"] in cmap]


def score_record(record: dict, scorer, corpus_map: dict, threshold: float) -> dict:
    contexts = contexts_for_record(record, record.get("chunk_mode", "paragraph"))
    premises: list[str] = []
    premise_meta: list[dict] = []
    for ctx in contexts:
        if ctx.get("para_id"):
            ps = build_premises(
                ctx,
                corpus_map,
                scorer.tokenizer,
                max_sentences=3,
                max_tokens=450,
            )
        else:
            ps = premises_from_text(ctx.get("text", ""), scorer.tokenizer)
        for p in ps:
            premises.append(p)
            premise_meta.append({"chunk_id": ctx.get("chunk_id"), "title": ctx.get("title")})
    result = scorer.score_answer(record.get("answer", ""), premises)
    result["unfaithful"] = float(result["score"] < threshold)
    result["threshold"] = threshold
    result["n_premises"] = len(premises)
    out = dict(record)
    out["nli"] = result
    return out
