"""HotpotQA distractor loading, fixed splits and pooled deduplicated corpus."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from src.util import DATA, SEED, ensure_dir, read_json, read_jsonl, write_json, write_jsonl

DATASET_ID = "hotpotqa/hotpot_qa"

SPLITS_PATH = DATA / "splits.json"
QUESTIONS_PATH = DATA / "questions.jsonl"
CORPUS_PATH = DATA / "corpus.jsonl"


def _paragraph_text(sentences: Iterable[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and s.strip())


def build_dataset(force: bool = False) -> dict:
    """Download HotpotQA, take 400 questions (100 dev / 300 test), pool their paragraphs."""
    if not force and SPLITS_PATH.exists() and QUESTIONS_PATH.exists() and CORPUS_PATH.exists():
        return read_json(SPLITS_PATH)

    from datasets import load_dataset

    ensure_dir(DATA)
    ds = load_dataset(DATASET_ID, "distractor", split="validation")
    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = sorted(indices[:400])
    dev_positions = chosen[:100]
    test_positions = chosen[100:]

    # The HotpotQA id is the stable key used everywhere else.
    dev_ids = [ds[i]["id"] for i in dev_positions]
    test_ids = [ds[i]["id"] for i in test_positions]
    splits = {
        "seed": SEED,
        "source": DATASET_ID,
        "config": "distractor",
        "source_split": "validation",
        "dev": dev_ids,
        "test": test_ids,
    }

    # Pool paragraphs from the 400 selected questions, deduplicated by (title, text).
    corpus: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    title_to_pids: dict[str, list[str]] = {}
    question_rows: list[dict] = []

    for qi, idx in enumerate(chosen):
        row = ds[idx]
        qid = row["id"]
        split = "dev" if qid in set(dev_ids) else "test"
        ctx = row["context"]
        local_title_to_pid: dict[str, list[str]] = {}
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            text = _paragraph_text(sentences)
            key = (title.strip(), text)
            if key not in seen:
                pid = f"p{len(corpus):05d}"
                seen[key] = pid
                corpus.append(
                    {
                        "para_id": pid,
                        "title": title.strip(),
                        "text": text,
                        "sentences": [s.strip() for s in sentences],
                    }
                )
            pid = seen[key]
            title_to_pids.setdefault(title.strip(), [])
            if pid not in title_to_pids[title.strip()]:
                title_to_pids[title.strip()].append(pid)
            local_title_to_pid.setdefault(title.strip(), []).append(pid)

        supporting = [
            {"title": t, "sent_id": int(s)}
            for t, s in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
        ]
        gold_titles = sorted({s["title"] for s in supporting})
        gold_para_ids: list[str] = []
        for t in gold_titles:
            for pid in local_title_to_pid.get(t, []):
                if pid not in gold_para_ids:
                    gold_para_ids.append(pid)

        question_rows.append(
            {
                "id": qid,
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "split": split,
                "gold_titles": gold_titles,
                "supporting": supporting,
                "gold_para_ids": gold_para_ids,
                "context_titles": [t.strip() for t in ctx["title"]],
            }
        )

    write_json(SPLITS_PATH, splits)
    write_jsonl(QUESTIONS_PATH, question_rows)
    write_jsonl(CORPUS_PATH, corpus)
    print(
        f"dataset: {len(question_rows)} questions "
        f"({len(dev_ids)} dev / {len(test_ids)} test), {len(corpus)} unique paragraphs"
    )
    return splits


def load_questions(split: str | None = None) -> list[dict]:
    rows = read_jsonl(QUESTIONS_PATH)
    if split is None:
        return rows
    return [r for r in rows if r["split"] == split]


def load_splits() -> dict:
    return read_json(SPLITS_PATH)


def load_corpus() -> list[dict]:
    return read_jsonl(CORPUS_PATH)


def corpus_by_id(corpus: list[dict] | None = None) -> dict[str, dict]:
    return {row["para_id"]: row for row in (corpus or load_corpus())}
