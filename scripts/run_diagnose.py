#!/usr/bin/env python
"""Failure attribution diagnostics (Phase 5)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.counterfactual import score_counterfactual_answer
from src.data.hotpot import corpus_by_id, load_questions
from src.diagnose.attribute import (
    CATEGORY_ORDER,
    attribute,
    gold_all_retrieved,
    is_failure,
)
from src.judge.em import answer_metrics
from src.judge.nli import NLIScorer, build_premises, split_answer_sentences
from src.models import NLI_ID, free_model, load_generator
from src.rag.generate import generate_with_kv
from src.rag.prompt import build_prompt
from src.util import (
    CACHE,
    DATA,
    RESULTS,
    cache_path,
    config_hash,
    deep_merge,
    ensure_dir,
    gpu_memory_report,
    load_experiment,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
)

CONFIG_PATH = "configs/audit.yaml"
FAILURES_PATH = RESULTS / "failures_dev.jsonl"
GOLD_CACHE_DIR = CACHE / "diagnose" / "gold"
FIGURE_PATH = RESULTS / "figures" / "failures_by_category.png"

KV_MODES = {
    "fp": "fp",
    "u8": "uniform",
    "u4": "uniform",
    "u2g32": "uniform",
    "u2g64": "uniform",
    "k4": "kivi",
    "k2g32": "kivi",
    "k2g64": "kivi",
}


def audit_record_path(dataset: str, split: str, k: int, config: str) -> Path:
    return RESULTS / f"audit_{dataset}_{split}_k{k}_{config}_scored.jsonl"


def load_audit_records(dataset: str, split: str, k: int, config: str) -> list[dict]:
    path = audit_record_path(dataset, split, k, config)
    return read_jsonl(path)


def fp_kv() -> dict:
    return {
        "mode": "fp",
        "bits": 16,
        "bits_high": 8,
        "bits_low": 2,
        "group": 32,
        "layout": "channel",
        "protected_p": 0,
        "seed": 1234,
        "sink_tokens": 4,
    }


def gold_chunks_for_question(q: dict, corpus_map: dict) -> list[dict]:
    chunks = []
    for pid in q.get("gold_para_ids", []) or []:
        para = corpus_map.get(pid)
        if para is None:
            continue
        chunks.append(
            {
                "chunk_id": pid,
                "title": para["title"],
                "text": para["text"],
                "para_id": pid,
                "sent_start": 0,
                "sent_end": len(para.get("sentences", [])) - 1,
                "reranker_score": 1.0,
                "retriever_score": 1.0,
            }
        )
    return chunks


class GoldContextProbe:
    def __init__(self, model_id: str, four_bit: bool, max_new_tokens: int, corpus_map: dict):
        self.model_id = model_id
        self.four_bit = four_bit
        self.max_new_tokens = max_new_tokens
        self.corpus_map = corpus_map
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, dict] = {}

    def _load(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = load_generator(
                self.model_id, four_bit=self.four_bit, device="cuda"
            )

    def _free(self) -> None:
        if self._model is not None:
            free_model(self._model)
            self._model = None
            self._tokenizer = None

    def record_for(self, qid: str, question: str, gold_answer: str,
                   gold_titles: list[str], gold_para_ids: list[str],
                   dataset: str, cf_lookup: dict) -> dict:
        if qid in self._cache:
            return self._cache[qid]
        self._load()
        assert self._model is not None and self._tokenizer is not None
        q = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "gold_titles": gold_titles,
            "gold_para_ids": gold_para_ids,
        }
        chunks = gold_chunks_for_question(q, self.corpus_map)
        prompt = build_prompt(question, chunks, self._tokenizer)
        gen = generate_with_kv(
            self._model,
            self._tokenizer,
            prompt,
            kv_cfg=fp_kv(),
            gold_titles=gold_titles,
            max_new_tokens=self.max_new_tokens,
        )
        answer = gen["answer"]
        metrics = answer_metrics(answer, gold_answer)
        rec = {
            "question_id": qid,
            "question": question,
            "dataset": dataset,
            "answer": answer,
            "gold_answer": gold_answer,
            "gold_titles": gold_titles,
            "em": metrics["em"],
            "f1": metrics["f1"],
            "nli": {"score": 0.0, "max_contradiction": 0.0, "unfaithful": 0.0,
                     "threshold": 0.5},
            "cf": {"label": "other", "followed_context": 1.0,
                   "reverted_to_memory": 0.0, "both_present": 0.0},
            "prompt_tokens": gen["prompt_tokens"],
            "bit_stats": gen["bit_stats"],
        }
        self._cache[qid] = rec
        return rec

    def score_nli_and_cf(self, records: list[dict], cf_lookup: dict) -> None:
        if not records:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.models import NLI_ID

        tokenizer = AutoTokenizer.from_pretrained(NLI_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            NLI_ID, torch_dtype=torch.float16
        ).to("cuda")
        model.eval()
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        ent_idx = labels.get("entailment", 1)
        con_idx = labels.get("contradiction", 0)

        pairs: list[tuple[str, str]] = []
        pair_index: list[int] = []
        for i, rec in enumerate(records):
            q_rec = {
                "id": rec["question_id"],
                "gold_titles": rec.get("gold_titles", []),
            }
            chunks = gold_chunks_for_question(q_rec, self.corpus_map)
            premises: list[str] = []
            for chunk in chunks:
                premises.extend(
                    build_premises(
                        {
                            "para_id": chunk.get("para_id"),
                            "text": chunk.get("text", ""),
                            "sent_start": int(chunk.get("sent_start", 0)),
                            "sent_end": int(chunk.get("sent_end", 0)),
                        },
                        self.corpus_map,
                        tokenizer,
                    )
                )
            if not premises:
                premises = [rec["question"]]
            for s in split_answer_sentences(rec["answer"]):
                for p in premises:
                    pairs.append((p, s))
                    pair_index.append(i)

        if pairs:
            all_probs = []
            for s in range(0, len(pairs), 32):
                enc = tokenizer(
                    [p for p, _ in pairs[s : s + 32]],
                    [h for _, h in pairs[s : s + 32]],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                with torch.no_grad():
                    logits = model(**enc).logits.float()
                all_probs.append(logits.softmax(dim=-1).cpu())
            probs = torch.cat(all_probs, dim=0)
            ent = probs[:, ent_idx]
            con = probs[:, con_idx]
            per_sentence: dict[int, list[float]] = defaultdict(list)
            per_contradiction: dict[int, list[float]] = defaultdict(list)
            for pi, idx in enumerate(pair_index):
                per_sentence[idx].append(float(ent[pi].item()))
                per_contradiction[idx].append(float(con[pi].item()))
            for i, rec in enumerate(records):
                sent_scores = per_sentence.get(i, [0.0])
                rec["nli"] = {
                    "score": float(min(sent_scores)),
                    "max_entailment": float(max(sent_scores)),
                    "max_contradiction": float(
                        max(per_contradiction.get(i, [0.0]))
                    ),
                    "unfaithful": float(min(sent_scores) < 0.5),
                    "threshold": 0.5,
                }
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for rec in records:
            dataset = rec["dataset"]
            cf = cf_lookup.get(rec["question_id"], {})
            if dataset == "counterfactual" and cf:
                sub = cf.get("substitute_answer", "")
                orig = cf.get("original_answer", "")
                rec["cf"] = score_counterfactual_answer(rec["answer"], orig, sub)
            else:
                rec["cf"] = {
                    "label": "other",
                    "followed_context": 1.0,
                    "reverted_to_memory": 0.0,
                    "both_present": 0.0,
                }
        self._free()


def collect_failures(datasets, splits, ks, configs, limit: int | None):
    fp_lookup: dict[tuple[str, str, int, str], dict] = {}
    cf_lookup: dict[str, dict] = {}
    question_lookup: dict[str, dict] = {}
    failures: list[dict] = []

    if "counterfactual" in datasets:
        cf_path = DATA / "counterfactual.jsonl"
        for row in read_jsonl(cf_path):
            cf_lookup[row["question_id"]] = {
                "original_answer": row["original_answer"],
                "substitute_answer": row["substitute_answer"],
                "answer_type": row["answer_type"],
            }

    normal_dev = load_questions("dev")
    for q in normal_dev:
        question_lookup[q["id"]] = q
    for q in load_questions():
        question_lookup.setdefault(q["id"], q)

    for dataset in datasets:
        for split in splits:
            for k in ks:
                for config in configs:
                    rows = load_audit_records(dataset, split, k, config)
                    if limit is not None:
                        rows = rows[:limit]
                    kv_mode = KV_MODES.get(config, "fp")
                    for rec in rows:
                        qid = rec["question_id"]
                        question_lookup.setdefault(
                            qid,
                            {
                                "id": qid,
                                "question": rec.get("question", ""),
                                "gold_answer": rec.get("gold_answer", ""),
                                "gold_titles": rec.get("gold_titles", []),
                                "gold_para_ids": rec.get("gold_para_ids", []),
                                "split": rec.get("split", split),
                            },
                        )
                        if config == "fp":
                            fp_lookup[(dataset, split, k, qid)] = rec
                        if not is_failure(rec, dataset):
                            continue
                        rec = dict(rec)
                        rec["dataset"] = dataset
                        rec["k"] = k
                        rec["split"] = split
                        failures.append(rec)
    return failures, fp_lookup, cf_lookup, question_lookup


def categorize(failures, fp_lookup, cf_lookup, question_lookup, probe: GoldContextProbe):
    step3_candidates: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)
    results: list[dict] = []

    for rec in failures:
        dataset = rec.get("dataset", "normal")
        qid = rec["question_id"]
        split = rec.get("split", "dev")
        k = int(rec.get("k", 10))
        config = rec.get("config_name", rec.get("config", "fp"))
        short = config
        for prefix in (f"audit_{dataset}_{split}_k{k}_",):
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        kv_mode = KV_MODES.get(short, "fp")

        q = question_lookup.get(qid, {})
        fp_passed = False
        chunking_broken = False
        retrieval_broken = not gold_all_retrieved(rec)

        if kv_mode != "fp":
            fp_rec = fp_lookup.get((dataset, split, k, qid))
            if fp_rec is not None:
                fp_passed = not is_failure(fp_rec, dataset)
        else:
            fp_passed = True

        if kv_mode != "fp" and fp_passed:
            category = "QUANT"
        elif chunking_broken:
            category = "CHUNKING"
        elif retrieval_broken:
            category = "RETRIEVAL"
        else:
            step3_candidates[qid].append((rec, dataset, kv_mode))
            category = None
        results.append(
            {
                "_rec": rec,
                "_category_pending": category is None,
                "_category": category,
                "_qid": qid,
                "_dataset": dataset,
                "_split": split,
                "_k": k,
                "_config": short,
                "_kv_mode": kv_mode,
            }
        )

    pending_qids = [qid for qid, recs in step3_candidates.items() if recs]
    probe_records: list[dict] = []
    probe_by_qid: dict[str, dict] = {}
    for qid in pending_qids:
        q = question_lookup.get(qid, {})
        dataset = step3_candidates[qid][0][1]
        pr = probe.record_for(
            qid=qid,
            question=q.get("question", ""),
            gold_answer=q.get("gold_answer", q.get("answer", "")),
            gold_titles=q.get("gold_titles", []),
            gold_para_ids=q.get("gold_para_ids", []),
            dataset=dataset,
            cf_lookup=cf_lookup,
        )
        probe_records.append(pr)
        probe_by_qid[qid] = pr
    probe.score_nli_and_cf(probe_records, cf_lookup)

    for entry in results:
        if entry["_category_pending"]:
            qid = entry["_qid"]
            pr = probe_by_qid[qid]
            gold_context_passed = not is_failure(pr, entry["_dataset"])
            entry["_category"] = attribute(
                entry["_rec"],
                entry["_dataset"],
                entry["_kv_mode"],
                fp_passed=False,
                chunking_broken=False,
                retrieval_broken=False,
                gold_context_passed=gold_context_passed,
            )
            entry["_gold_context"] = pr
        else:
            entry["_gold_context"] = None
    return results


def build_failure_record(entry) -> dict:
    rec = entry["_rec"]
    pr = entry["_gold_context"]
    out = {
        "category": entry["_category"],
        "config": entry["_config"],
        "kv_mode": entry["_kv_mode"],
        "dataset": entry["_dataset"],
        "split": entry["_split"],
        "k": entry["_k"],
        "question_id": rec["question_id"],
        "question": rec.get("question", ""),
        "answer": rec.get("answer", ""),
        "gold_answer": rec.get("gold_answer", ""),
        "em": float(rec.get("em", 0.0) or 0.0),
        "f1": float(rec.get("f1", 0.0) or 0.0),
        "nli_score": float((rec.get("nli") or {}).get("score", 0.0) or 0.0),
        "nli_unfaithful": float((rec.get("nli") or {}).get("unfaithful", 0.0) or 0.0),
        "followed_context": float((rec.get("cf") or {}).get("followed_context", 1.0) or 1.0),
        "cf_label": (rec.get("cf") or {}).get("label", ""),
        "gold_titles": rec.get("gold_titles", []),
        "gold_para_ids": rec.get("gold_para_ids", []),
        "retrieved_titles": [c.get("title") for c in rec.get("retrieved", []) or []],
        "all_gold_retrieved": gold_all_retrieved(rec),
    }
    if pr is not None:
        out["gold_context_answer"] = pr.get("answer", "")
        out["gold_context_em"] = float(pr.get("em", 0.0) or 0.0)
        out["gold_context_f1"] = float(pr.get("f1", 0.0) or 0.0)
        out["gold_context_nli_score"] = float((pr.get("nli") or {}).get("score", 0.0) or 0.0)
        out["gold_context_unfaithful"] = float((pr.get("nli") or {}).get("unfaithful", 0.0) or 0.0)
    return out


def draw_chart(failure_records: list[dict]) -> None:
    ensure_dir(RESULTS / "figures")
    by_config_cat: dict[str, Counter] = defaultdict(Counter)
    configs: list[str] = []
    for fr in failure_records:
        by_config_cat[fr["config"]][fr["category"]] += 1
        if fr["config"] not in configs:
            configs.append(fr["config"])
    configs = sorted(configs)
    if not configs:
        return
    bottom = {c: 0 for c in configs}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cat in CATEGORY_ORDER:
        vals = [by_config_cat[c].get(cat, 0) for c in configs]
        ax.bar(configs, vals, bottom=[bottom[c] for c in configs], label=cat)
        for c in configs:
            bottom[c] += by_config_cat[c].get(cat, 0)
    ax.set_ylabel("failed dev queries")
    ax.set_xlabel("config")
    ax.set_title("Phase 5 failure categories per config (dev)")
    ax.legend(title="category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 failure attribution")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, nargs="*", default=[5, 10])
    ap.add_argument("--dataset", choices=["normal", "counterfactual"],
                    nargs="*", default=["normal", "counterfactual"])
    args = ap.parse_args()

    base = load_experiment(args.config)
    model_id = base.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    four_bit = bool(base.get("four_bit", True))
    max_new_tokens = int(base.get("max_new_tokens", 64))
    set_seed(int(base.get("seed", 1234)))

    corpus_map = corpus_by_id()
    probe = GoldContextProbe(model_id, four_bit, max_new_tokens, corpus_map)

    failures, fp_lookup, cf_lookup, question_lookup = collect_failures(
        args.dataset, [args.split], args.k, list(KV_MODES.keys()), args.limit
    )
    results = categorize(failures, fp_lookup, cf_lookup, question_lookup, probe)

    records = [build_failure_record(e) for e in results]
    ensure_dir(RESULTS)
    write_jsonl(FAILURES_PATH, records)
    draw_chart(records)
    gpu_memory_report("run_diagnose exit",
                      write_path=RESULTS / "memory_run_diagnose.json")

    counts = Counter(r["category"] for r in records)
    print(f"attributed {len(records)} failures")
    for cat in CATEGORY_ORDER:
        print(f"  {cat}: {counts.get(cat, 0)}")
    print(f"wrote {FAILURES_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    main()
