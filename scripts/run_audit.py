#!/usr/bin/env python
"""Generate the Phase 4 audit matrix on HotpotQA, counterfactual, or RGB."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_baseline import run as run_normal
from scripts.run_counterfactual import run as run_counterfactual
from scripts.run_rgb import run as run_rgb
from src.util import (
    RESULTS,
    deep_merge,
    gpu_memory_report,
    load_experiment,
    mean_or_zero,
    write_json,
    write_jsonl,
)


def summary_for(name: str, dataset: str, split: str, k: int, cfg: dict, rows: list[dict]) -> dict:
    out = {
        "name": name,
        "dataset": dataset,
        "split": split,
        "k": k,
        "config": cfg.get("config_name") or name,
        "kv": cfg.get("kv", {}),
        "n": len(rows),
        "em": mean_or_zero([r["em"] for r in rows]),
        "f1": mean_or_zero([r["f1"] for r in rows]),
        "mean_effective_bits": mean_or_zero(
            [r["bit_stats"]["effective_bits"] for r in rows]
        ),
        "mean_prompt_tokens": mean_or_zero(
            [r["prompt_tokens"] for r in rows]
        ),
        "mean_latency_s": mean_or_zero([r["latency_s"] for r in rows]),
    }
    if dataset == "counterfactual":
        out["followed_context_rate"] = mean_or_zero(
            [r["cf"]["followed_context"] for r in rows]
        )
    if dataset == "normal" and rows:
        out["recall_at_k"] = mean_or_zero(
            [r["retrieval"]["recall_at_k"] for r in rows]
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["normal", "counterfactual", "rgb"], required=True)
    ap.add_argument("--split", choices=["dev", "test", "rgb"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    base = load_experiment("configs/audit.yaml")
    items = base["configs"]
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        items = [it for it in items if it["name"] in names]
    if not items:
        raise SystemExit("no configs selected")
    ks = base.get("ks", [5, 10])
    if args.dataset == "rgb":
        ks = base.get("rgb_ks", [5])
    all_rows = {}
    for item in items:
        for k in ks:
            cfg = deep_merge(base, item)
            cfg["k"] = k
            cfg["name"] = f"audit_{args.dataset}_{args.split}_k{k}_{item['name']}"
            cfg["dataset"] = args.dataset
            if args.dataset == "normal":
                runner = run_normal
            elif args.dataset == "counterfactual":
                runner = run_counterfactual
            else:
                runner = run_rgb
                cfg.pop("split", None)
            rows = runner(cfg, force=args.force)
            write_jsonl(RESULTS / f"{cfg['name']}_scored.jsonl", rows)
            all_rows[cfg["name"]] = rows
            print(f"{cfg['name']}: {len(rows)} rows", flush=True)
    write_json(
        RESULTS / "summary_audit_table.json",
        {"n_rows": sum(len(v) for v in all_rows.values()), "files": list(all_rows)},
    )
    gpu_memory_report("run_audit exit", write_path=RESULTS / "memory_audit_all.json")


if __name__ == "__main__":
    main()
