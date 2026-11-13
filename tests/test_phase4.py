"""Phase 4 gate: audit completeness, cache speed, bootstrap, table hygiene."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import pytest

from src.eval.bootstrap import paired_bootstrap
from scripts.run_baseline import run
from src.util import RESULTS, TABLES, load_experiment, read_jsonl


def test_bootstrap_interval_and_zero_width():
    a = [1.0, 0.0, 1.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    res = paired_bootstrap(a, b, n_boot=200, seed=7)
    assert res["lo"] <= res["mean_diff"] <= res["hi"]
    same = paired_bootstrap(a, a, n_boot=200, seed=7)
    assert same["lo"] == same["hi"] == 0.0


def test_audit_files_have_one_row_per_question():
    files = sorted(RESULTS.glob("audit_*_scored.jsonl"))
    assert files, "run scripts/run_audit.py and run_judge_batch.py first"
    for path in files:
        rows = read_jsonl(path)
        ids = [r["question_id"] for r in rows]
        assert len(ids) == len(set(ids)), path.name


def test_cached_config_rerun_is_fast():
    cfg = load_experiment("configs/baseline_k10.yaml")
    cfg["limit"] = 5
    t0 = time.perf_counter()
    rows = run(cfg)
    elapsed = time.perf_counter() - t0
    assert len(rows) == 5
    assert elapsed < 10.0, elapsed


def test_audit_table_has_no_nan_or_gaps():
    path = TABLES / "audit.csv"
    assert path.exists(), "run scripts/build_audit_table.py"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        for key, value in row.items():
            assert value.strip().lower() not in {"nan", "none"}, (key, value)
            if key in {
                "k", "n", "bits", "group", "em", "f1", "nli_faithfulness",
                "hallucination_rate", "effective_bits", "prompt_tokens",
            } and value.strip():
                assert math.isfinite(float(value)), (key, value)
    keys = {(r["dataset"], r["split"], r["k"], r["config"]) for r in rows}
    assert len(keys) == len(rows), "duplicate audit rows"


def test_judge_memory_and_plot():
    import json

    mem = RESULTS / "memory_judge_batch.json"
    assert mem.exists(), "run scripts/run_judge_batch.py"
    stats = json.load(open(mem))
    assert stats["max_memory_allocated_gb"] < 3.4
    assert (RESULTS / "figures" / "audit_effective_bits.png").exists()
