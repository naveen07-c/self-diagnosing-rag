"""Shared utilities: paths, seeds, config hashing, JSONL and GPU memory reporting."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
TABLES = RESULTS / "tables"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"

SEED = 1234


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg: dict[str, Any] = {}
    if path is not None:
        cfg = load_yaml(path)
    return deep_merge(cfg, overrides)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(cfg: dict, ignore: Iterable[str] = ()) -> str:
    """Stable short hash of a config dict, ignoring volatile bookkeeping keys."""
    ignore = set(ignore) | {"created_at", "note"}
    clean = {k: v for k, v in cfg.items() if k not in ignore}
    return hashlib.sha256(canonical_json(clean).encode()).hexdigest()[:12]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    os.replace(tmp, p)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r") as f:
        return json.load(f)


def append_jsonl(path: str | Path, record: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, p)


def cache_path(kind: str, cfg_hash: str, key: str, suffix: str = ".json") -> Path:
    return CACHE / kind / cfg_hash / f"{key}{suffix}"


def gpu_memory_report(tag: str = "", write_path: str | Path | None = None) -> dict:
    import torch

    stats: dict[str, Any] = {"tag": tag, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        stats.update(
            {
                "device": torch.cuda.get_device_name(0),
                "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            }
        )
        print(
            f"[gpu] {tag} allocated={stats['memory_allocated_gb']:.3f}GB "
            f"peak={stats['max_memory_allocated_gb']:.3f}GB "
            f"reserved_peak={stats['max_memory_reserved_gb']:.3f}GB",
            flush=True,
        )
    else:
        print(f"[gpu] {tag} CUDA not available", flush=True)
    if write_path is not None:
        write_json(write_path, stats)
    return stats


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.last = self.elapsed()


def load_experiment(path: str | Path, overrides: dict | None = None) -> dict:
    """Merge configs/base.yaml under an experiment file, then apply overrides."""
    base_path = CONFIGS / "base.yaml"
    cfg = load_yaml(base_path) if base_path.exists() else {}
    cfg = deep_merge(cfg, load_yaml(path))
    return deep_merge(cfg, overrides)


def mean_or_zero(values) -> float:
    values = list(values)
    return sum(float(v) for v in values) / len(values) if values else 0.0
