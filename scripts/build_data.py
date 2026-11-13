#!/usr/bin/env python
"""Phase 1 data build: fixed HotpotQA splits, pooled corpus, paragraph/window chunks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.chunking import build_chunks
from src.data.hotpot import build_dataset, load_splits
from src.util import set_seed


def main() -> None:
    set_seed(1234)
    splits = build_dataset()
    print("splits:", len(splits["dev"]), "dev,", len(splits["test"]), "test")
    build_chunks("paragraph")
    build_chunks("window")
    loaded = load_splits()
    assert loaded["dev"] == splits["dev"]
    assert loaded["test"] == splits["test"]


if __name__ == "__main__":
    main()
