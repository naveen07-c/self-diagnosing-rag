"""Fixed RGB noise-robustness subset with 5 provided documents per question."""
from __future__ import annotations

from src.util import DATA, read_jsonl

RGB_PATH = DATA / "rgb_noise.jsonl"


def load_rgb() -> list[dict]:
    return read_jsonl(RGB_PATH)
