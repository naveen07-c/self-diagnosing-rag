#!/usr/bin/env python
"""Phase 0 environment check: GPU, libraries, 4-bit generator load, KV geometry."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models import GENERATOR_ID, bytes_per_token_fp16, kv_geometry, load_generator
from src.util import RESULTS, ensure_dir, gpu_memory_report, set_seed, write_json


def main() -> None:
    set_seed(1234)
    ensure_dir(RESULTS)
    import accelerate
    import bitsandbytes as bnb
    import datasets
    import sentence_transformers
    import transformers

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bnb.__version__,
        "datasets": datasets.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "gpu": {
            "name": props.name if props else None,
            "total_vram_gb": round(props.total_memory / 1024**3, 3) if props else None,
            "capability": list(torch.cuda.get_device_capability(0)) if props else None,
            "cuda_available": torch.cuda.is_available(),
        },
        "models": {},
    }
    print(json.dumps(env, indent=2))

    model, tokenizer = load_generator(GENERATOR_ID, four_bit=True)
    geom = kv_geometry(model.config)
    geom["bytes_per_token_fp16"] = bytes_per_token_fp16(geom)
    env["models"][GENERATOR_ID] = geom
    print("KV geometry:", json.dumps(geom, indent=2))

    messages = [{"role": "user", "content": "Say hello in one short sentence."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("sample generation:", repr(answer))

    stats = gpu_memory_report("phase0_load_generate")
    env["phase0_memory"] = stats
    write_json(RESULTS / "env.json", env)
    print("wrote", RESULTS / "env.json")


if __name__ == "__main__":
    main()
