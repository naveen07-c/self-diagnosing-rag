# Phase 0 — environment

## What was built

- Python 3.12.3 venv at `.venv`, all dependencies pinned in `requirements.txt`
  (notably `torch==2.11.0+cu128`, `transformers==4.57.6`; the transformers pin is
  required because all KV-cache code is written against its `DynamicCache` API).
- `src/util.py`: seed handling, config hashing, JSONL/JSON cache helpers, GPU
  memory reporting.
- `src/models.py`: 4-bit NF4 generator loader, KV-geometry reader, model
  free/load helpers for the separate judge process.
- `scripts/env_check.py`: prints GPU, CUDA and library versions, loads
  `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit, generates 20 tokens, records KV geometry
  and peak memory into `results/env.json`.
- `tests/test_phase0.py` and `Makefile` (`setup`, `test`, `phase0`).

## Commands run

```
nvidia-smi
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
.venv/bin/pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python scripts/env_check.py
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_phase0.py -q
```

`PYTHONPATH` is cleared for every command because this machine exports ROS
Jazzy site-packages globally; leaving it set injects ROS pytest plugins into the
venv.

## Measured output

```
GPU: NVIDIA GeForce RTX 3050 A Laptop GPU, 4094 MiB (torch reports 3.648 GiB usable)
driver 580.178.04, CUDA 13.0 driver / torch CUDA 12.8, compute capability (8, 9)
torch 2.11.0+cu128, transformers 4.57.6, accelerate 1.12.0,
bitsandbytes 0.50.2, datasets 5.0.1, sentence-transformers 5.1.2

KV geometry from model config:
  num_hidden_layers = 28
  num_key_value_heads = 2
  head_dim = 128
  fp16 KV bytes/token = 28672 (28 KB)

4-bit load + 20-token generation:
  allocated 1.085 GB, peak 1.182 GB, reserved peak 1.221 GB

pytest: 4 passed in 9.06s
```

Raw values are in `results/env.json`.

## Gate status

PASS. CUDA available, total VRAM 3.648 GiB >= 3.5 GiB, peak generation memory
1.182 GB < 3.0 GB, bitsandbytes 4-bit linear forward works, and `env.json`
contains layers/KV heads/head dim for the generator.

## Open problems

- The advertised 4 GB card exposes only 3.648 GiB to torch. This is still above
the Phase 0 gate but leaves little headroom; all later scripts use batch size 1
and free models between stages.
- `torch_dtype` is deprecated in this transformers release; it is only a warning
and the model is loaded in fp16 as intended.
