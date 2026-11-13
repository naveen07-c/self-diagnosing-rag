"""Phase 0 gate: CUDA, 4-bit generator, bitsandbytes forward, KV geometry."""
from __future__ import annotations

import json

import pytest
import torch

from src.models import GENERATOR_ID, bytes_per_token_fp16, kv_geometry, load_generator
from src.util import RESULTS, gpu_memory_report


@pytest.fixture(scope="module")
def generator():
    model, tokenizer = load_generator(GENERATOR_ID, four_bit=True)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def test_cuda_and_vram():
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory / 1024**3 >= 3.5


def test_bitsandbytes_linear():
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16, quant_type="nf4").cuda()
    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    y = layer(x)
    assert y.shape == (2, 64)
    assert torch.isfinite(y).all()


def test_generator_memory_and_generation(generator):
    model, tokenizer = generator
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    assert out.shape[1] > inputs["input_ids"].shape[1]
    stats = gpu_memory_report("test_phase0")
    assert stats["max_memory_allocated_gb"] < 3.0


def test_env_json_geometry():
    env = json.loads((RESULTS / "env.json").read_text())
    geom = env["models"][GENERATOR_ID]
    assert geom["num_hidden_layers"] == 28
    assert geom["num_key_value_heads"] == 2
    assert geom["head_dim"] == 128
    assert geom["bytes_per_token_fp16"] == bytes_per_token_fp16(geom)
