"""Model loading helpers. Only one GPU model is loaded at a time (see plan rule 7)."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch

GENERATOR_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDER_ID = "BAAI/bge-small-en-v1.5"
RERANKER_ID = "BAAI/bge-reranker-base"
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def kv_geometry(config: Any) -> dict:
    """Read KV geometry from a model config (do not trust the plan's numbers)."""
    hidden = getattr(config, "hidden_size", None)
    n_heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden is not None and n_heads:
        head_dim = hidden // n_heads
    return {
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "num_attention_heads": n_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "model_type": getattr(config, "model_type", None),
    }


def bytes_per_token_fp16(geom: dict) -> float:
    return (
        2
        * int(geom["num_hidden_layers"])
        * int(geom["num_key_value_heads"])
        * int(geom["head_dim"])
        * 2
    )


def load_generator(
    model_id: str = GENERATOR_ID,
    four_bit: bool = True,
    device: str = "cuda",
    attn_implementation: str | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": {"": 0} if device == "cuda" else device}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def free_model(model: Any) -> None:
    """Drop a GPU model and release its memory before loading the next one."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_nli(model_id: str = NLI_ID, device: str = "cuda"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model, tokenizer


def load_sentence_encoder(model_id: str = EMBEDDER_ID, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def load_reranker(model_id: str = RERANKER_ID, device: str = "cpu"):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, device=device)
