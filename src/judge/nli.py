"""DeBERTa NLI faithfulness scorer over sentence/window premises."""
from __future__ import annotations

import re

import torch

from src.models import NLI_ID

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_answer_sentences(answer: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(answer.strip()) if p.strip()]
    return parts or ([answer.strip()] if answer.strip() else [])


def build_premises(
    chunk: dict,
    corpus_map: dict,
    tokenizer,
    max_sentences: int = 3,
    max_tokens: int = 450,
) -> list[str]:
    para = corpus_map.get(chunk.get("para_id"))
    if para and para.get("sentences"):
        sent_start = int(chunk.get("sent_start", 0))
        sent_end = int(chunk.get("sent_end", len(para["sentences"]) - 1))
        sentences = para["sentences"][sent_start : sent_end + 1]
    else:
        sentences = [chunk.get("text", "")]
    whole = " ".join(s for s in sentences if s.strip())
    if len(tokenizer(whole, add_special_tokens=False)["input_ids"]) <= max_tokens:
        return [whole] if whole.strip() else [chunk.get("text", "")]
    # Long chunk: score every 3-sentence window as well as the chunk's sentences.
    premises: list[str] = []
    i = 0
    while i < len(sentences):
        group = [s for s in sentences[i : i + max_sentences] if s.strip()]
        if not group:
            i += max_sentences
            continue
        text = " ".join(group)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_tokens:
            for s in group:
                if s.strip():
                    premises.append(s)
        else:
            premises.append(text)
        i += max_sentences
    return premises or [chunk.get("text", "")]


class NLIScorer:
    def __init__(
        self,
        model_id: str = NLI_ID,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entailment_idx = labels.get("entailment", 1)
        self.contradiction_idx = labels.get("contradiction", 0)

    def _probs(self, premises: list[str], hypotheses: list[str]) -> torch.Tensor:
        all_probs = []
        for i in range(0, len(premises), self.batch_size):
            p = premises[i : i + self.batch_size]
            h = hypotheses[i : i + self.batch_size]
            enc = self.tokenizer(
                p,
                h,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            all_probs.append(logits.softmax(dim=-1).cpu())
        return torch.cat(all_probs, dim=0)

    def score_answer(self, answer: str, premises: list[str]) -> dict:
        sentences = split_answer_sentences(answer)
        if not sentences or not premises:
            return {
                "score": 0.0,
                "max_entailment": 0.0,
                "max_contradiction": 0.0,
                "sentence_scores": [],
                "n_pairs": 0,
            }
        all_prem: list[str] = []
        all_hyp: list[str] = []
        for s in sentences:
            all_prem.extend(premises)
            all_hyp.extend([s] * len(premises))
        probs = self._probs(all_prem, all_hyp)
        ent = probs[:, self.entailment_idx].reshape(len(sentences), len(premises))
        con = probs[:, self.contradiction_idx].reshape(len(sentences), len(premises))
        per_sentence = ent.max(dim=1).values
        return {
            "score": float(per_sentence.min().item()),
            "max_entailment": float(ent.max().item()),
            "max_contradiction": float(con.max().item()),
            "sentence_scores": [float(x) for x in per_sentence],
            "n_pairs": int(ent.numel()),
        }

    def free(self) -> None:
        import gc

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
