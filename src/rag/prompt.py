"""Prompt construction with an explicit prefix/suffix split and chunk token spans.

The prefix is tokenized once as a whole and spans are derived from the fast
tokenizer's character offsets. Deriving spans from separately tokenized segments
is not safe: byte-level BPE can merge the two newlines that end a document with
the next document marker, which shifts token indices and makes chunk spans wrong.
"""
from __future__ import annotations

SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents. If the documents do not contain the answer, say that you do not know."
)


def _enc(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _boundary_at(offsets: list[tuple[int, int]], pos: int) -> int:
    """First token whose character start is at or after pos."""
    for i, (start, _) in enumerate(offsets):
        if start >= pos:
            return i
    return len(offsets)


def _overlap_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Token span covering every token that overlaps the character range [start, end)."""
    idx = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
    if idx:
        return [idx[0], idx[-1] + 1]
    b = _boundary_at(offsets, start)
    return [b, b]


def build_prompt(question: str, chunks: list[dict], tokenizer) -> dict:
    """Return prefix/suffix text, their ids, and text/block spans for each chunk."""
    header = f"assistant\n{SYSTEM}assistant\nuser\n"
    prefix_text = header
    pending = []
    for doc_index, chunk in enumerate(chunks, start=1):
        wrapper = f"[Doc {doc_index}]\n"
        text = chunk["text"].strip()
        block_start = len(prefix_text)
        text_start = block_start + len(wrapper)
        prefix_text += wrapper + text + "\n\n"
        text_end = text_start + len(text)
        block_end = len(prefix_text)
        pending.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "doc_index": doc_index,
                "text": text,
                "text_char": [text_start, text_end],
                "block_char": [block_start, block_end],
                "reranker_score": chunk.get("reranker_score"),
                "retriever_score": chunk.get("retriever_score"),
            }
        )

    enc = tokenizer(
        prefix_text, add_special_tokens=False, return_offsets_mapping=True
    )
    prefix_ids = enc["input_ids"]
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    chunk_records = []
    for rec in pending:
        ts, te = rec["text_char"]
        bs, be = rec["block_char"]
        rec = dict(rec)
        rec["text_span"] = _overlap_span(offsets, ts, te)
        rec["block_span"] = [_boundary_at(offsets, bs), _boundary_at(offsets, be)]
        rec.pop("text_char", None)
        rec.pop("block_char", None)
        chunk_records.append(rec)

    suffix = (
        f"Question: {question}\n"
        "Answer in one complete sentence using only the documents above."
        "assistant\n"
    )
    suffix_ids = _enc(tokenizer, suffix)
    return {
        "prefix_text": prefix_text,
        "suffix_text": suffix,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "full_ids": prefix_ids + suffix_ids,
        "chunks": chunk_records,
        "prefix_tokens": len(prefix_ids),
        "suffix_tokens": len(suffix_ids),
        "prompt_tokens": len(prefix_ids) + len(suffix_ids),
    }


def boundary_ok(prompt: dict, tokenizer) -> bool:
    joined = tokenizer(
        prompt["prefix_text"] + prompt["suffix_text"], add_special_tokens=False
    )["input_ids"]
    return joined == prompt["full_ids"]


def text_for_span(prompt: dict, tokenizer, span: list[int]) -> str:
    return tokenizer.decode(prompt["prefix_ids"][span[0]:span[1]], skip_special_tokens=False)
