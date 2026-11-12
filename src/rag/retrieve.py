"""Bi-encoder retrieval plus cross-encoder reranking, cached per (config, question)."""
from __future__ import annotations

import numpy as np

from src.data.chunking import load_chunks
from src.models import EMBEDDER_ID, RERANKER_ID, load_reranker, load_sentence_encoder
from src.util import DATA, cache_path, config_hash, ensure_dir, read_json, set_seed, write_json

EMBEDDING_META_VERSION = 1


def _embedding_paths(model_id: str, chunk_mode: str):
    tag = model_id.replace("/", "__")
    return (
        DATA / f"emb_{tag}_{chunk_mode}.npy",
        DATA / f"emb_{tag}_{chunk_mode}.json",
    )


def build_embeddings(
    chunk_mode: str = "paragraph",
    model_id: str = EMBEDDER_ID,
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Embed the chunk pool once; cache the matrix and ids on disk."""
    npy_path, meta_path = _embedding_paths(model_id, chunk_mode)
    chunks = load_chunks(chunk_mode)
    ids = [c["chunk_id"] for c in chunks]
    meta = read_json(meta_path, default={})
    if npy_path.exists() and meta.get("ids") == ids and meta.get("model") == model_id:
        return np.load(npy_path), ids

    set_seed(1234)
    model = load_sentence_encoder(model_id, device=device)
    texts = [c["text"] for c in chunks]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(npy_path, emb)
    write_json(
        meta_path,
        {
            "model": model_id,
            "chunk_mode": chunk_mode,
            "ids": ids,
            "shape": list(emb.shape),
            "version": EMBEDDING_META_VERSION,
        },
    )
    del model
    return emb, ids


def _query_embed(model, queries: list[str], batch_size: int = 32) -> np.ndarray:
    try:
        return model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt_name="query",
            show_progress_bar=False,
        ).astype(np.float32)
    except Exception:
        return model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def retrieval_key(cfg: dict) -> str:
    return config_hash(
        {
            "embedder": cfg.get("embedder", EMBEDDER_ID),
            "reranker": cfg.get("reranker", RERANKER_ID),
            "chunk_mode": cfg.get("chunk_mode", "paragraph"),
            "pool": cfg.get("pool", 30),
            "reranker_max_length": cfg.get("reranker_max_length", 512),
        }
    )


def build_retrieval(
    questions: list[dict],
    cfg: dict,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, dict]:
    """Return qid -> retrieval record with the top-pool chunks and both scores."""
    chunk_mode = cfg.get("chunk_mode", "paragraph")
    pool = int(cfg.get("pool", 30))
    embedder_id = cfg.get("embedder", EMBEDDER_ID)
    reranker_id = cfg.get("reranker", RERANKER_ID)
    key = retrieval_key(cfg)

    out: dict[str, dict] = {}
    todo: list[dict] = []
    for q in questions:
        path = cache_path("retrieval", key, q["id"])
        if not force and path.exists():
            out[q["id"]] = read_json(path)
        else:
            todo.append(q)

    if not todo:
        return out

    emb, ids = build_embeddings(chunk_mode, embedder_id, device=device)
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    chunks = load_chunks(chunk_mode)
    id_to_chunk = {c["chunk_id"]: c for c in chunks}

    set_seed(1234)
    encoder = load_sentence_encoder(embedder_id, device=device)
    queries = [q["question"] for q in todo]
    qemb = _query_embed(encoder, queries)
    del encoder

    pool_rows: list[list[tuple[str, float]]] = []
    for qv in qemb:
        scores = emb @ qv
        order = np.argsort(-scores, kind="stable")
        top = [(ids[i], float(scores[i])) for i in order[:pool]]
        pool_rows.append(top)

    set_seed(1234)
    reranker = load_reranker(reranker_id, device=device)
    pairs: list[tuple[str, str]] = []
    pair_owner: list[int] = []
    for qi, (q, top) in enumerate(zip(todo, pool_rows)):
        for cid, _ in top:
            pairs.append((q["question"], id_to_chunk[cid]["text"]))
            pair_owner.append(qi)
    scores = reranker.predict(
        pairs, batch_size=32, show_progress_bar=len(pairs) > 200, convert_to_numpy=True
    )
    del reranker

    per_q: list[list[dict]] = [[] for _ in todo]
    for (cid, ret_score), rk_score, qi in zip(
        [p for top in pool_rows for p in top], scores, pair_owner
    ):
        chunk = id_to_chunk[cid]
        per_q[qi].append(
            {
                "chunk_id": cid,
                "title": chunk["title"],
                "retriever_score": ret_score,
                "reranker_score": float(rk_score),
            }
        )

    for q, rows in zip(todo, per_q):
        rows.sort(key=lambda r: (-r["reranker_score"], -r["retriever_score"], r["chunk_id"]))
        for rank, row in enumerate(rows):
            row["rank"] = rank
        record = {"question_id": q["id"], "pool": rows, "retrieval_key": key}
        write_json(cache_path("retrieval", key, q["id"]), record)
        out[q["id"]] = record
    return out


def top_k(record: dict, k: int) -> list[dict]:
    return record["pool"][: int(k)]
