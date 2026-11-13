"""Failure attribution for Phase 5.

Every failed dev query is placed into exactly one category, applying the
decision procedure from plan.md in this order:

1. QUANT       - fails under a quantized config, passes under fp KV with the
   same retrieved chunks.
2. CHUNKING    - a gold supporting sentence is split across chunks and not all
   pieces are retrieved (window chunking); checked before RETRIEVAL.
3. RETRIEVAL   - not all gold supporting paragraphs are in the retrieved set.
4. CONTEXT_NOISE - fails even with only the gold supporting paragraphs under fp
   KV.
5. GENERATION  - cannot answer even with perfect context.
For failures under the fp config, step 1 is skipped.
"""
from __future__ import annotations

CATEGORY_ORDER = ["QUANT", "CHUNKING", "RETRIEVAL", "CONTEXT_NOISE", "GENERATION"]

# A query is "failed" when EM is 0 and F1 < 0.5, or the NLI flag is unfaithful,
# or (counterfactual) the answer did not follow the context.
F1_FAIL = 0.5


def is_failure(record: dict, dataset: str) -> bool:
    """Return True if the record counts as a failed query."""
    em = float(record.get("em", 0.0) or 0.0)
    f1 = float(record.get("f1", 0.0) or 0.0)
    nli = record.get("nli") or {}
    unfaithful = bool(nli.get("unfaithful", 0))
    cf = record.get("cf") or {}
    followed = cf.get("followed_context", 1.0)
    try:
        followed = float(followed)
    except (TypeError, ValueError):
        followed = 1.0
    if em == 0.0 and f1 < F1_FAIL:
        return True
    if unfaithful:
        return True
    if dataset == "counterfactual" and followed == 0.0:
        return True
    return False


def gold_all_retrieved(record: dict) -> bool:
    """True if every gold supporting title appears in the retrieved chunks."""
    gold = record.get("gold_titles") or []
    if not gold:
        return True
    titles = {c.get("title") for c in record.get("retrieved", []) or []}
    return set(gold).issubset(titles)


def chunking_broken(record: dict) -> bool:
    """True if a gold supporting sentence is split across chunks and not all
    pieces were retrieved. With paragraph chunking paragraphs are never split,
    so this is False; the hook exists for window chunking and is checked before
    RETRIEVAL."""
    return False


def attribute(
    record: dict,
    dataset: str,
    kv_mode: str,
    *,
    fp_passed: bool = False,
    chunking_broken: bool = False,
    retrieval_broken: bool = False,
    gold_context_passed: bool = False,
) -> str:
    """Assign exactly one failure category to a failed record.

    `fp_passed`, `chunking_broken`, `retrieval_broken` and `gold_context_passed`
    are the outcomes of decision steps 1-4. For `fp` configs step 1 is skipped.
    """
    if not is_failure(record, dataset):
        raise ValueError("attribute() must be called on a failed record")

    # Step 1: QUANT (skipped for fp configs).
    if kv_mode != "fp" and fp_passed:
        return "QUANT"

    # Step 2: CHUNKING is checked before RETRIEVAL.
    if chunking_broken:
        return "CHUNKING"
    if retrieval_broken:
        return "RETRIEVAL"

    # Step 3: CONTEXT_NOISE.
    if gold_context_passed:
        return "CONTEXT_NOISE"

    # Step 4.
    return "GENERATION"


def failure_definition_note() -> str:
    return (
        "A query is failed when (EM == 0 and F1 < 0.5), or the NLI judge flags "
        "it unfaithful, or (counterfactual) followed_context == 0."
    )
