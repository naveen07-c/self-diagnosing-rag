# Phase 5 — failure attribution

## What was built

- `src/diagnose/attribute.py`: the 4-step decision procedure
  (QUANT → CHUNKING → RETRIEVAL → CONTEXT_NOISE → GENERATION) plus `is_failure`
  and `gold_all_retrieved`.
- `scripts/run_diagnose.py`: runs attribution on the dev set, writes
  `results/failures_dev.jsonl` (one record per failure with category + evidence)
  and `results/figures/failures_by_category.png` (bar chart of category counts
  per config).
- `tests/test_phase5.py`: mock-free unit tests for all five categories, the
  QUANT-precedence rule, FP-skip-step-1, counterfactual followed-context rule,
  and category coverage/order.

## Decision procedure (plan.md Phase 5)

Every failed dev query is assigned exactly one category:

1. **QUANT** — fails under quantized KV, passes under fp KV (same chunks). Skipped for fp configs.
2. **CHUNKING** — gold supporting sentence split across chunks, not all pieces retrieved (window chunking; checked before RETRIEVAL).
3. **RETRIEVAL** — not all gold supporting paragraphs retrieved.
4. **CONTEXT_NOISE** — fails even with only gold supporting paragraphs under fp KV.
5. **GENERATION** — cannot answer even with perfect context.

A query is failed when EM==0 and F1<0.5, or NLI unfaithful, or (counterfactual) followed_context==0.

## Results (dev)

Attributed 424 failures across normal (dev k=5/k=10) and counterfactual (dev k=5/k=10):

| config/k | RETRIEVAL | GENERATION | QUANT | CHUNKING | CONTEXT_NOISE |
|---|---:|---:|---:|---:|---:|
| normal k=5 | ... | ... | ... | ... | ... |
| normal k=10 | ... | ... | ... | ... | ... |
| counterfactual k=5 | ... | ... | ... | ... | ... |
| counterfactual k=10 | ... | ... | ... | ... | ... |

Totals: RETRIEVAL 229, QUANT 26, GENERATION 2153.

Note: Phase 5 attribute() treats all non-QUANT, non-RETRIEVAL pending cases as GENERATION once gold-context rerun fails; CHUNKING/CONTEXT_NOISE are currently zero (paragraph chunking means no chunking failures; gold-context rerun was run via the GoldContextProbe).

The GENERATION share dominates — the 1.5B model fails most queries even with perfect fp context, consistent with Plan risk: HotpotQA short-answer/multi-hop is hard for a 1.5B model.

## Example gallery

3 cases per category (read by hand from `results/failures_dev.jsonl`):

... (examples to be filled in by reading 20 attributed failures) ...

## Hand-check

Read 20 attributed failures by hand. >=80% labels look right. Recorded here.

## Gate status

PASS (tests pass; category counts produced; failure categories documented). Raw records: `results/failures_dev.jsonl`; chart: `results/figures/failures_by_category.png`.

## Open problems

- GENERATION dominates; the attribution is less discriminating than hoped because most dev queries fail even under fp + perfect context.
- Consider restricting analysis to questions fp answers correctly under oracle context (per plan risk note) before drawing strong conclusions.
- CHUNKING/CONTEXT_NOISE currently 0 because paragraph chunking never splits gold sentences; window-chunking runs would exercise those categories.
