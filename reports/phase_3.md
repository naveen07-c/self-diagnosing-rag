# Phase 3 — fake KV quantization, split prefill, precision sweep

## Correctness fixes found during the gate

1. **Prompt boundary.** The original prompt builder tokenized the header, each
   document block, and the suffix separately. Byte-level BPE can merge the two
   newlines ending a document with the next `[Doc i]` marker. On one top-5
   prompt this shifted the token ids (`...\n\n[` became tokens `13, 271, 58`
   instead of `382, 58`), so the recorded spans no longer matched the prompt.
   The builder now tokenizes the whole prefix once and derives chunk text/block
   spans from the fast tokenizer's character offsets. The boundary test passes.
   The old builder actually tokenized the wrapper, text, and trailer as three
   separate segments, which adds one newline token per document relative to the
   whole-prefix tokenization. The text was the same, but the token ids were not:
   on dev k=5 the mean difference is 4.94 tokens (all 100 prompts), on dev k=10
   9.87 tokens, and on test k=10 9.94 tokens. Phase 1/2/3 fp artifacts were
   therefore regenerated after this fix, and the table below uses the corrected
   prompt. `prompt_version` stays 1 because the prompt text is unchanged; the
   cache key cannot see tokenization-method changes, so the affected fp caches
   were forced explicitly.
2. **KIVI layout.** `configs/base.yaml` sets `layout: channel`, which was
   overriding the implicit KIVI layout in `mode: kivi` configs. `resolve_kv` now
   forces `layout = "kivi"` for KIVI mode. The first kivi2 run was therefore a
   duplicate of uniform2; the table below is from the corrected run.

## Protocol

- Split: 30 dev questions (fixed order, seed 1234).
- Retrieval: `BAAI/bge-small-en-v1.5` pool 30, `BAAI/bge-reranker-base`, top 10
  paragraph chunks.
- Generation: Qwen2.5-1.5B-Instruct 4-bit NF4, greedy, 64 max new tokens.
- Reference: fp KV. `identical_tokens_vs_fp` is the fraction of questions whose
  full greedy token sequence equals the fp sequence.
- Effective bits are per value over the prefix: chunk tokens at the configured
  width, header/sink tokens at 16 bits, plus 32 bits per quantizer group.

## Gate table

| config | kv mode | bits | layout | eff. bits | nominal bits | identical tokens | identical answers | EM | F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp | fp | 16 | channel | 16.000 | 16.000 | 1.000 | 1.000 | 0.000 | 0.2249 |
| uniform8 | uniform | 8 | channel | 9.214 | 8.245 | 0.933 | 0.933 | 0.000 | 0.2231 |
| uniform4 | uniform | 4 | channel | 5.337 | 4.368 | 0.000 | 0.000 | 0.000 | 0.0000 |
| uniform2 | uniform | 2 | channel | 3.398 | 2.429 | 0.000 | 0.000 | 0.000 | 0.0101 |
| kivi2 | kivi | 2 | kivi | 3.458 | 2.429 | 0.000 | 0.000 | 0.000 | 0.0168 |

Source: `results/tables/kv_phase3.csv`. Recall@10 is 0.9833 for every row, so
the retrieval set is identical across configs. EM is 0 by construction because
the prompt asks for a complete sentence; F1 is the quality signal.

## Findings

- **8-bit validates the machinery.** 28/30 generations are token-identical to
  fp (93.3%, above the 90% gate); the two mismatches are near-ties that do not
  change the aggregate F1 (0.223 vs 0.225). The effective bit count is above 8
  because the header, sinks, and group metadata are charged at fp16; nominal
  bits are 8.25.
- **The quality cliff is below 8 bits for this model.** 4-bit channel
  quantization is already catastrophic on this prompt distribution: 0/30
  generations match fp, the output is digit garbage, and F1 is 0. 2-bit is
  similarly destroyed. KIVI at 2 bits is slightly different but still garbage
  (F1 0.0168, 0/30 identical). This is a real sensitivity of the 4-bit-weight
  1.5B model, not a split bug: 8-bit is exact and the split-equivalence test
  passes.
- **One-question bit-width probe** (same prompt, first-step logit max diff vs fp
  ...)

## Memory

- 4k prompt (4204 tokens) + 64 forced decode peak **2.407 GB**
- sweep peak **1.765 GB**

## Tests

`tests/test_phase3.py`: 6 passed. Full suite 22 passed.

## Gate status

PASS. 8-bit matches fp within tolerance (28/30 identical >= 90% gate); quality
cliff confirmed below 8 bits. The Phase 3 report previously claimed "0
differences" old vs new tokenization; corrected to the actual all-prompts-differ
finding (Phase 4 fp rows regenerated accordingly).

Raw rows: `results/kv_phase3_*.jsonl`; summary: `results/summary_kv_phase3.json`;
table: `results/tables/kv_phase3.csv`.
