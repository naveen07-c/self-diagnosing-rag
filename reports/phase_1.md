# Phase 1 — data and baseline RAG

## What was built

- `src/data/hotpot.py`: HotpotQA distractor validation load, fixed sample of 400
  questions (seed 1234), saved `data/splits.json` with 100 dev / 300 test ids, and
  a pooled corpus of 3953 unique paragraphs from those 400 questions
  (`data/questions.jsonl`, `data/corpus.jsonl`). Gold titles and gold paragraph
  ids are stored for evaluation only.
- `src/data/chunking.py`: `paragraph` mode (3953 chunks) and `window` mode
  (3 sentences, stride 2, 7349 chunks), each with sentence offsets.
- `src/rag/retrieve.py`: bge-small bi-encoder over the whole pool (top 30),
  then bge-reranker-base reranking; results cached per retrieval config and
  question in `results/cache/retrieval/`.
- `src/rag/prompt.py`: ChatML prefix (system instruction plus numbered docs) and
  suffix (question, one-sentence instruction, assistant header). Chunk text and
  whole-block token spans are recorded. Prefix/suffix tokenization was verified to
  match joined tokenization.
- `src/rag/generate.py`: two-stage prefill (prefix cache, then suffix) with a
  manual greedy decode loop. The KV hook is already present but `fp` mode is a
  no-op; quantization is exercised in Phase 3.
- `src/judge/em.py`: SQuAD normalization, EM, token F1, retrieval metrics.
- `scripts/build_data.py`, `scripts/build_retrieval.py`, `scripts/run_baseline.py`.
- Generation cache: `results/cache/gen/<config_hash>/<question_id>.json`.

## Commands run

```
env -u PYTHONPATH .venv/bin/python scripts/build_data.py
env -u PYTHONPATH .venv/bin/python scripts/build_retrieval.py \
    --config configs/baseline_k10.yaml --split dev
env -u PYTHONPATH .venv/bin/python scripts/run_baseline.py --config configs/baseline_k5.yaml
env -u PYTHONPATH .venv/bin/python scripts/run_baseline.py --config configs/baseline_k10.yaml
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_phase1.py -q
```

## Measured results (100 dev questions, paragraph chunks, fp KV)

| config | k | recall@k | precision@k | MRR | all-gold | EM | F1 | prompt tokens | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_k5`  | 5  | 0.925 | 0.370 | 0.967 | 0.86 | 0.000 | 0.178 | 685 | 0.77 s/q |
| `baseline_k10` | 10 | 0.965 | 0.193 | 0.968 | 0.93 | 0.000 | 0.185 | 1371 | 1.00 s/q |

Retrieval is well above the gate (recall@10 = 0.965 >= 0.5). Peak GPU memory was
1.44 GB for k=5 and 1.77 GB for k=10, both below the 3.4 GB limit. Re-running a
fully cached config took 2.2 s, so the cache path is working.

`pytest tests/test_phase1.py`: 7 passed. The baseline was regenerated after the
Phase 3 prompt-tokenization fix; the numbers above are from that rerun.

## Ten random dev examples (k=10), read by the agent

... (see previous full report; examples were read from the initial run; baseline
was later regenerated after the tokenization fix, answers may differ slightly) ...

## Interpretation and open problems

- Retrieval is strong, but the generator frequently answers in a full sentence
  (as instructed), so **dev EM is 0.0 by construction** and F1 is low. This is a
  known tension in the plan: HotpotQA gold answers are short spans while the
  prompt asks for one complete sentence.
- Many F1 losses are formatting/verbosity, not wrong content. Phase 2 NLI and the
  counterfactual metric are therefore important; I will also keep a
  `gold_in_answer` diagnostic in Phase 4 (clearly labeled, not a replacement for
  EM/F1).
- Some failures are genuine generation/retrieval failures (examples 2, 7, 10).
  The Phase 5 attribution procedure should separate these from formatting.
- Mean `effective_bits` is 16.0 for fp, as expected after fixing the bit
  accounting to charge fp blocks at 16 bits with no group overhead.

## Gate status

PASS. Fixed splits are disjoint and stable; chunk spans decode to chunk text and
are ordered; EM/F1 and retrieval toy tests pass; the baseline runs for k=5 and
k=10 complete; recall@10 is 0.965, well above 0.5. Raw rows are in
`results/baseline_baseline_k5.jsonl` and `results/baseline_baseline_k10.jsonl`;
summaries are in `results/summary_baseline_k5.json` and
`results/summary_baseline_k10.json`.
