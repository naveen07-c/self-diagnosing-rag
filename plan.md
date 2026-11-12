# Plan: self-diagnosing RAG with retrieval-guided KV-cache precision

This file is the build spec for a coding agent. Follow the phases in order. Do not start a phase until the gate of the previous phase passes. Every number that ends up in the README must come from a log file in `results/`.

## 0. Rules for the agent

1. Work phase by phase. At the end of each phase: run the phase tests, fill in `reports/phase_N.md` (what was built, commands run, test output, measured numbers, open problems), and make a git commit named `phase N: <summary>`.
2. If a gate fails, fix it or write the problem to `BLOCKERS.md` and stop. Do not weaken a test to make it pass. Do not skip a phase.
3. Never invent results, citations, or hyperparameters. If a number is not in a log, it does not go in a report.
4. If an expected effect does not show up (for example, quantization does not hurt faithfulness at this scale), report that plainly. A negative result with clean evidence is acceptable. A fabricated positive result is not.
5. Fix every random seed (`SEED = 1234`), record library versions in `results/env.json`, and cache every model output to disk keyed by `(config_hash, question_id)`. Never regenerate an output that is already cached.
6. GPU memory is the hard limit. Every script that touches the GPU must print `torch.cuda.max_memory_allocated()` at exit. If it exceeds 3.4 GB, treat that as a failure.
7. Use only one big model on the GPU at a time. Generation and NLI judging are separate scripts that never run together.
8. The citation arXiv:2608.30996 from the original proposal is unverified. Do not cite it as established fact. In the README, refer to the "faithfulness collapse" as a hypothesis this project tests, and leave a `[verify citation]` marker where the source belongs.
9. Use development questions (the `dev` split) for all tuning and diagnosis. The `test` split is touched once per final configuration.

## 1. Project summary

RAG systems that store a compressed KV cache can keep fluent answers while losing grounding in the retrieved text. This project does three things:

1. Measures grounding loss under KV quantization, using NLI entailment and a counterfactual test, not only exact match.
2. Diagnoses each failed query as a retrieval, chunking, quantization, or generation failure by rerunning it under controlled changes.
3. Uses reranker scores to decide which retrieved chunks keep high-precision KV entries and which are compressed to 2 bits, then compares that against uniform and random baselines at equal bit budgets.

The self-improvement loop reads the diagnosis and applies a fix per failure type (query rewriting, chunk size change, larger protected budget), then reports before and after numbers on held-out questions.

## 2. Hardware and constraints

- GPU: RTX 3050 with 4 GB VRAM. Ubuntu 22.04. Check with `nvidia-smi`.
- Main generator: `Qwen/Qwen2.5-1.5B-Instruct` loaded with 4-bit NF4 weights (bitsandbytes), compute dtype fp16. Expected weight memory is 1.5 to 2 GB. Measure it.
- Optional second generator: `meta-llama/Llama-3.2-1B-Instruct` (gated, needs HF login). Skip it if access is a problem.
- Embedder: `BAAI/bge-small-en-v1.5` (CPU is fine). Reranker: `BAAI/bge-reranker-base` (run on CPU or GPU, one at a time).
- NLI judge: a DeBERTa-v3 NLI checkpoint, for example `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` in fp16. If it does not fit, use `cross-encoder/nli-deberta-v3-base`. Verify the model id exists before coding against it.
- Context lengths: top-k chunks of 5, 10, and 20 (about 0.7k to 3k tokens). Do not attempt 32k or 64k locally. Report longer contexts analytically (section Phase 9).
- Verify the KV geometry from the model config instead of trusting this file: print `num_hidden_layers`, `num_key_value_heads`, `head_dim` for each model into `results/env.json`. For Qwen2.5-1.5B the expected values are 28 layers, 2 KV heads, head dim 128, which is about 28 KB of fp16 KV per token.

## 3. Repository layout

```
faithful-rag-kv/
  plan.md
  README.md
  BLOCKERS.md              # only if something blocks progress
  requirements.txt         # pinned versions
  Makefile                 # targets: setup, test, phaseN, all
  configs/                 # yaml, one file per experiment
  src/
    data/                  # loaders, counterfactual builder, splits
    rag/                   # chunking, embedding, retrieval, rerank, prompting, generation
    kv/                    # quantizer, fake-quant cache, bit accounting
    judge/                 # NLI scorer, EM/F1, counterfactual scorer
    diagnose/              # failure attribution
    improve/               # self-improvement loop
    viz/                   # plots
  scripts/                 # thin CLI entry points
  tests/                   # pytest, one file per phase
  results/                 # jsonl outputs, tables, plots, env.json
  reports/                 # phase_N.md
  demo/                    # gradio app
```

## Phase 0: environment

Tasks:
- Create a venv. Install torch (CUDA build matching the driver), transformers, accelerate, bitsandbytes, sentence-transformers, datasets, pytest, pandas, matplotlib, gradio, spacy (with `en_core_web_sm`), rank-bm25.
- Pin the transformers version. The `DynamicCache` API changes between releases, so record the version in `requirements.txt` and `results/env.json` and write all cache code against that version.
- Write `scripts/env_check.py` that prints the GPU name, VRAM, CUDA version, library versions, and loads the generator in 4-bit and generates 20 tokens.

Tests (`tests/test_phase0.py`):
- CUDA is available and total VRAM is at least 3.5 GB.
- The generator loads in 4-bit and `max_memory_allocated` after a 20-token generation is below 3.0 GB.
- `import bitsandbytes` works and a 4-bit linear layer runs a forward pass.

Gate: `pytest tests/test_phase0.py` passes. `results/env.json` exists and lists layers, KV heads, and head dim for each model.

## Phase 1: data and baseline RAG

Tasks:
1. Load HotpotQA distractor setting (`hotpotqa/hotpot_qa`, config `distractor`, validation split). Sample 400 questions with the fixed seed. Split into 100 `dev` and 300 `test`. Save the ids to `data/splits.json` and never change them.
2. Build one pooled corpus from the paragraphs of all 400 questions (about 4000 paragraphs, deduplicated by title and text). Retrieval runs over the whole pool, so wrong-question paragraphs act as realistic distractors. Keep the gold supporting paragraph titles per question for evaluation only.
3. Chunking, two modes selectable by config:
   - `paragraph`: one HotpotQA paragraph per chunk.
   - `window`: sentence windows of about 3 sentences with a stride of 2.
   Each chunk records its source title, sentence offsets, and text.
4. Retrieval: embed with bge-small, take the top 30 by cosine similarity, rerank with bge-reranker-base, keep the top k (k from config). Store both retriever and reranker scores per chunk.
5. Prompt template with a hard boundary so the KV cache can be split (see Phase 3):
   - Prefix part: system instruction plus the numbered chunks, each wrapped as `[Doc i]\n<text>\n\n`. End the prefix with a newline.
   - Suffix part: the question, the instruction "Answer in one complete sentence using only the documents above", and the assistant turn header.
   Record the exact token span `(start, end)` of every chunk in the prefix.
6. Generation: greedy decoding, `max_new_tokens=64`, fp (uncompressed) KV for now. Write one JSON line per question to `results/baseline_<config>.jsonl` with: question id, question, retrieved chunk ids and scores, prompt token count, answer, gold answer, gold supporting titles, latency.
7. Retrieval metrics: recall@k and precision@k on gold supporting titles, MRR of the first gold paragraph, and "all gold retrieved" rate.
8. Metrics for answers: exact match and token F1 after standard SQuAD normalization.

Tests (`tests/test_phase1.py`):
- `splits.json` has 100 dev and 300 test ids with no overlap, and loading it twice gives the same ids.
- Chunk token spans: slicing the prompt token ids with a recorded span and decoding it returns the chunk text (allow whitespace differences).
- Spans do not overlap and cover chunks in order.
- EM and F1 give known values on 5 hand-written pairs, including punctuation and article differences.
- Retrieval on a toy corpus of 5 paragraphs returns the obviously relevant one at rank 1.
- A rerun of the same config returns identical retrieved ids (determinism).

Gate: baseline run on `dev` finishes with k=5 and k=10. Print and save recall@k, EM, F1. Recall@10 of all gold paragraphs should be well above chance. If it is below 0.5, inspect the retrieval code for bugs before moving on. Write 10 random examples to `reports/phase_1.md` and read them.

## Phase 2: faithfulness metrics

Build the judging tools before any quantization experiment.

Tasks:
1. NLI faithfulness scorer (`src/judge/nli.py`). For each answer sentence, compute the entailment probability against each retrieved chunk (and each 3-sentence window inside long chunks, so inputs stay under 512 tokens). The score for an answer is the maximum entailment probability over chunks. Also store the maximum contradiction probability. Flag an answer as `unfaithful` when the score is below a threshold chosen on the hand-labeled set (below).
2. Counterfactual set (`src/data/counterfactual.py`):
   - Keep questions whose gold answer string appears verbatim in a gold supporting paragraph.
   - Replace every occurrence of the answer in the paragraphs with a substitute answer of the same coarse type, drawn from another question's answer (years with years, numbers with numbers, spaCy PERSON with PERSON, GPE with GPE, other with other).
   - Keep the original answer and the substitute together in the record.
   - Manually read 30 generated examples and write them to `reports/phase_2.md`. Drop the types where substitution reads as nonsense.
   - Scoring: `followed_context` if the substitute appears in the answer, `reverted_to_memory` if the original answer appears, otherwise `other`.
3. Hand-label 50 baseline answers as faithful or not (the human user does this, the agent prepares a `results/label_me.csv` with question, retrieved text, and answer columns and waits). Compute agreement between the NLI flag and the labels (accuracy and Cohen's kappa) and choose the threshold on this set. If the human is unavailable, use a proxy: the counterfactual `followed_context` label on 100 counterfactual answers, and say so in the report.
4. Judge runner script that reads a generation jsonl and writes a scored jsonl. It must free the generator from GPU before loading the NLI model, and run as its own process.

Tests (`tests/test_phase2.py`):
- NLI sanity: on 6 hand-written premise and hypothesis pairs (3 entailed, 3 contradicted), the scorer ranks every entailed pair above every contradicted pair.
- The scorer handles a 2000-token premise without crashing (windowing works).
- Counterfactual builder: for 20 samples, the original answer no longer appears in the edited paragraphs and the substitute does.
- Counterfactual scorer returns the three labels correctly on 6 synthetic answers.
- GPU memory during judging stays under 3.4 GB.

Gate: tests pass. NLI agreement with labels is reported with kappa. If kappa is below 0.4, try the other NLI checkpoint or sentence-level premises before continuing. Record the final threshold in `configs/judge.yaml`.

## Phase 3: fake KV quantization cache

This is the core engineering phase. Quantize and immediately dequantize, so quality effects are exact and no kernel is needed.

Design:
- Two-stage prefill. Run the prefix (system plus chunks) through the model and keep its cache. Apply quantize-dequantize to the prefix KV entries in place. Then run the suffix (question and assistant header) with that quantized cache as past, and decode from there. Causal attention makes this split exact for the fp case, which the test below checks. The suffix tokens and all generated tokens keep fp16 KV.
- Quantizer (`src/kv/quant.py`): asymmetric min-max with groups. For a tensor `x` and bit width `b`:

```python
def qdq(x, bits, dim, group):
    # x: tensor; groups of size `group` along `dim`, last group may be smaller
    # per group: zero = min, scale = (max - min) / (2**bits - 1), guard scale == 0
    # q = clamp(round((x - zero) / scale), 0, 2**bits - 1); return q * scale + zero
```

- Key and value layouts:
  - `uniform`: group along the channel dimension within each token, for both K and V.
  - `kivi`: keys grouped along the token dimension (per channel statistics), values grouped along the channel dimension (per token). This is the KIVI-style layout.
- Apply the quantizer per chunk span, never across a chunk boundary. A chunk shorter than the group size becomes one smaller group. This keeps each chunk a contiguous block, which is what a later kernel would need.
- The first 4 tokens of the prefix (attention sinks) always stay fp16 and are counted in the bit accounting.
- Bit accounting (`src/kv/bits.py`): effective bits per value = (sum over groups of `bits * n_values + 32`) divided by total values, where 32 is the fp16 scale plus fp16 zero point per group. Also expose nominal bits (without the overhead). Report both everywhere.
- Per-chunk precision map: a config gives each chunk a bit width, `{chunk_id: bits}`. Modes: `fp`, `uniform` (all chunks at b bits), `kivi` (all at b bits with the KIVI layout), `guided` (top p chunks by reranker score at `bits_high`, others at `bits_low`), `random_guided` (random p chunks), `oracle_guided` (gold supporting chunks protected, dev diagnostics only).
- Use the `DynamicCache` API of the pinned transformers version. If the API forbids in-place edits, subclass the cache and override `update`.

Tests (`tests/test_phase3.py`):
- Round trip: for random tensors, `qdq` at 8 bits has max abs error below 1% of the group range. At 2 bits the error is below the group range divided by 3 (half a step). A constant tensor returns the constant (scale guard).
- Idempotence: applying `qdq` twice gives the same result as once.
- Bit accounting: for a known toy case (one chunk of 64 tokens, group 32, 2 bits), effective bits equal `2 + 32/32 = 3.0` per value and nominal bits equal 2.0. Mixed chunks produce the weighted average.
- Split equivalence: with mode `fp`, logits of the last prefix token and the first 20 generated tokens under two-stage prefill match single-pass generation (greedy tokens identical, max logit difference below 1e-2 in fp16).
- Boundary check: tokenizing prefix and suffix separately gives the same ids as tokenizing the joined prompt. If it does not, change the split point to a place where it does (a newline or special token).
- Quantization must change the output: mode `uniform` at 2 bits gives a KV tensor that differs from fp and gives a different logit vector for at least one test prompt.
- 8-bit sanity: on 30 dev questions, uniform 8-bit generations match fp generations on at least 90% of questions. If not, there is a bug.
- Memory: a 4k-token prefill plus 64-token decode stays under 3.4 GB peak.

Gate: all tests pass, and a small table (fp, uniform 8, uniform 4, uniform 2, kivi 2) with effective bits and the fraction of generations identical to fp is saved to `reports/phase_3.md`.

## Phase 4: baseline faithfulness audit

Tasks:
- Run these configs on `dev` first, then `test`, for k in {5, 10}: `fp`, `uniform` at 8, 4, 2 bits, and `kivi` at 4 and 2 bits (group 32 and 64 for the 2-bit rows).
- Run on three sets: HotpotQA normal, HotpotQA counterfactual, and the RGB benchmark noisy-context subset (download from the RGB repository, use the English noise-robustness file, take 200 questions; if it cannot be downloaded, document it and skip RGB).
- Score every output with EM, F1, NLI faithfulness, hallucination rate (share with `unfaithful` flag), and counterfactual `followed_context` rate.
- Use paired bootstrap (1000 resamples) over questions for every comparison against `fp`. Report a 95% interval, not only point values.
- Produce `results/tables/audit.csv` and a plot of faithfulness and EM against effective bits.

Tests (`tests/test_phase4.py`):
- Every cached output file has exactly one record per question per config (no gaps, no duplicates).
- Re-running a config reads from the cache and finishes in under 10 seconds.
- Bootstrap function returns an interval that contains the mean difference, and returns a zero-width interval for two identical arrays.
- Result table has no NaN values.

Gate and interpretation: write what the data shows. Three outcomes are possible and all are valid findings: (a) EM stays flat while faithfulness or counterfactual following drops, which reproduces the hypothesis; (b) both drop; (c) neither drops at this model scale. In case (c), push the experiment (lower bits, longer context with k=20, group 128) until an effect appears or document that it does not. Do not continue to Phase 6 without a measurable gap between `fp` and some uniform low-bit mode, because the method has nothing to repair otherwise.

## Phase 5: failure attribution

Goal: put every failed dev query into exactly one category.

Definition of a failed query: EM is 0 and F1 is below 0.5, or the NLI flag is `unfaithful`, or (for counterfactual items) the answer did not follow the context.

Decision procedure for a failure under a quantized config (`src/diagnose/attribute.py`), applied in this order:
1. Rerun the same query with the same retrieved chunks under `fp` KV. If it now passes: `QUANT`.
2. Else, check retrieval. If not all gold supporting paragraphs are in the retrieved set: `RETRIEVAL`. Within window chunking, if a gold supporting sentence is split across two chunks and fewer than all pieces were retrieved: `CHUNKING` (checked before `RETRIEVAL`).
3. Else, rerun with only the gold supporting paragraphs as context, `fp` KV. If it passes: `CONTEXT_NOISE` (distractor chunks confuse the model).
4. Else: `GENERATION` (the model cannot answer even with perfect context).

For failures under the `fp` config, skip step 1.

Outputs: `results/failures_dev.jsonl` (one record per failure with category and the evidence for it), a bar chart of category counts per config, and an example gallery of 3 cases per category in `reports/phase_5.md`.

Tests (`tests/test_phase5.py`), using a mock generator and mock retriever so they run without a GPU:
- Synthetic fixtures for each of the five categories produce exactly that category.
- A query that fails under 2-bit and passes under fp is always `QUANT`, regardless of retrieval state.
- Every failure gets one category and categories sum to the failure count.
- Order check: a case where both quantization and retrieval are broken is labeled `QUANT` only if the fp rerun passes, and otherwise falls through to the next rule.

Gate: tests pass. On the real dev run, read 20 attributed failures by hand and confirm at least 80% of the labels look right. Record the check in the report. If under 80%, find the rule that misfires.

## Phase 6: retrieval-guided precision

Tasks:
- Implement `guided` mode (Phase 3 already defines it). Sweep p in {0, 1, 2, 3, 5} protected chunks out of k=10, with `bits_high` in {8} and `bits_low` in {2}. `p=0` equals uniform 2-bit and `p=k` equals uniform 8-bit.
- Baselines at matched effective bits: for each guided point, run uniform quantization at the bit width that gives the nearest effective bits (add a 3-bit mode if needed), and `random_guided` with the same p over 3 random seeds.
- Also run `oracle_guided` (protect gold chunks) as an upper bound.
- Plot faithfulness, counterfactual following, and F1 against effective bits for all methods. Put the memory axis in analytic KV size for a 4k context.

Tests (`tests/test_phase6.py`):
- The protected chunks in `guided` mode are exactly the top p by reranker score (check on 5 fixtures with ties broken by chunk id).
- The effective bits of each config match an independent recomputation from the chunk map.
- `p=0` reproduces `uniform` 2-bit outputs exactly, and `p=k` reproduces `uniform` 8-bit outputs exactly.
- Random-guided with the same seed gives the same chunk choice.

Gate: results exist for all points on dev and test. State the finding with intervals: does guided beat random and uniform at the same effective bits, by how much, and is the paired bootstrap interval clear of zero. If guided does not beat random, say so. In that case look at whether the reranker's top chunks contain the gold evidence (a retrieval problem) before concluding the idea fails.

## Phase 7: self-improvement loop

Tasks (`src/improve/loop.py`):
1. Read `failures_dev.jsonl` and count categories for the chosen baseline config (uniform 2-bit, k=10).
2. Policies, one per category:
   - `RETRIEVAL`: query rewriting with the generator (produce 2 alternative queries), retrieve for each, fuse with reciprocal rank fusion, and raise the rerank pool from 30 to 50.
   - `CHUNKING`: switch the chunk mode from `window` to `paragraph`, or change window size and overlap.
   - `QUANT`: raise the number of protected chunks p up to a budget cap (stop at 3.5 effective bits).
   - `CONTEXT_NOISE`: reduce k or add a reranker score cutoff.
   - `GENERATION`: log only, no fix.
3. Apply policies on `dev`, choose the combination with the best dev F1 and faithfulness, then evaluate it once on `test`. Compare against the starting config with a paired bootstrap.
4. Report before and after: recall@k, precision@k, MRR, EM, F1, NLI faithfulness, counterfactual following, effective bits, and the per-category failure counts.
5. Note in the README that diagnosis uses gold labels on dev as an offline signal. At inference time the loop uses only proxies (reranker score margin and NLI score). Do not claim online self-repair without gold unless a proxy-based version is also built and measured.

Tests (`tests/test_phase7.py`):
- A policy applied to a synthetic failure list changes only the config fields it owns.
- No test-split id appears in any file read during tuning (assert on the file access log or on the ids passed in).
- RRF fusion returns a stable ranking on a hand-computed example.
- The loop is idempotent: running it twice on the same inputs gives the same chosen config.

Gate: the before and after table exists. The improvement on `test` is reported with an interval. If there is no gain, report that along with the per-category counts, which explain why.

## Phase 8: ablations

Run on dev, report the best on test:
- Chunk-level reranker signal versus token-level attention saliency (mean attention received by each context token from the question tokens in a cheap prefill pass, summed per chunk). No gradients. This tests how much of the benefit the cheap signal keeps.
- Group size {16, 32, 64, 128} for 2-bit.
- `bits_low` in {2, 3} and `bits_high` in {4, 8}.
- Sink tokens protected versus not.
- Reranker choice: bge-reranker-base versus embedding score only.

Tests: each ablation writes a config hash into its output, and a test checks that no two ablation outputs share a hash with different settings.

Gate: one table per ablation in `results/tables/`, and a paragraph per ablation in `reports/phase_8.md` stating the effect with intervals.

## Phase 9: packaging

Deliverables:
- `README.md` that opens with what the project does and how to reproduce it, then the main result table, the Pareto plot (effective bits against faithfulness), the failure-category chart, before and after table, ablations, and a limitations section. Limitations to state plainly: small model (1.5B, 4-bit weights), 300 test questions, fake quantization instead of packed storage, diagnosis uses gold labels, short contexts.
- Analytic memory table for 4k, 16k, 32k, and 64k contexts for Qwen2.5-1.5B and for Llama-3-8B (computed from layers, KV heads, head dim, and measured effective bits, clearly labeled as computed not measured).
- `make all` reproduces the tables from cached outputs in under 10 minutes without a GPU.
- Gradio demo in `demo/app.py`: type a question, show the retrieved chunks, and show the answer under fp, uniform 2-bit, and guided KV side by side with NLI faithfulness scores and effective bits. It must run within 4 GB.
- `reports/resume_notes.md`: draft resume bullets with placeholders replaced only by measured values from `results/`. Include cost or latency numbers if they were logged.

Tests (`tests/test_phase9.py`):
- `make test` runs all phase tests green from a clean checkout.
- Every number in the README main table appears in a CSV under `results/tables/` (script that greps and compares).
- The demo starts and answers one query in under 60 seconds.
- `requirements.txt` installs in a fresh venv.

Gate: a fresh clone passes `make setup && make test && make all`.

## Stretch: packed 2-bit Triton kernel

Only start this after Phase 9 passes.

- Write a Triton kernel (Ampere, compute capability 8.6) that unpacks 2-bit values for one chunk block with group scales and zeros and writes fp16 output, then fuse it with the attention dot product if time allows.
- Test against the PyTorch `qdq` reference with `torch.allclose` (atol 1e-3 in fp16) on random inputs of several shapes, including a chunk shorter than the group size.
- Benchmark: measured KV memory and decode tokens per second versus the fake-quant path. Report the real effective bits including scale and zero overhead. Save to `results/kernel_bench.csv`.
- Do not claim FlashDecoding-level throughput unless it is measured.

## Final acceptance checklist

- [ ] All phase gates passed and committed.
- [ ] `results/env.json` has versions and KV geometry.
- [ ] Judge validated against hand or counterfactual labels, with kappa reported.
- [ ] Audit table shows fp versus quantized configs with intervals.
- [ ] Failure categories produced and spot checked.
- [ ] Guided versus random versus uniform compared at matched effective bits.
- [ ] Before and after loop results reported on held-out test questions.
- [ ] README limitations section written.
- [ ] No unverified citation stated as fact.
- [ ] Resume bullets contain only measured numbers.

## Known risks and what to do about them

- Qwen2.5-1.5B has only 2 KV heads, so each head carries a lot of information and may be fragile at 2 bits. That helps the audit show an effect. If the effect is too strong (everything fails at 2 bits), use 3-bit as the low setting.
- A 1.5B model may fail multi-hop questions even with perfect context. Check the `GENERATION` share in Phase 5. If it dominates, restrict the analysis to the questions the model answers correctly under `fp` and `oracle` context.
- bitsandbytes 4-bit weights add their own noise. Keep weights fixed across all configs so only the KV cache varies.
- HotpotQA is mostly short-answer. If NLI on one-sentence answers is unreliable, rely on the counterfactual metric as the primary faithfulness measure and keep NLI as secondary.
- transformers cache internals change between versions. Do not upgrade the library mid-project.
