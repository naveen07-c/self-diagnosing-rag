# self-diagnosing-rag

Self-diagnosing RAG with retrieval-guided KV-cache precision.

A project that measures grounding loss under KV-cache quantization (NLI entailment
+ a counterfactual test, not only exact match), diagnoses each failed query into
exactly one category (retrieval / chunking / quantization / context-noise /
generation), and experiments with retrieval-guided precision (protect the top
reranked chunks at high precision, compress the rest to 2 bits).

Built and measured on an RTX 3050 A Laptop GPU (3.648 GiB usable to torch) with
`Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4.

## What the project does

1. **Phase 0–1:** Fixed HotpotQA distractor splits (100 dev / 300 test), pooled
   corpus, paragraph + window chunking, bge-small + bge-reranker retrieval,
   two-stage prefill generation, EM/F1 metrics.
2. **Phase 2:** NLI faithfulness judge (DeBERTa-v3) + counterfactual answer
   substitution set (PERSON / GPE / year / number) for a context-following
   faithfulness signal.
3. **Phase 3:** Fake KV quantization (quantize–dequantize in place) with
   channel / KIVI layouts, split prefill, bit accounting (effective + nominal
   bits). 8-bit is token-identical to fp; below 8 bits quality collapses.
4. **Phase 4:** Baseline faithfulness audit across fp / uniform-8/4/2 / kivi-4/2
   on normal, counterfactual, and RGB noisy-context, dev + test, k = 5 / 10.
   Paired-bootstrap 95% intervals vs fp. Table + plot of faithfulness/F1/EM
   against effective bits.
5. **Phase 5:** Failure attribution: every failed dev query assigned to one of
   QUANT, CHUNKING, RETRIEVAL, CONTEXT_NOISE, GENERATION by rerunning under
   controlled changes (fp rerun, gold-paragraph context).

## Main result (Phase 4 audit, dev split)

| dataset | k | config | eff. bits | F1 | NLI faith. | halluc. | followed ctx |
|---|---:|---|---:|---:|---:|---:|---:|
| normal | 10 | fp | 16.00 | 0.185 | 0.501 | 0.520 | – |
| normal | 10 | u8 | 9.21 | 0.180 | 0.507 | 0.510 | – |
| normal | 10 | u4 | 5.33 | 0.001 | 0.156 | 0.910 | – |
| normal | 10 | k4 | 5.39 | 0.075 | 0.328 | 0.690 | – |
| counterfactual | 10 | fp | 16.00 | 0.073 | 0.377 | 0.623 | 0.283 |
| counterfactual | 10 | u8 | 9.21 | 0.073 | 0.367 | 0.623 | 0.302 |
| counterfactual | 10 | u4 | 5.33 | 0.000 | 0.122 | 0.981 | 0.000 |
| counterfactual | 10 | k4 | 5.39 | 0.020 | 0.319 | 0.698 | 0.057 |
| rgb | 5 | fp | 16.00 | 0.275 | 0.824 | 0.175 | – |
| rgb | 5 | u8 | 9.26 | 0.273 | 0.814 | 0.185 | – |
| rgb | 5 | u4 | 5.40 | 0.000 | 0.238 | 0.905 | – |

**Finding:** uniform 8-bit is statistically indistinguishable from fp (paired
bootstrap intervals straddle zero). Below 8 bits both quality and
faithfulness drop together — low precision causes *context neglect plus
hallucination*, not blind over-trust of the retrieved context. So the
audit confirms a measurable fp vs low-bit gap (case (b) in the plan), which is
what justifies Phase 6 (retrieval-guided precision).

## Repository layout

```
plan.md              # build spec, phases 0-9, rules
README.md            # this file
requirements.txt     # pinned library versions
Makefile             # setup, test, phase0..phase9, all, clean
configs/             # one yaml per experiment
src/
  data/              # HotpotQA loaders, splits, counterfactual builder, RGB
  rag/               # chunking, retrieval, rerank, prompting, generation
  kv/                # fake quantizer, bit accounting, cache modes
  judge/             # NLI scorer, EM/F1, counterfactual scorer
  diagnose/          # failure attribution (Phase 5)
  eval/              # paired bootstrap
scripts/             # thin CLI entry points (one per phase)
tests/               # pytest, one file per phase
reports/             # phase_N.md
results/             # gitignored: jsonl outputs, tables, plots, env.json
data/                # gitignored: splits, corpus, chunks, embeddings
```

## How to reproduce

```bash
# 0. environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm

# clear ROS-leaked PYTHONPATH (needed on this machine)
export PYTHONPATH=  # or prefix every command with `env -u PYTHONPATH`

# 1. data
.venv/bin/python scripts/build_data.py

# 2. counterfactual set
.venv/bin/python scripts/prepare_counterfactual.py

# 3. baseline (k=5, k=10)
.venv/bin/python scripts/run_baseline.py --config configs/baseline_k5.yaml
.venv/bin/python scripts/run_baseline.py --config configs/baseline_k10.yaml

# 4. NLI judge scoring
.venv/bin/python scripts/run_judge.py --mode score --input ... --output ...

# 5. KV quantization sweep
.venv/bin/python scripts/run_kv_sweep.py --config configs/kv_phase3.yaml

# 6. faithfulness audit (dev then test) + judge batch + table
.venv/bin/python scripts/run_audit.py --dataset normal --split dev
.venv/bin/python scripts/run_audit.py --dataset counterfactual --split dev
.venv/bin/python scripts/run_audit.py --dataset rgb --split rgb
.venv/bin/python scripts/run_audit.py --dataset normal --split test
.venv/bin/python scripts/run_audit.py --dataset counterfactual --split test
.venv/bin/python scripts/run_judge_batch.py
.venv/bin/python scripts/build_audit_table.py

# 7. failure attribution
.venv/bin/python scripts/run_diagnose.py --split dev

# tests
.venv/bin/python -m pytest tests -q
```

Every GPU-bound script prints `torch.cuda.max_memory_allocated()` at exit and
must stay under 3.4 GB.

## Requirements

**Python libraries (pinned, see `requirements.txt`):**
torch==2.11.0+cu128, transformers==4.57.6, accelerate==1.12.0,
bitsandbytes==0.50.2, datasets==5.0.1, sentence-transformers==5.1.2,
spacy==3.8.18 (+ `en_core_web_sm`), gradio==6.17.3, matplotlib==3.11.2,
pandas==3.1.1, pytest, pyyaml, numpy, rank-bm25.

**Models downloaded at runtime from Hugging Face (not stored in this repo):**
- `Qwen/Qwen2.5-1.5B-Instruct` (generator, 4-bit NF4)
- `BAAI/bge-small-en-v1.5` (embedder)
- `BAAI/bge-reranker-base` (reranker)
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (NLI judge)

**Large/generated artifacts are NOT in this repo** — regenerate them by running
the scripts above, or download where noted:
- `data/splits.json`, `data/questions.jsonl`, `data/corpus.jsonl` → `scripts/build_data.py` (downloads `hotpotqa/hotpot_qa`)
- `data/chunks_paragraph.jsonl`, `data/chunks_window.jsonl` → `scripts/build_data.py`
- `data/emb_*.npy` (embedding matrix) → `scripts/build_retrieval.py`
- `data/counterfactual.jsonl` → `scripts/prepare_counterfactual.py`
- `data/rgb_noise.jsonl` → `scripts/build_rgb.py` (downloads the RGB English
  noise-robustness file from github.com/chen700564/RGB)
- `results/*.jsonl`, `results/*.png`, `results/*.json` → the `run_*.py` scripts
  and `scripts/build_audit_table.py` (all cached per `config_hash` + question id)

## Citation note

The plan references `arXiv:2608.30996` for the original "faithfulness collapse"
proposal. That citation is **unverified** — it is treated as a hypothesis this
project tests, not established fact. See `plan.md` rule 8.

## Limitations

- Small model (1.5B, 4-bit weights), only 2 KV heads — 2-bit compression is
  harsh and may dominate the analysis.
- 300 test questions; short contexts (top-5 / top-10 paragraphs).
- Fake quantization (quantize–dequantize in fp16), not packed 2-bit storage.
- Diagnosis uses gold supporting paragraphs as an offline signal; at inference
  time only proxies (reranker score, NLI score) are available.
- NLI on one-sentence answers is weak; the counterfactual metric is primary,
  NLI is secondary.

## Phase status

- Phase 0–5: complete and committed. `pytest tests -q`: 27 passed.
- Phase 6+ (guided precision, self-improvement loop, ablations, packaging):
  not yet implemented.
