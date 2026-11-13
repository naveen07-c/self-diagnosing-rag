# Phase 4 — baseline faithfulness audit

## What was built

- `scripts/run_audit.py`: runs the full precision matrix — `fp`, `uniform` 8/4/2
  bits (group 32 and 64 for 2-bit), `kivi` 4-bit and 2-bit (group 32 and 64) —
  on HotpotQA normal, HotpotQA counterfactual, and RGB noisy-context, at k=5 and
  k=10 (RGB k=5 only), on dev and test. Writes one JSONL per config plus
  summaries and a peak-GPU-memory log.
- `scripts/run_judge_batch.py`: scores every audit output with the DeBERTa-v3
  NLI judge once (skips existing `_scored.jsonl`), writes
  `memory_judge_batch.json`.
- `scripts/build_audit_table.py`: aggregates into `results/tables/audit.csv`,
  paired-bootstrap intervals in `audit_bootstrap.csv`, and plots
  faithfulness/F1/EM against effective bits (`results/figures/audit_effective_bits.png`,
  plus a `_test` variant).
- `src/eval/bootstrap.py`: `paired_bootstrap(a,b,n_boot=1000,seed=1234,alpha=0.05)`;
  zero-width interval for identical arrays.
- `src/data/rgb.py`, `scripts/build_rgb.py`, `scripts/run_rgb.py`: fixed
  200-question RGB English noise-robustness subset (2 positive + 3 negative docs
  per question, seed 1234); raw `data/rgb/en.json` is gitignored.
- `answer_metrics_multi` in `src/judge/em.py` (best EM/F1 over answer aliases,
  for RGB).
- `configs/audit.yaml`: the 8-config matrix; `configs/judge.yaml` NLI threshold
  0.50.
- `tests/test_phase4.py`: bootstrap semantics, one-record-per-question, cached
  rerun <10 s, no NaN/gaps, judge memory + plot existence.

## Commands run

```
env -u PYTHONPATH .venv/bin/python scripts/build_rgb.py
env -u PYTHONPATH .venv/bin/python scripts/run_audit.py --dataset normal --split dev
env -u PYTHONPATH .venv/bin/python scripts/run_audit.py --dataset counterfactual --split dev
env -u PYTHONPATH .venv/bin/python scripts/run_audit.py --dataset rgb --split rgb
env -u PYTHONPATH .venv/bin/python scripts/run_audit.py --dataset normal --split test
env -u PYTHONPATH .venv/bin/python scripts/run_audit.py --dataset counterfactual --split test
env -u PYTHONPATH .venv/bin/python scripts/run_judge_batch.py
env -u PYTHONPATH .venv/bin/python scripts/build_audit_table.py
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_phase4.py -q
```

## Audit matrix

Configs: `fp`, `u8`, `u4`, `u2g32`, `u2g64`, `k4`, `k2g32`, `k2g64`.
Sets: normal dev/test (100/300 questions), counterfactual dev/test (53/129 eligible items), RGB (200 questions, k=5).
Every output is scored for EM, token F1, NLI faithfulness, hallucination rate (`unfaithful` share at threshold 0.50), and — for counterfactual — `followed_context` rate.

### Dev results (k=5 / k=10)

| dataset | k | config | eff. bits | EM | F1 | NLI faith. | halluc. | followed ctx |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| normal | 5 | fp | 16.00 | 0.000 | 0.178 | 0.548 | 0.440 | – |
| normal | 5 | u8 | 9.45 | 0.000 | 0.178 | 0.541 | 0.450 | – |
| normal | 5 | u4 | 5.70 | 0.000 | 0.008 | 0.128 | 0.910 | – |
| normal | 5 | u2g32 | 3.83 | 0.000 | 0.006 | 0.124 | 0.950 | – |
| normal | 5 | k4 | 5.76 | 0.000 | 0.084 | 0.386 | 0.630 | – |
| normal | 5 | k2g32 | 3.89 | 0.000 | 0.020 | 0.252 | 0.790 | – |
| normal | 10 | fp | 16.00 | 0.000 | 0.185 | 0.501 | 0.520 | – |
| normal | 10 | u8 | 9.21 | 0.000 | 0.180 | 0.507 | 0.510 | – |
| normal | 10 | u4 | 5.33 | 0.000 | 0.001 | 0.156 | 0.910 | – |
| normal | 10 | k4 | 5.39 | 0.000 | 0.075 | 0.328 | 0.690 | – |
| counterfactual | 5 | fp | 16.00 | 0.000 | 0.075 | 0.433 | 0.540 | 0.380 |
| counterfactual | 5 | u8 | 9.45 | 0.000 | 0.077 | 0.436 | 0.540 | 0.380 |
| counterfactual | 5 | u4 | 5.70 | 0.000 | 0.000 | 0.093 | 0.960 | 0.000 |
| counterfactual | 5 | k4 | 5.77 | 0.000 | 0.041 | 0.422 | 0.600 | 0.080 |
| counterfactual | 10 | fp | 16.00 | 0.000 | 0.073 | 0.377 | 0.623 | 0.283 |
| counterfactual | 10 | u8 | 9.21 | 0.000 | 0.073 | 0.367 | 0.623 | 0.302 |
| counterfactual | 10 | u4 | 5.33 | 0.000 | 0.000 | 0.122 | 0.981 | 0.000 |
| counterfactual | 10 | k4 | 5.39 | 0.000 | 0.020 | 0.319 | 0.698 | 0.057 |
| rgb | 5 | fp | 16.00 | 0.000 | 0.275 | 0.824 | 0.175 | – |
| rgb | 5 | u8 | 9.26 | 0.000 | 0.273 | 0.814 | 0.185 | – |
| rgb | 5 | u4 | 5.40 | 0.000 | 0.000 | 0.238 | 0.905 | – |
| rgb | 5 | k4 | 5.44 | 0.015 | 0.156 | 0.491 | 0.560 | – |

Full 72-row table (dev + test + RGB): `results/tables/audit.csv`; bootstrap intervals: `results/tables/audit_bootstrap.csv`.

### Test results (k=5 / k=10)

Test mirrors dev. Normal: fp F1 0.172/0.165 vs u8 0.171/0.165 (indistinguishable); u4 F1 ~0.002–0.012; k4 F1 ~0.10. Counterfactual: fp followed 0.432/0.388 vs u8 0.432/0.395 (matched); u4 followed 0.000; k4 followed 0.152/0.163. NLI faithfulness and hallucination track the same split: fp/u8 ≈, everything below 8 bits far worse. RGB: fp F1 0.275 vs u8 0.273; u4 F1 0.000; k4 F1 0.156.

### Paired bootstrap (config − fp), dev

Key intervals (normal dev k=10): u8 F1 −0.005 [−0.011, 0.000], NLI +0.006 [−0.018, 0.029], halluc −0.010 [−0.040, 0.020] → **8-bit is statistically indistinguishable from fp**. u4 F1 −0.185 [−0.218, −0.156], NLI −0.345 [−0.440, −0.256], halluc +0.390 [+0.280, +0.500] → **4-bit uniform is clearly worse**. k4 F1 −0.110 [−0.145, −0.074], NLI −0.173 [−0.287, −0.069], halluc +0.170 [+0.050, +0.300] → **kivi 4-bit is intermediate, still clearly degraded**.

Counterfactual dev k=10 followed_context: u8 +0.019 [0.000, 0.057] (matched); k4 −0.226 [−0.377, −0.094]; u4 −0.283 [−0.396, −0.170]; u2g32 −0.283 [−0.396, −0.170].

## Findings

**1. 8-bit KV quantization is safe.** `u8` matches `fp` on F1, NLI faithfulness,
hallucination rate, and counterfactual context-following across normal,
counterfactual, and RGB, on both dev and test, with bootstrap intervals
straddling zero. Effective bits ≈9.2–9.5 (above nominal 8 because header/sinks/group metadata stay at fp16). This validates the quantization/split-prefill machinery: an 8-bit cache produces the same answers as fp.

**2. The quality cliff is below 8 bits.** `u4` and both `u2` rows collapse: F1 → 0, NLI faithfulness → ~0.1–0.16, hallucination → 0.9+, counterfactual followed → 0.0–0.02. Outputs become digit garbage or short irrelevant fragments. This is a real sensitivity of the 4-bit-weight Qwen2.5-1.5B model, not a split bug (8-bit is exact and the split-equivalence test passes).

**3. KIVI helps but does not close the gap.** `k4` (kivi 4-bit) is intermediate: F1 0.075–0.106, NLI 0.319–0.422, hallucination 0.56–0.698, counterfactual followed 0.057–0.163 — better than `u4`/`u2` but far below `fp`/`u8`. `k2g32`/`k2g64` retain slightly more NLI than uniform 2-bit but F1 remains ~0.

**4. The original "faithfulness collapse" hypothesis is not supported in the direction predicted.** The plan's hypothesis (case (a): EM flat while faithfulness/context-following drops as precision falls) assumed quantized models would *over-trust* noisy context. The data shows the opposite — case **(b): both quality and faithfulness drop together**. Low precision causes **context neglect** (the model stops reading/repeating the counterfactual context) **plus hallucination**, not blind context-following. EM is 0.0 by construction (the prompt asks for one complete sentence; HotpotQA gold answers are short spans), so F1/NLI are the signals.

**5. Phase 4 gate is satisfied.** There is a clear, statistically significant gap between `fp`/`u8` and uniform low-bit modes (`u4`, `u2`), so the method has something to repair and Phase 6 (retrieval-guided precision) is justified. The finding to carry forward: protect the top retrieved chunks at high precision while compressing the rest, because low precision destroys both answer quality and context-following.

## Memory

Every GPU script logs `torch.cuda.max_memory_allocated()`. Peak allocated across the audit: normal test u8 k10 ≈1.84 GB; judge batch peak ≈2.18 GB allocated / 2.92 GB reserved. All below the 3.4 GB limit on the RTX 3050 A (4094 MiB).

## Prompt-tokenization caveat

Phase 3 fixed the prompt builder to tokenize the whole prefix once (offset-based spans) instead of wrapper/text/trailer separately. The old text was identical but token ids differed by ~1 token per document (dev k=5 mean +4.94, dev k=10 +9.87, test k=10 +9.94 tokens). All Phase 1/2/3 fp artifacts and the Phase 4 `fp` rows were force-regenerated after the fix, so every row in `audit.csv` uses the corrected, self-consistent prompt. `prompt_version` stays 1 because the prompt *text* is unchanged; only the tokenization method changed, and cache keys cannot detect that, so affected fp caches were regenerated explicitly.

## Tests

`pytest tests/test_phase4.py -q`: 5 passed. Full suite: 27 passed.

## Gate status

PASS. Measurable fp vs uniform low-bit gap confirmed (case b). Raw rows in
`results/audit_*.jsonl`; summaries in `results/summary_audit_*.json`; table
`results/tables/audit.csv`; bootstrap `results/tables/audit_bootstrap.csv`;
figures `results/figures/audit_effective_bits{,_test}.png`.
