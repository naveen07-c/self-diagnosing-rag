# Phase 2 — faithfulness metrics

## What was built

- `src/judge/nli.py`: DeBERTa-v3 NLI scorer. Each answer sentence is scored
  against each retrieved chunk; for chunks over 450 tokens it also scores
  3-sentence windows. The answer score is the minimum over answer sentences of
  the maximum entailment over premises (conservative), and the maximum
  contradiction is stored. Chunks whose full text fits under 450 tokens are
  scored whole, not only as 3-sentence windows.
- `src/data/counterfactual.py`: type-matched answer substitution. We replace the
  gold answer in the gold paragraphs with a same-type answer from another
  question. `followed_context`, `reverted_to_memory` and `other` are scored by
  phrase containment.
- `src/judge/score.py`, `scripts/run_counterfactual.py`, `scripts/run_judge.py`.
  The judge runner is a separate process and never loads the generator.
- `results/label_me.csv`: 50-row human-label sheet (question, retrieved text,
  answer, empty label column).

## Counterfactual set

`data/counterfactual.jsonl` contains 195 items (56 dev, 139 test):

| type | count |
|---|---:|
| PERSON | 112 |
| GPE | 52 |
| year | 23 |
| number | 8 |
| other | dropped |

The 30 examples below were read manually. Year, number, PERSON and GPE
substitutions form grammatical noun phrases and were kept. The `other` type was
dropped because replacements such as "the virus is fixed-roof, meaning..." or
"What colors are worn by ... Italian nationality law" break grammar. The
generated list is also saved at `results/counterfactual_examples.md`.

## Judge calibration

We prepared `results/label_me.csv` for human labels; it was not filled in, so we
used the plan's proxy: the counterfactual `followed_context` label. Of the 56
eligible dev items, 53 had at least one retrieved chunk containing the original
answer and no retrieved chunk already containing the substitute. Generation with
fp KV, k=10 produced:

- `followed_context`: 16
- `reverted_to_memory`: 4
- `other`: 33

Calibration was weak, and we report that plainly rather than hiding it:

| model | premises | proxy labels | best threshold | accuracy | Cohen's kappa |
|---|---|---|---:|---:|---:|
| DeBERTa-v3-base | 3-sentence windows | all 53 | 0.94 | 0.717 | 0.233 |
| DeBERTa-v3-large | 3-sentence windows | all 53 | 0.86 | 0.717 | 0.233 |
| DeBERTa-v3-large | whole chunk / windows | all 53 | 0.90 | 0.736 | 0.299 |
| DeBERTa-v3-large | whole chunk / windows | clear 20 (follow/revert only) | 0.01 | 0.850 | 0.483 |

The large checkpoint was kept (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`).
It cleanly separates the 4 reverted-to-memory answers (all scores <= 0.03), but
many answers that do mention the substitute are only partially entailed by any
single premise, and many `other` answers are faithful statements copied from the
context. The all-53 kappa stays below 0.4 even after trying both checkpoints and
whole-chunk premises. Following the plan's risk note, the counterfactual metric
is treated as the **primary** faithfulness measure and NLI as **secondary**.
The recorded operating threshold is 0.50 (neutral point); the best-kappa
threshold on all 53 was 0.90 (kappa 0.299, but it flags ~81% of answers
unfaithful). This choice and the weak kappa are stated in the README limitations.

Baseline k=10 fp NLI scoring (100 dev questions) gave mean faithfulness 0.513,
maximum contradiction 0.535, and hallucination rate 0.510 at threshold 0.50.
Raw output: `results/baseline_baseline_k10_scored.jsonl`.

## Tests

```
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_phase2.py -q
5 passed in 7.12s
```

- NLI sanity: all 3 entailed pairs rank above all 3 contradicted pairs.
- A 2000-token premise is split into windows without crashing.
- 20 counterfactual samples: original absent and substitute present after edit.
- Counterfactual scorer returns the three labels on 6 synthetic answers.
- Judge peak GPU memory in the test: below 3.4 GB (observed < 1.8 GB during the
  53-item scoring run; exact value in the memory log).

## Gate status

PASS with a documented caveat. NLI agreement with proxy labels is reported with
kappa above; it is below 0.4 on the full proxy, but the alternative checkpoint
and sentence-level/whole-chunk premises were tried. The final threshold, model
id and calibration metadata are in `configs/judge.yaml`; calibration curve in
`results/judge_calibration.json`. Per the risk note, the counterfactual metric is
primary for the audit.

## Appendix — 30 counterfactual examples (manually read)

... (see full report with examples 1-30) ...
