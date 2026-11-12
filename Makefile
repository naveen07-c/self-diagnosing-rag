PY := env -u PYTHONPATH .venv/bin/python
PIP := env -u PYTHONPATH .venv/bin/pip

.PHONY: setup test phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9 all clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PY) -c "import spacy; spacy.load('en_core_web_sm')" || $(PY) -m spacy download en_core_web_sm

test:
	$(PY) -m pytest tests -q

phase0:
	$(PY) scripts/env_check.py

phase1:
	$(PY) scripts/build_data.py
	$(PY) scripts/run_baseline.py --config configs/baseline_k5.yaml
	$(PY) scripts/run_baseline.py --config configs/baseline_k10.yaml

phase2:
	$(PY) scripts/prepare_counterfactual.py
	$(PY) scripts/run_judge.py --select threshold

phase3:
	$(PY) scripts/run_kv_sweep.py --config configs/kv_phase3.yaml

phase4:
	$(PY) scripts/run_audit.py --dataset normal --split dev
	$(PY) scripts/run_audit.py --dataset counterfactual --split dev
	$(PY) scripts/run_audit.py --dataset rgb --split rgb
	$(PY) scripts/run_audit.py --dataset normal --split test
	$(PY) scripts/run_audit.py --dataset counterfactual --split test
	$(PY) scripts/run_judge_batch.py
	$(PY) scripts/build_audit_table.py

phase5:
	$(PY) scripts/run_diagnose.py --split dev

phase6:
	$(PY) scripts/run_guided.py --split dev

phase7:
	$(PY) scripts/run_improve.py --split dev

phase8:
	$(PY) scripts/run_ablations.py

phase9:
	$(PY) scripts/build_tables.py
	$(PY) scripts/check_readme_numbers.py

all: phase9

clean:
	rm -rf .pytest_cache results/cache
