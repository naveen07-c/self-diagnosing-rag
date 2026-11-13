"""Phase 5 failure attribution tests (no GPU; mocks the decision outcomes)."""
from __future__ import annotations

import pytest

from src.diagnose.attribute import (
    CATEGORY_ORDER,
    attribute,
    is_failure,
)


def _fail(em=0.0, f1=0.0, unfaithful=False, followed=1.0, dataset="normal"):
    return {
        "em": em,
        "f1": f1,
        "nli": {"unfaithful": 1.0 if unfaithful else 0.0},
        "cf": {"followed_context": followed},
    }


def test_quant_takes_precedence_over_retrieval():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "uniform",
                    fp_passed=True, chunking_broken=True,
                    retrieval_broken=True, gold_context_passed=False)
    assert cat == "QUANT"


def test_quant_requires_fp_pass():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "uniform",
                    fp_passed=False, chunking_broken=False,
                    retrieval_broken=True, gold_context_passed=False)
    assert cat == "RETRIEVAL"


def test_chunking_before_retrieval():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "fp",
                    fp_passed=True, chunking_broken=True,
                    retrieval_broken=True, gold_context_passed=False)
    assert cat == "CHUNKING"


def test_retrieval_then_context_noise():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "fp",
                    fp_passed=True, chunking_broken=False,
                    retrieval_broken=False, gold_context_passed=True)
    assert cat == "CONTEXT_NOISE"


def test_generation_last_resort():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "fp",
                    fp_passed=True, chunking_broken=False,
                    retrieval_broken=False, gold_context_passed=False)
    assert cat == "GENERATION"


def test_fp_skips_quant_step():
    rec = _fail(unfaithful=True)
    cat = attribute(rec, "normal", "fp",
                    fp_passed=False, chunking_broken=False,
                    retrieval_broken=False, gold_context_passed=False)
    assert cat == "GENERATION"


def test_counterfactual_followed_context_is_failure():
    rec = _fail(em=1.0, f1=1.0, unfaithful=False, followed=0.0,
                dataset="counterfactual")
    assert is_failure(rec, "counterfactual") is True
    assert is_failure(rec, "normal") is False


def test_is_failure_em_f1_rule():
    assert is_failure({"em": 0.0, "f1": 0.1, "nli": {"unfaithful": 0.0},
                       "cf": {"followed_context": 1.0}}, "normal") is True
    assert is_failure({"em": 0.0, "f1": 0.6, "nli": {"unfaithful": 0.0},
                       "cf": {"followed_context": 1.0}}, "normal") is False
    assert is_failure({"em": 0.0, "f1": 0.0, "nli": {"unfaithful": 1.0},
                       "cf": {"followed_context": 1.0}}, "normal") is True


def test_categories_sum_and_cover_order():
    cases = [
        ("QUANT", dict(fp_passed=True)),
        ("CHUNKING", dict(chunking_broken=True)),
        ("RETRIEVAL", dict(retrieval_broken=True)),
        ("CONTEXT_NOISE", dict(gold_context_passed=True)),
        ("GENERATION", dict()),
    ]
    for expected, kwargs in cases:
        rec = _fail(unfaithful=True)
        cat = attribute(rec, "normal", "uniform", **kwargs)
        assert cat == expected, (kwargs, cat)
    assert list(CATEGORY_ORDER) == ["QUANT", "CHUNKING", "RETRIEVAL",
                                    "CONTEXT_NOISE", "GENERATION"]
    assert set(CATEGORY_ORDER) == {"QUANT", "CHUNKING", "RETRIEVAL",
                                   "CONTEXT_NOISE", "GENERATION"}
