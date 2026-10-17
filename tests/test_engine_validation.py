"""Input-validation and edge-case tests for the engine building blocks."""

import math

import pytest

from common.errors import ValidationError
from engine import evaluator
from engine.candidate_gen import CandidateGenerator
from engine.domain import ContentInfo, UserContext, interaction_strength
from engine.scorer import ScoredItem, WeightedScorer, diversify
from engine.similarity import (
    cosine_similarity,
    jaccard_similarity,
    top_k_similar,
    weighted_overlap,
)


def _content(i, category="A", popularity=1.0):
    return ContentInfo(i, f"c{i}", category, 2, popularity, frozenset({i}))


def _ctx(**kw):
    base = dict(
        user_id=1, name="t", interests=[], proficiency={}, strengths={}
    )
    base.update(kw)
    return UserContext(**base)


# ---------------------------------------------------------------- similarity
def test_cosine_rejects_non_mappings_and_handles_degenerate_input():
    for bad in ([1, 2], "ab", None, 5):
        with pytest.raises(ValidationError):
            cosine_similarity(bad, {"a": 1})
    assert cosine_similarity({"a": 0.0}, {"a": 1.0}) == 0.0
    assert cosine_similarity({"a": 1}, {"a": -1}) == pytest.approx(-1.0)
    assert -1.0 <= cosine_similarity({"a": 0.1}, {"a": 0.1}) <= 1.0
    assert cosine_similarity({"a": float("inf")}, {"a": 1.0}) == 0.0


def test_jaccard_and_overlap_reject_bad_types():
    for bad in ("abc", None, 5, b"x"):
        with pytest.raises(ValidationError):
            jaccard_similarity(bad, [1])
        with pytest.raises(ValidationError):
            weighted_overlap(bad, {1: 1.0})
    with pytest.raises(ValidationError):
        jaccard_similarity([[1]], [[1]])  # unhashable members
    with pytest.raises(ValidationError):
        weighted_overlap([1], [1.0])  # weights must be a mapping
    assert jaccard_similarity({1, 2}, (2, 3)) == pytest.approx(1 / 3)


def test_top_k_similar_validation_and_limits():
    others = {1: {"a": 1.0}, 2: {"a": 1.0, "b": 1.0}}
    assert top_k_similar({"a": 1.0}, others, 0) == []
    assert top_k_similar({"a": 1.0}, others, -2) == []
    assert [k for k, _ in top_k_similar({"a": 1.0}, others, 1)] == [1]
    for bad_k in ("3", None, 2.5, True):
        with pytest.raises(ValidationError):
            top_k_similar({"a": 1.0}, others, bad_k)
    with pytest.raises(ValidationError):
        top_k_similar({"a": 1.0}, [1, 2], 1)


# -------------------------------------------------------------------- domain
def test_interaction_strength_rejects_bad_ratings():
    for bad in ("5", True, float("nan"), float("inf"), [4]):
        with pytest.raises(ValidationError):
            interaction_strength("rate", bad)
    assert interaction_strength("view") == 0.2
    assert interaction_strength("unknown") == 0.0


# ----------------------------------------------------------------- evaluator
def test_metrics_validate_k_and_inputs():
    for bad_k in ("5", None, 2.0, True):
        for metric in (
            evaluator.precision_at_k,
            evaluator.recall_at_k,
            evaluator.ndcg_at_k,
        ):
            with pytest.raises(ValidationError):
                metric([1], {1}, bad_k)
    with pytest.raises(ValidationError):
        evaluator.precision_at_k("abc", {"a"}, 3)
    with pytest.raises(ValidationError):
        evaluator.evaluate_rankings([], {}, 5)


def test_negative_k_returns_zero_like_zero_k():
    for metric in (
        evaluator.precision_at_k,
        evaluator.recall_at_k,
        evaluator.ndcg_at_k,
    ):
        assert metric([1, 2], {1}, -3) == 0.0


def test_ndcg_never_exceeds_one_with_repeated_items():
    assert evaluator.ndcg_at_k([1, 1, 1, 1], {1}, 4) == pytest.approx(1.0)
    assert evaluator.precision_at_k([1, 1, 1], {1}, 3) == pytest.approx(1 / 3)


def test_metrics_accept_tuples_and_missing_users():
    assert evaluator.recall_at_k((1, 2, 3), {2}, 3) == 1.0
    result = evaluator.evaluate_rankings({}, {7: {1}}, 3)
    assert result["users"] == 1 and result["precision"] == 0.0
    assert not math.isnan(result["ndcg"])


# -------------------------------------------------------------------- scorer
def test_weighted_scorer_rejects_bad_weights():
    for weights in (
        {"popularity": -1.0},
        {"popularity": float("nan")},
        {"popularity": True},
        {"popularity": "1"},
        {},
    ):
        with pytest.raises(ValueError):
            WeightedScorer(weights)


def test_diversify_validates_penalty_and_handles_zero_limit():
    contents = {1: _content(1)}
    ranked = [ScoredItem(1, 0.5)]
    for penalty in (0, -1, 1.5):
        with pytest.raises(ValidationError):
            diversify(ranked, contents, 1, penalty)
    assert diversify(ranked, contents, 0) == []
    assert diversify([], contents, 3) == []


# ----------------------------------------------------------------- generator
def test_candidate_generator_validates_arguments():
    contents = {1: _content(1)}
    for bad in (-1, "5", None, True, 1.5):
        with pytest.raises(ValidationError):
            CandidateGenerator(contents, {}, n_neighbors=bad)
        with pytest.raises(ValidationError):
            CandidateGenerator(contents, {}, n_popular=bad)
    gen = CandidateGenerator(contents, {})
    with pytest.raises(ValidationError):
        gen.generate(_ctx(), "magic")
    with pytest.raises(ValidationError):
        gen.generate(_ctx(), "hybrid", min_candidates=-1)


def test_candidate_generator_respects_configured_sizes():
    contents = {i: _content(i, popularity=100 - i) for i in range(1, 9)}
    gen = CandidateGenerator(contents, {}, n_neighbors=0, n_popular=3)
    result = gen.generate(_ctx(), "cold_start")
    assert set(result.candidates) == {1, 2, 3}
    assert gen.find_neighbors(_ctx(strengths={1: 1.0})) == []
