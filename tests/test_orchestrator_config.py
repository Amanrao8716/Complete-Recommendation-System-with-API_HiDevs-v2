"""Orchestrator behaviour that is driven by settings and arguments."""

import json

import pytest

from common.config import Settings
from common.errors import ConfigError
from engine.orchestrator import (
    InvalidFeedbackError,
    InvalidRequestError,
    RecommendationOrchestrator,
)


def test_settings_flow_into_components(session_factory):
    cfg = Settings(n_neighbors=2, n_popular=3, max_limit=7, cache_ttl=0)
    orch = RecommendationOrchestrator(session_factory, settings=cfg)
    assert orch._generator.n_neighbors == 2
    assert orch._generator.n_popular == 3
    assert orch.max_limit == 7 and orch.cache_ttl == 0


def test_explicit_arguments_beat_settings(session_factory):
    cfg = Settings(cache_ttl=300, n_neighbors=2)
    orch = RecommendationOrchestrator(
        session_factory, settings=cfg, cache_ttl=0, n_neighbors=4
    )
    assert orch.cache_ttl == 0 and orch._generator.n_neighbors == 4


def test_zero_cache_ttl_is_honoured(session_factory):
    orch = RecommendationOrchestrator(session_factory, cache_ttl=0)
    orch.get_recommendations_detailed(1, 3)
    assert orch.get_recommendations_detailed(1, 3).cached is False


def test_invalid_tunables_raise_config_error(session_factory):
    bad_kwargs = [
        {"max_limit": 0},
        {"diversity_penalty": 0},
        {"diversity_penalty": 2},
        {"proficiency_gain": 0},
        {"cache_ttl": -1},
        {"cache_max_size": 0},
    ]
    for kwargs in bad_kwargs:
        with pytest.raises(ConfigError):
            RecommendationOrchestrator(session_factory, **kwargs)


def test_default_limit_is_used_when_none_given(session_factory):
    cfg = Settings(default_limit=2)
    orch = RecommendationOrchestrator(session_factory, settings=cfg)
    assert len(orch.get_recommendations(1)) == 2


def test_strategy_weights_override_and_validation(session_factory):
    orch = RecommendationOrchestrator(
        session_factory, strategy_weights={"popularity": {"popularity": 1.0}}
    )
    assert orch._weights["popularity"] == {"popularity": 1.0}
    assert "hybrid" in orch._weights  # others keep their defaults
    for bad in (
        {"magic": {"popularity": 1.0}},
        {"hybrid": {"nope": 1.0}},
        {"hybrid": {"popularity": 0.0}},
    ):
        with pytest.raises(ConfigError):
            RecommendationOrchestrator(session_factory, strategy_weights=bad)


def test_weights_file_changes_ranking(session_factory, tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"popularity": {"popularity": 1.0}}))
    orch = RecommendationOrchestrator(
        session_factory, settings=Settings(strategy_weights_file=str(path))
    )
    recs = orch.get_recommendations(14, 3, "popularity")
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_proficiency_gain_is_configurable(session_factory):
    orch = RecommendationOrchestrator(session_factory, proficiency_gain=0.5)
    ids = orch._contents[19].skill_ids  # Linux Command Line
    orch.record_feedback(14, 19, "complete")
    assert orch._user_skills[14][next(iter(ids))] == pytest.approx(0.5)


def test_non_integer_ids_are_rejected(orchestrator):
    for bad in ("1", None, 1.5, True):
        with pytest.raises(InvalidRequestError):
            orchestrator.get_recommendations(bad)
        with pytest.raises(InvalidFeedbackError):
            orchestrator.record_feedback(bad, 1, "like")
        with pytest.raises(InvalidFeedbackError):
            orchestrator.record_feedback(1, bad, "like")


def test_failed_refresh_keeps_previous_snapshot(session_factory):
    orch = RecommendationOrchestrator(session_factory, cache_ttl=0)
    before = orch.stats()["users"]

    def broken():
        raise RuntimeError("db down")

    orch._session_factory = broken
    with pytest.raises(RuntimeError):
        orch.refresh()
    assert orch.stats()["users"] == before
    assert orch.get_recommendations(1, 2)  # still serves from memory
