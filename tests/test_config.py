"""Tests for settings, strategy-weight loading and the error hierarchy."""

import json

import pytest

from common.config import Settings
from common.errors import (
    ConfigError,
    NotFoundError,
    RecsysError,
    ValidationError,
)
from engine.orchestrator import (
    ContentNotFoundError,
    InvalidFeedbackError,
    InvalidRequestError,
    UserNotFoundError,
)
from engine.scorer import STRATEGY_WEIGHTS, load_strategy_weights


def test_defaults_match_previous_hardcoded_values():
    s = Settings()
    assert (s.max_limit, s.default_limit, s.n_neighbors, s.n_popular) == (
        50,
        5,
        10,
        20,
    )
    assert s.diversity_penalty == 0.92 and s.proficiency_gain == 0.15
    assert s.cache_ttl == 300 and s.cache_max_size == 5000


def test_from_env_reads_and_normalises_values():
    s = Settings.from_env(
        {
            "PORT": "8080",
            "LOG_LEVEL": "debug",
            "LOG_FORMAT": "TEXT",
            "CACHE_TTL": "0",
            "MAX_LIMIT": "20",
            "API_KEY": " secret ",
            "EMPTY_IS_IGNORED": "",
        }
    )
    assert s.port == 8080 and s.log_level == "DEBUG"
    assert s.log_format == "text" and s.cache_ttl == 0.0
    assert s.max_limit == 20 and s.api_key == "secret"


def test_blank_variables_fall_back_to_defaults():
    assert Settings.from_env({"PORT": "  ", "CACHE_TTL": ""}) == Settings()


@pytest.mark.parametrize(
    "env",
    [
        {"PORT": "abc"},
        {"PORT": "0"},
        {"PORT": "70000"},
        {"CACHE_TTL": "-1"},
        {"CACHE_TTL": "nan"},
        {"LOG_LEVEL": "LOUD"},
        {"LOG_FORMAT": "xml"},
        {"MAX_LIMIT": "0"},
        {"DEFAULT_LIMIT": "999"},
        {"DIVERSITY_PENALTY": "0"},
        {"DIVERSITY_PENALTY": "1.5"},
        {"PROFICIENCY_GAIN": "2"},
        {"N_NEIGHBORS": "-3"},
        {"MAX_REQUEST_BYTES": "0"},
    ],
)
def test_invalid_environment_raises_config_error(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_error_message_names_the_variable():
    with pytest.raises(ConfigError, match="PORT"):
        Settings.from_env({"PORT": "abc"})


def test_secrets_are_masked():
    s = Settings(
        api_key="topsecret", database_url="postgresql://bob:pw123@db/x"
    )
    assert "topsecret" not in repr(s)
    shown = s.public_dict()
    assert shown["api_key"] == "***"
    assert "pw123" not in shown["database_url"]
    assert shown["database_url"] == "postgresql://bob:***@db/x"


def test_with_overrides_ignores_none_and_revalidates():
    s = Settings().with_overrides(cache_ttl=0, port=None)
    assert s.cache_ttl == 0 and s.port == 5000
    with pytest.raises(ConfigError):
        Settings().with_overrides(max_limit=0)
    with pytest.raises(ConfigError):
        Settings().with_overrides(nope=1)


def test_settings_are_immutable():
    with pytest.raises(Exception):
        Settings().port = 1


def test_strategy_weights_default_and_file_override(tmp_path):
    assert load_strategy_weights(None) == STRATEGY_WEIGHTS
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"hybrid": {"popularity": 1.0}}))
    loaded = load_strategy_weights(path)
    assert loaded["hybrid"] == {"popularity": 1.0}
    assert loaded["skill_based"] == STRATEGY_WEIGHTS["skill_based"]


def test_strategy_weights_file_errors(tmp_path):
    bad_files = {
        "list.json": "[1]",
        "broken.json": "{oops",
        "unknown_strategy.json": '{"magic": {"popularity": 1}}',
        "unknown_scorer.json": '{"hybrid": {"nope": 1}}',
        "negative.json": '{"hybrid": {"popularity": -1}}',
        "zero.json": '{"hybrid": {"popularity": 0}}',
        "not_object.json": '{"hybrid": 5}',
    }
    for name, body in bad_files.items():
        path = tmp_path / name
        path.write_text(body)
        with pytest.raises(ConfigError):
            load_strategy_weights(path)
    with pytest.raises(ConfigError):
        load_strategy_weights(tmp_path / "missing.json")


def test_error_hierarchy_keeps_builtin_bases():
    assert issubclass(UserNotFoundError, (NotFoundError, LookupError))
    assert issubclass(ContentNotFoundError, NotFoundError)
    assert issubclass(InvalidRequestError, (ValidationError, ValueError))
    assert issubclass(InvalidFeedbackError, ValidationError)
    assert issubclass(ConfigError, (RecsysError, ValueError))
