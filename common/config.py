"""Central, validated runtime configuration.

Every tunable lives in :class:`Settings` and is read from environment
variables (12-factor style), so behaviour can be changed without
touching code.  Unset or empty variables fall back to the defaults.

======================  =========  ======================================
Variable                Default    Meaning
======================  =========  ======================================
``DATABASE_URL``        sqlite     SQLAlchemy URL of the database
``API_KEY``             (empty)    Enables ``X-API-Key`` auth when set
``HOST`` / ``PORT``     127.0.0.1  Bind address / port of the dev server
                        / 5000
``LOG_LEVEL``           INFO       DEBUG, INFO, WARNING, ERROR, CRITICAL
``LOG_FORMAT``          json       ``json`` or ``text``
``CACHE_TTL``           300        Recommendation cache TTL in seconds
``CACHE_MAX_SIZE``      5000       Maximum cached result sets
``MAX_LIMIT``           50         Largest allowed ``limit`` parameter
``DEFAULT_LIMIT``       5          ``limit`` used when none is given
``N_NEIGHBORS``         10         Similar users used for collaborative
``N_POPULAR``           20         Extra popular items per request
``DIVERSITY_PENALTY``   0.92       Re-rank penalty per repeated category
``PROFICIENCY_GAIN``    0.15       Skill gain when content is completed
``MAX_REQUEST_BYTES``   16384      Largest accepted request body
``METRICS_WINDOW``      2000       Latency samples kept per endpoint
``STRATEGY_WEIGHTS_FILE``  (none)  JSON file overriding scorer weights
======================  =========  ======================================
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from common.errors import ConfigError

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_LOG_FORMATS = ("json", "text")
_URL_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@/\s]+@")


def _to_int(raw: str) -> int:
    return int(raw.strip())


def _to_float(raw: str) -> float:
    value = float(raw.strip())
    if not math.isfinite(value):
        raise ValueError("not finite")
    return value


def _to_str(raw: str) -> str:
    return raw.strip()


def _to_upper(raw: str) -> str:
    return raw.strip().upper()


def _to_lower(raw: str) -> str:
    return raw.strip().lower()


# field name -> (environment variable, parser, human description)
_ENV: dict[str, tuple[str, Callable[[str], Any], str]] = {
    "database_url": ("DATABASE_URL", _to_str, "a database URL"),
    "api_key": ("API_KEY", _to_str, "a string"),
    "host": ("HOST", _to_str, "a host name or address"),
    "port": ("PORT", _to_int, "an integer"),
    "log_level": ("LOG_LEVEL", _to_upper, "a log level name"),
    "log_format": ("LOG_FORMAT", _to_lower, "'json' or 'text'"),
    "cache_ttl": ("CACHE_TTL", _to_float, "a number of seconds"),
    "cache_max_size": ("CACHE_MAX_SIZE", _to_int, "an integer"),
    "max_limit": ("MAX_LIMIT", _to_int, "an integer"),
    "default_limit": ("DEFAULT_LIMIT", _to_int, "an integer"),
    "n_neighbors": ("N_NEIGHBORS", _to_int, "an integer"),
    "n_popular": ("N_POPULAR", _to_int, "an integer"),
    "diversity_penalty": ("DIVERSITY_PENALTY", _to_float, "a number"),
    "proficiency_gain": ("PROFICIENCY_GAIN", _to_float, "a number"),
    "max_request_bytes": ("MAX_REQUEST_BYTES", _to_int, "an integer"),
    "metrics_window": ("METRICS_WINDOW", _to_int, "an integer"),
    "strategy_weights_file": (
        "STRATEGY_WEIGHTS_FILE",
        _to_str,
        "a file path",
    ),
}


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings."""

    database_url: str = "sqlite:///recsys.db"
    api_key: str = field(default="", repr=False)
    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"
    log_format: str = "json"
    cache_ttl: float = 300.0
    cache_max_size: int = 5000
    max_limit: int = 50
    default_limit: int = 5
    n_neighbors: int = 10
    n_popular: int = 20
    diversity_penalty: float = 0.92
    proficiency_gain: float = 0.15
    max_request_bytes: int = 16384
    metrics_window: int = 2000
    strategy_weights_file: str | None = None

    def __post_init__(self) -> None:
        """Reject out-of-range values with a :class:`ConfigError`."""
        problems: list[str] = []

        def check(ok: bool, message: str) -> None:
            """Collect ``message`` when ``ok`` is false."""
            if not ok:
                problems.append(message)

        check(bool(self.database_url), "database_url must not be empty")
        check(1 <= self.port <= 65535, "port must be 1-65535")
        check(
            self.log_level in VALID_LOG_LEVELS,
            f"log_level must be one of {', '.join(VALID_LOG_LEVELS)}",
        )
        check(
            self.log_format in VALID_LOG_FORMATS,
            f"log_format must be one of {', '.join(VALID_LOG_FORMATS)}",
        )
        check(self.cache_ttl >= 0, "cache_ttl must be >= 0")
        check(self.cache_max_size >= 1, "cache_max_size must be >= 1")
        check(self.max_limit >= 1, "max_limit must be >= 1")
        check(
            1 <= self.default_limit <= self.max_limit,
            "default_limit must be between 1 and max_limit",
        )
        check(self.n_neighbors >= 0, "n_neighbors must be >= 0")
        check(self.n_popular >= 0, "n_popular must be >= 0")
        check(
            0 < self.diversity_penalty <= 1,
            "diversity_penalty must be in (0, 1]",
        )
        check(
            0 < self.proficiency_gain <= 1,
            "proficiency_gain must be in (0, 1]",
        )
        check(self.max_request_bytes >= 1, "max_request_bytes must be >= 1")
        check(self.metrics_window >= 1, "metrics_window must be >= 1")
        if problems:
            raise ConfigError("invalid settings: " + "; ".join(problems))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables.

        Args:
            env: Mapping to read instead of ``os.environ`` (for tests).

        Raises:
            ConfigError: If a variable cannot be parsed or is out of range.
        """
        source = os.environ if env is None else env
        values: dict[str, Any] = {}
        for name, (variable, parse, description) in _ENV.items():
            raw = source.get(variable)
            if raw is None or not raw.strip():
                continue
            try:
                values[name] = parse(raw)
            except ValueError as exc:
                raise ConfigError(
                    f"{variable}={raw!r} is invalid: expected {description}"
                ) from exc
        return cls(**values)

    def with_overrides(self, **changes: Any) -> Settings:
        """Return a copy with ``None`` values ignored (re-validated)."""
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise ConfigError(f"unknown settings: {sorted(unknown)}")
        return replace(
            self, **{k: v for k, v in changes.items() if v is not None}
        )

    def public_dict(self) -> dict[str, Any]:
        """Settings as a dict with secrets masked (safe to log)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["api_key"] = "***" if self.api_key else ""
        data["database_url"] = _URL_PASSWORD.sub(r"\1***@", self.database_url)
        return data
