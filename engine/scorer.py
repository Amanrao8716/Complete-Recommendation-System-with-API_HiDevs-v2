"""Scoring: individual scorers, weighted combination, re-ranking."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from common.errors import ConfigError, ValidationError
from engine.domain import ContentInfo, UserContext
from engine.similarity import weighted_overlap

STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "hybrid": {
        "skill_match": 0.30,
        "category_affinity": 0.20,
        "difficulty_fit": 0.15,
        "collaborative": 0.25,
        "popularity": 0.10,
    },
    "skill_based": {
        "skill_match": 0.60,
        "difficulty_fit": 0.25,
        "category_affinity": 0.15,
    },
    "collaborative": {"collaborative": 0.80, "popularity": 0.20},
    "popularity": {"popularity": 1.0},
    "cold_start": {
        "category_affinity": 0.40,
        "skill_match": 0.30,
        "popularity": 0.20,
        "difficulty_fit": 0.10,
    },
}
DEFAULT_DIVERSITY_PENALTY = 0.92


@dataclass
class ScoringContext:
    """Per-request inputs shared by all scorers."""

    user: UserContext
    cf_scores: dict[int, float]
    max_popularity: float


@dataclass
class ScoredItem:
    """A content id with its final score and per-component scores."""

    content_id: int
    score: float
    components: dict[str, float] = field(default_factory=dict)


class Scorer(Protocol):
    """Anything with a ``name`` and a ``score(content, ctx)`` method."""

    name: str

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """Return a value in ``[0, 1]`` for ``content``."""


class SkillMatchScorer:
    """How much of the item's skills the user needs/cares about."""

    name = "skill_match"

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """Average need of the skills the item teaches."""
        return weighted_overlap(content.skill_ids, ctx.user.need)


class CategoryAffinityScorer:
    """Affinity for the item's category (interests + behaviour)."""

    name = "category_affinity"

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """The user's affinity for the item's category."""
        return ctx.user.category_affinity.get(content.category, 0.0)


class DifficultyFitScorer:
    """Prefers items slightly above the user's current level."""

    name = "difficulty_fit"

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """1 at the target difficulty, falling off linearly."""
        target = min(5.0, ctx.user.level + 0.5)
        return max(0.0, 1.0 - abs(content.difficulty - target) / 3.0)


class CollaborativeScorer:
    """Opinion of users with similar interaction history."""

    name = "collaborative"

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """Neighbour-weighted score for the item, capped at 1."""
        return min(1.0, ctx.cf_scores.get(content.id, 0.0))


class PopularityScorer:
    """Global popularity normalised to ``[0, 1]``."""

    name = "popularity"

    def score(self, content: ContentInfo, ctx: ScoringContext) -> float:
        """Popularity relative to the most popular item."""
        if ctx.max_popularity <= 0:
            return 0.0
        return max(0.0, min(1.0, content.popularity / ctx.max_popularity))


SCORERS: dict[str, Scorer] = {
    s.name: s
    for s in (
        SkillMatchScorer(),
        CategoryAffinityScorer(),
        DifficultyFitScorer(),
        CollaborativeScorer(),
        PopularityScorer(),
    )
}


class WeightedScorer:
    """Combine scorers with (normalised) weights.

    Raises:
        ValidationError: If a weight names an unknown scorer, is negative
            or non-numeric, or if the weights do not sum to a positive
            number.  (``ValidationError`` is a ``ValueError``.)
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        scorers: Mapping[str, Scorer] | None = None,
    ) -> None:
        scorers = scorers or SCORERS
        unknown = set(weights) - set(scorers)
        if unknown:
            raise ValidationError(f"unknown scorers: {sorted(unknown)}")
        for name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValidationError(
                    f"weight for {name!r} must be a finite number >= 0"
                )
        total = sum(weights.values())
        if total <= 0:
            raise ValidationError("weights must sum to a positive number")
        self.weights: dict[str, float] = {
            k: v / total for k, v in weights.items()
        }
        self._scorers = {k: scorers[k] for k in weights}

    def score(self, content: ContentInfo, ctx: ScoringContext) -> ScoredItem:
        """Return a ``ScoredItem`` for ``content``."""
        parts = {n: s.score(content, ctx) for n, s in self._scorers.items()}
        total = sum(self.weights[n] * v for n, v in parts.items())
        return ScoredItem(content.id, total, parts)


def load_strategy_weights(
    path: str | Path | None,
) -> dict[str, dict[str, float]]:
    """Return strategy weights, optionally overridden from a JSON file.

    The file maps strategy names to ``{scorer: weight}`` objects, e.g.
    ``{"hybrid": {"skill_match": 0.5, "popularity": 0.5}}``.  Strategies
    not mentioned keep their defaults.  Every entry is validated up front
    so a typo fails at start-up rather than on the first request.

    Raises:
        ConfigError: If the file is unreadable, malformed or invalid.
    """
    weights = {name: dict(w) for name, w in STRATEGY_WEIGHTS.items()}
    if not path:
        return weights
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read weights file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("weights file must contain a JSON object")
    for strategy, override in data.items():
        if strategy not in STRATEGY_WEIGHTS:
            raise ConfigError(f"unknown strategy in weights file: {strategy}")
        if not isinstance(override, dict):
            raise ConfigError(f"weights for {strategy!r} must be an object")
        try:
            WeightedScorer(override)
        except ValidationError as exc:
            raise ConfigError(f"weights for {strategy!r}: {exc}") from exc
        weights[strategy] = {k: float(v) for k, v in override.items()}
    return weights


def rank(
    scored: Iterable[ScoredItem], contents: Mapping[int, ContentInfo]
) -> list[ScoredItem]:
    """Sort by score, then popularity, then id (deterministic)."""
    return sorted(
        scored,
        key=lambda s: (
            -s.score,
            -contents[s.content_id].popularity,
            s.content_id,
        ),
    )


def diversify(
    ranked: list[ScoredItem],
    contents: Mapping[int, ContentInfo],
    limit: int,
    penalty: float = DEFAULT_DIVERSITY_PENALTY,
) -> list[ScoredItem]:
    """Greedy re-rank that gently penalises repeated categories.

    Raises:
        ValidationError: If ``penalty`` is not in ``(0, 1]``.
    """
    if not 0 < penalty <= 1:
        raise ValidationError(f"penalty must be in (0, 1]: {penalty!r}")
    pool = list(ranked[: max(limit * 5, limit)])
    picked: list[ScoredItem] = []
    counts: Counter[str] = Counter()
    while pool and len(picked) < limit:
        best = max(
            pool,
            key=lambda s: (
                s.score * penalty ** counts[contents[s.content_id].category],
                -s.content_id,
            ),
        )
        pool.remove(best)
        picked.append(best)
        counts[contents[best.content_id].category] += 1
    return picked


def _join(names: Iterable[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _skill_phrase(
    content: ContentInfo, ctx: ScoringContext, skill_names: Mapping[int, str]
) -> str:
    ranked = sorted(content.skill_ids, key=lambda s: -ctx.user.need.get(s, 0))
    matched = [skill_names[s] for s in ranked if ctx.user.need.get(s, 0) > 0]
    return f"builds skills you care about ({_join(matched[:3])})"


def _difficulty_phrase(
    content: ContentInfo, ctx: ScoringContext, _names: Mapping[int, str]
) -> str:
    if ctx.user.is_cold:
        return f"is beginner-friendly (difficulty {content.difficulty})"
    return f"is pitched at your level (difficulty {content.difficulty})"


Phrase = Callable[[ContentInfo, ScoringContext, Mapping[int, str]], str]
_PHRASES: dict[str, Phrase] = {
    "skill_match": _skill_phrase,
    "difficulty_fit": _difficulty_phrase,
    "category_affinity": lambda c, ctx, n: (
        f"fits your interest in {c.category}"
    ),
    "collaborative": lambda c, ctx, n: (
        "was well received by learners with similar activity"
    ),
    "popularity": lambda c, ctx, n: "is popular among learners",
}


def explain(
    content: ContentInfo,
    item: ScoredItem,
    weights: Mapping[str, float],
    ctx: ScoringContext,
    skill_names: Mapping[int, str],
    max_reasons: int = 2,
) -> str:
    """Build a short human-readable explanation from the top components."""
    contributions = sorted(
        (
            (weights.get(name, 0.0) * value, name)
            for name, value in item.components.items()
            if value > 0
        ),
        reverse=True,
    )
    phrases = [
        _PHRASES[name](content, ctx, skill_names)
        for contribution, name in contributions
        if contribution > 0.01
    ][:max_reasons]
    if not phrases:
        return "Recommended as a good starting point."
    return "Recommended because it " + " and ".join(phrases) + "."
