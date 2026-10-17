"""RecommendationOrchestrator: wires data, engine components and cache."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from common.config import Settings
from common.errors import ConfigError, NotFoundError, ValidationError
from data.database import session_scope
from data.repositories import (
    ContentRepository,
    ContentSkillRepository,
    InteractionRepository,
    SkillRepository,
    UserRepository,
    UserSkillRepository,
)
from engine.candidate_gen import CandidateGenerator
from engine.domain import (
    INTERACTION_TYPES,
    POSITIVE_THRESHOLD,
    ContentInfo,
    Recommendation,
    RecommendationResult,
    UserContext,
    UserInfo,
    interaction_strength,
)
from engine.scorer import (
    STRATEGY_WEIGHTS,
    ScoringContext,
    WeightedScorer,
    diversify,
    explain,
    load_strategy_weights,
    rank,
)

logger = logging.getLogger(__name__)

MAX_LIMIT = 50  # default; override with Settings.max_limit / MAX_LIMIT env
STRATEGIES = tuple(STRATEGY_WEIGHTS)
VALID_STRATEGIES = ("auto",) + STRATEGIES
PROFICIENCY_GAIN = 0.15  # default; override with PROFICIENCY_GAIN env

SessionFactory = Callable[[], Any]
Strengths = dict[int, dict[int, float]]
CacheKey = tuple[int, int, str]


class UserNotFoundError(NotFoundError):
    """Raised when a user id does not exist."""


class ContentNotFoundError(NotFoundError):
    """Raised when a content id does not exist."""


class InvalidFeedbackError(ValidationError):
    """Raised when feedback fails validation."""


class InvalidRequestError(ValidationError):
    """Raised when limit/strategy arguments are invalid."""


def _merge_strength(
    strengths: Strengths,
    user_id: int,
    content_id: int,
    type_: str,
    rating: float | None,
) -> None:
    """Store the interaction signal; views never overwrite stronger ones."""
    per_user = strengths.setdefault(user_id, {})
    if type_ == "view" and content_id in per_user:
        return
    per_user[content_id] = interaction_strength(type_, rating)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class RecommendationOrchestrator:
    """Serve recommendations from an in-memory snapshot of the database.

    Strategies: ``hybrid``, ``skill_based``, ``collaborative``,
    ``popularity`` and ``cold_start`` (``auto`` picks between hybrid
    and cold start).  Results are cached per (user, limit, strategy)
    and invalidated whenever the user submits feedback.

    Tunables come from, in order of precedence: the explicit keyword
    argument, the ``settings`` object (see :mod:`common.config`), then the
    built-in default.

    Args:
        session_factory: Zero-argument callable returning a DB session.
        strategy_weights: Optional per-strategy scorer weights; strategies
            not listed keep their defaults.
        cache_ttl: Cache lifetime in seconds (``0`` disables caching).
        cache_max_size: Maximum number of cached result sets.
        diversity_penalty: Re-rank penalty per repeated category (0-1].
        clock: Monotonic clock (injectable for tests).
        settings: Application settings; defaults to ``Settings()``.
        n_neighbors: Similar users used for collaborative scores.
        n_popular: Extra popular items proposed per request.
        max_limit: Largest accepted ``limit``.
        proficiency_gain: Skill gain when content is completed.

    Raises:
        ConfigError: If a tunable or the weights are invalid.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        strategy_weights: Mapping[str, Mapping[str, float]] | None = None,
        cache_ttl: float | None = None,
        cache_max_size: int | None = None,
        diversity_penalty: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        settings: Settings | None = None,
        n_neighbors: int | None = None,
        n_popular: int | None = None,
        max_limit: int | None = None,
        proficiency_gain: float | None = None,
    ) -> None:
        cfg = settings or Settings()
        # Re-validating through Settings reuses its range checks.
        self.settings = cfg.with_overrides(
            cache_ttl=cache_ttl,
            cache_max_size=cache_max_size,
            diversity_penalty=diversity_penalty,
            n_neighbors=n_neighbors,
            n_popular=n_popular,
            max_limit=max_limit,
            proficiency_gain=proficiency_gain,
        )
        self._session_factory = session_factory
        self._weights = self._resolve_weights(strategy_weights)
        self.cache_ttl = self.settings.cache_ttl
        self.cache_max_size = self.settings.cache_max_size
        self.diversity_penalty = self.settings.diversity_penalty
        self.n_neighbors = self.settings.n_neighbors
        self.n_popular = self.settings.n_popular
        self.max_limit = self.settings.max_limit
        self.default_limit = self.settings.default_limit
        self.proficiency_gain = self.settings.proficiency_gain
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[CacheKey, tuple[float, RecommendationResult]] = {}
        self._cache_index: defaultdict[int, set[CacheKey]] = defaultdict(set)
        self._hits = 0
        self._misses = 0
        self._n_interactions = 0
        self.refresh()

    def _resolve_weights(
        self, override: Mapping[str, Mapping[str, float]] | None
    ) -> dict[str, dict[str, float]]:
        """Defaults < weights file < explicit argument, all validated."""
        weights = load_strategy_weights(self.settings.strategy_weights_file)
        for strategy, values in (override or {}).items():
            if strategy not in STRATEGIES:
                raise ConfigError(f"unknown strategy: {strategy!r}")
            weights[strategy] = dict(values)
        for strategy, values in weights.items():
            try:
                WeightedScorer(values)
            except ValidationError as exc:
                raise ConfigError(f"weights for {strategy!r}: {exc}") from exc
        return weights

    @property
    def session_factory(self) -> SessionFactory:
        """The SQLAlchemy session factory in use."""
        return self._session_factory

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """(Re)load every table into the in-memory snapshot.

        The snapshot is built completely before it replaces the old one,
        so a database failure here leaves the previous state intact.
        """
        started = time.perf_counter()
        with session_scope(self._session_factory) as session:
            skills = {s.id: s.name for s in SkillRepository(session).get_all()}
            links: defaultdict[int, set[int]] = defaultdict(set)
            for content_id, skill_id in ContentSkillRepository(
                session
            ).get_all():
                links[content_id].add(skill_id)
            contents = {
                c.id: ContentInfo(
                    id=c.id,
                    title=c.title,
                    category=c.category,
                    difficulty=c.difficulty,
                    popularity=float(c.popularity or 0.0),
                    skill_ids=frozenset(links[c.id]),
                )
                for c in ContentRepository(session).get_all()
            }
            users = {
                u.id: UserInfo(u.id, u.name, tuple(u.interest_list()))
                for u in UserRepository(session).get_all()
            }
            user_skills: defaultdict[int, dict[int, float]] = defaultdict(dict)
            for row in UserSkillRepository(session).get_all():
                user_skills[row.user_id][row.skill_id] = row.proficiency
            interactions = InteractionRepository(session).get_all()
            raw = [
                (i.user_id, i.content_id, i.type, i.rating)
                for i in interactions
            ]

        strengths: Strengths = {}
        for user_id, content_id, type_, rating in raw:
            _merge_strength(strengths, user_id, content_id, type_, rating)

        generator = CandidateGenerator(
            contents, strengths, self.n_neighbors, self.n_popular
        )
        with self._lock:
            self._skills = skills
            self._skill_words = {
                sid: (name.lower(), set(name.lower().split()))
                for sid, name in skills.items()
            }
            self._contents = contents
            self._users = users
            self._user_skills = user_skills
            self._strengths = strengths
            self._n_interactions = len(raw)
            self._max_popularity = max(
                (c.popularity for c in contents.values()), default=0.0
            )
            self._generator = generator
            self._cache.clear()
            self._cache_index.clear()
        logger.info(
            "snapshot loaded",
            extra={
                "users": len(users),
                "content_items": len(contents),
                "interactions": len(raw),
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000, 1
                ),
            },
        )

    def _ensure_user(self, user_id: int) -> None:
        if user_id in self._users:
            return
        with session_scope(self._session_factory) as session:
            exists = UserRepository(session).exists(user_id)
        if not exists:
            raise UserNotFoundError(f"user {user_id} not found")
        logger.info("new user found in database", extra={"user_id": user_id})
        self.refresh()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def _interest_match(
        self, tokens: list[str], name_lower: str, words: set[str]
    ) -> bool:
        return any(t == name_lower or t in words for t in tokens)

    def _build_context(self, user_id: int) -> UserContext:
        """Derive skill needs, category affinity and level for one user."""
        user = self._users[user_id]
        strengths = dict(self._strengths.get(user_id, {}))
        proficiency = dict(self._user_skills.get(user_id, {}))
        tokens = list(user.interests)

        pos_skill: Counter[int] = Counter()
        neg_skill: Counter[int] = Counter()
        pos_cat: Counter[str] = Counter()
        neg_cat: Counter[str] = Counter()
        for content_id, strength in strengths.items():
            item = self._contents.get(content_id)
            if item is None or strength == 0:
                continue
            skill_bucket = pos_skill if strength > 0 else neg_skill
            cat_bucket = pos_cat if strength > 0 else neg_cat
            for skill_id in item.skill_ids:
                skill_bucket[skill_id] += abs(strength)
            cat_bucket[item.category] += abs(strength)

        need: dict[int, float] = {}
        for skill_id, (name_lower, words) in self._skill_words.items():
            interest = (
                1.0 if self._interest_match(tokens, name_lower, words) else 0.0
            )
            prof = proficiency.get(skill_id, 0.0)
            affinity = _clamp(
                0.6 * interest
                + 0.6 * min(1.0, pos_skill[skill_id] / 2)
                - 0.4 * min(1.0, neg_skill[skill_id] / 2)
                + 0.25 * prof
            )
            if affinity > 0:
                need[skill_id] = affinity * (1.0 - 0.5 * prof)

        category_affinity: dict[str, float] = {}
        for category in {c.category for c in self._contents.values()}:
            words = set(category.lower().split())
            interest = (
                1.0
                if self._interest_match(tokens, category.lower(), words)
                else 0.0
            )
            affinity = _clamp(
                0.6 * interest
                + 0.6 * min(1.0, pos_cat[category] / 2)
                - 0.4 * min(1.0, neg_cat[category] / 2)
            )
            if affinity > 0:
                category_affinity[category] = affinity

        levels: list[float] = []
        if proficiency:
            levels.append(
                1.0 + 4.0 * sum(proficiency.values()) / len(proficiency)
            )
        liked = [
            self._contents[c].difficulty
            for c, s in strengths.items()
            if s >= POSITIVE_THRESHOLD and c in self._contents
        ]
        if liked:
            levels.append(sum(liked) / len(liked))
        level = sum(levels) / len(levels) if levels else 1.0

        return UserContext(
            user_id=user_id,
            name=user.name,
            interests=tokens,
            proficiency=proficiency,
            strengths=strengths,
            need=need,
            category_affinity=category_affinity,
            level=level,
        )

    def _weights_for(
        self, strategy: str, ctx: UserContext
    ) -> dict[str, float]:
        weights = dict(self._weights[strategy])
        if strategy == "hybrid" and len(ctx.strengths) < 3:
            weights["collaborative"] = weights.get("collaborative", 0) * 0.5
        return weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_recommendations(
        self,
        user_id: int,
        limit: int | None = None,
        strategy: str = "auto",
    ) -> list[dict[str, Any]]:
        """Return a list of recommendation dictionaries."""
        result = self.get_recommendations_detailed(user_id, limit, strategy)
        return [item.to_dict() for item in result.items]

    def get_recommendations_detailed(
        self,
        user_id: int,
        limit: int | None = None,
        strategy: str = "auto",
        use_cache: bool = True,
    ) -> RecommendationResult:
        """Return a ``RecommendationResult`` (items, strategy, cached).

        Args:
            user_id: Existing user id.
            limit: Number of items (``1..max_limit``; default
                ``Settings.default_limit``).
            strategy: ``auto`` or one of :data:`STRATEGIES`.
            use_cache: Set ``False`` to bypass the cache.

        Raises:
            InvalidRequestError: For a bad ``user_id``, ``limit`` or
                ``strategy``.
            UserNotFoundError: If the user does not exist.
        """
        if limit is None:
            limit = self.default_limit
        if not _is_int(user_id):
            raise InvalidRequestError("user_id must be an integer")
        if not _is_int(limit):
            raise InvalidRequestError("limit must be an integer")
        if not 1 <= limit <= self.max_limit:
            raise InvalidRequestError(
                f"limit must be between 1 and {self.max_limit}"
            )
        if strategy not in VALID_STRATEGIES:
            raise InvalidRequestError(
                f"strategy must be one of {', '.join(VALID_STRATEGIES)}"
            )

        key = (user_id, limit, strategy)
        with self._lock:
            if use_cache:
                entry = self._cache.get(key)
                if entry and entry[0] > self._clock():
                    self._hits += 1
                    return replace(entry[1], cached=True)
            self._misses += 1

            self._ensure_user(user_id)
            ctx = self._build_context(user_id)
            resolved = strategy
            if strategy == "auto":
                resolved = "cold_start" if ctx.is_cold else "hybrid"

            candidates = self._generator.generate(
                ctx, resolved, min_candidates=limit
            )
            weights = self._weights_for(resolved, ctx)
            scorer = WeightedScorer(weights)
            scoring_ctx = ScoringContext(
                ctx, candidates.cf_scores, self._max_popularity
            )
            scored = [
                scorer.score(self._contents[cid], scoring_ctx)
                for cid in candidates.candidates
            ]
            top = diversify(
                rank(scored, self._contents),
                self._contents,
                limit,
                self.diversity_penalty,
            )
            items: list[Recommendation] = []
            for item in top:
                content = self._contents[item.content_id]
                items.append(
                    Recommendation(
                        content_id=content.id,
                        title=content.title,
                        category=content.category,
                        difficulty=content.difficulty,
                        score=round(item.score, 4),
                        strategy=resolved,
                        explanation=explain(
                            content,
                            item,
                            scorer.weights,
                            scoring_ctx,
                            self._skills,
                        ),
                        components={
                            k: round(v, 4) for k, v in item.components.items()
                        },
                    )
                )
            result = RecommendationResult(user_id, resolved, items, False)
            if use_cache and self.cache_ttl > 0:
                self._store(key, result)
            logger.debug(
                "recommendations computed",
                extra={
                    "user_id": user_id,
                    "strategy": resolved,
                    "limit": limit,
                    "returned": len(items),
                },
            )
            return result

    def _store(self, key: CacheKey, result: RecommendationResult) -> None:
        if len(self._cache) >= self.cache_max_size:
            logger.info(
                "cache full; clearing", extra={"size": len(self._cache)}
            )
            self._cache.clear()
            self._cache_index.clear()
        self._cache[key] = (self._clock() + self.cache_ttl, result)
        self._cache_index[key[0]].add(key)

    def _invalidate_user(self, user_id: int) -> None:
        for key in self._cache_index.pop(user_id, set()):
            self._cache.pop(key, None)

    def record_feedback(
        self,
        user_id: int,
        content_id: int,
        type_: str,
        rating: float | None = None,
    ) -> dict[str, Any]:
        """Persist feedback and update the in-memory model immediately.

        Completing a content item also raises the user's proficiency in
        the skills it teaches.

        Raises:
            InvalidFeedbackError: For a bad type, rating or id type.
            UserNotFoundError: If the user does not exist.
            ContentNotFoundError: If the content item does not exist.
        """
        if not _is_int(user_id) or not _is_int(content_id):
            raise InvalidFeedbackError(
                "user_id and content_id must be integers"
            )
        if type_ not in INTERACTION_TYPES:
            raise InvalidFeedbackError(
                f"type must be one of {', '.join(INTERACTION_TYPES)}"
            )
        if rating is not None:
            valid = isinstance(rating, (int, float)) and not isinstance(
                rating, bool
            )
            if not valid or not 1 <= rating <= 5:
                raise InvalidFeedbackError(
                    "rating must be a number from 1 to 5"
                )
        if type_ == "rate" and rating is None:
            raise InvalidFeedbackError(
                "rating is required when type is 'rate'"
            )

        with self._lock:
            self._ensure_user(user_id)
            content = self._contents.get(content_id)
            if content is None:
                raise ContentNotFoundError(f"content {content_id} not found")

            updates: dict[int, float] = {}
            current = self._user_skills.get(user_id, {})
            with session_scope(self._session_factory) as session:
                row = InteractionRepository(session).record(
                    user_id, content_id, type_, rating
                )
                if type_ == "complete":
                    skills_repo = UserSkillRepository(session)
                    for skill_id in content.skill_ids:
                        old = current.get(skill_id, 0.0)
                        gain = self.proficiency_gain * (1.0 - old)
                        new = round(old + gain, 4)
                        skills_repo.set_proficiency(user_id, skill_id, new)
                        updates[skill_id] = new
                record = {
                    "interaction_id": row.id,
                    "user_id": user_id,
                    "content_id": content_id,
                    "type": type_,
                    "rating": rating,
                    "strength": interaction_strength(type_, rating),
                    "created_at": row.created_at.isoformat(),
                }

            _merge_strength(
                self._strengths, user_id, content_id, type_, rating
            )
            if updates:
                self._user_skills.setdefault(user_id, {}).update(updates)
            self._n_interactions += 1
            self._invalidate_user(user_id)
            logger.info(
                "feedback recorded",
                extra={
                    "user_id": user_id,
                    "content_id": content_id,
                    "feedback_type": type_,
                },
            )
            return record

    def stats(self) -> dict[str, Any]:
        """Return cache and dataset statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "cache": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": round(self._hits / total, 4) if total else 0.0,
                    "size": len(self._cache),
                    "ttl_seconds": self.cache_ttl,
                },
                "users": len(self._users),
                "content_items": len(self._contents),
                "interactions": self._n_interactions,
            }
