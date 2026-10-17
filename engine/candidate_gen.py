"""Candidate generation: decide which items are worth scoring."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from common.errors import ValidationError
from engine.domain import ContentInfo, UserContext
from engine.similarity import top_k_similar

logger = logging.getLogger(__name__)

DEFAULT_N_NEIGHBORS = 10
DEFAULT_N_POPULAR = 20

STRATEGY_SOURCES = {
    "hybrid": frozenset({"skill", "category", "collaborative", "popular"}),
    "skill_based": frozenset({"skill", "category"}),
    "collaborative": frozenset({"collaborative", "popular"}),
    "popularity": frozenset({"all"}),
    "cold_start": frozenset({"skill", "category", "popular"}),
}


def _require_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be an integer >= 0: {value!r}")
    return value


@dataclass
class CandidateSet:
    """Candidate ids with the sources that proposed them."""

    candidates: dict[int, set[str]] = field(default_factory=dict)
    cf_scores: dict[int, float] = field(default_factory=dict)
    neighbors: list[tuple[int, float]] = field(default_factory=list)


class CandidateGenerator:
    """Generate candidates from skills, categories, neighbours, popularity.

    ``contents`` and ``strengths`` are shared (live) references owned by
    the orchestrator, so feedback is visible without rebuilding.

    Args:
        contents: ``{content_id: ContentInfo}`` catalogue.
        strengths: ``{user_id: {content_id: signal}}`` interaction matrix.
        n_neighbors: How many similar users feed collaborative scores.
        n_popular: Extra popular items proposed besides the user's own.

    Raises:
        ValidationError: If the counts are not non-negative integers.
    """

    def __init__(
        self,
        contents: Mapping[int, ContentInfo],
        strengths: dict[int, dict[int, float]],
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        n_popular: int = DEFAULT_N_POPULAR,
    ) -> None:
        self.contents = contents
        self.strengths = strengths
        self.n_neighbors = _require_count("n_neighbors", n_neighbors)
        self.n_popular = _require_count("n_popular", n_popular)
        self._by_skill: defaultdict[int, list[int]] = defaultdict(list)
        self._by_category: defaultdict[str, list[int]] = defaultdict(list)
        for item in contents.values():
            for skill_id in item.skill_ids:
                self._by_skill[skill_id].append(item.id)
            self._by_category[item.category].append(item.id)
        ordered = sorted(
            contents.values(), key=lambda c: (-c.popularity, c.id)
        )
        self._popular = [c.id for c in ordered]

    def find_neighbors(self, ctx: UserContext) -> list[tuple[int, float]]:
        """Return users whose interaction vectors resemble ``ctx``'s."""
        if not ctx.strengths:
            return []
        return top_k_similar(  # type: ignore[return-value]
            ctx.strengths,
            self.strengths,  # type: ignore[arg-type]
            self.n_neighbors,
            exclude={ctx.user_id},
        )

    def collaborative_scores(
        self, ctx: UserContext, neighbors: list[tuple[int, float]]
    ) -> dict[int, float]:
        """Similarity-weighted opinion of neighbours, in ``[0, 1]``."""
        total = sum(sim for _, sim in neighbors)
        if total <= 0:
            return {}
        raw: defaultdict[int, float] = defaultdict(float)
        for user_id, sim in neighbors:
            for content_id, strength in self.strengths[user_id].items():
                if content_id not in ctx.strengths:
                    raw[content_id] += sim * strength
        return {cid: v / total for cid, v in raw.items() if v > 0}

    def generate(
        self, ctx: UserContext, strategy: str, min_candidates: int = 0
    ) -> CandidateSet:
        """Return unseen candidate items for ``strategy``.

        Raises:
            ValidationError: For an unknown strategy or bad
                ``min_candidates``.
        """
        if strategy not in STRATEGY_SOURCES:
            raise ValidationError(f"unknown strategy: {strategy!r}")
        _require_count("min_candidates", min_candidates)
        sources = STRATEGY_SOURCES[strategy]
        found: defaultdict[int, set[str]] = defaultdict(set)

        if "skill" in sources:
            for skill_id, need in ctx.need.items():
                if need > 0:
                    for cid in self._by_skill.get(skill_id, ()):
                        found[cid].add("skill")
        if "category" in sources:
            for category, aff in ctx.category_affinity.items():
                if aff > 0:
                    for cid in self._by_category.get(category, ()):
                        found[cid].add("category")

        neighbors: list[tuple[int, float]] = []
        cf_scores: dict[int, float] = {}
        if "collaborative" in sources:
            neighbors = self.find_neighbors(ctx)
            cf_scores = self.collaborative_scores(ctx, neighbors)
            for cid in cf_scores:
                found[cid].add("collaborative")

        if "all" in sources:
            for cid in self._popular:
                found[cid].add("popular")
        elif "popular" in sources:
            for cid in self._popular[: self.n_popular + len(ctx.strengths)]:
                found[cid].add("popular")

        candidates = {c: s for c, s in found.items() if c not in ctx.strengths}
        if len(candidates) < min_candidates:
            for cid in self._popular:
                if cid not in ctx.strengths and cid not in candidates:
                    candidates[cid] = {"fallback"}
                if len(candidates) >= min_candidates:
                    break
        logger.debug(
            "candidates generated",
            extra={
                "user_id": ctx.user_id,
                "strategy": strategy,
                "candidates": len(candidates),
                "neighbors": len(neighbors),
            },
        )
        return CandidateSet(candidates, cf_scores, neighbors)
