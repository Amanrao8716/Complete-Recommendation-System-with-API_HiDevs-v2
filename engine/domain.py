"""Plain data structures and constants shared by the engine modules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from common.errors import ValidationError

INTERACTION_TYPES = ("view", "like", "complete", "rate", "dislike")
POSITIVE_THRESHOLD = 0.5
_TYPE_STRENGTH = {"view": 0.2, "like": 0.7, "complete": 1.0, "dislike": -1.0}


def interaction_strength(type_: str, rating: float | None = None) -> float:
    """Convert an interaction into an implicit signal in ``[-1, 1]``.

    A rating of 1 maps to -1, 3 to 0 and 5 to +1.  For non-``rate``
    interactions the rating is averaged with the type's base strength.

    Raises:
        ValidationError: If ``rating`` is not a finite number.
    """
    base = _TYPE_STRENGTH.get(type_, 0.0)
    if rating is None:
        return base
    if (
        isinstance(rating, bool)
        or not isinstance(rating, (int, float))
        or not math.isfinite(rating)
    ):
        raise ValidationError(f"rating must be a finite number: {rating!r}")
    from_rating = max(-1.0, min(1.0, (float(rating) - 3.0) / 2.0))
    if type_ == "rate":
        return from_rating
    return (base + from_rating) / 2.0


@dataclass(frozen=True)
class ContentInfo:
    """Immutable view of a content item with its skills."""

    id: int
    title: str
    category: str
    difficulty: int
    popularity: float
    skill_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class UserInfo:
    """Immutable view of a user."""

    id: int
    name: str
    interests: tuple[str, ...] = ()


@dataclass
class UserContext:
    """Everything the engine knows about one user at request time."""

    user_id: int
    name: str
    interests: list[str]
    proficiency: dict[int, float]
    strengths: dict[int, float]
    need: dict[int, float] = field(default_factory=dict)
    category_affinity: dict[str, float] = field(default_factory=dict)
    level: float = 1.0

    @property
    def is_cold(self) -> bool:
        """True for users with neither history nor skill data."""
        return not self.strengths and not self.proficiency


@dataclass
class Recommendation:
    """One recommended item with its explanation."""

    content_id: int
    title: str
    category: str
    difficulty: int
    score: float
    strategy: str
    explanation: str
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


@dataclass
class RecommendationResult:
    """The outcome of one ``get_recommendations`` call."""

    user_id: int
    strategy: str
    items: list[Recommendation]
    cached: bool = False
