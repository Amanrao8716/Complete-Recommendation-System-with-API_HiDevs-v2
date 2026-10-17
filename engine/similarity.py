"""Similarity measures used by candidate generation and scoring."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import Any

from common.errors import ValidationError

Vector = Mapping[Hashable, float]
"""A sparse vector: ``{dimension: value}``."""


def _require_mapping(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"{name} must be a mapping, got {type(value).__name__}"
        )


def _as_set(name: str, values: Any) -> set[Any]:
    """Return ``values`` as a set, rejecting strings and non-iterables."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValidationError(
            f"{name} must be a collection, got {type(values).__name__}"
        )
    try:
        return set(values)
    except TypeError as exc:  # unhashable members
        raise ValidationError(f"{name} must contain hashable items") from exc


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two sparse ``{key: value}`` vectors.

    Returns ``0.0`` for empty or orthogonal vectors (never divides by
    zero) and clamps floating-point noise into ``[-1, 1]``.

    Raises:
        ValidationError: If an argument is not a mapping.
    """
    _require_mapping("a", a)
    _require_mapping("b", b)
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * large[k] for k, v in small.items() if k in large)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    result = dot / denominator
    if not math.isfinite(result):
        return 0.0
    return max(-1.0, min(1.0, result))


def jaccard_similarity(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Jaccard similarity between two collections treated as sets.

    Raises:
        ValidationError: If an argument is a string, not iterable, or
            holds unhashable items.
    """
    set_a, set_b = _as_set("a", a), _as_set("b", b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def weighted_overlap(
    items: Iterable[Hashable], weights: Mapping[Hashable, float]
) -> float:
    """Average weight of ``items`` looked up in ``weights`` (0 if absent).

    Raises:
        ValidationError: If ``items`` is not a collection or ``weights``
            is not a mapping.
    """
    _require_mapping("weights", weights)
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise ValidationError(
            f"items must be a collection, got {type(items).__name__}"
        )
    listed = list(items)
    if not listed:
        return 0.0
    return sum(weights.get(i, 0.0) for i in listed) / len(listed)


def top_k_similar(
    target: Vector,
    others: Mapping[Hashable, Vector],
    k: int,
    min_similarity: float = 0.0,
    exclude: Collection[Hashable] | None = None,
) -> list[tuple[Hashable, float]]:
    """Return the ``k`` most similar vectors as ``[(key, similarity)]``.

    Only similarities strictly above ``min_similarity`` are kept; ties are
    broken by key so results are deterministic.  ``k <= 0`` gives ``[]``.

    Raises:
        ValidationError: If ``k`` is not an integer or ``others`` is not
            a mapping.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError(f"k must be an integer, got {k!r}")
    _require_mapping("others", others)
    if k <= 0:
        return []
    skipped: Collection[Hashable] = exclude or ()
    scored: list[tuple[Hashable, float]] = []
    for key, vector in others.items():
        if key in skipped:
            continue
        sim = cosine_similarity(target, vector)
        if sim > min_similarity:
            scored.append((key, sim))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))  # type: ignore[type-var]
    return scored[:k]
