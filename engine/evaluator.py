"""Offline ranking metrics: precision@k, recall@k, NDCG@k."""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Mapping, Sequence
from math import log2
from typing import Any

from common.errors import ValidationError

logger = logging.getLogger(__name__)

Ranking = Sequence[Hashable]
Metrics = dict[str, float]


def _check_k(k: Any) -> int:
    """Return ``k`` if it is an integer (``bool`` is rejected)."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError(f"k must be an integer, got {k!r}")
    return k


def _top_k(recommended: Ranking, k: int) -> list[Hashable]:
    """First ``k`` distinct items of a ranking, order preserved."""
    if isinstance(recommended, (str, bytes)):
        raise ValidationError("recommended must be a list of ids")
    return list(dict.fromkeys(recommended))[:k]


def precision_at_k(
    recommended: Ranking, relevant: Collection[Hashable], k: int
) -> float:
    """Fraction of the top-k that is relevant (``0.0`` when ``k <= 0``).

    Raises:
        ValidationError: If ``k`` is not an integer.
    """
    if _check_k(k) <= 0:
        return 0.0
    hits = sum(1 for item in _top_k(recommended, k) if item in relevant)
    return hits / k


def recall_at_k(
    recommended: Ranking, relevant: Collection[Hashable], k: int
) -> float:
    """Fraction of relevant items found in the top-k.

    Raises:
        ValidationError: If ``k`` is not an integer.
    """
    _check_k(k)
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item in _top_k(recommended, k) if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: Ranking, relevant: Collection[Hashable], k: int
) -> float:
    """Binary-relevance normalised discounted cumulative gain.

    Repeated items are counted once, so the score can never exceed 1.

    Raises:
        ValidationError: If ``k`` is not an integer.
    """
    _check_k(k)
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 2)
        for rank, item in enumerate(_top_k(recommended, k))
        if item in relevant
    )
    ideal = sum(1.0 / log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def evaluate_rankings(
    recommended: Mapping[Hashable, Ranking],
    relevant: Mapping[Hashable, Collection[Hashable]],
    k: int = 5,
) -> dict[str, Any]:
    """Average the metrics over users that have held-out relevant items.

    ``recommended`` maps user -> ranked list, ``relevant`` maps user ->
    set of held-out relevant items.  Users with no relevant items are
    skipped; the result also reports how many users were evaluated.

    Raises:
        ValidationError: If an argument has the wrong type.
    """
    _check_k(k)
    if not isinstance(recommended, Mapping) or not isinstance(
        relevant, Mapping
    ):
        raise ValidationError("recommended and relevant must be mappings")
    users = [u for u in relevant if relevant[u]]
    if not users:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "users": 0}
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    for user in users:
        ranked = recommended.get(user, [])
        totals["precision"] += precision_at_k(ranked, relevant[user], k)
        totals["recall"] += recall_at_k(ranked, relevant[user], k)
        totals["ndcg"] += ndcg_at_k(ranked, relevant[user], k)
    result: dict[str, Any] = {n: v / len(users) for n, v in totals.items()}
    result["users"] = len(users)
    logger.debug("evaluated rankings", extra={"users": len(users), "k": k})
    return result


def catalog_coverage(
    recommended: Mapping[Hashable, Ranking], catalog_size: int
) -> float:
    """Share of the catalog that appears in at least one list."""
    if catalog_size <= 0:
        return 0.0
    seen = {item for ranked in recommended.values() for item in ranked}
    return len(seen) / catalog_size
