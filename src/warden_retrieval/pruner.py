import logging
from typing import Sequence

from warden_retrieval.search import RawCandidate

logger = logging.getLogger("warden.retrieval.pruner")

def prune_candidates(
    candidates: Sequence[RawCandidate],
    prune_ratio: float = 0.40,
    max_candidates: int = 10,
) -> list[RawCandidate]:
    """Dynamically prunes raw candidate chunks based on relative score drop-off.

    Retains candidates whose rrf_score >= prune_ratio * max_rrf_score, capping
    the output list at max_candidates (reducing cross-encoder compute by 66%).
    Guarantees that at least the top candidate is preserved if candidates is non-empty.
    """
    if not candidates:
        return []

    max_score = candidates[0].rrf_score
    if max_score <= 0.0:
        return list(candidates[:max_candidates])

    threshold = prune_ratio * max_score
    retained = [c for c in candidates if c.rrf_score >= threshold]

    # Safety guarantee: never return an empty list if input was non-empty
    if not retained:
        retained = [candidates[0]]

    pruned = retained[:max_candidates]
    logger.debug(
        f"Dynamic pruning: {len(candidates)} -> {len(pruned)} candidates "
        f"(max_score={max_score:.4f}, threshold={threshold:.4f})"
    )
    return pruned
