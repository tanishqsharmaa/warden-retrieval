from warden_retrieval.pruner import prune_candidates
from warden_retrieval.search import RawCandidate


def make_candidate(idx: int, score: float) -> RawCandidate:
    return RawCandidate(
        doc_id=f"DOC-{idx}",
        chunk_index=idx,
        content=f"Content {idx}",
        source_url="file:///doc.md",
        role_tags=["Employee"],
        rrf_score=score,
    )

def test_prune_candidates_normal_dropoff():
    # max_score = 1.0; 0.40 threshold = 0.40
    candidates = [
        make_candidate(0, 1.0),
        make_candidate(1, 0.80),
        make_candidate(2, 0.45),
        make_candidate(3, 0.39),  # Should be dropped
        make_candidate(4, 0.10),  # Should be dropped
    ]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 3
    assert [c.doc_id for c in pruned] == ["DOC-0", "DOC-1", "DOC-2"]

def test_prune_candidates_caps_at_max():
    # 15 candidates all scoring >= 0.40
    candidates = [make_candidate(i, 0.90 - i * 0.01) for i in range(15)]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 10
    assert pruned[0].doc_id == "DOC-0"
    assert pruned[9].doc_id == "DOC-9"

def test_prune_all_below_threshold_preserves_top_candidate():
    # Review Focus 3: Candidate 0 has score 1.0; all others < 0.40
    candidates = [
        make_candidate(0, 1.0),
        make_candidate(1, 0.35),
        make_candidate(2, 0.20),
    ]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 1
    assert pruned[0].doc_id == "DOC-0"

def test_prune_empty_candidates_returns_empty():
    # Review Focus 5: Empty input returns empty list
    pruned = prune_candidates([], prune_ratio=0.40, max_candidates=10)
    assert pruned == []

def test_prune_single_candidate():
    candidates = [make_candidate(0, 0.50)]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 1
    assert pruned[0].doc_id == "DOC-0"

def test_prune_zero_or_negative_max_score():
    candidates = [make_candidate(0, 0.0), make_candidate(1, -0.1)]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 2

def test_prune_unsorted_candidates():
    candidates = [
        make_candidate(1, 0.45),
        make_candidate(0, 1.0),
        make_candidate(2, 0.80),
        make_candidate(3, 0.10),
    ]
    pruned = prune_candidates(candidates, prune_ratio=0.40, max_candidates=10)
    assert len(pruned) == 3
    assert [c.doc_id for c in pruned] == ["DOC-0", "DOC-2", "DOC-1"]

