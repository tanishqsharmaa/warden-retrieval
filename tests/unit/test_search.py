from unittest.mock import AsyncMock, MagicMock

import pytest
from warden_shared.errors import QdrantUnavailableError

from warden_retrieval.search import HybridSearchEngine, RawCandidate


@pytest.mark.asyncio
async def test_search_injects_early_binding_acl_filter():
    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock()

    mock_response = MagicMock()
    mock_point = MagicMock()
    mock_point.score = 0.0328
    mock_point.payload = {
        "doc_id": "DOC-HR-LEAVE-2026",
        "chunk_index": 2,
        "content": (
            "Bereavement Leave: Employees are eligible for up to 5 consecutive paid days off..."
        ),
        "source_url": "file:///data/policies/pto_policy_2026.md",
        "role_tags": ["Employee", "Manager"],
    }
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    engine = HybridSearchEngine(client=mock_client, collection_name="warden_hr_policies")
    results = await engine.search(
        query_text="How many days of bereavement leave am I entitled to?",
        caller_role="Employee",
        top_k=30,
    )

    assert len(results) == 1
    cand = results[0]
    assert isinstance(cand, RawCandidate)
    assert cand.doc_id == "DOC-HR-LEAVE-2026"
    assert cand.chunk_index == 2
    assert cand.content.startswith("Bereavement Leave")
    assert cand.source_url == "file:///data/policies/pto_policy_2026.md"
    assert cand.role_tags == ["Employee", "Manager"]
    assert cand.rrf_score == 0.0328

    # Assert query_points was called with correct parameters
    mock_client.query_points.assert_awaited_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "warden_hr_policies"
    assert call_kwargs["limit"] == 30

    # Verify early-binding payload filter
    q_filter = call_kwargs["query_filter"]
    assert q_filter.must[0].key == "role_tags"
    assert q_filter.must[0].match.any == ["Employee"]

    # Verify SearchParams tuning
    s_params = call_kwargs["search_params"]
    assert s_params.hnsw_ef == 64
    assert s_params.exact is False
    assert s_params.quantization.rescore is True
    assert s_params.quantization.oversampling == 2.0

@pytest.mark.asyncio
async def test_search_zero_candidates_returns_empty():
    # Review Focus 5: empty candidates handled cleanly
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    engine = HybridSearchEngine(client=mock_client, collection_name="warden_hr_policies")
    results = await engine.search(query_text="nonexistent policy", caller_role="Employee", top_k=30)
    assert results == []

@pytest.mark.asyncio
async def test_search_qdrant_error_propagates():
    # Review Focus 2: Qdrant unreachable raises QdrantUnavailableError
    mock_client = AsyncMock()
    mock_client.query_points.side_effect = Exception("Qdrant connection dropped")

    engine = HybridSearchEngine(client=mock_client, collection_name="warden_hr_policies")
    with pytest.raises(QdrantUnavailableError):
        await engine.search(query_text="vacation", caller_role="Employee")
