from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from warden_shared.proto.v1.laya_pb2 import RankedCandidate, RerankResponse

from warden_retrieval.laya_client import RerankResult, SpeculativeLayaClient
from warden_retrieval.search import RawCandidate


def make_cand(idx: int, score: float) -> RawCandidate:
    return RawCandidate(
        doc_id=f"DOC-{idx}",
        chunk_index=idx,
        content=f"Policy chunk {idx}",
        source_url="file:///policy.md",
        role_tags=["Employee"],
        rrf_score=score,
    )

@pytest.mark.asyncio
async def test_laya_client_success():
    mock_stub = AsyncMock()
    mock_response = RerankResponse(
        query="leave",
        ranked_results=[
            RankedCandidate(candidate_id="DOC-1_1", probability_true=0.95, rank=1),
            RankedCandidate(candidate_id="DOC-0_0", probability_true=0.85, rank=2),
        ],
        inference_latency_ms=35.0,
    )
    mock_stub.Rerank = AsyncMock(return_value=mock_response)

    client = SpeculativeLayaClient(stub=mock_stub)
    candidates = [make_cand(0, 0.030), make_cand(1, 0.025)]
    result = await client.rerank(
        query="leave", candidates=candidates, final_limit=5, deadline_ms=150
    )

    assert isinstance(result, RerankResult)
    assert result.reranker_applied is True
    assert result.fallback_active is False
    assert len(result.passages) == 2
    # Verify passages reordered by calibrated_score (DOC-1 had 0.95 > DOC-0's 0.85)
    assert result.passages[0].doc_id == "DOC-1"
    assert result.passages[0].calibrated_score == 0.95
    assert result.passages[1].doc_id == "DOC-0"
    assert result.passages[1].calibrated_score == 0.85

@pytest.mark.asyncio
async def test_laya_client_timeout_triggers_fallback():
    # Review Focus 4: Deadline exceeded falls back to raw RRF
    mock_stub = AsyncMock()
    rpc_error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.DEADLINE_EXCEEDED,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Deadline exceeded in 150ms",
    )
    mock_stub.Rerank = AsyncMock(side_effect=rpc_error)

    client = SpeculativeLayaClient(stub=mock_stub)
    candidates = [make_cand(0, 0.030), make_cand(1, 0.025)]
    result = await client.rerank(
        query="leave", candidates=candidates, final_limit=5, deadline_ms=150
    )

    assert result.reranker_applied is False
    assert result.fallback_active is True
    assert len(result.passages) == 2
    # Preserves original RRF ordering
    assert result.passages[0].doc_id == "DOC-0"
    assert result.passages[0].rrf_score == 0.030
    assert result.passages[1].doc_id == "DOC-1"
    assert result.passages[1].rrf_score == 0.025

@pytest.mark.asyncio
async def test_laya_client_grpc_unavailable_triggers_fallback():
    mock_stub = AsyncMock()
    rpc_error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Laya service unreachable",
    )
    mock_stub.Rerank = AsyncMock(side_effect=rpc_error)

    client = SpeculativeLayaClient(stub=mock_stub)
    candidates = [make_cand(0, 0.040)]
    result = await client.rerank(
        query="leave", candidates=candidates, final_limit=5, deadline_ms=150
    )

    assert result.reranker_applied is False
    assert result.fallback_active is True
    assert len(result.passages) == 1
    assert result.passages[0].doc_id == "DOC-0"

@pytest.mark.asyncio
async def test_laya_client_empty_candidates():
    mock_stub = AsyncMock()
    client = SpeculativeLayaClient(stub=mock_stub)
    result = await client.rerank(query="leave", candidates=[], final_limit=5)

    assert result.reranker_applied is False
    assert result.fallback_active is False
    assert result.passages == []
    mock_stub.Rerank.assert_not_awaited()

@pytest.mark.asyncio
async def test_laya_client_circuit_breaker_trips_and_bypasses():
    mock_stub = AsyncMock()
    rpc_error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.DEADLINE_EXCEEDED,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Timeout",
    )
    mock_stub.Rerank = AsyncMock(side_effect=rpc_error)

    client = SpeculativeLayaClient(
        stub=mock_stub, failure_threshold=3, cooldown_seconds=5.0
    )
    candidates = [make_cand(0, 0.050)]

    # 3 failures should trip the circuit to OPEN
    for _ in range(3):
        res = await client.rerank("query", candidates)
        assert res.fallback_active is True

    assert client.is_circuit_open is True
    call_count = mock_stub.Rerank.await_count
    assert call_count == 3

    # 4th call should immediately bypass Laya without making a network call
    bypassed_res = await client.rerank("query", candidates)
    assert bypassed_res.fallback_active is True
    assert mock_stub.Rerank.await_count == call_count

@pytest.mark.asyncio
async def test_laya_client_empty_ranked_results_falls_back():
    mock_stub = AsyncMock()
    mock_response = RerankResponse(query="leave", ranked_results=[], inference_latency_ms=10.0)
    mock_stub.Rerank = AsyncMock(return_value=mock_response)

    client = SpeculativeLayaClient(stub=mock_stub)
    candidates = [make_cand(0, 0.040)]
    result = await client.rerank("leave", candidates)

    assert result.fallback_active is True
    assert len(result.passages) == 1
    assert result.passages[0].doc_id == "DOC-0"
