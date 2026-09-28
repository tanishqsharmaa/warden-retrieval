from unittest.mock import AsyncMock

import grpc
import pytest
from warden_shared.proto.v1.retrieval_pb2 import (
    IndexBatchRequest,
    IndexBatchResponse,
    PointData,
    RetrieveRequest,
    RetrieveResponse,
)

from warden_retrieval.grpc_server import RetrievalServiceServicerImpl, create_grpc_server
from warden_retrieval.indexer import IndexBatchResult
from warden_retrieval.laya_client import RerankResult, ScoredPassage
from warden_retrieval.search import RawCandidate


@pytest.mark.asyncio
async def test_grpc_retrieve_success():
    mock_search = AsyncMock()
    mock_search.search.return_value = [
        RawCandidate("DOC-1", 0, "Content 1", "file:///doc1", ["Employee"], 0.05),
        RawCandidate("DOC-2", 1, "Content 2", "file:///doc2", ["Employee"], 0.04),
    ]
    mock_laya = AsyncMock()
    mock_laya.rerank.return_value = RerankResult(
        passages=[
            ScoredPassage("DOC-2", 1, "Content 2", "file:///doc2", ["Employee"], 0.04, 0.95),
            ScoredPassage("DOC-1", 0, "Content 1", "file:///doc1", ["Employee"], 0.05, 0.85),
        ],
        reranker_applied=True,
        fallback_active=False,
        latency_ms=25.0,
    )
    mock_indexer = AsyncMock()

    servicer = RetrievalServiceServicerImpl(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
    )
    request = RetrieveRequest(
        query_text="health benefits",
        caller_role="Employee",
        top_k_candidates=30,
        final_rerank_limit=5,
        enable_reranker=True,
    )
    context = AsyncMock()
    response = await servicer.Retrieve(request, context)

    assert isinstance(response, RetrieveResponse)
    assert response.query_text == "health benefits"
    assert response.caller_role_applied == "Employee"
    assert response.reranker_applied is True
    assert response.fallback_active is False
    assert len(response.passages) == 2
    assert response.passages[0].doc_id == "DOC-2"
    assert response.passages[0].calibrated_score == 0.95
    assert response.raw_candidates_count == 2
    assert response.pruned_candidates_count == 2
    context.abort.assert_not_awaited()

@pytest.mark.asyncio
async def test_grpc_retrieve_invalid_role_aborts():
    # Review Focus 1: Invalid role aborts gRPC with INVALID_ARGUMENT
    servicer = RetrievalServiceServicerImpl(AsyncMock(), AsyncMock(), AsyncMock())
    request = RetrieveRequest(query_text="health", caller_role="Contractor")
    context = AsyncMock()
    await servicer.Retrieve(request, context)
    context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "Invalid or unauthorized role: Contractor"
    )

@pytest.mark.asyncio
async def test_grpc_retrieve_reranker_disabled():
    mock_search = AsyncMock()
    mock_search.search.return_value = [
        RawCandidate("DOC-1", 0, "Content 1", "file:///doc1", ["Employee"], 0.05)
    ]
    mock_laya = AsyncMock()
    mock_indexer = AsyncMock()

    servicer = RetrievalServiceServicerImpl(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
    )
    request = RetrieveRequest(
        query_text="health",
        caller_role="Employee",
        enable_reranker=False,
    )
    context = AsyncMock()
    response = await servicer.Retrieve(request, context)

    assert response.reranker_applied is False
    assert response.fallback_active is False
    assert len(response.passages) == 1
    assert response.passages[0].doc_id == "DOC-1"
    mock_laya.rerank.assert_not_awaited()

@pytest.mark.asyncio
async def test_grpc_index_batch_success():
    mock_indexer = AsyncMock()
    mock_indexer.index_points.return_value = IndexBatchResult(
        status="UPSERTED",
        points_count=2,
        execution_time_ms=12.5,
    )
    servicer = RetrievalServiceServicerImpl(
        search_engine=AsyncMock(),
        laya_client=AsyncMock(),
        indexer=mock_indexer,
    )
    point1 = PointData(point_id="1", doc_id="DOC-1")
    point2 = PointData(point_id="2", doc_id="DOC-2")
    request = IndexBatchRequest(points=[point1, point2], wait=False)
    context = AsyncMock()
    response = await servicer.IndexBatch(request, context)

    assert isinstance(response, IndexBatchResponse)
    assert response.status == "UPSERTED"
    assert response.points_count == 2
    assert response.execution_time_ms == 12.5
    mock_indexer.index_points.assert_awaited_once_with(request.points, wait=False)

@pytest.mark.asyncio
async def test_grpc_retrieve_qdrant_error_aborts_unavailable():
    from warden_shared.errors import QdrantUnavailableError

    mock_search = AsyncMock()
    mock_search.search.side_effect = QdrantUnavailableError("Qdrant down")
    servicer = RetrievalServiceServicerImpl(
        search_engine=mock_search,
        laya_client=AsyncMock(),
        indexer=AsyncMock(),
    )
    request = RetrieveRequest(query_text="health", caller_role="Employee")
    context = AsyncMock()
    await servicer.Retrieve(request, context)
    context.abort.assert_awaited_once()
    assert context.abort.call_args[0][0] == grpc.StatusCode.UNAVAILABLE

@pytest.mark.asyncio
async def test_grpc_index_qdrant_error_aborts_unavailable():
    from warden_shared.errors import QdrantUnavailableError

    mock_indexer = AsyncMock()
    mock_indexer.index_points.side_effect = QdrantUnavailableError("Qdrant timeout")
    servicer = RetrievalServiceServicerImpl(
        search_engine=AsyncMock(),
        laya_client=AsyncMock(),
        indexer=mock_indexer,
    )
    request = IndexBatchRequest(points=[PointData(point_id="1")], wait=False)
    context = AsyncMock()
    await servicer.IndexBatch(request, context)
    context.abort.assert_awaited_once()
    assert context.abort.call_args[0][0] == grpc.StatusCode.UNAVAILABLE

@pytest.mark.asyncio
async def test_create_grpc_server():
    servicer = RetrievalServiceServicerImpl(AsyncMock(), AsyncMock(), AsyncMock())
    server = create_grpc_server(servicer=servicer, port=50051)
    assert server is not None
    await server.stop(grace=0)
