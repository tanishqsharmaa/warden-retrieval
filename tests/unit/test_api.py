from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from warden_retrieval.api import create_app
from warden_retrieval.indexer import IndexBatchResult
from warden_retrieval.laya_client import RerankResult, ScoredPassage
from warden_retrieval.search import RawCandidate


@pytest.fixture
def mock_dependencies():
    mock_search = AsyncMock()
    mock_search.search.return_value = [
        RawCandidate("DOC-1", 0, "PTO details", "file:///pto.md", ["Employee"], 0.05)
    ]
    mock_laya = AsyncMock()
    mock_laya.rerank.return_value = RerankResult(
        passages=[
            ScoredPassage("DOC-1", 0, "PTO details", "file:///pto.md", ["Employee"], 0.05, 0.95)
        ],
        reranker_applied=True,
        fallback_active=False,
        latency_ms=30.0,
    )
    mock_indexer = AsyncMock()
    mock_indexer.index_points.return_value = IndexBatchResult(
        status="UPSERTED",
        points_count=1,
        execution_time_ms=5.0,
    )
    mock_client_mgr = AsyncMock()
    mock_client_mgr.is_healthy.return_value = True

    return mock_search, mock_laya, mock_indexer, mock_client_mgr

@pytest.mark.asyncio
async def test_retrieve_missing_role_returns_401(mock_dependencies):
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/retrieve", json={"query_text": "PTO"})

    assert resp.status_code == 401
    data = resp.json()
    assert data["error_code"] == "ERR_AUTH_ROLE_MISSING"

@pytest.mark.asyncio
async def test_retrieve_invalid_role_returns_403(mock_dependencies):
    # Review Focus 1: Invalid role returns HTTP 403
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/retrieve",
            json={"query_text": "PTO"},
            headers={"X-User-Role": "Contractor"},
        )

    assert resp.status_code == 403
    data = resp.json()
    assert data["error_code"] == "ERR_AUTH_ROLE_INVALID"

@pytest.mark.asyncio
async def test_retrieve_success(mock_dependencies):
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/retrieve",
            json={
                "query_text": "How many days of PTO?",
                "top_k_candidates": 30,
                "final_rerank_limit": 5,
                "enable_reranker": True,
            },
            headers={"X-User-Role": "Employee"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query_text"] == "How many days of PTO?"
    assert data["caller_role_applied"] == "Employee"
    assert data["reranker_applied"] is True
    assert data["fallback_active"] is False
    assert len(data["passages"]) == 1
    assert data["passages"][0]["doc_id"] == "DOC-1"
    assert data["passages"][0]["calibrated_relevance_score"] == 0.95
    assert "latency_breakdown_ms" in data

@pytest.mark.asyncio
async def test_internal_index_success(mock_dependencies):
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    payload = {
        "points": [
            {
                "point_id": "e81d7f3e-2b58-5d32-9f37-123456789abc",
                "doc_id": "DOC-HR-LEAVE-2026",
                "chunk_index": 0,
                "content": "PTO policy details...",
                "dense_vector": [0.01] * 768,
                "role_tags": ["Employee"],
                "redacted": True,
                "source_url": "file:///doc.md",
                "token_count": 512,
            }
        ]
    }
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/internal/index", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "UPSERTED"
    assert data["points_count"] == 1
    assert "execution_time_ms" in data

@pytest.mark.asyncio
async def test_health_healthy_returns_200(mock_dependencies):
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert data["qdrant_connected"] is True

@pytest.mark.asyncio
async def test_health_qdrant_down_returns_503(mock_dependencies):
    # Review Focus 2: Qdrant unreachable returns HTTP 503
    mock_search, mock_laya, mock_indexer, mock_client_mgr = mock_dependencies
    mock_client_mgr.is_healthy.return_value = False

    app = create_app(
        search_engine=mock_search,
        laya_client=mock_laya,
        indexer=mock_indexer,
        client_manager=mock_client_mgr,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["error_code"] == "ERR_QDRANT_UNAVAILABLE"
