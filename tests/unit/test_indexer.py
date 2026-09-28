from unittest.mock import AsyncMock

import pytest
from warden_shared.errors import QdrantUnavailableError
from warden_shared.proto.v1.retrieval_pb2 import PointData

from warden_retrieval.indexer import BatchIndexer, IndexBatchResult


@pytest.mark.asyncio
async def test_index_points_success():
    mock_client = AsyncMock()
    mock_client.upload_points = AsyncMock()

    indexer = BatchIndexer(client=mock_client, collection_name="warden_hr_policies")
    point = PointData(
        point_id="e81d7f3e-2b58-5d32-9f37-123456789abc",
        doc_id="DOC-HR-LEAVE-2026",
        chunk_index=0,
        content=(
            "Paid Time Off (PTO) Policy: Full-time employees accrue 18 days "
            "of paid leave annually..."
        ),
        dense_vector=[0.0124] * 768,
        role_tags=["Employee", "Manager"],
        redacted=True,
        source_url="file:///data/policies/pto_policy_2026.md",
        token_count=512,
    )
    result = await indexer.index_points([point], wait=False)

    assert isinstance(result, IndexBatchResult)
    assert result.status == "UPSERTED"
    assert result.points_count == 1
    assert result.execution_time_ms >= 0.0
    mock_client.upload_points.assert_awaited_once()

    call_kwargs = mock_client.upload_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "warden_hr_policies"
    assert call_kwargs["batch_size"] == 64
    assert call_kwargs["parallel"] == 2
    assert call_kwargs["wait"] is False
    assert call_kwargs["method"] == "grpc"

    points_arg = call_kwargs["points"]
    assert len(points_arg) == 1
    p = points_arg[0]
    assert p.id == "e81d7f3e-2b58-5d32-9f37-123456789abc"
    assert p.vector["dense"] == pytest.approx([0.0124] * 768, rel=1e-5)
    assert p.payload["doc_id"] == "DOC-HR-LEAVE-2026"
    assert p.payload["role_tags"] == ["Employee", "Manager"]
    assert p.payload["redacted"] is True

@pytest.mark.asyncio
async def test_index_points_empty_batch():
    mock_client = AsyncMock()
    indexer = BatchIndexer(client=mock_client, collection_name="warden_hr_policies")
    result = await indexer.index_points([], wait=False)
    assert result.status == "UPSERTED"
    assert result.points_count == 0
    mock_client.upload_points.assert_not_awaited()

@pytest.mark.asyncio
async def test_index_points_qdrant_error_propagates():
    mock_client = AsyncMock()
    mock_client.upload_points.side_effect = Exception("Write timeout")

    indexer = BatchIndexer(client=mock_client, collection_name="warden_hr_policies")
    point = PointData(
        point_id="11111111-1111-1111-1111-111111111111",
        doc_id="DOC-1",
        chunk_index=0,
        content="Test",
        dense_vector=[0.1] * 768,
        role_tags=["Employee"],
    )
    with pytest.raises(QdrantUnavailableError):
        await indexer.index_points([point], wait=False)
