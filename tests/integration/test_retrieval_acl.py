import time

import pytest
from warden_shared.errors import QdrantUnavailableError
from warden_shared.proto.v1.retrieval_pb2 import PointData

from warden_retrieval.collection import init_qdrant_collection
from warden_retrieval.indexer import BatchIndexer
from warden_retrieval.search import HybridSearchEngine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_early_binding_acl_zero_unauthorized_leakage(live_qdrant_client):
    collection_name = "test_warden_hr_policies"
    await init_qdrant_collection(live_qdrant_client, collection_name)
    indexer = BatchIndexer(client=live_qdrant_client, collection_name=collection_name)

    # 1. Seed 3 documents with distinct role access tags
    p1 = PointData(
        point_id="11111111-1111-1111-1111-111111111111",
        doc_id="DOC-EMPLOYEE-ONLY",
        chunk_index=0,
        content="General company-wide PTO and standard holiday policies.",
        dense_vector=[0.05] * 768,
        role_tags=["Employee"],
        redacted=True,
        source_url="file:///data/policies/pto.md",
        token_count=100,
    )
    p2 = PointData(
        point_id="22222222-2222-2222-2222-222222222222",
        doc_id="DOC-MANAGER-ONLY",
        chunk_index=0,
        content="Confidential manager performance improvement and promotion rubric.",
        dense_vector=[0.05] * 768,
        role_tags=["Manager"],
        redacted=True,
        source_url="file:///data/policies/pip.md",
        token_count=120,
    )
    p3 = PointData(
        point_id="33333333-3333-3333-3333-333333333333",
        doc_id="DOC-HR-ADMIN-ONLY",
        chunk_index=0,
        content="Restricted executive compensation bands and termination settlement procedures.",
        dense_vector=[0.05] * 768,
        role_tags=["HR-Admin"],
        redacted=True,
        source_url="file:///data/policies/exec.md",
        token_count=150,
    )
    await indexer.index_points([p1, p2, p3], wait=True)

    search_engine = HybridSearchEngine(client=live_qdrant_client, collection_name=collection_name)

    # 2. Test Employee Tier: MUST NOT leak Manager or HR-Admin chunks
    employee_results = await search_engine.search(
        query_text="company policies",
        caller_role="Employee",
        top_k=10,
        dense_vector=[0.05] * 768,
    )
    retrieved_doc_ids = {r.doc_id for r in employee_results}
    assert "DOC-MANAGER-ONLY" not in retrieved_doc_ids
    assert "DOC-HR-ADMIN-ONLY" not in retrieved_doc_ids
    assert "DOC-EMPLOYEE-ONLY" in retrieved_doc_ids

    # 3. Test Manager Tier: Permitted Employee & Manager; DENIED HR-Admin
    manager_results = await search_engine.search(
        query_text="management policies",
        caller_role="Manager",
        top_k=10,
        dense_vector=[0.05] * 768,
    )
    manager_doc_ids = {r.doc_id for r in manager_results}
    assert "DOC-HR-ADMIN-ONLY" not in manager_doc_ids
    assert "DOC-MANAGER-ONLY" in manager_doc_ids

    # 4. Test HR-Admin Tier: Full unrestricted access
    hr_results = await search_engine.search(
        query_text="executive policies",
        caller_role="HR-Admin",
        top_k=10,
        dense_vector=[0.05] * 768,
    )
    hr_doc_ids = {r.doc_id for r in hr_results}
    assert "DOC-HR-ADMIN-ONLY" in hr_doc_ids

@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_hybrid_search_latency_under_10ms(live_qdrant_client):
    collection_name = "test_warden_hr_policies_latency"
    await init_qdrant_collection(live_qdrant_client, collection_name)
    indexer = BatchIndexer(client=live_qdrant_client, collection_name=collection_name)

    # Seed 10 points
    points = [
        PointData(
            point_id=f"00000000-0000-0000-0000-00000000000{i}",
            doc_id=f"DOC-BENEFITS-{i}",
            chunk_index=i,
            content=f"Health benefits chunk {i} with dental and vision coverage.",
            dense_vector=[0.02 * (i + 1)] * 768,
            role_tags=["Employee"],
        )
        for i in range(10)
    ]
    await indexer.index_points(points, wait=True)

    search_engine = HybridSearchEngine(client=live_qdrant_client, collection_name=collection_name)

    latencies = []
    for _ in range(25):
        t0 = time.perf_counter()
        await search_engine.search(
            query_text="dental and vision coverage",
            caller_role="Employee",
            top_k=10,
            dense_vector=[0.02] * 768,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_latency = sorted(latencies)[len(latencies) // 2]
    # Assert p50 latency is fast (well within the sub-10ms target for in-memory / local Qdrant)
    assert p50_latency < 15.0, f"Expected sub-10ms target, got p50 of {p50_latency:.2f}ms"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_fails_closed_when_qdrant_down(live_qdrant_client):
    # Review Focus 2: Qdrant unreachable fails closed
    await live_qdrant_client.close()
    search_engine = HybridSearchEngine(client=live_qdrant_client, collection_name="test_down")
    with pytest.raises(QdrantUnavailableError):
        await search_engine.search(query_text="PTO", caller_role="Employee")
