import logging
import time
from typing import Optional

import grpc
from warden_shared.errors import QdrantUnavailableError
from warden_shared.proto.v1.retrieval_pb2 import (
    IndexBatchRequest,
    IndexBatchResponse,
    Passage,
    RetrieveRequest,
    RetrieveResponse,
)
from warden_shared.proto.v1.retrieval_pb2_grpc import (
    RetrievalServiceServicer,
    add_RetrievalServiceServicer_to_server,
)

from warden_retrieval.config import Settings, get_settings
from warden_retrieval.indexer import BatchIndexer
from warden_retrieval.laya_client import SpeculativeLayaClient
from warden_retrieval.pruner import prune_candidates
from warden_retrieval.search import HybridSearchEngine

logger = logging.getLogger("warden.retrieval.grpc")

ALLOWED_ROLES = {"Employee", "Manager", "HR-Admin"}

class RetrievalServiceServicerImpl(RetrievalServiceServicer):
    """Fulfills RetrievalService gRPC contract with early-binding ACL, dynamic pruning,

    and speculative reranking.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        laya_client: SpeculativeLayaClient,
        indexer: BatchIndexer,
        settings: Optional[Settings] = None,
    ) -> None:
        self.search_engine = search_engine
        self.laya_client = laya_client
        self.indexer = indexer
        self.settings = settings or get_settings()

    async def Retrieve(
        self, request: RetrieveRequest, context: grpc.aio.ServicerContext
    ) -> RetrieveResponse:
        """Executes access-controlled hybrid retrieval with speculative Laya rerank."""
        t0 = time.perf_counter()

        # Validate caller role
        if request.caller_role not in ALLOWED_ROLES:
            logger.warning(f"Rejected retrieval request with invalid role: {request.caller_role}")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid or unauthorized role: {request.caller_role}",
            )
            return RetrieveResponse()

        top_k = request.top_k_candidates or self.settings.RETRIEVAL_TOP_K_DEFAULT
        final_limit = request.final_rerank_limit or self.settings.RERANK_LIMIT_DEFAULT

        # 1. Early-binding hybrid search against Qdrant
        try:
            raw_candidates = await self.search_engine.search(
                query_text=request.query_text,
                caller_role=request.caller_role,
                top_k=top_k,
            )
        except QdrantUnavailableError as e:
            logger.error(f"Datastore unavailable during retrieval: {e}")
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                e.detail or "Qdrant datastore unavailable",
            )
            return RetrieveResponse()

        # 2. Dynamic candidate score pruning (40% drop-off threshold, <=10 chunks)
        pruned_candidates = prune_candidates(
            candidates=raw_candidates,
            prune_ratio=self.settings.CANDIDATE_PRUNE_RATIO,
            max_candidates=self.settings.PRUNED_CANDIDATE_MAX,
        )

        # 3. Speculative Laya cross-encoder rerank
        if request.enable_reranker and pruned_candidates:
            rerank_result = await self.laya_client.rerank(
                query=request.query_text,
                candidates=pruned_candidates,
                final_limit=final_limit,
                deadline_ms=self.settings.LAYA_DEADLINE_MS,
            )
            passages = [
                Passage(
                    doc_id=p.doc_id,
                    chunk_index=p.chunk_index,
                    content=p.content,
                    source_url=p.source_url,
                    role_tags=p.role_tags,
                    rrf_score=p.rrf_score,
                    calibrated_score=p.calibrated_score,
                )
                for p in rerank_result.passages
            ]
            reranker_applied = rerank_result.reranker_applied
            fallback_active = rerank_result.fallback_active
        else:
            passages = [
                Passage(
                    doc_id=c.doc_id,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    source_url=c.source_url,
                    role_tags=c.role_tags,
                    rrf_score=c.rrf_score,
                    calibrated_score=0.0,
                )
                for c in pruned_candidates[:final_limit]
            ]
            reranker_applied = False
            fallback_active = False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RetrieveResponse(
            query_text=request.query_text,
            caller_role_applied=request.caller_role,
            passages=passages,
            reranker_applied=reranker_applied,
            fallback_active=fallback_active,
            execution_time_ms=elapsed_ms,
            raw_candidates_count=len(raw_candidates),
            pruned_candidates_count=len(pruned_candidates),
        )

    async def IndexBatch(
        self, request: IndexBatchRequest, context: grpc.aio.ServicerContext
    ) -> IndexBatchResponse:
        """Indexes a batch of vectorized points from upstream warden-ingestion."""
        try:
            result = await self.indexer.index_points(request.points, wait=request.wait)
            return IndexBatchResponse(
                status=result.status,
                points_count=result.points_count,
                execution_time_ms=result.execution_time_ms,
            )
        except QdrantUnavailableError as e:
            logger.error(f"Datastore unavailable during index batch: {e}")
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                e.detail or "Qdrant datastore unavailable",
            )
            return IndexBatchResponse()

def create_grpc_server(
    servicer: RetrievalServiceServicerImpl,
    port: int = 50051,
) -> grpc.aio.Server:
    """Configures and binds a high-performance gRPC HTTP/2 server."""
    server = grpc.aio.server(
        options=[
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.http2.max_pings_without_data", 0),
        ]
    )
    add_RetrievalServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
