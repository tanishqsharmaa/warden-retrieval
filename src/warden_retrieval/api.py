import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field
from warden_shared.errors import (
    QdrantUnavailableError,
    RoleInvalidError,
    RoleMissingError,
    register_error_handlers,
)
from warden_shared.proto.v1.retrieval_pb2 import PointData

from warden_retrieval.config import Settings, get_settings
from warden_retrieval.indexer import BatchIndexer
from warden_retrieval.laya_client import SpeculativeLayaClient
from warden_retrieval.pruner import prune_candidates
from warden_retrieval.qdrant_client import QdrantClientManager
from warden_retrieval.search import HybridSearchEngine

logger = logging.getLogger("warden.retrieval.api")

ALLOWED_ROLES = {"Employee", "Manager", "HR-Admin"}

class RetrieveRequestBody(BaseModel):
    query_text: str = Field(..., description="Natural language search query")
    top_k_candidates: Optional[int] = Field(
        30, description="Raw candidates to retrieve from Qdrant"
    )
    final_rerank_limit: Optional[int] = Field(5, description="Passages to return post-rerank")
    enable_reranker: Optional[bool] = Field(
        True, description="Enable speculative Laya cross-encoder reranker"
    )

class PointDataInput(BaseModel):
    point_id: str
    doc_id: str
    chunk_index: int
    content: str
    dense_vector: list[float]
    role_tags: list[str]
    redacted: bool = False
    source_url: str = ""
    token_count: int = 0

class IndexBatchRequestBody(BaseModel):
    points: list[PointDataInput]
    wait: bool = False

def create_app(
    search_engine: Optional[HybridSearchEngine] = None,
    laya_client: Optional[SpeculativeLayaClient] = None,
    indexer: Optional[BatchIndexer] = None,
    client_manager: Optional[QdrantClientManager] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """Factory creating and configuring the FastAPI REST application."""
    cfg = settings or get_settings()
    app = FastAPI(
        title="warden-retrieval",
        description="Project Warden Storage Custodian & Hybrid Retrieval Subsystem (Tier 3)",
        version="1.0.0",
    )
    register_error_handlers(app)

    @app.post("/retrieve")
    async def retrieve_endpoint(
        request: Request,
        body: RetrieveRequestBody,
        x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    ) -> dict[str, Any]:
        """Access-controlled hybrid retrieval with early-binding ACL and speculative reranking."""
        if not x_user_role:
            raise RoleMissingError(detail="Missing mandatory X-User-Role header.")
        if x_user_role not in ALLOWED_ROLES:
            raise RoleInvalidError(detail=f"Role '{x_user_role}' is unauthorized.")

        engine = getattr(request.app.state, "search_engine", None) or search_engine
        laya = getattr(request.app.state, "laya_client", None) or laya_client
        if engine is None or laya is None:
            raise QdrantUnavailableError(detail="Retrieval service dependencies not initialized.")

        t_total_0 = time.perf_counter()
        top_k = body.top_k_candidates or cfg.RETRIEVAL_TOP_K_DEFAULT
        final_limit = body.final_rerank_limit or cfg.RERANK_LIMIT_DEFAULT

        # 1. Early-binding hybrid search against Qdrant
        t_search_0 = time.perf_counter()
        raw_candidates = await engine.search(
            query_text=body.query_text,
            caller_role=x_user_role,
            top_k=top_k,
        )
        qdrant_search_ms = (time.perf_counter() - t_search_0) * 1000.0

        # 2. Dynamic candidate score pruning
        t_prune_0 = time.perf_counter()
        pruned_candidates = prune_candidates(
            candidates=raw_candidates,
            prune_ratio=cfg.CANDIDATE_PRUNE_RATIO,
            max_candidates=cfg.PRUNED_CANDIDATE_MAX,
        )
        dynamic_pruning_ms = (time.perf_counter() - t_prune_0) * 1000.0

        # 3. Speculative Laya cross-encoder rerank
        laya_rerank_ms = 0.0
        if body.enable_reranker and pruned_candidates:
            rerank_res = await laya.rerank(
                query=body.query_text,
                candidates=pruned_candidates,
                final_limit=final_limit,
                deadline_ms=cfg.LAYA_DEADLINE_MS,
            )
            passages_data = [
                {
                    "doc_id": p.doc_id,
                    "chunk_index": p.chunk_index,
                    "content": p.content,
                    "source_url": p.source_url,
                    "role_tags": p.role_tags,
                    "rrf_score": p.rrf_score,
                    "calibrated_relevance_score": p.calibrated_score,
                }
                for p in rerank_res.passages
            ]
            reranker_applied = rerank_res.reranker_applied
            fallback_active = rerank_res.fallback_active
            laya_rerank_ms = rerank_res.latency_ms
        else:
            passages_data = [
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "source_url": c.source_url,
                    "role_tags": c.role_tags,
                    "rrf_score": c.rrf_score,
                    "calibrated_relevance_score": 0.0,
                }
                for c in pruned_candidates[:final_limit]
            ]
            reranker_applied = False
            fallback_active = False

        total_ms = (time.perf_counter() - t_total_0) * 1000.0

        return {
            "query_text": body.query_text,
            "caller_role_applied": x_user_role,
            "candidates_count": len(passages_data),
            "raw_candidates_retrieved": len(raw_candidates),
            "pruned_candidates_reranked": len(pruned_candidates),
            "reranker_applied": reranker_applied,
            "fallback_active": fallback_active,
            "passages": passages_data,
            "latency_breakdown_ms": {
                "qdrant_hybrid_search_ms": round(qdrant_search_ms, 2),
                "dynamic_pruning_ms": round(dynamic_pruning_ms, 2),
                "laya_rerank_ms": round(laya_rerank_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    @app.post("/internal/index")
    async def index_endpoint(
        request: Request, body: IndexBatchRequestBody
    ) -> dict[str, Any]:
        """Bulk point upsertion endpoint invoked by warden-ingestion."""
        idxer = getattr(request.app.state, "indexer", None) or indexer
        if idxer is None:
            raise QdrantUnavailableError(detail="Indexer dependency not initialized.")

        pb_points = [
            PointData(
                point_id=p.point_id,
                doc_id=p.doc_id,
                chunk_index=p.chunk_index,
                content=p.content,
                dense_vector=p.dense_vector,
                role_tags=p.role_tags,
                redacted=p.redacted,
                source_url=p.source_url,
                token_count=p.token_count,
            )
            for p in body.points
        ]
        result = await idxer.index_points(pb_points, wait=body.wait)
        return {
            "status": result.status,
            "points_count": result.points_count,
            "collection_name": cfg.QDRANT_COLLECTION,
            "execution_time_ms": round(result.execution_time_ms, 2),
        }

    @app.get("/health")
    async def health_endpoint(request: Request) -> dict[str, Any]:
        """Health check endpoint probing Qdrant connectivity."""
        client_mgr = getattr(request.app.state, "client_manager", None) or client_manager
        if client_mgr is None:
            raise QdrantUnavailableError(detail="Qdrant client manager not initialized.")

        is_healthy = await client_mgr.is_healthy()
        if not is_healthy:
            raise QdrantUnavailableError(
                detail="Qdrant datastore unreachable or collection missing."
            )

        vectors_count = 0
        try:
            client = await client_mgr.get_client()
            col_info = await client.get_collection(cfg.QDRANT_COLLECTION)
            vectors_count = getattr(col_info, "points_count", 0) or 0
        except Exception:
            pass

        return {
            "status": "HEALTHY",
            "service": "warden-retrieval",
            "qdrant_connected": True,
            "collection_exists": True,
            "vectors_count": vectors_count,
            "grpc_server_active": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return app
