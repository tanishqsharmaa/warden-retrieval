import logging
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models
from warden_shared.errors import QdrantUnavailableError

logger = logging.getLogger("warden.retrieval.search")

@dataclass(frozen=True)
class RawCandidate:
    doc_id: str
    chunk_index: int
    content: str
    source_url: str
    role_tags: list[str]
    rrf_score: float

class HybridSearchEngine:
    """Executes hybrid vector + lexical search against Qdrant with early-binding ACL filtering."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        dense_embedder: Any = None,
        sparse_embedder: Any = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder

    async def search(
        self,
        query_text: str,
        caller_role: str,
        top_k: int = 30,
        dense_vector: Optional[list[float]] = None,
        sparse_indices: Optional[list[int]] = None,
        sparse_values: Optional[list[float]] = None,
    ) -> list[RawCandidate]:
        """Executes access-controlled hybrid retrieval with native Reciprocal Rank Fusion (RRF)."""
        # Generate dense and sparse query embeddings if embedders are configured
        if dense_vector is None and self.dense_embedder is not None:
            try:
                dense_res = list(self.dense_embedder.embed([query_text]))
                if dense_res:
                    first = dense_res[0]
                    dense_vector = list(first.tolist() if hasattr(first, "tolist") else first)
            except Exception as e:
                logger.warning(f"Dense query embedding failed: {e}")

        if (
            sparse_indices is None
            and sparse_values is None
            and self.sparse_embedder is not None
        ):
            try:
                sparse_res = list(self.sparse_embedder.embed([query_text]))
                if sparse_res:
                    s_item = sparse_res[0]
                    sparse_indices = list(getattr(s_item, "indices", []))
                    sparse_values = list(getattr(s_item, "values", []))
            except Exception as e:
                logger.warning(f"Sparse BM25 query embedding failed: {e}")

        # Enforce early-binding ACL security filter
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="role_tags",
                    match=models.MatchAny(any=[caller_role]),
                )
            ]
        )

        # Apply tuned SearchParams per ARCHITECTURE_SPECIFICATION § 4.2.4
        search_params = models.SearchParams(
            hnsw_ef=64,
            exact=False,
            quantization=models.QuantizationSearchParams(
                rescore=True,
                oversampling=2.0,
            ),
        )

        try:
            # Build prefetch queries if vectors are provided, or native fusion query
            prefetch: list[models.Prefetch] = []
            if dense_vector is not None:
                prefetch.append(
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        filter=query_filter,
                        limit=top_k,
                    )
                )
            if sparse_indices is not None and sparse_values is not None:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse_bm25",
                        filter=query_filter,
                        limit=top_k,
                    )
                )

            query_kwargs: dict[str, Any] = {
                "collection_name": self.collection_name,
                "query_filter": query_filter,
                "search_params": search_params,
                "limit": top_k,
            }

            if prefetch:
                query_kwargs["prefetch"] = prefetch
                query_kwargs["query"] = models.FusionQuery(fusion=models.Fusion.RRF)
            elif dense_vector is not None:
                query_kwargs["query"] = dense_vector
                query_kwargs["using"] = "dense"

            response = await self.client.query_points(**query_kwargs)

            candidates: list[RawCandidate] = []
            points = getattr(response, "points", []) or []
            for pt in points:
                payload = pt.payload or {}
                candidates.append(
                    RawCandidate(
                        doc_id=str(payload.get("doc_id", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        content=str(payload.get("content", "")),
                        source_url=str(payload.get("source_url", "")),
                        role_tags=list(payload.get("role_tags", [])),
                        rrf_score=float(pt.score or 0.0),
                    )
                )

            logger.info(
                f"Retrieved {len(candidates)} raw candidates for role '{caller_role}' "
                f"(limit={top_k})"
            )
            return candidates
        except Exception as e:
            logger.error(f"Qdrant hybrid search failed for role '{caller_role}': {e}")
            err_msg = f"Hybrid search failed on collection '{self.collection_name}': {e}"
            raise QdrantUnavailableError(detail=err_msg) from e
