import logging
import time
from dataclasses import dataclass
from typing import Sequence

from qdrant_client import AsyncQdrantClient, models
from warden_shared.errors import QdrantUnavailableError
from warden_shared.proto.v1.retrieval_pb2 import PointData

logger = logging.getLogger("warden.retrieval.indexer")

@dataclass(frozen=True)
class IndexBatchResult:
    status: str
    points_count: int
    execution_time_ms: float

class BatchIndexer:
    """High-throughput point batch indexer communicating with Qdrant over gRPC."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    async def index_points(
        self, points: Sequence[PointData], wait: bool = False
    ) -> IndexBatchResult:
        """Upserts a batch of vectorized points into the Qdrant collection."""
        if not points:
            return IndexBatchResult(status="UPSERTED", points_count=0, execution_time_ms=0.0)

        t0 = time.perf_counter()
        try:
            point_structs: list[models.PointStruct] = []
            for p in points:
                payload = {
                    "doc_id": p.doc_id,
                    "chunk_index": p.chunk_index,
                    "content": p.content,
                    "role_tags": list(p.role_tags),
                    "redacted": p.redacted,
                    "source_url": p.source_url,
                    "token_count": p.token_count,
                }
                point_structs.append(
                    models.PointStruct(
                        id=p.point_id,
                        vector={"dense": list(p.dense_vector)},
                        payload=payload,
                    )
                )

            upload_res = self.client.upload_points(
                collection_name=self.collection_name,
                points=point_structs,
                batch_size=64,
                parallel=2,
                wait=wait,
                method="grpc",
            )
            import inspect
            if inspect.isawaitable(upload_res):
                await upload_res
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"Successfully upserted {len(points)} points into '{self.collection_name}' "
                f"in {elapsed_ms:.2f}ms (wait={wait})"
            )
            return IndexBatchResult(
                status="UPSERTED",
                points_count=len(points),
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"Failed to upsert points batch into '{self.collection_name}': {e}")
            err_msg = f"Failed to upsert batch into '{self.collection_name}': {e}"
            raise QdrantUnavailableError(detail=err_msg) from e
