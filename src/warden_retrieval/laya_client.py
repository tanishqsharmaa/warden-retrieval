import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import grpc
from warden_shared.proto.v1.laya_pb2 import CandidateItem, RerankRequest
from warden_shared.proto.v1.laya_pb2_grpc import LayaInferenceServiceStub

from warden_retrieval.search import RawCandidate

logger = logging.getLogger("warden.retrieval.laya")

@dataclass(frozen=True)
class ScoredPassage:
    doc_id: str
    chunk_index: int
    content: str
    source_url: str
    role_tags: list[str]
    rrf_score: float
    calibrated_score: float

@dataclass(frozen=True)
class RerankResult:
    passages: list[ScoredPassage]
    reranker_applied: bool
    fallback_active: bool
    latency_ms: float

class SpeculativeLayaClient:
    """Speculatively dispatches pruned candidates to warden-laya-service with strict 150ms deadline.

    Implements a circuit breaker state machine: on consecutive failures, trips to OPEN for a
    cooldown window, immediately bypassing Laya to protect overall query p50/p95 latency.
    """

    def __init__(
        self,
        target_url: str = "warden-laya-service:50051",
        stub: Optional[LayaInferenceServiceStub] = None,
        failure_threshold: int = 5,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self.target_url = target_url
        self._stub = stub
        self._channel: Optional[grpc.aio.Channel] = None
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._circuit_tripped_at = 0.0
        self._init_lock = asyncio.Lock()

    @property
    def is_circuit_open(self) -> bool:
        """Returns True if the circuit breaker is open (tripped) and in cooldown."""
        if self._consecutive_failures >= self.failure_threshold:
            if time.monotonic() - self._circuit_tripped_at < self.cooldown_seconds:
                return True
        return False

    async def get_stub(self) -> LayaInferenceServiceStub:
        """Returns or thread-safely lazily creates the gRPC client stub."""
        if self._stub is None:
            async with self._init_lock:
                if self._stub is None:
                    self._channel = grpc.aio.insecure_channel(
                        self.target_url,
                        options=[
                            ("grpc.keepalive_time_ms", 30000),
                            ("grpc.keepalive_timeout_ms", 5000),
                        ],
                    )
                    self._stub = LayaInferenceServiceStub(self._channel)
        return self._stub

    def _make_fallback_result(
        self, candidates: Sequence[RawCandidate], final_limit: int, latency_ms: float = 0.0
    ) -> RerankResult:
        """Creates fallback result using raw RRF candidate rankings."""
        fallback_passages = [
            ScoredPassage(
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                content=c.content,
                source_url=c.source_url,
                role_tags=c.role_tags,
                rrf_score=c.rrf_score,
                calibrated_score=0.0,
            )
            for c in candidates
        ]
        fallback_passages.sort(key=lambda p: p.rrf_score, reverse=True)
        return RerankResult(
            passages=fallback_passages[:final_limit],
            reranker_applied=False,
            fallback_active=True,
            latency_ms=latency_ms,
        )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RawCandidate],
        final_limit: int = 5,
        deadline_ms: int = 150,
    ) -> RerankResult:
        """Dispatches candidate list to Laya cross-encoder with circuit breaker fallback."""
        if not candidates:
            return RerankResult(
                passages=[],
                reranker_applied=False,
                fallback_active=False,
                latency_ms=0.0,
            )

        # Immediate bypass if circuit breaker is tripped
        if self.is_circuit_open:
            logger.warning(
                f"Laya circuit breaker is OPEN (failures={self._consecutive_failures}). "
                "Immediately bypassing reranking."
            )
            return self._make_fallback_result(candidates, final_limit, latency_ms=0.0)

        cand_map = {f"{c.doc_id}_{c.chunk_index}": c for c in candidates}
        items = [
            CandidateItem(candidate_id=cid, text=c.content)
            for cid, c in cand_map.items()
        ]
        request = RerankRequest(query=query, candidates=items, primitive="noul")

        t0 = time.perf_counter()
        stub = await self.get_stub()
        try:
            timeout_sec = deadline_ms / 1000.0
            response = await stub.Rerank(request, timeout=timeout_sec)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            scored_passages: list[ScoredPassage] = []
            for item in response.ranked_results:
                orig = cand_map.get(item.candidate_id)
                if orig:
                    scored_passages.append(
                        ScoredPassage(
                            doc_id=orig.doc_id,
                            chunk_index=orig.chunk_index,
                            content=orig.content,
                            source_url=orig.source_url,
                            role_tags=orig.role_tags,
                            rrf_score=orig.rrf_score,
                            calibrated_score=float(item.probability_true),
                        )
                    )

            # If Laya returns empty ranked_results, fall back to input candidates
            if not scored_passages and candidates:
                logger.warning("Laya returned empty ranked_results. Falling back to raw RRF order.")
                return self._make_fallback_result(candidates, final_limit, latency_ms=elapsed_ms)

            # Success: reset circuit failure counter
            self._consecutive_failures = 0

            # Sort descending by calibrated cross-encoder probability score
            scored_passages.sort(key=lambda p: p.calibrated_score, reverse=True)
            logger.info(
                f"Laya rerank completed in {elapsed_ms:.2f}ms for {len(candidates)} "
                f"candidates -> top {final_limit}"
            )
            return RerankResult(
                passages=scored_passages[:final_limit],
                reranker_applied=True,
                fallback_active=False,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._consecutive_failures += 1
            self._circuit_tripped_at = time.monotonic()
            logger.warning(
                f"Laya speculative rerank tripped circuit breaker after {elapsed_ms:.2f}ms: {e}. "
                f"Failures: {self._consecutive_failures}. Falling back to raw RRF order."
            )
            return self._make_fallback_result(candidates, final_limit, latency_ms=elapsed_ms)

    async def close(self) -> None:
        """Closes gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
