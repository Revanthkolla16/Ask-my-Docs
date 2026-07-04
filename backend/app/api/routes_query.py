"""
backend/app/api/routes_query.py
---------------------------------
POST /query endpoint.

Pipeline:
    QueryRequest
        → HybridRetriever.retrieve(query, top_k)
        → CrossEncoderReranker.rerank(query, candidates, top_n)
        → CitationGenerator.generate(query, context_chunks)
        → HallucinationDetector.detect(answer, citations, context_chunks)
        → QueryResponse

The endpoint is wired into FastAPI via ``app.main`` and exported from
``app.api.__init__``.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.generation import CitationGenerator, GenerationResult
from app.hallucination import HallucinationDetector
from app.models import QueryRequest, QueryResponse
from app.retrieval import CrossEncoderReranker, HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Singleton pipeline components ────────────────────────────────────────────
# Instantiated once per process; lazy model loading keeps startup fast.

_hybrid_retriever = HybridRetriever(
    embedding_model=settings.embedding_model,
    persist_dir=settings.chroma_persist_dir,
    rrf_k=settings.rrf_k,
)

_reranker = CrossEncoderReranker(model_name=settings.reranker_model)

_generator = CitationGenerator(
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)

_detector = HallucinationDetector(
    model_name=settings.nli_model,
    threshold=settings.hallucination_threshold,
)


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the indexed documents",
    description=(
        "Run the full RAG pipeline: hybrid retrieval → cross-encoder reranking "
        "→ Groq citation-grounded generation.  Returns the answer, source "
        "citations, and optional hallucination flags."
    ),
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    End-to-end RAG query endpoint.

    1. Hybrid retrieval (dense + BM25, RRF fusion).
    2. Cross-encoder reranking.
    3. Citation-grounded generation via Groq (JSON mode).
    4. Claim-level hallucination detection via NLI cross-encoder.
    """
    t_start = time.perf_counter()

    top_k = request.top_k or settings.retrieval_top_k
    top_n = request.top_n or settings.rerank_top_n

    # ── 1. Hybrid retrieval ───────────────────────────────────────────────────
    logger.info("Query pipeline: retrieving top-%d candidates …", top_k)
    try:
        candidates = _hybrid_retriever.retrieve(query=request.query, top_k=top_k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    if not candidates:
        logger.warning("No chunks retrieved for query: %r", request.query)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return QueryResponse(
            answer="I couldn't find any relevant documents. Please ingest some documents first.",
            citations=[],
            hallucination_flags=[],
            confidence=None,
            latency_ms=total_ms,
        )

    # ── 2. Reranking ──────────────────────────────────────────────────────────
    logger.info(
        "Query pipeline: reranking %d candidates → top-%d …",
        len(candidates),
        top_n,
    )
    try:
        reranked = _reranker.rerank(
            query=request.query,
            candidates=candidates,
            top_n=top_n,
        )
    except Exception as exc:
        logger.warning(
            "Reranking failed (%s) — falling back to raw retrieval results.", exc
        )
        # Graceful degradation: use the top-n retrieval results without reranking
        reranked = candidates[:top_n]

    # ── 3. Citation-grounded generation ──────────────────────────────────────
    logger.info(
        "Query pipeline: generating answer from %d context chunks …", len(reranked)
    )
    try:
        result: GenerationResult = _generator.generate(
            query=request.query,
            context_chunks=reranked,
        )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation failed: {exc}",
        ) from exc

    # ── 4. Hallucination detection ────────────────────────────────────────────
    hallucination_flags: list = []
    confidence: float | None = None

    if request.detect_hallucinations and result.answer:
        logger.info("Query pipeline: running hallucination detection …")
        try:
            hallucination_flags = _detector.detect(
                answer=result.answer,
                citations=result.citations,
                context_chunks=reranked,
            )
            # Overall confidence = fraction of claims that are entailed
            if hallucination_flags:
                entailed = sum(
                    1 for f in hallucination_flags if f.label == "entailment"
                )
                confidence = round(entailed / len(hallucination_flags), 4)
        except Exception as exc:
            logger.warning(
                "Hallucination detection failed (%s) — returning empty flags.", exc
            )

    total_ms = (time.perf_counter() - t_start) * 1000.0

    flagged_count = sum(1 for f in hallucination_flags if f.flagged)
    logger.info(
        "Query pipeline complete: latency=%.1f ms, citations=%d, "
        "hallucination_flags=%d (flagged=%d).",
        total_ms,
        len(result.citations),
        len(hallucination_flags),
        flagged_count,
    )

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        hallucination_flags=hallucination_flags,
        confidence=confidence,
        latency_ms=total_ms,
    )
