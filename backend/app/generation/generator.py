"""
backend/app/generation/generator.py
--------------------------------------
Citation-grounded answer generation using the Groq API (llama-3.3-70b-versatile
by default).

The generator:
  1. Formats context chunks into the prompt via ``build_messages``.
  2. Calls the Groq chat-completion endpoint requesting JSON mode.
  3. Parses the JSON response and validates that every cited chunk_id is
     actually present in the supplied context (reject any ghost citations).
  4. Converts valid citations to ``Citation`` Pydantic models and returns a
     ``GenerationResult`` with latency.

Classes
-------
GenerationResult
    Dataclass holding answer, citations, and wall-clock latency.
CitationGenerator
    Wraps the Groq client and exposes ``generate(query, context_chunks)``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List

from app.generation.prompts import build_messages
from app.models.response import Citation
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Output of ``CitationGenerator.generate()``.

    Attributes
    ----------
    answer:
        The LLM-generated answer string (may contain markdown).
    citations:
        Validated ``Citation`` objects whose chunk_ids were present in the
        supplied context.
    latency_ms:
        Wall-clock time from the start of the Groq API call to parse
        completion, in milliseconds.
    raw_chunk_ids:
        The raw list of chunk_ids returned by the model before validation
        (useful for debugging / logging).
    """

    answer: str
    citations: List[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    raw_chunk_ids: List[str] = field(default_factory=list)


# ── CitationGenerator ─────────────────────────────────────────────────────────

class CitationGenerator:
    """
    Generate citation-grounded answers via the Groq LLM API.

    Parameters
    ----------
    api_key:
        Groq API key.  Falls back to the ``LLM_API_KEY`` env variable if the
        ``groq`` client supports it, but explicit passing is preferred.
    model:
        Groq model name.  Defaults to ``llama-3.3-70b-versatile``.
    temperature:
        Sampling temperature.  Use 0 for fully deterministic outputs.
    max_tokens:
        Maximum tokens in the completion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self):
        """Lazily instantiate the Groq client."""
        if self._client is None:
            from groq import Groq  # type: ignore[import]
            self._client = Groq(api_key=self.api_key)
            logger.info("Groq client initialised (model=%s).", self.model)
        return self._client

    @staticmethod
    def _build_context_index(chunks: List[RetrievalResult]) -> dict[str, RetrievalResult]:
        """Map chunk_id → RetrievalResult for O(1) look-ups."""
        return {c.chunk_id: c for c in chunks}

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        """
        Parse the LLM response as JSON, stripping markdown fences if present.

        Raises
        ------
        ValueError
            If the content cannot be parsed as JSON or is missing required keys.
        """
        # Strip common markdown code fences the model might add
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening and closing fence lines
            inner = [l for l in lines if not l.startswith("```")]
            text = "\n".join(inner).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}\nRaw:\n{content}") from exc

        if "answer" not in data:
            raise ValueError(f"LLM JSON missing 'answer' key. Got keys: {list(data.keys())}")
        if "citations" not in data:
            raise ValueError(f"LLM JSON missing 'citations' key. Got keys: {list(data.keys())}")

        return data

    @staticmethod
    def _validate_citations(
        raw_ids: List[str],
        context_index: dict[str, RetrievalResult],
    ) -> List[Citation]:
        """
        Reject any chunk_id not present in the supplied context and convert
        the remaining ones to ``Citation`` Pydantic models.

        This is the security / hallucination guard: the model must not be
        allowed to cite documents it was not given.

        Parameters
        ----------
        raw_ids:
            chunk_ids returned by the LLM.
        context_index:
            Map of chunk_id → RetrievalResult for all chunks in the context.

        Returns
        -------
        List[Citation]
            Only citations whose chunk_id exists in ``context_index``.
        """
        citations: List[Citation] = []
        seen: set[str] = set()

        for cid in raw_ids:
            if not isinstance(cid, str):
                logger.warning("Ignoring non-string citation: %r", cid)
                continue
            if cid in seen:
                continue
            seen.add(cid)

            if cid not in context_index:
                logger.warning(
                    "LLM cited chunk_id '%s' which is not in the provided context — REJECTED.",
                    cid,
                )
                continue

            chunk = context_index[cid]
            meta = chunk.metadata or {}

            citations.append(
                Citation(
                    chunk_id=cid,
                    source=str(meta.get("source", "unknown")),
                    page_num=meta.get("page_num"),
                    section=meta.get("section"),
                    snippet=chunk.text[:200].strip(),
                )
            )

        return citations

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
    ) -> GenerationResult:
        """
        Generate a citation-grounded answer for *query* given *context_chunks*.

        Steps:
          1. Format chunks into the prompt.
          2. Call Groq with JSON mode enabled.
          3. Parse and validate the JSON response.
          4. Reject any cited chunk_id not in the provided context.
          5. Return a ``GenerationResult`` with latency.

        Parameters
        ----------
        query:
            The natural-language question.
        context_chunks:
            Reranked list of ``RetrievalResult`` objects; these are the only
            chunks the model is allowed to cite.

        Returns
        -------
        GenerationResult
        """
        client = self._get_client()
        context_index = self._build_context_index(context_chunks)
        messages = build_messages(query, context_chunks)

        logger.info(
            "CitationGenerator: calling Groq model='%s' with %d context chunks …",
            self.model,
            len(context_chunks),
        )

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_content = response.choices[0].message.content or ""
        logger.debug("Groq raw response (%d chars): %.200s…", len(raw_content), raw_content)

        # Parse JSON
        try:
            data = self._parse_llm_json(raw_content)
        except ValueError as exc:
            logger.error("Failed to parse LLM JSON response: %s", exc)
            # Return a graceful degradation rather than 500-ing the whole request
            return GenerationResult(
                answer="I was unable to generate a structured response. Please try again.",
                citations=[],
                latency_ms=latency_ms,
                raw_chunk_ids=[],
            )

        answer: str = data.get("answer", "")
        raw_ids: List[str] = data.get("citations", [])

        # Validate citations — reject any chunk_id not in our context
        valid_citations = self._validate_citations(raw_ids, context_index)

        logger.info(
            "CitationGenerator: answer generated (latency=%.1f ms, "
            "raw_citations=%d, valid_citations=%d).",
            latency_ms,
            len(raw_ids),
            len(valid_citations),
        )

        return GenerationResult(
            answer=answer,
            citations=valid_citations,
            latency_ms=latency_ms,
            raw_chunk_ids=raw_ids,
        )
