"""
backend/app/ingestion/chunker.py
---------------------------------
Three chunking strategies + a factory function.

Strategies
----------
* ``FixedSizeChunker``     – sliding window over token-counted text (512 tok / 50 overlap)
* ``SemanticChunker``      – embeds sentences, splits where cosine similarity drops
* ``SentenceWindowChunker``– 3-sentence chunks + ±2 sentence window stored in metadata

All chunkers accept a ``List[DocumentPage]`` and return a ``List[Chunk]``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Literal

from app.ingestion.parser import DocumentPage


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Atomic retrieval unit produced by a chunker."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys carried forward from DocumentPage:
    #   source, page_num, section
    # metadata keys added by chunkers:
    #   chunk_index  – position of this chunk in the document
    #   strategy     – which chunker produced it
    #   window       – surrounding sentence context (SentenceWindowChunker only)


def _new_id() -> str:
    return str(uuid.uuid4())


# ── FixedSizeChunker ──────────────────────────────────────────────────────────

class FixedSizeChunker:
    """
    Sliding-window chunker based on token counts.

    Parameters
    ----------
    chunk_size:
        Target size of each chunk in tokens (default 512).
    overlap:
        Number of overlapping tokens between consecutive chunks (default 50).
    encoding_name:
        Tiktoken encoding to use for token counting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = None
        self._encoding_name = encoding_name

    def _get_encoder(self):
        if self._enc is None:
            try:
                import tiktoken  # type: ignore[import]
                self._enc = tiktoken.get_encoding(self._encoding_name)
            except ImportError:
                self._enc = None  # fall back to word-split approximation
        return self._enc

    def _tokenize(self, text: str) -> List[int]:
        enc = self._get_encoder()
        if enc is not None:
            return enc.encode(text)
        # rough fallback: split on whitespace
        return text.split()  # type: ignore[return-value]

    def _decode(self, tokens) -> str:
        enc = self._get_encoder()
        if enc is not None:
            return enc.decode(tokens)
        return " ".join(tokens)

    def chunk(self, pages: List[DocumentPage]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in pages:
            tokens = self._tokenize(page.content)
            start = 0

            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                window_tokens = tokens[start:end]
                text = self._decode(window_tokens)

                chunks.append(
                    Chunk(
                        chunk_id=_new_id(),
                        text=text,
                        metadata={
                            **page.metadata,
                            "chunk_index": chunk_index,
                            "strategy": "fixed",
                        },
                    )
                )
                chunk_index += 1

                if end == len(tokens):
                    break
                start = end - self.overlap  # slide back by overlap

        return chunks


# ── SemanticChunker ───────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits text where the cosine similarity between adjacent sentences drops
    below ``similarity_threshold``.

    Uses ``sentence-transformers`` for embeddings.

    Parameters
    ----------
    model_name:
        Sentence-transformers model used for embedding sentences.
    similarity_threshold:
        Cosine-similarity cutoff below which a new chunk is started.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naive sentence splitter – replace with NLTK if available."""
        import re
        # Split on '.', '!', '?' followed by whitespace or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]

    def chunk(self, pages: List[DocumentPage]) -> List[Chunk]:
        import numpy as np  # type: ignore[import]

        model = self._get_model()
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in pages:
            sentences = self._split_sentences(page.content)
            if not sentences:
                continue

            if len(sentences) == 1:
                chunks.append(
                    Chunk(
                        chunk_id=_new_id(),
                        text=sentences[0],
                        metadata={**page.metadata, "chunk_index": chunk_index, "strategy": "semantic"},
                    )
                )
                chunk_index += 1
                continue

            embeddings = model.encode(sentences, show_progress_bar=False)

            # Cosine similarity between consecutive embeddings
            def _cosine(a, b):
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return 1.0
                return float(np.dot(a, b) / denom)

            buffer: List[str] = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = _cosine(embeddings[i - 1], embeddings[i])
                if sim < self.similarity_threshold:
                    # flush buffer → new chunk
                    chunks.append(
                        Chunk(
                            chunk_id=_new_id(),
                            text=" ".join(buffer),
                            metadata={
                                **page.metadata,
                                "chunk_index": chunk_index,
                                "strategy": "semantic",
                            },
                        )
                    )
                    chunk_index += 1
                    buffer = [sentences[i]]
                else:
                    buffer.append(sentences[i])

            # flush remainder
            if buffer:
                chunks.append(
                    Chunk(
                        chunk_id=_new_id(),
                        text=" ".join(buffer),
                        metadata={
                            **page.metadata,
                            "chunk_index": chunk_index,
                            "strategy": "semantic",
                        },
                    )
                )
                chunk_index += 1

        return chunks


# ── SentenceWindowChunker ─────────────────────────────────────────────────────

class SentenceWindowChunker:
    """
    Groups text into ``window_size``-sentence chunks.  Each chunk also stores
    a surrounding context window (±``context_sentences`` sentences) in
    ``metadata["window"]`` for use during answer generation.

    Parameters
    ----------
    window_size:
        Number of sentences per chunk core (default 3).
    context_sentences:
        How many sentences to include on each side of the core (default 2).
    """

    def __init__(self, window_size: int = 3, context_sentences: int = 2) -> None:
        self.window_size = window_size
        self.context_sentences = context_sentences

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]

    def chunk(self, pages: List[DocumentPage]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in pages:
            sentences = self._split_sentences(page.content)
            if not sentences:
                continue

            # Slide in steps of window_size (no overlap on core text)
            step = self.window_size
            for start in range(0, len(sentences), step):
                core = sentences[start: start + self.window_size]
                core_text = " ".join(core)

                # Context window (may extend beyond page boundaries — clamp)
                ctx_start = max(0, start - self.context_sentences)
                ctx_end = min(len(sentences), start + self.window_size + self.context_sentences)
                window_text = " ".join(sentences[ctx_start:ctx_end])

                chunks.append(
                    Chunk(
                        chunk_id=_new_id(),
                        text=core_text,
                        metadata={
                            **page.metadata,
                            "chunk_index": chunk_index,
                            "strategy": "sentence_window",
                            "window": window_text,
                        },
                    )
                )
                chunk_index += 1

        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

ChunkingStrategy = Literal["fixed", "semantic", "sentence_window"]


def get_chunker(
    strategy: ChunkingStrategy = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.75,
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """
    Return the appropriate chunker instance for the given *strategy*.

    Parameters
    ----------
    strategy:
        One of ``"fixed"``, ``"semantic"``, or ``"sentence_window"``.
    chunk_size:
        Token budget per chunk (used by ``FixedSizeChunker``).
    chunk_overlap:
        Overlap in tokens between consecutive chunks (used by ``FixedSizeChunker``).
    similarity_threshold:
        Cosine-similarity cutoff (used by ``SemanticChunker``).
    embedding_model:
        Model name for sentence embeddings (used by ``SemanticChunker``).
    """
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    if strategy == "semantic":
        return SemanticChunker(
            model_name=embedding_model,
            similarity_threshold=similarity_threshold,
        )
    if strategy == "sentence_window":
        return SentenceWindowChunker()
    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Choose one of: 'fixed', 'semantic', 'sentence_window'."
    )
