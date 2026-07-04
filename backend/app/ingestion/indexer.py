"""
backend/app/ingestion/indexer.py
---------------------------------
Dual-index manager: dense (ChromaDB) + sparse (BM25).

Classes
-------
IndexManager
    High-level facade that owns both indexes and exposes ``add_documents()``,
    ``clear()``, and ``get_stats()``.

Helpers
-------
build_dense_index(chunks)   – embed with all-MiniLM-L6-v2, upsert into ChromaDB
build_sparse_index(chunks)  – build BM25Okapi, pickle to disk
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_CHROMA_COLLECTION = "ask_my_docs"
_BM25_PICKLE_NAME = "bm25_index.pkl"


# ── Dense index ───────────────────────────────────────────────────────────────

def build_dense_index(
    chunks: List[Chunk],
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_dir: str = "./data/chroma",
) -> None:
    """
    Embed *chunks* with *embedding_model* and upsert them into a persistent
    ChromaDB collection.

    Parameters
    ----------
    chunks:
        The list of chunks to index.
    embedding_model:
        SentenceTransformers model name used to produce embeddings.
    persist_dir:
        Directory where ChromaDB persists its data.
    """
    if not chunks:
        logger.warning("build_dense_index called with empty chunk list — skipping.")
        return

    import chromadb  # type: ignore[import]
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    logger.info("Loading embedding model '%s' …", embedding_model)
    model = SentenceTransformer(embedding_model)

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    logger.info("Upserted %d chunks into ChromaDB collection '%s'.", len(chunks), _CHROMA_COLLECTION)


# ── Sparse index ──────────────────────────────────────────────────────────────

def build_sparse_index(
    chunks: List[Chunk],
    persist_dir: str = "./data/chroma",
) -> None:
    """
    Build a ``BM25Okapi`` index from *chunks* and pickle it to *persist_dir*.

    The pickle file stores a dict:
    ``{"bm25": BM25Okapi, "chunk_ids": List[str], "texts": List[str]}``

    Parameters
    ----------
    chunks:
        The list of chunks to index.
    persist_dir:
        Directory where the BM25 pickle file is written.
    """
    if not chunks:
        logger.warning("build_sparse_index called with empty chunk list — skipping.")
        return

    from rank_bm25 import BM25Okapi  # type: ignore[import]

    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    pickle_path = persist_path / _BM25_PICKLE_NAME

    payload = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in chunks],
        "texts": [c.text for c in chunks],
        "metadatas": [c.metadata for c in chunks],
    }

    with open(pickle_path, "wb") as fh:
        pickle.dump(payload, fh)

    logger.info("BM25 index with %d chunks pickled to '%s'.", len(chunks), pickle_path)


# ── IndexManager ──────────────────────────────────────────────────────────────

class IndexManager:
    """
    Unified manager for the dense (ChromaDB) and sparse (BM25) indexes.

    Parameters
    ----------
    embedding_model:
        SentenceTransformers model used for dense embeddings.
    persist_dir:
        Directory for both ChromaDB data and the BM25 pickle.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./data/chroma",
    ) -> None:
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self._chroma_client: Optional[object] = None
        self._collection: Optional[object] = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_collection(self):
        """Lazily initialise ChromaDB client and return the collection."""
        if self._collection is None:
            import chromadb  # type: ignore[import]
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Chunk]) -> None:
        """
        Add *chunks* to both the dense and sparse indexes.

        **Note**: The BM25 index is rebuilt from scratch each time (because
        BM25Okapi is not incrementally updatable).  For large corpora this may
        be slow, but it guarantees correctness.
        """
        if not chunks:
            logger.warning("add_documents called with empty chunk list.")
            return

        # 1. Dense
        build_dense_index(
            chunks,
            embedding_model=self.embedding_model,
            persist_dir=self.persist_dir,
        )

        # 2. Sparse — we need *all* existing chunks + the new ones so that
        #    BM25 covers the entire corpus.  Load existing texts from ChromaDB.
        collection = self._get_collection()
        existing_result = collection.get(include=["documents", "metadatas"])

        existing_chunks: List[Chunk] = []
        if existing_result["ids"]:
            for cid, doc, meta in zip(
                existing_result["ids"],
                existing_result["documents"],
                existing_result["metadatas"],
            ):
                existing_chunks.append(Chunk(chunk_id=cid, text=doc, metadata=meta))

        # Merge: existing (excluding newly upserted) + new
        new_ids = {c.chunk_id for c in chunks}
        merged = [c for c in existing_chunks if c.chunk_id not in new_ids] + chunks

        build_sparse_index(merged, persist_dir=self.persist_dir)

    def clear(self) -> None:
        """Delete all chunks from both indexes."""
        import chromadb  # type: ignore[import]

        # Dense
        try:
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(_CHROMA_COLLECTION)
            self._collection = None
            logger.info("ChromaDB collection '%s' deleted.", _CHROMA_COLLECTION)
        except Exception as exc:
            logger.warning("Failed to delete ChromaDB collection: %s", exc)

        # Sparse
        pickle_path = Path(self.persist_dir) / _BM25_PICKLE_NAME
        if pickle_path.exists():
            pickle_path.unlink()
            logger.info("BM25 pickle deleted from '%s'.", pickle_path)

    def get_stats(self) -> dict:
        """
        Return a dict with index statistics.

        Returns
        -------
        dict with keys:
            * ``dense_chunk_count``   – number of vectors in ChromaDB
            * ``sparse_index_exists`` – whether the BM25 pickle file exists
            * ``persist_dir``         – path to the storage directory
        """
        collection = self._get_collection()
        dense_count = collection.count()

        pickle_path = Path(self.persist_dir) / _BM25_PICKLE_NAME

        return {
            "dense_chunk_count": dense_count,
            "sparse_index_exists": pickle_path.exists(),
            "persist_dir": str(self.persist_dir),
        }
