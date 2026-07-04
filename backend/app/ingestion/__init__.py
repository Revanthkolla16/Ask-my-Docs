"""
app.ingestion — document ingestion pipeline

Stages:
    parser  → parse_document(path) → list[DocumentPage]
    chunker → get_chunker(strategy).chunk(pages) → list[Chunk]
    indexer → IndexManager.add_documents(chunks) → dense + sparse index
"""
from app.ingestion.parser import parse_document, DocumentPage
from app.ingestion.chunker import get_chunker, Chunk
from app.ingestion.indexer import IndexManager

__all__ = ["parse_document", "DocumentPage", "get_chunker", "Chunk", "IndexManager"]
