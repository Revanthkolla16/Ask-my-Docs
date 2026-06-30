"""
app.ingestion — document ingestion pipeline

Stages:
    parser  → parse_document(path) → list[DocumentPage]
    chunker → get_chunker(strategy).chunk(pages) → list[Chunk]
    indexer → IndexManager.add_documents(chunks) → dense + sparse index
"""
