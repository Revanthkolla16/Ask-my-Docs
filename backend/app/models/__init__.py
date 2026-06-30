"""
app.models — Pydantic request/response schemas

Exports:
    IngestRequest   — body for POST /ingest
    QueryRequest    — body for POST /query
    Citation        — a single source citation
    HallucinationFlag — per-claim NLI result
    QueryResponse   — full response from POST /query
    IngestResponse  — response from POST /ingest
"""
from app.models.request import IngestRequest, QueryRequest
from app.models.response import (
    Citation,
    HallucinationFlag,
    IngestResponse,
    QueryResponse,
)
