"""
app.api — FastAPI route handlers

Routers (included in app.main):
    routes_ingest  → POST /ingest
    routes_query   → POST /query
    routes_eval    → POST /eval/run, GET /eval/results, GET /eval/ablation/{experiment}
"""
from app.api.routes_ingest import router as ingest_router

__all__ = ["ingest_router"]
