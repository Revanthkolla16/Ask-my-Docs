from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    # Startup: preload heavy models / connections here in later milestones
    print("Ask My Docs API starting up…")
    yield
    # Shutdown cleanup
    print("Ask My Docs API shutting down…")


app = FastAPI(
    title="Ask My Docs",
    description="Citation-grounded RAG API with hallucination detection and RAGAS evaluation.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
from app.api import ingest_router
# from app.api import query_router   # Milestone 3
# from app.api import eval_router     # Milestone 5

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
# app.include_router(query_router, prefix="/query", tags=["Query"])
# app.include_router(eval_router, prefix="/eval", tags=["Evaluation"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": app.version}
