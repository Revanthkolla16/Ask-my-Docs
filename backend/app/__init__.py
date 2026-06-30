"""
ask-my-docs · backend application package

Entry points:
    app.main      — FastAPI application instance
    app.config    — settings singleton (loaded from .env)
"""
from app.config import settings