from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import admin, chat_completions, health, ocr, provider
from app.config import get_settings
from app.models.openai import make_error
from app.security.api_keys import get_api_key_store
from app.services.paddle_ocr import warm_up as warm_up_ocr

logger = logging.getLogger("dsa_practice_solver")

app = FastAPI(title="DSA Practice Solver (Gemini + Grok consensus)", version="1.0.0")

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat_completions.router)
app.include_router(ocr.router)
app.include_router(admin.router)
app.include_router(provider.router)


@app.on_event("startup")
async def on_startup() -> None:
    # Generate the backend API key on first-ever startup; on subsequent
    # restarts the persisted key is loaded instead (see security/api_keys.py).
    # Read settings fresh (not the module-level `settings` captured at
    # import time) so this stays correct if settings are reloaded, e.g.
    # between test runs that use different DATA_DIR values.
    current_settings = get_settings()
    store = get_api_key_store(current_settings.DATA_DIR, current_settings.PROVIDER_API_KEY, current_settings.ADMIN_SESSION_SECRET)
    store.get_or_create()
    logger.info("DSA Practice Solver started. Provider API key is ready (not logged).")

    if current_settings.PRELOAD_OCR_ON_STARTUP:
        asyncio.create_task(_warm_up_ocr_safely(current_settings))


async def _warm_up_ocr_safely(settings) -> None:
    """Loads the PaddleOCR engine pool in the background so it doesn't
    delay server startup, but never crashes the app if it fails (e.g.
    paddleocr isn't installed yet, or there's no network on first run
    to download model weights) — the first real OCR request will just
    retry the load and surface a normal 502 if it's still broken."""
    try:
        await warm_up_ocr(settings)
        logger.info("PaddleOCR engine pool warmed up.")
    except Exception:  # noqa: BLE001
        logger.exception("PaddleOCR warm-up failed; will retry lazily on first OCR request.")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # `detail` is already an OpenAI-compatible error dict for most routes;
    # fall back to wrapping plain-string details (e.g. admin auth errors).
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = make_error(str(exc.detail), "invalid_request_error")
    return JSONResponse(status_code=exc.status_code, content=content, headers=getattr(exc, "headers", None) or {})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces or internal details to the client.
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=make_error("An internal error occurred.", "internal_error", code="internal_error"),
    )
