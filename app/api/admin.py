from __future__ import annotations

import io
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.models.solver import ExtractedImage
from app.security.admin_auth import (
    clear_session_cookie,
    issue_session_cookie,
    require_admin_session,
    verify_admin_password,
)
from app.security.api_keys import get_api_key_store
from app.services.gemini_client import GeminiError, call_gemini
from app.services.grok_client import GrokError, call_grok
from app.services.paddle_ocr import PaddleOcrError, run_ocr_on_images
from app.services.prompts import SOLVER_SYSTEM_PROMPT
from app.services.provider_config import build_provider_config

router = APIRouter(tags=["admin"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _make_test_image() -> ExtractedImage:
    """A tiny in-memory PNG containing legible text, used only for the
    admin OCR connectivity check. Never persisted to disk."""
    img = Image.new("RGB", (300, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "OCR TEST 12345", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ExtractedImage(mime_type="image/png", data=buf.getvalue())


class LoginRequest(BaseModel):
    password: str


@router.post("/admin/login")
async def admin_login(body: LoginRequest, response: Response):
    if not verify_admin_password(body.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid administrator password.")
    issue_session_cookie(response)
    return {"status": "ok"}


@router.post("/admin/logout", dependencies=[Depends(require_admin_session)])
async def admin_logout(response: Response):
    clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/admin", response_class=FileResponse)
async def admin_page():
    """Serves the admin single-page UI. The page itself performs the
    password login via POST /admin/login before calling any protected
    endpoint — no credentials are embedded in this file."""
    return FileResponse(_STATIC_DIR / "admin.html")


@router.get("/admin/provider", dependencies=[Depends(require_admin_session)])
async def admin_get_provider(request: Request, reveal: bool = False, settings: Settings = Depends(get_settings)):
    store = get_api_key_store(settings.DATA_DIR, settings.PROVIDER_API_KEY, settings.ADMIN_SESSION_SECRET)
    config = build_provider_config(settings, store, request=request, reveal_key=reveal)
    return config


@router.post("/admin/provider/regenerate", dependencies=[Depends(require_admin_session)])
async def admin_regenerate_key(settings: Settings = Depends(get_settings)):
    store = get_api_key_store(settings.DATA_DIR, settings.PROVIDER_API_KEY, settings.ADMIN_SESSION_SECRET)
    try:
        store.regenerate()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return {"status": "ok", "message": "API key regenerated. The previous key is no longer valid."}


_TEST_PROBLEM = "Given an array of n integers, find the maximum subarray sum. n <= 10^5."


@router.post("/admin/test", dependencies=[Depends(require_admin_session)])
async def admin_test_backend(settings: Settings = Depends(get_settings)):
    """Runs a simple text-only round trip against Gemini and Grok
    independently, plus a small in-memory image through the local
    PaddleOCR engine, reporting connectivity/success/latency for each
    without exposing provider credentials. Unlike Gemini/Grok, the
    PaddleOCR check never leaves this server — "connectivity" here
    means "the local engine loaded and ran successfully", which also
    exercises the model-download-and-cache path on a fresh deploy."""
    store = get_api_key_store(settings.DATA_DIR, settings.PROVIDER_API_KEY, settings.ADMIN_SESSION_SECRET)

    result = {
        "authentication": "ok",  # we're already authenticated as admin to reach here
        "gemini_connectivity": "unknown",
        "gemini_success": False,
        "gemini_latency_ms": None,
        "grok_connectivity": "unknown",
        "grok_success": False,
        "grok_latency_ms": None,
        "paddle_ocr_connectivity": "unknown",
        "paddle_ocr_success": False,
        "paddle_ocr_latency_ms": None,
    }

    # --- Gemini check ---
    gemini_start = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            text = await call_gemini(settings, SOLVER_SYSTEM_PROMPT, _TEST_PROBLEM, client=client)
        result["gemini_connectivity"] = "ok"
        result["gemini_success"] = bool(text)
    except GeminiError as exc:
        result["gemini_connectivity"] = "failed"
        result["gemini_error"] = str(exc)
    finally:
        result["gemini_latency_ms"] = round((time.perf_counter() - gemini_start) * 1000, 1)

    # --- Grok check ---
    grok_start = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            text = await call_grok(settings, SOLVER_SYSTEM_PROMPT, _TEST_PROBLEM, client=client)
        result["grok_connectivity"] = "ok"
        result["grok_success"] = bool(text)
    except GrokError as exc:
        result["grok_connectivity"] = "failed"
        result["grok_error"] = str(exc)
    finally:
        result["grok_latency_ms"] = round((time.perf_counter() - grok_start) * 1000, 1)

    # --- PaddleOCR check (local, in-process — no network call) ---
    paddle_start = time.perf_counter()
    try:
        test_image = _make_test_image()
        ocr_results = await run_ocr_on_images(settings, [test_image])
        extracted_text = ocr_results[0].text if ocr_results else ""
        result["paddle_ocr_connectivity"] = "ok"
        result["paddle_ocr_success"] = bool(extracted_text.strip())
        if not result["paddle_ocr_success"]:
            result["paddle_ocr_error"] = "OCR call succeeded but returned no text."
    except PaddleOcrError as exc:
        result["paddle_ocr_connectivity"] = "failed"
        result["paddle_ocr_error"] = str(exc)
    finally:
        result["paddle_ocr_latency_ms"] = round((time.perf_counter() - paddle_start) * 1000, 1)

    # ensure the key store has been initialized even if this is the very
    # first admin action taken after a fresh deploy
    store.get_or_create()
    return result
