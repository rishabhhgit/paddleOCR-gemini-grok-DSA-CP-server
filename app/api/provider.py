from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.config import Settings, get_settings
from app.security.admin_auth import require_admin_session
from app.security.api_keys import get_api_key_store
from app.services.provider_config import build_provider_config

router = APIRouter(tags=["provider"])


@router.get("/api/provider/config", dependencies=[Depends(require_admin_session)])
async def get_provider_config(request: Request, settings: Settings = Depends(get_settings)):
    """Returns the full client-ready provider configuration, including the
    plaintext backend API key. Administrator authentication required —
    this is never publicly readable."""
    store = get_api_key_store(settings.DATA_DIR, settings.PROVIDER_API_KEY, settings.ADMIN_SESSION_SECRET)
    config = build_provider_config(settings, store, request=request, reveal_key=True)
    return config
