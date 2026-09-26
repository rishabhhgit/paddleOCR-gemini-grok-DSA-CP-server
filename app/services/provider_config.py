from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.models.provider import ProviderConfig
from app.security.api_keys import EnvAPIKeyStore, FileAPIKeyStore


def resolve_public_base_url(settings: Settings, request: Request | None) -> str:
    if settings.PUBLIC_BASE_URL:
        return settings.PUBLIC_BASE_URL.rstrip("/")
    if request is not None:
        return str(request.base_url).rstrip("/")
    return "https://YOUR-DOMAIN.com"


def build_provider_config(
    settings: Settings,
    key_store: "FileAPIKeyStore | EnvAPIKeyStore",
    request: Request | None = None,
    reveal_key: bool = True,
) -> ProviderConfig:
    base_url = resolve_public_base_url(settings, request)
    api_key = key_store.get_or_create() if reveal_key else key_store.get_masked()

    return ProviderConfig(
        name=settings.PUBLIC_PROVIDER_NAME,
        model=settings.PUBLIC_MODEL_NAME,
        endpoint=f"{base_url}/v1/chat/completions",
        api_key=api_key,
        status="active",
    )
