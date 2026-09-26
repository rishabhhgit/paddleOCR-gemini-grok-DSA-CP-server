from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings
from app.security.api_keys import get_api_key_store


def _invalid_key_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": {
                "message": "Invalid authentication credentials.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )


async def require_backend_api_key(authorization: str | None = Header(default=None)) -> str:
    """Validates `Authorization: Bearer <backend-generated key>`.

    Never accepts GEMINI_API_KEY or GROK_API_KEY as a valid credential —
    those are never compared against here at all, only the independently
    generated/stored backend key is. (PaddleOCR has no API key at all;
    it runs locally on this server.)
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _invalid_key_error()

    candidate = authorization.split(" ", 1)[1].strip()
    settings = get_settings()
    store = get_api_key_store(settings.DATA_DIR, settings.PROVIDER_API_KEY, settings.ADMIN_SESSION_SECRET)

    if not store.verify(candidate):
        raise _invalid_key_error()

    return candidate
