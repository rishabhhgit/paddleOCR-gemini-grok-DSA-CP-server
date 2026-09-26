"""
Admin authentication.

Administrators authenticate once with APP_ADMIN_PASSWORD via
POST /admin/login, which sets a signed, HTTP-only session cookie.
Subsequent requests to admin endpoints are authenticated via that
cookie — the plaintext admin password is never embedded in frontend
JavaScript and is never written to logs.
"""
from __future__ import annotations

import hmac

from fastapi import Cookie, HTTPException, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import get_settings

_SESSION_COOKIE_NAME = "dsa_admin_session"
_SESSION_VALUE = "admin-authenticated"


def _serializer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.ADMIN_SESSION_SECRET, salt="admin-session")


def verify_admin_password(password: str) -> bool:
    settings = get_settings()
    return hmac.compare_digest(password or "", settings.APP_ADMIN_PASSWORD)


def issue_session_cookie(response: Response) -> None:
    token = _serializer().dumps(_SESSION_VALUE)
    settings = get_settings()
    response.set_cookie(
        key=_SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="strict",
        max_age=settings.ADMIN_SESSION_TTL_SECONDS,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(_SESSION_COOKIE_NAME, path="/")


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {"message": "Administrator authentication required.", "type": "invalid_request_error"}},
    )


async def require_admin_session(dsa_admin_session: str | None = Cookie(default=None)) -> None:
    settings = get_settings()
    if not dsa_admin_session:
        raise _unauthorized()
    try:
        value = _serializer().loads(dsa_admin_session, max_age=settings.ADMIN_SESSION_TTL_SECONDS)
    except (BadSignature, SignatureExpired):
        raise _unauthorized()
    if value != _SESSION_VALUE:
        raise _unauthorized()
