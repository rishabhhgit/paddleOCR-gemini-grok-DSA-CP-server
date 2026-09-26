"""
Thin Gemini API client.

Exposes a single generic `call_gemini` used both for the independent
"solve" step and (when Gemini is configured as the tiebreaker) the
final arbitration step. The model name (GEMINI_MODEL) is only ever
used internally and is never echoed back to the client.
"""
from __future__ import annotations

import httpx

from app.config import Settings

GEMINI_URL_TEMPLATE = "{base}/v1beta/models/{model}:generateContent"


class GeminiError(Exception):
    """Raised when the upstream Gemini provider fails or is unreachable."""


def _build_payload(system_prompt: str, user_content: str) -> dict:
    return {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {"temperature": 0.2},
    }


def _endpoint(settings: Settings) -> str:
    return GEMINI_URL_TEMPLATE.format(base=settings.GEMINI_BASE_URL.rstrip("/"), model=settings.GEMINI_MODEL)


async def call_gemini(
    settings: Settings,
    system_prompt: str,
    user_content: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Sends a single-turn request to Gemini and returns the text content."""
    if not settings.GEMINI_MODEL:
        raise GeminiError("GEMINI_MODEL is not configured on the server.")
    if not settings.GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not configured on the server.")

    url = _endpoint(settings)
    payload = _build_payload(system_prompt, user_content)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        try:
            resp = await client.post(
                url,
                params={"key": settings.GEMINI_API_KEY},
                json=payload,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise GeminiError(f"Failed to reach Gemini: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code >= 400:
        raise GeminiError(f"Gemini returned HTTP {resp.status_code}.")

    try:
        data = resp.json()
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini response format: {exc}") from exc

    if not text:
        raise GeminiError("Gemini returned an empty response.")

    return text
