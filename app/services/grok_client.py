"""
Thin Grok (xAI) API client.

xAI's API is OpenAI-compatible (POST {base_url}/chat/completions with
{model, messages}), so this mirrors gemini_client.py's shape but speaks
that dialect instead. Exposes a single generic `call_grok` used both
for the independent "solve" step and (when Grok is configured as the
tiebreaker) the final arbitration step. The model name (GROK_MODEL) is
only ever used internally and is never echoed back to the client.
"""
from __future__ import annotations

import httpx

from app.config import Settings


class GrokError(Exception):
    """Raised when the upstream Grok provider fails or is unreachable."""


def _build_payload(model: str, system_prompt: str, user_content: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }


async def call_grok(
    settings: Settings,
    system_prompt: str,
    user_content: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Sends a single-turn request to Grok and returns the text content."""
    if not settings.GROK_MODEL:
        raise GrokError("GROK_MODEL is not configured on the server.")
    if not settings.GROK_API_KEY:
        raise GrokError("GROK_API_KEY is not configured on the server.")

    base_url = settings.GROK_BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = _build_payload(settings.GROK_MODEL, system_prompt, user_content)
    headers = {"Authorization": f"Bearer {settings.GROK_API_KEY}"}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        try:
            resp = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise GrokError(f"Failed to reach Grok: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code >= 400:
        raise GrokError(f"Grok returned HTTP {resp.status_code}.")

    try:
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise GrokError(f"Unexpected Grok response format: {exc}") from exc

    if not text:
        raise GrokError("Grok returned an empty response.")

    return text
