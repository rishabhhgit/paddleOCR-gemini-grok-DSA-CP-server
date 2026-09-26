"""
Generation, persistence, and validation of the backend's own API key.

This is the credential that CLIENTS use to authenticate against this
backend's /v1/chat/completions endpoint. It is completely unrelated to
GEMINI_API_KEY or GROK_API_KEY, which never leave the server. (PaddleOCR
runs locally and has no API key of its own.)

Persistence strategy (in priority order):
    1. `PROVIDER_API_KEY` env var, if set — used verbatim. Highest
       priority, full manual control.
    2. Otherwise, the key is deterministically DERIVED from
       `ADMIN_SESSION_SECRET` (an HMAC-SHA256 of a fixed context
       string, keyed by that secret) rather than generated randomly.
       Since `ADMIN_SESSION_SECRET` is already a required setting that
       lives in the host's environment variables (not on disk), this
       means the very same client-facing API key is reconstructed on
       every single startup automatically — no persistent disk and no
       extra configuration step required. This is what keeps the key
       stable across Render free-tier redeploys, which wipe the
       container filesystem each time but never touch env vars.
    3. The derived (or regenerated) key is still cached to a JSON file
       under `settings.DATA_DIR` (default: ./data/provider_key.json)
       purely so repeated reads don't require re-deriving, and so an
       explicit `regenerate()` call can persist a fresh random key for
       as long as the disk survives (real persistent volumes, or
       between restarts that don't wipe disk). If that file is ever
       missing (fresh disk), step 2 reconstructs the same default key
       again, so clients are never permanently locked out by a wipe.

    For multi-instance deployments, replace `FileAPIKeyStore` with a
    shared store (e.g. a database row or a secrets manager entry) that
    all instances read from — the interface (`get_or_create`,
    `regenerate`, `get_masked`, `verify`) is intentionally small so
    this is a drop-in swap.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

_KEY_PREFIX = "dsa_sk_"
_KEY_BYTES = 32  # 256 bits of entropy
_DERIVATION_CONTEXT = b"dsa-practice-solver:provider-api-key:v1"


def _generate_raw_key() -> str:
    """Cryptographically secure, high-entropy key. Never derived from
    any predictable value (domain, timestamp, other provider keys)."""
    return f"{_KEY_PREFIX}{secrets.token_urlsafe(_KEY_BYTES)}"


def _derive_key(admin_session_secret: str) -> str:
    """Deterministic key derived from ADMIN_SESSION_SECRET via HMAC.

    Same secret always produces the same key -- that's the point. The
    fixed context string provides domain separation so this derived
    value can never collide with, or be confused for, a signature
    produced elsewhere using the same secret (e.g. the admin session
    cookie).
    """
    digest = hmac.new(admin_session_secret.encode("utf-8"), _DERIVATION_CONTEXT, hashlib.sha256).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class FileAPIKeyStore:
    """File-cached persistence for the backend API key, with a
    deterministic fallback so the key survives disk wipes automatically.
    """

    def __init__(self, data_dir: str, admin_session_secret: str = ""):
        self._path = Path(data_dir) / "provider_key.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[dict] = None
        self._admin_session_secret = admin_session_secret

    def _load(self) -> Optional[dict]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            return None
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._cache = data
        return data

    def _save(self, raw_key: str) -> None:
        data = {"key_hash": _hash_key(raw_key), "raw_key": raw_key}
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        tmp_path.replace(self._path)
        self._cache = data

    def get_or_create(self) -> str:
        """Return the persisted key. If none is on disk yet (first ever
        boot, or a redeploy wiped an ephemeral filesystem), reconstruct
        the deterministic key from ADMIN_SESSION_SECRET instead of
        minting a random one -- that's what makes the key automatically
        stable across redeploys with zero extra configuration. Only
        falls back to a random key if ADMIN_SESSION_SECRET is somehow
        empty, which shouldn't happen in normal use.
        """
        data = self._load()
        if data is not None:
            return data["raw_key"]
        raw_key = _derive_key(self._admin_session_secret) if self._admin_session_secret else _generate_raw_key()
        self._save(raw_key)
        return raw_key

    def regenerate(self) -> str:
        """Issue a new, random (non-deterministic) key and persist it.
        The previous key stops working immediately for the currently
        running instance. Note: on hosts with no persistent disk, this
        rotation only lasts until the next redeploy wipes the
        filesystem -- at that point `get_or_create` reconstructs the
        deterministic default again (see above), rather than silently
        losing access forever. For a rotation that survives redeploys
        on such hosts, set `PROVIDER_API_KEY` explicitly instead.
        """
        raw_key = _generate_raw_key()
        self._save(raw_key)
        return raw_key

    def get_masked(self) -> str:
        raw = self.get_or_create()
        if len(raw) <= 12:
            return "*" * len(raw)
        return f"{raw[:10]}{'*' * 8}{raw[-4:]}"

    def verify(self, candidate: str) -> bool:
        if not candidate:
            return False
        raw = self.get_or_create()
        return hmac.compare_digest(_hash_key(candidate), _hash_key(raw))


class EnvAPIKeyStore:
    """Fixed-key store backed by an env var instead of disk.

    Used when `PROVIDER_API_KEY` is explicitly set. Takes priority over
    the derived-key default in `FileAPIKeyStore`. There is nothing to
    regenerate here — the key only changes if you edit the env var
    yourself and redeploy.
    """

    def __init__(self, raw_key: str):
        self._raw_key = raw_key
        self._hash = _hash_key(raw_key)

    def get_or_create(self) -> str:
        return self._raw_key

    def regenerate(self) -> str:
        raise RuntimeError(
            "PROVIDER_API_KEY is set via environment variable, so the key "
            "can't be regenerated from the admin panel. To rotate it, set "
            "a new value for PROVIDER_API_KEY in your host's environment "
            "variables and redeploy."
        )

    def get_masked(self) -> str:
        raw = self._raw_key
        if len(raw) <= 12:
            return "*" * len(raw)
        return f"{raw[:10]}{'*' * 8}{raw[-4:]}"

    def verify(self, candidate: str) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(_hash_key(candidate), self._hash)


_store: Optional["FileAPIKeyStore | EnvAPIKeyStore"] = None
_store_key_signature: Optional[str] = None


def get_api_key_store(
    data_dir: str = "data",
    provider_api_key: str = "",
    admin_session_secret: str = "",
) -> "FileAPIKeyStore | EnvAPIKeyStore":
    """Return the process-wide key store.

    If `provider_api_key` is non-empty, an env-backed store is used
    (no disk needed, stable across redeploys, fully manual).
    Otherwise, a file-cached store is used whose key is automatically
    derived from `admin_session_secret` whenever the cache file is
    missing -- so it's stable across redeploys too, with no extra
    configuration needed.
    """
    global _store, _store_key_signature
    signature = f"env:{provider_api_key}" if provider_api_key else f"file:{data_dir}:{admin_session_secret}"
    if _store is None or _store_key_signature != signature:
        if provider_api_key:
            _store = EnvAPIKeyStore(provider_api_key)
        else:
            _store = FileAPIKeyStore(data_dir, admin_session_secret)
        _store_key_signature = signature
    return _store
