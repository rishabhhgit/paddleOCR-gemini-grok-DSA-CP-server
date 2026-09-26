"""
Central application configuration.

All values are loaded from environment variables (see .env.example).
Nothing here should be hardcoded to a specific deployment domain.
"""
import os
from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _strip_whitespace(cls, data):
        """
        Normalise every env var before it is assigned to a field.

        Two things happen here, regardless of source (real environment
        variable, .env file, or a host's dashboard secret):

        1. Leading/trailing whitespace is stripped, so a stray pasted
           space in e.g. GEMINI_API_KEY or PUBLIC_BASE_URL can't
           silently turn into an invalid key or a malformed URL like
           "onrender.com /v1/...".
        2. Values that are empty *after* stripping are dropped
           entirely, so the field falls back to its declared default.
           Pasting a whole .env.example into a host's dashboard (or
           leaving a variable declared but unset) otherwise yields "" —
           which pydantic cannot coerce into an int/bool field and turns
           into a startup crash rather than "use the default".
           For str fields an explicit empty value is indistinguishable
           from the default "", so dropping is never a behaviour change.

        Only strips values that are actually strings — non-string
        values (if any ever appear) pass through untouched.
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                cleaned[key] = value
            return cleaned
        return data

    # --- Admin ---
    APP_ADMIN_PASSWORD: str = "change-this"
    ADMIN_SESSION_SECRET: str = "dev-session-secret-change-this"
    ADMIN_SESSION_TTL_SECONDS: int = 3600

    # --- Public provider identity ---
    PUBLIC_BASE_URL: str = ""
    PUBLIC_MODEL_NAME: str = "dsa-solver"
    PUBLIC_PROVIDER_NAME: str = "multi-model-dsa"

    # --- PaddleOCR (runs locally in this process; no API key, no
    # external network call) ---
    PADDLE_OCR_LANG: str = "en"
    # "cpu", "gpu", "gpu:0", etc. — see PaddleOCR's device docs.
    PADDLE_OCR_DEVICE: str = "cpu"
    # Explicit model names. Setting these (instead of letting `lang`
    # pick them) is what lets us use the *mobile* detection model: the
    # default PP-OCRv5_server_det has a much bigger backbone, and its
    # peak activation memory during load/inference is enough to get the
    # process OOM-killed on a 2 GB host (Vercel Hobby's fixed ceiling).
    # NOTE: paddleocr ignores `lang` whenever any model name is set, so
    # BOTH names must be given together. Set PADDLE_OCR_DET_MODEL=lang to
    # hand model selection back to PADDLE_OCR_LANG.
    PADDLE_OCR_DET_MODEL: str = "PP-OCRv5_mobile_det"
    PADDLE_OCR_REC_MODEL: str = "en_PP-OCRv5_mobile_rec"
    # Caps the side length handed to the detection model. "max" means
    # the *longest* side never exceeds the limit, so a phone-photo
    # screenshot can't blow up the model's activation memory.
    PADDLE_OCR_DET_LIMIT_TYPE: str = "max"
    PADDLE_OCR_DET_LIMIT_SIDE_LEN: int = 2000
    PADDLE_OCR_USE_TEXTLINE_ORIENTATION: bool = True
    PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY: bool = False
    PADDLE_OCR_USE_DOC_UNWARPING: bool = False
    # Number of long-lived PaddleOCR engine instances kept warm in
    # memory, i.e. how many screenshots can be OCR'd truly
    # concurrently. Each instance holds its own copy of the model
    # weights in memory. Defaults to 1 because two copies is enough to
    # blow the memory ceiling on small hosts (Vercel Hobby hard-caps
    # functions at 2 GB, which is not configurable). Raise it if you
    # have real RAM/GPU headroom and multi-image traffic to serve.
    OCR_MAX_CONCURRENCY: int = 1
    # Eagerly load the PaddleOCR engine pool at process startup
    # instead of on the first incoming request, so the first real
    # user isn't the one who pays the model load/download cost (see
    # app/services/paddle_ocr.py:warm_up, called from app/main.py).
    PRELOAD_OCR_ON_STARTUP: bool = True

    # --- Gemini solver (server-side only, never exposed to clients) ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = ""
    # Mirrors GROK_BASE_URL. Point it at a proxy or a recording mock to
    # inspect exactly what is sent to the model (the local end-to-end
    # check in scripts/solve_screenshots.py relies on this); normal
    # deployments leave it alone.
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"

    # --- Grok solver (server-side only, never exposed to clients) ---
    GROK_API_KEY: str = ""
    GROK_MODEL: str = ""
    GROK_BASE_URL: str = "https://api.x.ai/v1"

    # --- Consensus / verification behaviour ---
    # Which model arbitrates when Gemini's and Grok's independent
    # solutions disagree after cross-verification: "gemini" or "grok".
    # Only used as a last-resort tiebreaker (see services/consensus_solver.py).
    TIEBREAKER_PROVIDER: str = "gemini"
    # If true, run the (more expensive) full cross-verification pipeline
    # even when both models already agree token-for-token. Off by default
    # since identical output from two independent models is already a
    # strong correctness signal.
    ALWAYS_VERIFY: bool = False

    # --- Image handling ---
    # Total screenshots (inline images + previously-OCR'd results echoed
    # back through `ocr_results`) accepted by ONE /v1/chat/completions
    # call. This is a *solve* limit, not a transport limit: with the
    # two-phase flow (POST /v1/ocr per batch, then one solve request) the
    # payload is text and stays tiny regardless of image count.
    MAX_IMAGES: int = 20
    MAX_IMAGE_SIZE_MB: int = 10
    # Screenshots accepted by ONE POST /v1/ocr call. Deliberately small:
    # Vercel caps request bodies at 4.5 MB, and base64 inflates by 4/3,
    # so a batch has to leave room for the images themselves. 4 x ~1 MB
    # stays comfortably inside that.
    MAX_OCR_BATCH_IMAGES: int = 4
    # Total *decoded* image bytes accepted by one POST /v1/ocr call.
    # This is the real constraint behind MAX_OCR_BATCH_IMAGES: base64
    # encoding adds 33%, so 3 MB of image data is ~4 MB on the wire —
    # under Vercel's 4.5 MB body cap with headroom. Exceeding that cap
    # fails at the platform with an opaque 413 before our code ever
    # runs, so it is checked here where the client gets an actionable
    # message instead. Raise it if your host has no request-body limit
    # (self-hosted Docker has none).
    MAX_OCR_BATCH_BYTES: int = 3_000_000
    # Hard ceiling on the assembled prompt handed to Gemini/Grok, in
    # characters (~0.75 chars/token). Rejected locally with a clear 413
    # rather than letting the provider return an opaque context-length
    # error after the OCR work is already done.
    MAX_PROMPT_CHARS: int = 250_000

    # --- Timeouts ---
    # Per upstream model call (Gemini or Grok), applied independently to
    # each request httpx makes. A hard problem with a long, detailed
    # answer can legitimately take several minutes to generate — 90 s
    # was cutting those off mid-answer (httpx.ReadTimeout -> the call is
    # counted as failed, so you'd silently get the *other* provider's
    # answer, or nothing if both timed out). 540 s (9 min) gives a single
    # call plenty of room without being effectively infinite.
    #
    # There are at most two SEQUENTIAL rounds per request (solve, then —
    # only if the two providers disagree — arbitration; see
    # consensus_solver.py), so the worst case is OCR time plus up to
    # 2 x REQUEST_TIMEOUT_SECONDS. At 540 s that's up to ~18 minutes,
    # which is longer than the total request time this server itself
    # enforces anywhere else. If you deploy on Vercel (Dockerfile.vercel),
    # the PLATFORM also kills the whole request at its own function
    # duration limit regardless of this setting (300 s by default, up to
    # 800 s on Pro/Enterprise, or up to 1800 s with the "extended max
    # duration" beta enabled) — raising REQUEST_TIMEOUT_SECONDS alone
    # does nothing there unless you also raise Project Settings ->
    # Functions -> Function Max Duration to comfortably cover the worst
    # case above, or accept that disagreement-triggered arbitration on a
    # very slow provider response could still hit a platform 504. On
    # self-hosted Docker/docker-compose there is no such outer ceiling.
    REQUEST_TIMEOUT_SECONDS: int = 540

    # --- Misc ---
    ALLOWED_ORIGINS: str = ""  # comma-separated list

    # --- Storage ---
    DATA_DIR: str = "data"  # where the generated API key is persisted

    # --- Provider API key override ---
    # Optional. If set, THIS becomes the client-facing API key instead of
    # one generated on first boot and written to DATA_DIR. Use this on
    # hosts with no persistent disk (e.g. Vercel's serverless functions,
    # whose filesystem is read-only/ephemeral). Set this once as a host
    # env var (e.g. to the output of
    # `python3 -c "import secrets; print('dsa_sk_' + secrets.token_urlsafe(32))"`)
    # and it will survive every redeploy since env vars aren't wiped.
    PROVIDER_API_KEY: str = ""

    @model_validator(mode="after")
    def _default_data_dir_for_vercel(self):
        """
        On Vercel, the deployed project directory is read-only at
        request time -- only `/tmp` is writable, and it's wiped between
        cold starts / separate function instances anyway. If DATA_DIR
        is still at its default ("data", meant for Docker/local use)
        and we detect we're running on Vercel (it sets VERCEL=1
        automatically, no manual config needed), redirect it to /tmp so
        the app doesn't crash trying to create a directory it can't
        write to. This doesn't change the key-stability guarantee:
        the derived-from-ADMIN_SESSION_SECRET key (see
        app/security/api_keys.py) is what actually keeps the API key
        constant across invocations, not this file cache.
        """
        if os.environ.get("VERCEL") and self.DATA_DIR == "data":
            self.DATA_DIR = "/tmp/dsa-practice-solver-data"
        return self

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
