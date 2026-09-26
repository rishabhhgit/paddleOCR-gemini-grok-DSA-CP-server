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
    # Google's OpenAI-compatible surface; the client appends
    # `/chat/completions` and authenticates with a bearer key. Mirrors
    # GROK_BASE_URL — point it at a proxy or a recording mock to inspect
    # exactly what is sent to the model; normal deployments leave it
    # alone (trailing slash optional).
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # --- Grok solver, served by Groq (never exposed to clients) ---
    GROK_API_KEY: str = ""
    GROK_MODEL: str = ""
    # OpenAI-compatible; the client appends `/chat/completions`.
    GROK_BASE_URL: str = "https://api.groq.com/openai/v1"

    # --- Consensus / verification behaviour ---
    # Which model arbitrates when Gemini's and Grok's independent
    # solutions disagree after cross-verification: "gemini" or "grok".
    # Only used as a last-resort tiebreaker (see services/consensus_solver.py).
    TIEBREAKER_PROVIDER: str = "gemini"
    # If true, run the (more expensive) full cross-verification pipeline
    # even when both models already agree token-for-token. On by default:
    # every answer is re-derived and judged by both models rather than
    # being trusted because two independent first passes happened to
    # match, which is the strongest correctness signal this pipeline can
    # buy. Costs 2 extra model calls per request. Set false to trade
    # that safety net for latency/cost when the models already agree.
    ALWAYS_VERIFY: bool = True

    # --- Request limits ---
    # EVERY limit in this file follows one convention: 0 (or negative)
    # means NO LIMIT. The defaults are all 0 — nothing this server owns
    # may refuse a request or truncate what it assembles.
    #
    # Limits imposed by someone else still exist and are not ours to
    # remove: the model's own context window, and, when deployed on
    # Vercel, its 4.5 MB request-body cap and function duration limit.
    # Those surface as provider/platform errors rather than as a locally
    # rejected request. On self-hosted Docker/docker-compose there is no
    # such outer ceiling at all.

    # Total screenshots (inline images + previously-OCR'd results echoed
    # back through `ocr_results`) accepted by ONE /v1/chat/completions
    # call. 0 = any number. This is a *solve* limit, not a transport
    # limit: with the two-phase flow (POST /v1/ocr per batch, then one
    # solve request) the payload is text and stays tiny regardless of
    # image count.
    MAX_IMAGES: int = 0
    # Per-image decoded size in MB. 0 = any size.
    MAX_IMAGE_SIZE_MB: int = 0
    # Screenshots accepted by ONE POST /v1/ocr call. 0 = any number.
    # Keep > 0 only when the host enforces a request-body cap (Vercel:
    # 4.5 MB, and base64 inflates by 4/3) and you would rather give the
    # client an actionable message than let the platform fail it with a
    # bare 413.
    MAX_OCR_BATCH_IMAGES: int = 0
    # Total *decoded* image bytes accepted by one POST /v1/ocr call.
    # 0 = any size. Same Vercel caveat as MAX_OCR_BATCH_IMAGES: 3_000_000
    # is ~4 MB of base64 on the wire, just under Vercel's 4.5 MB cap.
    MAX_OCR_BATCH_BYTES: int = 0
    # Assembled prompt handed to Gemini/Grok, in characters (~0.75
    # chars/token). 0 = any length. Set > 0 only to fail fast with a
    # clear 413 instead of letting the provider return an opaque
    # context-length error after the OCR work is already done.
    MAX_PROMPT_CHARS: int = 0

    # --- Output budget ---
    # Hard ceiling on how many tokens ONE provider call (Gemini or Grok)
    # may generate. 0 = NO LIMIT: each client asks for the largest budget
    # its provider documents (see UNLIMITED_BUDGET in gemini_client.py /
    # grok_client.py), which is as close to "no cap" as an API that
    # requires a number can get. A positive value imposes a cap.
    #
    # An explicit maximum matters. Omitting the field does NOT mean
    # "unlimited": Gemini falls back to a model-dependent default that is
    # often far below its real output limit, and that default is what
    # used to cut solutions off mid-function. Gemini 2.5 makes it worse
    # because THINKING tokens count against max_tokens too — a hard
    # problem can spend the whole default thinking and leave too little
    # budget for the code.
    #
    # A model whose own maximum is lower than what we ask for rejects the
    # request with a 400 naming the cap; the client retries once
    # at the ceiling the model names (or with no budget if it names
    # none), remembers it for that model, and — whatever happens — never
    # returns a response the provider marked as truncated.
    SOLVER_MAX_TOKENS: int = 0

    # --- Timeouts ---
    # Per upstream model call (Gemini or Grok), applied independently to
    # each request httpx makes. 0 = NO TIMEOUT: the call waits however
    # long the model takes. A positive value is a cap in seconds.
    #
    # Timeouts used to cut answers off mid-generation: httpx.ReadTimeout
    # made the call count as failed, so you'd silently get the *other*
    # provider's answer, or nothing at all if both timed out.
    #
    # There are at most two SEQUENTIAL rounds per request (solve, then —
    # only when the two providers disagree — arbitration; see
    # consensus_solver.py). This server enforces no total deadline of its
    # own. The one ceiling that remains is whichever the host imposes:
    # Vercel kills the whole request at its function duration limit (300 s
    # by default, up to 800 s on Pro/Enterprise, or 1800 s with the
    # extended-max-duration beta) no matter what this is set to, so raise
    # Project Settings -> Functions -> Function Max Duration there too.
    # Self-hosted Docker/docker-compose has no such outer ceiling.
    REQUEST_TIMEOUT_SECONDS: int = 0

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
