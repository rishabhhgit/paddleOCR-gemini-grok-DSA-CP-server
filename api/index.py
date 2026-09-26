"""
Vercel serverless entry point.

Vercel's Python runtime auto-detects an ASGI application exported as
`app` from a file under `api/` and serves it directly -- no adapter
(e.g. Mangum, which is AWS Lambda-specific) is needed.

All incoming requests are routed here via the rewrite rule in
`vercel.json`, so this single file serves every route the app defines
(/health, /admin, /v1/chat/completions, etc.) exactly as it would under
uvicorn on any other host.
"""
import sys
from pathlib import Path

# Make the project root (which contains the `app` package) importable.
# Vercel invokes this file directly, so it isn't automatically on
# sys.path the way it would be when run as part of a normal package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402  (import must follow the sys.path fix above)
