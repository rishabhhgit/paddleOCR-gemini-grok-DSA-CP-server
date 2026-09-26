"""
Safe logging.

Only ever log the metadata fields explicitly allowed below. Never pass
Authorization headers, API keys (backend, Gemini, or Grok), base64
image data, or full request/response bodies to `log_request_event`.
"""
from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger("dsa_practice_solver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def log_request_event(
    request_id: str,
    *,
    num_images: int = 0,
    ocr_duration_ms: float | None = None,
    solver_duration_ms: float | None = None,
    total_latency_ms: float | None = None,
    success: bool,
    http_status: int,
) -> None:
    logger.info(
        "request_id=%s num_images=%d ocr_ms=%s solver_ms=%s total_ms=%s success=%s status=%d",
        request_id,
        num_images,
        f"{ocr_duration_ms:.1f}" if ocr_duration_ms is not None else "-",
        f"{solver_duration_ms:.1f}" if solver_duration_ms is not None else "-",
        f"{total_latency_ms:.1f}" if total_latency_ms is not None else "-",
        success,
        http_status,
    )


class Timer:
    """Small helper: `with Timer() as t: ...` then `t.elapsed_ms`."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False
