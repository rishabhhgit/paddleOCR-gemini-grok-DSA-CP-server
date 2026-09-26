#!/usr/bin/env python3
"""
Solve a stack of DSA screenshots with one command.

Handles the two-phase flow for you: OCRs the images in small batches
(Vercel caps request bodies at 4.5 MB, so 15-20 screenshots can never
travel in one request), then sends every parsed result back in a single
solve request that Gemini and Grok answer from.

Usage:
    export DSA_API_KEY=dsa_sk_...          # from /admin -> Reveal key
    python3 scripts/solve_screenshots.py shots/*.png -q "Solve in C++17"

Stdlib only — no pip install needed.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib
import sys
import time
import urllib.error
import urllib.request

# Client-side chunking defaults. They no longer mirror the server's
# MAX_OCR_BATCH_IMAGES/MAX_OCR_BATCH_BYTES (both 0 = unlimited by
# default); they stay conservative because a single oversized POST can
# still be refused by a proxy or by Vercel's 4.5 MB body cap. Override
# with --batch-images/--batch-bytes when your host has no such limit.
DEFAULT_BATCH_IMAGES = 4
DEFAULT_BATCH_BYTES = 3_000_000
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def post_json(url: str, payload: dict, api_key: str, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # non-2xx: body still carries the error JSON
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": {"message": body[:500]}}
    except urllib.error.URLError as exc:
        return 0, {"error": {"message": f"Could not reach {url}: {exc.reason}"}}


def error_message(body: dict) -> str:
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("message", json.dumps(err))
    return json.dumps(err) if err else json.dumps(body)[:500]


def to_data_url(path: pathlib.Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise SystemExit(f"Unsupported image type: {path} (need png, jpg, webp)")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def make_batches(paths: list[pathlib.Path], max_images: int, max_bytes: int) -> list[list[pathlib.Path]]:
    """Splits by image count *and* by encoded size, so a batch never
    trips the server's byte budget (which fails as an opaque 413 at
    Vercel's edge otherwise)."""
    batches: list[list[pathlib.Path]] = []
    current: list[pathlib.Path] = []
    current_bytes = 0
    for path in paths:
        size = path.stat().st_size
        if current and (len(current) >= max_images or current_bytes + size > max_bytes):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(path)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="15-20 screenshot files (png/jpg/webp)")
    parser.add_argument("--base-url", default="https://paddle-ocr-gemini-grok-dsa-cp-serve.vercel.app")
    parser.add_argument("--api-key", default="", help="or set DSA_API_KEY")
    parser.add_argument("-q", "--question", default="Solve the problem shown, in C++17, with complexity analysis.")
    parser.add_argument("--model", default="dsa-solver")
    parser.add_argument("--batch-images", type=int, default=DEFAULT_BATCH_IMAGES)
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES)
    parser.add_argument("--timeout", type=float, default=600, help="seconds per HTTP call")
    parser.add_argument(
        "--show-ocr",
        action="store_true",
        help="print the text parsed from each screenshot, so OCR problems are visible",
    )
    parser.add_argument("--ocr-only", action="store_true", help="stop after OCR; skip the solve call")
    parser.add_argument(
        "--independent",
        action="store_true",
        help=(
            "treat screenshots as unrelated items instead of one continuously-"
            "scrolled problem: disables scroll/chrome de-duplication so nothing "
            "is ever merged or dropped. Use this when the images are different "
            "problems, a problem plus unrelated material (e.g. your own code), "
            "or otherwise not one continuous scroll -- the default assumes they "
            "are, and can silently delete real content if they are not."
        ),
    )
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("DSA_API_KEY", "")
    if not api_key:
        raise SystemExit("Provide --api-key or set DSA_API_KEY (from /admin -> Reveal key).")

    paths = [pathlib.Path(p) for p in args.images]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Not found: {', '.join(missing)}")

    base = args.base_url.rstrip("/")
    batches = make_batches(paths, args.batch_images, args.batch_bytes)
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    print(f"{len(paths)} screenshots ({total_mb:.1f} MB) -> {len(batches)} OCR batch(es)", flush=True)

    results: list[dict] = []
    start_index = 0
    for i, batch in enumerate(batches, 1):
        began = time.time()
        batch_offset = start_index
        status, body = post_json(
            f"{base}/v1/ocr",
            {
                "images": [to_data_url(p) for p in batch],
                "start_index": start_index,
                "stitch": not args.independent,
            },
            api_key,
            args.timeout,
        )
        if status != 200:
            print(f"  batch {i}/{len(batches)} failed (HTTP {status}): {error_message(body)}", file=sys.stderr)
            return 1
        results.extend(body["results"])
        start_index = body["next_start_index"]
        unreadable = sum(1 for r in body["results"] if r["unreadable"])
        print(
            f"  batch {i}/{len(batches)}: {body['count']} image(s) in {time.time() - began:.1f}s"
            f" -> screenshots {batch_offset + 1}-{start_index}"
            f"{f', {unreadable} unreadable' if unreadable else ''}",
            flush=True,
        )
        if args.show_ocr:
            for r in body["results"]:
                source = batch[r["index"] - batch_offset].name
                print(f"\n--- screenshot {r['index'] + 1} ({source}) ---")
                print(r["text"] if not r["unreadable"] else "[UNREADABLE]", flush=True)

    if args.ocr_only:
        print(f"\n{len(results)} screenshot(s) parsed; skipping solve (--ocr-only).")
        return 0

    began = time.time()
    status, body = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.question}],
            "ocr_results": results,
            "stitch": not args.independent,
        },
        api_key,
        args.timeout,
    )
    if status != 200:
        print(f"solve failed (HTTP {status}): {error_message(body)}", file=sys.stderr)
        return 1

    print(f"solved in {time.time() - began:.1f}s from {len(results)} parsed screenshots\n", flush=True)
    print(body["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
