# DSA Practice Solver — PaddleOCR + Gemini + Grok Consensus

An OpenAI-compatible `/v1/chat/completions` backend for competitive
programming / DSA practice. It takes screenshots (or plain text) of a
problem, OCRs them locally with **PaddleOCR**, then gets **Gemini**
and **Grok** to solve the problem *independently*, cross-checks their
answers against each other, and returns one final, verified,
submission-ready solution.

This is a fork of the Mistral-OCR-based three-provider server that
swaps the OCR stage for **PaddleOCR**, which runs in-process on this
server instead of calling a hosted OCR API. Same client contract
(drop-in for any app that expects an OpenAI-style `endpoint` / `model`
/ `api key`), same Gemini+Grok consensus pipeline — only the OCR stage
changed.

## How a request is solved

```
screenshots ──▶ PaddleOCR (local) ──▶ reconstructed problem text
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                             ▼
                     Gemini solves                 Grok solves
                    (independently)               (independently)
                          │                             │
                          └─────────────┬───────────────┘
                                        ▼
                     same answer? ──yes──▶ return it
                          │no
                          ▼
      both models cross-verify each other's candidate solution
      against the problem's constraints/edge cases and each
      produce one final corrected answer
                          │
                          ▼
                 both agree now? ──yes──▶ return it
                          │no (rare — genuinely ambiguous problem)
                          ▼
        TIEBREAKER_PROVIDER (default: gemini) decides
```

Agreement between two *independently* prompted models is itself a
strong correctness signal — that's the actual "verification" the
person asked for, not a rubber-stamp. Disagreement triggers a second,
more expensive round where each model has to defend/fix its answer
against the other's, before any tiebreak is used as a last resort.

A Gemini or Grok outage degrades gracefully (falls back to whichever
model is still up) instead of failing the whole request.

## PaddleOCR: what's different from a hosted OCR API

- **No API key, no external network call for OCR.** PaddleOCR's
  detection + recognition models run in this server's own process
  (CPU by default, GPU if you configure `PADDLE_OCR_DEVICE`). The
  trade-off is that this server now needs real CPU (or GPU) and
  memory headroom to run OCR itself — see "Resource requirements"
  below.
- **Plain text, not Markdown.** Unlike a hosted OCR model that can
  return structured Markdown (headings, tables, inline math), local
  PaddleOCR returns individual recognized text lines, which this
  server sorts into top-to-bottom reading order and joins into plain
  text (see `app/services/paddle_ocr.py`). This is normally enough for
  DSA problem screenshots — prose, constraints, code blocks — but is a
  real capability drop for anything table- or formula-heavy.
- **Model weights download on first use** (tens to a few hundred MB,
  depending on `PADDLE_OCR_LANG`) and are cached on disk afterwards.
  The server warms the OCR engine up in the background at startup
  (`PRELOAD_OCR_ON_STARTUP`) so the first real request doesn't pay
  that cost, but a completely fresh deploy's *first ever* OCR call can
  still be slow if the weights aren't cached yet — make sure the cache
  directory is on a persisted volume in production (see Docker setup
  below) so this only ever happens once.

## Resource requirements

Because OCR now runs on this server instead of a hosted API, size your
host accordingly:

- At least 1–2 CPU cores and ~1–2 GB of free RAM per concurrent OCR
  engine instance (`OCR_MAX_CONCURRENCY`) is a reasonable starting
  point for CPU inference with the default models; test with your own
  screenshots and adjust.
- **Not a fit for Vercel's default zip-based Python runtime**: the
  paddlepaddle/paddleocr dependencies and model weights are far larger
  than that runtime's 500 MB deployment-size limit.
  `api/index.py` (the old zip-runtime entrypoint) is left in the repo
  for reference but is no longer what `vercel.json` points at.

  This project instead deploys to Vercel as a **container image
  Function** (`Dockerfile.vercel`), which supports uncompressed
  bundles up to 5 GB on Fluid Compute — enough room for
  paddlepaddle/paddleocr and their weights. Two things make this work
  well:

  1. `Dockerfile.vercel` instantiates PaddleOCR once *during the
     image build*, so the model weights are baked into the image
     layer instead of being downloaded on every cold start (container
     Functions are stateless — nothing written at runtime persists
     between instances, so a runtime download would otherwise repeat
     on every scale-from-zero).
  2. Vercel injects `$PORT` at runtime rather than using a fixed port,
     so the container's `CMD` binds uvicorn to `$PORT` instead of a
     hardcoded `8000` (the plain `Dockerfile`, used for
     docker-compose/local Docker, still binds to a fixed `8000` since
     that's a normal long-lived host).

  To deploy:
  1. In the Vercel project settings, set the environment variable
     `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (only needed for projects
     created before large-function support became the default) and
     make sure **Fluid Compute** is enabled.
  2. Set `PROVIDER_API_KEY` as a project env var. Container Function
     instances are stateless/ephemeral, so relying on the
     file-persisted key in `DATA_DIR` (fine for Docker with a real
     volume) would give you a different client-facing API key per
     cold start; `PROVIDER_API_KEY` pins it.
  3. Set `GEMINI_API_KEY`, `GEMINI_MODEL`, `GROK_API_KEY`,
     `GROK_MODEL`, `APP_ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`, and
     `PUBLIC_BASE_URL` as usual (see `.env.example`).
  4. Deploy. Vercel builds `Dockerfile.vercel` as the `api` service
     declared in `vercel.json` and routes all traffic to it.

  If you change `PADDLE_OCR_LANG`/`PADDLE_OCR_DEVICE`/etc. away from
  the `en`/`cpu` defaults via env vars, update the matching
  `PaddleOCR(...)` call inside `Dockerfile.vercel` too — otherwise the
  weights baked into the image won't match what's requested at
  runtime, and the first request after each cold start will fall back
  to downloading the right ones on the fly.

  For a long-lived process on a host with real persistent disk
  instead (no cold-start weight/key concerns at all), the original
  `Dockerfile` + `docker-compose.yml` still work as-is — Render,
  Fly.io, a VPS, etc.

## Output contract

Both the solver prompt and the arbiter prompt enforce the same rules:

- Default language: C++17.
- **Code only.** No explanations, no comments, no headings, no
  Markdown code fences (` ``` `, ` ```cpp `, etc.) — unless the user's
  message explicitly asks for an explanation and/or comments, in which
  case exactly what was asked for is included.
- Directly copy-paste/submit-ready.

See `app/services/prompts.py` for the exact prompts — tuned for DSA
and competitive-programming constraints (overflow, TLE, edge cases,
exact I/O format, etc.), and shared identically by the solver and
arbiter stages so their outputs are comparable.

## No limits: what "0" means, and what it can't mean

Every limit this server owns defaults to `0`, which means **no limit** —
nothing here refuses a request, truncates what it assembles, or puts a
clock on an answer:

| Setting | Default | `0` means |
|---|---|---|
| `SOLVER_MAX_TOKENS` | `0` | ask for the largest budget the provider documents |
| `REQUEST_TIMEOUT_SECONDS` | `0` | no timeout; this server sets no total deadline |
| `MAX_PROMPT_CHARS` | `0` | any prompt length |
| `MAX_IMAGES` | `0` | any number of screenshots per solve request |
| `MAX_IMAGE_SIZE_MB` | `0` | any image size |
| `MAX_OCR_BATCH_IMAGES` | `0` | any number of images per `/v1/ocr` call |
| `MAX_OCR_BATCH_BYTES` | `0` | any batch size |
| `ALWAYS_VERIFY` | `true` | (not a limit) every answer is cross-verified by both models |

Set any of them to a positive value to put that cap back.

**The output budget still has to be a number.** The APIs require one, and
*omitting* it is not the same as unlimited: Gemini falls back to a
model-dependent default far below its real output limit, and that default
is what used to cut solutions off mid-function — Gemini 2.5 makes it
worse because thinking tokens count against it too, so a hard problem
could spend the entire default thinking and leave too little room for the
code. So `0` sends the largest budget each provider documents
(`UNLIMITED_BUDGET` in `gemini_client.py` / `grok_client.py`). A model
with a smaller maximum rejects that with a 400 naming its own ceiling;
the client retries there, remembers it for that model, and — whatever
happens — never returns a response the provider marked as truncated. A
truncation shows up as a logged 502, never as code that silently stops
mid-function.

A client may pass `max_tokens` on `/v1/chat/completions` to ask for
*less* than the server budget. It can only ever lower it.

**Ceilings that aren't ours to remove:** the model's own context window,
and — only if you deploy on Vercel — its 4.5 MB request-body cap and
function duration limit. Those surface as provider/platform errors rather
than as a locally rejected request. Self-hosted Docker/docker-compose has
neither. No configuration can guarantee a model's answer is *correct*;
`ALWAYS_VERIFY=true` (the default) is the strongest signal this pipeline
can buy — both models independently re-derive and judge the answer.

## What your app needs to plug in (the "AI providers" form)

| Field | Value |
|---|---|
| Name | anything, e.g. `DSA Solver (Gemini+Grok)` |
| Model | `dsa-solver` (or whatever you set `PUBLIC_MODEL_NAME` to) |
| Endpoint URL | `https://YOUR-DEPLOYMENT-URL/v1/chat/completions` |
| API key | the backend-generated key, from `/admin` → *Show API Key* |

The Gemini and Grok API keys are server-side only and are never
exposed through this endpoint or the admin panel's config export in a
way your client app would see — your app only ever talks to the
single generated `dsa_sk_...` key above. PaddleOCR has no API key of
its own; it runs locally.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `APP_ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET` (pick strong values)
   - `GEMINI_API_KEY` + `GEMINI_MODEL` (e.g. `gemini-2.5-pro`) —
     called over Google's OpenAI-compatible endpoint
     (`https://generativelanguage.googleapis.com/v1beta/openai`)
   - `GROK_API_KEY` + `GROK_MODEL` — the second solver, called over
     Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`)
     with whatever model slug you pick there (e.g. `openai/gpt-oss-120b`
     — see [Groq's model catalog](https://console.groq.com/docs/models);
     slugs get retired/renamed over time)
   - PaddleOCR's own settings (`PADDLE_OCR_LANG`, `PADDLE_OCR_DEVICE`,
     `OCR_MAX_CONCURRENCY`, ...) have sensible CPU/English defaults —
     adjust only if you need another language or have a GPU.
2. Local run: `pip install -r requirements.txt && uvicorn app.main:app --reload`
   (the first run downloads PaddleOCR's model weights, so expect the
   first OCR request — or the background warm-up at startup — to take
   longer than usual).
3. Visit `/admin`, log in with `APP_ADMIN_PASSWORD`, click **Show API
   Key** and **Test Backend** (this pings Gemini and Grok, and runs a
   small local image through PaddleOCR, reporting latency/success for
   each).
4. Paste the endpoint (`.../v1/chat/completions`), model name, and API
   key into your app's "AI providers" form.

## Deploying with Docker

`Dockerfile` and `docker-compose.yml` are set up for this. Two volumes
matter in production:

- `/app/data` — the persisted, backend-generated client API key.
- `/root/.paddlex` — PaddleOCR's downloaded model weight cache. Mount
  this too, or every container replacement re-downloads the models on
  its first OCR request.

```
docker compose up -d --build
```

## Consensus tuning

- `TIEBREAKER_PROVIDER` (`gemini` | `grok`, default `gemini`) — who
  decides in the rare case both models still disagree after
  cross-verifying each other.
- `ALWAYS_VERIFY` (`true`/`false`, default `true`) — run the
  cross-verification round even when both models already agree on the
  first pass. On by default, because a matching first pass is weaker
  evidence than both models independently re-deriving the same answer.
  Turning it off costs 2 fewer model calls per request.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible; accepts text and/or
  `image_url` (base64 data URL) parts, plus an optional `ocr_results`
  array of screenshots OCR'd earlier (see below), and an optional
  `stitch` boolean (default `true` — see "Multiple screenshots in one
  request" above). Requires `Authorization: Bearer <backend-generated
  key>`.
- `POST /v1/ocr` — OCR-only batch endpoint for the two-phase flow. Same
  auth; accepts any number of images (`MAX_OCR_BATCH_IMAGES` defaults to
  `0` = unlimited), plus the same optional `stitch` boolean (only
  affects this response's convenience `text` field).
- `GET /admin` — admin UI (password-protected).
- `GET /health` — liveness check.

## Why a 5-15 screenshot scroll of ONE problem could still come out wrong

Even with `stitch: true` and screenshots that really are one continuous
scroll (as intended), the header/footer/scroll-seam matching in
`app/services/problem_reconstructor.py` used to get less reliable the
*more* screenshots were in one request — the opposite of what you'd
want. Two specific false-positive bugs, both now fixed and covered by
tests in `tests/test_reconstruct.py`:

- **A per-screenshot counter looked like chrome.** Many judge UIs show
  something like a question index, page number, or submission id right
  next to genuinely fixed chrome (`Q46`, `3/15`, ...). That counter is
  ~98% the same text from one screenshot to the next — it differs only
  in its digits — which a plain character-similarity comparison cannot
  tell apart from ordinary OCR noise on truly-static chrome. The old
  logic treated the whole line (counter included) as repeated chrome and
  stripped it from every screenshot but the first, taking real
  per-screenshot content with it. Digit-differing lines are no longer
  treated as a chrome match at all, no matter how similar the rest of
  the line is.
- **Nearly-swallowed screenshots.** Once chrome and seam-overlap are
  stripped, a screenshot's remaining *unique* content can be short — and
  a short remainder is more likely to coincidentally share a phrase with
  the previous screenshot's remainder (shared constraints wording, a
  repeated example format), which the scroll-seam matcher could
  misjudge as "this whole thing is overlap" and drop entirely. A fuzzy
  (non-exact) match can no longer swallow 100% of a screenshot, and a
  match covering nearly all of a longer screenshot needs stronger
  evidence than before.

Both failure modes get more likely with more screenshots simply because
there are more headers and more seams for one coincidence to slip
through — which is why this was more visible at 5-15 images than at 2.
If you still see missing content after this fix, run with `--show-ocr`
on the failing batch and compare it to what actually reaches the model
(see below) — that will show directly whether it's these dedup passes
or something else.

## Multiple screenshots in one request: `stitch`

Both `/v1/chat/completions` and `/v1/ocr` accept a `stitch` boolean
(default `true`). When `true`, screenshots are assumed to be
**consecutive views of one continuously-scrolled problem**:
`app/services/problem_reconstructor.py` looks for repeated fixed
chrome (nav bars, tab strips, the code panel) and scroll-seam overlap
between consecutive shots and removes the repeats, so the model reads
one continuous statement instead of the same paragraph twice.

That assumption only holds when the screenshots really are one
continuous scroll. If instead you send **several unrelated
screenshots in one request** — different problems, a problem plus a
separate shot of your own code, screenshots in the wrong order, or
anything else that isn't one continuous page — that same matching can
misfire: judge sites repeat a lot of boilerplate verbatim across
*different* problems too ("Time Limit: 1 second", a shared
`Constraints:`/example layout), and the heuristic can't tell "this is
chrome/overlap" from "this problem's text happens to look like that
one's." When it gets it wrong, it silently deletes real, unique
content from one of the screenshots before Gemini/Grok ever see it —
which is the most common way this pipeline produces a confidently
wrong answer from screenshots that individually OCR'd just fine.

**Set `stitch: false`** whenever your screenshots aren't one
continuous scroll. Every screenshot's OCR'd text is then kept in full
under its own `SCREENSHOT n` block — nothing is ever merged, trimmed,
or marked `DUPLICATE` — at the cost of not collapsing genuine scroll
repeats when they do occur. `scripts/solve_screenshots.py` exposes
this as `--independent`:

```bash
python3 scripts/solve_screenshots.py shots/*.png --independent -q "Solve in C++17."
```

If you're not sure which case you're in, run with `--show-ocr` first
(see below) and compare it to the final solve prompt: if text you can
see in `--show-ocr` output is missing from what actually reaches the
model, `stitch` is very likely the fix.

## Many screenshots (15-20): the two-phase flow

Vercel caps request bodies at **4.5 MB** and base64 inflates image
bytes by 4/3, so 15-20 screenshots can never arrive in a single
`/v1/chat/completions` call. OCR them in small batches first, then solve
once with the accumulated text. The server keeps no state between the
two phases — ordering survives because each batch carries its own
`start_index` and reports back a `next_start_index`.

### Ready-made client

`scripts/solve_screenshots.py` does the whole flow for you — chunks the
files by count *and* by encoded size, OCRs each batch, then solves once.
Standard library only, no `pip install`:

```bash
export DSA_API_KEY=dsa_sk_...   # from /admin -> Reveal key
python3 scripts/solve_screenshots.py shots/*.png -q "Solve in C++17."
```

Add `--show-ocr` to print the text parsed from every screenshot (the
fastest way to see whether OCR is the problem), and `--ocr-only` to stop
before the solve step:

```bash
python3 scripts/solve_screenshots.py shots/*.png --show-ocr --ocr-only
```

Point it elsewhere with `--base-url` (e.g. `http://127.0.0.1:8000` against
a local Docker instance).

### Rolling your own

```js
const AUTH = { Authorization: `Bearer ${apiKey}` };
const ocrResults = [];
let start = 0;

// 1. OCR every screenshot in batches of <= 4
for (const batch of chunk(screenshots, 4)) {
  const res = await fetch(`${base}/v1/ocr`, {
    method: "POST",
    headers: { ...AUTH, "Content-Type": "application/json" },
    body: JSON.stringify({ images: batch, start_index: start }),
  }).then((r) => r.json());
  ocrResults.push(...res.results);
  start = res.next_start_index;
}

// 2. Solve everything in one small, text-only request
const answer = await fetch(`${base}/v1/chat/completions`, {
  method: "POST",
  headers: { ...AUTH, "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "dsa-solver",
    messages: [{ role: "user", content: "Solve in C++17, O(n log n)." }],
    ocr_results: ocrResults,
  }),
}).then((r) => r.json());
```

- Screenshots across `ocr_results` + inline `image_url` parts in one
  solve request have no cap by default (`MAX_IMAGES=0`).
- Each `/v1/ocr` response also has `text`: numbered `SCREENSHOT n`
  blocks with no preamble. Concatenate those across batches if you
  prefer sending plain text instead of echoing `results`.
- Inline `image_url` parts and `ocr_results` can be mixed in one solve
  request; inline images are OCR'd and appended after the earlier ones.
- There is no batch limit by default (`MAX_OCR_BATCH_IMAGES=0`,
  `MAX_OCR_BATCH_BYTES=0`). **If you deploy on Vercel**, set both back to
  positive values (the old `4` images / `3 MB` split things just under its
  4.5 MB body cap) so an oversized batch gets a 413 with instructions to
  split it — rather than the platform's opaque 413, which is all you'd get
  if the check only happened at the edge. `MAX_IMAGE_SIZE_MB` applies per
  image on the same terms.
- Prefer screenshotting at ~1600 px wide or smaller: smaller files fit
  more images per batch and OCR faster.

## Notes

- Supported image formats: PNG, JPEG, WEBP — any size, any count
  (`MAX_IMAGE_SIZE_MB`, `MAX_OCR_BATCH_IMAGES`, and `MAX_IMAGES` all
  default to `0` = unlimited).
- **Tall screenshots are sliced, not shrunk.** A full-page capture
  (e.g. 900 x 14000) would otherwise be squeezed to 2000px on its
  longest side — 1/7 scale — and OCR would return gibberish. Images
  taller than `TILE_HEIGHT_PX` are cut into ~1800px tiles *at blank
  rows* (so no line is ever split) and OCR'd near native resolution.
  Narrow screenshots (< 960px wide) are upscaled for the same reason.
- **The background colour does not change the reading.** Images are
  converted to grayscale before detection and recognition, so white,
  black, green, terminal, washed-out, gradient, wallpaper and
  syntax-highlighted backgrounds all return the same text.
- Assembled prompts over `MAX_PROMPT_CHARS` are rejected locally with a
  413 rather than an opaque provider context-length error. The default
  is `0`, so this only applies if you set a cap.
- Streaming (`stream: true`) is not supported — returns a 400.
- Only base64 `data:` URLs are accepted for `image_url.url` (no
  external image fetching).
