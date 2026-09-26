"""
Combines per-screenshot OCR output into a single reconstructed problem
statement, in screenshot order, ready to hand to the solver.

Two kinds of repetition show up across screenshots of one problem and
have to be collapsed before the solver sees the text:

* fixed chrome — the icon rail, the tab strip above the description and
  the code panel beside it are pinned to the window, so every shot
  repeats them verbatim. The header is kept from the first shot and the
  footer from the last one; every other copy is dropped, otherwise the
  statement opens with the same navigation line three times and the
  scroll seam never matches.
* scroll overlap — screenshots are taken by scrolling, so the top of one
  shot repeats the bottom of the previous one. Those repeated lines are
  removed here, and a shot that contributes nothing new is marked as a
  duplicate, so the solver reads one continuous statement instead of the
  same paragraph appearing under two SCREENSHOT headers.

This module deliberately does NOT try to be clever about correcting OCR
errors beyond whitespace cleanup and that overlap stitching — it is the
solver's job (with the full assembled context) to reason about ambiguous
or unreadable text, and it is explicitly told to never invent missing
constraints or examples.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_DIGITS = re.compile(r"\d+")

from app.models.solver import OcrResult, ReconstructedProblem

# How far back a scroll can repeat itself: a screenshot is at most a
# couple of screens of text, and an overlap larger than this is more
# likely to be a genuinely repeated section than a scroll seam.
_MAX_OVERLAP_LINES = 30
# The same line re-shot at a different pixel offset OCRs slightly
# differently, so matching is fuzzy rather than exact.
_OVERLAP_SIMILARITY = 0.8

# Fixed chrome (sidebar, tabs, code panel) is checked position-for-
# position against the first screenshot. This has to be a HIGH bar, not
# a loose one: with only two screenshots, a coincidental near-match is
# rare, but with 5-15 (a normal scroll of one long problem) it stops
# being rare, and real screenshots often carry a small per-shot counter
# or index right next to genuinely fixed chrome ("Q46", "3/15", a
# submission id) that differs by only a character or two — which is
#*exactly* what a loose fuzzy threshold cannot tell apart from OCR
# noise on truly-identical chrome, since both look like "≈98% the same
# line". 0.92 still comfortably tolerates a single misread character on
# a normal-length line (real OCR noise), while no longer treating a
# line that differs by a whole token/number as the same chrome merely
# because most of its characters happen to match.
_PANEL_SIMILARITY = 0.92
# A run this short is not chrome: two shared lines are two lines that
# happen to look alike, and dropping them would eat real content. Three
# is enough, because the lines are compared *position for position* with
# the first screenshot: anything that matches there is already printed on
# the first screenshot, so the copies on the others add no information.
# An icon rail with a near-black background loses its glyph boxes to the
# detector and comes back as three labels rather than four, and the
# fourth line of the header is scrolling content — waiting for it would
# print the rail and the tab strip twice.
_PANEL_MIN_RUN = 3
# Extending a run past a line one frame failed to read is a bigger leap
# than accepting consecutive matches, so slack still needs the longer
# run before it may skip anything.
_PANEL_SLACK_RUN = 4
# ... and a run that long is worth extending past the odd line that one
# frame failed to read.
_PANEL_SLACK = 2

# Scroll seams are matched at character level rather than line level,
# because the same paragraph re-wraps differently once the viewport
# moves: one shot breaks the sentence at "(inclusive)," and the next
# packs it into a single line. A seam is a chain of matching runs
# anchored on the end of the previous screenshot.
_CHAIN_BLOCK = 8  # shorter runs are word-level coincidence
_CHAIN_MATCH = 24  # a chain this short is not a seam
_CHAIN_GAP = 40  # a run may be interrupted by about a line of junk
_CHAIN_TAIL = 24  # tolerate this much junk at the very end of the previous shot
_CHAIN_LINE_SHARE = 0.8  # a mostly matched line goes with the seam
# Lines sitting in front of the seam (chrome the header pass declined to
# strip, or a line the viewport clipped) are only dropped when the
# previous screenshot showed them too. A few characters we cannot account
# for are tolerated — a clip can turn a line into garbage — but only when
# the overlap behind them is substantial, so a stray shared line between
# two unrelated screenshots never earns the right to delete a heading.
_LEAD_SHARE = 0.8
_LEAD_MATCH = 6
_CHAIN_LEAD_UNKNOWN = 48
_CHAIN_STITCH_MIN = 80

# Safety nets against false-positive stitching. Both kinds of matching
# above were tuned on pairs of screenshots; with more screenshots in one
# request (a 5-15 shot upload of a long problem is normal) there are many
# more headers and many more seams for a coincidence to slip through one
# of, and any single garbled OCR line among 5-15 real photos is common
# enough that "requires every screenshot to agree" is not as strong a
# guarantee as it sounds. A false NEGATIVE here just leaves a line of
# chrome or an overlapping sentence repeated once more than necessary —
# harmless clutter. A false POSITIVE deletes real problem text before the
# solver ever sees it, which is how this pipeline produces a confident,
# wrong answer from screenshots that individually OCR'd fine. These caps
# bias hard toward the harmless failure mode.
#
# Real fixed chrome (an icon rail, a tab strip, a code panel) is a
# handful of lines; a "shared header" run this long is far more likely to
# be a coincidence (or, with stitch left on by mistake, actual content
# that happens to repeat across screenshots of different problems) than
# genuine chrome, so it is not stripped past this length.
_PANEL_MAX_LINES = 12
# A scroll seam is, by definition, the part of `current` that repeats
# onto the tail of `previous` — visually at most one screen's worth. A
# fuzzy, partial match that would remove almost all of a screenshot's
# lines is not that; it is generic wording (shared constraints/example
# boilerplate) coincidentally resembling the previous shot's tail. The
# one legitimate case where a whole screenshot is already-seen content
# (a re-uploaded or fully-scrolled-back shot) is caught separately, up
# front, by the exact `previous == current` equality check below, so
# anything reaching this cap is a fuzzy/partial match, not that case.
_SEAM_MAX_DROP_FRACTION = 0.85
# Below this many lines, "fraction of the screenshot" is too coarse a
# signal to trust: a genuine rewrap can legitimately cover 3 of a
# 4-line screenshot. The cap only kicks in once there is enough length
# for "nearly the whole thing matched" to be meaningful evidence of a
# coincidence rather than expected behaviour.
_SEAM_FRACTION_MIN_LINES = 8

_UNREADABLE = "[UNREADABLE: no text could be extracted from this screenshot]"
_DUPLICATE = "[DUPLICATE: this screenshot showed only content already given above]"


def _body_lines(text: str) -> list[str]:
    """Non-blank lines, indentation preserved — what the solver reads."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def _norm(line: str) -> str:
    """Whitespace-collapsed view of a line — what matching compares."""
    return " ".join(line.split())


def _similar(a: str, b: str, threshold: float = _OVERLAP_SIMILARITY) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _panel_similar(a: str, b: str, threshold: float = _PANEL_SIMILARITY) -> bool:
    """Like `_similar`, but two lines whose only textual difference is
    which digits they contain are never treated as the same fixed
    chrome. A page number, question index, submission id, or
    incrementing example label ("Example 1" vs "Example 2") sits right
    next to genuinely fixed chrome in a lot of real screenshots, and
    against a plain character-ratio comparison it looks exactly like
    OCR noise on truly-static chrome — both are "almost all the same
    characters". Real fixed chrome never has its digits change between
    screenshots, so when the only difference is digits, that is treated
    as decisive: not a match, regardless of how high the raw ratio is.
    Everything else about the comparison (tolerating an actually-garbled
    character elsewhere in the line) is unchanged."""
    if a == b:
        return True
    if not a or not b:
        return False
    if _DIGITS.sub("#", a) == _DIGITS.sub("#", b) and _DIGITS.findall(a) != _DIGITS.findall(b):
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _panel_run(views: list[list[str]], reverse: bool = False) -> int:
    """Length of the run of lines every screenshot shares at the same
    position — the fixed header (or, reversed, the fixed footer)."""
    if len(views) < 2:
        return 0
    seqs = [list(reversed(view)) if reverse else view for view in views]
    limit = min(len(seq) for seq in seqs)
    committed = 0
    slack = _PANEL_SLACK
    for i in range(limit):
        first = seqs[0][i]
        if all(
            _panel_similar(first, seq[i]) for seq in seqs[1:]
        ):
            committed = i + 1
        elif slack and committed >= _PANEL_SLACK_RUN:
            slack -= 1
        else:
            break
    if committed < _PANEL_MIN_RUN:
        return 0
    # See _PANEL_MAX_LINES: a run this long stops looking like fixed
    # chrome and starts looking like a false positive, so it is capped
    # rather than trusted past that length.
    return min(committed, _PANEL_MAX_LINES)


def _covered_lines(lines: list[str], cut: int) -> int:
    """Leading lines of `lines` that the character position `cut`
    covers. A line is taken when it is fully covered, or mostly covered
    — the remainder is then a small OCR difference, not new content."""
    start = 0
    drop = 0
    for line in lines:
        end = start + len(line)
        if cut <= start:
            break
        if cut < end:
            if cut - start >= _CHAIN_LINE_SHARE * len(line):
                drop += 1
            break
        drop += 1
        start = end + 1
    return drop


def _present_in(line: str, previous: list[str]) -> bool:
    """Whether an earlier screenshot already showed this line: exactly,
    or close enough to be the same text with the viewport cutting it
    half way through."""
    for candidate in previous:
        if line == candidate:
            return True
        shared = min(len(line), len(candidate))
        if shared < _LEAD_MATCH:
            continue
        matched = sum(
            block.size
            for block in SequenceMatcher(
                None, line, candidate, autojunk=False
            ).get_matching_blocks()
        )
        if matched >= _LEAD_MATCH and matched >= _LEAD_SHARE * shared:
            return True
    return False


def _unknown_lead(b_lines: list[str], seam_at: int, previous: list[str]) -> int:
    """Characters sitting in front of the seam that the previous
    screenshot never showed."""
    start = 0
    unknown = 0
    for line in b_lines:
        end = start + len(line)
        if end > seam_at:
            # The line the chain starts on is judged by the chain itself.
            break
        if not _present_in(line, previous):
            unknown += len(line) + 1
        start = end + 1
    return unknown


def _seam_drop(previous: list[str], current: list[str]) -> int:
    """Number of leading lines of `current` that repeat the tail of
    `previous` — the scroll seam between two screenshots."""
    if not previous or not current:
        return 0
    if previous == current:
        # Same picture uploaded twice (or a shot fully inside the previous
        # one): every line of `current` is already on screen.
        return len(current)

    a_lines = previous[-_MAX_OVERLAP_LINES:]
    b_lines = current[:_MAX_OVERLAP_LINES]
    a = " ".join(a_lines)
    b = " ".join(b_lines)
    blocks = [
        match
        for match in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
        if match.size >= _CHAIN_BLOCK
    ]
    # The overlap is a suffix of the previous shot by construction, so the
    # chain has to reach the end of `a` — a run matching somewhere in the
    # middle is content the next screenshot happens to mention again.
    anchor = None
    for match in blocks:
        if match.a + match.size >= len(a) - _CHAIN_TAIL:
            anchor = match
    if anchor is None:
        return 0

    total = anchor.size
    leftmost = anchor
    for match in reversed(blocks[: blocks.index(anchor)]):
        if (
            leftmost.a - (match.a + match.size) > _CHAIN_GAP
            or leftmost.b - (match.b + match.size) > _CHAIN_GAP
        ):
            break
        leftmost = match
        total += match.size
    # Junk in front of the seam (a tab strip the header pass declined to
    # strip) is fine as long as the previous screenshot showed it too;
    # unique content the previous shot never showed is not.
    if total < _CHAIN_MATCH:
        return 0
    unknown = _unknown_lead(b_lines, leftmost.b, previous)
    if unknown > _CHAIN_LEAD_UNKNOWN or (unknown and total < _CHAIN_STITCH_MIN):
        return 0
    covered = _covered_lines(b_lines, anchor.b + anchor.size)
    if covered >= len(current):
        # A *fuzzy* match should never fully swallow a screenshot. The
        # legitimate "this whole screenshot is already-seen content" case
        # is caught above, up front, by the exact `previous == current`
        # equality check — so any other path that reaches full coverage
        # is a coincidence (shared generic wording), not a real seam, and
        # is most dangerous exactly when `current` is short: a one- or
        # two-line remainder swallowed whole loses 100% of what that
        # screenshot contributed.
        return 0
    # A fuzzy match covering nearly all (but not literally all) of a
    # *longer* screenshot is the same false-positive risk on a longer
    # remainder. Only applied once `current` is long enough for a
    # fraction to mean anything — on a short screenshot (a rewrapped
    # paragraph is often only 3-4 lines) a genuine seam routinely covers
    # most of it, and the existing chain/lead checks above already guard
    # the short case well.
    if len(current) >= _SEAM_FRACTION_MIN_LINES and covered >= round(
        _SEAM_MAX_DROP_FRACTION * len(current)
    ):
        return 0
    return covered


def _strip_fixed_panels(
    keys: list[int], raws: list[list[str]], norms: list[list[str]]
) -> dict[int, tuple[list[str], list[str]]]:
    """Keeps the shared header on the first screenshot and the shared
    footer on the last one, and drops them from every shot in between."""
    if len(norms) >= 2:
        head = _panel_run(norms, reverse=False)
        # The footer is searched for only below the header. Without that
        # bound a reverse run skips the one or two body lines it fails to
        # match, resumes on the header — which every shot shares after all
        # — and swallows the statement between the two panels.
        tails = [view[head:] for view in norms]
        tail = _panel_run(tails, reverse=True)
    else:
        head = tail = 0
    stripped: dict[int, tuple[list[str], list[str]]] = {}
    for position, key in enumerate(keys):
        raw, norm = raws[position], norms[position]
        start = head if position else 0
        stop = len(raw) - tail if position < len(keys) - 1 else len(raw)
        stripped[key] = (raw[start:stop], norm[start:stop])
    return stripped


def format_screenshot_blocks(ocr_results: list[OcrResult], stitch: bool = True) -> str:
    """Numbered SCREENSHOT blocks in order.

    When `stitch` is True (the default, and the only behaviour this
    function had before this parameter existed), the fixed chrome and
    the scroll overlap between consecutive screenshots are removed —
    correct when every screenshot is a consecutive view of ONE
    continuously-scrolled problem, which is what the header/footer and
    seam matching below assumes.

    When the caller's screenshots are NOT that — several different
    problems, a problem plus an unrelated shot of their own code,
    screenshots taken out of order, or anything else that is not one
    continuous scroll — that assumption is false, and the matching
    below can misfire: generic boilerplate a judge repeats on every
    problem page ("Time Limit: 1 second", "Constraints:", a shared
    example format) looks exactly like repeated chrome or a scroll seam
    to a heuristic that only compares text, and gets silently deleted
    from screenshots where it was actually the only copy. Pass
    `stitch=False` in that case: every screenshot's OCR text is kept in
    full (still marking the ones OCR could not read at all), so nothing
    is ever dropped, at the cost of not collapsing genuine scroll
    repeats or truly duplicate screenshots."""
    ordered = sorted(ocr_results, key=lambda r: r.index)

    if not stitch:
        blocks: list[str] = []
        for r in ordered:
            header = f"SCREENSHOT {r.index + 1}"
            if r.unreadable or not r.text.strip():
                blocks.append(f"{header}\n{_UNREADABLE}")
            else:
                blocks.append(f"{header}\n" + "\n".join(_body_lines(r.text)))
        return "\n\n".join(blocks)

    keys: list[int] = []
    raws: list[list[str]] = []
    norms: list[list[str]] = []
    for r in ordered:
        if r.unreadable or not r.text.strip():
            continue
        raw = _body_lines(r.text)
        if not raw:
            continue
        keys.append(r.index)
        raws.append(raw)
        norms.append([_norm(line) for line in raw])

    stripped = _strip_fixed_panels(keys, raws, norms)

    blocks: list[str] = []
    # The last screenshot that actually contributed text: the seam is
    # always measured against it, so an unreadable or duplicate shot in
    # the middle does not break the chain.
    previous: tuple[list[str], list[str]] | None = None

    for r in ordered:
        header = f"SCREENSHOT {r.index + 1}"
        if r.unreadable or not r.text.strip():
            blocks.append(f"{header}\n{_UNREADABLE}")
            continue

        raw, norm = stripped[r.index]
        if previous is not None:
            drop = _seam_drop(previous[1], norm)
            if drop:
                raw, norm = raw[drop:], norm[drop:]

        if not raw:
            blocks.append(f"{header}\n{_DUPLICATE}")
            continue
        previous = (raw, norm)
        blocks.append(f"{header}\n" + "\n".join(raw))

    return "\n\n".join(blocks)


def reconstruct_problem(ocr_results: list[OcrResult], stitch: bool = True) -> ReconstructedProblem:
    ordered = sorted(ocr_results, key=lambda r: r.index)

    combined = format_screenshot_blocks(ordered, stitch=stitch)

    if stitch:
        preamble = (
            "The following problem statement was reconstructed from "
            f"{len(ordered)} screenshot(s), preserved in their original order. "
            "They are consecutive views of the SAME problem — read them as one "
            "continuous statement, not as separate problems. Where the top of a "
            "screenshot repeated the bottom of the previous one, those repeated "
            "lines have already been removed; fixed interface chrome (navigation, "
            "sidebars, code panels) was likewise kept only once, at the edge "
            "where it belongs. A screenshot marked DUPLICATE showed only content "
            "already given above. Screenshots marked UNREADABLE could not be "
            "OCR'd — treat any such gaps as genuinely missing information; do "
            "not invent constraints, examples, or input/output formats to fill "
            "them in.\n\n"
        )
    else:
        preamble = (
            "The following was reconstructed from "
            f"{len(ordered)} screenshot(s), preserved in their original order. "
            "No de-duplication was performed between them, so treat each "
            "SCREENSHOT block on its own merits — they may be different parts "
            "of one problem, entirely different problems, or a mix of problem "
            "statement and other material (e.g. the user's own code); use the "
            "surrounding text and ADDITIONAL USER NOTES (if present) to tell "
            "them apart rather than assuming they are one continuous statement. "
            "Screenshots marked UNREADABLE could not be OCR'd — treat any such "
            "gaps as genuinely missing information; do not invent constraints, "
            "examples, or input/output formats to fill them in.\n\n"
        )

    return ReconstructedProblem(
        text=preamble + combined,
        screenshot_count=len(ordered),
    )
