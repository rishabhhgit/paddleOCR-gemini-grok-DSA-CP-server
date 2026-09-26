"""End-to-end OCR reading-order tests on realistic screenshots.

PaddleOCR returns one recognized fragment per detected region, and those
fragments arrive in whatever order the detector fired. Two real layouts
broke badly with a naive top-then-left sort:

* split-view pages (problem left, code editor right) interleaved the two
  columns line by line;
* code blocks reordered fragments inside a single line, because glyphs
  like `[` start higher than lowercase letters (`[]` before `return`).

These tests render those layouts — plus dark mode, small browser text,
a tall scroll, and a long comment line — and assert the text comes back
in the order a human reads it.
"""

import asyncio
import io
import random
import re

import pytest

from PIL import Image, ImageDraw, ImageFont

from app.services.problem_reconstructor import reconstruct_problem

pytest.importorskip("paddleocr")

_SANS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_MONO = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _font(paths: list[str], size: int) -> ImageFont.ImageFont:
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _norm(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def ocr():
    """OCR a PNG with the real engine and return the ordered text."""
    from app.config import Settings
    from app.models.solver import ExtractedImage
    from app.services.paddle_ocr import run_ocr_on_images

    settings = Settings(_env_file=None)

    def run(png: bytes) -> str:
        results = asyncio.run(
            run_ocr_on_images(settings, [ExtractedImage(mime_type="image/png", data=png)])
        )
        return results[0].text

    def run_many(png: bytes, start_index: int = 0) -> list:
        results = asyncio.run(
            run_ocr_on_images(settings, [ExtractedImage(mime_type="image/png", data=png)])
        )
        return [r.model_copy(update={"index": start_index + r.index}) for r in results]

    run.many = run_many
    return run


@pytest.fixture(scope="module")
def two_pane_png() -> bytes:
    """Split view: problem description on the left, code editor on the right."""
    img = Image.new("RGB", (1600, 1000), "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.rectangle([800, 0, 1600, 1000], fill="#1e1e1e")
    sans, mono = _font(_SANS, 17), _font(_MONO, 16)
    left = [
        "Two Sum",
        "",
        "Given an array of integers nums and an integer",
        "target, return indices of the two numbers such",
        "that they add up to target.",
        "",
        "You may assume that each input would have",
        "exactly one solution, and you may not use the",
        "same element twice.",
        "",
        "Constraints:",
        "2 <= nums.length <= 10^4",
    ]
    right = [
        "class Solution:",
        "    def twoSum(self, nums, target):",
        "        seen = {}",
        "        for i, x in enumerate(nums):",
        "            if target - x in seen:",
        "                return [seen[target - x], i]",
    ]
    y = 40
    for line in left:
        draw.text((40, y), line, fill="#202124", font=sans)
        y += 27
    y = 40
    for line in right:
        draw.text((830, y), line, fill="#d4d4d4", font=mono)
        y += 26
    return _png(img)


@pytest.fixture(scope="module")
def code_png() -> bytes:
    """Syntax-highlighted code block on a dark background."""
    img = Image.new("RGB", (1100, 700), "#0d1117")
    draw = ImageDraw.Draw(img)
    mono = _font(_MONO, 17)
    lines = [
        "class Solution:",
        "    def twoSum(self, nums, target):",
        "        seen = {}",
        "        for i, x in enumerate(nums):",
        "            if target - x in seen:",
        "                return [seen[target - x], i]",
        "            seen[x] = i",
        "        return []",
    ]
    y = 40
    for line in lines:
        draw.text((30, y), line, fill="#d4d4d4", font=mono)
        y += 30
    draw.text((30, y + 20), "# time O(n) space O(n)", fill="#6a737d", font=mono)
    return _png(img)


def _browser_body(img: Image.Image, font_size: int) -> bytes:
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, font_size)
    title = _font(_SANS, int(font_size * 1.3))
    draw.rectangle([0, 0, img.width, 40], fill="#f1f3f4")
    draw.text((14, 10), "leetcode.com/problems/two-sum - Chrome", fill="#5f6368", font=_font(_SANS, 13))
    draw.text((40, 70), "Two Sum", fill="#202124", font=title)
    body = [
        "Given an array of integers nums and an integer target, return indices",
        "of the two numbers such that they add up to target.",
        "You may assume that each input would have exactly one solution, and",
        "you may not use the same element twice.",
        "Input: nums = [2,7,11,15], target = 9",
        "Output: [0,1]",
        "Constraints:",
        "2 <= nums.length <= 10^4",
        "-10^9 <= nums[i] <= 10^9",
    ]
    y = 120
    for line in body:
        draw.text((40, y), line, fill="#202124", font=sans)
        y += int(font_size * 1.75)
    return _png(img)


def test_split_view_reads_left_column_before_right(ocr, two_pane_png):
    """The description must not be interleaved with the code editor."""
    text = _norm(ocr(two_pane_png))
    assert "same element twice." in text
    assert "def twoSum(self, nums, target):" in text
    # Every description line comes before every code line.
    assert text.index("same element twice.") < text.index("class Solution:")
    assert text.index("class Solution:") < text.index("for i, x in enumerate(nums):")


def test_code_block_keeps_fragments_in_line_order(ocr, code_png):
    """Fragments inside one line must stay left-to-right."""
    text = _norm(ocr(code_png))
    # Brackets are taller than the letters around them, so a naive sort by
    # the box's top edge put `[]` first.
    assert "return []" in text
    assert "[] return" not in text
    # `#` must lead its own comment line, not trail it.
    comment = text.index("#")
    assert comment < text.index("time O(n)") < text.index("space O(n)")


def test_dark_mode_screenshot(ocr):
    img = Image.new("RGB", (900, 700), "#1c1c1e")
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, 22)
    draw.text((40, 40), "Problem 3: Longest Substring", fill="#f2f2f7", font=sans)
    draw.text((40, 90), "Find the length of the longest substring without repeats.", fill="#f2f2f7", font=sans)
    draw.text((40, 140), "Constraints: 0 <= s.length <= 5 * 10^4", fill="#f2f2f7", font=sans)
    text = _norm(ocr(_png(img)))
    assert "Problem 3: Longest Substring" in text
    assert "longest substring without repeats" in text


def test_small_browser_text(ocr):
    """Desktop screenshots carry ~14px text — the smallest we must read."""
    text = _norm(ocr(_browser_body(Image.new("RGB", (1440, 900), "#ffffff"), 14)))
    assert "target, return indices of the two numbers such that they add up to target." in text
    assert "2 <= nums.length <= 10^4" in text
    assert "-10^9 <= nums" in text and "<= 10^9" in text


def test_tall_screenshot_is_read_in_chunks(ocr):
    """A long page is tiled vertically; the tiles must concatenate in order."""
    img = Image.new("RGB", (900, 4200), "#ffffff")
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, 26)
    y = 40
    for n in range(1, 16):
        draw.text((40, y), f"Problem {n}: Example heading {n}", fill="#202124", font=sans)
        y += 40
        draw.text((40, y), f"Detail line for item {n} with several words of prose.", fill="#202124", font=sans)
        y += 60
        draw.text((40, y), "Edge cases: empty input, single element, all duplicates.", fill="#202124", font=sans)
        y += 160
    text = _norm(ocr(_png(img)))
    assert text.index("Problem 1: Example heading 1") < text.index("Problem 15: Example heading 15")
    for n in (1, 8, 15):
        assert f"Problem {n}: Example heading {n}" in text


def test_scroll_sequence_of_one_problem_is_stitched(ocr):
    """Three scrolling screenshots of one statement must come back as one
    continuous problem, with the shared rows appearing only once."""
    lines = [
        "Two Sum",
        "Given an array of integers nums and an integer target, return indices of",
        "the two numbers such that they add up to target.",
        "You may assume that each input would have exactly one solution, and you",
        "may not use the same element twice.",
        "Example 1:",
        "Input: nums = [2,7,11,15], target = 9",
        "Output: [0,1]",
        "Explanation: Because nums[0] + nums[1] == 9, we return [0,1].",
        "Example 2:",
        "Input: nums = [3,2,4], target = 6",
        "Output: [1,2]",
        "Constraints:",
        "2 <= nums.length <= 10^4",
        "-10^9 <= nums[i] <= 10^9",
        "Only one valid answer exists.",
    ]
    img = Image.new("RGB", (900, 780), "#ffffff")
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, 24)
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="#202124", font=sans)
        y += 46

    # Cuts sit in the blank gap between text lines; each window shares two
    # lines with its neighbour, exactly like scrolling down a page.
    windows = [(0, 390), (296, 618), (526, 766)]
    results = []
    for position, (top, bottom) in enumerate(windows):
        buf = io.BytesIO()
        img.crop((0, top, 900, bottom)).save(buf, format="PNG")
        results.extend(ocr.many(buf.getvalue(), position))

    text = reconstruct_problem(results).text

    # Nothing from the shared rows is duplicated ...
    for repeated in ("Output: [0,1]", "Output: [1,2]", "Example 1:", "Constraints:"):
        assert text.count(repeated) == 1, repeated
    # ... and the statement still reads start-to-finish.
    for a, b in zip(lines, lines[1:]):
        if a in text and b in text:
            assert text.index(a) < text.index(b), (a, b)


_DESCRIPTION = [
    "Get Minimum Removals",
    "Amazon   Medium   65% acceptance",
    "Problem Description",
    "Amazon developers are building a prototype feature that helps",
    "customers manage their cart within a given budget.",
    "You are given:",
    "- An integer budget, representing the budget of the customer.",
    "- An integer array cart_items of length n, where cart_items[i]",
    "is the price of item i.",
    "For each index i (0-based), consider the sub-cart containing",
    "items from index 0 to i, i.e., sub-cart = cart_items[0...i].",
    "For this sub-cart, find the minimum number of removals required",
    "so that the sum of the remaining items is within the budget.",
    "Removals are applied to the original cart, so an item removed",
    "for an earlier index stays removed for every later index as well.",
    "Return an array answer of length n where answer[i] is the minimum",
    "number of removals required after processing the first i+1 items.",
    "Example 1:",
    "Input: budget = 8, cart_items = [2, 3, 7, 6]",
    "Output: [0, 0, 2, 3]",
    "Explanation: With a budget of 8 and the cart [2, 3, 7, 6], the",
    "sub-cart [2] needs none. The sub-cart [2, 3] needs none either.",
    "The sub-cart [2, 3, 7] sums to 12, so we remove 7 and 3 to bring",
    "the sum down to 2, which takes 2 removals.",
    "The sub-cart [2, 3, 7, 6] sums to 18, so we remove 7, 6 and 3 to",
    "bring the sum down to 2, which takes 3 removals.",
    "Example 2:",
    "Input: budget = 5, cart_items = [1, 1, 1, 1, 1]",
    "Output: [0, 0, 0, 0, 0]",
    "Explanation: Every sub-cart sums to at most 5, so no removals.",
    "Constraints:",
    "1 <= budget <= 10^6",
    "1 <= n <= 10^5",
    "1 <= cart_items[i] <= 10^6",
]

_CODE = [
    "class Solution:",
    "    def minRemovals(self, budget, cart_items):",
    "        answer = []",
    "        removed = 0",
    "        total = 0",
    "        for i, price in enumerate(cart_items):",
    "            total += price",
    "            while total > budget:",
    "                total -= cart_items[i - removed]",
    "                removed += 1",
    "            answer.append(removed)",
    "        return answer",
]


@pytest.fixture(scope="module")
def three_pane_scroll() -> tuple[list[bytes], list[str]]:
    """A browser shot of a problem page with the editor docked right: a
    fixed icon rail and tab strip on the left, the description scrolling
    between them, and a code panel that stays put — cut at three scroll
    offsets, each landing in the blank gap between two text lines."""
    strip = Image.new("RGB", (760, 1360), "#ffffff")
    sans = _font(_SANS, 15)
    y = 20
    for line in _DESCRIPTION:
        ImageDraw.Draw(strip).text((25, y), line, fill="#202124", font=sans)
        y += 30

    shots: list[bytes] = []
    for top, bottom in ((10, 670), (340, 1000), (670, 1330)):
        img = Image.new("RGB", (1500, 820), "#ffffff")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 60, 820], fill="#f6f7f8")
        rail = _font(_SANS, 13)
        draw.text((14, 120), "Q46", fill="#3c4043", font=rail)
        draw.text((14, 400), "Menu", fill="#3c4043", font=rail)
        draw.text((14, 640), "Notes", fill="#3c4043", font=rail)
        draw.text((84, 18), "Description Solution Discussions Submissions", fill="#5f6368", font=sans)
        img.paste(strip.crop((0, top, 760, bottom)), (74, 56))
        draw.rectangle([900, 40, 1500, 400], fill="#1e1e1e")
        mono, cy = _font(_MONO, 14), 70
        for row in _CODE:
            draw.text((930, cy), row, fill="#d4d4d4", font=mono)
            cy += 26
        draw.text((100, 740), "Test Results", fill="#202124", font=sans)
        draw.text((300, 740), "Run", fill="#1a73e8", font=sans)
        draw.text((420, 740), "Submit", fill="#1a73e8", font=sans)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        shots.append(buf.getvalue())
    return shots, list(_DESCRIPTION)


def test_three_pane_scroll_keeps_pinned_panels_once(ocr, three_pane_scroll):
    """The rail, the tab strip and the docked editor are pinned to the
    window, so all three shots repeat them — and the icon rail is close
    enough to the text that it welds itself onto a sentence unless it is
    read as a column of its own."""
    shots, lines = three_pane_scroll
    results: list = []
    for position, png in enumerate(shots):
        results.extend(ocr.many(png, position))
    rec = reconstruct_problem(results)
    text = _norm(rec.text)

    assert rec.screenshot_count == 3
    # Pinned chrome shows up exactly once, on the edge it belongs to.
    assert text.count("Description Solution Discussions Submissions") == 1
    assert text.count("Notes") == 1
    assert text.count("Test Results") == 1
    assert text.count("Submit") == 1
    assert text.count("def minRemovals") == 1
    # The rail reads above the statement rather than inside a sentence.
    assert text.index("Notes") < text.index("Amazon developers are building")
    # Rows the scroll window shared are reported once ...
    assert text.count("where answer[i] is the minimum") == 1
    assert text.count("sub-cart [2] needs none") == 1
    # ... in reading order: rail, tabs, statement, then the editor.
    assert text.index(lines[0]) < text.index(lines[15])
    assert text.index(lines[15]) < text.index(lines[30])
    assert text.index(lines[30]) < text.index("def minRemovals")
    assert text.index(lines[30]) < text.index("Test Results")


_STATEMENT = [
    "Two Sum",
    "Given an array of integers nums and an integer target, return indices",
    "of the two numbers such that they add up to target.",
    "You may assume that each input would have exactly one solution, and",
    "you may not use the same element twice.",
    "Constraints:",
    "2 <= nums.length <= 10^4",
    "-10^9 <= nums[i] <= 10^9",
]


def _statement_on(bg: str, fg: str, size: int = 14) -> bytes:
    """The same statement at browser size on another palette."""
    img = Image.new("RGB", (900, 340), bg)
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, size)
    y = 30
    for line in _STATEMENT:
        draw.text((40, y), line, fill=fg, font=sans)
        y += 26
    return _png(img)


def _assert_statement_read(text: str) -> None:
    # the trailing period of a line is dropped often enough on every
    # palette that it is not worth pinning down here.
    assert "of the two numbers such that they add up to target" in text
    assert "you may not use the same element twice." in text
    assert "Constraints:" in text
    assert "2 <= nums.length <= 10^4" in text
    assert "-10^9 <= nums" in text and "<= 10^9" in text


@pytest.mark.parametrize(
    "bg,fg",
    [
        ("#ffffff", "#202124"),
        ("#000000", "#ffffff"),
        ("#1c1c1e", "#7a7a80"),
        ("#d9fdd3", "#111b21"),
        ("#001a00", "#33ff66"),
        ("#000000", "#ffb000"),
        ("#ffffff", "#c9c9ce"),
        ("#0b57d0", "#ffffff"),
        ("#ffd900", "#1a1a1a"),
    ],
)
def test_statement_reads_the_same_on_any_flat_background(ocr, bg, fg):
    """Themes, terminals, highlighters and washed-out screenshots all
    change the background colour without changing a word of the text, so
    the background must not decide what the reader survives."""
    _assert_statement_read(_norm(ocr(_statement_on(bg, fg))))


def test_statement_reads_over_a_gradient(ocr):
    """No flat colour to normalise against: the ground moves from near
    black at the top to near white at the bottom under one paragraph."""
    img = Image.new("RGB", (900, 340), "#0b1020")
    draw = ImageDraw.Draw(img)
    top, bottom = (11, 16, 32), (223, 230, 245)
    for y in range(img.height):
        t = y / img.height * 0.45
        draw.line([(0, y), (img.width, y)],
                  fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    sans = _font(_SANS, 14)
    y = 30
    for line in _STATEMENT:
        draw.text((40, y), line, fill="#ffffff", font=sans)
        y += 26
    _assert_statement_read(_norm(ocr(_png(img))))


def _wallpaper(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (900, 340), "#20304a")
    draw = ImageDraw.Draw(img)
    for _ in range(60):
        x, y = rng.randrange(900), rng.randrange(340)
        r = rng.randrange(40, 220)
        colour = tuple(rng.randrange(180, 255) for _ in range(3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
    return img


def test_statement_reads_on_a_wallpaper_behind_a_card(ocr):
    """A shared screenshot: the desktop shows through a translucent card,
    so the background varies within a single line of text."""
    img = _wallpaper(1).convert("RGBA")
    img = Image.alpha_composite(img, Image.new("RGBA", img.size, (255, 255, 255, 205)))
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, 14)
    y = 30
    for line in _STATEMENT:
        draw.text((40, y), line, fill="#202124", font=sans)
        y += 26
    _assert_statement_read(_norm(ocr(_png(img.convert("RGB")))))


def test_statement_reads_straight_on_a_busy_background(ocr):
    """No card at all — white text painted on random light patches."""
    img = _wallpaper(3)
    draw = ImageDraw.Draw(img)
    sans = _font(_SANS, 16)
    y = 30
    for line in _STATEMENT:
        draw.text((40, y), line, fill="#ffffff", font=sans)
        y += 30
    _assert_statement_read(_norm(ocr(_png(img))))


def test_syntax_highlighted_code_on_dark(ocr):
    """Editors separate tokens by hue, not only by brightness — and the
    statement to the left of them must not be sacrificed for it."""
    lines = [
        "class Solution:",
        "    def minRemovals(self, budget, cart_items):",
        "        answer = []",
        "        removed = 0",
        "        for i, price in enumerate(cart_items):",
        "            total += price",
        "            while total > budget:",
        "                total -= cart_items[i - removed]",
        "                removed += 1",
        "            answer.append(removed)",
        "        return answer",
    ]
    palette = ["#ff7b72", "#79c0ff", "#d2a8ff", "#a5d6ff", "#7ee787", "#ffa657"]
    img = Image.new("RGB", (980, 420), "#010409")
    draw = ImageDraw.Draw(img)
    mono = _font(_MONO, 16)
    y = 30
    for line in lines:
        x = 30.0
        for n, token in enumerate(re.split(r"(\W+)", line)):
            if token:
                draw.text((x, y), token, fill=palette[n % len(palette)], font=mono)
                x += mono.getlength(token)
        y += 32
    text = _norm(ocr(_png(img)))
    for line in lines:
        assert _norm(line) in text, line
