"""Stitching of consecutive screenshots of the same problem.

Screenshots of one problem are taken by scrolling, so the top of each
shot repeats the bottom of the previous one — and anything pinned to the
window (the icon rail, the tab strip, the code editor) repeats in every
shot whether it scrolled or not. Left alone, the same paragraph ends up
under two SCREENSHOT headers and the model reads the editor three times.
These tests cover that dedup, the duplicate/unreadable markers, and the
ordering guarantees the two-phase upload relies on.
"""

from app.models.solver import OcrResult
from app.services.problem_reconstructor import format_screenshot_blocks, reconstruct_problem


def _r(index: int, text: str, unreadable: bool = False) -> OcrResult:
    return OcrResult(index=index, text=text, unreadable=unreadable)


SEAM = "Input: nums = [2,7,11,15], target = 9"
FIRST_SHOT = f"""Two Sum
Given an array of integers nums and an integer target, return indices of
the two numbers such that they add up to target.
Example 1:
{SEAM}
Output: [0,1]"""
SECOND_SHOT = f"""{SEAM}
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0,1].
Constraints:
2 <= nums.length <= 10^4"""


def test_scroll_seam_is_not_repeated():
    out = format_screenshot_blocks([_r(0, FIRST_SHOT), _r(1, SECOND_SHOT)])
    assert out.count(SEAM) == 1
    # The repeated line stays with the earlier screenshot ...
    assert out.index(SEAM) < out.index("SCREENSHOT 2")
    # ... and the later one picks up exactly where the first ended.
    assert "SCREENSHOT 2\nExplanation: Because nums[0] + nums[1] == 9, we return [0,1]." in out


def test_overlapping_seam_is_fuzzy_match():
    """The same line re-shot at another pixel offset OCRs slightly apart."""
    prev = "Constraints:\n2 <= nums.length <= 10^4\nExplanation: Because nums[0] + nums[1] == 9, we return [0,1]."
    cur = "Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].\n-10^9 <= nums[i] <= 10^9"
    out = format_screenshot_blocks([_r(0, prev), _r(1, cur)])
    assert "SCREENSHOT 2\n-10^9 <= nums[i] <= 10^9" in out
    assert out.count("Explanation: Because nums[0]") == 1


def test_identical_screenshot_is_marked_duplicate():
    out = format_screenshot_blocks([_r(0, FIRST_SHOT), _r(1, FIRST_SHOT)])
    assert out.count("Given an array of integers") == 1
    assert "DUPLICATE" in out.split("SCREENSHOT 2", 1)[1]


def test_unreadable_shot_does_not_break_the_chain():
    """Overlap is measured against the last shot that had text, so an
    unreadable image in the middle does not leave the seam un-stitched."""
    out = format_screenshot_blocks(
        [_r(0, FIRST_SHOT), _r(1, "", unreadable=True), _r(2, SECOND_SHOT)]
    )
    assert out.count(SEAM) == 1
    assert "[UNREADABLE" in out
    assert "SCREENSHOT 3\nExplanation: Because nums[0]" in out


def test_unrelated_screenshots_keep_all_their_text():
    a = "Two Sum\nGiven an array of integers nums and a target"
    b = "House Robber\nYou are a robber facing a row of houses"
    out = format_screenshot_blocks([_r(0, a), _r(1, b)])
    assert "Given an array of integers" in out
    assert "SCREENSHOT 2\nHouse Robber" in out
    assert "row of houses" in out


def test_short_repeated_line_is_not_a_seam():
    """A single common heading is not enough evidence to delete a line."""
    out = format_screenshot_blocks(
        [_r(0, "Two Sum\nConstraints:"), _r(1, "Constraints:\n2 <= nums.length <= 10^4")]
    )
    assert out.count("Constraints:") == 2


def test_results_are_ordered_by_index_not_call_order():
    out = format_screenshot_blocks([_r(1, "second"), _r(0, "first")])
    assert out.index("SCREENSHOT 1\nfirst") < out.index("SCREENSHOT 2\nsecond")


# --- fixed chrome: pinned to the window, so it repeats in every shot ---

RAIL_TABS = "0\nQ46\nNotes\nDescription Solution Discussions Submissions"
EDITOR = "Code C++\n10 // write your code here\n14 int main() {\n37 return 0;\nSaved"


def test_fixed_chrome_survives_only_at_the_edges():
    """The rail and tab strip belong to the first shot, the code panel to
    the last; the shots in between show them without contributing them."""
    shot1 = f"{RAIL_TABS}\nAmazon developers are building a prototype feature\nYou are given:\n- An integer budget, representing the budget of the customer.\n{EDITOR}"
    shot2 = f"{RAIL_TABS}\n- An integer budget, representing the budget of the customer.\n- An integer array cart_items of length n.\n{EDITOR}"
    shot3 = f"{RAIL_TABS}\n- An integer array cart_items of length n.\nConstraints:\n2 <= nums.length <= 10^4\n{EDITOR}"

    out = format_screenshot_blocks([_r(0, shot1), _r(1, shot2), _r(2, shot3)])

    assert out.count("SCREENSHOT") == 3  # nothing collapsed to DUPLICATE
    assert out.count("Description Solution Discussions Submissions") == 1
    assert out.count("Notes") == 1
    assert out.count("Saved") == 1
    assert out.count("Amazon developers") == 1
    # Each bullet shows once, in reading order, with no other shot's copy.
    assert out.count("- An integer budget, representing the budget of the customer.") == 1
    assert out.count("- An integer array cart_items of length n.") == 1
    assert out.count("Constraints:") == 1
    order = ["Amazon developers", "- An integer budget", "- An integer array", "Constraints:", "Saved"]
    positions = [out.index(needle) for needle in order]
    assert positions == sorted(positions)


def test_one_frame_that_failed_to_read_does_not_end_the_header():
    """A chrome line that came back as noise in a single frame is still
    chrome: the run only pauses for it, it does not stop being stripped."""
    header = "0\nQ46\nNotes\nDescription Solution Discussions Submissions"
    foot = EDITOR
    shot1 = f"{header}\nSave Images Report\nHelp Center\nTwo Sum\nGiven an array of integers nums and an integer target, return the indices\n{foot}"
    shot2 = f"{header}\n@@@@\nHelp Center\nYou are given an array of integers\nand a target value; find the two numbers\n{foot}"
    shot3 = f"{header}\nSave Images Report\nHelp Center\nReturn the indices of the two numbers\nwhose sum equals the target\n{foot}"

    out = format_screenshot_blocks([_r(0, shot1), _r(1, shot2), _r(2, shot3)])

    assert "@@@@" not in out
    assert out.count("Help Center") == 1
    assert out.count("Save Images Report") == 1
    assert "Two Sum" in out and "Return the indices" in out


def test_seam_is_found_when_the_paragraph_rewraps():
    """The same sentence broken at a different word: line-level matching
    would see no overlap at all, character-level matching sees one."""
    prev = (
        "Two Sum\n"
        "Given an array of integers nums and an integer target, return\n"
        "indices of the two numbers such that they add up to target.\n"
        "Constraints:"
    )
    cur = (
        "Given an array of integers nums and an integer target, return indices of the two\n"
        "numbers such that they add up to target.\n"
        "Constraints:\n"
        "2 <= nums.length <= 10^4"
    )
    out = format_screenshot_blocks([_r(0, prev), _r(1, cur)])
    assert out.count("Constraints:") == 1
    assert "SCREENSHOT 2\n2 <= nums.length <= 10^4" in out
    assert out.count("add up to target.") == 1


def test_a_shared_tail_line_does_not_delete_new_content():
    """Two unrelated screenshots that both end on the constraints block:
    the overlap is far too short to license dropping the new heading."""
    prev = "Two Sum\nGiven an array of integers nums and a target\nConstraints:\n2 <= nums.length <= 10^4"
    cur = "House Robber\nConstraints:\n2 <= nums.length <= 10^4"
    out = format_screenshot_blocks([_r(0, prev), _r(1, cur)])
    assert "SCREENSHOT 2\nHouse Robber" in out
    assert out.count("2 <= nums.length <= 10^4") == 2


def test_reconstruction_frames_the_shots_as_one_problem():
    rec = reconstruct_problem([_r(0, FIRST_SHOT), _r(1, SECOND_SHOT)])
    assert rec.screenshot_count == 2
    assert "SAME problem" in rec.text
    assert "repeated lines have already been removed" in rec.text
    # The stitched statement itself follows the preamble.
    assert rec.text.index("SCREENSHOT 1") > rec.text.index("SAME problem")
    assert rec.text.count(SEAM) == 1


def test_stitch_false_keeps_every_screenshot_verbatim():
    """Two screenshots that share several lines of judge boilerplate but are
    actually DIFFERENT problems: with stitching on, the shared lines would be
    treated as repeated chrome/scroll overlap and dropped from one of them.
    With stitch=False nothing is ever merged or deleted."""
    boilerplate = "Time Limit: 1 second\nMemory Limit: 256 MB\nConstraints:\n1 <= n <= 10^5"
    shot_a = f"Problem A: Sum Pairs\n{boilerplate}\nReturn the count of pairs."
    shot_b = f"Problem B: Max Subarray\n{boilerplate}\nReturn the maximum sum."

    out = format_screenshot_blocks([_r(0, shot_a), _r(1, shot_b)], stitch=False)

    assert "Problem A: Sum Pairs" in out
    assert "Problem B: Max Subarray" in out
    assert "Return the count of pairs." in out
    assert "Return the maximum sum." in out
    assert "DUPLICATE" not in out
    # Boilerplate legitimately appears once per screenshot, not collapsed.
    assert out.count("Time Limit: 1 second") == 2


def test_stitch_false_still_marks_unreadable():
    out = format_screenshot_blocks([_r(0, "", unreadable=True), _r(1, "Some text")], stitch=False)
    assert "SCREENSHOT 1\n" in out
    assert "[UNREADABLE" in out
    assert "SCREENSHOT 2\nSome text" in out


def test_stitch_false_never_marks_duplicate_even_for_identical_shots():
    """stitch=True collapses an identical re-upload into DUPLICATE (see
    test_identical_screenshot_is_marked_duplicate above); stitch=False must
    never do that, since the caller has said not to de-duplicate at all."""
    out = format_screenshot_blocks([_r(0, FIRST_SHOT), _r(1, FIRST_SHOT)], stitch=False)
    assert "DUPLICATE" not in out
    assert out.count("Given an array of integers") == 2


def test_reconstruct_problem_stitch_false_preamble_warns_not_continuous():
    result = reconstruct_problem([_r(0, "A"), _r(1, "B")], stitch=False)
    assert "SAME problem" not in result.text
    assert "may be different parts" in result.text or "entirely different problems" in result.text


def test_seam_false_positive_on_long_screenshot_is_rejected_not_swallowed():
    """A later screenshot whose entire (long) body happens to fuzzy-match the
    tail of the previous one -- generic constraints/example wording repeated
    verbatim, not a real scroll seam covering nearly 100% of a real chunk of
    problem text -- must not be dropped wholesale. Losing an entire
    screenshot's real content this way is exactly the silent-data-loss
    failure mode that produces a wrong final answer."""
    shared_tail = (
        "Constraints:\n1 <= n <= 10^5\n1 <= arr[i] <= 10^9\nTime limit: 2 seconds\n"
        "Memory limit: 256 MB\nOutput the answer modulo 10^9 + 7.\n"
        "Print a single integer.\nRead input from standard input."
    )
    prev = f"Some earlier part of the statement.\n{shared_tail}"
    # `cur` is a DIFFERENT, later part of the same long problem that happens
    # to restate the same boilerplate near-verbatim (common on CP judges)
    # before its own genuinely new content.
    cur = f"{shared_tail}\nThis is new content that must not be lost: return the count of valid arrays."

    out = format_screenshot_blocks([_r(0, prev), _r(1, cur)])
    assert "This is new content that must not be lost" in out


def test_many_screenshots_of_one_long_problem_keep_all_unique_content():
    """A 10-screenshot scroll of one long problem: every screenshot
    contributes some genuinely new line, and none of it should vanish just
    because there are many screenshots (and therefore many chances for a
    coincidental match) instead of just two."""
    shots = []
    for i in range(10):
        shots.append(
            _r(
                i,
                f"Constraints:\n1 <= n <= 10^5\nExample section shared wording here\n"
                f"UNIQUE_MARKER_{i}: this line must survive to the final prompt",
            )
        )
    out = format_screenshot_blocks(shots)
    for i in range(10):
        assert f"UNIQUE_MARKER_{i}: this line must survive to the final prompt" in out


def test_panel_run_capped_at_max_lines():
    """A run of identical lines across every screenshot longer than
    _PANEL_MAX_LINES is capped rather than trusted in full, since real fixed
    chrome is short and a run this long is more likely a false positive."""
    long_shared_header = "\n".join(f"chrome line {i}" for i in range(20))
    shot1 = f"{long_shared_header}\nFirst unique body line."
    shot2 = f"{long_shared_header}\nSecond unique body line."
    out = format_screenshot_blocks([_r(0, shot1), _r(1, shot2)])
    # Some of the shared header survives on SCREENSHOT 2 rather than all 20
    # lines being silently trusted as chrome.
    assert out.count("chrome line 0") == 1
    kept_on_second = out.split("SCREENSHOT 2", 1)[1]
    assert "chrome line 19" in kept_on_second
    assert "Second unique body line." in kept_on_second


def test_incrementing_page_indicator_is_not_treated_as_fixed_chrome():
    """A per-screenshot counter/page indicator ('Q46', 'Q47', ...) sits at a
    fixed position right next to real chrome and is textually almost
    identical to it -- differing only in the number. A plain character-ratio
    comparison sees ~98% similarity and would wrongly treat the *whole*
    line, counter included, as repeated chrome to be stripped from every
    screenshot but the first. It must not be: the counter (and anything
    bundled with it) is per-screenshot content, not chrome."""
    shots = [
        _r(i, f"Q{46 + i}\nDescription Solution Discussions\nActual unique problem text {i}")
        for i in range(6)
    ]
    out = format_screenshot_blocks(shots)
    for i in range(6):
        assert f"Actual unique problem text {i}" in out
