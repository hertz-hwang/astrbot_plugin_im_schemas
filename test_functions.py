"""Test the core encoding functions directly."""

import sys
sys.path.insert(0, '/Users/bennett/workspace/astrbot/data/plugins/astrbot_plugin_im_schemas')

# Copy just the functions we need to test
import unicodedata
from typing import Optional

def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")

def _code_display(code: str, max_len: int, pos: int = 1, select_keys: str = "", candidate_count: int = 1) -> str:
    if pos > 1:
        sym = select_keys[pos - 2] if pos <= len(select_keys) else ""
        if code.endswith(sym):
            return code
        return code + sym
    if len(code) < max_len or candidate_count > 1:
        first_key = select_keys[0] if select_keys else "_"
        return code + first_key
    return code

def _omit_in_punct_context(
    code: str,
    pos: int,
    max_len: int,
    select_keys: str,
    is_punct_follow: bool,
    candidate_count: int = 1,
) -> Optional[tuple[int, str]]:
    if pos == 1:
        # Followed by punct: always omit (any len(code))
        if is_punct_follow:
            return len(code), code
        # Followed by space/letter/digit: only omit if len(code)<max_len and candidate_count==1
        if len(code) < max_len and candidate_count == 1:
            return len(code), code
    return None

# Simulate segments
class Segment:
    def __init__(self, text, code, pos, candidate_count):
        self.text = text
        self.code = code
        self.pos = pos
        self.candidate_count = candidate_count
        self.is_self_coded = (code is None)

def simulate_encode(text, max_len=4, select_keys="_"):
    """Simulates the encoding logic."""
    segments = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch.isdigit() or (ch.isascii() and ch.isalpha()):
            segments.append(Segment(ch, None, 1, 1))
            i += 1
        elif _is_punct(ch):
            segments.append(Segment(ch, None, 1, 1))
            i += 1
        else:
            if i + 1 < len(text) and text[i:i+2] == "木工":
                segments.append(Segment("木工", "yqiv", 1, 2))
                i += 2
            elif i + 1 < len(text) and text[i:i+2] == "瓦匠":
                segments.append(Segment("瓦匠", "ijxt", 1, 1))
                i += 2
            elif ch == "木":
                segments.append(Segment("木", "yqiv", 1, 2))
                i += 1
            elif ch == "工":
                segments.append(Segment("工", "ijxt", 1, 2))
                i += 1
            else:
                segments.append(Segment(ch, "xxxx", 1, 1))
                i += 1

    n = len(segments)
    key_seq_parts = []

    def _next_self_coded_punct(i):
        return i + 1 < n and segments[i + 1].is_self_coded and _is_punct(segments[i + 1].text)

    def _next_is_word(i):
        return i + 1 < n and not segments[i + 1].is_self_coded

    for idx, seg in enumerate(segments):
        is_last = (idx == n - 1)

        if seg.is_self_coded:
            if seg.text == " ":
                key_seq_parts.append("＿")
            else:
                key_seq_parts.append(seg.text)
        else:
            code = seg.code
            pos = seg.pos
            candidate_count = seg.candidate_count

            # Check for omit (only when followed by punctuation)
            omit = _omit_in_punct_context(code, pos, max_len, select_keys, True, candidate_count) if _next_self_coded_punct(idx) else None

            if omit is not None:
                key_seq_parts.append(code)
            elif is_last:
                disp = _code_display(code, max_len, pos, select_keys, candidate_count)
                key_seq_parts.append(disp)
            elif _next_is_word(idx):
                # Followed by Chinese word: omit if pos==1 (regardless of candidate_count)
                if pos == 1:
                    key_seq_parts.append(code)
                else:
                    disp = _code_display(code, max_len, pos, select_keys, candidate_count)
                    key_seq_parts.append(disp)
            else:
                # Followed by space/punct: use _code_display
                disp = _code_display(code, max_len, pos, select_keys, candidate_count)
                key_seq_parts.append(disp)

    return "".join(key_seq_parts)

# Run tests
tests = [
    ("木工，瓦匠", "yqiv，ijxt"),
    ("木工 瓦匠", "yqiv_＿ijxt"),
    ("木工瓦匠", "yqivijxt"),  # When followed by Chinese, omit first key if pos==1 (regardless of candidate_count)
]

print("Testing encoding functions...")
all_pass = True
for text, expected in tests:
    result = simulate_encode(text, max_len=4, select_keys="_")
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    print(f"{status} Input: {text!r}")
    print(f"  Expected: {expected!r}")
    print(f"  Got:      {result!r}")
    print()

print("All tests passed!" if all_pass else "Some tests failed!")