"""Token counting and the 4K tool-output cap.

Eval plan §4.4: "tool outputs capped at 4K tokens (beginning and end kept, plus error lines)".
One fixed tokenizer (cl100k_base, the one Qoherent's SFT release counts with) is used for every
model, so the cap is identical across models. The same function is handed to the training team,
because the proposal truncates training sessions at 4K tokens too (proposal §4.3). Truncating an
output is not a failure; it only shortens what the model sees.
"""

from __future__ import annotations

import re
from functools import lru_cache

TOOL_OUTPUT_CAP = 4096
ERROR_LINE = re.compile(r"(error|exception|traceback|failed|fatal|warning|invalid|not defined|"
                        r"cannot|refused|timed out)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


def _take(tokens: list[int], n: int, from_end: bool = False) -> str:
    part = tokens[-n:] if from_end else tokens[:n]
    return _enc().decode(part) if n > 0 else ""


def truncate_tool_output(text: str, cap: int = TOOL_OUTPUT_CAP) -> str:
    """Keep the beginning, the end and every error line, within `cap` cl100k tokens."""
    enc = _enc()
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= cap:
        return text
    lines = text.splitlines()
    errors = [ln for ln in lines if ERROR_LINE.search(ln)]
    error_block = ""
    err_budget = cap // 4
    if errors:
        kept, used = [], 0
        for ln in errors:
            n = len(enc.encode(ln, disallowed_special=())) + 1
            if used + n > err_budget:
                break
            kept.append(ln)
            used += n
        error_block = "\n".join(kept)
    marker_tmpl = "\n[... {n} tokens truncated by the GRC-Bench 4K tool-output cap ...]\n"
    err_part = (f"[error lines kept]\n{error_block}\n" if error_block else "")
    fixed = len(enc.encode(marker_tmpl.format(n=len(toks)) + err_part, disallowed_special=()))
    budget = max(cap - fixed, 0)
    head_n = budget // 2
    tail_n = budget - head_n
    removed = len(toks) - head_n - tail_n
    out = _take(toks, head_n) + marker_tmpl.format(n=removed) + err_part + _take(toks, tail_n, True)
    # guarantee the cap even after re-tokenisation at the joins
    while count(out) > cap and tail_n > 0:
        tail_n -= 16
        head_n = max(head_n - 16, 0)
        removed = len(toks) - head_n - tail_n
        out = _take(toks, head_n) + marker_tmpl.format(n=removed) + err_part + _take(toks, tail_n, True)
    return out
