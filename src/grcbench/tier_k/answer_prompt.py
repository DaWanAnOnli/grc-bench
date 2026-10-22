"""The Tier-K question prompt: how a question is put to a model under test (B0, and B1 on top of it).

Tier K measures what a model knows. It is closed-book: the model gets one user message holding a
fixed instruction and the question, with no GRC-Agent system prompt and no tools (eval plan §4.2). The
B1 baseline ("+ documentation search", eval plan Table 1 and Table 6) places the retrieved
documentation block before the same B0 text (`retriever.add_to_prompt`), so B0 and B1 differ only by
that block.

The instruction asks for a closing "Final answer:" line, which the rule cascade reads first
(rules.py, "1. Parse"); reasoning before it is allowed and is removed before scoring. A reply is
limited to MAX_ANSWER_TOKENS tokens, reasoning included (the engine's max_tokens; eval plan §7.1:
"an answer cut off at the limit is scored as wrong"). Sampling follows each model's recommended
settings (eval plan §7.1); nothing here sets them.

Pre-registration: PROMPT_VERSION and prompt_sha256() go into the G2 pre-registration (Appendix B,
`./gb prereg`). The hash covers the instruction, the template, the limit and the complete messages
built for PROBE_QUESTIONS in B0 and B1 form, so any change to what a model sees changes the hash;
the tests pin it. The canary string is never part of a prompt (eval plan §6): build_messages refuses
any question or retrieved block that carries it.

Template status: the instruction is drafted by Claude (Anthropic), pending team review.

    python -m grcbench.tier_k.answer_prompt [--show]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from grcbench.common.io import sha256_text
from grcbench.tier_k.judge_prompt import CanaryError, contains_canary

PROMPT_VERSION = "tier-k-answer-v1"
MAX_ANSWER_TOKENS = 4096                 # eval plan §7.1: Tier-K answers are limited to 4K tokens
INSTRUCTION = (
    "Answer the following question about radio, signal processing or GNU Radio. You may reason step by "
    "step first. End your reply with one line of the form\n"
    "Final answer: <your answer>\n"
    "If the answer is a number, give it with its unit when it has one. If the question asks you to choose, "
    "name the option.")
TEMPLATE = "{instruction}\n\nQuestion:\n{question}"
PROBE_QUESTIONS = (
    "What is the Nyquist rate of a real baseband signal limited to 10 kHz?",
    "Which GNU Radio block lowers the sample rate of a stream by keeping one item in every N?",
)
PROBE_BLOCK = "<retrieved_documentation>\nprobe documentation block\n</retrieved_documentation>"


def b0_text(question: str) -> str:
    """The B0 user message: the instruction, then the question."""
    if contains_canary(question):
        raise CanaryError("the question carries the GRC-Bench canary; it must never be inside a prompt")
    q = question.strip()
    if not q:
        raise ValueError("empty question")
    return TEMPLATE.format(instruction=INSTRUCTION, question=q)


def build_messages(question: str, retrieved: str | None = None) -> list[dict[str, str]]:
    """The request messages for one Tier-K question: B0, or B1 when `retrieved` (the retriever's block)
    is given."""
    from grcbench.tier_k.retriever import add_to_prompt
    text = b0_text(question)
    if retrieved is not None:
        if contains_canary(retrieved):
            raise CanaryError("the retrieved block carries the GRC-Bench canary")
        text = add_to_prompt(text, retrieved)
    return [{"role": "user", "content": text}]


def probe_requests() -> list[list[dict[str, str]]]:
    return [build_messages(q) for q in PROBE_QUESTIONS] + [build_messages(q, PROBE_BLOCK) for q in PROBE_QUESTIONS]


def prompt_sha256() -> str:
    parts = (PROMPT_VERSION, INSTRUCTION, TEMPLATE, str(MAX_ANSWER_TOKENS),
             json.dumps(probe_requests(), sort_keys=True, ensure_ascii=False))
    return sha256_text("\n\0\n".join(parts))


def snapshot() -> dict[str, Any]:
    """Everything the pre-registration records about the Tier-K question prompt."""
    return {"prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(),
            "max_answer_tokens": MAX_ANSWER_TOKENS, "instruction": INSTRUCTION, "template": TEMPLATE,
            "b1": "the retrieved documentation block, a blank line, then the B0 text (retriever.add_to_prompt)",
            "probes": len(PROBE_QUESTIONS)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m grcbench.tier_k.answer_prompt", description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="print the B0 message for a probe question")
    args = ap.parse_args(argv)
    print(f"{PROMPT_VERSION} {prompt_sha256()}")
    if args.show:
        print(build_messages(PROBE_QUESTIONS[0])[0]["content"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
