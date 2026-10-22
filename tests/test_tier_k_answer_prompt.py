"""The Tier-K question prompt (B0 and B1): pinned, closed-book, canary-free, readable by the rule cascade."""

from __future__ import annotations

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.tier_k import answer_prompt as ap
from grcbench.tier_k import retriever, rules
from grcbench.tier_k.judge_prompt import CanaryError

PINNED = "16891eff0f9f05392c48e461148daef38e5b8b502e8c5c5b52296c57378f0d65"


def test_prompt_is_pinned():
    # changing what a model under test sees needs a new PROMPT_VERSION and a new pinned hash
    assert ap.PROMPT_VERSION == "tier-k-answer-v1"
    assert ap.prompt_sha256() == PINNED


def test_b0_is_one_user_message_without_tools_or_agent_prompt():
    msgs = ap.build_messages("What is 2 + 2?")
    assert [m["role"] for m in msgs] == ["user"]
    text = msgs[0]["content"]
    assert text.startswith(ap.INSTRUCTION) and text.endswith("Question:\nWhat is 2 + 2?")
    assert "GRC-Agent" not in text and "tool" not in text.lower()


def test_b1_puts_the_retrieved_block_before_the_unchanged_b0_text():
    block = "<retrieved_documentation>\nsome page\n</retrieved_documentation>"
    b0 = ap.build_messages("What is 2 + 2?")[0]["content"]
    b1 = ap.build_messages("What is 2 + 2?", block)[0]["content"]
    assert b1 == retriever.add_to_prompt(b0, block) == f"{block}\n\n{b0}"


def test_the_canary_never_enters_a_prompt():
    with pytest.raises(CanaryError):
        ap.build_messages(f"Question with {CANARY_GUID}")
    with pytest.raises(CanaryError):
        ap.build_messages("What is 2 + 2?", f"block {CANARY_GUID}")


def test_the_requested_answer_line_is_what_the_rule_cascade_reads():
    reply = "The sum of 2 and 3 is 5.\nFinal answer: 5"
    assert rules.score(reply, {"value": 5.0, "unit": "", "tolerance": 1e-9, "abs_tolerance": 0.01})["credit"] == 1.0
