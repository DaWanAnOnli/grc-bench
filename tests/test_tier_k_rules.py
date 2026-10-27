"""Tests for `./gb score-k`: the rule-first Tier-K scoring cascade (proposal §5.1; eval plan §4.2, §5.2,
§7.1). Every answer here is hand-made; no benchmark item is used and nothing is sent to a model."""

from __future__ import annotations

import json
import math

import pytest

from grcbench.common.io import read_jsonl, write_jsonl
from grcbench.common.paths import REPO
from grcbench.tier_k import rules
from grcbench.tier_k.rules import (extract_quantities, final_answer, parse_reference, parse_unit,
                                   rule_only_credit, score, strip_reasoning)

K30 = {"value": 30, "unit": "kHz"}
TYPES = ["byte", "complex", "float", "int", "short"]
JUDGE = ("judge", None)


def credit(answer, reference, **kw):
    r = score(answer, reference, **kw)
    return r["method"], r["credit"]


def reason(answer, reference, **kw):
    return score(answer, reference, **kw)["details"]["reason"]


# --- the proposal's requirements ------------------------------------------------------------------


@pytest.mark.parametrize("answer", [
    "The output runs at 2500 kS/s.", "2.5 MS/s", "2.5 Msps", "2,500,000 samples per second",
    "2.5e6 S/s", "2.5 × 10^6 Sa/s", r"$2.5\times10^{6}\,\text{S/s}$", "samp_rate = 2.5 MHz",
    "2500 ksamples/s", "Answer: 2.5M", "Answer: 2500 kSamples/s", "Answer: 2.5 MSample/s",
    "Answer: 2500 KSamples/sec", "Answer: 2500 kilo samples per second",
])
def test_unit_conversion_equal_rates(answer):
    assert credit(answer, {"value": 2.5, "unit": "MS/s"}) == ("numeric", 1.0)
    assert credit(answer, {"value": 2500, "unit": "kS/s"}) == ("numeric", 1.0)


@pytest.mark.parametrize("answer,reference", [
    ("The output power is 27 dB.", {"value": 27, "unit": "dBm"}),     # dB where dBm is meant
    ("27 dBm", {"value": 27, "unit": "dB"}),
    ("27 dBW", {"value": 27, "unit": "dBi"}),
    ("The period is 0.4 µs.", {"value": 2.5, "unit": "MHz"}),        # Hz vs s
    ("About 1.2 MHz.", {"value": 1.2, "unit": "Mbit/s"}),            # bit/s vs Hz
    ("2.5 Msym/s", {"value": 2.5, "unit": "MS/s"}),                  # symbols vs samples
    ("6.28 rad/s", {"value": 1, "unit": "Hz"}),                      # rad/s vs Hz
])
def test_unit_dimension_mismatch_scores_zero(answer, reference):
    r = score(answer, reference)
    assert (r["method"], r["credit"]) == ("numeric", 0.0)
    assert r["details"]["reason"] == "unit_mismatch"


def test_bits_and_bytes():
    assert credit("Answer: 1.2 Mbps", {"value": 1.2, "unit": "Mbit/s"}) == ("numeric", 1.0)
    assert credit("Answer: 1.2 MBps", {"value": 9.6, "unit": "Mbit/s"}) == ("numeric", 1.0)
    assert credit("Answer: 1.2 MBps", {"value": 1.2, "unit": "Mbit/s"}) == ("numeric", 0.0)
    assert credit("150 kB/s", {"value": 1.2, "unit": "Mbit/s"}) == ("numeric", 1.0)
    assert credit("1 KiB", {"value": 8192, "unit": "bits"}) == ("numeric", 1.0)


def test_number_words_with_digits():
    assert credit("1.5 million samples per second", {"value": 1.5, "unit": "MS/s"}) == ("numeric", 1.0)
    assert credit("Answer: 30 thousand", K30) == ("numeric", 1.0)


def test_db_and_linear_power_convert():
    assert credit("Final answer: 1 W", {"value": 30, "unit": "dBm"}) == ("numeric", 1.0)
    assert credit("0 dBW", {"value": 30, "unit": "dBm"}) == ("numeric", 1.0)
    assert credit("30 dBm", {"value": 1, "unit": "W"}) == ("numeric", 1.0)
    assert credit("10 dBi", {"value": 7.85, "unit": "dBd"}) == ("numeric", 1.0)


def test_temperatures_convert():
    assert credit("About 17 degrees Celsius", {"value": 290, "unit": "K"}) == ("numeric", 1.0)
    assert credit("Answer: 16.85 °C", {"value": 290, "unit": "K"}) == ("numeric", 1.0)
    assert credit("Answer: 290 K", {"value": 16.85, "unit": "°C"}) == ("numeric", 1.0)
    assert credit("Answer: 62.33 °F", {"value": 290, "unit": "K"}) == ("numeric", 1.0)


@pytest.mark.parametrize("answer", ["About 25 MHz.", "About 0.25 MHz.", "2.5 GHz", "2.5 kHz", "250 kHz"])
def test_order_of_magnitude_error_scores_zero_even_with_wide_tolerance(answer):
    r = score(answer, {"value": 2.5, "unit": "MHz", "tolerance": 0.95})
    assert (r["method"], r["credit"], r["details"]["reason"]) == ("numeric", 0.0, "order_of_magnitude")


def test_order_of_magnitude_threshold():
    ref = {"value": 2.0, "unit": "", "tolerance": 100.0}           # tiers alone would give full credit
    assert credit("18.8", ref) == ("numeric", 1.0)
    assert credit("19", ref) == ("numeric", 0.0)                     # 9.5 x
    assert credit("0.212", ref) == ("numeric", 1.0)
    assert credit("0.21", ref) == ("numeric", 0.0)                   # 1/9.5 x
    assert credit("-2", ref) == ("numeric", 0.0)                     # a sign flip
    assert reason("40 dBm", {"value": 30, "unit": "dBm", "abs_tolerance": 20}) == "within_tolerance"
    assert reason("40 dBm", {"value": 30, "unit": "dBm", "tolerance_db": 9}) == "order_of_magnitude"
    # an explicit absolute tolerance (values near zero) is the one exemption
    assert credit("0.04 V", {"value": 0.5, "unit": "V", "abs_tolerance": 1}) == ("numeric", 1.0)
    assert credit("-0.2 V", {"value": 0.5, "unit": "V", "abs_tolerance": 1}) == ("numeric", 1.0)
    assert credit("0.04 V", {"value": 0.5, "unit": "V", "tolerance": 1}) == ("numeric", 0.0)


# --- credit tiers (finding F7: the half band is capped; F1: no absolute floor) ---------------------


@pytest.mark.parametrize("answer,expected", [
    ("1 kHz", 1.0), ("1.01 kHz", 1.0), ("0.99 kHz", 1.0), ("1.011 kHz", 0.5), ("1.05 kHz", 0.5),
    ("0.95 kHz", 0.5), ("1.051 kHz", 0.0), ("1.2 kHz", 0.0),
])
def test_tolerance_tiers_default(answer, expected):
    assert rules.CREDIT_TIERS == ((1.0, 1.0), (5.0, 0.5)) and rules.DEFAULT_TOLERANCE == 0.01
    assert rules.HALF_BAND_EXTRA == 0.04
    assert credit(answer, {"value": 1, "unit": "kHz"}) == ("numeric", expected)


def test_half_band_is_capped_at_tolerance_plus_four_points():
    ref = {"value": 100, "unit": "Hz", "tolerance": 0.02}             # full <= 2 %, half <= 6 %
    assert credit("102 Hz", ref) == ("numeric", 1.0)
    assert credit("106 Hz", ref) == ("numeric", 0.5)
    assert credit("107 Hz", ref) == ("numeric", 0.0)
    small = {"value": 100, "unit": "Hz", "tolerance": 0.005}          # 5 x 0.5 % = 2.5 % < 4.5 %
    assert credit("102.5 Hz", small) == ("numeric", 0.5)
    assert credit("103 Hz", small) == ("numeric", 0.0)
    wide = {"value": 1, "unit": "kHz", "tolerance": 0.1}              # full <= 10 %, half <= 14 %
    assert credit("Final answer: 1.12 kHz", wide) == ("numeric", 0.5)
    assert credit("Final answer: 1.15 kHz", wide) == ("numeric", 0.0)


@pytest.mark.parametrize("answer,reference", [
    ("Final answer: 0.5 kHz", {"value": 1, "unit": "kHz", "tolerance": 0.1}),     # a halving
    ("Final answer: 2 kHz", {"value": 1, "unit": "kHz", "tolerance": 0.2}),       # a doubling
    ("Final answer: 6 s", {"value": 10, "unit": "s", "tolerance": 0.08}),         # 40 % off
    ("Final answer: 0.5 kHz", {"value": 1, "unit": "kHz", "tolerance": 0.2}),
])
def test_factor_two_mistakes_get_no_credit_at_wide_tolerances(answer, reference):
    assert credit(answer, reference) == ("numeric", 0.0)


def test_exact_and_zero_references():
    exact = {"value": 4, "unit": "", "tolerance": 0}
    assert credit("4", exact) == ("numeric", 1.0)
    assert credit("4.001", exact) == ("numeric", 0.0)
    assert credit("2.4 GHz", {"value": 2.4e9, "unit": "Hz", "tolerance": 0}) == ("numeric", 1.0)
    assert credit("0 Hz", {"value": 0, "unit": "Hz"}) == ("numeric", 1.0)
    assert credit("0.5 Hz", {"value": 0, "unit": "Hz", "abs_tolerance": 1}) == ("numeric", 1.0)
    assert credit("1e-15 Hz", {"value": 0, "unit": "Hz"}) == ("numeric", 0.0)


@pytest.mark.parametrize("value,unit", [(3e-38, ""), (4.2e-21, "W/Hz"), (1.4e-23, "J/K"), (2e-20, "W")])
def test_tiny_references_keep_relative_tiers(value, unit):
    ref = {"value": value, "unit": unit}
    assert credit(f"Final answer: {value:.6g} {unit}", ref) == ("numeric", 1.0)
    assert credit(f"Final answer: {value * 1.005:.6g} {unit}", ref) == ("numeric", 1.0)
    assert credit(f"Final answer: {value * 1.03:.6g} {unit}", ref) == ("numeric", 0.5)
    for factor in (2, 1.5, 1.25, 0.6):
        assert credit(f"Final answer: {value * factor:.6g} {unit}", ref) == ("numeric", 0.0), factor


# --- decibel tolerances (finding F6) ----------------------------------------------------------------


def test_db_tolerance_is_absolute_in_db():
    assert credit("-114.05 dBm", {"value": -114, "unit": "dBm"}) == ("numeric", 1.0)    # default 0.1 dB
    assert credit("-114.2 dBm", {"value": -114, "unit": "dBm"}) == ("numeric", 0.5)     # <= 0.1 + 0.17 dB
    assert credit("-114.3 dBm", {"value": -114, "unit": "dBm"}) == ("numeric", 0.0)
    assert credit("-114.6 dBm", {"value": -114, "unit": "dBm", "abs_tolerance": 0.5}) == ("numeric", 0.5)
    assert credit("-114.6 dBm", {"value": -114, "unit": "dBm", "tolerance_db": 1}) == ("numeric", 1.0)
    assert credit("Final answer: 20.05 dB", {"value": 20, "unit": "dB", "tolerance_db": 0.1}) == ("numeric", 1.0)


def test_db_reference_with_only_a_relative_tolerance_is_refused():
    for ref in ({"value": -114, "unit": "dBm", "tolerance": 0.01}, {"value": 20, "unit": "dB", "tolerance": 0.0233},
                {"value": 6, "unit": "dBi", "tolerance": 0.5}):
        with pytest.raises(ValueError, match="in dB"):
            score("20 dB", ref)
    with pytest.raises(ValueError):
        parse_reference({"value": 1, "unit": "kHz", "tolerance_db": 0.1})          # dB tolerance, linear unit
    with pytest.raises(ValueError):
        parse_reference({"value": 1, "unit": "dB", "abs_tolerance": 0.1, "tolerance_db": 0.2})
    # calc.py writes both: the relative equivalent and the tolerance in dB, which wins
    calc_style = {"value": 20, "unit": "dB", "tolerance": 10 ** 0.01 - 1, "abs_tolerance": 0.1}
    assert parse_reference(calc_style).db_tolerance == 0.1
    assert credit("Final answer: 20.05 dB", calc_style) == ("numeric", 1.0)
    # the rubric guide's decibel form (methodology-F6; docs/guidelines/rubric_writing.md)
    assert parse_reference({"value": -114, "unit": "dBm", "abs_tolerance": 0.5}).db_tolerance == 0.5


# --- reasoning text and truncation ----------------------------------------------------------------


@pytest.mark.parametrize("answer", [
    "<think>Maybe 20 kHz? No.</think>The minimum rate is 30 kHz.",
    "<thinking>20 kHz</thinking>\n30 kHz",
    "Let me think: 20 kHz.</think>\n\nThe minimum rate is 30 kHz.",      # template opened the block
    "[THINK]20 kHz[/THINK]30 kHz",
    "<|channel|>analysis<|message|>20 kHz<|end|><|start|>assistant<|channel|>final<|message|>30 kHz",
])
def test_thinking_is_stripped(answer):
    assert credit(answer, K30) == ("numeric", 1.0)


def test_only_the_answer_after_thinking_counts():
    assert credit("<think>It is 30 kHz.</think>The minimum rate is 20 kHz.", K30) == ("numeric", 0.0)
    assert credit("<think>So the rate is 30 kHz and", K30) == ("numeric", 0.0)    # cut off while thinking
    assert strip_reasoning("<think>a</think>b") == "b"
    assert strip_reasoning("<think>never closed 30 kHz") == ""
    assert strip_reasoning("b <reasoning>x</reasoning> c") == "b  c"


def test_truncated_and_empty_answers_score_zero():
    assert reason("30 kHz", K30, truncated=True) == "truncated"
    assert credit("30 kHz", K30, truncated=True) == ("numeric", 0.0)
    assert credit("", K30) == ("numeric", 0.0)
    assert credit(None, K30) == ("numeric", 0.0)
    assert credit("<think>x</think>", "blocks_throttle") == ("symbolic", 0.0)


# --- several numbers: the final stated answer (finding F2) ------------------------------------------


@pytest.mark.parametrize("answer", [
    "With B = 15 kHz, fs = 2 × 15 kHz = 30 kHz.",                     # a derivation chain
    "We need 2 × 15 kHz = 30 kHz.",
    "fs = 2B = 2 x 15 kHz = 30 kHz minimum",
    "The bandwidth is 15 kHz, so the sample rate must be 30 kHz.",    # premise, conclusion word
    "The bandwidth is 15 kHz, so the sample rate should be 30 kHz.",
    "Since B = 15 kHz, Nyquist gives 30 kHz.",
    "We decimate the 120 kHz input by 4, giving 30 kHz.",
    "Nyquist: 2 × 15 kHz, so 30 kHz.",
    "The rate is 30 kHz, so each sample lasts 33.3 µs.",              # a later quantity of another unit
    "The rate is 30 kHz (twice the 15 kHz bandwidth).",               # a parenthetical explanation
    "Final answer: 30 kHz\n\nA 37.5 kHz rate leaves a guard band.",   # the marker decides
    r"So \boxed{30\ \text{kHz}}. Audio often uses 48 kHz.",
    "Final answer: 30 kHz or 30000 Hz",                               # one value, two spellings
    "fs = 30000 Hz = 30 kHz",
])
def test_the_final_stated_answer_is_decided(answer):
    assert credit(answer, K30) == ("numeric", 1.0)


@pytest.mark.parametrize("answer", [
    "Final answer: 30 kHz, giving a Nyquist frequency of 15 kHz.",
    "Final answer: 30 kHz, so the Nyquist frequency is 15 kHz.",
    "**Answer:** fs = 30 kHz, so the usable bandwidth is 15 kHz.",
    r"\boxed{f_s = 30\text{ kHz}, \text{ so } f_N = 15\text{ kHz}}",
])
def test_a_consequence_after_the_stated_answer_goes_to_the_judge(answer):
    """Round-2 finding R1: the first value after an explicit marker (or in a box) is the stated answer,
    so a later derived value is a rival, not the answer; whichever is right, the judge decides."""
    assert credit(answer, K30) == JUDGE
    assert credit(answer, {"value": 15, "unit": "kHz"}) == JUDGE


@pytest.mark.parametrize("answer", [
    "Final answer: 12 kHz\n\nNote: this answer is for a 48 kHz input; for 44.1 kHz it would be 11.025 kHz.",
    "Final answer: 12 kHz. The answer is valid for the 48 kHz input only.",
    "**Final Answer:** 12 kHz\n\nThe answer is the same for a 96 kHz input decimated by 8.",
])
def test_a_prose_marker_never_replaces_an_explicit_one(answer):
    """Round-2 finding R2: 'the answer is' in a note after 'Final answer:' is not a new marker."""
    assert credit(answer, {"value": 12, "unit": "kHz"}) == ("numeric", 1.0)
    assert credit("The answer is 12 kHz.", {"value": 12, "unit": "kHz"}) == ("numeric", 1.0)   # alone it counts


@pytest.mark.parametrize("answer,values", [
    ("It is 30 kHz, or maybe 20 kHz.", (20, 30)), ("It is 20 kHz, or maybe 30 kHz.", (20, 30)),   # both orders
    ("Final answer: 15 kHz, 20 kHz, 25 kHz or 30 kHz", (15, 20, 25, 30)),
    ("Final answer: 10 kHz / 20 kHz / 30 kHz", (10, 20, 30)),
    ("Final answer: either 60 kHz or 30 kHz", (30, 60)), ("Final answer: 30 kHz or 60 kHz", (30, 60)),
    ("Final answer: 20-30 kHz", (20, 30)), ("Final answer: 20 to 30 kHz", (20, 30)),
    ("Somewhere between 20 kHz and 30 kHz.", (20, 30)),
    ("If the input is complex the rate is 20 kHz; otherwise 30 kHz.", (20, 30)),
    ("A first guess is 20 kHz; with the factor of two it is 30 kHz.", (20, 30)),     # a revision
    ("Final answer: 15 kHz (no, it should be 30 kHz)", (15, 30)),
    ("30 kHz = 2 × 15 kHz", (15, 30)),                                               # the candidate is an operand
    ("It must be at least 2 × 15 kHz = 30 kHz, or 32 kHz with margin", (30, 32)),
])
def test_hedged_or_conflicting_answers_go_to_the_judge_whichever_value_is_right(answer, values):
    for value in values:
        assert credit(answer, {"value": value, "unit": "kHz"}) == JUDGE, (answer, value)
    assert credit(answer, {"value": 90, "unit": "kHz"}) == ("numeric", 0.0)   # no reading earns credit


def test_hedge_reasons_and_details():
    r = score("It is 30 kHz, or maybe 20 kHz.", K30)
    assert r["details"]["reason"] == "hedged_answer" and r["details"]["rule_method"] == "numeric"
    assert set(r["details"]["candidates"]) == {"20 kHz", "30 kHz"}
    r = score("A first guess is 20 kHz; with the factor of two it is 30 kHz.", K30)
    assert r["details"]["reason"] == "ambiguous_final_answer" and r["details"]["rivals"] == ["20 kHz"]
    # values that would all earn the same credit do not matter
    assert credit("Answer: 10 kHz or 20 kHz", K30) == ("numeric", 0.0)


def test_context_values_go_to_the_judge_in_both_directions():
    text = "Decimating by 4 gives 12 kHz from the original 48 kHz."
    assert credit(text, {"value": 48, "unit": "kHz"}) == JUDGE
    assert credit(text, {"value": 12, "unit": "kHz"}) == JUDGE
    text = "Answer: 30. The input was 48 kHz."                  # a bare candidate next to a rate
    assert credit(text, K30) == JUDGE and credit(text, {"value": 48, "unit": "kHz"}) == JUDGE
    text = "Answer: 30.\n\nThe input was 48 kHz."               # another paragraph: the marker decides
    assert credit(text, K30) == ("numeric", 1.0) and credit(text, {"value": 48, "unit": "kHz"}) == ("numeric", 0.0)
    assert credit("Final answer: 1. 30 kHz", K30) == JUDGE       # an inline list marker or the number 1


def test_negation_near_the_answer_goes_to_the_judge():
    for text in ("The answer is not 30 kHz.", "Final answer: not 30 kHz but 40 kHz", "Answer: 30 kHz, not 20 kHz.",
                 "The answer is 30 kHz rather than 20 kHz.", "Final answer: 30 kHz (not 20 kHz)"):
        assert credit(text, K30) == JUDGE, text
        assert credit(text, {"value": 20, "unit": "kHz"}) == JUDGE, text
    assert credit("The minimum rate is 30 kHz; anything lower would not work.", K30) == ("numeric", 1.0)


def test_rule_only_credit_counts_a_judge_result_as_zero():
    assert rule_only_credit(score("It is 20 kHz, or maybe 30 kHz.", K30)) == 0.0
    assert rule_only_credit(score("Final answer: 30 kHz", K30)) == 1.0
    assert rule_only_credit(score("1.03 kHz", {"value": 1, "unit": "kHz"})) == 0.5
    parts = [{"value": 1.25, "unit": "kHz", "label": "B"}, {"value": 1, "unit": "ms", "label": "Ts"}]
    r = score("B = 1.25 kHz; Ts is one millisecond", parts)
    assert r["method"] == "judge" and rule_only_credit(r) == 0.5


def test_list_markers_and_labels_are_not_answers():
    four = {"value": 4, "unit": ""}
    assert credit("Answer: 4.", four) == ("numeric", 1.0)
    assert credit("4.", four) == ("numeric", 1.0)
    assert credit("Final answer:\n1. 30 kHz", K30) == ("numeric", 1.0)
    assert credit("## Final Answer\n\n1. 30 kHz", K30) == ("numeric", 1.0)
    assert credit("Step 1: compute. Step 2: 30 kHz", K30) == ("numeric", 1.0)
    assert credit("Final answer:\n1. Decimate by 4\n2. Then low-pass filter", four) == ("numeric", 1.0)
    # a number before a colon inside a sentence is a value
    assert credit("The decimation factor is 4: 48 kHz in, 12 kHz out.", four) == ("numeric", 1.0)
    assert credit("Answer: 4: the decimation factor", four) == ("numeric", 1.0)


# --- labels and parts (finding F3, and value-first answers) -----------------------------------------

TS = {"value": 1, "unit": "ms", "label": "Ts"}


def test_the_final_answer_beats_an_earlier_label():
    assert credit("Ts = 2 ms at first.\nFinal answer: 1 ms", TS) == ("numeric", 1.0)
    assert credit("Ts = 1 ms if no oversampling.\nFinal answer: 2 ms", TS) == ("numeric", 0.0)
    assert credit(r"Ts = 1 ms (first try). Then \boxed{2 ms}", TS) == ("numeric", 0.0)
    assert credit("Final answer: 1 ms\n\nNote: Ts = 2 ms without oversampling.", TS) == ("numeric", 1.0)
    # a label inside the final answer still anchors there
    assert credit("Final answer: B = 2 ms, Ts = 1 ms", TS) == ("numeric", 1.0)
    assert credit(r"\boxed{B = 2\text{ ms},\ T_s = 3\text{ ms},\ Ts = 1\text{ ms}}", TS) == ("numeric", 1.0)


def test_labelled_parts():
    text = "B = 1.25 kHz and Ts = 1 ms. Note that Ts is the inverse of the symbol rate."
    assert credit(text, TS) == ("numeric", 1.0)
    parts = [{"value": 1.25, "unit": "kHz", "label": "B"}, TS]
    assert credit(text, parts) == ("numeric", 1.0)
    half = [{"value": 1.25, "unit": "kHz", "label": "B"}, {"value": 2, "unit": "ms", "label": "Ts"}]
    assert credit(text, half) == ("numeric", 0.5)
    assert credit("B and Ts are 1.25 kHz and 1 ms", parts) == ("numeric", 1.0)
    assert credit("f1=10 kHz,f2=20 kHz", [{"value": 10, "unit": "kHz", "label": "f1"},
                                           {"value": 20, "unit": "kHz", "label": "f2"}]) == ("numeric", 1.0)
    assert credit("Final answer: 1.25 kHz, 1 ms", [{"value": 1.25, "unit": "kHz"}, {"value": 1, "unit": "ms"}]) \
        == ("numeric", 1.0)
    # score() gives the mean over the parts with each part's result; a unit or order-of-magnitude
    # error in any part zeroes the whole answer in judge_prompt (DU) and in rule_only_credit
    power = [{"value": 1.25, "unit": "kHz", "label": "B"}, {"value": 27, "unit": "dBm", "label": "P"}]
    r = score("B = 1.25 kHz, P = 27 dB", power)
    assert (r["method"], r["credit"]) == ("numeric", 0.5)
    assert [p["details"]["reason"] for p in r["details"]["parts"]] == ["within_tolerance", "unit_mismatch"]
    assert rule_only_credit(r) == 0.0
    assert rule_only_credit(score("B = 12.5 kHz and Ts = 1 ms", parts)) == 0.0         # order of magnitude
    r = score("B = 12.5 kHz; Ts is one millisecond", parts)                           # a part undecided
    assert r["method"] == "judge" and rule_only_credit(r) == 0.0
    assert rule_only_credit(score(text, half)) == 0.5                                  # no DU: the mean
    # a part the answer gives no number for goes to the judge (it may be in words)
    assert credit("Final answer: 30 kHz", [K30, {"value": 1, "unit": "ms"}]) == JUDGE
    # unlabelled parts of one unit cannot be told apart
    assert credit("Final answer: 3 MS/s and 0.75 MS/s", [{"value": 3, "unit": "MS/s", "label": "input"},
                                                        {"value": 0.75, "unit": "MS/s", "label": "output"}]) == JUDGE


RATES = [{"value": 50, "unit": "MS/s", "label": ["real sampler", "real"]},
         {"value": 25, "unit": "MS/s", "label": ["complex sampler", "complex"]}]


@pytest.mark.parametrize("answer,expected", [
    ("Real sampler: 50 MS/s. Complex sampler: 25 MS/s.", 1.0),
    ("You need 50 MS/s for a real sampler and 25 MS/s for a complex sampler.", 1.0),
    ("50 MS/s with real sampling; 25 MS/s with complex sampling.", 1.0),
    ("Answer: 50 MS/s (real) and 25 MS/s (complex).", 1.0),
    ("- 50 MS/s for the real sampler\n- 25 MS/s for the complex sampler", 1.0),
    ("| sampler | rate |\n|---|---|\n| real | 50 MS/s |\n| complex | 25 MS/s |", 1.0),
    ("Real: 2B; complex: B.\n\\boxed{50\\text{ MS/s (real)},\\ 25\\text{ MS/s (complex)}}", 1.0),
    ("- 25 MS/s for the real sampler\n- 50 MS/s for the complex sampler", 0.0),       # swapped
    ("You need 25 MS/s for a real sampler and 50 MS/s for a complex sampler.", 0.0),
    ("50 MS/s (real),25 MS/s (complex)", None),                                       # no separator
])
def test_parts_written_label_first_or_value_first(answer, expected):
    r = score(answer, RATES)
    assert r["credit"] == expected, r["details"]


def test_three_parts_value_first():
    parts = [{"value": 12, "unit": "m", "label": ["range resolution", "resolution"]},
             {"value": 80, "unit": "", "label": ["time-bandwidth product", "time-bandwidth", "TBP"]},
             {"value": 120, "unit": "km", "label": ["unambiguous range", "maximum unambiguous range"]}]
    for text in ("Range resolution: 12 m. Time-bandwidth product: 80. Maximum unambiguous range: 120 km.",
                 "This gives a 12 m range resolution, a time-bandwidth product of 80, and a 120 km unambiguous "
                 "range (for a 1 ms PRI).",
                 "So: 12 m range resolution, TBP = 80, and 120 km maximum unambiguous range. The PRI is 1 ms."):
        assert credit(text, parts) == ("numeric", 1.0), text


# --- bare numbers, unknown units, numbers in words ---------------------------------------------------


def test_bare_numbers():
    assert credit("Answer: 30", K30) == ("numeric", 1.0)                       # read in the reference unit
    assert reason("Answer: 30", {**K30, "unit_required": True}) == "unit_missing"
    assert credit("Answer: 30000", K30) == ("numeric", 0.0)                     # read in kHz only (P4)
    assert credit("Answer: 0.125", {"value": 12.5, "unit": "%"}) == ("numeric", 0.0)
    assert credit("a voltage gain of 2", {"value": 6.02, "unit": "dB"}) == ("numeric", 0.0)   # power: 3.01 dB
    assert credit("a voltage gain of 2", {"value": 6.02, "unit": "dB", "ratio_kind": "amplitude"}) == ("numeric", 1.0)
    assert credit("Answer: 10k", {"value": 10, "unit": "kHz"}) == ("numeric", 1.0)
    assert credit("48K", {"value": 48, "unit": "kHz"}) == ("numeric", 1.0)
    assert credit("290K", {"value": 290, "unit": "K"}) == ("numeric", 1.0)
    assert credit("Twice the highest frequency.", K30) == JUDGE


def test_unknown_unit_spellings_go_to_the_judge():
    for text in ("Answer: 2500 kSamp/s", "Answer: 2.5 Msamp/s", "Answer: 1 kc/s"):
        r = score(text, {"value": 2.5, "unit": "MS/s"})
        assert (r["method"], r["details"]["reason"]) == ("judge", "unit_unparsed"), text
    assert credit("Use 4 taps per symbol", {"value": 4, "unit": ""}) == JUDGE
    assert credit("Answer: 2.5 Mega Hertz", {"value": 2.5, "unit": "MHz"}) == ("numeric", 1.0)


def test_numbers_in_words_go_to_the_judge():
    assert reason("Sample at thirty kilohertz, twice the 15 kHz bandwidth.", K30) == "number_in_words"
    assert credit("The decimation factor is four, taking 48 kHz down to 12 kHz.", {"value": 4, "unit": ""}) == JUDGE
    assert credit("Use one sample per symbol.", {"value": 1, "unit": "samples/symbol"}) == JUDGE
    # number words that cannot be the answer do not matter
    assert credit("Each of the two channels runs at 30 kHz", K30) == ("numeric", 1.0)
    assert credit("The rate is 30 kHz, which is twice the bandwidth.", K30) == ("numeric", 1.0)
    assert credit("A half-band filter; the output rate is 30 kHz.", K30) == ("numeric", 1.0)


def test_unclear_digit_grouping_goes_to_the_judge():
    assert credit("2,5 MHz", {"value": 2.5, "unit": "MHz"}) == JUDGE
    assert credit("1 000 000 samples", {"value": 1e6, "unit": "samples"}) == JUDGE
    assert credit("1 000 000 samples", {"value": 1e6, "unit": "samples"}) == ("numeric", 1.0)  # thin spaces
    assert credit("taps [1,2,1]; cutoff 30 kHz", K30) == ("numeric", 1.0)    # not the candidate


# --- equivalent units and other readings (findings F8, F10) ------------------------------------------


def test_cycles_are_a_count():
    assert credit("1000 cycles per second", {"value": 1, "unit": "kHz"}) == ("numeric", 1.0)
    assert credit("Answer: 1000 cycles/s", {"value": 1, "unit": "kHz"}) == ("numeric", 1.0)
    assert credit("Final answer: 0.25 cycles/sample", {"value": 0.25, "unit": ""}) == ("numeric", 1.0)
    assert credit("Final answer: 0.25 cycles", {"value": 90, "unit": "deg"}) == ("numeric", 1.0)
    assert credit("1000 cycles per second", {"value": 6283.19, "unit": "rad/s"}) == ("numeric", 1.0)


def test_the_other_form_is_converted():
    assert reason("Processing gain = 20 dB", {"value": 100, "unit": ""}) == "converted_db"
    assert credit("Processing gain = 20 dB", {"value": 10, "unit": ""}) == ("numeric", 0.0)   # a power ratio
    assert credit("Processing gain = 20 dB", {"value": 10, "unit": "", "ratio_kind": "amplitude"}) == ("numeric", 1.0)
    assert credit("Processing gain = 100", {"value": 20, "unit": "dB"}) == ("numeric", 1.0)   # and the reverse
    assert credit("Gain = 30 dB", {"value": 4, "unit": ""}) == ("numeric", 0.0)      # no reading fits
    r = score("Final answer: 1 %", {"value": -20, "unit": "dB", "tolerance_db": 0.1})  # a ratio for a dB value
    assert (r["method"], r["credit"], r["details"]["reason"]) == ("numeric", 1.0, "converted_ratio")
    assert credit("Final answer: 50 %", {"value": -20, "unit": "dB"}) == ("numeric", 0.0)
    assert credit("Final answer: 1 %", {"value": -20, "unit": "dBm"}) == ("numeric", 0.0)   # not a ratio
    assert reason("π/2 rad", {"value": 1.5708, "unit": ""}) == "converted_angle"
    assert reason("There is a loss of 3 dB.", {"value": -3, "unit": "dB"}) == "loss_word_sign"
    assert credit("The spur is 40 dBc below the carrier.", {"value": -40, "unit": "dBc"}) == ("numeric", 1.0)
    assert credit("Final answer: -3 dB", {"value": -3, "unit": "dB"}) == ("numeric", 1.0)
    assert credit("Final answer: -3 dB", {"value": -6, "unit": "dB", "tolerance_db": 0.1}) == ("numeric", 0.0)
    # without a loss word the sign is the answer's own: a sign error scores 0
    assert credit("Final answer: -4.5 dB", {"value": 4.5, "unit": "dB"}) == ("numeric", 0.0)
    assert credit("The gain is 4.5 dB.", {"value": -4.5, "unit": "dB"}) == ("numeric", 0.0)
    assert credit("Gain = 11 dB", {"value": 11, "unit": "dBi", "accept_units": ["dB"]}) == ("numeric", 1.0)


@pytest.mark.parametrize("reference", [
    {"value": 30, "unit": "kHz"}, {"value": 2.5, "unit": "MS/s"}, {"value": -114, "unit": "dBm", "abs_tolerance": 0.2},
    {"value": 4.5, "unit": "dB", "tolerance_db": 0.1}, {"value": -40, "unit": "dBc"}, {"value": 7.15, "unit": "dBi"},
    {"value": 0.125, "unit": ""}, {"value": 12.5, "unit": "%"}, {"value": 90, "unit": "deg"},
    {"value": 290, "unit": "K"},
    {"value": 4, "unit": "samples/symbol"}, {"value": 1.2, "unit": "Mbit/s"}, {"value": 3e-38, "unit": ""},
    {"value": 1, "unit": "ms", "label": "Ts"}, {"value": 0.25, "unit": "", "unit_required": True},
])
def test_a_lone_value_with_its_unit_is_always_decided(reference):
    """judge_prompt.grade_value scores each value the judge reports as "<value> <unit>" through
    score() and needs a decided result."""
    ref = parse_reference(reference)
    for factor in (1, -1, 0.5, 0.97, 1.03, 2, 9.5, 1e-3, 1e3):
        text = f"{float(ref.value * factor)!r} {reference['unit']}".strip()     # grade_value's format
        r = score(text, reference)
        assert r["method"] == "numeric", (text, r["details"])


# --- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,value,unit", [
    ("2.5e6 Hz", 2.5e6, "Hz"), ("−3 dB", -3.0, "dB"), ("1/4", 0.25, ""), (r"\frac{1}{4}", 0.25, ""),
    ("π/2 rad", math.pi / 2, "rad"), ("3\\pi/4", 3 * math.pi / 4, ""), ("2^10", 1024.0, ""),
    ("1.5*10**6 bit/s", 1.5e6, "bit/s"), ("2.5×10⁶ S/s", 2.5e6, "S/s"), ("1,024 bytes", 1024.0, "bytes"),
    ("12.5%", 12.5, "%"), ("90°", 90.0, "°"), ("10 kHz ± 5 Hz", 10.0, "kHz"), ("5 MHz.", 5.0, "MHz"),
    ("0.4 µs", 0.4, "μs"), ("4x", 4.0, ""), ("16.85 °C", 16.85, "°C"), ("2500 kSamples/s", 2500.0, "kSamples/s"),
])
def test_extract_quantities(text, value, unit):
    qs = extract_quantities(text)
    assert len(qs) == 1, qs
    assert math.isclose(qs[0].value, value) and qs[0].unit == unit


@pytest.mark.parametrize("text", ["log10", "f1", "8PSK", "16QAM", "GNU Radio 3.10.12", "2nd", "x_2"])
def test_numbers_inside_identifiers_are_skipped(text):
    assert extract_quantities(text) == []


@pytest.mark.parametrize("unit,si", [
    ("kHz", 1e3), ("KHz", 1e3), ("mhz", 1e6), ("mHz", 1e-3), ("MS/s", 1e6), ("Msps", 1e6), ("kbps", 1e3),
    ("Mbit/s", 1e6), ("MB/s", 8e6), ("us", 1e-6), ("ns", 1e-9), ("samples per second", 1.0), ("deg", math.pi / 180),
    ("kSamples/s", 1e3), ("Mega Hertz", 1e6), ("cycles/s", 1.0),
])
def test_unit_scales(unit, si):
    assert math.isclose(parse_unit(unit).scale, si)


def test_final_answer_regions():
    assert final_answer("x\nFinal answer: 30 kHz. Then 5 Hz.").regions[0] == "30 kHz"
    assert final_answer("## Final Answer\n\n1. 30 kHz").line_start is True
    assert final_answer("Ts = 2 ms.\nFinal answer: 1 ms", ("Ts",)).rule == "marker"
    assert final_answer("B = 1 kHz, Ts = 1 ms", ("Ts",), ("B",)).regions[0] == "Ts = 1 ms"


def test_unknown_units_and_bad_references_raise():
    with pytest.raises(ValueError):
        parse_unit("furlongs")
    for bad in ({"unit": "Hz"}, {"value": 1, "unit": "zorks"}, {"value": 1, "tolerance": -1},
                {"set": []}, {"exact": ""}, 42, [{"value": 1}, {"set": ["a"]}], {"numeric": 3}):
        with pytest.raises(ValueError):
            score("1", bad)


# --- rubrics without a rule target (finding F9) ------------------------------------------------------


def test_open_ended_rubric_goes_to_the_judge():
    for rubric in ({"numeric": None, "points": ["p1"], "numeric_tolerance": None},
                   {"required_points": [{"id": "P1", "text": "..."}]}, {"numeric": [], "points": ["p1"]}):
        r = score("Throttle paces a simulation without hardware.", rubric)
        assert (r["method"], r["credit"], r["details"]["reason"]) == ("judge", None, "no_rule_reference")
        assert r["details"]["final_answer"]
    assert credit("x", {"numeric": None, "points": ["p1"]}, truncated=True) == ("rules", 0.0)   # F9


def test_rubric_numeric_field():
    assert credit("2.5 MHz", {"numeric": {"value": 2.5, "unit": "MHz", "tolerance": 0.01}}) == ("numeric", 1.0)
    rubric = {"numeric": [{"value": 1.25, "unit": "kHz", "label": "B"}, TS], "points": ["p1"]}
    assert credit("B = 1.25 kHz and Ts = 1 ms", rubric) == ("numeric", 1.0)


# --- symbolic (findings F4, F5) -----------------------------------------------------------------------


@pytest.mark.parametrize("answer,expected", [
    ("Answer: float, complex", ("symbolic", 1.0)),
    ("- complex\n- float", ("symbolic", 1.0)),
    ("Answer: complex and float types", ("symbolic", 1.0)),
    ("**Answer:** `complex`, `float`", ("symbolic", 1.0)),
    ("Answer: both complex and float", ("symbolic", 1.0)),
    ("Answer: complex, float, int", ("symbolic", 0.0)),
    ("Answer: complex", ("symbolic", 0.0)),
    ("Only float.", ("symbolic", 0.0)),
    ("It accepts complex and float inputs.", JUDGE),                   # prose: the judge
    ("It takes complex and float, not int.", JUDGE),
    ("Only complex is supported; float was dropped.", JUDGE),
    ("complex rather than float", JUDGE),
    ("Final answer: either complex or float", JUDGE),
    ("Answer: complex or float", JUDGE),                                # "or" is a hedge, not a list
    ("Answer: complex, float, vectors", JUDGE),                         # an unknown piece
    ("It takes vectors.", JUDGE),
])
def test_option_sets(answer, expected):
    assert credit(answer, {"set": ["complex", "float"], "universe": TYPES}) == expected


@pytest.mark.parametrize("reference", [{"set": ["complex", "float"]}, ["complex", "float"], {"complex", "float"}])
def test_sets_without_a_universe_cannot_rule_out_extras(reference):
    assert credit("Answer: complex, float, int", reference) == JUDGE
    assert credit("It accepts complex, float and int.", reference) == JUDGE
    assert credit("Answer: complex, float", reference) == ("symbolic", 1.0)
    assert credit("Answer: complex", reference) == ("symbolic", 0.0)
    assert credit("Answer: complex, float, int", {"set": ["complex", "float"], "universe": TYPES}) == ("symbolic", 0.0)


def test_option_aliases():
    ref = {"set": ["complex"], "universe": TYPES, "aliases": {"complex": ["gr_complex", "fc32"]}}
    assert credit("Answer: fc32", ref) == ("symbolic", 1.0)


ABCD = ["A", "B", "C", "D"]


@pytest.mark.parametrize("answer,expected", [
    ("Final answer: (A) and (C)", ("symbolic", 1.0)),
    ("Final answer: A, C", ("symbolic", 1.0)),
    ("Final answer: options A and C", ("symbolic", 1.0)),
    ("Final answer: (A)", ("symbolic", 0.0)),
    ("Final answer: A, B, C", ("symbolic", 0.0)),
    ("Final answer: (A) and (C), since (B) doubles the rate", ("symbolic", 1.0)),
    ("Final answer: (A) and (C) -- (D) would alias", ("symbolic", 1.0)),
    ("The correct options are A and C.", JUDGE),                    # letters in prose: the judge
    ("Final answer: A is wrong, C is right", JUDGE),
    ("Final answer: I think A and C", JUDGE),
])
def test_multiple_choice_letters(answer, expected):
    assert credit(answer, {"set": ["A", "C"], "universe": ABCD}) == expected


@pytest.mark.parametrize("answer,expected", [
    ("blocks_throttle", ("symbolic", 1.0)),
    ("Insert a `blocks_throttle` after the source.", ("symbolic", 1.0)),
    ("Use blocks_throttle.", ("symbolic", 1.0)),
    ("The block is `blocks_throttle`; it limits the rate.", ("symbolic", 1.0)),
    ("Use `blocks.throttle(gr.sizeof_gr_complex, samp_rate)`.", ("symbolic", 1.0)),
    ("The block is blocks_throttle_0.", ("symbolic", 1.0)),
    ("Insert a `blocks_head` after the source.", ("symbolic", 0.0)),
    ("Use `blocks_throttle2`.", ("symbolic", 0.0)),
    ("Use `analog_sig_source_x` and `blocks_throttle`.", JUDGE),
    ("Use blocks_throttle instead of blocks_head.", JUDGE),
    ("Do not use blocks_throttle.", JUDGE),
    ("Insert a rate limiter after the source.", JUDGE),
    ("You might think of blocks_throttle, but the right block is the Head block.", JUDGE),
    ("blocks_throttle would be wrong here; Head is what you want.", JUDGE),
    ("Don't use `blocks_head`; use the Throttle block.", JUDGE),
    ("Use `analog.sig_source_c`.", JUDGE),                            # a call that is not the target
])
def test_exact_block_id(answer, expected):
    assert credit(answer, "blocks_throttle") == expected


def test_exact_plain_words_and_letters_are_strict():
    assert credit("Answer: decim", "decim") == ("symbolic", 1.0)
    assert credit("The parameter is `decim`.", "decim") == ("symbolic", 1.0)
    assert credit("The output is complex or float.", "complex") == JUDGE
    assert credit("It is B, because a filter is needed.", "A") == JUDGE
    assert credit("Final answer: B", {"exact": "B", "universe": ["A", "B", "C"]}) == ("symbolic", 1.0)
    assert credit("Final answer: C", {"exact": "B", "universe": ["A", "B", "C"]}) == ("symbolic", 0.0)
    assert credit("The output is complex.", {"exact": "complex", "universe": TYPES}) == JUDGE
    assert credit("Answer: Throttle", {"exact": "blocks_throttle", "aliases": ["Throttle"]}) == ("symbolic", 1.0)


# --- methodology review, round 3 (findings methodology-F1 .. F8) -------------------------------------

K20 = {"value": 20, "unit": "kHz"}


@pytest.mark.parametrize("answer,values,retracted", [
    ("Answer: 30 kHz. Actually, no: it is 20 kHz.", (20, 30), False),
    ("Final answer: 30 kHz. Wait, that's wrong, the correct rate is 20 kHz.", (20, 30), True),
    ("Answer: 30 kHz? No, 20 kHz.", (20, 30), False),
    ("Answer: 30 kHz\n\nWait, I made an error. Corrected: 24 kHz.", (24, 30), True),
    ("Final answer: 24 kHz.\nHmm, actually it should be 30 kHz.", (24, 30), False),
    ("Answer: 30 kHz. Hmm, recheck: 2 x 12 kHz = 24 kHz.", (24, 30), False),     # 12 kHz is an operand
    (r"\boxed{30\text{ kHz}}. Wait, actually 20 kHz.", (20, 30), False),
    ("Final answer: 30 kHz\n\nNote: with a real input you need 60 kHz instead.", (30, 60), False),
    ("Answer: 30 kHz. That's wrong.", (30,), True),                                # a retraction alone
    ("fs = 20 kHz.\nfs = 30 kHz.", (20, 30), False),                               # restated, not derived
])
def test_a_revision_after_the_final_answer_goes_to_the_judge(answer, values, retracted):
    """F1: the rules no longer decide from the first clause after the marker when later text revises
    it, whichever value is right. When no value involved earns credit, the rules still decide 0
    (both readings agree), unless the answer retracts itself."""
    for value in values:
        r = score(answer, {"value": value, "unit": "kHz"})
        assert (r["method"], r["credit"]) == JUDGE, (answer, value, r["details"])
    other = score(answer, {"value": 12, "unit": "kHz"})
    assert (other["method"], other["credit"]) == (JUDGE if retracted else ("numeric", 0.0)), other["details"]


@pytest.mark.parametrize("answer", [
    "Final answer: 30 kHz\n\nNo aliasing occurs at this rate.",
    "Final answer: 30 kHz. This is twice the 15 kHz bandwidth.",
    "Final answer: 30 kHz\n\nA common mistake is to use 15 kHz.",
    "Final answer: 30 kHz\n\nThe anti-alias cutoff should be 12 kHz.",
    "Final answer: 30 kHz\n\nYou may need to wait 10 ms for the PLL to lock.",
    "Final answer: 30 kHz. Or equivalently 30000 Hz.",                    # the same value: no revision
    "Final answer: 30 kHz\n\nRecheck: 2 × 15 kHz = 30 kHz.",              # a check that agrees
    "Final answer: 30 kHz\n\nA 37.5 kHz rate leaves a guard band.",
])
def test_commentary_after_the_final_answer_keeps_the_decision(answer):
    assert credit(answer, K30) == ("numeric", 1.0)
    assert credit(answer, K20) == ("numeric", 0.0)


def test_revision_words_need_a_value_of_the_reference_dimension_unless_strong():
    four = {"value": 12, "unit": "kHz"}
    assert credit("Final answer: 12 kHz\n\nThe decimation factor is 4, not 2.", four) == ("numeric", 1.0)
    assert credit("Final answer: 1.2 Mbit/s\n\nThose are corrected bits; the raw rate is 2.4 Mbit/s.",
                  {"value": 1.2, "unit": "Mbit/s"}) == ("numeric", 1.0)
    assert credit("Answer: 30 kHz? No, 20.", K30) == JUDGE                 # a strong revision: bare values count
    # labelled parts: a revision after the part's own segments
    assert credit("Ts = 1 ms. Hmm, actually 2 ms.", TS) == JUDGE
    r = score("Final answer: Ts = 1 ms, B = 2 kHz. Wait, B = 3 kHz.", [TS, {"value": 3, "unit": "kHz", "label": "B"}])
    assert r["method"] == "judge" and [p["method"] for p in r["details"]["parts"]] == ["numeric", "judge"]


@pytest.mark.parametrize("answer,reference", [
    ("Answer: complex (or float)", "complex"),
    ("Answer: complex (or float)", {"exact": "complex", "universe": TYPES}),
    ("Answer: complex (float also works)", "complex"),
    ("Answer: complex (float also works)", {"exact": "complex", "universe": TYPES}),
    ("Answer: complex? No, float.", "complex"),
    ("Answer: complex. Correction: float.", "complex"),
    ("Answer: complex, float (and possibly int)", {"set": ["complex", "float"]}),
    ("Answer: complex, float (and possibly int)", {"set": ["complex", "float"], "universe": TYPES}),
    ("Answer: complex, float. Actually only complex.", {"set": ["complex", "float"]}),
    ("Answer: `blocks_throttle` (or `blocks_head`)", "blocks_throttle"),
    ("Answer: `blocks_throttle`. No wait, `blocks_head`.", "blocks_throttle"),
    ("Use `blocks_head`.\n\nActually, `blocks_throttle`.", "blocks_throttle"),
    ("Final answer: B. Hmm, actually C.", {"set": ["B"], "universe": ABCD}),
    ("Final answer: (B)\nOn second thought, (C).", {"set": ["B"], "universe": ABCD}),
    ("Final answer: B\n\nOption A would also work.", {"set": ["B"], "universe": ABCD}),
    ("Answer: complex. The float type is also accepted.", {"exact": "complex", "universe": TYPES}),
])
def test_hedged_or_revised_option_answers_go_to_the_judge(answer, reference):
    """F2: a parenthetical or a later sentence that hedges, extends or revises an option answer."""
    r = score(answer, reference)
    assert (r["method"], r["credit"]) == JUDGE, r["details"]
    assert r["details"]["rule_method"] == "symbolic"


@pytest.mark.parametrize("answer,reference,expected", [
    ("Answer: complex (the default)", "complex", 1.0),
    ("Answer: complex\n\nIt is the default output type.", {"exact": "complex", "universe": TYPES}, 1.0),
    ("Final answer: B\n\nIt doubles the decimation.", {"set": ["B"], "universe": ABCD}, 1.0),
    ("Final answer: B\n\nThis is a filter tap issue, not a rate issue.", {"set": ["B"], "universe": ABCD}, 1.0),
    ("Final answer: complex, float\n\nOnly these two are listed.", {"set": ["complex", "float"], "universe": TYPES},
     1.0),
    ("Final answer: (A) and (C), since (B) doubles the rate", {"set": ["A", "C"], "universe": ABCD}, 1.0),
    ("Final answer: C\n\nIt is a filter issue.", {"set": ["B"], "universe": ABCD}, 0.0),
])
def test_option_commentary_without_another_option_keeps_the_decision(answer, reference, expected):
    assert credit(answer, reference) == ("symbolic", expected)


@pytest.mark.parametrize("answer,reference", [
    ("To get the final answer, multiply 15 kHz by 2: 30 kHz.", K30),
    ("The final answer depends on the 15 kHz bandwidth: fs = 2 x 15 kHz = 30 kHz.", K30),
])
def test_final_answer_in_prose_is_not_a_marker(answer, reference):
    """F3: 'final answer' is a marker only before ':', '=' or 'is', alone on its line or bold."""
    assert credit(answer, reference) == ("numeric", 1.0)
    assert final_answer(answer).rule == "last"


def test_final_answer_marker_forms():
    for text in ("Final answer: 30 kHz", "**Final answer:** 30 kHz", "**Final Answer**: 30 kHz",
                 "The final answer is 30 kHz.", "Final answer = 30 kHz", "## Final Answer\n30 kHz",
                 "**Final Answer**\n\nThe output rate is 30 kHz.", "Final Answer\n\n30 kHz", "> **Final answer:** 30 kHz",
                 "Some work: 15 kHz.\n\n**Final answer** 30 kHz"):
        assert final_answer(text).rule == "marker", text
        assert credit(text, K30) == ("numeric", 1.0), text
    # prose: the context value no longer becomes the decided candidate
    text = "Before giving the final answer, note that the input is 48 kHz. Decimating by 4, the output rate is 12 kHz."
    assert final_answer(text).rule == "last"
    assert credit(text, {"value": 12, "unit": "kHz"}) == JUDGE


PARTS_TB = [{"value": 1, "unit": "ms", "label": "Ts"}, {"value": 1, "unit": "kHz", "label": "B"}]


@pytest.mark.parametrize("answer,part_reasons", [
    ("Ts = 1 ms and B = 1 dB", ["within_tolerance", "unit_mismatch"]),
    ("Ts = 1 ms and B = 10 kHz", ["within_tolerance", "order_of_magnitude"]),
    ("Ts = 1 ms and B = 1 ms", ["within_tolerance", "unit_mismatch"]),      # B given another part's unit
])
def test_a_part_error_zeroes_the_item_credit(answer, part_reasons):
    """F4: the item credit (details, rule_only_credit, --batch rows) is 0; `credit` stays the mean over
    the parts, the per-part input judge_prompt combines (its tests pin it)."""
    r = score(answer, PARTS_TB)
    assert [p["details"]["reason"] for p in r["details"]["parts"]] == part_reasons
    assert (r["method"], r["credit"]) == ("numeric", 0.5)
    assert r["details"]["reason"] == "part_zero" and r["details"]["part_zero"] == part_reasons[1]
    assert r["details"]["item_credit"] == 0.0 and rule_only_credit(r) == 0.0


def test_item_credit_without_a_part_error():
    r = score("Ts = 1 ms and B = 1 kHz", PARTS_TB)
    assert (r["method"], r["credit"], r["details"]["reason"], r["details"]["item_credit"]) == ("numeric", 1.0, "parts", 1.0)
    assert "part_zero" not in r["details"]
    r = score("B and Ts are 1 kHz and 1 ms", PARTS_TB)                       # respectively: not a unit error
    assert (r["method"], r["credit"]) == ("numeric", 1.0)
    r = score("B = 1 kHz; Ts is one millisecond", PARTS_TB)                  # a part undecided, none zero
    assert r["method"] == "judge" and r["details"]["item_credit"] is None and rule_only_credit(r) == 0.5
    r = score("B = 1 dB; Ts is one millisecond", PARTS_TB)                   # undecided, but one part zeroes it
    assert r["method"] == "judge" and r["details"]["item_credit"] == 0.0 and r["details"]["part_zero"] == "unit_mismatch"


def test_rule_only_summary_and_the_answer_format_line(tmp_path, capsys):
    """F5: the deferral rate, the lower-bound mean and the decided-only mean are reported together, and
    answers that end with ANSWER_FORMAT_INSTRUCTION's line are decided whatever comes before."""
    assert "Final answer: <value> <unit>" in rules.ANSWER_FORMAT_INSTRUCTION
    formatted = [
        ("For a 15 kHz bandwidth signal, the Nyquist rate is 30 kHz.\nFinal answer: 30 kHz", K30),
        ("Input 48 kHz, decimation 4, so 12 kHz; a real input would need 24 kHz.\n\nFinal answer: 12 kHz",
         {"value": 12, "unit": "kHz"}),
        ("Ts is 1 ms because B = 1 kHz.\nFinal answer: Ts = 1 ms, B = 1 kHz", PARTS_TB),
        ("The port takes complex or float.\nFinal answer: complex, float", {"set": ["complex", "float"],
                                                                            "universe": TYPES}),
    ]
    for text, ref in formatted:
        r = score(text, ref)
        assert r["method"] != "judge" and r["credit"] == 1.0, (text, r["details"])
    unformatted = [score(t.split("Final answer")[0], ref) for t, ref in formatted[:2]]
    assert all(r["method"] == "judge" for r in unformatted)
    results = [score(t, ref) for t, ref in formatted] + unformatted + [score("Final answer: 20 kHz", K30)]
    assert rules.rule_only_summary(results) == {"n": 7, "decided": 5, "deferred": 2,
                                                "deferral_rate": round(2 / 7, 6),
                                                "rule_only_mean": round(4 / 7, 6), "decided_mean": 0.8}
    assert rules.rule_only_summary([])["deferral_rate"] is None


def test_decibel_scale_and_bare_numbers_are_documented():
    """F7: the reference's unit sets the scale (P1); a bare number is read in the reference unit only
    (P4)."""
    assert credit("1.02 mW", {"value": 0, "unit": "dBm"}) == ("numeric", 1.0)          # 0.086 dB
    assert credit("0.086 dBm", {"value": 1, "unit": "mW"}) == ("numeric", 0.5)         # 2 %
    assert credit("-114.05 dBm", {"value": -114, "unit": "dBm"}) == ("numeric", 1.0)   # rounded to 0.1 dB
    doc = " ".join(rules.__doc__.split())
    assert "dB items are graded somewhat more leniently than linear ones" in doc
    assert credit("Answer: 30", K30) == ("numeric", 1.0)
    assert credit("Answer: 30", {"value": 30000, "unit": "Hz"}) == ("numeric", 0.0)    # 30 Hz
    assert credit("Answer: 2500", {"value": 2.5, "unit": "MS/s"}) == ("numeric", 0.0)  # 2500 MS/s
    assert credit("Answer: 25", {"value": 30000, "unit": "Hz"}) == ("numeric", 0.0)    # no reading fits
    assert credit("Answer: 30", {"value": 30000, "unit": "Hz", "unit_required": True}) == ("numeric", 0.0)
    assert credit("Answer: 4", {"value": 4, "unit": ""}) == ("numeric", 1.0)           # no prefix for a count


@pytest.mark.parametrize("answer", [
    "Answer: less than 30 kHz", "Answer: more than 30 kHz", "Answer: at least 30 kHz", "Answer: at most 30 kHz",
    "Answer: up to 30 kHz", "Answer: fs > 30 kHz", "Answer: fs ≥ 30 kHz", "Answer: fs >= 30 kHz",
    "Answer: 30 kHz or more", "Answer: 30 kHz or higher", "Answer: no more than 30 kHz", "Answer: below 30 kHz",
])
def test_a_comparator_next_to_the_answer_goes_to_the_judge(answer):
    """F8: a bound is not a point value."""
    r = score(answer, K30)
    assert (r["method"], r["details"]["reason"]) == ("judge", "comparator_near_answer")


def test_comparators_elsewhere_and_arrows_do_not_count():
    for text in ("fs must be at least 2B = 2 × 15 kHz = 30 kHz", "fs = 2B -> 30 kHz", "fs = 2B => 30 kHz",
                 "> **Final answer:** 30 kHz", "> 30 kHz", "fs = 2 × 15 kHz = 30 kHz minimum"):
        assert credit(text, K30) == ("numeric", 1.0), text


def test_sign_words_on_linear_quantities_and_named_db_references():
    """F8: a direction word carries the sign of a linear value as a loss word does of a dB value; R9:
    a plain dB value is read in the reference's relative decibel unit (dBc, dBi)."""
    assert reason("The image lies 10 kHz below the carrier.", {"value": -10, "unit": "kHz"}) == "loss_word_sign"
    assert credit("The output lags by 90 degrees.", {"value": -90, "unit": "deg"}) == ("numeric", 1.0)
    assert credit("The image lies 10 kHz above the carrier.", {"value": -10, "unit": "kHz"}) == ("numeric", 0.0)
    assert credit("Final answer: -10 kHz", {"value": 10, "unit": "kHz"}) == ("numeric", 0.0)
    r = score("The spur is 40 dB below the carrier.", {"value": -40, "unit": "dBc"})
    assert (r["method"], r["credit"], r["details"]["reason"]) == ("numeric", 1.0, "loss_word_sign")
    assert credit("The antenna gain is 6 dB over isotropic.", {"value": 6, "unit": "dBi"}) == ("numeric", 1.0)
    assert credit("The spur is at -40 dB.", {"value": -40, "unit": "dBc"}) == ("numeric", 1.0)    # read as dBc
    assert credit("The spur is 40 dB above the carrier.", {"value": -40, "unit": "dBc"}) == ("numeric", 0.0)
    assert credit("The spur is 40 dB below the carrier.", {"value": -40, "unit": "dBc", "accept_units": ["dB"]}) == ("numeric", 1.0)


# --- unit forms, relative decibels, voltage conventions (round-2 findings R4, R5, R9, R10) ---------


def test_sec_family_and_kbit_units_are_read():
    assert score("The symbol period is 1 msec.", {"value": 0.001, "unit": "s"})["credit"] == 1.0
    assert score("Answer: 400 nsec", {"value": 0.4, "unit": "us"})["credit"] == 1.0
    assert score("Answer: 20 usec", {"value": 2e-05, "unit": "s"})["credit"] == 1.0
    assert score("Answer: 125 microsec", {"value": 0.000125, "unit": "s"})["credit"] == 1.0
    assert score("Answer: 64 Kbit", {"value": 64000, "unit": "bit"})["credit"] == 1.0


def test_unknown_unit_word_goes_to_the_judge():
    r = score("Final answer: 3.84 Mcps", {"value": 3.84e6, "unit": "Hz"})
    assert r["method"] == "judge" and r["details"]["reason"] == "unknown_unit"


def test_count_words_after_a_number_are_not_units():
    assert score("Final answer: 4 taps", {"value": 4, "unit": ""})["credit"] == 1.0


def test_latex_tilde_between_number_and_unit_is_a_space():
    assert score("Final answer: $30~\\text{kHz}$", {"value": 30000, "unit": "Hz"})["credit"] == 1.0


def test_plain_db_reads_in_the_relative_family_of_the_reference():
    assert score("Final answer: -40 dB", {"value": -40, "unit": "dBc"})["credit"] == 1.0
    r = score("Final answer: -40 dB", {"value": -40, "unit": "dBm"})
    assert r["credit"] == 0.0 and r["details"]["reason"] == "unit_mismatch"


def test_voltage_conventions():
    assert score("Final answer: 2 Vpp", {"value": 1, "unit": "Vpk"})["credit"] == 1.0
    assert score("Final answer: 1 Vp", {"value": 2, "unit": "Vpp"})["credit"] == 1.0
    assert score("Final answer: 500 mVpp", {"value": 0.25, "unit": "Vpk"})["credit"] == 1.0
    assert score("Final answer: 0.7071 Vrms", {"value": 1, "unit": "Vpk", "waveform": "sine"})["credit"] == 1.0
    r = score("Final answer: 0.7071 Vrms", {"value": 1, "unit": "Vpk"})
    assert r["credit"] == 0.0 and r["details"]["reason"] == "voltage_convention"
    assert score("Final answer: 1 V", {"value": 1, "unit": "Vpk"})["credit"] == 1.0   # no convention named
    with pytest.raises(ValueError):
        parse_reference({"value": 1, "unit": "Vpk", "waveform": "square"})
    assert score("Final answer: 1.414 Vpk", {"value": 1, "unit": "Vrms", "waveform": "sine"})["credit"] == 1.0


def test_voltage_conventions_in_other_spellings():
    for ans in ("Final answer: 2 Vp-p", "Final answer: 2 Vpk-pk", "Final answer: 2 V peak-to-peak",
                "Final answer: 2 volts p-p", "Final answer: 2000 mVp-p", "Final answer: 2 V pp",
                "Final answer: 2 V peak to peak", "Final answer: 2 volts peak to peak"):
        assert score(ans, {"value": 1, "unit": "Vpk"})["credit"] == 1.0, ans
    for ans in ("Final answer: 1 V rms", "Final answer: $1\\,V_{rms}$", "Final answer: 1 V RMS"):
        assert score(ans, {"value": 1, "unit": "Vpk"})["details"]["reason"] == "voltage_convention", ans
    assert score("The 5 V pin is used.", {"value": 5, "unit": "V"})["credit"] == 1.0
    assert score("Final answer: 1 Vrms^2", {"value": 1, "unit": "V^2"})["credit"] == 1.0   # an exponent stays


def test_dbv_levels_are_rms():
    assert score("Final answer: 1 Vpp", {"value": 0, "unit": "dBV"})["details"]["reason"] == "voltage_convention"
    assert score("Final answer: 1 Vrms", {"value": 0, "unit": "dBV"})["credit"] == 1.0
    assert score("Final answer: 0 dBV", {"value": 1.414, "unit": "Vpk", "waveform": "sine"})["credit"] == 1.0
    assert score("Final answer: 0 dBV", {"value": 1, "unit": "Vpk"})["details"]["reason"] == "voltage_convention"


def test_a_sign_word_flips_the_converted_voltage():
    r = score("The output drops by 1 Vpp.", {"value": -0.5, "unit": "Vpk"})
    assert (r["method"], r["credit"], r["details"]["reason"]) == ("numeric", 1.0, "loss_word_sign")
    r = score("The output is 1 Vpp, after the drop.", {"value": -0.5, "unit": "Vpk"})   # a sign word, not next to it
    assert (r["credit"], r["details"]["reason"], r["details"]["alt_credit"]) == (0.0, "tie_break_lower", 1.0)


def test_a_voltage_convention_error_zeroes_a_multi_part_answer():
    parts = [{"value": 1, "unit": "Vpk", "label": "amplitude"}, {"value": 1, "unit": "kHz", "label": "frequency"}]
    r = score("Final answer: amplitude = 1 Vrms, frequency = 1 kHz", parts)
    assert r["details"]["item_credit"] == 0.0 and r["details"]["part_zero"] == "voltage_convention"


def test_plain_db_against_dbd_and_dbc_per_hz():
    assert score("Final answer: 2 dB", {"value": 2, "unit": "dBd"})["credit"] == 1.0      # read as dBd
    assert score("Final answer: -90 dB/Hz", {"value": -90, "unit": "dBc/Hz"})["credit"] == 1.0
    assert score("Final answer: -90 dB", {"value": -90, "unit": "dBc/Hz"})["credit"] == 0.0


def test_ordinary_words_after_a_bare_number_are_not_units():
    for ans in ("Final answer: 4 dimensionless", "Final answer: 4 unitless", "Final answer: 4 complex samples",
                "Final answer: 4-tap", "Final answer: 4 filter taps", "Final answer: 4 thus the output",
                "Final answer: 4\nThe input is a stream.", "Final answer: 4 assuming a stream",
                "Final answer: 4-QAM"):
        assert score(ans, {"value": 4, "unit": ""})["credit"] == 1.0, ans


def test_gnu_radio_item_words_after_a_count_are_not_units():
    # the behaviour questions ask "How many short items / vectors of 4 values ..."
    for ans in ("34 vectors", "34 short items", "34 int items", "34 char items", "34 messages", "34 tags",
                "34 PDUs", "34 zeros", "34 ones", "34 chunks"):
        r = score("Final answer: " + ans, {"value": 34, "unit": ""})
        assert (r["method"], r["credit"]) == ("numeric", 1.0), ans


def test_bits_bytes_and_symbols_count_items_against_a_reference_with_no_unit():
    for ans in ("34 bytes", "34 byte items", "34 bits", "34 symbols"):
        r = score("Final answer: " + ans, {"value": 34, "unit": ""})
        assert (r["method"], r["credit"]) == ("numeric", 1.0), ans
    assert score("Final answer: 34 bytes", {"value": 272, "unit": ""})["credit"] == 0.0       # 34 items
    assert score("Final answer: 34 bytes", {"value": 272, "unit": "bits"})["credit"] == 1.0   # a unit: converted
    r = score("Final answer: 34 symbols", {"value": 34, "unit": "samples"})
    assert r["details"]["reason"] == "unit_mismatch"                                           # P3


def test_an_unknown_word_after_a_multiplier_goes_to_the_judge():
    for ans in ("Final answer: 3.84M chips", "Final answer: 3.84 million chips"):
        r = score(ans, {"value": 3.84e6, "unit": "Hz"})
        assert r["method"] == "judge" and r["details"]["reason"] == "unknown_unit", ans


def test_latex_tilde_after_math_or_bold():
    for ans in ("Final answer: $48$~kHz", "Final answer: **48**~kHz"):
        assert score(ans, {"value": 48000, "unit": "Hz"})["credit"] == 1.0, ans


# --- decision 7: one automatic reading for every answer (pre-registration §8) -----------------------

DB30 = {"value": 30, "unit": "dB", "abs_tolerance": 0.1}


def test_bare_number_against_a_db_reference_is_a_ratio():
    r = score("Final answer: 1000", DB30)
    assert r["credit"] == 1.0 and r["details"]["reason"] == "converted_ratio"
    assert score("Final answer: 30", DB30)["credit"] == 0.0          # a ratio of 30 is 14.8 dB
    assert score("Final answer: 10", {"value": 20, "unit": "dB", "ratio_kind": "amplitude"})["credit"] == 1.0
    assert score("Final answer: 2", {"value": 6.02, "unit": "dBV"})["credit"] == 1.0   # ratio to 1 V
    assert score("Final answer: 1000", {"value": 30, "unit": "dBm"})["credit"] == 1.0  # ratio to 1 mW
    with pytest.raises(ValueError):
        parse_reference({"value": 30, "unit": "dBm", "ratio_kind": "amplitude"})
    with pytest.raises(ValueError):
        parse_reference({"value": 3, "unit": "dB", "quantity": "Gain"})   # only "gain"


def test_ratio_words_are_units_of_a_plain_ratio():
    for ans in ("Final answer: 1000 times", "Final answer: 1000x", "Final answer: 1000 x"):
        assert score(ans, DB30)["credit"] == 1.0, ans
    assert score("Final answer: 1000 times", {**DB30, "unit_required": True})["credit"] == 1.0
    assert score("Final answer: 1000", {**DB30, "unit_required": True})["details"]["reason"] == "unit_missing"
    assert score("Final answer: 4 times", {"value": 4, "unit": ""})["credit"] == 1.0
    assert score("Final answer: 4x", {"value": 4, "unit": "samples/symbol"})["credit"] == 1.0   # read bare


def test_bare_number_elsewhere_is_read_in_the_reference_unit_only():
    assert score("Final answer: 30", {"value": 30, "unit": "kHz"})["credit"] == 1.0
    r = score("Final answer: 30000", {"value": 30, "unit": "kHz"})
    assert r["method"] == "numeric" and r["credit"] == 0.0           # no SI-base second reading
    assert score("Final answer: 30", {"value": 30000, "unit": "Hz"})["credit"] == 0.0


def test_db_answer_against_a_ratio_reference_is_converted():
    r = score("The power ratio is 20 dB.", {"value": 100, "unit": ""})
    assert r["credit"] == 1.0 and r["details"]["reason"] == "converted_db"
    assert score("The voltage gain is 20 dB.", {"value": 10, "unit": "", "ratio_kind": "amplitude"})["credit"] == 1.0
    assert score("The voltage gain is 20 dB.", {"value": 10, "unit": ""})["credit"] == 0.0   # power: 100
    r = score("Final answer: 90 degrees", {"value": 1.5708, "unit": ""})
    assert r["credit"] == 1.0 and r["details"]["reason"] == "converted_angle"
    assert score("Final answer: 1 %", {"value": -20, "unit": "dB", "tolerance_db": 0.1})["credit"] == 1.0
    assert score("Final answer: 1 %", {"value": -20, "unit": "dBm"})["credit"] == 0.0   # not a ratio


def test_form_required_zeroes_the_other_form_only_when_it_would_earn_credit():
    r = score("Final answer: 20 dB", {"value": 100, "unit": "", "form_required": True})
    assert r["credit"] == 0.0 and r["details"]["reason"] == "wrong_form"
    assert score("Final answer: 1000", {**DB30, "form_required": True})["details"]["reason"] == "wrong_form"
    r = score("Final answer: 25 dB", {"value": 100, "unit": "", "form_required": True})
    assert r["credit"] == 0.0 and r["details"]["reason"] != "wrong_form"   # wrong in either form


def test_loss_words_set_the_sign_for_a_gain():
    r = score("Final answer: a loss of 3 dB", {"value": -3, "unit": "dB"})
    assert r["credit"] == 1.0 and r["details"]["reason"] == "loss_word_sign"
    assert score("Final answer: a loss of 3 dB", {"value": 3, "unit": "dB"})["credit"] == 1.0   # a loss quantity
    assert score("Final answer: a loss of 3 dB", {"value": 3, "unit": "dB", "quantity": "gain"})["credit"] == 0.0
    assert score("Final answer: 3 dB", {"value": -3, "unit": "dB"})["credit"] == 0.0
    assert score("Final answer: -3 dB (a loss of 3 dB)", {"value": -3, "unit": "dB"})["credit"] == 1.0
    for ans, ref in (("Final answer: 10 kHz below the carrier", {"value": -10, "unit": "kHz"}),
                     ("The output is 3 dB down.", {"value": -3, "unit": "dB"}),
                     ("The output lags by 90 degrees.", {"value": -90, "unit": "deg"}),
                     ("The phase is lagging by 90 degrees.", {"value": -90, "unit": "deg"}),
                     ("The spur is 40 dBc below the carrier.", {"value": -40, "unit": "dBc"}),
                     ("The level is reduced by 3 dB.", {"value": -3, "unit": "dB"})):
        assert score(ans, ref)["credit"] == 1.0, ans
    assert score("Final answer: a loss of -3 dB", {"value": -3, "unit": "dB"})["details"]["reason"] == "sign_unclear"


def test_unknown_unit_word_is_never_read_as_bare():
    assert score("The symbol period is 1 msec.", {"value": 0.001, "unit": "s"})["credit"] == 1.0
    assert score("Final answer: 3.84 Mcps", {"value": 3.84e6, "unit": "Hz"})["method"] == "judge"


def test_tie_break_takes_the_lower_credit():
    r = rules.tie_break({"method": "judge", "credit": None,
                         "details": {"reason": "sign_convention", "literal_credit": 0.0, "alt_credit": 1.0}})
    assert r["method"] == "numeric" and r["credit"] == 0.0 and r["details"]["reason"] == "tie_break_lower"
    part = {"method": "judge", "credit": None, "details": {"reason": "sign_convention",
                                                          "literal_credit": 0.0, "alt_credit": 1.0}}
    done = {"method": "numeric", "credit": 1.0, "details": {"reason": "within_tolerance"}}
    r = rules.tie_break({"method": "judge", "credit": None,
                         "details": {"reason": "part_undecided", "parts": [part, done], "item_credit": None}})
    assert r["method"] == "numeric" and r["credit"] == 0.5 and r["details"]["item_credit"] == 0.5
    assert r["details"]["parts"][0]["details"]["reason"] == "tie_break_lower"
    assert rules.reading_counts([r]) == {"tie_break": 1, "wrong_form": 0}


def test_a_value_in_the_reference_form_is_the_candidate_before_a_converted_one():
    db13 = {"value": 13, "unit": "dB"}
    for ans in ("The gain is 13 dB, so N = 16.", "Final answer: 13 dB for 16-QAM", "Final answer: 13 dB at 25 % load"):
        assert score(ans, db13)["credit"] == 1.0, ans
    assert score("The processing gain is 20 dB, a factor of 100.", {"value": 100, "unit": ""})["credit"] == 1.0


def test_a_ratio_word_before_a_word_or_a_full_stop():
    assert score("Final answer: 1000 times.", {**DB30, "unit_required": True})["credit"] == 1.0
    assert score("Final answer: 1000 times larger", {**DB30, "unit_required": True})["credit"] == 1.0
    r = score("The voltage ratio is 10, so the power gain is 20 dB.", {"value": 100, "unit": ""})
    assert r["method"] == "judge"                  # the native 10 is a step towards the converted 20 dB
    r = score("Final answer: 24 kHz. Recheck: fs = 2 × B = 2 × 12 kHz = 24 kHz.", {"value": 24, "unit": "kHz"})
    assert r["credit"] == 1.0


def test_a_loss_word_leaves_a_ratio_against_a_db_reference_alone():
    assert score("Final answer: the power is reduced by 50 %", {"value": -3, "unit": "dB"})["credit"] == 1.0
    r = score("Final answer: a loss of 3 dB", {"value": 3, "unit": "dB", "quantity": "gain"})
    assert r["credit"] == 0.0 and r["details"]["reason"] != "tie_break_lower"
    assert rules.reading_counts([r])["tie_break"] == 0
    assert score("Final answer: a loss\n3 dB", {"value": -3, "unit": "dB"})["credit"] == 0.0   # not across a line
    r = score("Final answer: a loss of 3 dB. Actually the cable's loss is 3 dB.", {"value": -3, "unit": "dB"})
    assert r["credit"] == 1.0


def test_the_tie_break_keeps_the_error_of_the_lower_reading():
    r = score("The spur is 40 dB, well below the carrier.", {"value": -40, "unit": "dBc"})
    assert r["details"]["reason"] == "tie_break_lower" and r["details"]["lower_reason"] == "order_of_magnitude"
    part = {"method": "judge", "credit": None, "details": {"reason": "db_sign_convention", "literal_credit": 0.0,
                                                          "alt_credit": 1.0, "literal_reason": "order_of_magnitude"}}
    done = {"method": "numeric", "credit": 1.0, "details": {"reason": "within_tolerance"}}
    r = rules.tie_break({"method": "judge", "credit": None,
                         "details": {"reason": "part_undecided", "parts": [part, done], "item_credit": None}})
    assert r["details"]["item_credit"] == 0.0 and r["details"]["part_zero"] == "order_of_magnitude"
    assert rule_only_credit(r) == 0.0


READING_PROBES = [  # the inputs of the decision-7 tests above, as (answer, reference) cases
    {"answer": a, "reference": ref} for a, ref in (
        ("Final answer: 1000", DB30), ("Final answer: 30000", {"value": 30, "unit": "kHz"}),
        ("The power ratio is 20 dB.", {"value": 100, "unit": ""}),
        ("Final answer: 90 degrees", {"value": 1.5708, "unit": ""}),
        ("Final answer: 3 dB, the loss of the cable", {"value": -3, "unit": "dB"}),
        ("Final answer: Ts = 30, B = 1 kHz", [{"value": 30000, "unit": "Hz", "label": "Ts"},
                                             {"value": 1, "unit": "kHz", "label": "B"}]))]


def test_no_result_leaves_two_readings_open():
    for case in list(rules.SELFTEST) + READING_PROBES:
        d = score(case["answer"], case["reference"])
        for p in d["details"].get("parts") or [d]:
            assert not (p["method"] == "judge" and "alt_credit" in p["details"]), case["answer"]


# --- round-2 findings R3, R7, R8, F9 and pins of those already handled -------------------------------

DECIMATION = [{"value": 3, "unit": "MS/s", "label": ["input rate", "input"]},
         {"value": 0.75, "unit": "MS/s", "label": ["output rate", "output"]}]


def test_same_dimension_parts_without_matched_labels_go_to_the_judge():
    for ans in ("Decimating the 3 MS/s input by 4 gives an output rate of 0.75 MS/s.",
                "Interpolating the 0.75 MS/s input by 4 gives an output rate of 3 MS/s."):
        r = score(ans, DECIMATION)
        assert r["method"] == "judge", ans
        assert "part_unassigned" in [p["details"]["reason"] for p in r["details"]["parts"]], ans
    assert score("Final answer: input rate = 3 MS/s, output rate = 0.75 MS/s", DECIMATION)["credit"] == 1.0


def test_sentence_initial_article_is_not_option_a():
    r = score("Final answer: A low-pass filter is needed, so option C.",
              {"set": ["C"], "universe": ["A", "B", "C", "D"]})
    assert r["method"] == "judge" or r["credit"] == 1.0
    assert score("Final answer: A", {"set": ["A"], "universe": ["A", "B", "C", "D"]})["credit"] == 1.0
    assert score("Final answer: A, C", {"set": ["A", "C"], "universe": ["A", "B", "C", "D"]})["credit"] == 1.0
    ac, abcd = {"set": ["A", "C"], "universe": list("ABCD")}, list("ABCD")
    for ans in ("Final answer: A, C are correct.", "Final answer: A and C are needed"):
        assert score(ans, ac)["credit"] == 1.0, ans
        assert score(ans, {"set": ["A"], "universe": abcd})["credit"] == 0.0, ans     # C is an extra option
    assert score("Final answer: B & D should both be selected", {"set": ["B", "D"], "universe": abcd})["credit"] == 1.0
    for ans in ("**Final answer: B** — the low-pass filter before the decimator.", "Final answer: B – low-pass filter"):
        assert score(ans, {"set": ["B"], "universe": abcd})["credit"] == 1.0, ans
    r = score("Final answer: A **low-pass filter** before decimation (option C).", {"set": ["C"], "universe": abcd})
    assert r["method"] == "judge" or r["credit"] == 1.0


def test_unevaluated_expression_goes_to_the_judge():
    for ans in ("Final answer: 2 × 15 kHz", "Final answer: 2 x 15 kHz"):
        r = score(ans, {"value": 30, "unit": "kHz"})
        assert r["method"] == "judge" and r["details"]["reason"] == "expression_answer", ans
    assert score("Final answer: 2 × 15 kHz = 30 kHz", {"value": 30, "unit": "kHz"})["credit"] == 1.0
    for ans, ref in (("Final answer: (2)(15 kHz)", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 2(15 kHz)", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 48 kHz / 4", {"value": 12, "unit": "kHz"})):
        assert score(ans, ref)["details"]["reason"] == "expression_answer", ans
    assert score("Final answer: 10 kHz / 20 kHz / 30 kHz", {"value": 90, "unit": "kHz"})["method"] == "numeric"
    for ans in ("Final answer: fs = 2 × 15 kHz", "Final answer: with x = 5, fs = 2 × 15 kHz",
                "The sample rate is 2 × 15 kHz. With N = 4 taps, the filter is short."):
        r = score(ans, {"value": 30, "unit": "kHz"})
        assert r["method"] == "judge" and r["details"]["reason"] == "expression_answer", ans
    for ans, ref in (("Sample rate:\n* 30 kHz", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 30 kHz double-sided bandwidth", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 1000 times the input power", {"value": 1000, "unit": ""})):
        assert score(ans, ref)["credit"] == 1.0, ans


def test_empty_open_ended_answer_has_its_own_method():
    assert score("", {"required_points": ["x"]})["method"] == "rules"
    assert score("x", {"required_points": ["x"]}, truncated=True)["method"] == "rules"


def test_prose_final_answer_is_never_a_marker():                       # F3
    for ans in ("To get the final answer, multiply the 15 kHz bandwidth by 2. The sample rate is 30 kHz.",
                "Before giving the final answer, note the 15 kHz bandwidth.\nThe sample rate must be 30 kHz."):
        assert score(ans, {"value": 30, "unit": "kHz"})["credit"] != 0.0


def test_part_zero_reaches_the_batch_rows(tmp_path):                  # F4, R11
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps({"answer": "Final answer: Ts = 1 s, B = 1 kHz",
                             "reference": [{"value": 1, "unit": "ms", "label": "Ts"},
                                           {"value": 1, "unit": "kHz", "label": "B"}]}) + "\n")
    rows, errors = rules._batch(p)
    assert errors == 0 and rows[0]["rule_only_credit"] == 0.0


def test_rubric_guide_db_template_parses():                           # F6
    guide = (REPO / "docs" / "guidelines" / "rubric_writing.md").read_text(encoding="utf-8")
    assert '{"value": <number>, "unit": "dBm", "abs_tolerance": <tolerance in dB>}' in guide
    parse_reference(json.loads('{"value": -90, "unit": "dBm", "abs_tolerance": 0.5}'))


def test_comparator_and_named_block():                                # F8, R6
    assert score("Final answer: at least 30 kHz", {"value": 30, "unit": "kHz"})["method"] == "judge"
    assert score("Final answer: the Throttle block (set `sample_rate` to `samp_rate`).",
                 {"exact": "blocks_throttle", "aliases": ["Throttle"]})["credit"] != 0.0


def test_a_wrong_identifier_must_be_a_real_block_id():                       # R6
    r = score("Add a Throttle block and set its `samples_per_second` to `samp_rate`.", {"exact": "blocks_throttle"})
    assert r["method"] == "judge"
    r = score("Final answer: `blocks_head`", {"exact": "blocks_throttle"})
    assert (r["credit"], r["details"]["reason"]) == (0.0, "wrong_identifier")


def test_sums_and_quotients_left_unevaluated_go_to_the_judge():               # R8
    for ans, ref in (("Final answer: 15 kHz + 15 kHz", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 60 kHz over 2", {"value": 30, "unit": "kHz"}),
                     ("Final answer: 35 kHz minus 5 kHz", {"value": 30, "unit": "kHz"})):
        assert score(ans, ref)["details"]["reason"] == "expression_answer", ans
    assert score("Final answer: 15 kHz + 15 kHz = 30 kHz", {"value": 30, "unit": "kHz"})["credit"] == 1.0


# --- interface, determinism, CLI ------------------------------------------------------------------


def test_result_shape_and_json():
    for case in rules.SELFTEST:
        r = score(case["answer"], case["reference"], truncated=case.get("truncated", False))
        assert set(r) == {"method", "credit", "details"} and r["method"] in {"numeric", "symbolic", "rules", "judge"}
        assert (r["credit"] is None) == (r["method"] == "judge")
        assert r["credit"] is None or 0.0 <= r["credit"] <= 1.0
        assert "reason" in r["details"]
        json.dumps(r, allow_nan=False)
        assert score(case["answer"], case["reference"], truncated=case.get("truncated", False)) == r
    judged = score("It is 30 kHz, or maybe 20 kHz.", K30)
    assert judged["details"]["rule_method"] == "numeric" and judged["details"]["final_answer"]


def test_policies_are_listed_for_the_preregistration():
    doc = rules.__doc__
    for name in ("CREDIT_TIERS", "HALF_BAND_EXTRA", "DEFAULT_TOLERANCE", "DB_TOLERANCE", "MAGNITUDE_RATIO",
                 "unit_required", "rule_only_credit", "rule_only_summary", "ANSWER_FORMAT_INSTRUCTION",
                 "item_credit", "revised_after_answer"):
        assert name in doc, name
    assert all(f"P{i} " in doc for i in range(1, 11))


def test_selftest_passes():
    rows = rules.selftest()
    assert len(rows) >= 70 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert rules.main(["--selftest"]) == 0


def test_cli_single_and_batch(tmp_path, capsys):
    assert rules.main(["--answer", "2500 kS/s", "--reference", '{"value": 2.5, "unit": "MS/s"}']) == 0
    assert json.loads(capsys.readouterr().out)["credit"] == 1.0
    rows = [{"id": "a", "answer": "27 dB", "reference": {"value": 27, "unit": "dBm"}},
            {"id": "b", "answer": "<think>x</think>30 kHz", "reference": K30},
            {"id": "c", "answer": "30 kHz", "reference": K30, "truncated": True},
            {"id": "d", "answer": "maybe 30 kHz or 20 kHz", "reference": K30},
            {"id": "e", "answer": "prose", "reference": {"numeric": None, "points": ["x"]}}]
    src = tmp_path / "in.jsonl"
    write_jsonl(src, rows)
    outs = []
    for name in ("o1.jsonl", "o2.jsonl"):
        assert rules.main(["--batch", str(src), "--out", str(tmp_path / name)]) == 0
        outs.append((tmp_path / name).read_bytes())
    assert outs[0] == outs[1]
    got = {r["id"]: (r["method"], r["credit"]) for r in read_jsonl(tmp_path / "o1.jsonl")}
    assert got == {"a": ("numeric", 0.0), "b": ("numeric", 1.0), "c": ("numeric", 0.0), "d": JUDGE, "e": JUDGE}
    assert all(r["rules_version"] == rules.RULES_VERSION for r in read_jsonl(tmp_path / "o1.jsonl"))
    err = capsys.readouterr().err
    assert "deferred to the judge 2/5" in err and "decided-only mean credit 0.3333 over 3 decided" in err
    assert rules.main([]) == 2


def test_batch_rows_carry_the_item_credit(tmp_path):
    """F4: a consumer of --batch rows gets the P9/P10 credit as rule_only_credit."""
    rows = [{"id": "p", "answer": "Ts = 1 ms and B = 10 kHz", "reference": PARTS_TB},
            {"id": "q", "answer": "Ts = 1 ms and B = 1 kHz", "reference": PARTS_TB},
            {"id": "r", "answer": "maybe 30 kHz or 20 kHz", "reference": K30}]
    src = tmp_path / "in.jsonl"
    write_jsonl(src, rows)
    assert rules.main(["--batch", str(src), "--out", str(tmp_path / "o.jsonl")]) == 0
    out = {r["id"]: r for r in read_jsonl(tmp_path / "o.jsonl")}
    assert (out["p"]["credit"], out["p"]["details"]["item_credit"], out["p"]["rule_only_credit"]) == (0.5, 0.0, 0.0)
    assert (out["q"]["credit"], out["q"]["rule_only_credit"]) == (1.0, 1.0)
    assert (out["r"]["credit"], out["r"]["rule_only_credit"]) == (None, 0.0)


def test_batch_keeps_going_past_a_malformed_reference(tmp_path, capsys):
    rows = [{"id": "a", "answer": "30 kHz", "reference": K30},
            {"id": "b", "answer": "20 dB", "reference": {"value": 20, "unit": "dB", "tolerance": 0.01}},
            {"id": "c", "answer": "x"},
            {"id": "d", "answer": "1 ms", "reference": TS}]
    src = tmp_path / "in.jsonl"
    write_jsonl(src, rows)
    assert rules.main(["--batch", str(src), "--out", str(tmp_path / "o.jsonl")]) == 1
    out = {r["id"]: r for r in read_jsonl(tmp_path / "o.jsonl")}
    assert len(out) == 4 and out["a"]["credit"] == 1.0 and out["d"]["credit"] == 1.0
    for bad in ("b", "c"):
        assert out[bad]["method"] is None and out[bad]["details"]["reason"] == "bad_reference"
    assert "2 row(s) with a malformed reference" in capsys.readouterr().err
