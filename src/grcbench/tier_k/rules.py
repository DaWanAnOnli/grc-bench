"""Rule-first scoring cascade for Tier-K answers: parse -> numeric -> symbolic -> judge.

Implements proposal §5.1 (Tier K: "Calculation items get tiered relative-error credit, and unit or
order-of-magnitude errors (e.g., dB vs dBm) score zero. A rule-first cascade (parse -> numeric ->
symbolic) runs before any LLM judging"; "Development-set checks between training runs are scored by the
rule cascade only") and eval plan §4.2 "Scoring" (rules first, the judge only when rules cannot decide;
partial credit by relative error; a wrong unit scores zero), §5.2 (the rubric's numeric tolerance) and
§7.1 (reasoning text is removed before scoring; an answer cut off at the length cap is scored as wrong).

    score(answer_text, reference, truncated=False)
        -> {"method": "numeric" | "symbolic" | "rules" | "judge", "credit": float in [0, 1] | None,
            "details": {...}}

"rules" is a truncated or empty answer to an open-ended rubric (no rule target), scored 0 by rule.

"judge" means the rules cannot decide: credit is None, and details carry the reason, the extracted
candidate(s) and the final-answer text for the judge (eval plan §8). Every result names its reason in
details["reason"]. The rules decide from the answer's form, never from which reading happens to be
right: a number that could be read two ways is read by the pre-registered conventions (P4, P8), and
a result that still holds two readings of different credit gets the lower one (P11); the judge
reads only what the rules cannot (pre-registration §8, decision 7). On rule-only checks (dev-split checkpoint screening,
eval plan §4.1 and §10.3, "scored by rules only") a judge result scores 0 (rule_only_credit) and the
deferral rate is reported with the score (P10).

Reference forms (a rubric's numeric field, or a target for block and choice questions):
  {"value": 2.5, "unit": "MS/s", "tolerance": 0.01}   numeric. A rubric {"numeric": {...} | [...]}
      also works; a rubric whose numeric is null or absent (an open-ended item) goes to the judge.
      tolerance: relative (default DEFAULT_TOLERANCE). abs_tolerance: absolute, in the reference unit
      (values near zero; an error within it also lifts the order-of-magnitude and sign checks).
      A decibel reference takes its tolerance in dB only: abs_tolerance (a rubric; the rubric guide
      and judge_prompt.NUMERIC_KEYS) or tolerance_db (the calculation key), default DB_TOLERANCE. A
      relative `tolerance` alone on a decibel reference is refused (ValueError: it is ambiguous), and
      next to a dB tolerance it is ignored (calc.py writes both).
      unit_required: a bare number scores 0 instead of being read in the reference unit.
      waveform: "sine" or absent; with "sine" an RMS voltage converts to peak (factor √2).
      accept_units: spellings treated as the reference unit (e.g. ["dB"] for a dBi gain).
      label: str or list, the quantity's name ("Ts"), for questions asking for several numbers.
  [{...}, {...}]          several numeric parts; credit = mean of the parts (the per-part input
                          judge_prompt.explain_score combines), each part's result in details["parts"];
                          the judge decides if any part is undecided. The item's credit under P9 is
                          details["item_credit"]: 0 when any decided part has a unit or order-of-
                          magnitude error (reason "part_zero", details["part_zero"]), else the mean;
                          judge_prompt.explain_score (error DU) and rule_only_credit apply the same rule
  {"set": [...], "universe": [...], "aliases": {...}}    the answer names exactly this set of options
  {"exact": "blocks_throttle", "aliases": [...], "universe": [...]}   one identifier or option; a bare
                          str means this; a list or set means {"set": ...} without a universe

1. Parse. Reasoning text is stripped first (Tier F's rule, plus unclosed and template-opened blocks).
   The final answer is, in order: the last \\boxed{...}; else the text after the last "final answer" /
   "answer is" / "answer:" marker; else the whole answer. "final answer" is a marker only before ":",
   "=" or "is", alone on its line (a heading, "## Final Answer") or as a bold span ("**Final
   answer**"); in prose ("To get the final answer, multiply ...", "Before giving the final answer,
   ...") it is not one. The final answer is read in widening regions: the first clause (ended by ". ",
   "; " or ": "), the line, the paragraph and the rest; for options also the clause without its
   parenthetical remarks. The text after the region a decision is read from (the rest of the
   paragraph, later paragraphs, the text after a box) is where revisions are looked for (2, 3).
   Labels (numeric parts): inside the final answer (the boxed text or the first paragraph after the
   marker; the whole answer when there is neither) the label scopes it: the segments (sentences,
   clauses, comma or "and" items, lines) that mention the label and no other part's label, then those
   segments with the unlabelled segments that follow them ("15 m range resolution, a time-bandwidth
   product of 100" gives each part its own segment). Failing that, the text after the label's last
   mention that has a number nearby, cut at another part's label. A label outside the final answer is
   not used ("Ts = 2 ms at first. Final answer: 1 ms" reads 1 ms). A quantity whose unit fits another
   part's dimension and not this part's belongs to the other part and is skipped.
   Numbers: signs (also U+2212), decimals, thousands separators ("," "_", thin and no-break spaces, in
   groups of three), scientific notation (2.5e6, 2.5 x 10^6, 2.5×10⁶, 1.5*10**6, \\times 10^{6}),
   powers (2^10), fractions (1/4, \\frac{1}{4}), multiples of pi (π/2, 3\\pi/4), multipliers (10k,
   2.5M samples/s, 1.5 million samples per second; a bare "10k" or "10 thousand" is in SI base units,
   10 000 Hz for a kHz reference; an attached K alone is kelvin for a temperature reference and 1 000
   otherwise, as in 290K and 48K). A comma never marks decimals and a plain space never groups digits.
   Numbers inside identifiers (log10, f1, 8PSK, version 3.10.12), list markers ("1. text" opening a
   line) and labels ("Step 2:") are skipped. An uncertainty after "±" is dropped and lends its unit to
   the number before it.
   Units: SI prefixes T G M k m μ/u n p on Hz, s, m, W, V, A, Ω, J, bit, Sa, sym, Bd, rad; S/s, Sa/s,
   ksamples/s in any case, sps/Msps (a sample OR symbol rate: whichever fits the reference), bit/s,
   b/s, bps, B/s, bytes, samples, symbols, bits, deg/°, cycles (a count, so cycles/s = Hz and
   cycles/sample is a normalised frequency, or 2π rad), %, ppm, K, °C, °F, min, h, dB, dBm, dBW, dBV,
   dBmV, dBuV, dBi, dBd, dBc, dBFS, dB-Hz, dBm/Hz, dBc/Hz; compounds with "/", "per" and "·";
   exponents; unit words (megahertz, "kilo samples per second", msec, μsec, nsec, microsec, ...);
   voltage conventions Vpp, Vpk, Vp, Vrms. Common miscapitalisations (KHz, Kbit, mhz, Mbps, dbm) are
   read as meant; "mHz" written exactly so is millihertz. A LaTeX "~" between a number and its unit
   is a space ("30~\\text{kHz}"); before a number it means "about".
2. Numeric. In the decision region (the narrowest region holding a quantity) the candidate is the last
   quantity outside parentheses with the reference's dimension; else the last with another unit (a
   unit error); else the last bare number, read in the reference unit. The rule for several numbers
   (the final stated answer): the rules decide only if every other value in the region that would
   earn a different credit is a step towards the candidate: an operand ("2 × 15 kHz", "twice the
   15 kHz"), a value followed by "=", "≈", "→" or a conclusion word ("so", "therefore", "gives") before
   the candidate, or a parenthetical explanation ("(twice 15 kHz)"); and the candidate is not itself
   an operand ("30 kHz = 2 × 15 kHz"). The first value after an explicit final-answer marker, or in a
   box, is the stated answer and never a step: "Final answer: 30 kHz, giving a Nyquist frequency of
   15 kHz" goes to the judge. Without a marker the rules cannot tell a consequence from a conclusion:
   "The minimum sample rate is 30 kHz, which gives a Nyquist frequency of 15 kHz" is decided on
   15 kHz like "The bandwidth is 15 kHz, so the sample rate must be 30 kHz" on 30 kHz (a known limit
   of the rule). A prose marker ("the answer is") counts only where the answer has no explicit one,
   so "Final answer: 12 kHz. The answer is valid for the 48 kHz input only." is decided on 12 kHz. Hedges ("30 kHz, or maybe 20 kHz", "10/20/30 kHz", "20-30 kHz",
   "between 20 and 30 kHz"), revisions ("first 20 kHz; with the factor of two it is 30 kHz") and
   context values ("the 48 kHz input ... 12 kHz") go to the judge, whichever value is right; so does
   a bare candidate whose line or paragraph in the final answer holds a value with the reference's
   dimension at another credit ("Answer: 30. The input was 48 kHz", "Answer: 1. 30 kHz"). So does
   a candidate with: a negation in its clause ("not 30 kHz"); a comparator next to it ("less than
   30 kHz", "at least 30 kHz", "≥ 30 kHz", "30 kHz or more": a bound, not a value; one earlier in a
   derivation, "fs ≥ 2B = 30 kHz", does not count); digits grouped by a plain space or a decimal
   comma; an unknown unit after it ("2500 kSamp/s"; after a bare candidate, any word that is no unit
   and not in UNIT_STOP_WORDS: "3.84 Mcps", but not "4 taps"); a number in words that could be the answer
   ("thirty kilohertz"); a loss word next to a negative value ("a loss of -3 dB", P8); or an
   unevaluated expression ("Final answer: 2 × 15 kHz", "15 kHz + 15 kHz", "60 kHz over 2": a factor
   of ×, ·, "times", "twice", "half" ..., or a term of "+", "plus", "minus", "over", with no
   relation (=, ≈, →, ...) after it in its sentence, expression_answer; a difference written with
   "-" is read on its last term, as "-" also marks ranges and separators). So does a decided candidate
   that the text after it revises ("revised_after_answer"): a retraction ("that's wrong", "I made an
   error"); a revision word ("actually", "wait,", "no,", "correction", "hmm", "on second thought",
   "recheck", ...) with a value of any unit (or a bare one) at another credit; or a hedge word ("or",
   "maybe", "instead", "not", ...) with a value of the reference's dimension at another credit.
   Operands there are steps ("recheck: 2 × 12 kHz = 24 kHz" counts 24 kHz, not 12 kHz); commentary
   without such a word ("Final answer: 30 kHz\\n\\nA 37.5 kHz rate leaves a guard band", "A common
   mistake is 15 kHz") leaves the decision alone. A single quantity with a unit of the reference's
   dimension and no other text ("-114.2 dBm", as judge_prompt passes the judge's values) is always
   decided, except for the readings in P8.
   Credit: the units must have the same dimension, else credit 0 ("unit_mismatch": dB vs dBm, Hz vs s,
   bit/s vs Hz, symbols vs samples, rad/s vs Hz). Samples count as dimensionless, so a sample rate in
   Hz (GNU Radio's samp_rate) equals the same rate in S/s. dBm/dBW convert to W, dBV/dBmV/dBuV to V,
   dBd to dBi (+2.15 dB), °C/°F to K. A plain dB (dB/Hz) value is read in the reference's relative
   decibel unit (dBc, dBc/Hz, dBi, dBd, dBFS); against an absolute one (dBm, dBW, dBV, ...) it is a
   unit error. Voltages: Vp is Vpk; Vpp and Vpk convert exactly (factor 2); Vrms converts to peak only
   when the reference records waveform "sine" (factor √2), otherwise a different convention scores 0
   ("voltage_convention", a unit error under P9); a plain V names no convention and is compared as
   written. Linear units: a ratio to the reference of >= MAGNITUDE_RATIO or
   <= 1/MAGNITUDE_RATIO is an order-of-magnitude error and a sign flip is wrong; both score 0 whatever
   the tolerance, unless the error is within an explicit abs_tolerance. Otherwise the relative error
   |a - r| / |r| is graded: <= tolerance -> 1.0; <= min(5 x tolerance, tolerance + HALF_BAND_EXTRA)
   -> 0.5 (with the default 1 %: 1 % and 5 %); otherwise 0. Decibel references: the error is the
   absolute difference in dB, graded the same way (the half band ends at min(5 x tol, tol + the dB
   equivalent of HALF_BAND_EXTRA)); an error of 10*log10(MAGNITUDE_RATIO) dB or more (20*log10 for
   dBV) is an order of magnitude (score 0).
3. Symbolic (no computer algebra). Sets: decided only from a clean list of known options ("A, C",
   "complex and float", bullets; "both"/"only" allowed); "or" is a hedge, not a list. With one-letter
   options, the letters leading an explicit final answer ("(B), since (A) is wrong" reads B); a bare
   capital is a letter only before the end of the clause, punctuation, "&", "+" or "and"/"or" and
   another letter, so in "A low-pass filter is needed, so option C" the "A" is the article. The set
   found must equal the target: an extra or a missing option scores 0. Without a universe an unknown
   piece cannot be told from an extra option, so it goes to the judge. Prose naming options goes to
   the judge. Exact: the normalised answer (a region) equals the target or an alias; or, with no
   negation, contrast or hedge word in the final answer ("not", "but", "only", "rather", "wrong",
   "either", ...), the first clause of the final answer names identifiers (code spans, snake_case
   names; `blocks.throttle` reads blocks_throttle, an instance blocks_throttle_0 reads blocks_throttle)
   and they are exactly the target (1.0) or only other identifiers with the target named nowhere
   (0.0). When the target is a GNU Radio block id, those other identifiers must all be block ids
   of the frozen catalog (reports/block_catalog.json): parameter and variable names ("set
   `samples_per_second` to `samp_rate`") name a block only in words, so they go to the judge
   (not_a_block_id). A plain word named in prose ("complex or float" for "complex") goes to the judge.
   Every symbolic decision is then checked, as numbers are (P7): it goes to the judge if a
   parenthetical the match left out holds a hedge, contrast or addition word or another option or
   identifier ("complex (or float)", "complex (float also works)", "`blocks_throttle` (or
   `blocks_head`)"); or if the text after it revises it (a retraction or a revision word: "complex?
   No, float.", "B. Hmm, actually C.") or holds a hedge, contrast or addition word next to another
   option or identifier. Without a universe a plain word is not known to be an option, so only
   revision words, identifiers and the target's aliases are seen there.
4. Otherwise "judge", with the candidate.

POLICIES that change credit (PROPOSAL, to be fixed in the G2 pre-registration):
  P1  credit tiers: CREDIT_TIERS (full credit within the tolerance; half credit up to 5 x tolerance,
      capped at tolerance + HALF_BAND_EXTRA), DEFAULT_TOLERANCE, DB_TOLERANCE. The error is relative,
      |a - r| / |r|, as in the key sheet's blind recomputation ("within ±1 %"), not a log ratio; it is
      not symmetric in ratio terms, but with the cap a halving earns credit only from a tolerance of
      0.46 and a doubling only from 0.96 (the guideline forbids tolerances that let a factor-2 slip
      pass). The reference's unit sets the scale the error is measured on: a decibel reference is
      graded in dB, by default 0.1 dB for full credit and about 0.27 dB for half (about 2.3 % and
      6.4 % in power, 1.2 % and 3.2 % in amplitude; for dBV, dBmV and dBuV, whose decibels are
      20*log10 of a voltage, the half band ends at about 0.44 dB, 5.2 % in voltage), a linear one at
      1 % and 5 %. So dB items are
      graded somewhat more leniently than linear ones, deliberately: 0.1 dB is the usual precision of
      a decibel value, and an answer correctly rounded to it errs by up to 0.05 dB (1.2 % in power),
      more than 10*log10(1.01) = 0.043 dB, the dB equivalent of the linear 1 %. An answer in the
      other scale is converted to the reference's (1.02 mW for 0 dBm: 0.086 dB, full credit; 0.086 dBm
      for 1 mW: 2 %, half credit). Rubric writers give the reference in the unit the question asks for.
  P2  MAGNITUDE_RATIO: an order-of-magnitude error scores 0; so does a sign flip on a linear unit;
      an explicit abs_tolerance lifts both.
  P3  a unit of another dimension scores 0; samples are dimensionless (Hz == S/s); cycles are a count.
      Against a reference with no unit, bits, bytes and symbols count items (ITEM_COUNT_UNITS: "34 bytes"
      is 34 byte items), and a GNU Radio item word after a number is no unit (UNIT_STOP_WORDS: "34
      vectors", "34 short items").
  P4  a bare number (decision 7; the Tier-K prompt says "a number without a unit is never read as
      decibels"): against a decibel reference it is a linear ratio, to the unit's reference level for an
      absolute unit (1 mW for dBm, 1 W for dBW, 1 V, 1 mV or 1 μV for dBV, dBmV, dBuV) and of kind
      ratio_kind for a relative one (power by default: 1000 is 30 dB; amplitude: 1000 is 60 dB),
      reason converted_ratio; against a dimensionless reference it is read as is; against any other
      reference it is read in the reference unit, with no other reading (30000 for 30 kHz is 30000 kHz).
      A bare number with an attached multiplier (10k, 2.5M) is in SI base units (against a decibel
      reference it is a ratio: 10k is 40 dB); an attached K is kelvin only for a temperature reference.
      Where a value in the reference's own form is also given, it is the candidate before a bare or
      converted one ("13 dB, so N = 16"). unit_required: a bare number scores 0 (unit_missing).
      A ratio word after it (RATIO_WORDS: "1000 times", "1000x") makes it a plain ratio with a unit
      against a dimensionless or decibel reference; against any other reference the word is ignored.
  P5  the final-answer rule, the rule for several numbers and the revision rule (1 and 2 above): the
      final stated answer counts, and a later revision of it, a hedge or a comparator next to it
      sends the answer to the judge.
  P6  truncated and empty answers score 0 (eval plan §7.1), reasoning text is removed first.
  P7  sets are scored by exact equality (an extra or a missing option scores 0); a hedged, extended
      or revised option answer goes to the judge (3 above).
  P8  the other form (decision 7): a decibel answer for a dimensionless reference (a ratio of kind
      ratio_kind: 20 dB is 100 in power, 10 in amplitude), a ratio with a unit (%, ppm, a ratio word)
      for a relative decibel reference and an angle for a dimensionless reference (degrees to radians)
      are converted and graded (converted_db, converted_ratio, converted_angle). With form_required
      (the question fixes the form) the other form scores 0 as wrong_form when its converted value
      would have earned credit. A plain dB value for a relative decibel reference (dBc, dBi, dBd,
      dBFS) is read in that unit (2 above). Sign: a loss word (LOSS_WORDS) right before the value ("a
      loss of 3 dB", "attenuated by 3 dB", "lags by 90 degrees") or right after it ("3 dB down",
      "40 dBc below the carrier", "10 kHz below the carrier") negates a positive relative-decibel or
      linear value when the reference is negative or marked quantity "gain" (loss_word_sign); a
      positive reference without that mark is a loss quantity, so "a loss of 3 dB" stays 3 dB; a
      negative value with a loss word goes to the judge (sign_unclear). Without a loss word next to
      it, a value of the opposite sign scores 0 (with a sign word elsewhere in its clause, by P11).
  P9  parts: two parts of one dimension decided on the same value, one of them without its label
      (scope or anchor), go to the judge (part_unassigned): the rules cannot tell whose value it is.
      A unit or order-of-magnitude error in any decided part scores the whole answer 0
      (PART_ZERO_REASONS; details["item_credit"] and reason "part_zero"); otherwise the item's credit
      is the mean over the parts. A label whose own segments give it only a value in another part's
      unit ("Ts = 1 ms and B = 1 ms") has a unit error. score()'s `credit` stays the mean over the
      parts (the per-part input judge_prompt.explain_score combines, as its tests pin); the item
      credit is applied by judge_prompt.explain_score (DU), by rule_only_credit and in --batch rows.
  P10 rule-only checks (dev-split screening): report together, as rule_only_summary gives them, the
      deferral rate, the mean of rule_only_credit (a judge result counts 0, so it is a lower bound)
      and the mean over the decided items (the rules' accuracy where they decide, which favours
      answers whose form they read). The deferral rate depends on the answer's form more than on its
      content, so the Tier-K prompt should ask for ANSWER_FORMAT_INSTRUCTION's final-answer line
      (proposed here; it is part of the prompt fixed at G2), and before G2 the deferral rate and the
      decided items' accuracy are measured on dev-split answers of the smoke-test model.
  P11 tie-break (decision 7): a result, or a part, that still holds two readings of different credit
      (details literal_credit and alt_credit, READING_REASONS) is decided with the lower one
      (tie_break_lower). reading_counts gives, per model, the answers decided by the tie-break and the
      answers scored 0 for the wrong form alone, reported next to every score.

`./gb score-k --selftest` runs the hand-made cases in SELFTEST; `--answer/--reference` scores one
answer; `--batch` scores a JSONL file: each row carries score()'s result and its rule_only_credit
(P9, P10), stderr the rule_only_summary figures (a row whose reference is malformed gets an error row
and the exit code is 1). Nothing here calls a model. Outputs are deterministic.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from grcbench.common.io import iter_jsonl, write_jsonl
from grcbench.common.paths import REPORTS
from grcbench.tier_f.grcfile import strip_reasoning as _strip_think

# --- PROPOSAL values, to be fixed in the G2 pre-registration -------------------------------------
# Tiered credit: (multiple of the item's tolerance, credit); every tier after the first also ends at
# tolerance + HALF_BAND_EXTRA (relative). With the default 1 % tolerance: <= 1 % -> 1.0,
# <= 5 % -> 0.5, otherwise 0. With a 10 % tolerance: <= 10 % -> 1.0, <= 14 % -> 0.5.
CREDIT_TIERS: tuple[tuple[float, float], ...] = ((1.0, 1.0), (5.0, 0.5))
HALF_BAND_EXTRA = 0.04          # relative: the half-credit band is at most 4 points wider than full
DEFAULT_TOLERANCE = 0.01        # relative, when the rubric gives none
DB_TOLERANCE = 0.1              # dB, absolute, for decibel references (about 2.3 % in power)
MAGNITUDE_RATIO = 9.5           # answer/reference >= 9.5 or <= 1/9.5 (0.105): order-of-magnitude error

# The answer-format line proposed for the Tier-K prompt (P10; fixed with the prompt at G2): an answer
# that ends with it is read from its marker, so the rules decide it whatever the reasoning before it.
ANSWER_FORMAT_INSTRUCTION = (
    'End your answer with one line "Final answer: <value> <unit>". If the question asks for several '
    'quantities, name each: "Final answer: <name> = <value> <unit>, <name> = <value> <unit>". If it asks '
    'for options or identifiers, list exactly the ones you choose: "Final answer: <option>, <option>". '
    "Write nothing after that line.")

RULES_VERSION = "tier-k-rules 0.5"
_CLIP = 600                     # characters of final-answer text kept in details

# --- units ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One reading of a unit. Linear units: value * scale + zero is SI (zero: °C, °F). Decibel units
    of a linear quantity (log = 10 for power-like, 20 for amplitude-like): SI value = scale * 10 **
    (x / log). Relative decibel units (dB, dBi, dBc, ...) have a pseudo-dimension of their own and
    log = 0."""

    name: str
    dims: tuple[tuple[str, int], ...] = ()
    scale: float = 1.0
    log: int = 0
    db: bool = False
    offset: float = 0.0          # relative dB units: added to reach the family's base (dBd -> dBi)
    zero: float = 0.0            # linear units with an offset zero (°C, °F)


def _d(**exps: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((k, v) for k, v in exps.items() if v))


DIMENSIONLESS = Unit("")
_F, _T = _d(time=-1), _d(time=1)
_TEMP = _d(temperature=1)
_ANGLE = _d(angle=1)
_KELVIN = Unit("K", _TEMP)
_PREFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3, "c": 1e-2, "m": 1e-3, "μ": 1e-6,
           "u": 1e-6, "n": 1e-9, "p": 1e-12}
_WORD_PREFIX = {"": 1.0, "kilo": 1e3, "mega": 1e6, "giga": 1e9, "milli": 1e-3, "micro": 1e-6,
                "nano": 1e-9, "pico": 1e-12, "centi": 1e-2}


def _pow(u: Unit, e: int) -> Unit:
    if e == 1:
        return u
    return Unit(f"{u.name}^{e}", tuple(sorted((k, v * e) for k, v in u.dims)), u.scale ** e)


def _combine(a: Unit, b: Unit, sign: int) -> Unit | None:
    if a.db or b.db or a.zero or b.zero:
        return None
    c = Counter(dict(a.dims))
    for k, v in b.dims:
        c[k] += sign * v
    return Unit(f"{a.name}{'/' if sign < 0 else '·'}{b.name}",
                tuple(sorted((k, v) for k, v in c.items() if v)), a.scale * b.scale ** sign)


def _tables() -> tuple[dict[str, tuple[Unit, ...]], dict[str, tuple[Unit, ...]],
                       dict[str, Unit]]:
    exact: dict[str, tuple[Unit, ...]] = {}
    folded: dict[str, tuple[Unit, ...]] = {}

    def put(table: dict, key: str, *units: Unit) -> None:
        table.setdefault(key, units)

    # SI symbols, exact case: (symbol, dims, scale, allowed prefixes, usable without a prefix)
    volt = _d(voltage=1)
    symbols = [
        ("Hz", _F, 1.0, "TGMkm", True), ("s", _T, 1.0, "mμunp", True),
        ("m", _d(length=1), 1.0, "kcmμun", True), ("W", _d(power=1), 1.0, "GMkmμunp", True),
        ("V", volt, 1.0, "kmμun", True), ("Vpp", volt, 1.0, "mμu", True),
        ("Vrms", volt, 1.0, "mμu", True), ("Vpk", volt, 1.0, "mμu", True), ("Vp", volt, 1.0, "mμu", True),
        ("A", _d(current=1), 1.0, "mμun", True), ("Ω", _d(voltage=1, current=-1), 1.0, "GMkm", True),
        ("J", _d(power=1, time=1), 1.0, "kmμunp", True),
        ("sec", _T, 1.0, "mμun", False), ("secs", _T, 1.0, "mμun", False),
        ("bit", _d(bit=1), 1.0, "kKMGT", True), ("bits", _d(bit=1), 1.0, "kKMGT", True),
        ("b", _d(bit=1), 1.0, "kMGT", False), ("B", _d(bit=1), 8.0, "kKMGT", False),
        ("Sa", (), 1.0, "kMG", True), ("sample", (), 1.0, "kMG", True), ("samples", (), 1.0, "kMG", True),
        ("sym", _d(symbol=1), 1.0, "kMG", True), ("symbol", _d(symbol=1), 1.0, "kMG", True),
        ("symbols", _d(symbol=1), 1.0, "kMG", True), ("Bd", _d(symbol=1, time=-1), 1.0, "kMG", True),
        ("rad", _ANGLE, 1.0, "mμu", True),
    ]
    for sym, dims, scale, prefixes, bare in symbols:
        if bare:
            put(exact, sym, Unit(sym, dims, scale))
        for p in prefixes:
            put(exact, p + sym, Unit(p + sym, dims, scale * _PREFIX[p]))
    for name, power in (("KiB", 1), ("MiB", 2), ("GiB", 3)):
        put(exact, name, Unit(name, _d(bit=1), 8.0 * 1024 ** power))
    for name, unit in (("min", Unit("min", _T, 60.0)), ("h", Unit("h", _T, 3600.0)),
                       ("deg", Unit("deg", _ANGLE, math.pi / 180)),
                       ("°", Unit("°", _ANGLE, math.pi / 180)), ("K", _KELVIN),
                       ("%", Unit("%", (), 0.01)), ("ppm", Unit("ppm", (), 1e-6)),
                       ("ppb", Unit("ppb", (), 1e-9)), ("kohm", Unit("kΩ", _d(voltage=1, current=-1), 1e3)),
                       ("Mohm", Unit("MΩ", _d(voltage=1, current=-1), 1e6))):
        put(exact, name, unit)

    # unit words, any case: (forms, dims, scale, takes word prefixes such as kilo/mega)
    words = [
        (("hertz",), _F, 1.0, True), (("second", "seconds"), _T, 1.0, True),
        (("sec", "secs"), _T, 1.0, True), (("minute", "minutes", "mins"), _T, 60.0, False),
        (("hour", "hours", "hr", "hrs"), _T, 3600.0, False),
        (("meter", "meters", "metre", "metres"), _d(length=1), 1.0, True),
        (("watt", "watts"), _d(power=1), 1.0, True), (("volt", "volts"), volt, 1.0, True),
        (("amp", "amps", "ampere", "amperes"), _d(current=1), 1.0, True),
        (("ohm", "ohms"), _d(voltage=1, current=-1), 1.0, True),
        (("joule", "joules"), _d(power=1, time=1), 1.0, True),
        (("bit", "bits"), _d(bit=1), 1.0, True), (("byte", "bytes"), _d(bit=1), 8.0, True),
        (("sample", "samples"), (), 1.0, True), (("symbol", "symbols"), _d(symbol=1), 1.0, True),
        (("radian", "radians"), _ANGLE, 1.0, True),
        (("degree", "degrees"), _ANGLE, math.pi / 180, False),
        (("kelvin",), _TEMP, 1.0, False), (("percent",), (), 0.01, False),
    ]
    for forms, dims, scale, prefixed in words:
        for wp, f in (_WORD_PREFIX.items() if prefixed else (("", 1.0),)):
            for form in forms:
                put(folded, wp + form, Unit(wp + form, dims, scale * f))
    put(folded, "kilohm", Unit("kΩ", _d(voltage=1, current=-1), 1e3))
    put(folded, "kilohms", Unit("kΩ", _d(voltage=1, current=-1), 1e3))
    for form in ("cycle", "cycles"):        # a count (cycles/s = Hz, cycles/sample) or a turn of 2π rad
        put(folded, form, Unit(form, (), 1.0), Unit(form, _ANGLE, 2 * math.pi))

    # families read in any case: an "m" here means mega (mhz, mbps, msps, msamples)
    for p, f, label in (("", 1.0, ""), ("k", 1e3, "k"), ("m", 1e6, "M"), ("g", 1e9, "G"), ("t", 1e12, "T")):
        put(folded, p + "hz", Unit(label + "Hz", _F, f))
        put(folded, p + "bps", Unit(label + "bit/s", _d(bit=1, time=-1), f))
        put(folded, p + "baud", Unit(label + "Bd", _d(symbol=1, time=-1), f))
        put(folded, p + "bd", Unit(label + "Bd", _d(symbol=1, time=-1), f))
        sps = [Unit(label + "S/s", _F, f), Unit(label + "sym/s", _d(symbol=1, time=-1), f)]
        if not p:   # GNU Radio's "sps" is also samples per symbol
            sps.append(Unit("samples/symbol", _d(symbol=-1)))
        put(folded, p + "sps", *sps)
        if p:       # kSamples, MSample, Gsamples
            for form in ("sample", "samples", "sa"):
                put(folded, p + form, Unit(label + "S", (), f))

    # decibels, any case
    for key, unit in (("db", Unit("dB", _d(dB=1), db=True)), ("decibel", Unit("dB", _d(dB=1), db=True)),
                      ("decibels", Unit("dB", _d(dB=1), db=True)),
                      ("dbm", Unit("dBm", _d(power=1), 1e-3, 10, True)),
                      ("dbw", Unit("dBW", _d(power=1), 1.0, 10, True)),
                      ("dbv", Unit("dBV", volt, 1.0, 20, True)),
                      ("dbmv", Unit("dBmV", volt, 1e-3, 20, True)),
                      ("dbuv", Unit("dBuV", volt, 1e-6, 20, True)),
                      ("dbμv", Unit("dBuV", volt, 1e-6, 20, True)),
                      ("dbi", Unit("dBi", _d(dBi=1), db=True)),
                      ("dbd", Unit("dBd", _d(dBi=1), db=True, offset=2.15)),
                      ("dbc", Unit("dBc", _d(dBc=1), db=True)),
                      ("dbfs", Unit("dBFS", _d(dBFS=1), db=True)),
                      ("dbhz", Unit("dB-Hz", _d(dB_Hz=1), db=True))):
        put(folded, key, unit)
    density = {"dbm": Unit("dBm/Hz", _d(power=1, time=1), 1e-3, 10, True),
               "dbw": Unit("dBW/Hz", _d(power=1, time=1), 1.0, 10, True),
               "dbc": Unit("dBc/Hz", _d(dBc_Hz=1), db=True),
               "db": Unit("dB/Hz", _d(dB_per_Hz=1), db=True)}
    return exact, folded, density


_EXACT, _FOLDED, _DENSITY = _tables()
_RATE_PREFIX = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_CELSIUS = Unit("°C", _TEMP, 1.0, zero=273.15)
_FAHRENHEIT = Unit("°F", _TEMP, 5 / 9, zero=273.15 - 32 * 5 / 9)
_V_PREFIX = {"": 1.0, "k": 1e3, "m": 1e-3, "μ": 1e-6, "u": 1e-6}


def _volt_convention(m: re.Match) -> Unit:
    """Vp-p, Vpk-pk, V peak-to-peak, volts p-p, V rms, V_rms, V pk (the voltage conventions as written)."""
    name = "Vpp" if m[2] else "Vrms" if m[3] else "Vpk"
    return Unit(m[1] + name, _d(voltage=1), _V_PREFIX[m[1]])


_SPECIALS = (   # units that are not a plain quotient of two table entries
    (re.compile(r"([kmμu]?)(?:V|[Vv]olts?)[ \t_]?(?i:(p-p|pp|pk-pk|peak[ \t-]*to[ \t-]*peak)|(rms)|(pk|peak|p))"
                r"(?![\w-])"),
     _volt_convention),
    (re.compile(r"([kKMGT]?)(?:Sa|S)/s(?![\w])"), lambda m: Unit(m[1] + "S/s", _F, _RATE_PREFIX[m[1]])),
    (re.compile(r"([kKMGT])s/s(?![\w])"), lambda m: Unit(m[1] + "S/s", _F, _RATE_PREFIX[m[1]])),
    (re.compile(r"([kKMGT]?)b/s(?![\w])"),
     lambda m: Unit(m[1] + "bit/s", _d(bit=1, time=-1), _RATE_PREFIX[m[1]])),
    (re.compile(r"([kKMGT]?)B/s(?![\w])|([kKMGT]?)Bps(?![\w])"),       # bytes: capital B, exact case
     lambda m: Unit((m[1] or m[2] or "") + "B/s", _d(bit=1, time=-1), 8.0 * _RATE_PREFIX[m[1] or m[2] or ""])),
    (re.compile(r"dB-Hz(?![\w])", re.I), lambda m: Unit("dB-Hz", _d(dB_Hz=1), db=True)),
    (re.compile(r"(?:[°º][ \t]?|(?i:deg(?:rees?)?)[ \t]*)(?:C|(?i:celsius))(?![\w])|℃|(?i:celsius)(?![\w])"),
     lambda m: _CELSIUS),
    (re.compile(r"(?:[°º][ \t]?|(?i:deg(?:rees?)?)[ \t]*)(?:F|(?i:fahrenheit))(?![\w])|℉|(?i:fahrenheit)(?![\w])"),
     lambda m: _FAHRENHEIT),
)
_SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_ATOM = re.compile(r"%|°|[A-Za-zμΩ]+")
_EXP = re.compile(rf"\^\(?([-+]?\d+)\)?|([⁻⁺]?[{_SUPS}]+)")
_SEP = re.compile(r"[ \t]*(/)[ \t]*|[ \t]*([·*])[ \t]*|[ \t]+(per)[ \t]+")
_GAP = re.compile(r"[ \t\u00a0\u2007\u2009\u202f]*(?:-(?=[A-Za-zμΩ°%]))?")
_SPACED_PREFIX = {k: v for k, v in _WORD_PREFIX.items() if k}       # "kilo samples", "Mega Hertz"
_PREFIX_GAP = re.compile(r"[ \t-]+")


def _atom_at(text: str, i: int) -> tuple[tuple[Unit, ...], int, str] | None:
    for pat, make in _SPECIALS:
        m = pat.match(text, i)
        if m:
            units, j = (make(m),), m.end()
            e = _EXP.match(text, j)
            if e and not any(u.db or u.zero for u in units):          # Vrms^2
                n = int(e.group(1) if e.group(1) else e.group(2).translate(_SUPER))
                units, j = tuple(_pow(u, n) for u in units), e.end()
            return units, j, m.group(0)
    m = _ATOM.match(text, i)
    if not m:
        return None
    word = m.group(0)
    units = _EXACT.get(word) or _FOLDED.get(word.casefold())
    j = m.end()
    if not units and word.casefold() in _SPACED_PREFIX:
        g = _PREFIX_GAP.match(text, j)
        nxt = _ATOM.match(text, g.end()) if g else None
        base = _FOLDED.get(nxt.group(0).casefold()) if nxt else None
        if base and not any(u.db or u.zero for u in base):
            f = _SPACED_PREFIX[word.casefold()]
            units = tuple(Unit(f"{word} {u.name}", u.dims, u.scale * f) for u in base)
            word, j = text[m.start():nxt.end()], nxt.end()
    if not units:
        return None
    e = _EXP.match(text, j)
    if e and not any(u.db or u.zero for u in units):
        n = int(e.group(1) if e.group(1) else e.group(2).translate(_SUPER))
        units, j = tuple(_pow(u, n) for u in units), e.end()
    return units, j, word


def _unit_at(text: str, i: int) -> tuple[tuple[Unit, ...], int] | None:
    """The unit expression starting at text[i], as its possible readings and its end."""
    first = _atom_at(text, i)
    if not first:
        return None
    units, j, _ = first
    while True:
        s = _SEP.match(text, j)
        if not s:
            break
        nxt = _atom_at(text, s.end())
        if not nxt:
            break
        more, k, word = nxt
        divide = bool(s.group(1) or s.group(3))
        if len(units) == 1 and units[0].db:          # dBm/Hz, dBc per Hz: a density, or the end
            dens = _DENSITY.get(units[0].name.casefold()) if divide and word.casefold() == "hz" else None
            if dens is None:
                break
            units, j = (dens,), k
            continue
        combos: list[Unit] = []
        for a, b in itertools.product(units, more):
            c = _combine(a, b, -1 if divide else 1)
            if c is not None and c not in combos:
                combos.append(c)
        if not combos:
            break
        units, j = tuple(combos), k
    if j < len(text) and (text[j].isalnum() or text[j] == "_"):
        return None
    return units, j


def parse_unit(text: str | None) -> Unit:
    """A reference unit string ("MS/s", "dBm", "samples/symbol", "" for none) as one Unit."""
    s = normalize(text or "").strip()
    if not s:
        return DIMENSIONLESS
    got = _unit_at(s, 0)
    if not got or got[1] != len(s):
        raise ValueError(f"unknown unit {text!r}")
    return got[0][0]


def unit_readings(text: str) -> tuple[Unit, ...]:
    """Every reading of a unit string (e.g. 'Msps' is a sample rate or a symbol rate)."""
    s = normalize(text).strip()
    got = _unit_at(s, 0) if s else None
    if not got or got[1] != len(s):
        raise ValueError(f"unknown unit {text!r}")
    return got[0]


# --- normalisation and number extraction ---------------------------------------------------------

_LATEX_WRAP = re.compile(r"\\(?:text|textrm|mathrm|mathit|mathbf|textbf|boldsymbol|operatorname|mbox|"
                         r"mathsf|rm)\s*\{([^{}]*)\}")
_LATEX_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_PLAIN_NUM = re.compile(r"\s*-?(?:\d+(?:\.\d+)?|\.\d+)?\s*(?:\\pi|π)?\s*")
_LATEX_SUB = (("\\left", ""), ("\\right", ""), ("\\times", "×"), ("\\cdot", "·"), ("\\approx", "≈"),
              ("\\simeq", "≈"), ("\\sim", "~"), ("\\pm", "±"), ("\\mu", "μ"), ("\\Omega", "Ω"),
              ("\\pi", "π"), ("\\%", "%"), ("\\degree", "°"), ("\\circ", "°"), ("\\,", " "),
              ("\\;", " "), ("\\:", " "), ("\\!", ""), ("\\ ", " "), ("\\quad", " "), ("\\qquad", " "),
              ("\\(", " "), ("\\)", " "), ("\\[", " "), ("\\]", " "), ("$", " "))
_CHAR_SUB = str.maketrans({"\u2212": "-", "\u2010": "-", "\u2011": "-", "\u00b5": "μ", "\u2126": "Ω",
                           "\u22c5": "·", "\u2219": "·"})


def normalize(text: str) -> str:
    """Answer text with LaTeX and Markdown reduced to plain characters the number parser reads."""
    s = text.translate(_CHAR_SUB)
    s = re.sub(r"(?<=[\d}$*])~(?=[A-Za-zμΩ°%\\])", " ", s)  # LaTeX's non-breaking space; "~30" stays "about"
    for _ in range(4):                                  # nested \text{...} and \frac{..}{..}
        new = _LATEX_WRAP.sub(r"\1", s)
        new = _LATEX_FRAC.sub(lambda m: (f"{m[1].strip()}/{m[2].strip()}"
                                         if _PLAIN_NUM.fullmatch(m[1]) and _PLAIN_NUM.fullmatch(m[2])
                                         else f"({m[1]})/({m[2]})"), new)
        if new == s:
            break
        s = new
    for old, new in _LATEX_SUB:
        s = s.replace(old, new)
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    s = re.sub(r"\^\s*°", "°", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s).replace("{", "").replace("}", "")
    s = re.sub(r"μ\s+(?=[A-Za-zΩ])", "μ", s)            # "\mu s" -> "μs"
    s = re.sub(r"(?<![\d)])\*\*|\*\*(?![-+]?\d)", "", s)  # bold, but keep 10**6
    s = re.sub(r"(?<![\w)*])\*(?=[^\s*])|(?<=[^\s*])\*(?![\w(*])", "", s)   # italics, but keep 10**6
    s = re.sub(r"(?<!\w)__|__(?!\w)", "", s).replace("`", "")
    s = re.sub(r"\+\s*/\s*-|\+-(?=\s*[\d.])", "±", s)
    s = re.sub(r"(?:(?<=^)|(?<=[\s(=]))[–—](?=\d)", "-", s, flags=re.M)   # an en dash used as minus
    s = re.sub(r"(?<![A-Za-z])pi(?![A-Za-z])", "π", s)
    return s


_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-+](?=\.?[\dπ]))?"
    r"(?:"
    r"(?P<mant>(?:\d{1,3}(?P<sep>[,_\u00a0\u2007\u2009\u202f])\d{3}(?:(?P=sep)\d{3})*(?!\d)|\d+)"
    r"(?:\.\d+)?|\.\d+)(?!\d)(?!\.\d)"
    r"(?:"
    r"(?P<pi>[ ]?π)"
    r"|[eE](?P<e1>[-+]?\d+)"
    r"|[ ]*[×x·*][ ]*10[ ]*(?:\^|\*\*)[ ]*\(?(?P<e2>[-+]?\d+)\)?"
    rf"|[ ]*[×x·*][ ]*10(?P<e3>[⁻⁺]?[{_SUPS}]+)"
    r"|[ ]*(?:\^|\*\*)[ ]*\(?(?P<p1>[-+]?\d+)\)?"
    rf"|(?P<p2>[⁻⁺]?[{_SUPS}]+)"
    r")?"
    r"|(?P<pi0>π)"
    r")"
    r"(?:[ ]*/[ ]*(?P<den>\d+(?:\.\d+)?)(?![\d.]))?"
)
_MULT = {"k": 1e3, "K": 1e3, "M": 1e6}     # no G: "5G" is a network generation
_WORD_MULT_VALUE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_WORD_MULT = re.compile(r"[ \t ]+(thousand|million|billion)(?![\w])", re.I)
# a word after a number that looks like a unit but is not one ("2500 kSamp/s", "4 taps per symbol")
_UNPARSED = re.compile(r"[A-Za-zμΩ]+(?:[ \t]*/[ \t]*[A-Za-zμΩ]|[ \t]+per[ \t]+[A-Za-z])|[°º][ \t]?[A-Za-z]")
# Words that may follow a bare number without being a unit (compared case-insensitively). Any other
# word right after a bare candidate is a unit the rules do not know ("3.84 Mcps"): the judge decides.
UNIT_STOP_WORDS = frozenset((
    "of", "and", "or", "nor", "to", "for", "in", "at", "by", "with", "from", "per", "on", "into", "over",
    "after", "before", "than", "times", "fold", "x", "is", "are", "was", "were", "be", "been", "being",
    "which", "so", "gives", "giving", "because", "since", "as", "if", "when", "while", "then", "that",
    "this", "these", "those", "it", "its", "the", "a", "an", "each", "every", "but", "not", "no", "has",
    "have", "had", "can", "will", "would", "should", "must", "may", "might", "here", "there", "where",
    "we", "you", "i", "they", "also", "only", "more", "less", "about", "approximately", "exactly",
    "roughly", "total", "overall", "thus", "hence", "therefore", "given", "assuming", "using", "via",
    "due", "respectively", "such", "instead", "dimensionless", "unitless", "complex", "real", "float",
    "floats", "integer", "integers", "qam", "psk", "fsk", "ask", "ary", "filter", "fft",
    "tap", "taps", "stage", "stages", "step", "steps", "item", "items", "bin", "bins", "point", "points",
    "coefficient", "coefficients", "channel", "channels", "entry", "entries", "value", "values",
    "output", "outputs", "input", "inputs", "element", "elements", "frame", "frames", "packet", "packets",
    "block", "blocks", "port", "ports", "stream", "streams", "level", "levels", "state", "states",
    "carrier", "carriers", "subcarrier", "subcarriers", "tone", "tones", "iteration", "iterations",
    "row", "rows", "column", "columns", "digit", "digits", "decimal", "places",
    # GNU Radio item types and the behaviour questions' nouns ("34 short items", "34 vectors")
    "short", "shorts", "int", "ints", "char", "chars", "vector", "vectors", "message", "messages", "tag",
    "tags", "pdu", "pdus", "zero", "zeros", "one", "ones", "chunk", "chunks",
))
# units that count items against a reference with no unit (P3): "34 bytes" is 34 byte items
ITEM_COUNT_UNITS = ("bit", "bits", "byte", "bytes", "symbol", "symbols")
_WORD_AFTER = re.compile(r"[A-Za-zμΩ]+(?![\w])")
# words that make a bare number a plain ratio (decision 7): "1000 times", "1000x", "a 1000 ratio";
# not before a number or a word, so "2 x 15 kHz" and "2 × B" stay expressions
RATIO_WORDS = ("times", "x", "×", "ratio")
_RATIO_WORD = re.compile(r"(?i:times|ratio|x|×)(?![\w])(?![ \t]*(?:[-+]?[\d(π]|\.\d|"
                         r"(?!(?i:more|less|larger|higher|greater|bigger|smaller|lower)\b)[A-Za-zμΩ]))")
# a number before ":" is a label only after a label word or a capitalised word, or opening a line
_LABEL_WORD = re.compile(r"(?i:\b(?:step|part|case|example|method|option|question|problem|item|stage|figure|"
                         r"fig|table|eq|equation|note|scenario|q))\.?[ \t]*$|\b[A-Z][a-z]*[ \t]*$")


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str                       # the unit as written, "" for a bare number
    units: tuple[Unit, ...]         # its readings (empty for a bare number)
    text: str                       # the quantity as it reads in the normalised answer
    span: tuple[int, int] = (0, 0)  # where it is in the normalised text
    base: bool = False              # bare, with an attached multiplier (10k, 2.5M): in SI base units
    kilo_or_kelvin: bool = False    # an attached K with no unit after it (290K, 48K)
    unparsed: str = ""              # bare, followed by a unit-like word the parser does not know
    unknown: str = ""               # bare, followed by a word that is no unit and not in UNIT_STOP_WORDS
    ratio_word: str = ""            # bare, followed by a ratio word: "1000 times", "1000x" (P4)


def _number_value(m: re.Match) -> float | None:
    try:
        if m["pi0"]:
            v = math.pi
        else:
            v = float(re.sub(r"[,_\u00a0\u2007\u2009\u202f]", "", m["mant"]))
            if m["pi"]:
                v *= math.pi
            elif m["e1"]:
                v *= 10.0 ** int(m["e1"])
            elif m["e2"] or m["e3"]:
                v *= 10.0 ** int((m["e2"] or m["e3"]).translate(_SUPER))
            elif m["p1"] or m["p2"]:
                v = v ** int((m["p1"] or m["p2"]).translate(_SUPER))
        if m["den"]:
            den = float(m["den"])
            if den == 0:
                return None
            v /= den
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    if m["sign"] == "-":
        v = -v
    return float(v) if math.isfinite(v) else None


def _is_marker_number(s: str, m: re.Match, at_line_start: bool = True) -> bool:
    """List markers ("1. text", "2) text" at the start of a line) and labels ("Step 2:") are not
    quantities. A number that ends its line ("Answer: 4.") is a quantity, and so is one before a colon
    in a sentence ("The factor is 4: 48 kHz in"). `at_line_start` says whether the start of `s` is
    the start of a line (False for text cut after "Answer:")."""
    line_start = s.rfind("\n", 0, m.start()) + 1
    rest = s[m.end():]
    opens_line = (line_start > 0 or at_line_start) and s[line_start:m.start()].strip(" \t*-#>•") == ""
    if (opens_line and not m["sign"] and m["mant"] and m["mant"].isdigit()
            and re.match(r"[.)][ \t]+\S", rest)):
        return True
    if not re.match(r":(?!\d)", rest):
        return False
    return opens_line or bool(_LABEL_WORD.search(s[line_start:m.start()]))


def extract_quantities(text: str, at_line_start: bool = True) -> list[Quantity]:
    """Every number in the text with its unit, in order of appearance."""
    s = normalize(text)
    out: list[Quantity] = []
    pos = 0
    while True:
        m = _NUMBER.search(s, pos)
        if not m:
            break
        pos = max(m.end(), m.start() + 1)
        if _is_marker_number(s, m, at_line_start):
            continue
        value = _number_value(m)
        if value is None:
            continue
        j, units, unit_text, base, kilo_or_kelvin, unparsed, unknown, ratio = m.end(), (), "", False, False, "", "", ""
        run = _ATOM.match(s, j)
        word = _WORD_MULT.match(s, j)
        if word:                                                 # 1.5 million samples per second
            value *= _WORD_MULT_VALUE[word[1].casefold()]
            j = word.end()
            g = _GAP.match(s, j)
            after = _unit_at(s, g.end()) if g.end() > j else None
            if after:
                units, j = after
                unit_text = s[g.end():j]
            else:
                base = True
        elif run and run.group(0) in _MULT:                     # attached multiplier: 10k, 2.5M samples/s
            g = _GAP.match(s, run.end())
            after = _unit_at(s, g.end()) if g.end() > run.end() else None
            if after:
                value *= _MULT[run.group(0)]
                units, j = after
                unit_text = s[g.end():j]
            elif run.group(0) == "K":                            # 290K: kelvin, or 48K = 48 000
                units, j, unit_text, kilo_or_kelvin = (_KELVIN,), run.end(), "K", True
            else:                                                # 10k, 2.5M: SI base units
                value *= _MULT[run.group(0)]
                j, base = run.end(), True
        else:
            g = _GAP.match(s, j)
            got = _unit_at(s, g.end())
            if got:
                units, end = got
                unit_text, j = s[g.end():end], end
            else:
                u = _UNPARSED.match(s, g.end())
                if u:
                    unparsed = u.group(0)
        if not units and not unparsed:                           # bare, or a multiplier in SI base units
            g = _GAP.match(s, j)
            rw = _RATIO_WORD.match(s, g.end())
            w = _WORD_AFTER.match(s, g.end()) if g.end() > j else None
            if rw and (g.end() > j or rw.group(0) in "xX×"):        # "1000 times", "1000x"
                ratio, j = rw.group(0), rw.end()
            elif w and w.group(0).casefold() not in UNIT_STOP_WORDS:
                unknown = w.group(0)
        if not units and j < len(s) and (s[j].isalnum() or s[j] == "_"):
            if not (s[j] in "x×" and (j + 1 >= len(s) or not s[j + 1].isalnum())):
                continue                                         # glued to a word: 8PSK, 0.5z
        if s[:m.start()].rstrip().endswith("±"):                  # an uncertainty, not an answer
            if out and units and not out[-1].units and not out[-1].base:
                prev = out[-1]
                out[-1] = Quantity(prev.value, unit_text, units, f"{prev.text} ±{s[m.start():j]}",
                                   (prev.span[0], j), kilo_or_kelvin=kilo_or_kelvin)
            pos = max(pos, j)
            continue
        out.append(Quantity(value, unit_text.strip(), units, re.sub(r"\s+", " ", s[m.start():j].strip()),
                            (m.start(), j), base, kilo_or_kelvin, unparsed, unknown, ratio))
        pos = max(pos, j)
    return out


# --- reasoning and the final answer --------------------------------------------------------------

_THINK_BLOCKS = tuple(re.compile(p, re.S | re.I) for p in (
    r"<think(?:ing)?>.*?</think(?:ing)?>", r"<reasoning>.*?</reasoning>", r"\[THINK\].*?\[/THINK\]",
    r"<\|START_THINKING\|>.*?<\|END_THINKING\|>",
    r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|(?=<\|start\|>)|(?=<\|channel\|>))"))
_CLOSERS = re.compile(r"</think(?:ing)?>|</reasoning>|\[/THINK\]|<\|END_THINKING\|>", re.I)
_OPENERS = re.compile(r"<think(?:ing)?>|<reasoning>|\[THINK\]|<\|START_THINKING\|>|"
                      r"<\|channel\|>analysis", re.I)
_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>", re.I)


def strip_reasoning(answer: str) -> str:
    """Remove reasoning text (eval plan §7.1).

    Closed blocks go (Tier F's rule, plus <thinking>, <reasoning>, [THINK], Cohere and harmony
    channels). A closing tag with no opening one means the chat template opened the block in the
    prompt, so only the text after it is the answer. An opening tag with no closing one means the
    answer was cut off inside its reasoning, so nothing from it on counts.
    """
    text = _strip_think(answer or "")
    for pat in _THINK_BLOCKS:
        text = pat.sub("", text)
    finals = list(_HARMONY_FINAL.finditer(text))
    if finals:
        text = text[finals[-1].end():]
    closers = list(_CLOSERS.finditer(text))
    if closers:
        text = text[closers[-1].end():]
    opener = _OPENERS.search(text)
    if opener:
        text = text[:opener.start()]
    return re.sub(r"<\|[A-Za-z_]+\|>", " ", text).strip()


_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
# The marker never takes the line break after it, so a list marker on the next line is seen as one.
# "final answer" is a marker only before ":", "=" or "is", alone on its line (a heading) or as a bold
# span; in prose ("to get the final answer, multiply ...") it is not one.
_MARKER = re.compile(r"\bfinal\s+answer\b[ \t*_]*(?:is\b[ \t*_]*[:=]?|[:=])|"
                     r"^[ \t>#*_-]*final\s+answer\b(?:[ \t]*(?:\*\*|__))?[ \t]*(?=\n|$)|"
                     r"(?:\*\*|__)[ \t]*final\s+answer[ \t]*(?:\*\*|__)|"
                     r"\banswer\s+is\b[ \t*_]*:?|\banswer[ \t*_]*[:=]", re.I | re.M)
# A prose marker ("the answer is ...") counts only where no explicit one ("Final answer:", "Answer:", a
# heading or a bold span) exists, so a note after the explicit answer ("this answer is for a 48 kHz
# input") cannot replace it (round-2 finding R2).
_PROSE_MARKER = re.compile(r"answer\s+is\b[ \t*_]*:?", re.I)
_CLAUSE_END = re.compile(r"(?<=\S)[.!?](?=\s|$)|;|:(?=\s)")
_PAREN = re.compile(r"\([^()]*\)")
_LEAD = " \t\r\n*_:=#>"
# label segments: sentences, clauses, lines, comma items and "and" items
_SEGMENT_END = re.compile(r"(?<=\S)[.!?](?=\s|$)|;|\n|,(?=\s)|(?<=\s)and(?=\s)", re.I)


@dataclass(frozen=True)
class FinalAnswer:
    rule: str                       # boxed | marker | last, with "+label" (or "label") when a label scoped it
    regions: tuple[str, ...]        # narrow to wide
    mention: str                    # the text in which options are looked for
    clause: int = 0                 # index of the first clause in regions (before it: the core)
    line_start: bool = True         # whether the regions begin at the start of a line
    tails: tuple[str, ...] = ()     # per region, the answer text after it (where revisions are looked for)
    kind: str = ""                  # "scope" | "anchor" for a label's place (_label_scope, _label_anchor)

    def tail(self, i: int) -> str:
        return self.tails[i] if 0 <= i < len(self.tails) else ""


def _last_boxed(text: str) -> tuple[str, str] | None:
    """The last \\boxed{...} body and the text after it."""
    found = list(_BOXED.finditer(text))
    if not found:
        return None
    i = j = found[-1].end()
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return (text[i:j - 1], text[j:]) if depth == 0 else (text[i:], "")


def _first_clause(line: str, line_start: bool) -> str:
    """The line up to its first sentence end, ";" or ": ". A list marker that opens a line ("1. ")
    does not end a clause."""
    for m in _CLAUSE_END.finditer(line):
        head = line[:m.start()]
        if (line_start and m.group() == "." and re.fullmatch(r"\s*(?:\d{1,2}|[A-Za-z])", head)
                and line[m.end():].strip()):
            continue
        return head
    return line


def _core(clause: str) -> str:
    """The clause without parenthetical remarks ("complex (the default)" -> "complex")."""
    s = re.sub(r"\\[()]", " ", clause)
    for _ in range(4):
        new = _PAREN.sub(" ", s)
        if new == s:
            break
        s = new
    return s


def _sections(after: str, rule: str, beyond: str = "", kind: str = "") -> FinalAnswer:
    """Regions after an anchor (a marker or a label): core, clause, line, paragraph, the rest; each
    with the text after it (and `beyond`, text past the anchor's area)."""
    rest = after.lstrip(_LEAD)
    line_start = "\n" in after[:len(after) - len(rest)]
    paragraph = re.split(r"\n[ \t]*\n", rest, maxsplit=1)[0]
    line = rest.split("\n", 1)[0]
    clause = _first_clause(line, line_start)
    regions: list[str] = []
    tails: list[str] = []
    clause_index = 0
    # clause, line and paragraph are prefixes of rest, so what follows a region is rest[len(raw):]
    for name, r, raw in (("core", _core(clause), clause), ("clause", clause, clause), ("line", line, line),
                         ("paragraph", paragraph, paragraph), ("rest", rest, rest)):
        r = r.strip()
        if name == "clause":
            clause_index = len(regions) - 1 if regions and regions[-1] == r else len(regions)
        if r and (not regions or regions[-1] != r):
            regions.append(r)
            tails.append(rest[len(raw):] + beyond)
    if not regions:
        return FinalAnswer(rule, ("",), "", 0, line_start, (beyond,), kind)
    return FinalAnswer(rule, tuple(regions), paragraph.strip(), max(0, clause_index), line_start, tuple(tails),
                       kind)


def _without_labels(text: str, stops: tuple[str, ...]) -> str:
    """The segments of `text` that mention no other part's label (they belong to that part)."""
    if not stops or not text:
        return text
    mentions = _label_mentions(text, (), stops)
    cuts = sorted({0, len(text)} | {m.end() for m in _SEGMENT_END.finditer(text)})
    return "".join(text[a:b] for a, b in itertools.pairwise(cuts)
                   if not any(a <= s < b for s, _, _ in mentions))


def _label_mentions(text: str, own: tuple[str, ...], others: tuple[str, ...]) -> list[tuple[int, int, bool]]:
    """Mentions of labels as (start, end, is_own), longest label first, never overlapping. Labels of
    up to 3 characters ("B", "Ts") match their case."""
    taken: list[tuple[int, int, bool]] = []
    labels = sorted({(lab, True) for lab in own if lab} | {(lab, False) for lab in others if lab},
                    key=lambda x: (-len(x[0]), x[0], not x[1]))
    for lab, is_own in labels:
        flags = re.I if len(lab) > 3 else 0
        for m in re.finditer(r"(?<![\w])" + re.escape(lab) + r"(?![\w])", text, flags):
            if not any(m.start() < e and s < m.end() for s, e, _ in taken):
                taken.append((m.start(), m.end(), is_own))
    return sorted(taken)


def _label_scope(area: str, labels: tuple[str, ...], stops: tuple[str, ...], rule: str,
                 beyond: str = "") -> FinalAnswer | None:
    """The segments of `area` that mention this label and no other part's (narrow), and those with the
    unlabelled segments that follow them (wide). The tail is every later segment of the area that is
    neither this label's nor another part's, then `beyond`."""
    mentions = _label_mentions(area, labels, stops)
    if not any(own for _, _, own in mentions):
        return None
    cuts = sorted({0, len(area)} | {m.end() for m in _SEGMENT_END.finditer(area)})
    segs = [(a, b) for a, b in itertools.pairwise(cuts) if area[a:b].strip()]

    def owners(seg: tuple[int, int]) -> set[bool]:
        return {own for s, _, own in mentions if seg[0] <= s < seg[1]}

    narrow: list[str] = []
    wide: list[str] = []
    first = None
    i = 0
    while i < len(segs):
        if owners(segs[i]) == {True}:
            first = i if first is None else first
            j = i + 1
            while j < len(segs) and not owners(segs[j]):
                j += 1
            narrow.append(area[segs[i][0]:segs[i][1]].strip())
            wide.append(area[segs[i][0]:segs[j - 1][1]].strip())
            i = j
        else:
            i += 1
    if not narrow:
        return None
    regions = tuple(dict.fromkeys(("\n".join(narrow), "\n".join(wide))))
    tail = " ".join(area[a:b] for a, b in segs[first + 1:] if not owners((a, b))) + beyond
    return FinalAnswer(rule, regions, regions[-1], 0, True, (tail,) * len(regions), "scope")


def _label_anchor(area: str, labels: tuple[str, ...], stops: tuple[str, ...], rule: str,
                  beyond: str = "") -> FinalAnswer | None:
    """The text after the label's last mention that has a number nearby, cut at another part's label.
    With other parts, a mention that a number precedes in its segment ("50 MS/s (real),25 MS/s") is
    written value first, so the number after it is not its value."""
    mentions = _label_mentions(area, labels, stops)
    seg_starts = [0] + [m.end() for m in _SEGMENT_END.finditer(area)]
    for start, end, own in reversed(mentions):
        if not own:
            continue
        if stops:
            seg = max(s for s in seg_starts if s <= start)
            prev = max([e for _, e, _ in mentions if e <= start] + [seg])
            if extract_quantities(area[prev:start]):
                continue
        after = area[end:]
        cut = next((s - end for s, _, o in mentions if not o and s >= end), None)
        rest = ""
        if cut is not None:
            after, rest = after[:cut], _without_labels(after[cut:], stops)
        fa = _sections(after, rule, rest + beyond, "anchor")
        if any(extract_quantities(r, fa.line_start) for r in fa.regions[:fa.clause + 2]):
            return fa
    return None


def _final_answers(text: str, labels: tuple[str, ...] = (), stops: tuple[str, ...] = ()) -> list[FinalAnswer]:
    """Where the final stated answer may be, most specific first (the rule in the module docstring)."""
    boxed = _last_boxed(text)
    markers = list(_MARKER.finditer(text))
    markers = [m for m in markers if not _PROSE_MARKER.fullmatch(m.group())] or markers
    if boxed is not None:
        body, beyond = boxed[0].strip(), boxed[1]
        base, area, rule = FinalAnswer("boxed", (body,), body, 0, False, (beyond,)), body, "boxed+label"
    elif markers:
        after = text[markers[-1].end():]
        base = _sections(after, "marker")
        lead = after.lstrip(_LEAD)
        area = re.split(r"\n[ \t]*\n", lead, maxsplit=1)[0]
        beyond, rule = lead[len(area):], "marker+label"
    else:
        base, area, beyond, rule = FinalAnswer("last", (text.strip(),), text.strip(), 0, True, ("",)), text, "", "label"
    out: list[FinalAnswer] = []
    if labels:
        area = normalize(area)              # segments and labels are found in plain text, not LaTeX
        beyond = _without_labels(normalize(beyond), stops)
        for fa in (_label_scope(area, labels, stops, rule, beyond),
                   _label_anchor(area, labels, stops, rule, beyond)):
            if fa is not None:
                out.append(fa)
    return out + [base]


def final_answer(text: str, labels: tuple[str, ...] = (), stops: tuple[str, ...] = ()) -> FinalAnswer:
    """Where the final stated answer is: the first place in _final_answers that holds a number."""
    found = _final_answers(text, labels, stops)
    for fa in found:
        if any(extract_quantities(r, fa.line_start) for r in fa.regions):
            return fa
    return found[-1]


# --- references ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericRef:
    value: float
    unit: str
    unit_obj: Unit
    tolerance: float
    abs_tolerance: float | None
    unit_required: bool
    accept_units: frozenset[str]
    labels: tuple[str, ...]
    db_tolerance: float = DB_TOLERANCE      # decibel references: allowance in dB
    waveform: str | None = None             # "sine": an RMS voltage converts to peak (factor √2)
    ratio_kind: str = "power"               # a linear ratio read for a decibel value: 10 or 20 log10 (P4, P8)
    form_required: bool = False             # the question fixes the form: the other form scores 0 (P8)
    quantity: str | None = None             # "gain": a loss word negates a positive value (P8)

    @property
    def dimensionless(self) -> bool:
        return not self.unit_obj.dims and not self.unit_obj.db

    @property
    def base_unit(self) -> Unit:
        """The SI base unit of the reference's dimension (Hz for kHz, W for dBm, K for °C, 1 for %)."""
        return Unit("SI", self.unit_obj.dims)


@dataclass(frozen=True)
class SetRef:
    items: frozenset[str]
    names: tuple[tuple[str, str], ...]     # normalised spelling -> canonical option
    closed: bool = False                   # a universe was given: every valid option is known

    @property
    def lookup(self) -> dict[str, str]:
        return dict(self.names)


@dataclass(frozen=True)
class ExactRef:
    target: str
    accepted: frozenset[str]
    universe: frozenset[str]
    aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NoRuleRef:
    """A rubric with no rule target (an open-ended item): the judge decides."""


Reference = NumericRef | SetRef | ExactRef | NoRuleRef | tuple
_RUBRIC_KEYS = ("numeric", "required_points", "points", "disqualifying_errors", "numeric_tolerance")


def _norm_str(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("`", "").replace("*", "")
    s = re.sub(r"\s+", " ", s).strip(" \t\r\n.,;:!?\"'()[]{}<>")
    return s.casefold()


def _alias_pairs(aliases: Any) -> list[tuple[str, str]]:
    """Aliases as {alias: option}, {option: [alias, ...]} or [[alias, option], ...]."""
    pairs: list[tuple[str, str]] = []
    if isinstance(aliases, dict):
        for k, v in aliases.items():
            if isinstance(v, (list, tuple, set, frozenset)):
                pairs += [(a, k) for a in v]
            else:
                pairs.append((k, v))
    elif aliases:
        pairs += [(a, b) for a, b in aliases]
    return [(_norm_str(a), _norm_str(b)) for a, b in pairs]


def _numeric_ref(d: dict) -> NumericRef:
    if d.get("value") is None:
        raise ValueError(f"a numeric reference needs a value: {d!r}")
    value = float(d["value"])
    if not math.isfinite(value):
        raise ValueError(f"a numeric reference needs a finite value: {d!r}")
    unit = str(d.get("unit") or "").strip()
    unit_obj = parse_unit(unit)
    given = d.get("tolerance")
    tol = DEFAULT_TOLERANCE if given is None else float(given)
    abs_tol = None if d.get("abs_tolerance") is None else float(d["abs_tolerance"])
    tol_db = None if d.get("tolerance_db") is None else float(d["tolerance_db"])
    if tol < 0 or any(t is not None and t < 0 for t in (abs_tol, tol_db)):
        raise ValueError("tolerances must be >= 0")
    db_tol = DB_TOLERANCE
    if unit_obj.db:
        # docs/guidelines/rubric_writing.md: a rubric gives a decibel tolerance in abs_tolerance (the
        # calculation key's tolerance_db goes there); next to it calc.py's relative `tolerance` is ignored
        if abs_tol is not None and tol_db is not None and abs_tol != tol_db:
            raise ValueError(f"abs_tolerance and tolerance_db differ: {d!r}")
        in_db = abs_tol if abs_tol is not None else tol_db
        if in_db is None and given is not None:
            raise ValueError(f"a decibel reference takes its tolerance in dB (abs_tolerance; tolerance_db in "
                             f"the calculation key); a relative tolerance alone is ambiguous: {d!r}")
        if in_db is not None:
            db_tol = abs_tol = in_db
    elif tol_db is not None:
        raise ValueError(f"tolerance_db is for decibel units only: {d!r}")
    waveform = d.get("waveform")
    if waveform is not None and waveform not in WAVEFORMS:
        raise ValueError(f"waveform must be one of {list(WAVEFORMS)}: {d!r}")
    kind = d.get("ratio_kind")
    if kind is not None and kind not in RATIO_KINDS:
        raise ValueError(f"ratio_kind must be one of {list(RATIO_KINDS)}: {d!r}")
    if unit_obj.log:                    # dBm, dBW: power; dBV, dBmV, dBuV: amplitude, fixed by the unit
        fixed = "power" if unit_obj.log == 10 else "amplitude"
        if kind is not None and kind != fixed:
            raise ValueError(f"{unit} is a {fixed} level; ratio_kind {kind!r} contradicts it: {d!r}")
        kind = fixed
    quantity = d.get("quantity")
    if quantity is not None and quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {list(QUANTITIES)}: {d!r}")
    label = d.get("label")
    labels = (label,) if isinstance(label, str) else tuple(label or ())
    return NumericRef(value, unit, unit_obj, tol, abs_tol, bool(d.get("unit_required", False)),
                      frozenset(_norm_str(u) for u in d.get("accept_units") or ()), labels, db_tol, waveform,
                      kind or "power", bool(d.get("form_required", False)), quantity)


def _set_ref(items: Any, universe: Any = None, aliases: Any = None) -> SetRef:
    if isinstance(items, str):
        items = [items]
    canon = frozenset(_norm_str(x) for x in items or ())
    if not canon or "" in canon:
        raise ValueError("a set reference needs non-empty options")
    names = {u: u for u in sorted(canon | {_norm_str(x) for x in universe or ()})}
    for alias, option in _alias_pairs(aliases):
        names.setdefault(option, option)
        names[alias] = option
    return SetRef(canon, tuple(sorted(names.items())), bool(universe))


def parse_reference(reference: Any) -> Reference:
    """A rubric numeric field, a set target or a string target (forms in the module docstring)."""
    if isinstance(reference, NumericRef | SetRef | ExactRef | NoRuleRef):
        return reference
    if isinstance(reference, str):
        reference = {"exact": reference}
    if isinstance(reference, (set, frozenset)):
        return _set_ref(sorted(map(str, reference)))
    if isinstance(reference, (list, tuple)):
        if reference and all(isinstance(r, dict) for r in reference):
            parts = tuple(parse_reference(r) for r in reference)
            if not all(isinstance(p, NumericRef) for p in parts):
                raise ValueError("a list of parts must hold numeric references")
            return parts if len(parts) > 1 else parts[0]
        return _set_ref(list(reference))
    if not isinstance(reference, dict):
        raise ValueError(f"unsupported reference: {reference!r}")
    kind = reference.get("kind")
    target_keys = {"value", "set", "exact", "items", "target"} & set(reference)
    if "numeric" in reference and not target_keys and kind is None:        # a rubric record
        num = reference["numeric"]
        if num is None or num == []:
            return NoRuleRef()
        if isinstance(num, (dict, list)):
            return parse_reference(num)
        raise ValueError(f"unsupported rubric numeric field: {num!r}")
    if "value" in reference or kind == "numeric":
        return _numeric_ref(reference)
    if "set" in reference or kind == "set":
        return _set_ref(reference.get("set", reference.get("items")), reference.get("universe"),
                        reference.get("aliases"))
    if "exact" in reference or kind == "exact":
        target = _norm_str(reference.get("exact", reference.get("target")))
        if not target:
            raise ValueError("an exact reference needs a target")
        aliases = reference.get("aliases") or []
        if isinstance(aliases, (list, tuple)) and all(isinstance(a, str) for a in aliases):
            pairs = [(_norm_str(a), target) for a in aliases]
        else:
            pairs = _alias_pairs(aliases)
        accepted = frozenset({target} | {a for a, o in pairs if o == target})
        universe = frozenset(_norm_str(u) for u in reference.get("universe") or ())
        return ExactRef(target, accepted, universe, tuple(sorted(pairs)))
    if not target_keys and kind is None and any(k in reference for k in _RUBRIC_KEYS):
        return NoRuleRef()                                  # a rubric without a rule target
    raise ValueError(f"unsupported reference: {reference!r}")


# --- results -------------------------------------------------------------------------------------


def _clip(text: str) -> str:
    return text if len(text) <= _CLIP else text[:_CLIP] + " ..."


def _decided(method: str, credit: float, **details: Any) -> dict[str, Any]:
    return {"method": method, "credit": round(float(credit), 6), "details": details}


def _judge(reason: str, rule_method: str, fa: FinalAnswer | None = None, region: str | None = None,
           **details: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"reason": reason, "rule_method": rule_method}
    if fa is not None:
        d["rule"] = fa.rule
        d["final_answer"] = _clip(region if region is not None else fa.regions[-1])
    d.update(details)
    return {"method": "judge", "credit": None, "details": d}


def rule_only_credit(result: dict[str, Any]) -> float:
    """Credit on a rule-only check (dev-split screening; proposal §5.1), the aggregation
    judge_prompt.explain_score applies to the numbers the rules decided: a judge result counts 0
    (P10); for parts, a unit or order-of-magnitude error in any decided part gives 0 (P9), otherwise
    the mean with every undecided part at 0."""
    parts = result.get("details", {}).get("parts")
    if parts:
        if _part_zero(parts):
            return 0.0
        return sum(rule_only_credit(p) for p in parts) / len(parts)
    if result.get("method") != "judge":
        return float(result.get("credit") or 0.0)
    return 0.0


def rule_only_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The figures a rule-only check reports together (P10): the deferral rate, the mean rule-only
    credit with deferrals at 0 (a lower bound) and the mean credit over the decided items (accuracy
    where the rules decide; it favours answers whose form the rules read). Rows without a method
    (a malformed reference) are left out."""
    rows = [r for r in results if r.get("method") is not None]
    decided = [r for r in rows if r["method"] != "judge"]
    n = len(rows)
    return {"n": n, "decided": len(decided), "deferred": n - len(decided),
            "deferral_rate": round((n - len(decided)) / n, 6) if n else None,
            "rule_only_mean": round(sum(rule_only_credit(r) for r in rows) / n, 6) if n else None,
            "decided_mean": (round(sum(rule_only_credit(r) for r in decided) / len(decided), 6)
                             if decided else None)}


# --- numeric: credit -----------------------------------------------------------------------------


def _tier(error: float, allowance: float, extra: float, noise: float) -> tuple[float, str, int | None]:
    """Grade an error: tier 1 ends at the allowance, every later tier at min(multiple x allowance,
    allowance + extra). `noise` absorbs float error from unit conversion."""
    for k, (mult, credit) in enumerate(CREDIT_TIERS, start=1):
        bound = mult * allowance if k == 1 else min(mult * allowance, allowance + extra)
        if error <= bound * (1 + 1e-9) + noise:
            return credit, ("within_tolerance" if credit >= 1.0 else "partial_credit"), k
    return 0.0, "outside_tolerance", None


def _to_si(x: float, u: Unit) -> float:
    try:
        return u.scale * 10.0 ** (x / u.log) if u.log else x * u.scale + u.zero
    except OverflowError:
        return math.inf


def _credit_linear(a: float, ref: NumericRef) -> dict[str, Any]:
    """Credit for a, already in the reference unit: relative error, sign, order of magnitude."""
    r = ref.value
    if not math.isfinite(a):
        return {"credit": 0.0, "reason": "order_of_magnitude"}
    error = abs(a - r)
    allowance = max(ref.tolerance * abs(r), ref.abs_tolerance or 0.0)
    out: dict[str, Any] = {"compared": a, "error": error, "allowance": allowance,
                           "basis": "relative" if r else "absolute"}
    if r != 0:
        out["relative_error"] = error / abs(r)
        if not (ref.abs_tolerance is not None and error <= ref.abs_tolerance):
            if a * r < 0:
                return {**out, "credit": 0.0, "reason": "sign_mismatch"}
            ratio = a / r
            if ratio >= MAGNITUDE_RATIO or ratio <= 1 / MAGNITUDE_RATIO:
                return {**out, "ratio": ratio, "credit": 0.0, "reason": "order_of_magnitude"}
    # the half band and the float noise are relative to the reference: no absolute floor
    credit, reason, tier = _tier(error, allowance, HALF_BAND_EXTRA * abs(r), 1e-9 * abs(r))
    return {**out, "credit": credit, "reason": reason, "tier": tier}


def _credit_db(a: float, ref: NumericRef) -> dict[str, Any]:
    """Credit for a, already in the reference's decibel unit: absolute error in dB."""
    r, ru = ref.value, ref.unit_obj
    if not math.isfinite(a):
        return {"credit": 0.0, "reason": "order_of_magnitude"}
    error, allowance = abs(a - r), ref.db_tolerance
    out: dict[str, Any] = {"compared": a, "error": error, "allowance": allowance, "basis": "absolute_db"}
    exempt = ref.abs_tolerance is not None and error <= ref.abs_tolerance
    log = ru.log or 10
    if not exempt and error >= log * math.log10(MAGNITUDE_RATIO):
        return {**out, "credit": 0.0, "reason": "order_of_magnitude"}
    credit, reason, tier = _tier(error, allowance, log * math.log10(1 + HALF_BAND_EXTRA),
                                 1e-9 * max(abs(r), 1.0))
    return {**out, "credit": credit, "reason": reason, "tier": tier}


def _compare(x: float, u: Unit | None, ref: NumericRef) -> dict[str, Any]:
    """Credit for the value x in unit u (None: already in the reference unit). u has the
    reference's dimension."""
    ru = ref.unit_obj
    if not ru.db:
        return _credit_linear(x if u is None else (_to_si(x, u) - ru.zero) / ru.scale, ref)
    if u is None:
        return _credit_db(x, ref)
    if u.db and not u.log and not ru.log:           # relative dB units: dBd -> dBi
        return _credit_db(x + u.offset - ru.offset, ref)
    v = _to_si(x, u)                                # an absolute log unit (dBm) or its linear unit (W)
    if not v > 0:
        return {"credit": 0.0, "reason": "not_positive_for_db"}
    return _credit_db(ru.log * math.log10(v / ru.scale), ref)


RATIO_KINDS = ("power", "amplitude")
QUANTITIES = ("gain",)


def _ratio_log(ref: NumericRef) -> int:
    """10 or 20: the dB of a linear ratio for this reference (its unit's own for dBm, dBV, ...;
    ratio_kind for a relative decibel unit or a plain ratio)."""
    return ref.unit_obj.log or (20 if ref.ratio_kind == "amplitude" else 10)


def _ratio_to_db(x: float, ref: NumericRef) -> dict[str, Any]:
    """A linear ratio x graded against a decibel reference: to the unit's reference level for an
    absolute unit (1 mW for dBm, 1 V for dBV), of kind ratio_kind for a relative one (P4, P8)."""
    if not x > 0:
        return {"credit": 0.0, "reason": "not_positive_for_db"}
    return _credit_db(_ratio_log(ref) * math.log10(x), ref)


def _converted(g: dict[str, Any], reason: str, ref: NumericRef) -> dict[str, Any]:
    """A value read in the other form (P4, P8): named by `reason` when it earns credit; with
    form_required it scores 0 as wrong_form instead (and keeps its own reason when it earns none)."""
    g = {"class": "compatible", "reading": "converted", **g}
    if g["credit"] > 0:
        if ref.form_required:
            return {**g, "converted_credit": g["credit"], "credit": 0.0, "reason": "wrong_form"}
        return {**g, "graded_reason": g["reason"], "reason": reason}
    return g


def _grade_bare(q: Quantity, ref: NumericRef) -> dict[str, Any]:
    """P4: a bare number against a decibel reference is a linear ratio; against a dimensionless one
    it is read as is; against any other reference it is read in the reference unit (in SI base units
    with an attached multiplier: 10k, 2.5M), with no other reading. A ratio word ("1000 times",
    "1000x") makes it a plain ratio with a unit, against a dimensionless or decibel reference."""
    ru = ref.unit_obj
    ratio_word = bool(q.ratio_word) and (ref.dimensionless or ru.db)
    cls = "compatible" if ref.dimensionless else "bare"
    if cls == "bare" and ref.unit_required and not ratio_word:
        return {"class": cls, "credit": 0.0, "reason": "unit_missing"}
    if ru.db:           # still a bare number (a ratio word makes it a ratio with a unit): it is the
        # candidate only where no value has the reference's form, and the line around it is checked
        g = _converted({"unit_assumed": "ratio", **_ratio_to_db(q.value, ref)}, "converted_ratio", ref)
        return {**g, "class": "compatible" if ratio_word else cls}
    if q.base:          # 10k, 2.5M: the multiplier implies the SI base unit
        return {"class": cls, "unit_assumed": "SI base", **_compare(q.value, ref.base_unit, ref)}
    return {"class": cls, "unit_assumed": "reference" if cls == "bare" else None, **_compare(q.value, None, ref)}


# Voltage conventions, in volts per volt of peak: Vpp = 2 Vpk exactly; Vrms = Vpk / √2 for a sine only
# (a reference's `waveform`). dBV, dBmV and dBuV levels are RMS. A plain V names no convention and is
# compared as written.
WAVEFORMS = ("sine",)
_V_CONVENTIONS = (("Vpp", "pp", 2.0), ("Vpk", "pk", 1.0), ("Vp", "pk", 1.0), ("Vrms", "rms", math.sqrt(0.5)))
# relative decibel units a plain dB (dB/Hz) value is read in (round-2 finding R9)
_RELATIVE_DB = ("dBc", "dBc/Hz", "dBi", "dBd", "dBFS")


def _v_convention(u: Unit) -> tuple[str, float] | None:
    if u.dims != _d(voltage=1):
        return None
    if u.log:                                       # dBV, dBmV, dBuV: referenced to 1 V (mV, μV) RMS
        return "rms", math.sqrt(0.5)
    return next(((conv, f) for suffix, conv, f in _V_CONVENTIONS if u.name.endswith(suffix)), None)


def _voltage_value(x: float, u: Unit, ref: NumericRef) -> float | None:
    """x in unit u expressed in the reference's voltage convention, or None if it cannot be."""
    a, r = _v_convention(u), _v_convention(ref.unit_obj)
    if a is None or r is None or a[0] == r[0]:
        return x
    if "rms" in (a[0], r[0]) and ref.waveform != "sine":
        return None
    return x + 20 * math.log10(r[1] / a[1]) if u.log else x / a[1] * r[1]


def _grade(q: Quantity, ref: NumericRef) -> dict[str, Any]:
    ru = ref.unit_obj
    if q.kilo_or_kelvin and ru.dims != _KELVIN.dims:     # 48K is 48 000
        q = Quantity(q.value * 1e3, "", (), q.text, q.span, base=True)
    if not q.units:
        return _grade_bare(q, ref)

    def signed(g: dict[str, Any], u: Unit | None, x: float | None = None) -> dict[str, Any]:
        """The credit of the opposite sign, which a sign word may carry ("a loss of 3 dB" for -3 dB).
        x: the value as compared (after a voltage-convention conversion), q.value by default."""
        x = q.value if x is None else x
        uu = ru if u is None else u
        relative_db = ru.db and not ru.log and uu.db and not uu.log
        linear = not ru.db and not uu.db and not ru.zero and not uu.zero
        if (relative_db or linear) and x * ref.value < 0:
            alt = _compare(-x, u, ref)["credit"]
            if alt > g["credit"]:
                g.update(alt_credit=alt, alt_reason="db_sign_convention" if relative_db else "sign_convention")
        return g

    if _norm_str(q.unit) in ref.accept_units:
        return signed({"class": "compatible", **_compare(q.value, None, ref)}, None)
    if not ref.unit and _norm_str(q.unit) in ITEM_COUNT_UNITS:        # P3: a count of items
        return signed({"class": "compatible", "reading": "items", **_compare(q.value, None, ref)}, None)
    for u in q.units:
        if u.dims == ru.dims:
            x = _voltage_value(q.value, u, ref)
            if x is None:
                return {"class": "compatible", "reading": u.name, "credit": 0.0, "reason": "voltage_convention"}
            return signed({"class": "compatible", "reading": u.name, **_compare(x, u, ref)}, u, x)
    if ru.name in _RELATIVE_DB:                     # a plain dB value is read in the reference's unit
        plain = _d(dB_per_Hz=1) if ru.name.endswith("/Hz") else _d(dB=1)
        if any(u.db and not u.log and u.dims == plain for u in q.units):
            return signed({"class": "compatible", "reading": ru.name, **_compare(q.value, None, ref)}, None)
    # P8: the other form, converted ("20 dB" for a power ratio of 100, "90 degrees" for 1.5708, "1 %"
    # for -20 dB); with form_required it scores 0 (wrong_form)
    for u in q.units:
        if ref.dimensionless and u.db and not u.log:
            try:
                lin = 10.0 ** ((q.value + u.offset) / _ratio_log(ref))
            except OverflowError:
                lin = math.inf
            return _converted({"readings": [u.name], **_credit_linear(lin, ref)}, "converted_db", ref)
        if ref.dimensionless and u.dims == _ANGLE:
            return _converted({"readings": [u.name], **_credit_linear(_to_si(q.value, u), ref)},
                              "converted_angle", ref)
        if ru.db and not ru.log and not u.dims and not u.db:
            return _converted({"readings": [u.name], **_ratio_to_db(_to_si(q.value, u), ref)},
                              "converted_ratio", ref)
    return {"class": "other_unit", "credit": 0.0, "reason": "unit_mismatch", "readings": [u.name for u in q.units]}


def _rounded(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            v = float(f"{v:.10g}") if math.isfinite(v) else str(v)
        out[k] = v
    return out


# --- numeric: which number is the answer ---------------------------------------------------------

# digit groups the parser does not join: a plain space ("1 000 000") or a comma before one, two or
# four or more digits ("2,5" or a list "1,2,3")
_UNCLEAR_DIGITS = re.compile(r"(?<![\w.,])\d{1,3}(?: \d{3})+(?![\w.,])|(?<![\w.,])\d+,(?:\d{1,2}|\d{4,})(?!\d)")
_SENTENCE_END = re.compile(r"(?<=\S)[.!?](?=\s|$)|[;\n]")
_NUM_NEGATION = re.compile(r"\b(?:not|never|cannot|neither|nor|unlike|except|wrong|incorrect|mistaken|false)\b|"
                           r"n't\b|\brather\s+than\b|\binstead\s+of\b", re.I)
# a hedge or a correction: blocks an explanation and marks alternatives
_HEDGE = re.compile(r"\b(?:or|either|maybe|perhaps|possibly|probably|alternatively|otherwise|else|actually|"
                    r"correction|instead|rather|wait|no|not|never|wrong|incorrect)\b|n't\b", re.I)
# the text between two values that only joins them as alternatives or a range
_CONNECTOR = re.compile(r"[\s,;]*(?:(?:either|or|nor|and|to|through|thru|until|maybe|perhaps|possibly|"
                        r"probably|else|otherwise|alternatively|vs\.?|versus|[-–—~/|])[\s,;]*)+", re.I)
_RELATION = re.compile(r"=|≈|≅|→|⇒|⟹")
_CONCLUSION = re.compile(r"\b(?:so|therefore|thus|hence|consequently|giving|gives|give|yields?|yielding|"
                         r"results?\s+in|resulting\s+in|get|gets|becomes|making|makes)\b|∴", re.I)
# After the final answer (P5, P7): words that revise it, words that retract it, and the weaker hedges
# that count there only next to another value of the reference's dimension. Words common in
# commentary are left out or narrowed: "No aliasing occurs" (but "No, 20 kHz"), "wait 10 ms for the
# PLL" (but "Wait, ..."), "error-corrected bits" (but "Correction:"), "a common mistake is 15 kHz",
# "the cutoff should be 12 kHz".
_REVISION = re.compile(r"\b(?:actually|correction|oops|sorry|my\s+(?:mistake|bad|error)|re-?check\w*|"
                       r"re-?calculat\w*|re-?comput\w*|reconsider\w*|revis(?:e|ed|ing)\b|hmm+|scratch\s+that|"
                       r"on\s+second\s+thought|make\s+that|i\s+mean|or\s+rather|should\s+have\s+been)\b|"
                       r"\bcorrected\s*(?::|answer|value|result)|\bwait\b[ \t]*(?:[,.;:!—–]|-[ \t])|"
                       r"\b(?:no|but|oh)[ \t,]+wait\b|\bno\b[ \t]*(?:[,.;:!?—–]|-[ \t])", re.I)
_RETRACT = re.compile(r"\b(?:that|this)(?:'s|’s|\s+is|\s+was)\s+(?:wrong|incorrect|not\s+(?:right|correct))\b|"
                      r"\bscratch\s+that\b|\bi\s+made\s+an?\s+(?:error|mistake)\b|\bignore\s+(?:that|the\s+above)\b|"
                      r"\b(?:my|the)\s+(?:previous|above|earlier)\s+answer\s+(?:is|was)\s+(?:wrong|incorrect)\b", re.I)
_TAIL_HEDGE = re.compile(r"\b(?:or|either|maybe|perhaps|possibly|probably|alternatively|otherwise|instead|rather|"
                         r"not|wrong|incorrect)\b|n't\b", re.I)
# a comparator next to the candidate makes it a bound, not a value: "less than 30 kHz", "≥ 30 kHz",
# "30 kHz or more" (P5)
_COMPARATOR_BEFORE = re.compile(
    r"(?:\b(?:(?:no[ \t]+)?(?:less|fewer|smaller|lower|more|greater|larger|higher|bigger|longer|shorter)[ \t]+than|"
    r"at[ \t]+(?:least|most)|up[ \t]+to|below|above|under|over|beyond|exceed(?:s|ing)?)|"
    r"(?:(?<![-=<>])[<>]|[≤≥⩽⩾≲≳])=?)"                       # not the arrows "->" and "=>"
    r"[ \t]*(?:(?:about|approximately|approx\.?|around|roughly|~|≈)[ \t]*)?$", re.I)
_COMPARATOR_AFTER = re.compile(r"^[ \t]*(?:(?:or|and)[ \t]+(?:more|less|fewer|greater|higher|lower|larger|smaller|"
                               r"above|below|over|under|up|down)|at[ \t]+(?:least|most))\b", re.I)
_OP_WORDS = (r"times|twice|half(?:\s+of)?|double|triple|quadruple|divided\s+by|multiplied\s+by|over|plus|minus|"
             r"multiply(?:ing)?|divid(?:e|ing)")
_OP_BEFORE = re.compile(rf"(?:[×*/÷+·^−–-]|(?<![\w])x|\b(?:{_OP_WORDS})(?:\s+(?:the|a|an|its|this|that))?)"
                        r"[\s(]*$", re.I)
_OP_AFTER = re.compile(rf"^[\s)]*(?:[×*/÷+·^]|[−–-]\s*[\d.(]|x(?=\s*[\d(.])|\b(?:{_OP_WORDS})\b)", re.I)
# multiplication and division by a written factor, sums and differences: an answer that is such a
# term with no relation after it in its sentence is an expression left unevaluated (R8). "/" between
# values and "-" are left out: lists, ranges and bullets use them ("10 / 20 / 30 kHz", "20-30 kHz",
# "- 50 MS/s"); "+" counts with a space after it or between two terms ("15 kHz + 15 kHz",
# "15+15 kHz"), as a sign glued to a number ("to +3 dB") belongs to the number
_EXPR_WORDS = (r"times|twice|half(?:\s+of)?|double|triple|quadruple|divided\s+by|multiplied\s+by|"
               r"plus|minus|over")
# an operator symbol needs a term before it on the same line ("2 × 15 kHz", not a "* 30 kHz" bullet);
# after the candidate, an operator needs a number after it ("30 kHz double-sided" is not a product)
# Two terms side by side are a product ("(2)(15 kHz)", "2(15 kHz)"); "/" and a plain number after a
# value with a unit is a division ("48 kHz / 4"), not a list
_EXPR_BEFORE = re.compile(rf"(?:[\w)][ \t]*[×*÷·+]|[\w)][ \t]+x|[\d)][ \t]*\(|\b(?:{_EXPR_WORDS})(?:[ \t]+(?:the|a|an|its|this|that))?)"
                          r"[ \t(]*$", re.I)
_EXPR_AFTER = re.compile(rf"^\((?=[ \t]*[\d.])|^[ \t)]*(?:[×*÷·x+][ \t]*[\d(.]|/[ \t]*\d+(?:\.\d+)?(?![\w.]|[ \t]*[A-Za-zμΩ%])|"
                         rf"(?:{_EXPR_WORDS})\b(?=[ \t]+(?:(?:the|a|an|its|this|that|of)[ \t]+)?[\d(.]))", re.I)
_EXPLAINS = re.compile(rf"[×*/÷+=≈·^]|(?<![\w])x(?![\w])|\b(?:{_OP_WORDS}|i\.?e|that\s+is|since|because|from|"
                       r"per|as)\b", re.I)
_NUM_WORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
             r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
             r"ninety|hundred|thousand|million|billion|twice|half|double|triple|quadruple)")
_NUM_WORDS = re.compile(rf"(?<![\w-]){_NUM_WORD}(?:[\s-]+(?:and[\s-]+)?{_NUM_WORD})*(?![\w])", re.I)
# words that carry a value's sign: "a loss of 3 dB" is -3 dB, "40 dBc below the carrier" -40 dBc,
# "10 kHz below the carrier" -10 kHz, "lags by 90 degrees" -90 deg (P8)
_SIGN_WORD = re.compile(r"\b(?:loss|losses|lose|loses|attenuat\w*|below|beneath|under|down|drops?|dropped|lower|"
                        r"reduc\w*|decreas\w*|suppress\w*|reject\w*|rejection|lag|lags|lagging|lagged|behind|"
                        r"delay|delays|delayed|retard\w*)\b", re.I)
_SIGN_REASONS = ("db_sign_convention", "sign_convention")
# the reasons of a result that still holds two readings (literal_credit, alt_credit); P11 decides it
READING_REASONS = _SIGN_REASONS
# Loss words (decision 7): next to a positive value they negate it when the reference is negative or
# marked quantity "gain" ("a loss of 3 dB" for -3 dB, "3 dB down", "40 dBc below the carrier", "lags by
# 90 degrees"). Every stem of _SIGN_WORD except "under", which is a comparator.
LOSS_WORDS = ("loss", "losses", "lose", "loses", "losing", "attenuate", "attenuates", "attenuated",
              "attenuating", "attenuation", "below", "beneath", "down", "drop", "drops", "dropped", "lower",
              "lowered", "reduce", "reduces", "reduced", "reduction", "decrease", "decreases", "decreased",
              "suppress", "suppressed", "suppression", "reject", "rejected", "rejection", "lag", "lags",
              "lagging", "lagged", "behind", "delay", "delays", "delayed", "retard", "retarded", "retardation")
_LOSS = "(?:" + "|".join(sorted(LOSS_WORDS, key=len, reverse=True)) + ")"
_LOSS_BEFORE = re.compile(rf"\b{_LOSS}(?:[ \t]+(?:of|by|is|was|about|approximately|roughly|around))*[ \t]*[:=~≈]?[ \t]*\Z",
                          re.I)
_LOSS_AFTER = re.compile(rf"^[ \t]*(?:of[ \t]+)?{_LOSS}\b", re.I)
# a part with one of these reasons zeroes the whole answer (as judge_prompt.DU_REASONS does)
PART_ZERO_REASONS = ("unit_mismatch", "order_of_magnitude", "voltage_convention")


def _paren_spans(s: str) -> list[tuple[int, int]]:
    spans, stack = [], []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            a = stack.pop()
            if not stack:
                spans.append((a, i + 1))
    return spans


def _clause_around(s: str, span: tuple[int, int]) -> str:
    start, end = 0, len(s)
    for m in _SENTENCE_END.finditer(s):
        if m.end() <= span[0]:
            start = m.end()
        elif m.start() >= span[1]:
            end = m.start()
            break
    return s[start:end]


def _is_operand(s: str, q: Quantity) -> bool:
    return bool(_OP_BEFORE.search(s[:q.span[0]]) or _OP_AFTER.match(s[q.span[1]:]))


def _number_words_unclear(s: str, g: dict[str, Any], ref: NumericRef) -> bool:
    """A number in words could be the answer: one followed by a unit of the reference's dimension
    ("thirty kilohertz"), or any one when the candidate does not have the reference's dimension."""
    for m in _NUM_WORDS.finditer(s):
        if re.search(r"[\d.][ \t]*$", s[:m.start()]):            # "1.5 million": a parsed multiplier
            continue
        if g["class"] != "compatible":
            return True
        gap = _GAP.match(s, m.end())
        got = _unit_at(s, gap.end()) if gap.end() > m.end() else None
        if got and any(u.dims == ref.unit_obj.dims or (ref.dimensionless and not u.dims and not u.db)
                       for u in got[0]):
            return True
    return False


def _hedged(s: str, q: Quantity, g: dict[str, Any], graded: list[tuple[Quantity, dict]], ref: NumericRef) -> bool:
    """The candidate is joined to a neighbouring value only by "or", "to", "-", "/", "and", ... and that
    value (a bare one read in the candidate's unit: "20-30 kHz") would earn a different credit."""
    order = sorted(graded, key=lambda x: x[0].span)
    i = next(k for k, (p, _) in enumerate(order) if p is q)
    for j in (i - 1, i + 1):
        if not 0 <= j < len(order):
            continue
        p, pg = order[j]
        if pg["class"] == "other_unit":
            continue
        a, b = (p.span[1], q.span[0]) if j < i else (q.span[1], p.span[0])
        if not _CONNECTOR.fullmatch(s[a:b]):
            continue
        if not p.units and q.units:
            alt = _grade(Quantity(p.value, q.unit, q.units, p.text, p.span), ref)["credit"]
        else:
            alt = pg["credit"]
        if alt != g["credit"]:
            return True
    return False


_ASSIGNED = re.compile(r"(?<![\w.])([A-Za-z_]\w*)[ \t]*=[ \t]*$")       # "fs = " right before a value
_ASSIGNS = re.compile(r"[\s*_>-]*([A-Za-z_]\w*)[ \t]*=")                 # "fs =" opening a clause
_CLAUSE_BREAK = re.compile(r"(?<=\S)[.!?](?=\s)|[;\n,]")


def _restated(s: str, p: Quantity, between: str) -> bool:
    """The name p is assigned to is assigned again in a later clause ("fs = 20 kHz. fs = 30 kHz",
    "Ts = 1 ms.\\nTs = 2 ms" in a label's segments): a restatement, not a step towards the candidate."""
    pm = _ASSIGNED.search(s[:p.span[0]])
    breaks = list(_CLAUSE_BREAK.finditer(between))
    if not pm or not breaks:
        return False
    qm = _ASSIGNS.match(between[breaks[-1].end():])
    return bool(qm) and qm[1] == pm[1]


_OWN_ASSIGN = re.compile(r"[A-Za-z_]\w*[ \t]*=[ \t]*$")      # "f_N = " just before the candidate


def _explained(s: str, p: Quantity, q: Quantity, paren: tuple[int, int] | None, q_operand: bool,
               anchored: bool = False) -> bool:
    """Whether p, another value in the decision region, is a step towards the candidate q rather than
    an alternative to it. `anchored`: p is the first value after an explicit final-answer marker or
    in a box, i.e. the stated answer."""
    if paren is not None:                       # "(twice 15 kHz)" explains; "(or 20 kHz)" does not
        t = s[paren[0]:paren[1]]
        return bool(_EXPLAINS.search(t)) and not _HEDGE.search(t)
    if q_operand:
        return False
    if _is_operand(s, p):
        return True
    if p.span[1] <= q.span[0]:
        between = s[p.span[1]:q.span[0]]
        if _HEDGE.search(between) or _restated(s, p, between):
            return False
        # A relation ("=", "→") or a conclusion word makes p a step ("15 kHz, so the sample rate must
        # be 30 kHz"), except when p is the stated answer: in "Final answer: 30 kHz, giving a Nyquist
        # frequency of 15 kHz" the later value is a consequence, and there the candidate's own
        # assignment ("so f_N = 15 kHz") is no relation either; a rival with another credit goes to
        # the judge (round-2 finding R1).
        if _RELATION.search(_OWN_ASSIGN.sub("", between) if anchored else between):
            return True
        return bool(_CONCLUSION.search(between)) and not anchored
    return False


def _comparator(s: str, q: Quantity) -> str | None:
    """A comparator next to the candidate ("less than 30 kHz", "≥ 30 kHz", "30 kHz or more"). A ">"
    opening a line is a Markdown quote, not "greater than"."""
    before = s[:q.span[0]]
    m = _COMPARATOR_BEFORE.search(before)
    if m and not (m.group().lstrip().startswith(">") and not before[:m.start()].rsplit("\n", 1)[-1].strip()):
        return m.group().strip()
    m = _COMPARATOR_AFTER.match(s[q.span[1]:])
    return m.group().strip() if m else None


def _revision(tail: str, credit: float, ref: NumericRef, keep: Any) -> str | None:
    """Whether the text after the final answer revises it (P5): a retraction; a revision word with a
    value of any unit (or a bare one) at another credit; a hedge word with a value of the reference's
    dimension at another credit ("not 2" after "Final answer: 12 kHz" is a factor, not a rate).
    Operands ("2 × 12 kHz") are steps, not values."""
    s = normalize(tail)
    if not s.strip():
        return None
    if _RETRACT.search(s):
        return "retracted"
    strong = bool(_REVISION.search(s))
    if not (strong or _TAIL_HEDGE.search(s)):
        return None
    for q in extract_quantities(tail, False):
        if (keep is not None and not keep(q)) or _is_operand(s, q):
            continue
        g = _loss_word_grade(s, q, _grade(q, ref), ref)          # "the cable's loss is 3 dB" restates -3 dB
        if (strong or g["class"] == "compatible") and g["credit"] != credit:
            return q.text
    return None


def _ignored(q: Quantity, ref: NumericRef, other_dims: frozenset) -> bool:
    """A quantity whose unit fits another part's dimension and not this part's belongs to that part."""
    if not q.units or not other_dims or _norm_str(q.unit) in ref.accept_units:
        return False
    dims = {u.dims for u in q.units}
    return ref.unit_obj.dims not in dims and bool(dims & other_dims)


def _unit_elsewhere(wider: list[str], fa: FinalAnswer, ref: NumericRef, credit: float,
                    keep: Any) -> list[str]:
    """Values with the reference's dimension in the line or paragraph around a bare candidate that
    would earn a different credit ("Answer: 1. 30 kHz", "Answer: 30. The input was 48 kHz")."""
    for region in wider:
        found = [q for q in extract_quantities(region, fa.line_start) if keep(q)]
        graded = [(q, _grade(q, ref)) for q in found]
        compatible = [(q, g) for q, g in graded if g["class"] == "compatible"]
        if compatible:
            return sorted({q.text for q, g in compatible if g["credit"] != credit})
    return []


def _loss_applies(q: Quantity, ref: NumericRef) -> bool:
    """The sign rule applies to relative decibel and linear values (not dBm, dBV or temperatures);
    against a decibel reference only to a decibel value (a ratio carries its own sign: "reduced by
    50 %" is a ratio of 0.5)."""
    ru = ref.unit_obj
    if ru.log or ru.zero:
        return False
    if ru.db:
        return any(u.db and not u.log for u in q.units)
    return not any(u.log or u.zero for u in q.units)


def _loss_word_grade(s: str, q: Quantity, g: dict[str, Any], ref: NumericRef) -> dict[str, Any]:
    """Decision 7's sign rule (P8): a loss word right before the value ("a loss of 3 dB", "attenuated
    by 3 dB") or right after it ("3 dB down", "10 kHz below the carrier") negates a positive value when
    the reference is negative or marked quantity "gain"; a negative value with a loss word is unclear."""
    if not (_LOSS_BEFORE.search(s[:q.span[0]]) or _LOSS_AFTER.match(s[q.span[1]:])) or not _loss_applies(q, ref):
        return g
    if q.value < 0:
        return {**g, "sign_unclear": True}
    if q.value == 0 or not (ref.value < 0 or ref.quantity == "gain"):
        return g
    neg = {k: v for k, v in _grade(replace(q, value=-q.value), ref).items() if k not in ("alt_credit", "alt_reason")}
    if neg["credit"] > 0:
        neg = {**neg, "graded_reason": neg["reason"], "reason": "loss_word_sign"}
    return {**neg, "loss_word": True}


def _decide_numeric(region: str, qs: list[Quantity], fa: FinalAnswer, ref: NumericRef,
                    wider: list[str] = (), keep: Any = None) -> dict[str, Any]:
    s = normalize(region)
    graded = [(q, _loss_word_grade(s, q, _grade(q, ref), ref)) for q in qs]
    parens = _paren_spans(s)

    def paren_of(x: Quantity) -> tuple[int, int] | None:
        return next(((a, b) for a, b in parens if a <= x.span[0] and x.span[1] <= b), None)

    main = [x for x in graded if paren_of(x[0]) is None] or graded
    compatible = [x for x in main if x[1]["class"] == "compatible"]
    native = [x for x in compatible if x[1].get("reading") != "converted"]   # the reference's own form first
    other = [x for x in main if x[1]["class"] == "other_unit"]
    q, g = (native or compatible or other or main)[-1]
    base = {"candidate": q.text, "candidates": sorted({p.text for p, _ in graded})}

    def undecided(reason: str, **more: Any) -> dict[str, Any]:
        return _judge(reason, "numeric", fa, region, **base, **more)

    if any(m.start() < q.span[1] and q.span[0] < m.end() for m in _UNCLEAR_DIGITS.finditer(s)):
        return undecided("unclear_digit_grouping")
    if q.unparsed:
        return undecided("unit_unparsed", unparsed=q.unparsed)
    if q.unknown:
        return undecided("unknown_unit", unknown_unit=q.unknown)
    if _number_words_unclear(s, g, ref):
        return undecided("number_in_words")
    clause = _clause_around(s, q.span)
    if _NUM_NEGATION.search(clause):
        return undecided("negation_near_answer")
    comparator = _comparator(s, q)
    if comparator:
        return undecided("comparator_near_answer", comparator=comparator)
    if g.get("sign_unclear"):
        return undecided("sign_unclear")
    if g.get("alt_credit", 0.0) > g["credit"] and (g["alt_reason"] not in _SIGN_REASONS or _SIGN_WORD.search(clause)):
        return undecided(g["alt_reason"], literal_credit=g["credit"], alt_credit=g["alt_credit"],
                         literal_reason=g["reason"])
    if _hedged(s, q, g, graded, ref):
        return undecided("hedged_answer")
    q_operand = _is_operand(s, q)
    if _EXPR_BEFORE.search(s[:q.span[0]]) or _EXPR_AFTER.match(s[q.span[1]:]):
        # the stated answer is a term of an expression left unevaluated ("2 × 15 kHz", "fs = 2 × 15 kHz";
        # R8): no relation after it in its clause, and no value before a relation before it ("30 kHz =
        # 2 × 15 kHz" states 30 kHz, which the rule for several numbers reads)
        start = max((m.end() for m in _SENTENCE_END.finditer(s) if m.end() <= q.span[0]), default=0)
        before = s[start:q.span[0]]
        end = _SENTENCE_END.search(s, q.span[1])        # from the candidate, so "15 kHz." ends there
        after = s[q.span[1]:end.start() if end else len(s)]
        rel = list(_RELATION.finditer(before))
        # the left side of the last relation is a value ("30 kHz =") rather than a name ("fs =")
        lhs = normalize(before[:rel[-1].start()]) if rel else ""
        lhs_qs = extract_quantities(lhs, False) if lhs else []
        stated_before = bool(lhs_qs) and not lhs[lhs_qs[-1].span[1]:].strip()
        if not _RELATION.search(after) and not stated_before:
            return undecided("expression_answer")
    explicit = fa.rule.split("+")[0] in ("marker", "boxed")
    first = min((x[0] for x in main), key=lambda x: x.span[0])
    native_q = g.get("reading") != "converted"
    rivals = sorted({p.text for p, pg in graded if p is not q and pg["class"] == g["class"]
                     and pg["credit"] != g["credit"]
                     # a converted value is no rival of a native one ("13 dB at 25 % load") unless the
                     # native one is a step towards it ("the voltage ratio is 10, so ... 20 dB")
                     and not (native_q and pg.get("reading") == "converted"
                              and not _explained(s, q, p, paren_of(q), False))
                     and not _explained(s, p, q, paren_of(p), q_operand, anchored=explicit and p is first)})
    if rivals:
        return undecided("ambiguous_final_answer", rivals=rivals)
    if g["class"] == "bare":
        rivals = _unit_elsewhere(list(wider), fa, ref, g["credit"], keep or (lambda x: True))
        if rivals:
            return undecided("ambiguous_final_answer", rivals=rivals)
    details = {"rule": fa.rule, "final_answer": _clip(region), "candidate": q.text, "value": q.value,
               "unit": q.unit, "reference": {"value": ref.value, "unit": ref.unit}}
    details.update({k: v for k, v in g.items() if k not in {"credit"}})
    return _decided("numeric", g["credit"], **_rounded(details))


def _score_numeric(text: str, ref: NumericRef, others: tuple[NumericRef, ...] = ()) -> dict[str, Any]:
    stops = tuple(lab for o in others for lab in o.labels)
    other_dims = frozenset(o.unit_obj.dims for o in others) - {ref.unit_obj.dims}
    places = _final_answers(text, ref.labels, stops)

    def keep(q: Quantity) -> bool:
        return not _ignored(q, ref, other_dims)

    def decide(fa: FinalAnswer, i: int, region: str, qs: list[Quantity], wider: list[str]) -> dict[str, Any]:
        res = _decide_numeric(region, qs, fa, ref, wider, keep)
        if res["method"] == "judge":
            return res
        revised = _revision(fa.tail(fa.clause + i), res["credit"], ref, keep)
        if revised:
            return _judge("revised_after_answer", "numeric", fa, region, candidate=res["details"]["candidate"],
                          revision=revised, literal_credit=res["credit"])
        return res

    for fa in places:
        regions = list(fa.regions[fa.clause:])
        for i, region in enumerate(regions):
            qs = [q for q in extract_quantities(region, fa.line_start) if keep(q)]
            if qs:
                # a bare candidate is checked against the wider regions inside the final answer's
                # paragraph (the line and the paragraph after a marker; a label's scope), not the rest
                wider = [r for r in regions[i + 1:] if r in fa.mention]
                return decide(fa, i, region, qs, wider)
    # Nowhere a quantity this part could take, but the label's own segments give it one in another
    # part's unit ("Ts = 1 ms and B = 1 ms"): that is this part's value, with a unit error (P9).
    scope = next((fa for fa in places if fa.kind == "scope"), None)
    found = extract_quantities(scope.regions[0], scope.line_start) if scope is not None else []
    if found:
        return decide(scope, 0, scope.regions[0], found, [])
    return _judge("no_quantity", "numeric", places[-1])


def _part_zero(parts: list[dict[str, Any]]) -> str | None:
    """The first decided part with a unit or order-of-magnitude error (also as the lower reading a
    tie-break kept, P11): it zeroes the whole answer."""
    for p in parts:
        d = p["details"]
        if p["method"] == "numeric":
            for r in (d.get("reason"), d.get("lower_reason")):
                if r in PART_ZERO_REASONS:
                    return r
    return None


def _score_parts(text: str, refs: tuple[NumericRef, ...]) -> dict[str, Any]:
    """Several numeric parts (P9). `credit` is the mean over the parts, the per-part input that
    judge_prompt.explain_score combines; details["item_credit"] is the item's credit under P9 (0 when
    a decided part has a unit or order-of-magnitude error; None while the judge must still read a part
    and none has)."""
    parts = [_score_numeric(text, r, refs[:i] + refs[i + 1:]) for i, r in enumerate(refs)]
    # R3: two parts of one dimension decided on the same value, one of them without its label, could
    # each be the other's value ("Decimating the 3 MS/s input by 4 gives an output rate of 0.75 MS/s")
    for i, j in itertools.combinations(range(len(parts)), 2):
        di, dj = parts[i]["details"], parts[j]["details"]
        if (refs[i].unit_obj.dims == refs[j].unit_obj.dims and parts[i]["method"] == parts[j]["method"] == "numeric"
                and di.get("candidate") == dj.get("candidate")):
            for k, d in ((i, di), (j, dj)):
                if "label" not in str(d.get("rule", "")):
                    parts[k] = _judge("part_unassigned", "numeric", candidate=d.get("candidate"),
                                      rule=d.get("rule"), final_answer=d.get("final_answer"))
    zero = _part_zero(parts)
    flag = {"part_zero": zero} if zero else {}
    if any(p["method"] == "judge" for p in parts):
        return {"method": "judge", "credit": None,
                "details": {"reason": "part_undecided", "rule_method": "numeric", "parts": parts,
                            "item_credit": 0.0 if zero else None, **flag}}
    mean = sum(p["credit"] for p in parts) / len(parts)
    return _decided("numeric", mean, reason="part_zero" if zero else "parts",
                    item_credit=0.0 if zero else round(mean, 6), **flag, parts=parts)


# --- symbolic ------------------------------------------------------------------------------------

_LIST_SPLIT = re.compile(r"\s*(?:[,;/&+\n]|\band\b)\s*", re.I)       # no "or": a disjunction is a hedge
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_FILLER = re.compile(r"^(?:(?:both|only|just|the|options?|choices?|types?)\s+)+|\s+(?:types?|only)$", re.I)
# negation, contrast and hedge words: options or identifiers named near them are not decided by rules
_CONTRAST = re.compile(r"\b(?:not|no|never|cannot|except|without|unsupported|neither|nor|unlike|instead|only|"
                       r"rather|but|wrong|incorrect|dropped|removed|deprecated|obsolete|replaced|either|or|"
                       r"avoid|unless|although|however|whereas|maybe|perhaps)\b|n't\b", re.I)
_SNAKE = re.compile(r"(?<![\w.])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w])")
_PLAIN_ID = re.compile(r"[A-Za-z_]\w*")
_DOTTED_ID = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\(.*\))?", re.S)


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("`", " ").replace("**", " ")
    return " " + re.sub(r"\s+", " ", s).casefold() + " "


def _clean_list(region: str, names: dict[str, str]) -> set[str] | None:
    """The options of a region that is nothing but a list of known options, else None."""
    pieces = [_norm_str(_FILLER.sub("", _norm_str(_BULLET.sub("", p)))) for p in _LIST_SPLIT.split(region)]
    pieces = [p for p in pieces if p]
    if not pieces or any(p not in names for p in pieces):
        return None
    return {names[p] for p in pieces}


_IDIOMS = re.compile(r"\b(?:in short|for short|short of|in brief)\b")     # not the option "short"


def _mentions(region: str, names: dict[str, str]) -> set[str]:
    """The options a text names (only to show the judge; never decides)."""
    text = _IDIOMS.sub(" ", _norm_text(region))
    found: set[str] = set()
    for spelling in sorted(names, key=lambda n: (-len(n), n)):
        pat = re.compile(r"(?<![\w])" + re.escape(spelling) + r"(?![\w])")
        if pat.search(text):
            found.add(names[spelling])
            text = pat.sub(" ", text)
    return found


# words that make a parenthetical or a later sentence an alternative or an addition (P7)
_ALSO = re.compile(r"\b(?:also|too|as\s+well|works?|equally|likewise|alternatively|either|plus)\b", re.I)


def _parenthetical(fa: FinalAnswer, k: int) -> str:
    """The parenthetical remarks that region k leaves out (k is the core of the first clause), else ''."""
    if k >= fa.clause or fa.clause >= len(fa.regions):
        return ""
    clause = fa.regions[fa.clause]
    return " ".join(clause[a + 1:b - 1] for a, b in _paren_spans(clause))


def _named_options(text: str, known: dict[str, str]) -> set[str]:
    """The options a text names; a one-letter option only as a capital or in brackets ("C", "(c)"), so
    the article "a" is not option A."""
    found = _mentions(text, {n: o for n, o in known.items() if len(n) > 1})
    for n, o in known.items():
        if len(n) == 1 and re.search(rf"(?<![\w]){re.escape(n.upper())}(?![\w])|"
                                     rf"[(\[][ \t]*(?i:{re.escape(n)})[ \t]*[)\]]", text):
            found.add(o)
    return found


def _foreign_ids(text: str, known: dict[str, str], accepted: frozenset[str] | set[str]) -> set[str]:
    """Identifiers (code spans, snake_case names) in the text that are neither known options nor accepted."""
    ids, _ = _identifiers(text, frozenset(accepted))
    return {i for i in ids if i not in accepted and i not in known}


def _symbolic_doubt(fa: FinalAnswer, k: int, known: dict[str, str], found: set[str],
                    accepted: frozenset[str] | set[str], tail: str | None = None) -> str | None:
    """Why a symbolic decision read from region k goes to the judge (P7), else None: a parenthetical it
    leaves out holds a hedge or another option ("complex (or float)"); or the text after it revises it
    ("Answer: B. Hmm, actually C."), or holds a hedge, contrast or addition word next to another option
    or identifier."""
    paren = _parenthetical(fa, k)
    if paren.strip():
        if _HEDGE.search(paren) or _CONTRAST.search(paren) or _ALSO.search(paren):
            return "hedged_parenthetical"
        if _named_options(paren, known) - found or _foreign_ids(paren, known, accepted):
            return "option_in_parenthetical"
    tail = fa.tail(k) if tail is None else tail
    if tail.strip():
        if _RETRACT.search(tail) or _REVISION.search(tail):
            return "revised_after_answer"
        if ((_TAIL_HEDGE.search(tail) or _CONTRAST.search(tail) or _ALSO.search(tail))
                and (_named_options(tail, known) - found or _foreign_ids(tail, known, accepted))):
            return "revised_after_answer"
    return None


def _set_verdict(found: set[str], ref: SetRef, fa: FinalAnswer, how: str) -> dict[str, Any]:
    missing, extra = sorted(ref.items - found), sorted(found - ref.items)
    reason = "set_match" if not (missing or extra) else ("missing_options" if missing else "extra_options")
    return _decided("symbolic", 0.0 if (missing or extra) else 1.0, reason=reason, rule=fa.rule, how=how,
                    found=sorted(found), expected=sorted(ref.items), missing=missing, extra=extra,
                    final_answer=_clip(fa.mention))


# the option letters leading an explicit final answer: "(A) and (C)", "options A, C", "B) text"
_LETTER = r"(?:[(\[]\s*[A-Za-z]\s*[)\]]|[A-Z](?![\w]))"
# the first letter of a leading list: a bare capital counts only where a letter answer can go on or
# end (the end of the clause, punctuation, Markdown emphasis, a spaced dash, "&", "+", or "and"/"or"
# and another letter), so in "A low-pass filter is needed" the "A" is the article (R7); a letter after
# a separator cannot be the article
_FIRST_LETTER = (r"(?:[(\[]\s*[A-Za-z]\s*[)\]]|[A-Z](?![\w])"
                 r"(?=[*_]*(?:\s*(?:$|[,.;:)\]&+!?]|(?:and|or)\s+(?:[(\[]\s*)?[A-Za-z](?![\w]))|\s+[-–—](?:\s|$))))")
_LEADING_LETTERS = re.compile(rf"\s*(?:(?i:options?|choices?)\s*:?\s*)?"
                              rf"({_FIRST_LETTER}(?:\s*(?:,|&|\+|\band\b)\s*{_LETTER})*)")


def _answer_clause(fa: FinalAnswer) -> str:
    """The first clause of the final answer (for an unmarked answer, of its first line)."""
    if fa.rule == "last":
        first = next((line for line in fa.mention.split("\n") if line.strip()), "")
        return _first_clause(first, True).strip()
    return fa.regions[min(fa.clause, len(fa.regions) - 1)]


def _clause_tail(fa: FinalAnswer) -> str:
    """The answer text after _answer_clause(fa)."""
    if fa.rule == "last":
        clause = _answer_clause(fa)
        at = fa.mention.find(clause) if clause else -1
        return fa.mention[at + len(clause):] if at >= 0 else ""
    return fa.tail(min(fa.clause, len(fa.regions) - 1))


def _score_set(text: str, ref: SetRef) -> dict[str, Any]:
    fa = final_answer(text)
    names = ref.lookup
    for k in reversed(range(len(fa.regions))):                  # widest clean list first
        found = _clean_list(fa.regions[k], names)
        if found is not None:
            doubt = _symbolic_doubt(fa, k, names, found, found)
            if doubt:
                return _judge(doubt, "symbolic", fa, fa.mention, candidate=sorted(found))
            return _set_verdict(found, ref, fa, "clean_list")
    candidate = sorted(_mentions(fa.mention, names))
    if any(len(n) <= 1 for n in names) and fa.rule != "last":
        clause = _answer_clause(fa)
        m = _LEADING_LETTERS.match(clause)
        if m and not _CONTRAST.search(_norm_text(clause)):
            letters = {c.casefold() for c in re.findall(r"[A-Za-z]", re.sub(r"(?i)\band\b", " ", m[1]))}
            if letters and letters <= set(names):
                found = {names[c] for c in letters}
                doubt = _symbolic_doubt(fa, fa.clause, names, found, found, _clause_tail(fa))
                if doubt:
                    return _judge(doubt, "symbolic", fa, fa.mention, candidate=sorted(found))
                return _set_verdict(found, ref, fa, "letters")
    reason = "unknown_option" if not ref.closed and candidate else "not_a_clean_list"
    return _judge(reason, "symbolic", fa, fa.mention, candidate=candidate)


def _identifiers(clause: str, accepted: frozenset[str]) -> tuple[set[str], bool]:
    """The identifiers a clause names: code spans and snake_case names, normalised (`blocks.throttle`
    reads blocks_throttle; an instance name blocks_throttle_0 reads blocks_throttle). The flag is set
    when a code span is not a plain identifier that could be read."""
    found: set[str] = set()
    unclear = False
    for span in re.findall(r"`([^`\n]+)`", clause):
        t = span.strip()
        dotted = _DOTTED_ID.fullmatch(t)
        if _PLAIN_ID.fullmatch(t):
            found.add(t.casefold())
        elif dotted and "_".join(dotted[1].split(".")[-2:]).casefold() in accepted:
            found.add("_".join(dotted[1].split(".")[-2:]).casefold())
        else:
            unclear = True
    found |= set(_SNAKE.findall(re.sub(r"`[^`\n]+`", " ", clause).casefold()))
    out = set()
    for ident in found:
        stem = re.sub(r"_\d+$", "", ident)
        out.add(stem if ident not in accepted and stem in accepted else ident)
    return out, unclear


BLOCK_CATALOG = REPORTS / "block_catalog.json"


@functools.cache
def _block_ids() -> frozenset[str]:
    """The block ids of GNU Radio's catalog, as frozen in reports/block_catalog.json (`./gb blocks`), so
    the rule reads the same on a machine without GNU Radio."""
    return frozenset(b["id"].casefold() for b in json.loads(BLOCK_CATALOG.read_text(encoding="utf-8"))["blocks"])


def _score_exact(text: str, ref: ExactRef) -> dict[str, Any]:
    fa = final_answer(text)
    known = {a: ref.target for a in ref.accepted} | {u: u for u in ref.universe} | dict(ref.aliases)
    target = {ref.target}
    for k, region in enumerate(fa.regions):
        if _norm_str(region) in ref.accepted:
            doubt = _symbolic_doubt(fa, k, known, target, ref.accepted)
            if doubt:
                return _judge(doubt, "symbolic", fa, fa.mention, candidate=ref.target)
            return _decided("symbolic", 1.0, reason="exact_match", rule=fa.rule, final_answer=_clip(region))
    if ref.universe:
        names = ref.universe | ref.accepted
        return _score_set(text, _set_ref([ref.target], sorted(names), list(ref.aliases)))
    region = fa.mention
    norm = _norm_text(region)
    named = any(re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", norm) for a in ref.accepted)
    if _CONTRAST.search(norm):
        return _judge("negation_or_contrast", "symbolic", fa, region)
    clause = _answer_clause(fa)
    ids, unclear = _identifiers(clause, ref.accepted)
    hits, others = ids & ref.accepted, sorted(ids - ref.accepted)
    if (hits and not others and not unclear) or (others and not hits and not unclear and not named):
        doubt = _symbolic_doubt(fa, fa.clause, known, set(hits) or set(others), ref.accepted | set(others),
                                _clause_tail(fa))
        if doubt:
            return _judge(doubt, "symbolic", fa, region, candidate=sorted(ids))
    blocks = _block_ids()
    if others and not hits and not unclear and not named and ref.target in blocks and not set(others) <= blocks:
        return _judge("not_a_block_id", "symbolic", fa, region, candidate=others)     # R6
    if hits and not others and not unclear:
        return _decided("symbolic", 1.0, reason="identifier_match", rule=fa.rule, found=sorted(hits),
                        final_answer=_clip(clause))
    if others and not hits and not unclear and not named:
        return _decided("symbolic", 0.0, reason="wrong_identifier", rule=fa.rule, found=others,
                        expected=ref.target, final_answer=_clip(clause))
    if hits or others or unclear:
        return _judge("several_identifiers", "symbolic", fa, region, candidate=sorted(ids))
    if not named:
        return _judge("target_not_found", "symbolic", fa, region)
    return _judge("named_in_prose", "symbolic", fa, region, candidate=ref.target)


# --- the cascade ---------------------------------------------------------------------------------


def tie_break(result: dict[str, Any]) -> dict[str, Any]:
    """P11: a result (or a part) that still holds two readings of different credit (literal_credit and
    alt_credit) is decided with the lower one (reason tie_break_lower); for parts, the mean and the
    item credit (P9) are recomputed."""
    d = result.get("details") or {}
    parts = d.get("parts")
    if parts is not None:
        new = [tie_break(p) for p in parts]
        if new == parts:
            return result
        zero = _part_zero(new)
        flag = {"part_zero": zero} if zero else {}
        rest = {k: v for k, v in d.items() if k not in ("parts", "item_credit", "part_zero", "reason")}
        if any(p["method"] == "judge" for p in new):
            return {"method": "judge", "credit": None,
                    "details": {**rest, "reason": d.get("reason"), "parts": new,
                                "item_credit": 0.0 if zero else None, **flag}}
        mean = sum(p["credit"] for p in new) / len(new)
        rest.pop("rule_method", None)
        return _decided("numeric", mean, **rest, reason="part_zero" if zero else "parts",
                        item_credit=0.0 if zero else round(mean, 6), **flag, parts=new)
    if result.get("method") == "judge" and "literal_credit" in d and "alt_credit" in d:
        rest = {k: v for k, v in d.items() if k not in ("reason", "rule_method")}
        literal = d["literal_credit"] <= d["alt_credit"]
        # the lower reading's own reason (an order-of-magnitude error still zeroes an item, P9)
        lower = {"lower_reason": d["literal_reason"]} if literal and d.get("literal_reason") else {}
        return _decided("numeric", min(d["literal_credit"], d["alt_credit"]), **rest,
                        reason="tie_break_lower", undecided_reason=d.get("reason"), **lower)
    return result


def reading_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    """Per model, the answers the tie-break decided and the answers scored 0 for the wrong form
    (decision 7: reported next to every score), counting the parts of an answer as one answer."""
    def reasons(r: dict[str, Any]) -> set[str]:
        d = r.get("details") or {}
        own = {d.get("reason")}
        return own.union(*(reasons(p) for p in d.get("parts") or ()))
    sets = [reasons(r) for r in results]
    return {"tie_break": sum("tie_break_lower" in s for s in sets),
            "wrong_form": sum("wrong_form" in s for s in sets)}


def score(answer_text: str | None, reference: Any, *, truncated: bool = False) -> dict[str, Any]:
    """Score one Tier-K answer against its reference: parse -> numeric -> symbolic -> judge, and
    the tie-break (P11) on what is still undecided between two readings."""
    return tie_break(_score(answer_text, reference, truncated=truncated))


def _score(answer_text: str | None, reference: Any, *, truncated: bool = False) -> dict[str, Any]:
    ref = parse_reference(reference)
    # an open-ended rubric has no rule target: its early zeros are decided by rule (F9)
    method = ("numeric" if isinstance(ref, (NumericRef, tuple)) else "rules" if isinstance(ref, NoRuleRef)
              else "symbolic")
    if truncated:
        return _decided(method, 0.0, reason="truncated")
    text = strip_reasoning(answer_text or "")
    if not text:
        return _decided(method, 0.0, reason="empty_answer")
    if isinstance(ref, NoRuleRef):
        return _judge("no_rule_reference", "none", final_answer(text))
    if isinstance(ref, tuple):
        return _score_parts(text, ref)
    if isinstance(ref, NumericRef):
        return _score_numeric(text, ref)
    if isinstance(ref, SetRef):
        return _score_set(text, ref)
    return _score_exact(text, ref)


# --- self-test: hand-made cases (no benchmark item, nothing sent to a model) ---------------------

_TYPES = ["byte", "complex", "float", "int", "short"]
_K30 = {"value": 30, "unit": "kHz"}
SELFTEST: tuple[dict[str, Any], ...] = (
    {"name": "kS/s equals MS/s", "answer": "The decimated stream runs at 2500 kS/s.",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 1.0},
    {"name": "sample rate in Hz equals S/s", "answer": "Set samp_rate = 2.5 MHz.",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 1.0},
    {"name": "scientific notation", "answer": "fs = 2.5e6 Hz", "reference": {"value": 2.5, "unit": "MHz"},
     "method": "numeric", "credit": 1.0},
    {"name": "thousands separators", "answer": "It produces 2,500,000 samples per second.",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 1.0},
    {"name": "times ten to the power", "answer": "rate = 2.5 × 10^6 Sa/s",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 1.0},
    {"name": "LaTeX", "answer": r"$f_s = 2.5\times10^{6}\,\text{S/s}$",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 1.0},
    {"name": "dB where dBm is meant", "answer": "The output power is 27 dB.",
     "reference": {"value": 27, "unit": "dBm"}, "method": "numeric", "credit": 0.0},
    {"name": "dBm converts to W", "answer": "Final answer: 1 W", "reference": {"value": 30, "unit": "dBm"},
     "method": "numeric", "credit": 1.0},
    {"name": "time where frequency is meant", "answer": "The period is 0.4 µs.",
     "reference": {"value": 2.5, "unit": "MHz"}, "method": "numeric", "credit": 0.0},
    {"name": "factor-10 too large", "answer": "About 25 MHz.",
     "reference": {"value": 2.5, "unit": "MHz", "tolerance": 0.5}, "method": "numeric", "credit": 0.0},
    {"name": "factor-10 too small", "answer": "About 0.25 MHz.",
     "reference": {"value": 2.5, "unit": "MHz", "tolerance": 0.5}, "method": "numeric", "credit": 0.0},
    {"name": "tier 1 (<= 1 %)", "answer": "1.005 kHz", "reference": {"value": 1, "unit": "kHz"},
     "method": "numeric", "credit": 1.0},
    {"name": "tier 2 (<= 5 %)", "answer": "1.03 kHz", "reference": {"value": 1, "unit": "kHz"},
     "method": "numeric", "credit": 0.5},
    {"name": "outside the tiers", "answer": "1.06 kHz", "reference": {"value": 1, "unit": "kHz"},
     "method": "numeric", "credit": 0.0},
    {"name": "half band capped at tolerance + 4 %", "answer": "Final answer: 0.5 kHz",
     "reference": {"value": 1, "unit": "kHz", "tolerance": 0.1}, "method": "numeric", "credit": 0.0},
    {"name": "tiny reference keeps relative tiers", "answer": "Final answer: 6e-38",
     "reference": {"value": 3e-38, "unit": ""}, "method": "numeric", "credit": 0.0},
    {"name": "dB error is absolute", "answer": "Noise floor: -117.2 dBm",
     "reference": {"value": -117, "unit": "dBm"}, "method": "numeric", "credit": 0.5},
    {"name": "thinking stripped", "answer": "<think>Maybe 20 kHz? No.</think>The minimum rate is 30 kHz.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "only the answer counts", "answer": "<think>It is 30 kHz.</think>The minimum rate is 20 kHz.",
     "reference": _K30, "method": "numeric", "credit": 0.0},
    {"name": "cut off while thinking", "answer": "<think>So the rate is 30 kHz and",
     "reference": _K30, "method": "numeric", "credit": 0.0},
    {"name": "several numbers: a derivation chain", "answer": "With B = 15 kHz, fs = 2 × 15 kHz = 30 kHz.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "several numbers: premise then conclusion",
     "answer": "The bandwidth is 15 kHz, so the sample rate must be 30 kHz.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "several numbers: a revision goes to the judge",
     "answer": "A first guess is 20 kHz; with the factor of two it is 30 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "hedged answer goes to the judge", "answer": "It is 30 kHz, or maybe 20 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "hedge in the other order too", "answer": "It is 20 kHz, or maybe 30 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "range goes to the judge", "answer": "Final answer: 20-30 kHz",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "negated answer goes to the judge", "answer": "The answer is not 30 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "final-answer marker", "answer": "Final answer: 30 kHz\n\nA 37.5 kHz rate leaves a guard band.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "boxed answer", "answer": r"So \boxed{30\ \text{kHz}}. Audio often uses 48 kHz.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "bare number read in the reference unit", "answer": "Answer: 30",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "no number goes to the judge", "answer": "Twice the highest frequency in the signal.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "number in words goes to the judge", "answer": "Sample at thirty kilohertz, twice the 15 kHz bandwidth.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "fraction", "answer": "Final answer: 1/4", "reference": {"value": 0.25, "unit": ""},
     "method": "numeric", "credit": 1.0},
    {"name": "pi in radians vs degrees", "answer": "The phase shift is π/2 rad.",
     "reference": {"value": 90, "unit": "deg"}, "method": "numeric", "credit": 1.0},
    {"name": "percent", "answer": "About 12.5% of the samples.", "reference": {"value": 0.125, "unit": ""},
     "method": "numeric", "credit": 1.0},
    {"name": "cycles per second are hertz", "answer": "1000 cycles per second",
     "reference": {"value": 1, "unit": "kHz"}, "method": "numeric", "credit": 1.0},
    {"name": "dB for a linear ratio is converted", "answer": "Processing gain = 20 dB",
     "reference": {"value": 100, "unit": ""}, "method": "numeric", "credit": 1.0},
    {"name": "truncated answer", "answer": "The rate is 30 kHz", "truncated": True,
     "reference": _K30, "method": "numeric", "credit": 0.0},
    {"name": "samples per symbol", "answer": "Use 4 sps.", "reference": {"value": 4, "unit": "samples/symbol"},
     "method": "numeric", "credit": 1.0},
    {"name": "bit rate vs bandwidth", "answer": "About 1.2 MHz.", "reference": {"value": 1.2, "unit": "Mbit/s"},
     "method": "numeric", "credit": 0.0},
    {"name": "symbol rate is not a sample rate", "answer": "2.5 Msym/s",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "numeric", "credit": 0.0},
    {"name": "unknown unit spelling goes to the judge", "answer": "Answer: 2500 kSamp/s",
     "reference": {"value": 2.5, "unit": "MS/s"}, "method": "judge", "credit": None},
    {"name": "labelled part", "answer": "B = 1.25 kHz and Ts = 1 ms.",
     "reference": {"value": 1, "unit": "ms", "label": "Ts"}, "method": "numeric", "credit": 1.0},
    {"name": "the final answer beats an earlier label", "answer": "Ts = 2 ms at first.\nFinal answer: 1 ms",
     "reference": {"value": 1, "unit": "ms", "label": "Ts"}, "method": "numeric", "credit": 1.0},
    {"name": "parts: value before label",
     "answer": "You need 50 MS/s for a real sampler and 25 MS/s for a complex sampler.",
     "reference": [{"value": 50, "unit": "MS/s", "label": ["real sampler", "real"]},
                   {"value": 25, "unit": "MS/s", "label": ["complex sampler", "complex"]}],
     "method": "numeric", "credit": 1.0},
    {"name": "set as a clean list", "answer": "Answer: float, complex",
     "reference": {"set": ["complex", "float"], "universe": _TYPES}, "method": "symbolic", "credit": 1.0},
    {"name": "set with an extra option", "answer": "Answer: complex, float, int",
     "reference": {"set": ["complex", "float"], "universe": _TYPES}, "method": "symbolic", "credit": 0.0},
    {"name": "set with a missing option", "answer": "Answer: complex",
     "reference": {"set": ["complex", "float"], "universe": _TYPES}, "method": "symbolic", "credit": 0.0},
    {"name": "set: unknown piece without a universe", "answer": "Answer: complex, float, int",
     "reference": ["complex", "float"], "method": "judge", "credit": None},
    {"name": "set in prose goes to the judge", "answer": "Only complex is supported; float was dropped.",
     "reference": {"set": ["complex", "float"], "universe": _TYPES}, "method": "judge", "credit": None},
    {"name": "multiple choice letters", "answer": "Final answer: (A) and (C)",
     "reference": {"set": ["A", "C"], "universe": ["A", "B", "C", "D"]}, "method": "symbolic", "credit": 1.0},
    {"name": "letter answer with an explanation", "answer": "Final answer: (B), since (A) doubles the rate",
     "reference": {"set": ["B"], "universe": ["A", "B", "C", "D"]}, "method": "symbolic", "credit": 1.0},
    {"name": "block id in code", "answer": "Insert a `blocks_throttle` after the source.",
     "reference": "blocks_throttle", "method": "symbolic", "credit": 1.0},
    {"name": "wrong block id", "answer": "Insert a `blocks_head` after the source.",
     "reference": "blocks_throttle", "method": "symbolic", "credit": 0.0},
    {"name": "block id named in contrast goes to the judge",
     "answer": "You might think of blocks_throttle, but the right block is the Head block.",
     "reference": "blocks_throttle", "method": "judge", "credit": None},
    {"name": "block described in words goes to the judge", "answer": "Insert a rate limiter after the source.",
     "reference": "blocks_throttle", "method": "judge", "credit": None},
    {"name": "open-ended rubric goes to the judge", "answer": "Throttle paces a simulation.",
     "reference": {"numeric": None, "required_points": ["P1"]}, "method": "judge", "credit": None},
    # regressions found while verifying 0.1 and 0.2
    {"name": "number ending the answer is not a list marker", "answer": "Answer: 4.",
     "reference": {"value": 4, "unit": ""}, "method": "numeric", "credit": 1.0},
    {"name": "list marker opening a line is skipped", "answer": "Final answer:\n1. 30 kHz",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "paragraph after the marker decides", "answer": "Answer: 30.\n\nThe input was 48 kHz.",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "bare number next to a rate goes to the judge", "answer": "Answer: 30. The input was 48 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "parenthetical explanation is not an alternative", "answer": "Final answer: 30 kHz (twice 15 kHz)",
     "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "attached multiplier is in SI base units", "answer": "Answer: 10k",
     "reference": {"value": 10, "unit": "kHz"}, "method": "numeric", "credit": 1.0},
    {"name": "48K is 48 000 for a rate", "answer": "48K", "reference": {"value": 48, "unit": "kHz"},
     "method": "numeric", "credit": 1.0},
    {"name": "a bare number is read in the reference unit only", "answer": "Answer: 10000",
     "reference": {"value": 10, "unit": "kHz"}, "method": "numeric", "credit": 0.0},
    {"name": "power of ten with **", "answer": "Throughput is 1.5*10**6 bit/s",
     "reference": {"value": 1.5, "unit": "Mbit/s"}, "method": "numeric", "credit": 1.0},
    {"name": "decimal comma goes to the judge", "answer": "2,5 MHz",
     "reference": {"value": 2.5, "unit": "MHz"}, "method": "judge", "credit": None},
    {"name": "dB tolerance is given in dB", "answer": "-117.4 dBm",
     "reference": {"value": -117, "unit": "dBm", "abs_tolerance": 0.5}, "method": "numeric", "credit": 1.0},
    {"name": "huge dBm does not overflow", "answer": "1e6 dBm", "reference": {"value": 30, "unit": "dBW"},
     "method": "numeric", "credit": 0.0},
    {"name": "label: last mention with a number", "answer": "Ts = 1 ms. Here Ts is the symbol period.",
     "reference": {"value": 1, "unit": "ms", "label": "Ts"}, "method": "numeric", "credit": 1.0},
    {"name": "plain-word target named in prose goes to the judge", "answer": "The output is complex or float.",
     "reference": "complex", "method": "judge", "credit": None},
    {"name": "one-letter target is not the article 'a'", "answer": "It is B, because a filter is needed.",
     "reference": "A", "method": "judge", "credit": None},
    {"name": "option letters in the final answer", "answer": "Final answer: options A and C",
     "reference": {"set": ["A", "C"], "universe": ["A", "B", "C", "D"]}, "method": "symbolic", "credit": 1.0},
    # regressions found while verifying 0.3 (methodology review, round 3)
    {"name": "a revision after the final answer goes to the judge", "answer": "Answer: 30 kHz? No, 20 kHz.",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "a revision after the final answer, other order", "answer": "Final answer: 24 kHz.\nHmm, actually it "
     "should be 30 kHz.", "reference": _K30, "method": "judge", "credit": None},
    {"name": "commentary after the final answer is not a revision",
     "answer": "Final answer: 30 kHz\n\nNo aliasing occurs at this rate.", "reference": _K30,
     "method": "numeric", "credit": 1.0},
    {"name": "'final answer' in prose is not a marker", "answer": "To get the final answer, multiply 15 kHz by 2: "
     "30 kHz.", "reference": _K30, "method": "numeric", "credit": 1.0},
    {"name": "a comparator makes a bound, not a value", "answer": "Answer: less than 30 kHz",
     "reference": _K30, "method": "judge", "credit": None},
    {"name": "a hedged parenthetical option goes to the judge", "answer": "Answer: complex (or float)",
     "reference": "complex", "method": "judge", "credit": None},
    {"name": "a revised option goes to the judge", "answer": "Final answer: B. Hmm, actually C.",
     "reference": {"set": ["B"], "universe": ["A", "B", "C", "D"]}, "method": "judge", "credit": None},
    {"name": "a revised block id goes to the judge", "answer": "Answer: `blocks_throttle`. No wait, `blocks_head`.",
     "reference": "blocks_throttle", "method": "judge", "credit": None},
    {"name": "a bare number takes no other SI prefix", "answer": "Answer: 30",
     "reference": {"value": 30000, "unit": "Hz"}, "method": "numeric", "credit": 0.0},
    {"name": "a direction word next to the value sets its sign",
     "answer": "The image lies 10 kHz below the carrier.", "reference": {"value": -10, "unit": "kHz"},
     "method": "numeric", "credit": 1.0},
    {"name": "dB below the carrier for a dBc value",
     "answer": "The spur is 40 dB below the carrier.", "reference": {"value": -40, "unit": "dBc"},
     "method": "numeric", "credit": 1.0},
    {"name": "plain dB for dBc is read as dBc", "answer": "The spur is at -40 dB.",
     "reference": {"value": -40, "unit": "dBc"}, "method": "numeric", "credit": 1.0},
    {"name": "plain dB for dBm is a unit error", "answer": "The level is -40 dB.",
     "reference": {"value": -40, "unit": "dBm"}, "method": "numeric", "credit": 0.0},
)


def selftest(cases: tuple[dict[str, Any], ...] = SELFTEST) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        got = score(case["answer"], case["reference"], truncated=case.get("truncated", False))
        ok = got["method"] == case["method"] and got["credit"] == case["credit"]
        rows.append({"name": case["name"], "ok": ok, "expected": (case["method"], case["credit"]),
                     "got": (got["method"], got["credit"]), "reason": got["details"].get("reason")})
    return rows


# --- CLI -----------------------------------------------------------------------------------------


def _batch(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows, errors = [], 0
    for row in iter_jsonl(path):
        try:
            res = score(row.get("answer"), row["reference"], truncated=bool(row.get("truncated")))
        except (KeyError, TypeError, ValueError) as exc:        # one malformed reference: an error row
            errors += 1
            res = {"method": None, "credit": None,
                   "details": {"reason": "bad_reference", "error": f"{type(exc).__name__}: {exc}"}}
        only = rule_only_credit(res) if res["method"] is not None else None
        rows.append({"id": row.get("id"), "rules_version": RULES_VERSION, **res, "rule_only_credit": only})
    return rows, errors


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb score-k", description="Rule-cascade scorer for Tier-K answers "
                                 "(parse -> numeric -> symbolic -> judge).")
    ap.add_argument("--selftest", action="store_true", help="run the hand-made cases in SELFTEST")
    ap.add_argument("--answer", help="one answer text")
    ap.add_argument("--answer-file", type=Path, help="read the answer text from a file")
    ap.add_argument("--reference", help='reference as JSON, e.g. \'{"value": 2.5, "unit": "MS/s"}\'')
    ap.add_argument("--truncated", action="store_true", help="the answer hit the length cap")
    ap.add_argument("--batch", type=Path, help="JSONL rows {id, answer, reference[, truncated]}")
    ap.add_argument("--out", type=Path, help="with --batch: write scored rows here (else stdout)")
    args = ap.parse_args(argv)

    if args.selftest:
        rows = selftest()
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            exp, got = r["expected"], r["got"]
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['name']:<{width}}  {got[0]:<8} {str(got[1]):<5} "
                  f"{r['reason']}" + ("" if r["ok"] else f"   (expected {exp[0]} {exp[1]})"))
        bad = sum(not r["ok"] for r in rows)
        print(f"\n{len(rows) - bad}/{len(rows)} cases pass ({RULES_VERSION})")
        return 1 if bad else 0

    if args.batch:
        out_rows, errors = _batch(args.batch)
        counts = Counter(str(r["method"]) for r in out_rows)
        if args.out:
            write_jsonl(args.out, out_rows)
            print(f"wrote {len(out_rows)} rows to {args.out}")
        else:
            for r in out_rows:
                print(json.dumps(r, sort_keys=True, ensure_ascii=False))
        summary = rule_only_summary(out_rows)
        if summary["n"]:
            decided = ("n/a" if summary["decided_mean"] is None
                       else f"{summary['decided_mean']:.4f} over {summary['decided']} decided")
            print("methods: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                  + f"; deferred to the judge {summary['deferred']}/{summary['n']} ({summary['deferral_rate']:.1%});"
                  f" rule-only mean credit {summary['rule_only_mean']:.4f} (deferrals at 0);"
                  f" decided-only mean credit {decided}", file=sys.stderr)
        if errors:
            print(f"{errors} row(s) with a malformed reference (method null, reason bad_reference)",
                  file=sys.stderr)
        return 1 if errors else 0

    if args.reference is None or (args.answer is None and args.answer_file is None):
        ap.print_help()
        return 2
    answer = args.answer if args.answer is not None else args.answer_file.read_text(encoding="utf-8")
    res = score(answer, json.loads(args.reference), truncated=args.truncated)
    print(json.dumps(res, sort_keys=True, ensure_ascii=False, indent=1))
    return 0
