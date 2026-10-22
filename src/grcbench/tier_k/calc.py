"""`./gb calc` computes the answers to the Tier-K calculation test questions by script (eval plan §5.2).

Eval plan §5.2: "Calculation answers are computed by a script from the question's numbers", and "a
question whose correct answer depends on assumptions the question does not state is rewritten, or
removed and counted (§5.3)". Eval plan §4.2 "Scoring" and proposal §5.1: calculation items get tiered
relative-error credit, and a wrong unit or order of magnitude scores zero (tier_k/rules.py applies it).

Role table (eval plan §5.1, Table 5; decision of 14 Oct). Claude may compute calculation answers by
script; every record is "pending blind team recomputation", and a team member recomputes each answer
without seeing the script's. That exception covers only values computed from numbers the question
states. Claude drafts no reference answer and no rubric prose:
  - a question labelled `calculation` whose answer is an expression or an example (no number to
    compute from) gets no script value: its record is a `team reference`, and its function computes
    nothing and holds no answer content (no formula, no source, no check of a candidate answer);
  - the parts of a question that ask for an explanation are listed under `not_computed`;
  - tolerance notes say what a tolerance admits, never which answers are wrong or earn partial
    credit (the rubric's errors are the team's);
  - `numeric_proposal` is only a proposal: the team writes the rubric's `numeric` and
    `numeric_tolerance` from the confirmed value (docs/guidelines/rubric_writing.md).

Which questions. The 25 frozen test questions whose clean task type is `calculation`
(splits/task_type_labels.jsonl), cross-checked against items/tier_k/test/frozen_questions.jsonl and the
raw spellings `calculation` (23) and `calculations` (2) in 04_eval_frozen/eval_questions.jsonl. The
build stops if the three sets differ, if the question text differs from the dataset, if a question has
no function here (or a function has no question), or if a number a function relies on is missing
from its question text (`stated`).

One function per question, named by the question id. Its keyword defaults are its inputs: the numbers
stated in the question (recorded as `stated_inputs`) and the constants the question does not state but
the value uses, declared with `assumed=` (recorded as `assumed_constants`, each named under
Assumptions). It re-checks its result independently where it can (numpy FFT, numerical integration, a
filter simulation, a direct sum) and returns an Answer. Its docstring holds the sections the record
copies (a line that opens with another "Name:" is refused):
  Formula:       the formula, with the numbers substituted
  Assumptions:   "- " items: the definitions and conventions the value uses
  Missing:       "- " items, only when the question does not state everything a unique answer needs
  Source:        a trusted source: a textbook, a standard, PySDR or the GNU Radio documentation
  Tolerance:     the relative tolerance and what it admits
  Not computed:  "- " items: sub-questions that ask for prose (left to the team's rubric)

Specification (one rule for every number a question asks for; `specification` in the record). Eval plan
§5.2: "A question whose correct answer depends on assumptions the question does not state is rewritten,
or removed and counted (§5.3)". So a number is either determined or not:
  determined      the question states every quantity the value depends on, or names it through a term
                  or theorem whose definition fixes it, with no rival convention in a trusted source (a
                  textbook, a standard, PySDR or the GNU Radio documentation) whose value lies outside
                  the tolerance. Assumptions name the definition and the words that select it; readings
                  (other conventions) are listed only if their value lies within the tolerance;
  underspecified  the question leaves a choice open that a rival convention in a trusted source fills
                  with a value outside the tolerance, or omits a quantity the value needs: the number has
                  no value (null), Missing names each unstated assumption, the values under the named
                  options are listed under `readings` (never as the answer), and the team rewrites or
                  removes the question (§5.2, §5.3);
  team reference  no number to compute from (see the role table above).
The record is `underspecified` exactly when a part has no value, and a part with a value lists no
reading outside its tolerance (the build enforces both). The blind recomputation checks the readings:
a team member who finds a convention the script missed marks the question underspecified too.

Tolerance (one rule). 1 % relative, or 0.1 dB for a decibel value: the scorer's defaults
(tier_k/rules.py), which admit the exact value rounded to three significant figures (0.1 dB) and
standard constants such as c = 3e8 m/s. It is widened only to admit a standard approximation named in
the Formula, or a convention for writing the value or for an input (room temperature, GB vs GiB),
whose value differs by at most 10 % (0.41 dB); a convention that differs by more is a reading that
makes the number underspecified. For a decibel quantity the tolerance is set in dB
(`tolerance_db`; `abs_tolerance` in the proposal, which rules.py uses for decibel units) and
`tolerance` is its relative equivalent on the linear quantity, 10^(dB/10) - 1.

Answer kinds: numeric (one or more numbers with units; the record's value/unit/tolerance are scalars
for one number and lists, in the question's order, for several), vector (one number per element),
symbolic and example (team references: no value). `numeric_proposal` (the reference in the form
rules.parse_reference reads) is given only for a determined numeric record whose parts the current
scorer handles; otherwise `numeric_withheld` says why. A part is withheld for the scorer when
rules.py would score an equivalent form of the value as wrong (a symbol, list or LaTeX label included),
or a value a factor 2 off as right.

Record fields (calc_answers.jsonl): question_id, question_sha256, function, calc_version, kind, title,
specification, reference_route (where the reference comes from), labels (one per number asked), value,
unit, tolerance, tolerance_db, tolerance_note, numeric_proposal, numeric_withheld, equivalents (the same
result in another form), readings (other readings of the question; `within_tolerance` null for an
undetermined part), formula, assumptions, source, underspecified, undetermined_parts,
missing_assumptions, not_computed, stated_inputs, assumed_constants, status ("pending blind team
recomputation", for every record), computed_by ("script drafted by Claude (Anthropic)"), canary.

Outputs (deterministic: sorted keys, stable order, no timestamps; a re-run is byte-identical):
  items/tier_k/test/calc_answers.jsonl                 one record per calculation question
  handoff/review_sheets/tier_k_calc_blind_recompute.csv question id and text, empty team columns only
                                                        (a template: annotators fill per-annotator copies)
  handoff/review_sheets/tier_k_calc_key.csv            the script's answers, for after the blind pass
  handoff/review_sheets/tier_k_calc_instructions.md    how to fill the blind sheet
The CSVs are UTF-8 with a byte-order mark (as labels/task_types.write_csv), so Excel shows σ, ω and μ.

`./gb calc` writes them; `--out ROOT` writes under another root; `--check` compares a fresh build with
the files on disk. A blind sheet whose team columns someone has filled in is never overwritten, and
`--check` accepts it as long as its question ids and texts are current. The default output prints
counts only, no question ids. Inputs are read-only. Nothing here calls a model or sends a question
anywhere.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import inspect
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from grcbench.common.canary import CANARY
from grcbench.common.io import read_jsonl, sha256_file, sha256_text
from grcbench.common.paths import FROZEN, REPO, SPLITS

CALC_VERSION = "tier-k-calc 3"
STATUS = "pending blind team recomputation"
COMPUTED_BY = "script drafted by Claude (Anthropic)"
# where each specification class's reference comes from (`reference_route` in the record)
ROUTES = {
    "determined": "script value, confirmed by the blind team recomputation (eval plan §5.2)",
    "underspecified": "none until the team rewrites the question, or removes and counts it (eval plan "
                      "§5.2, §5.3); the readings are not answers",
    "team reference": "rubric reference by the Table-5 drafting model (eval plan §5.2) or by the team (a "
                      "deviation to record, docs/guidelines/rubric_writing.md); no script value",
}

ITEMS_TEST = REPO / "items" / "tier_k" / "test" / "frozen_questions.jsonl"
LABELS = SPLITS / "task_type_labels.jsonl"
RAW_TEST = FROZEN / "eval_questions.jsonl"
RAW_SPELLINGS = ("calculation", "calculations")
CLEAN_TYPE = "calculation"

OUT_ANSWERS = Path("items/tier_k/test/calc_answers.jsonl")
OUT_BLIND = Path("handoff/review_sheets/tier_k_calc_blind_recompute.csv")
OUT_KEY = Path("handoff/review_sheets/tier_k_calc_key.csv")
OUT_INSTRUCTIONS = Path("handoff/review_sheets/tier_k_calc_instructions.md")
BLIND_COLUMNS = ("question_id", "question", "annotator", "value", "unit", "notes")
BOM = "﻿"

# Physical constants (exact in the 2019 SI) and the standard noise reference temperature.
C = 299_792_458.0               # speed of light, m/s
K_B = 1.380649e-23              # Boltzmann constant, J/K
T0 = 290.0                      # K, reference temperature of noise figure (IEEE; ITU-R P.372)
SIG_DIGITS = 10                 # values are stored to 10 significant digits
MAX_WIDENED = 0.10              # a tolerance is never widened beyond 10 % ...
MAX_WIDENED_DB = 0.41           # ... or 0.41 dB (10 % in power)

KINDS = ("numeric", "vector", "symbolic", "example")
VALUE_KINDS = ("numeric", "vector")         # the script computes a value from the question's numbers
TEAM_KINDS = ("symbolic", "example")        # no number to compute from: the team writes the reference
SPECIFICATIONS = ("determined", "underspecified", "team reference")
SECTIONS = ("Formula", "Assumptions", "Missing", "Source", "Tolerance", "Not computed")
LIST_SECTIONS = ("Assumptions", "Missing", "Not computed")
REQUIRED_SECTIONS = ("Formula", "Assumptions", "Source", "Tolerance")


# ---------------------------------------------------------------- answers ------------------------

@dataclass(frozen=True)
class Part:
    """One number a question asks for. value None: the question does not determine it (see Missing)."""

    label: str
    value: float | None
    unit: str = ""                                   # a spelling rules.parse_unit reads; "" = dimensionless
    tolerance: float | None = None                   # relative
    abs_db: float | None = None                      # decibel units: the tolerance in dB
    accept_units: tuple[str, ...] = ()               # spellings the scorer treats as this unit
    anchors: tuple[str, ...] = ()                    # names the scorer anchors on (questions with parts)
    readings: tuple[tuple[str, float, str], ...] = ()     # (reading, value, unit), in this part's unit
    equivalents: tuple[tuple[str, float, str], ...] = ()  # (form, value, unit): the same result
    withhold: str = ""                               # why the scorer cannot use this part (no proposal)

    def rel_tolerance(self) -> float | None:
        if self.abs_db is not None:
            return 10 ** (self.abs_db / 10) - 1
        return self.tolerance

    def within(self, x: float) -> bool:
        """Whether x lies within this part's tolerance of its value."""
        if self.value is None:
            raise ValueError(f"{self.label}: no value")
        if self.abs_db is not None:
            return abs(x - self.value) <= self.abs_db * (1 + 1e-9)
        return abs(x - self.value) <= (self.tolerance or 0.0) * abs(self.value) * (1 + 1e-9)


@dataclass(frozen=True)
class Answer:
    kind: str
    parts: tuple[Part, ...] = ()                     # numeric and vector; none for team references


@dataclass(frozen=True)
class Calc:
    qid: str
    fn: Callable[..., Answer]
    stated: tuple[str, ...]                          # substrings the question text must contain
    assumed: tuple[str, ...] = ()                    # parameters whose defaults the question does not state


REGISTRY: dict[str, Calc] = {}


def calc(qid: str, stated: tuple[str, ...], assumed: tuple[str, ...] = ()
         ) -> Callable[[Callable[..., Answer]], Callable[..., Answer]]:
    """Register the function computing question `qid`. `stated` are the numbers it takes from the text;
    `assumed` names the parameters whose defaults are constants the question does not state."""
    def wrap(fn: Callable[..., Answer]) -> Callable[..., Answer]:
        if fn.__name__ != qid or qid in REGISTRY:
            raise ValueError(f"calc function {fn.__name__} must be named after its question and be unique")
        unknown = set(assumed) - set(inspect.signature(fn).parameters)
        if unknown:
            raise ValueError(f"{qid}: assumed {sorted(unknown)} are not parameters")
        REGISTRY[qid] = Calc(qid, fn, stated, assumed)
        return fn
    return wrap


def _q(x: float) -> float:
    """Gaussian tail probability Q(x)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _db(ratio: float) -> float:
    return 10.0 * math.log10(ratio)


def _dbm(watts: float) -> float:
    return 10.0 * math.log10(watts / 1e-3)


def _sinc(x: float) -> float:
    return 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise AssertionError(f"calc self-check failed: {what}")


# ---------------------------------------------------------------- the 25 questions ---------------

@calc("q_41dd72957bec640b", stated=("16-QAM", "normalized to 1", "SNR of 20 dB"))
def q_41dd72957bec640b(m: int = 16, snr_db: float = 20.0) -> Answer:
    """Symbol error rate of square 16-QAM in AWGN at an SNR of 20 dB.

    Formula: P_s = 1 - (1 - p)^2 with p = 2 (1 - 1/sqrt(M)) Q(sqrt(3 g / (M - 1))), Q(x) = erfc(x/sqrt(2))/2
        and g = E_s/N_0 the SNR per symbol (exact for square M-QAM with ML detection); E_s/N_0 = 20 dB
        (g = 100) gives P_s = 1.16e-5; E_b/N_0 = 20 dB (E_s/N_0 = E_b/N_0 + 10 log10 4 = 26.0 dB) gives 5.6e-19.
    Assumptions:
      - coherent maximum-likelihood detection, AWGN only, perfect synchronisation; Gray labelling does not
        change the symbol error rate
    Missing:
      - which SNR the 20 dB is: the SNR per symbol E_s/N_0 (signal over noise power in the symbol-spaced
        model with unit average symbol power) or the SNR per bit E_b/N_0 (E_s/N_0 = 26.0 dB); both are in
        textbook use, the symbol error rates differ by 13 orders of magnitude, and the question itself asks
        which assumptions the answer makes
    Source: Proakis & Salehi, Digital Communications, 5th ed., §4.3 (error probability of square M-ary QAM;
        SNR per symbol and SNR per bit); Goldsmith, Wireless Communications (2005), ch. 6 (MQAM symbol
        error probability).
    Tolerance: none: the value is undetermined (see Missing). Once the SNR is fixed, 5 % relative admits Q(x)
        evaluated with the standard approximation exp(-x²/2) / (x sqrt(2π)) (+4.6 % at E_s/N_0 = 20 dB) and
        rounding to three significant figures.
    Not computed:
      - "What assumptions are you making?" (prose, for the team's rubric)
    """
    def ser(es_n0: float, q: Callable[[float], float] = _q) -> float:
        p = 2 * (1 - 1 / math.sqrt(m)) * q(math.sqrt(3 * es_n0 / (m - 1)))
        return p * (2 - p)                               # = 1 - (1 - p)^2 without cancellation at tiny p

    g = 10 ** (snr_db / 10)
    es, eb = ser(g), ser(g * math.log2(m))
    approx = ser(g, lambda x: math.exp(-x * x / 2) / (x * math.sqrt(2 * math.pi)))
    _check(abs(es / (2 * (1 - 1 / math.sqrt(m)) * 2 * _q(math.sqrt(3 * g / (m - 1)))) - 1) < 1e-4,
           "16-QAM: exact SER equals the nearest-neighbour approximation at high SNR")
    _check(0.04 < approx / es - 1 < 0.05, "16-QAM: the asymptotic Q approximation is +4.6 % (docstring)")
    _check(round(es, 7) == 1.16e-5 and f"{eb:.1e}" == "5.6e-19", "16-QAM: docstring values")
    return Answer("numeric", (Part("symbol error rate", None, "", readings=(
        ("SNR per symbol: E_s/N_0 = 20 dB", es, ""),
        ("SNR per bit: E_b/N_0 = 20 dB (E_s/N_0 = 26.0 dB)", eb, ""))),))


@calc("q_1fe7817ee6a8873f", stated=("Z = X + jY", "independent Gaussian", "mean 0", "variance σ²"))
def q_1fe7817ee6a8873f() -> Answer:
    """Distribution and statistical properties of Z = X + jY, with X and Y independent N(0, σ²).

    Formula: none: the question asks for a distribution and its properties, not a value computed from
        numbers it states; the reference is written by the team (eval plan §5.2, Table 5).
    Assumptions:
      - none (no value is computed)
    Source: none: the reference's writer names its source (docs/guidelines/rubric_writing.md).
    Tolerance: none: no value is computed.
    Not computed:
      - the whole question (team reference)
    """
    return Answer("symbolic")


@calc("q_5ed3c28aa3901b62", stated=("32-bit IEEE 754", "smallest positive normalized"), assumed=("exponent_bits",))
def q_5ed3c28aa3901b62(exponent_bits: int = 8) -> Answer:
    """Smallest positive normalized IEEE 754 binary32 number.

    Formula: bias = 2^(w-1) - 1 = 127 for w = 8 exponent bits; e_min = 1 - bias = -126; smallest normal
        number = 2^e_min = 2^-126 = 1.17549435e-38 (checked against numpy.finfo(float32).tiny).
    Assumptions:
      - "32-bit IEEE 754" is the binary32 (single-precision) format, with w = 8 exponent bits (assumed
        constant, from the standard); "normalized" excludes subnormals
    Source: IEEE Std 754-2019, §3.3 (binary32: emax = 127, emin = 1 - emax = -126); Goldberg, "What every
        computer scientist should know about floating-point arithmetic", ACM Computing Surveys 23(1), 1991.
    Tolerance: 1 % relative: the value is exact; admits rounding to three significant figures (1.18e-38).
    Not computed:
      - "explain its significance for DSP" (prose, for the team's rubric)
    """
    bias = 2 ** (exponent_bits - 1) - 1
    value = 2.0 ** (1 - bias)
    _check(value == float(np.finfo(np.float32).tiny), "binary32 smallest normal")
    return Answer("numeric", (Part("smallest positive normalized number", value, "", 0.01),))


@calc("q_bc3fb92ae337a426", stated=("stereo", "96kHz", "32-bit"))
def q_bc3fb92ae337a426(channels: int = 2, sample_rate_hz: float = 96_000.0, bits_per_sample: int = 32) -> Answer:
    """Memory bandwidth of real-time stereo audio at 96 kHz with 32-bit samples.

    Formula: stream rate R = channels x f_s x bits per sample = 2 x 96 000 x 32 = 6.144 Mbit/s (= 768 kB/s =
        750 KiB/s); a stream buffered in memory, each sample written in and read out once, makes memory
        traffic of 2R = 12.288 Mbit/s.
    Assumptions:
      - uncompressed linear PCM, 32 bits stored per sample per channel, no framing or padding
    Missing:
      - what the memory bandwidth counts: the stream's data rate (each sample crossing the memory once,
        6.144 Mbit/s) or the memory traffic of a buffer that each sample is written into and read out of
        (two transfers per sample, 12.288 Mbit/s); the question states no system architecture
    Source: data rate of linear PCM = channels x sampling frequency x word length (Pohlmann, Principles of
        Digital Audio, 6th ed., 2011); IEC 80000-13 (k = 10^3, Ki = 2^10); JEDEC JESD100B.01 (KB = 2^10 bytes
        in memory usage).
    Tolerance: none: the value is undetermined (see Missing). Once the reading is fixed, 5 % relative admits
        the value written with JEDEC binary prefixes read as decimal ones (750 KB/s, -2.3 %; 0.732 MB/s,
        -4.6 %) and rounding to three significant figures.
    Not computed:
      - "Discuss system design impact" (prose, for the team's rubric)
    """
    bits_per_s = channels * sample_rate_hz * bits_per_sample
    _check(bits_per_s / 8 == 768_000 and bits_per_s / 8 / 1024 == 750, "audio: 768 kB/s = 750 KiB/s")
    _check(round(1 - bits_per_s / 8 / 2**20 / 0.768, 3) == 0.046, "audio: JEDEC MB reading (docstring)")
    return Answer("numeric", (Part("memory bandwidth", None, "Mbit/s", readings=(
        ("the stream's data rate: one memory transfer per sample", bits_per_s / 1e6, "Mbit/s"),
        ("a buffer each sample is written into and read out of: two transfers per sample",
         2 * bits_per_s / 1e6, "Mbit/s"))),))


@calc("q_66b264c8786f2018", stated=("x(t - t0)", "X(ω)"))
def q_66b264c8786f2018() -> Answer:
    """Fourier transform Y(ω) of the time-shifted signal y(t) = x(t - t0).

    Formula: none: the question asks for an expression, not a value computed from numbers it states; the
        reference is written by the team (eval plan §5.2, Table 5).
    Assumptions:
      - none (no value is computed)
    Source: none: the reference's writer names its source (docs/guidelines/rubric_writing.md).
    Tolerance: none: no value is computed.
    Not computed:
      - the whole question (team reference)
    """
    return Answer("symbolic")


@calc("q_f7bced7a5d660e9d", stated=("3-point moving average",))
def q_f7bced7a5d660e9d() -> Answer:
    """Frequency response of the 3-point moving-average filter.

    Formula: none: the question asks for a function of frequency, not a value computed from numbers it
        states; the reference is written by the team (eval plan §5.2, Table 5).
    Assumptions:
      - none (no value is computed)
    Source: none: the reference's writer names its source (docs/guidelines/rubric_writing.md).
    Tolerance: none: no value is computed.
    Not computed:
      - the whole question (team reference)
    """
    return Answer("symbolic")


@calc("q_a598b884fbd1d1f6", stated=("QPSK", "2 bits per symbol"))
def q_a598b884fbd1d1f6() -> Answer:
    """Mapping bits to QPSK symbols, with an example mapping for 2 bits per symbol.

    Formula: none: the question asks for a method and an example, not a value computed from numbers it
        states; the reference is written by the team (eval plan §5.2, Table 5).
    Assumptions:
      - none (no value is computed)
    Source: none: the reference's writer names its source (docs/guidelines/rubric_writing.md).
    Tolerance: none: no value is computed.
    Not computed:
      - the whole question (team reference)
    """
    return Answer("example")


@calc("q_63f558e028ba13d5", stated=("bandwidth of 10 MHz", "Nyquist"))
def q_63f558e028ba13d5(bandwidth_hz: float = 10e6) -> Answer:
    """Minimum (Nyquist) sampling rate of a signal of 10 MHz bandwidth.

    Formula: real samples: f_s,min = 2 B = 2 x 10 MHz = 20 MS/s (the theorem needs f_s > 2B; 2B, the Nyquist
        rate, is the limit quoted as the minimum); complex (I/Q) samples: f_s,min = B = 10 MS/s.
    Assumptions:
      - no carrier is given, so the signal is lowpass: with real samples its 10 MHz bandwidth is its highest
        frequency; with complex samples it occupies -5 ... +5 MHz
    Missing:
      - whether the samples are real or complex (I/Q): the theorem as Shannon states it, for a real signal,
        gives 2B = 20 MS/s; complex (I/Q) samples, which SDR front ends deliver to GNU Radio, need f_s = B =
        10 MS/s for a band of width B (PySDR, "IQ Sampling"); the question does not say which
    Source: Shannon, "Communication in the presence of noise", Proc. IRE 37(1), 1949, Theorem 1; Oppenheim &
        Schafer, Discrete-Time Signal Processing, 3rd ed., ch. 4 (the sampling theorem); PySDR (M. Lichtman),
        chapter "IQ Sampling".
    Tolerance: none: the value is undetermined (see Missing). Once the reading is fixed, 1 % relative: the
        value is exact.
    """
    return Answer("numeric", (Part("minimum sampling rate", None, "MS/s", readings=(
        ("real samples: f_s = 2B", 2 * bandwidth_hz / 1e6, "MS/s"),
        ("complex (I/Q) samples: f_s = B", bandwidth_hz / 1e6, "MS/s"))),))


@calc("q_c955829cc73f83d0", stated=("4-point sequence", "[1, 0, 0, 0]"))
def q_c955829cc73f83d0(x: tuple[int, ...] = (1, 0, 0, 0)) -> Answer:
    """4-point DFT of the sequence [1, 0, 0, 0].

    Formula: X[k] = c Σ_{n=0}^{N-1} x[n] e^{-j2πkn/N} = c x[0] = c for k = 0, 1, 2, 3, so X = [c, c, c, c], with
        the normalisation factor c = 1, 1/sqrt(N) = 0.5 or 1/N = 0.25 (direct sum, checked against
        numpy.fft.fft in its three normalisation modes; every imaginary part is 0).
    Assumptions:
      - the forward DFT, with the kernel e^{-j2πkn/N}
    Missing:
      - the DFT's normalisation factor: none in Oppenheim & Schafer, numpy.fft.fft and MATLAB fft
        (X = [1, 1, 1, 1]); 1/sqrt(N) in the unitary DFT, numpy norm="ortho" and the Wolfram Language
        Fourier default (X = [0.5, 0.5, 0.5, 0.5]); 1/N in numpy norm="forward" (X = [0.25, 0.25, 0.25,
        0.25]); the question does not say which
    Source: Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed., ch. 8 (definition of the DFT);
        NumPy numpy.fft documentation (normalisation modes "backward", "ortho", "forward"); Wolfram Language
        documentation, Fourier (default FourierParameters {0, 1}: factor 1/sqrt(n)).
    Tolerance: none: the value is undetermined (see Missing). Once the normalisation is fixed, 1 % relative
        per element: every element is exact.
    Not computed:
      - "What does the result represent?" (prose, for the team's rubric)
    """
    n = len(x)
    xs = np.asarray(x, dtype=float)
    direct = [sum(x[i] * cmath.exp(-2j * math.pi * k * i / n) for i in range(n)) for k in range(n)]
    _check(bool(np.allclose(direct, np.fft.fft(xs), atol=1e-12)) and all(abs(v.imag) < 1e-12 for v in direct),
           "DFT")
    modes = (("no normalisation factor", 1.0, "backward"), ("unitary DFT, factor 1/sqrt(N)", 1 / math.sqrt(n), "ortho"),
             ("factor 1/N", 1 / n, "forward"))
    for _, c, mode in modes:
        _check(bool(np.allclose(np.fft.fft(xs, norm=mode), c * np.asarray(direct), atol=1e-12)), f"DFT: {mode}")
    return Answer("vector", tuple(
        Part(f"X[{k}]", None, "", readings=tuple((name, c * round(v.real, 12) + 0.0, "") for name, c, _ in modes))
        for k, v in enumerate(direct)))


@calc("q_75980861b0e8d95d", stated=("Cooley-Tukey FFT", "N-point"))
def q_75980861b0e8d95d() -> Answer:
    """Computational complexity of the Cooley-Tukey FFT of an N-point signal.

    Formula: none: the question asks for an order of growth in N, not a value computed from numbers it
        states; the reference is written by the team (eval plan §5.2, Table 5).
    Assumptions:
      - none (no value is computed)
    Source: none: the reference's writer names its source (docs/guidelines/rubric_writing.md).
    Tolerance: none: no value is computed.
    Not computed:
      - the whole question (team reference)
    """
    return Answer("symbolic")


@calc("q_693ffc952e4828ed", stated=("15 kHz", "5 microseconds"))
def q_693ffc952e4828ed(subcarrier_spacing_hz: float = 15e3, max_delay_spread_s: float = 5e-6) -> Answer:
    """Minimum OFDM cyclic-prefix length for a 5 μs maximum delay spread.

    Formula: T_CP,min = τ_max = 5 μs: the cyclic prefix must cover the channel's delay spread for the
        linear convolution to look circular (no inter-symbol or inter-carrier interference). The subcarrier
        spacing only sets the useful symbol length T_u = 1/Δf = 66.7 μs (CP overhead 7.5 %).
    Assumptions:
      - the CP covers exactly the maximum delay spread, with no extra margin for filter transients or
        timing error (the question asks for the minimum)
    Source: Goldsmith, Wireless Communications (2005), ch. 12 (cyclic prefix at least as long as the channel
        delay spread).
    Tolerance: 1 % relative: the value is exact.
    """
    tu = 1 / subcarrier_spacing_hz
    _check(round(tu * 1e6, 1) == 66.7 and round(max_delay_spread_s / tu, 3) == 0.075, "OFDM: Tu and overhead")
    return Answer("numeric", (Part("minimum cyclic-prefix length", max_delay_spread_s * 1e6, "μs", 0.01,
                                   equivalents=(("seconds", max_delay_spread_s, "s"),)),))


@calc("q_a12e73b62696c2fe", stated=("chip rate is 10 Mbps", "data rate is 10 kbps"))
def q_a12e73b62696c2fe(chip_rate: float = 10e6, data_rate: float = 10e3) -> Answer:
    """Processing gain of a direct-sequence spread-spectrum system.

    Formula: G_p = R_c / R_b = 10e6 / 10e3 = 1000, i.e. 10 log10(1000) = 30 dB.
    Assumptions:
      - the "10 Mbps" chip rate means 10 Mchip/s; processing gain is the chip rate over the data rate (the
        spread over the unspread bandwidth)
    Source: Sklar, Digital Communications, 2nd ed. (2001), ch. 12 (processing gain G_p = W/R); Proakis &
        Salehi, Digital Communications, 5th ed., ch. 12 (spread-spectrum signals).
    Tolerance: 0.1 dB (2.3 % on the ratio): the value is exact; 0.1 dB is the scorer's default for decibels.
    """
    ratio = chip_rate / data_rate
    return Answer("numeric", (Part("processing gain", _db(ratio), "dB", abs_db=0.1,
                                   equivalents=(("power ratio", ratio, ""),)),))


@calc("q_673b0e732a9bccad", stated=("bandwidth of 2 MHz", "real sampler", "complex sampler"))
def q_673b0e732a9bccad(bandwidth_hz: float = 2e6) -> Answer:
    """Minimum sample rate for a 2 MHz-wide signal with a real and with a complex (I/Q) sampler.

    Formula: real sampler: f_s = 2 f_max = 2 B = 4 MS/s; complex sampler: f_s = B = 2 MS/s (complex samples
        capture the band -f_s/2 ... +f_s/2, so the two-sided width B needs f_s = B).
    Assumptions:
      - no carrier frequency is given, so the signal is lowpass: real sampler, band 0 ... 2 MHz; complex
        sampler, band -1 ... +1 MHz after quadrature downconversion
      - a band at a carrier (f_L ... f_H) sampled by a real sampler needs 2 f_H / floor(f_H / B), which is
        2 B only when f_H is a whole multiple of B and more otherwise (bandpass sampling); that would need
        the carrier, which the question does not give
    Source: PySDR (M. Lichtman), chapter "IQ Sampling" (Nyquist rate; with IQ sampling the observable
        bandwidth equals the sample rate); Lyons, Understanding Digital Signal Processing, 3rd ed., ch. 2
        (bandpass sampling) and ch. 8 (quadrature signals).
    Tolerance: 1 % relative per part: both values are exact.
    """
    f_h, b = 3e6, bandwidth_hz                            # a band at 1 ... 3 MHz, for the bandpass statement
    n = np.arange(64)
    _check(bool(np.allclose(np.cos(2 * np.pi * 1.5e6 * n / (2 * b)), np.cos(2 * np.pi * 2.5e6 * n / (2 * b)))),
           "bandpass: 1-3 MHz aliases onto itself at 2B")
    _check(2 * f_h / math.floor(f_h / b) == 6e6, "bandpass: 2 fH / floor(fH / B) = 6 MS/s for 1-3 MHz")
    return Answer("numeric", (
        Part("real sampler", 2 * bandwidth_hz / 1e6, "MS/s", 0.01, anchors=(
            "real sampler", "real", "real-valued", "real sampling", "real ADC")),
        Part("complex sampler", bandwidth_hz / 1e6, "MS/s", 0.01, anchors=(
            "complex sampler", "complex", "complex-valued", "complex sampling", "I/Q", "IQ", "quadrature")),
    ))


@calc("q_976f44ddd3de3998", stated=("1 MHz bandwidth", "room temperature", "dBm"), assumed=("temperature_k",))
def q_976f44ddd3de3998(bandwidth_hz: float = 1e6, temperature_k: float = T0) -> Answer:
    """Thermal noise floor of a 1 MHz bandwidth at room temperature, in dBm.

    Formula: N = 10 log10(k T B / 1 mW) = -174.0 dBm/Hz + 10 log10(10^6) = -113.98 dBm,
        with k = 1.380649e-23 J/K, T = 290 K, B = 1 MHz.
    Assumptions:
      - room temperature is the standard noise reference T0 = 290 K (assumed constant); 293 K and 300 K are
        listed under readings (within the tolerance)
      - the "thermal noise floor" is kTB: noise figure 0 dB, B the noise bandwidth
    Source: Pozar, Microwave Engineering, 4th ed., ch. 10 (noise power kTB; T0 = 290 K); Johnson and Nyquist,
        Phys. Rev. 32 (1928).
    Tolerance: 0.2 dB (4.7 % in power): admits room temperatures from 290 K to 300 K (0.15 dB) and the
        rounding -174 dBm/Hz + 60 dB = -114 dBm.
    """
    n_dbm = _dbm(K_B * temperature_k * bandwidth_hz)
    _check(abs(_dbm(K_B * 300.0 * bandwidth_hz) - n_dbm - 0.147) < 0.001, "kTB: 290 K vs 300 K")
    return Answer("numeric", (Part(
        "thermal noise floor", n_dbm, "dBm", abs_db=0.2,
        equivalents=(("dBW", n_dbm - 30, "dBW"), ("watts", K_B * temperature_k * bandwidth_hz, "W")),
        readings=(("room temperature 293 K (20 °C)", _dbm(K_B * 293.0 * bandwidth_hz), "dBm"),
                  ("room temperature 300 K", _dbm(K_B * 300.0 * bandwidth_hz), "dBm"))),))


@calc("q_d00f22f579b94d32", stated=("3 GHz", "10 km", "30 dB antenna gain", "SNR of 20 dB", "10 MHz"))
def q_d00f22f579b94d32(freq_hz: float = 3e9, distance_m: float = 10e3, gain_db: float = 30.0,
                       snr_db: float = 20.0, bandwidth_hz: float = 10e6) -> Answer:
    """Transmit power for a 20 dB received SNR on a 3 GHz, 10 km link with 30 dB antennas at each end.

    Formula: P_t = SNR + N + FSPL - G_t - G_r, with N = 10 log10(k T0 B / 1 mW) + NF and
        FSPL = 20 log10(4π d f / c) = 121.99 dB; with NF = 0 dB and no margin:
        P_t = 20 - 103.98 + 121.99 - 60 = -21.98 dBm (6.3 μW).
    Assumptions:
      - free-space propagation (line of sight, first Fresnel zone clear), isotropic-referenced gains, T0 = 290 K
    Missing:
      - the receiver noise figure (or system noise temperature): each dB of noise figure adds 1 dB to P_t
      - any fade, implementation or feeder-loss margin (none is stated; each adds dB for dB)
    Source: ITU-R Recommendation P.525 (free-space attenuation); Friis, "A note on a simple transmission
        formula", Proc. IRE 34(5), 1946; Pozar, Microwave Engineering, 4th ed., ch. 10 (kTB noise).
    Tolerance: none: the value is undetermined (see Missing).
    """
    nf_db = 0.0                                          # the named ideal reading only; the question gives none
    fspl = 20 * math.log10(4 * math.pi * distance_m * freq_hz / C)
    noise = _dbm(K_B * T0 * bandwidth_hz) + nf_db
    pt = snr_db + noise + fspl - 2 * gain_db
    p525 = 32.45 + 20 * math.log10(freq_hz / 1e6) + 20 * math.log10(distance_m / 1e3)
    _check(abs(fspl - p525) < 0.01, "FSPL equals ITU-R P.525 (32.45 + 20 log f_MHz + 20 log d_km)")
    _check((round(fspl, 2), round(noise, 2), round(pt, 2)) == (121.99, -103.98, -21.98)
           and round(10 ** (pt / 10) * 1e3, 1) == 6.3, "link budget: docstring values")
    return Answer("numeric", (Part("transmit power", None, "dBm", readings=(
        ("noise figure 0 dB, T0 = 290 K, free space, no losses or margin", pt, "dBm"),)),))


@calc("q_9849c620793a77be", stated=("clipped at 3 dB above its average power", "approximate PAR"))
def q_9849c620793a77be(clip_db: float = 3.0) -> Answer:
    """Approximate peak-to-average power ratio of a signal clipped 3 dB above its average power.

    Formula: PAR = P_peak / P_avg ≈ P_clip / P_avg = 3 dB (power ratio 10^0.3 ≈ 2).
    Assumptions:
      - "approximate": the clip level is taken as the peak power and the average power as unchanged by the
        clipping (clipping lowers it by an amount that depends on the signal's amplitude distribution, which
        the question does not give)
    Source: Goldsmith, Wireless Communications (2005), ch. 12 (peak-to-average power ratio, PAR = max |x|² /
        E|x|²); Proakis & Salehi, Digital Communications, 5th ed., ch. 11 (PAR of multicarrier signals).
    Tolerance: 0.1 dB (2.3 % in power): the scorer's default for decibels; admits the power ratio 2 (3.01 dB).
    """
    gamma2 = 10 ** (clip_db / 10)                     # clip level over average power (linear)
    _check(abs(_db(2.0) - clip_db) < 0.1, "PAR: a power ratio of 2 is within 0.1 dB of 3 dB")
    return Answer("numeric", (Part("peak-to-average power ratio", clip_db, "dB", abs_db=0.1,
                                   equivalents=(("power ratio", gamma2, ""),)),))


@calc("q_ff18653bc3179694", stated=("rate-1/2", "d_free = 10", "BER of 1e-5"))
def q_ff18653bc3179694(rate: float = 0.5, d_free: int = 10, ber: float = 1e-5) -> Answer:
    """Coding gain of a rate-1/2 convolutional code with d_free = 10 at a BER of 1e-5.

    Formula: asymptotic coding gain G = 10 log10(R d_free) = 10 log10(5) = 6.99 dB for soft-decision
        decoding and 10 log10(R d_free / 2) = 3.98 dB for hard-decision decoding.
    Assumptions:
      - BPSK/QPSK on AWGN, maximum-likelihood (Viterbi) decoding
    Missing:
      - soft- or hard-decision decoding (6.99 dB vs 3.98 dB)
      - whether the asymptotic gain is meant: the gain at a stated BER of 1e-5 also depends on the code's
        distance spectrum (the number of paths at d_free and beyond), which the question does not give
    Source: Proakis & Salehi, Digital Communications, 5th ed., ch. 8 (convolutional codes: coding gain
        R d_free for soft and R d_free / 2 for hard decisions); Sklar, Digital Communications, 2nd ed., ch. 7.
    Tolerance: none: the value is undetermined (see Missing).
    """
    soft, hard = _db(rate * d_free), _db(rate * d_free / 2)
    return Answer("numeric", (Part("coding gain", None, "dB", readings=(
        ("asymptotic, soft-decision decoding", soft, "dB"), ("asymptotic, hard-decision decoding", hard, "dB"))),))


@calc("q_eabcdb1b438a5afb", stated=("100 Hz frequency offset", "15 kHz subcarrier spacing"))
def q_eabcdb1b438a5afb(offset_hz: float = 100.0, spacing_hz: float = 15e3) -> Answer:
    """Inter-carrier interference power in OFDM for a 100 Hz frequency offset and 15 kHz spacing.

    Formula: normalised offset ε = Δf_off / Δf = 100 / 15000 = 6.67e-3. With independent unit-power symbols,
        a subcarrier whose neighbours on both sides are loaded (every subcarrier of a fully loaded symbol)
        receives P_ICI = 1 - sinc²(ε) ≈ (πε)²/3 = 1.462e-4 = -38.35 dB relative to the signal power (N → ∞;
        checked against the direct finite-N sum for N = 64 and 1024); a subcarrier at the edge of the loaded
        band receives about half, (πε)²/6 = -41.36 dB (direct sum, N = 1024 with 600 loaded subcarriers);
        Moose's upper bound 0.5947 sin²(πε) is 2.61e-4 = -35.84 dB.
    Assumptions:
      - all loaded subcarriers carry independent symbols of equal power; many subcarriers (N ≥ 64 changes
        the result by less than 0.01 dB)
    Missing:
      - the reference power: no signal power is given, so the ICI power can only be stated relative to the
        signal power (in dB, as a power ratio or as a percentage)
      - which subcarrier, and how many subcarriers are loaded: one with loaded neighbours on both sides
        receives -38.35 dB, one at the edge of the loaded band about -41.36 dB
      - which expression is meant: the ICI power of this model or Moose's upper bound on it, -35.84 dB, which
        is also quoted as the ICI power
    Source: Goldsmith, Wireless Communications (2005), ch. 12 (frequency offset in OFDM: subcarrier attenuation
        and inter-carrier interference); Pollet, Van Bladel & Moeneclaey, "BER sensitivity of OFDM systems to
        carrier frequency offset and Wiener phase noise", IEEE Trans. Commun. 43(2/3/4), 1995 (ICI power
        (πε)²/3 for small ε); Moose, "A technique for orthogonal frequency division multiplexing frequency
        offset correction", IEEE Trans. Commun. 42(10), 1994 (upper bound 0.5947 sin²(πε) |X|²).
    Tolerance: none: the value is undetermined (see Missing). Once the reading is fixed, 0.1 dB (2.3 % in
        power), the scorer's default for decibels: the small-offset approximation and any N ≥ 64 move the
        value by less than 0.01 dB.
    """
    eps = offset_hz / spacing_hz
    ici = 1 - _sinc(eps) ** 2
    _check(abs(ici / ((math.pi * eps) ** 2 / 3) - 1) < 1e-3, "ICI: small-offset approximation")

    def leak(n: int, k: int) -> float:
        """|coefficient|² from the subcarrier k places away (n-point symbol, offset ε)."""
        return (math.sin(math.pi * eps) / (n * math.sin(math.pi * (k + eps) / n))) ** 2

    for n in (64, 1024):
        _check(abs(_db(sum(leak(n, k) for k in range(1, n))) - _db(ici)) < 0.01, f"ICI: direct sum N={n}")
    edge = ici / 2
    n, loaded = 1024, 600
    for side in (1, -1):                                  # the lowest and the highest loaded subcarrier
        _check(abs(_db(sum(leak(n, side * k) for k in range(1, loaded))) - _db(edge)) < 0.1, "ICI: band edge")
    moose = 0.5947 * math.sin(math.pi * eps) ** 2
    _check((round(_db(ici), 2), round(_db(edge), 2), round(_db(moose), 2)) == (-38.35, -41.36, -35.84),
           "ICI: docstring values")
    return Answer("numeric", (Part("ICI power relative to the signal power", None, "dB", readings=(
        ("a subcarrier with loaded neighbours on both sides (fully loaded symbol): (πε)²/3", _db(ici), "dB"),
        ("a subcarrier at the edge of the loaded band: about (πε)²/6", _db(edge), "dB"),
        ("Moose's upper bound 0.5947 sin²(πε) (1994)", _db(moose), "dB"))),))


@calc("q_ccf5d9913ed192e4", stated=("FMCW", "0.1 m range resolution"))
def q_ccf5d9913ed192e4(range_resolution_m: float = 0.1) -> Answer:
    """Sweep bandwidth an FMCW radar needs for a 0.1 m range resolution.

    Formula: ΔR = c / (2B), so B = c / (2 ΔR) = 299 792 458 / 0.2 = 1.499 GHz (1.5 GHz with c = 3e8 m/s).
    Assumptions:
      - propagation at c in free space; resolution set by the swept bandwidth alone (no windowing widening)
    Source: Richards, Fundamentals of Radar Signal Processing, 2nd ed. (2014), range resolution ΔR = c/(2B);
        Rao, "Introduction to mmWave Sensing: FMCW Radars", Texas Instruments (2017).
    Tolerance: 1 % relative: admits c = 3e8 m/s (+0.07 %) and rounding to three significant figures.
    """
    return Answer("numeric", (Part("sweep bandwidth", C / (2 * range_resolution_m) / 1e9, "GHz", 0.01),))


@calc("q_94a0c101dbf3ee99", stated=("interpolates by 7", "decimates by 5", "1 MS/s", "0-400 kHz"))
def q_94a0c101dbf3ee99(interp: int = 7, decim: int = 5, fs_in: float = 1e6, f_max: float = 400e3) -> Answer:
    """Output rate of a 7/5 rational resampler at 1 MS/s, and its prototype filter cutoff.

    Formula: f_out = f_in L / M = 1 MS/s x 7/5 = 1.4 MS/s. The prototype lowpass runs at L f_in = 7 MS/s. To
        keep the 0-400 kHz signal its passband must reach 400 kHz; to stop the images its stopband must begin
        by the first image, at f_in - 400 kHz = 600 kHz; and a cutoff below 600 kHz is below the output
        Nyquist frequency f_out/2 = 700 kHz, so nothing aliases. Any ideal cutoff from 400 kHz to just below
        600 kHz therefore stops images and aliases (checked by simulation: scipy upfirdn, Kaiser FIR). The
        general rule for a rational resampler, cutoff min(f_in, f_out)/2 = 500 kHz (π/max(L, M) = π/7
        rad/sample at 7 MS/s), is for an input that fills 0-500 kHz; here it is also the middle of the
        400-600 kHz transition band.
    Assumptions:
      - an ideal L-fold zero-stuffing interpolator, one lowpass filter, then M-fold decimation
    Missing:
      - which cutoff the "minimum required cutoff" is: the smallest passband edge that keeps the signal
        (400 kHz), the general min(f_in, f_out)/2 rule (500 kHz) or the limit the first image sets (just
        below 600 kHz); imaging and aliasing, the constraints the question names, bound the cutoff from
        above, not from below
    Source: Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed., §4.6 (changing the sampling rate by a
        non-integer factor: cutoff min(π/L, π/M), gain L); GNU Radio 3.10 rational_resampler block documentation.
    Tolerance: output rate 1 % relative (the value is exact); the cutoff has none: undetermined (see Missing).
    """
    from scipy import signal
    fs_out, fs_up = fs_in * interp / decim, fs_in * interp
    rule, image = min(fs_in, fs_out) / 2, fs_in - f_max
    _check(math.isclose(rule / fs_up, 1 / (2 * max(interp, decim))) and image == 600e3 and image < fs_out / 2,
           "resampler: min(f_in, f_out)/2 is pi/max(L, M) at 7 MS/s; first image at 600 kHz, below 700 kHz")
    tones = np.array([50e3, 200e3, 330e3, 390e3])        # inside the 0-400 kHz band
    x = np.cos(2 * np.pi * tones[:, None] * np.arange(4096)[None, :] / fs_in).sum(axis=0)

    def worst_spur_db(cutoff_hz: float) -> float:
        """Largest output component away from the tones, relative to the strongest tone (dBc)."""
        taps, beta = signal.kaiserord(80.0, 20e3 / (fs_up / 2))
        h = interp * signal.firwin(taps | 1, cutoff_hz, window=("kaiser", beta), fs=fs_up)
        start = len(h) // decim + 1                       # past the start-up transient, before the end one
        y = signal.upfirdn(h, x, up=interp, down=decim)[start:start + 4096]
        _check(start + 4096 < len(x) * interp // decim, "resampler: steady-state output only")
        spec = np.abs(np.fft.rfft(y * np.kaiser(len(y), 16))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / fs_out)
        away = np.all(np.abs(f[:, None] - tones[None, :]) > 4e3, axis=1)
        return _db(spec[away].max() / spec.max())

    _check(all(worst_spur_db(c) < -70 for c in (420e3, rule, 570e3)) and worst_spur_db(640e3) > -10,
           "resampler: cutoffs 420-570 kHz stop images and aliases; 640 kHz lets the 610 kHz image through")
    return Answer("numeric", (
        Part("output sample rate", fs_out / 1e6, "MS/s", 0.01, anchors=("output sample rate", "output rate")),
        Part("prototype filter cutoff", None, "kHz", anchors=("cutoff",), readings=(
            ("smallest passband edge that keeps the 0-400 kHz signal", f_max / 1e3, "kHz"),
            ("general rational-resampler rule min(f_in, f_out)/2, for an input filling 0-500 kHz", rule / 1e3, "kHz"),
            ("limit set by the first image at f_in - 400 kHz: ideal cutoffs just below it stop images and aliases",
             image / 1e3, "kHz"))),
    ))


@calc("q_7acc67b2a3b45407", stated=("100 MHz", "75 kHz", "15 kHz", "99% power"))
def q_7acc67b2a3b45407(deviation_hz: float = 75e3, message_bw_hz: float = 15e3) -> Answer:
    """Carson's-rule bandwidth of FM with 75 kHz deviation and a 15 kHz message, and the 99 % power bandwidth.

    Formula: B_C = 2 (Δf + W) = 2 (75 + 15) kHz = 180 kHz (= 2 (β + 1) W with β = Δf / W = 5). The carrier
        frequency does not enter. For single-tone modulation the spectrum has lines at f_c ± n W with powers
        J_n(β)²; the 99 % bandwidth is 2 n W for the smallest n with J_0² + 2 Σ_{k=1..n} J_k² ≥ 0.99.
    Assumptions:
      - Carson's rule with the peak deviation and the highest message frequency W = 15 kHz
    Missing:
      - the message waveform for the 99 % power bandwidth: a 15 kHz-wide message is not a tone, and the
        99 % bandwidth of FM depends on the message's spectrum. For a single 15 kHz tone (β = 5), 6 sideband
        pairs hold 99.4 % of the power, so B_99 = 180 kHz, the same as Carson's rule.
    Source: Proakis & Salehi, Communication Systems Engineering, 2nd ed. (2002), angle modulation (Carson's
        rule, Bessel-function spectrum of FM); Haykin, Communication Systems, 4th ed., ch. 2 (FM bandwidth).
    Tolerance: Carson bandwidth 1 % relative (the value is exact); the 99 % bandwidth has none: undetermined
        (see Missing).
    """
    from scipy.special import jv
    beta = deviation_hz / message_bw_hz
    carson = 2 * (deviation_hz + message_bw_hz)
    power = [jv(0, beta) ** 2]
    for n in range(1, 40):
        power.append(power[-1] + 2 * jv(n, beta) ** 2)
    _check(abs(power[-1] - 1) < 1e-12, "FM: Bessel powers sum to 1")
    n99 = next(n for n, p in enumerate(power) if p >= 0.99)
    _check(n99 == 6 and round(power[6] * 100, 1) == 99.4, "FM: 6 sideband pairs hold 99.4 % (docstring)")
    return Answer("numeric", (
        Part("Carson bandwidth", carson / 1e3, "kHz", 0.01, anchors=("Carson",)),
        Part("99 % power bandwidth", None, "kHz", anchors=("99%", "99 %"), readings=(
            ("single-tone modulation at 15 kHz (β = 5)", 2 * n99 * message_bw_hz / 1e3, "kHz"),)),
    ))


# rules.normalize turns LaTeX commands it does not know into spaces, so "$\Delta R = 1.5$ m" loses its label
RADAR_LATEX_WITHHOLD = (
    "a range resolution labelled in LaTeX (\"$\\Delta R = 1.5$ m\") loses its label in rules.py (tier-k-rules "
    "0.3 and 0.4, 18 Oct 2026: normalize drops \\Delta), which then reads the unambiguous range (150 km) as the range "
    "resolution, an order-of-magnitude error that zeroes the whole answer; the anchors here already decide "
    "plain-text symbols (ΔR, TB, R_max) correctly")


@calc("q_487f23bc5c3635f0", stated=("bandwidth 100 MHz", "10 μs", "1 ms"))
def q_487f23bc5c3635f0(bandwidth_hz: float = 100e6, pulse_s: float = 10e-6, pri_s: float = 1e-3) -> Answer:
    """Range resolution, time-bandwidth product and maximum unambiguous range of an LFM pulse radar.

    Formula: ΔR = c / (2B) = 1.499 m; TB = B τ = 100e6 x 10e-6 = 1000; R_ua = c PRI / 2 = 149.9 km
        (1.5 m and 150 km with c = 3e8 m/s).
    Assumptions:
      - the range resolution of the chirp after pulse compression (matched filtering), c / (2B), as defined
        for a frequency-modulated pulse; free-space propagation at c
    Source: Richards, Fundamentals of Radar Signal Processing, 2nd ed. (2014) (ΔR = c/(2B), unambiguous range
        c PRI / 2, LFM time-bandwidth product); Skolnik, Introduction to Radar Systems, 3rd ed., ch. 1.
    Tolerance: 1 % relative per part: admits c = 3e8 m/s (+0.07 %) and rounding to three significant figures.
    """
    return Answer("numeric", (
        Part("range resolution", C / (2 * bandwidth_hz), "m", 0.01,
             anchors=("range resolution", "resolution", "ΔR", "δR", "Δr", "δr"),
             withhold=RADAR_LATEX_WITHHOLD),
        Part("time-bandwidth product", bandwidth_hz * pulse_s, "", 0.01,
             anchors=("time-bandwidth product", "time-bandwidth", "time bandwidth product", "time bandwidth",
                      "TBP", "TB", "BT", "Bτ", "τB", "B·τ", "B*τ", "B×τ")),
        Part("maximum unambiguous range", C * pri_s / 2 / 1e3, "km", 0.01,
             anchors=("maximum unambiguous range", "unambiguous range", "max unambiguous range", "maximum range",
                      "max range", "R_ua", "R_unamb", "R_unambiguous", "R_max", "Rmax", "R_u")),
    ))



@calc("q_25f9f7406f3df97e", stated=("decimation=4", "1 GB", "100 MS/s", "complex samples"),
      assumed=("bytes_per_item",))
def q_25f9f7406f3df97e(file_bytes: float = 1e9, bytes_per_item: int = 8, rate_sps: float = 100e6,
                       decimation: int = 4) -> Answer:
    """Run time of a headless GNU Radio flowgraph reading a 1 GB complex file at 100 MS/s.

    Formula: N = 1e9 B / 8 B per gr_complex = 1.25e8 samples; t = N / 100 MS/s = 1.25 s. The decimation by 4
        only shrinks the output (3.125e7 samples, 250 MB); the rate applies to the input samples.
    Assumptions:
      - GNU Radio's complex stream item is gr_complex = std::complex<float>, 8 bytes (assumed constant);
        1 GB = 10^9 bytes
        (1 GiB = 2^30 bytes is listed under readings, within the tolerance)
      - the 100 MS/s processing rate is the bottleneck (disk reads and writes keep up); no Throttle block
    Source: GNU Radio 3.10 manual: gr_complex (gnuradio/gr_complex.h) and the File Source/File Sink blocks
        (raw items, no header); IEC 80000-13 (G = 10^9, Gi = 2^30).
    Tolerance: 8 % relative: admits 1 GB read as 1 GiB (+7.4 %) and rounding to three significant figures.
    Not computed:
      - "What GNU Radio mechanism ensures the flowgraph runs as fast as possible?" (prose, for the team's rubric)
    """
    samples = file_bytes / bytes_per_item
    t = samples / rate_sps
    gib = 2**30 / bytes_per_item / rate_sps
    _check(round(gib, 2) == 1.34 and round(gib / t - 1, 3) == 0.074, "GNU Radio file: GiB reading (docstring)")
    _check(samples / decimation == 3.125e7, "GNU Radio file: output samples")
    return Answer("numeric", (Part("run time", t, "s", 0.08, readings=(("1 GB read as 1 GiB = 2^30 bytes", gib, "s"),)),))


@calc("q_cf42443d85802560", stated=("77 GHz", "bandwidth of 4 GHz"))
def q_cf42443d85802560(carrier_hz: float = 77e9, bandwidth_hz: float = 4e9) -> Answer:
    """Theoretical range resolution of a 77 GHz radar with 4 GHz bandwidth.

    Formula: ΔR = c / (2B) = 299 792 458 / 8e9 = 0.03747 m (3.75 cm with c = 3e8 m/s); the 77 GHz carrier
        does not enter.
    Assumptions:
      - propagation at c in free space; resolution set by the swept bandwidth alone
    Source: Richards, Fundamentals of Radar Signal Processing, 2nd ed. (2014), range resolution ΔR = c/(2B);
        Rao, "Introduction to mmWave Sensing: FMCW Radars", Texas Instruments (2017).
    Tolerance: 1 % relative: admits c = 3e8 m/s (+0.07 %) and rounding to three significant figures.
    """
    return Answer("numeric", (Part("range resolution", C / (2 * bandwidth_hz), "m", 0.01),))


@calc("q_072c5b5c4eb8bcc9", stated=("EIRP of 30 dBm", "transmitter power of 20 dBm"))
def q_072c5b5c4eb8bcc9(eirp_dbm: float = 30.0, tx_dbm: float = 20.0) -> Answer:
    """Antenna gain for an EIRP of 30 dBm from a 20 dBm transmitter.

    Formula: EIRP = P_t + G_t - L_feed, so G_t = 30 dBm - 20 dBm = 10 dBi.
    Assumptions:
      - no feeder or cable loss between the transmitter and the antenna (none is stated)
      - a gain written in dB is the same number in dBi (EIRP is referenced to an isotropic antenna)
    Source: ITU Radio Regulations, Article 1 (definition of equivalent isotropically radiated power, e.i.r.p.:
        power supplied to the antenna times its gain relative to an isotropic antenna); Sklar, Digital
        Communications, 2nd ed., ch. 5 (link budget, EIRP = P_t G_t).
    Tolerance: 0.1 dB (2.3 % in power): the value is exact; 0.1 dB is the scorer's default for decibels.
    """
    return Answer("numeric", (Part("antenna gain", eirp_dbm - tx_dbm, "dBi", abs_db=0.1, accept_units=("dB",)),))


# ---------------------------------------------------------------- docstrings -> record fields ----

_HEADER = re.compile(rf"^({'|'.join(SECTIONS)}):\s*(.*)$")
_ANY_HEADER = re.compile(r"^([A-Z][A-Za-z -]*):(?:\s|$)")     # a line that opens a section


def doc_sections(fn: Callable[..., Any]) -> dict[str, Any]:
    """Parse a calc function's docstring into its sections (lists for the "- " item sections). A line
    that opens a section other than SECTIONS is refused, so nothing is kept that the record does not copy."""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    out: dict[str, Any] = {"title": lines[0].strip() if lines else ""}
    current: str | None = None
    buf: dict[str, list[str]] = {}
    for line in lines[1:]:
        m = _HEADER.match(line)
        other = _ANY_HEADER.match(line)
        if other and not m:
            raise ValueError(f"{fn.__name__}: unknown docstring section {other.group(1)!r}")
        if m:
            current = m.group(1)
            if current in buf:
                raise ValueError(f"{fn.__name__}: section {current} twice")
            buf[current] = [m.group(2).strip()] if m.group(2).strip() else []
            continue
        if current is None or not line.strip():
            continue
        text = line.strip()
        if current in LIST_SECTIONS and text.startswith("- "):
            buf[current].append("\0" + text[2:].strip())
        elif buf[current]:
            buf[current][-1] += " " + text
        else:
            buf[current].append(text)
    for name, chunks in buf.items():
        if name in LIST_SECTIONS:
            items = [c.lstrip("\0") for c in chunks]
            if any(not c.startswith("\0") for c in chunks):
                raise ValueError(f"{fn.__name__}: section {name} must be a list of '- ' items")
            out[name] = items
        else:
            out[name] = " ".join(chunks)
    for name in REQUIRED_SECTIONS:
        if not out.get(name):
            raise ValueError(f"{fn.__name__}: docstring has no {name} section")
    for name in LIST_SECTIONS:
        out.setdefault(name, [])
    return out


# ---------------------------------------------------------------- records ------------------------

def _r(x: float | None) -> float | None:
    """Round to SIG_DIGITS significant digits (stable output across platforms); None stays None."""
    if x is None:
        return None
    return float(f"{x:.{SIG_DIGITS}g}") + 0.0


def inputs_of(calc_: Calc) -> tuple[dict[str, Any], dict[str, Any]]:
    """A function's keyword defaults, split into (stated in the question, assumed constants)."""
    stated: dict[str, Any] = {}
    assumed: dict[str, Any] = {}
    for name, p in inspect.signature(calc_.fn).parameters.items():
        v = p.default
        if v is inspect.Parameter.empty:
            raise ValueError(f"{calc_.qid}: parameter {name} has no default (the question's number)")
        (assumed if name in calc_.assumed else stated)[name] = list(v) if isinstance(v, tuple) else v
    return stated, assumed


def _numeric_ref(p: Part, with_label: bool) -> dict[str, Any]:
    """The part as a reference rules.parse_reference reads."""
    ref: dict[str, Any] = {"value": _r(p.value), "unit": p.unit, "tolerance": _r(p.rel_tolerance())}
    if p.abs_db is not None:
        ref["abs_tolerance"] = p.abs_db
    if p.accept_units:
        ref["accept_units"] = list(p.accept_units)
    if with_label and p.anchors:
        ref["label"] = list(p.anchors)
    return ref


def evaluate(qid: str) -> Answer:
    """Run the function of question `qid` on the numbers stated in the question."""
    ans = REGISTRY[qid].fn()
    if ans.kind not in KINDS:
        raise ValueError(f"{qid}: unknown answer kind {ans.kind!r}")
    if (ans.kind in VALUE_KINDS) != bool(ans.parts):
        raise ValueError(f"{qid}: numeric and vector answers have parts, team references none")
    for p in ans.parts:
        if p.value is not None and p.rel_tolerance() is None:
            raise ValueError(f"{qid}: {p.label} has a value but no tolerance")
        if p.value is None and (p.tolerance is not None or p.abs_db is not None or p.equivalents or p.withhold):
            raise ValueError(f"{qid}: {p.label} has no value, so no tolerance, equivalents or withhold")
        if any(u != p.unit for _, _, u in p.readings):
            raise ValueError(f"{qid}: {p.label}: readings are given in the part's unit")
        if p.value is not None and any(not p.within(v) for _, v, _ in p.readings):
            raise ValueError(f"{qid}: {p.label}: a reading outside the tolerance leaves the part undetermined: "
                             "give it no value and a Missing section (module docstring, Specification)")
        widened = (p.abs_db or 0.0) > MAX_WIDENED_DB or (p.abs_db is None and (p.tolerance or 0.0) > MAX_WIDENED)
        if widened:
            raise ValueError(f"{qid}: {p.label}: tolerance wider than 10 % (0.41 dB)")
    return ans


def answer_value(ans: Answer) -> Any:
    """The record's `value`: a number, a list of numbers (several parts or a vector), or None."""
    if ans.kind not in VALUE_KINDS:
        return None
    vals = [_r(p.value) for p in ans.parts]
    return vals[0] if len(vals) == 1 else vals


def specification(ans: Answer) -> str:
    """The specification class of an answer (module docstring, "Specification")."""
    if ans.kind in TEAM_KINDS:
        return "team reference"
    if any(p.value is None for p in ans.parts):
        return "underspecified"
    return "determined"


def withheld(ans: Answer, spec: str) -> str | None:
    """Why the record has no numeric_proposal (None: it has one)."""
    if spec == "team reference":
        return "team reference: no script value"
    if spec == "underspecified":
        return "underspecified: no value until the team rewrites the question"
    if ans.kind != "numeric":
        return f"a {ans.kind} answer has no scalar reference form"
    reasons = [f"{p.label}: {p.withhold}" for p in ans.parts if p.withhold]
    return "; ".join(reasons) if reasons else None


def record(q: dict[str, Any], calc_: Calc) -> dict[str, Any]:
    """The calc_answers.jsonl record of one question."""
    ans = evaluate(calc_.qid)
    doc = doc_sections(calc_.fn)
    parts = ans.parts
    spec = specification(ans)
    team = spec == "team reference"
    undetermined = [p.label for p in parts if p.value is None]
    if bool(undetermined) != bool(doc["Missing"]):
        raise ValueError(f"{calc_.qid}: a part without a value needs a Missing section, and only then")
    stated_inputs, assumed_constants = inputs_of(calc_)
    if team and (stated_inputs or assumed_constants or not doc["Formula"].startswith("none: ")
                 or not doc["Source"].startswith("none: ")):
        raise ValueError(f"{calc_.qid}: a team reference computes nothing and names no formula or source")
    single = len(parts) == 1
    if parts:
        unit: Any = parts[0].unit if single else [p.unit for p in parts]
        tols = [_r(p.rel_tolerance()) if p.value is not None else None for p in parts]
        dbs = [p.abs_db if p.value is not None else None for p in parts]
        tolerance: Any = tols[0] if single else tols
        tolerance_db: Any = dbs[0] if single else dbs
    else:
        unit, tolerance, tolerance_db = "", None, None
    reason = withheld(ans, spec)
    proposal = None
    if reason is None:
        refs = [_numeric_ref(p, with_label=not single) for p in parts]
        proposal = refs[0] if single else refs
    return {
        "question_id": calc_.qid,
        "question_sha256": sha256_text(q["question"]),
        "function": f"{__name__}.{calc_.qid}",
        "calc_version": CALC_VERSION,
        "kind": ans.kind,
        "title": doc["title"],
        "specification": spec,
        "reference_route": ROUTES[spec],
        "labels": [p.label for p in parts],
        "value": answer_value(ans),
        "unit": unit,
        "tolerance": tolerance,
        "tolerance_db": tolerance_db,
        "tolerance_note": doc["Tolerance"],
        "numeric_proposal": proposal,
        "numeric_withheld": reason,
        "equivalents": [{"part": p.label, "form": f, "value": _r(v), "unit": u}
                        for p in parts for f, v, u in p.equivalents],
        "readings": [{"part": p.label, "reading": r, "value": _r(v), "unit": u,
                      "within_tolerance": None if p.value is None else p.within(v)}
                     for p in parts for r, v, u in p.readings],
        "formula": doc["Formula"],
        "assumptions": doc["Assumptions"],
        "source": doc["Source"],
        "underspecified": spec == "underspecified",
        "undetermined_parts": undetermined,
        "missing_assumptions": doc["Missing"],
        "not_computed": doc["Not computed"],
        "stated_inputs": stated_inputs,
        "assumed_constants": assumed_constants,
        "status": STATUS,
        "computed_by": COMPUTED_BY,
        "canary": CANARY,
    }


# ---------------------------------------------------------------- inputs -------------------------

def load_questions() -> list[dict[str, Any]]:
    """The calculation test questions in frozen-split order, after the three-way cross-check."""
    items = read_jsonl(ITEMS_TEST)
    labels = {r["id"]: r for r in read_jsonl(LABELS) if r["split"] == "eval"}
    raw = {r["id"]: r for r in read_jsonl(RAW_TEST)}
    problems = []
    if len(items) != 185 or len(labels) != 185 or len(raw) != 185:
        problems.append(f"expected 185 test questions, got items {len(items)}, labels {len(labels)}, raw {len(raw)}")
    by_items = [r for r in items if r["task_type"] == CLEAN_TYPE]
    ids_items = {r["id"] for r in by_items}
    ids_labels = {i for i, r in labels.items() if r["clean"] == CLEAN_TYPE}
    ids_raw = {i for i, r in raw.items() if r["task_type"] in RAW_SPELLINGS}
    if not ids_items == ids_labels == ids_raw:
        problems.append(f"calculation sets differ: items {len(ids_items)}, labels {len(ids_labels)}, raw {len(ids_raw)}")
    for r in by_items:
        src = raw.get(r["id"])
        if src is None or src["question"] != r["question"] or src["task_type"] != r["task_type_raw"]:
            problems.append(f"{r['id']}: item differs from the frozen dataset")
    if ids_items != set(REGISTRY):
        problems.append(f"no function for {sorted(ids_items - set(REGISTRY))}; "
                        f"function without question {sorted(set(REGISTRY) - ids_items)}")
    for r in by_items:
        c = REGISTRY.get(r["id"])
        missing = [s for s in (c.stated if c else ()) if s not in r["question"]]
        if missing:
            problems.append(f"{r['id']}: question text lacks {missing}")
    if problems:
        raise SystemExit("calc: " + "; ".join(problems))
    return by_items


# ---------------------------------------------------------------- outputs ------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{SIG_DIGITS}g}"
    if isinstance(v, list):
        return "; ".join("(undetermined)" if x is None else _fmt(x) for x in v)
    return str(v)


def _csv(header: tuple[str, ...] | list[str], rows: list[list[str]]) -> str:
    """CSV text with a byte-order mark (Excel reads it as UTF-8) and LF line ends."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return BOM + buf.getvalue()


def blind_sheet(questions: list[dict[str, Any]]) -> str:
    """Question id and text with empty team columns: nothing the script computed."""
    return _csv(BLIND_COLUMNS, [[q["id"], q["question"], "", "", "", ""] for q in questions])


def read_sheet(text: str) -> list[list[str]]:
    """Rows of a CSV sheet, with or without a byte-order mark."""
    return list(csv.reader(io.StringIO(text.removeprefix(BOM))))


def annotated(text: str) -> bool:
    """Whether someone has filled in a team column of a blind sheet."""
    return any(any(c.strip() for c in row[2:]) for row in read_sheet(text)[1:])


def same_questions(a: str, b: str) -> bool:
    """Whether two blind sheets hold the same header, question ids and question texts."""
    rows_a, rows_b = read_sheet(a), read_sheet(b)
    return bool(rows_a) and rows_a[0] == rows_b[0] and [r[:2] for r in rows_a] == [r[:2] for r in rows_b]


KEY_COLUMNS = ("question_id", "question", "kind", "specification", "reference_route", "labels", "value", "unit",
               "tolerance", "tolerance_db", "tolerance_note", "readings", "equivalents", "missing_assumptions",
               "formula", "assumptions", "source", "not_computed", "status", "computed_by")


def key_sheet(questions: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    """The script's answers, one row per question, for comparison after the blind pass."""
    text = {q["id"]: q["question"] for q in questions}
    rows = []
    for r in records:
        unit = r["unit"] if isinstance(r["unit"], str) else "; ".join(r["unit"])
        rows.append([
            r["question_id"], text[r["question_id"]], r["kind"], r["specification"], r["reference_route"],
            "; ".join(r["labels"]),
            _fmt(r["value"]), unit, _fmt(r["tolerance"]), _fmt(r["tolerance_db"]), r["tolerance_note"],
            " | ".join(f"{c['part']}: {c['reading']}: {_fmt(c['value'])} {c['unit']}".rstrip()
                       + ("" if c["within_tolerance"] is None else
                          " (within tolerance)" if c["within_tolerance"] else " (outside tolerance)")
                       for c in r["readings"]),
            " | ".join(f"{e['part']}: {e['form']}: {_fmt(e['value'])} {e['unit']}".rstrip() for e in r["equivalents"]),
            " | ".join(r["missing_assumptions"]), r["formula"], " | ".join(r["assumptions"]), r["source"],
            " | ".join(r["not_computed"]), r["status"], r["computed_by"],
        ])
    return _csv(KEY_COLUMNS, rows)


def instructions(n: int) -> str:
    return f"""# Tier-K calculation answers: blind recomputation

Eval plan §5.2: calculation answers are computed by a script from the question's numbers, and a team
member recomputes each one **without seeing the script's answer**. The script was drafted by Claude
(Anthropic), so every answer is "{STATUS}" (role table, Table 5).
The script gives a value only where it computes one from numbers the question states. A question
whose answer is an expression or an example has no script value: its reference is written
separately (`reference_route` in the key).

1. Copy `tier_k_calc_blind_recompute.csv` ({n} questions) to
   `tier_k_calc_blind_recompute_<your initials>.csv` and fill in your copy only: the original is a
   generated template. Until you have finished, do **not** open `tier_k_calc_key.csv`,
   `items/tier_k/test/calc_answers.jsonl`, or any file under `src/grcbench/tier_k/` or `tests/`:
   they hold the script's answers.
2. For each question, fill in `annotator` (your name), `value` and `unit`. Use only the numbers the
   question states plus standard constants. If the question asks for several numbers, give them in
   the question's order, separated by `;` (same for `unit`). If the answer is an expression or an
   example rather than a number, write it in `notes` and leave `value` empty. Leave out the
   explanations the question asks for: those go into the rubric, which is written separately.
3. In `notes`, write any assumption you had to make and the source you checked (textbook, standard,
   PySDR or the GNU Radio documentation). If the question does not state everything needed for a
   unique answer, do not guess: write `underspecified:`, the missing assumption and the answer under
   each option, and leave `value` empty. This includes a term with two definitions in use (in a
   textbook, a standard, PySDR or the GNU Radio documentation) that give different answers.
4. Then open `tier_k_calc_key.csv` and compare. A value agrees when it lies within the key's relative
   `tolerance` (decibel values: within `tolerance_db` dB). The `specification` column says how the
   script classed the question:
   - `determined`: one value;
   - `underspecified`: no value; `missing_assumptions` names what the question leaves open, and the
     values under each named option are under `readings` (they are not answers);
   - `team reference`: no script value (the answer is not a number computed from the question).
   The `tolerance_note` says only what the tolerance admits. The errors that make an answer wrong
   belong to the rubric, which the team writes.
5. The Qoherent expert decides every disagreement between the script and a recomputation, including
   one about whether a question is underspecified. An underspecified question is rewritten, or
   removed and counted (eval plan §5.2, §5.3). Record in your copy's `notes` who decided and what.
6. The team then writes the rubric's `numeric` and `numeric_tolerance` from the confirmed value
   (`docs/guidelines/rubric_writing.md`). The script's `numeric_proposal` in `calc_answers.jsonl` is
   only a starting point.
"""


@dataclass
class Build:
    questions: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(default_factory=list)

    def outputs(self) -> dict[Path, str]:
        """Every output, as the exact text written (UTF-8; the CSVs start with a byte-order mark)."""
        jsonl = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in self.records)
        return {OUT_ANSWERS: jsonl, OUT_BLIND: blind_sheet(self.questions),
                OUT_KEY: key_sheet(self.questions, self.records), OUT_INSTRUCTIONS: instructions(len(self.questions))}


def build() -> Build:
    questions = load_questions()
    return Build(questions, [record(q, REGISTRY[q["id"]]) for q in questions])


def _read(path: Path) -> str | None:
    return path.read_bytes().decode("utf-8") if path.exists() else None


def write(b: Build, root: Path) -> tuple[dict[str, str], list[str]]:
    """Write every output under `root`; return ({relative path: sha256}, kept annotated sheets).

    A blind sheet whose team columns are filled in is kept, never overwritten; if its questions are not
    the current ones the build stops instead."""
    hashes, kept = {}, []
    for rel, text in b.outputs().items():
        path = root / rel
        old = _read(path)
        if rel == OUT_BLIND and old is not None and annotated(old):
            if not same_questions(old, text):
                raise SystemExit(f"calc: {rel.as_posix()} is filled in but its questions are out of date; "
                                 "move it aside and run ./gb calc again")
            kept.append(rel.as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        hashes[rel.as_posix()] = sha256_file(path)
    return hashes, kept


def check(b: Build, root: Path) -> list[str]:
    """Outputs under `root` that differ from a fresh build (empty: up to date). A filled-in blind sheet
    is up to date when its header, question ids and question texts are."""
    stale = []
    for rel, text in b.outputs().items():
        old = _read(root / rel)
        if old == text:
            continue
        if rel == OUT_BLIND and old is not None and annotated(old) and same_questions(old, text):
            continue
        stale.append(rel.as_posix())
    return stale


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb calc", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    ap.add_argument("--check", action="store_true", help="compare a fresh build with the files on disk")
    args = ap.parse_args(argv)
    b = build()
    if args.check:
        stale = check(b, args.out)
        for rel in stale:
            print(f"stale: {rel}")
        print("calc outputs are up to date" if not stale else f"{len(stale)} calc output(s) differ: run ./gb calc")
        return 1 if stale else 0
    hashes, kept = write(b, args.out)
    specs = {s: sum(r["specification"] == s for r in b.records) for s in SPECIFICATIONS}
    proposals = sum(r["numeric_proposal"] is not None for r in b.records)
    print(f"calculation test questions {len(b.records)}: " + ", ".join(f"{k} {v}" for k, v in specs.items())
          + f"; numeric proposals {proposals}; status: {STATUS}")
    for rel, h in hashes.items():
        print(f"  {rel:55s} sha256 {h[:16]}" + ("  (filled in: kept, not overwritten)" if rel in kept else ""))
    return 0
