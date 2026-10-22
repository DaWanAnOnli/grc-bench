"""Authoring script for GRC-Bench Tier-F item gbf-test-11 (modify, L1, composed chain).

Re-themed on 18 Oct (fix plan G5): the old item (retune a DAC test tone, parameter-tune) shared the
cell filter|tone with gbf-test-10 and overlapped gbf-test-04/-12. The slot becomes a modify L1 item
because gbf-test-06 moved to parameter-tune (plan R6). New theme: add automatic gain control.

Start graph (48 kHz, complex): a Signal Source (cosine, 1.2 kHz, amplitude 0.5) is multiplied by a
repeating Vector Source envelope (4000 samples at 0.1, then 4000 samples at 3.0) and written to a
File Sink rx.dat, so rx.dat alternates between weak (0.05) and strong (1.5) bursts, a 29.5 dB step
every 4000 samples. The user asks for automatic gain control so that every burst has RMS 1.0
(+-10 %) from 240 samples (5 ms) after each level change on, without changing the sources.

Feasibility (scratch simulation in GNU Radio 3.10.12, 18 Oct). analog_agc_xx updates
gain += rate * (reference - |out|), so its time constant is 1 / (rate * input level): settling the
weak bursts (0.05) needs rate >= ~0.15 for the stated 5 ms, and the strong bursts (1.5) diverge for
rate >= ~0.8. The fix plan's "1 ms" was infeasible for analog_agc_xx (best ~74 samples at rate 0.6),
so the item states 5 ms. Measured RMS (first 240 samples after the settling time / rest of the
stretch), weak bursts: rate 0.1 0.836/0.995 (fails), 0.15 0.928/0.999, 0.2 0.968/1.000,
0.3 0.993/1.000 (reference), 0.7 1.000/1.000, 0.8 inf (diverges); AGC3 0.1/0.01 0.968/1.000;
AGC2 0.5/0.2 0.967/1.000; feed-forward AGC (any num_samples < 240) 0.929 (its |x| estimate
r + 0.4 i overestimates the magnitude by up to 8 %); magnitude / moving average / divide 1.000.
A single RMS over the whole stretch would not see the settling time (rate 0.1 gives 0.986), hence
the separate first-240-samples window.

V6 oracle on rx.dat (6 stretches, 3 envelope periods; stretch k starts at sample 4000 k):
  f2p  rms over [4000k + 240, 4000k + 480)    1.0 +- 10 %   (settled within 5 ms)
  f2p  rms over [4000k + 480, 4000k + 4000)   1.0 +- 10 %   (stays there)
  p2p  peak_freq over [4000k + 240, 4000k + 4000)   +1200 Hz +- 1 %   (94 periods: on an FFT bin)
  p2p  band_fraction 1150-1250 Hz, same window      >= 0.99
  graph checks:
  p2p  Signal Source unchanged (complex, 1200 Hz, amplitude 0.5) and Vector Source unchanged
       (complex, the same envelope) -- without them, flattening the envelope would pass every
       signal check
  p2p  a path from the Signal Source to a File Sink
  f2p  a path from the Vector Source to a File Sink through a gain-control block: AGC, AGC2, AGC3,
       Feed-Forward AGC, a Divide (normalisers) or an Embedded Python block (a custom AGC)

Review round 2 (finding G5-test11-train-hashed-alternatives): the solutions the gain-control check
accepts do not all share the reference's v2 signature, because the class map puts the AGC blocks in
class agc but a Divide by a level estimate in divide + the estimator's classes. Every such design
that passes the oracle is a variant here, so that compose.audit_items puts its signature on the
reserved list (5 of the 8 accepted signatures are not test, 4 hash to train):
  agc|data_source|mix                                 test   reference, v1-v3 (AGC, AGC2, AGC3, FF AGC)
  data_source|divide|envelope|filter|mix              test   v4  |x| -> moving average -> Divide
  data_source|divide|mix|power_measure                train  v5  RMS block -> Divide
  data_source|divide|envelope|filter|mix|nonlinear    train  v6  |x|^2 -> moving average -> sqrt -> Divide
  data_source|divide|envelope|mix|nonlinear           train  v7  sqrt(|x|^2) -> Divide (no averaging)
  data_source|divide|envelope|mix                     dev    v8  x / |x|
  custom|data_source|mix                              train  v9  Embedded Python feedback AGC
(|x| -> single-pole IIR -> Divide and RMS -> IIR -> Divide hash to test.) Measured RMS in the first
window of the weak bursts: v5 0.9998, v6-v8 1.000, v9 0.993 (the same loop as the reference AGC).

Deterministic; regenerate with (inside WSL, from the repository root):

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-11/build.py

Needs GNU Radio (catalog defaults are used to write the compact flowgraphs).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from grcbench.common.canary import CANARY             # noqa: E402
from grcbench.tier_f import compose as C              # noqa: E402
from grcbench.tier_f import grc, grcfile, taskio      # noqa: E402

TASK_ID = "gbf-test-11"
SPLIT = "test"

# ---- the specification (every number below is stated in the instruction) -----------------------
FS = 48000                 # sample rate, Hz
F_TONE = 1200              # tone frequency, Hz (40 samples per period)
A_TONE = 0.5               # Signal Source amplitude
ENV_WEAK, ENV_STRONG = 0.1, 3.0
STRETCH = 4000             # samples per envelope level
ENVELOPE = f"[{ENV_WEAK}] * {STRETCH} + [{ENV_STRONG}] * {STRETCH}"
OUT_FILE = "rx.dat"
SETTLE = 240               # 5 ms at 48 kHz
FIRST = 240                # the first window after the settling time (another 5 ms)
TARGET_RMS = 1.0
RMS_REL = 0.10
FREQ_REL = 0.01
PURITY, PURITY_HW = 0.99, 50
N_STRETCH = 6              # 3 envelope periods
N_ITEMS = N_STRETCH * STRETCH

AGC_RATE = 0.3             # reference AGC rate (weak bursts settle in ~150 samples)

GAIN_CONTROL_IDS = ["analog_agc_xx", "analog_agc2_xx", "analog_agc3_xx", "analog_feedforward_agc_cc",
                    "blocks_divide_xx", "epy_block"]

INSTRUCTION = (
    "This flowgraph runs at a sample rate of 48 kHz and simulates a receiver recording with weak and "
    "strong bursts: a complex Signal Source makes a 1.2 kHz tone with amplitude 0.5, a repeating Vector "
    "Source supplies the burst envelope (4000 samples at 0.1, then 4000 samples at 3.0), the two are "
    "multiplied, and the product is written to a File Sink that stores complex samples in rx.dat. So "
    "the level in rx.dat jumps between 0.05 and 1.5 every 4000 samples, counted from its first sample. "
    "Add automatic gain control before the File Sink so that, from 240 samples (5 ms) after every level "
    "change until the next change, rx.dat has an RMS level of 1.0 within 10 %, both over the first 240 "
    "samples of that stretch and over the rest of it. Do not change the Signal Source or the Vector "
    "Source, and keep the envelope multiplying the tone. Keep the 1.2 kHz tone: in each of those "
    "stretches the strongest component must be at +1.2 kHz (within 1 %), with at least 99 % of the "
    "power within 50 Hz of it. The flowgraph should keep producing samples until it is stopped; do not "
    "limit the number of samples."
)


# ---- start and reference --------------------------------------------------------------------------
def _sources() -> list[dict]:
    return [C.sig_source("tone", "complex", freq=F_TONE, amp=A_TONE),
            C.block("envelope", "blocks_vector_source_x", type="complex", vector=ENVELOPE, repeat="True")]


def _graph(chain: list[dict], extra_conns: list[list[str]] | None = None, extra_blocks: list[dict] | None = None) -> dict:
    """tone -> mult:0, envelope -> mult:1, mult -> <chain...> -> rx_sink."""
    blocks = _sources() + [C.block("bursts", "blocks_multiply_xx", type="complex")] + chain + \
        (extra_blocks or []) + [C.file_sink("rx_sink", "complex", OUT_FILE)]
    conns = [["tone", "0", "bursts", "0"], ["envelope", "0", "bursts", "1"]]
    prev = "bursts"
    for b in chain:
        conns.append([prev, "0", b["name"], "0"])
        prev = b["name"]
    conns.append([prev, "0", "rx_sink", "0"])
    return C.graph("burst_agc", blocks, conns + (extra_conns or []), {"samp_rate": str(FS)})


def start_full() -> dict:
    return _graph([])


def reference_full() -> dict:
    return _graph([C.block("agc", "analog_agc_xx", type="complex", rate=str(AGC_RATE), reference="1.0",
                           gain="1.0", max_gain="65536")])


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, f"compact form differs from the full form: {diffs}"
    return out


# ---- equivalent variants (genuinely different correct solutions) --------------------------------
def variant_agc2() -> dict:
    """AGC2 with separate attack and decay rates."""
    return _graph([C.block("agc2", "analog_agc2_xx", type="complex", attack_rate="0.5", decay_rate="0.3",
                           reference="1.0", gain="1.0", max_gain="65536")])


def variant_agc3() -> dict:
    """AGC3 (gain IIR towards reference / |x|), attack 0.1, decay 0.05."""
    return _graph([C.block("agc3", "analog_agc3_xx", type="complex", attack_rate="0.1", decay_rate="0.05",
                           reference="1.0", gain="1.0", max_gain="65536", iir_update_decim="1")])


def variant_feedforward() -> dict:
    """Feed-Forward AGC over a 48-sample (1 ms) look-ahead window (output delayed by 47 samples)."""
    return _graph([C.block("ffagc", "analog_feedforward_agc_cc", num_samples="48", reference="1.0")])


def _divide_normaliser(level: list[dict]) -> dict:
    """bursts / level(bursts): `level` is a chain from the bursts to a float level estimate, which a
    Float To Complex turns into the divisor of a complex Divide (bursts on input 0, level on input 1)."""
    blocks = level + [C.block("to_c", "blocks_float_to_complex"), C.block("norm", "blocks_divide_xx", type="complex")]
    conns = [["tone", "0", "bursts", "0"], ["envelope", "0", "bursts", "1"]]
    prev = "bursts"
    for b in level:
        conns.append([prev, "0", b["name"], "0"])
        prev = b["name"]
    conns += [[prev, "0", "to_c", "0"], ["bursts", "0", "norm", "0"], ["to_c", "0", "norm", "1"],
              ["norm", "0", "rx_sink", "0"]]
    return C.graph("burst_agc",
                   _sources() + [C.block("bursts", "blocks_multiply_xx", type="complex")] + blocks
                   + [C.file_sink("rx_sink", "complex", OUT_FILE)], conns, {"samp_rate": str(FS)})


def variant_normaliser() -> dict:
    """Divide the signal by its own magnitude, averaged over 48 samples (1 ms): a normalising AGC."""
    return _divide_normaliser([C.block("mag", "blocks_complex_to_mag"),
                               C.block("avg", "blocks_moving_average_xx", type="float", length="48",
                                       scale="1.0 / 48")])


# Round 2 (finding G5-test11-train-hashed-alternatives): Divide-based normalisers whose v2 signature
# differs from the reference's. Each passes every check, and three of them hash to train, so they are
# listed as variants to put their signatures on the reserved list (compose.audit_items reserves the
# signatures of a test task's reference and variants).
def _sqrt_block() -> dict:
    """Transcendental block computing sqrt (its function parameter is called 'name', like a block's name)."""
    return {"name": "sq", "id": "blocks_transcendental", "parameters": {"type": "float", "name": "sqrt"}}


def variant_rms_normaliser() -> dict:
    """Divide by the RMS block's output (alpha 0.05): data_source|divide|mix|power_measure (train)."""
    return _divide_normaliser([C.block("rms", "blocks_rms_xx", type="complex", alpha="0.05")])


def variant_power_sqrt_normaliser() -> dict:
    """Divide by sqrt of the 48-sample average of |x|^2 (a moving RMS):
    data_source|divide|envelope|filter|mix|nonlinear (train)."""
    return _divide_normaliser([C.block("msq", "blocks_complex_to_mag_squared"),
                               C.block("avg", "blocks_moving_average_xx", type="float", length="48",
                                       scale="1.0 / 48"),
                               _sqrt_block()])


def variant_instant_power_sqrt() -> dict:
    """Divide by sqrt(|x|^2) sample by sample (no averaging): data_source|divide|envelope|mix|nonlinear (train)."""
    return _divide_normaliser([C.block("msq", "blocks_complex_to_mag_squared"),
                               _sqrt_block()])


def variant_magnitude_limiter() -> dict:
    """Divide by |x| sample by sample (x / |x|): data_source|divide|envelope|mix (dev). Shown in review
    round 2 (probe limiter_x_over_absx); for a constant-envelope burst it is an AGC with no lag."""
    return _divide_normaliser([C.block("mag", "blocks_complex_to_mag")])


EPY_AGC_SOURCE = '''"""Feedback AGC: gain += rate * (reference - |output|), like GNU Radio's AGC block."""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Feedback AGC: gain += rate * (reference - |output|), like GNU Radio's AGC block."""

    def __init__(self, rate=0.3, reference=1.0):
        gr.sync_block.__init__(self, name="Feedback AGC", in_sig=[np.complex64], out_sig=[np.complex64])
        self.rate = rate
        self.reference = reference
        self._g = 1.0

    def work(self, input_items, output_items):
        g, out = self._g, []
        for x in input_items[0].tolist():
            y = x * g
            out.append(y)
            g = min(g + self.rate * (self.reference - abs(y)), 65536.0)
        output_items[0][:] = out
        self._g = g
        return len(output_items[0])
'''


def variant_epy_agc() -> dict:
    """An Embedded Python feedback AGC (the custom AGC the gain-control path check accepts):
    custom|data_source|mix (train)."""
    return _graph([{"name": "epy_agc", "id": "epy_block",
                    "parameters": {"_source_code": EPY_AGC_SOURCE, "rate": "0.3", "reference": "1.0"}}])


# ---- task record ---------------------------------------------------------------------------------
def record(sig: tuple[str, ...], start_sig: tuple[str, ...], variant_sigs: dict[str, tuple]) -> dict:
    spec = {
        "settled": ("from 240 samples (5 ms) after every level change, the first 240 samples of the stretch have "
                    "RMS 1.0 within 10 %"),
        "level": "the rest of every stretch (480 samples after the change until the next change) has RMS 1.0 within 10 %",
        "tone": "the strongest component of every stretch (from 240 samples after the change) is at +1.2 kHz within 1 %",
        "pure": "in every stretch (from 240 samples after the change), >= 99 % of the power lies within 1150-1250 Hz",
        "sources_kept": ("the Signal Source (complex, 1.2 kHz, amplitude 0.5) and the Vector Source (complex, envelope "
                         f"{ENVELOPE}) are unchanged, and the tone reaches the File Sink"),
        "agc": ("automatic gain control: the bursts reach the File Sink through a gain-control block (AGC, AGC2, AGC3, "
                "Feed-Forward AGC, a Divide-based normaliser or an Embedded Python block)"),
        "rate": "sample rate 48 kHz (frequencies are measured on the sample index at 48 kHz)",
        "observation": ("complex samples written to the File Sink rx.dat (sink type complex), running continuously "
                        f"(V5 liveness: at least {N_ITEMS} items, 3 envelope periods)"),
    }
    checks = []
    for k in range(N_STRETCH):
        s0 = k * STRETCH
        kind = "weak" if k % 2 == 0 else "strong"
        checks += [
            {"point": "rx", "feature": "rms", "skip": s0 + SETTLE, "n": FIRST, "expect": TARGET_RMS,
             "rel_tol": RMS_REL, "group": "f2p", "property": "settled", "stretch": f"{k} ({kind})",
             "requires": spec["settled"]},
            {"point": "rx", "feature": "rms", "skip": s0 + SETTLE + FIRST, "n": STRETCH - SETTLE - FIRST,
             "expect": TARGET_RMS, "rel_tol": RMS_REL, "group": "f2p", "property": "level",
             "stretch": f"{k} ({kind})", "requires": spec["level"]},
            {"point": "rx", "feature": "peak_freq", "fs": FS, "skip": s0 + SETTLE, "n": STRETCH - SETTLE,
             "expect": F_TONE, "tol": FREQ_REL * F_TONE, "group": "p2p", "property": "tone",
             "stretch": f"{k} ({kind})", "requires": spec["tone"]},
            {"point": "rx", "feature": "band_fraction", "fs": FS, "lo": F_TONE - PURITY_HW, "hi": F_TONE + PURITY_HW,
             "skip": s0 + SETTLE, "n": STRETCH - SETTLE, "min": PURITY, "group": "p2p", "property": "pure",
             "stretch": f"{k} ({kind})", "requires": spec["pure"]},
        ]
    graph_checks = [
        {"kind": "block_present", "ids": ["analog_sig_source_x"], "min": 1,
         "params": {"type": "complex", "freq": str(F_TONE), "amp": str(A_TONE)}, "group": "p2p",
         "property": "sources_kept", "note": "the Signal Source is unchanged"},
        {"kind": "block_present", "ids": ["blocks_vector_source_x"], "min": 1,
         "params": {"type": "complex", "vector": ENVELOPE}, "group": "p2p", "property": "sources_kept",
         "note": "the burst envelope is unchanged (flattening it would pass every signal check)"},
        {"kind": "path", "from_ids": ["analog_sig_source_x"], "to_ids": ["blocks_file_sink"], "group": "p2p",
         "property": "sources_kept", "note": "the tone still reaches a File Sink"},
        {"kind": "path", "from_ids": ["blocks_vector_source_x"], "to_ids": ["blocks_file_sink"],
         "via_ids": GAIN_CONTROL_IDS, "group": "f2p", "property": "agc",
         "note": "the bursts reach a File Sink through a gain-control block"},
    ]
    return {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": (
            "Re-themed 18 Oct (fix plan G5); relabelled modify (the old item was parameter-tune, plan R6: adding "
            "a block is a structural change). Old round-1 findings are moot except F1 (stated structural rules "
            "unchecked), which the new graph checks address for this item's 'do not change the sources' rule. "
            "Deviation from the plan's theme numbers: settling time 5 ms instead of 1 ms (1 ms is infeasible for "
            "analog_agc_xx with a 29.5 dB step; see build.py), and a 1.2 kHz / amplitude 0.5 tone with a 0.1 / "
            "3.0 envelope (same 0.05 / 1.5 burst levels) so the kept-source graph checks can name non-default "
            "parameter values. Known oracle limits: (1) 'automatic' is checked structurally (a path through an "
            "AGC-type block, a Divide or an Embedded Python block); dividing by a copy of the known Vector Source "
            "envelope would pass although it is not adaptive. (2) Only the first 240 samples after the settling "
            "time and the rest of each stretch are measured, not every 5 ms sub-window. (3) The kept-source "
            "graph checks compare parameter strings (numeric spellings such as 1.2e3 match; a re-written but "
            "equivalent envelope expression would not). (4) Writing a fresh constant tone to rx.dat while the "
            "original chain feeds another File Sink would pass (contrived). Round 2 (finding "
            "G5-test11-train-hashed-alternatives, accepted): Divide-based normalisers and an Embedded Python AGC do "
            "the AGC's job but get other v2 signatures (the class map puts a Divide by a level estimate in divide + "
            "the estimator's classes, not in agc), and four of them hash to train; every accepted design found is "
            "now a variant (v4-v9), so its signature is reserved. For the team (tooling, outside this item): "
            "(a) compose's class map should treat a Divide by a level estimate (RMS, |x| or |x|^2 through an "
            "average and/or sqrt) as agc, and the handoff note's claim that swapping in an equivalent block never "
            "turns a non-train graph into a train graph does not hold for these; (b) the reserved check is exact, "
            "so a superset of a reserved signature (e.g. the reference plus a short Delay, agc|data_source|delay|mix, "
            "train) is not caught; (c) the non-adaptive shortcut in limit (1), dividing by the Vector Source, has "
            "signature data_source|divide|mix (train) and passes the oracle, but is not a correct answer, so it is "
            "not a variant and therefore not reserved.")},
        "family": "modify", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": "start.grc",
        "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "rx", "sink": {"block_id": "blocks_file_sink",
                                               "params": {"file": OUT_FILE, "type": "complex"}}}],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 10.0},
            "contract": "the complex samples written to the File Sink whose file name is rx.dat",
        },
        "oracle": {"checks": checks, "graph_checks": graph_checks},
        "spec_properties": list(spec.values()),
        "mutations": [
            {"op": "set_param", "block": "agc", "param": "rate", "value": "1e-4", "breaks": spec["settled"],
             "note": "GRC's default rate: the gain needs seconds to follow a burst"},
            {"op": "set_param", "block": "agc", "param": "rate", "value": "0.1", "breaks": spec["settled"],
             "note": "near miss: weak bursts settle in ~450 samples (RMS 0.84 in the first window)"},
            {"op": "set_param", "block": "agc", "param": "rate", "value": "0.8", "breaks": spec["level"],
             "note": "too fast for the strong bursts: the gain loop diverges"},
            {"op": "set_param", "block": "agc", "param": "reference", "value": "0.5", "breaks": spec["level"]},
            {"op": "set_param", "block": "agc", "param": "reference", "value": "1.15", "breaks": spec["level"],
             "note": "near miss: 15 % high"},
            {"op": "set_param", "block": "agc", "param": "max_gain", "value": "10", "breaks": spec["level"],
             "note": "the weak bursts need gain 20 and stay at 0.5"},
            {"op": "bypass", "block": "agc", "breaks": spec["agc"], "note": "no gain control (the start behaviour)"},
            {"op": "set_param", "block": "tone", "param": "freq", "value": "1300", "breaks": spec["tone"]},
            {"op": "set_param", "block": "tone", "param": "amp", "value": "1.0", "breaks": spec["sources_kept"],
             "note": "the AGC hides the change: only the kept-source graph check sees it"},
            {"op": "set_param", "block": "envelope", "param": "vector", "value": "[1.0]",
             "breaks": spec["sources_kept"], "note": "bursts flattened: every signal check passes, the graph check fails"},
            {"op": "set_param", "block": "samp_rate", "param": "value", "value": "96000", "breaks": spec["rate"]},
        ],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Adding a fast automatic gain control to a burst recording (weak/strong envelope) under a settling-time "
            "spec. Nearest training-side specs (reports/reference/training_task_specs.json): t19_rssi_meter_build "
            "and t08_run_verify_probe measure a level (moving average / probe) but never control a gain; GRC-Agent "
            "scenario 06_query_knowledge_multiply multiplies two tones. No training spec uses an AGC block. "
            f"v2 signatures: reference {'|'.join(sig)} -> {SPLIT}; start {'|'.join(start_sig)} -> "
            f"{C.signature_split(start_sig)}; variants " + "; ".join(
                f"{k} {'|'.join(s)} -> {C.signature_split(s)}" for k, s in variant_sigs.items()) + ". "
            "Signatures to reserve for this item (reference and variants, compose.signature_digest): " + "; ".join(
                f"{'|'.join(s)} {C.signature_digest(s)}" for s in sorted({sig, *variant_sigs.values()})) + "."
        ),
        "notes": (
            "Reference: analog_agc_xx rate 0.3 (weak bursts settle in ~150 samples, strong in ~10; overshoot up "
            "to |x| = 30 in the first samples of a strong burst, which the instruction leaves unconstrained). "
            "Variants: AGC2 (0.5 / 0.3), AGC3 (0.1 / 0.05), Feed-Forward AGC (48 samples; RMS 0.929 because its "
            "magnitude estimate is approximate; output delayed by 47 samples, within the 240-sample settling "
            "allowance), and Divide-based normalisers and a custom AGC, each a genuine structural alternative "
            "listed for the reserved set: v4 |x| / 48-sample moving average / Divide "
            "(data_source|divide|envelope|filter|mix, test); v5 RMS block (alpha 0.05) / Divide "
            "(data_source|divide|mix|power_measure, train); v6 |x|^2 / moving average / sqrt / Divide "
            "(data_source|divide|envelope|filter|mix|nonlinear, train); v7 sqrt(|x|^2) / Divide without averaging "
            "(data_source|divide|envelope|mix|nonlinear, train); v8 x / |x| (data_source|divide|envelope|mix, dev); "
            "v9 Embedded Python feedback AGC, the reference's loop in Python (custom|data_source|mix, train). "
            "v5 and v6 are the designs of the round-2 finding, v8 the round-2 probe limiter_x_over_absx; v7 and v9 "
            "were found by the fixer in the same search."
        ),
    }


def main() -> int:
    start = compact(start_full())
    ref = compact(reference_full())
    variants = {
        "v1_agc2": compact(variant_agc2()),
        "v2_agc3": compact(variant_agc3()),
        "v3_feedforward_agc": compact(variant_feedforward()),
        "v4_magnitude_normaliser": compact(variant_normaliser()),
        "v5_rms_normaliser": compact(variant_rms_normaliser()),
        "v6_power_sqrt_normaliser": compact(variant_power_sqrt_normaliser()),
        "v7_instant_power_sqrt": compact(variant_instant_power_sqrt()),
        "v8_magnitude_limiter": compact(variant_magnitude_limiter()),
        "v9_epy_agc": compact(variant_epy_agc()),
    }
    sig, start_sig = C.signature(ref), C.signature(start)
    assert C.signature_split(sig) == SPLIT, f"reference signature {sig} -> {C.signature_split(sig)}, not {SPLIT}"
    variant_sigs = {k: C.signature(v) for k, v in variants.items()}
    print("v2 signatures:")
    print(f"  start      {'|'.join(start_sig)} -> {C.signature_split(start_sig)}")
    print(f"  reference  {'|'.join(sig)} -> {C.signature_split(sig)}")
    for k, s in variant_sigs.items():
        print(f"  {k:24s} {'|'.join(s)} -> {C.signature_split(s)}")
    files = {"start.grc": grcfile.dump(start), "reference.grc": grcfile.dump(ref)}
    files.update({f"variants/{k}.grc": grcfile.dump(v) for k, v in variants.items()})
    errors = taskio.save(HERE, record(sig, start_sig, variant_sigs), files)
    if errors:
        print("NOT WRITTEN:", *errors, sep="\n  ")
        return 1
    for stale in (HERE / "variants").glob("*.grc"):     # variants no longer produced by this script
        if f"variants/{stale.name}" not in files:
            stale.unlink()
    print(f"written {HERE.relative_to(REPO)} (signature {'|'.join(sig)} -> {SPLIT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
