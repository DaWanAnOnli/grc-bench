"""Authoring script for GRC-Bench Tier-F item gbf-test-10 (repair, L1, fault: nonexistent_block).

Deterministic: re-running it rewrites task.json, reference.grc, start.grc and variants/*.grc
byte-identically (and removes variant files it no longer writes). Run inside WSL from the repository root:

    PYTHONPATH=src ~/grc-agent/.venv/bin/python items/tier_f/test/gbf-test-10/build.py

The item: a composed L1 chain (Signal Source -> High Pass Filter -> Multiply Const -> File Sink)
that removes the DC offset from a 1.2 kHz test tone and scales it to a peak of 1.0. In the starting
flowgraph the filter's block id is replaced by 'filter_high_pass_filter', a plausible but
nonexistent id (it is the catalog *file* name filter_high_pass_filter.block.yml; the block id
inside it is 'high_pass_filter'). GRC loads the block as a "Missing Block", so the start fails at
rung V2. Drafted by Claude (Anthropic); pending team review.

Round-1 fixes (18 Oct 2026, reports/tier_f_reviews/round1/gbf-test-10.resolution.json):
  * the kept tone source is observed through an edge point on block 'tone' (no extra block) with p2p
    checks, and a graph check pins its stated parameters (a consistent rescaling of the sample rate
    and the tone frequency gives identical samples, so only the parameters can show it);
  * the instruction requires the DC offset to be removed by a filter block (graph check: a path from
    the Signal Source through a filter block to the File Sink; every filter block id is accepted);
    the 300 Hz / 200 Hz design is context, not a requirement (it is not measurable without a hum
    source, and a hum source would move the v2 cell to combine|filter|tone = train);
  * the tolerances and the 1024-sample start-up allowance are stated; the run-length sentence added;
  * groups follow fix-plan rule R8: the start fails at V2, so p2p = properties of the parts the fault
    does not touch (the tone source), and every output check is f2p (it depends on the repaired block);
  * the reviewers' passing alternatives are variants (FFT filter, Kaiser window with an absolute
    output path, first-order IIR high-pass).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # .../grc-bench
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY                      # noqa: E402
from grcbench.tier_f import compose as C, edit, grc, grcfile, taskio   # noqa: E402

TASK_ID = "gbf-test-10"
SPLIT = "test"
FS = 32000            # sample rate (Hz)
TONE_F = 1200         # tone frequency (Hz)
TONE_AMP = 0.4        # tone amplitude at the source
DC_OFFSET = 0.75      # DC offset of the source signal
HPF_CUTOFF = 300      # high-pass cutoff (Hz): context only
HPF_WIDTH = 200       # high-pass transition width (Hz): context only
GAIN = 2.5            # Multiply Const value
OUT_FILE = "clean_tone.dat"
WRONG_ID = "filter_high_pass_filter"
FAULT_BLOCK = "hpf"
TONE = "tone"

N_ITEMS = 8192
SKIP = 1024           # > the start-up of the 385-tap Hamming high-pass (and of Kaiser/Blackman designs up to ~1000 taps)
OUT_PEAK = TONE_AMP * GAIN                               # 1.0
OUT_RMS = round(OUT_PEAK / math.sqrt(2), 5)              # 0.70711
SRC_RMS = round(math.sqrt(DC_OFFSET ** 2 + TONE_AMP ** 2 / 2), 5)   # 0.80156
DC_TOL = 0.05
REL_TOL = 0.05
FREQ_TOL = round(0.01 * TONE_F, 3)                       # 12 Hz

DC_FILTERS = ["high_pass_filter", "band_pass_filter", "band_reject_filter", "low_pass_filter",
              "fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx", "iir_filter_xxx",
              "dc_blocker_xx", "single_pole_iir_filter_xx", "blocks_moving_average_xx"]

# ---------------------------------------------------------------- reference (compact form)
full = C.chain("clean_tone", [
    C.sig_source(TONE, "float", freq=TONE_F, amp=TONE_AMP, offset=DC_OFFSET),
    C.high_pass(FAULT_BLOCK, "float", cutoff=HPF_CUTOFF, width=HPF_WIDTH),
    C.multiply_const("gain", "float", GAIN),
    C.file_sink("clean_sink", "float", OUT_FILE),
], {"samp_rate": FS})


def compact(data: dict) -> dict:
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(data, defaults)
    diffs = grc.compact_equivalent(data, out)
    assert not diffs, diffs
    return out


ref = compact(full)
sig = C.signature(ref)
assert C.signature_split(sig) == SPLIT, (sig, C.signature_split(sig))
ids = grc.block_ids()
assert all(b["id"] in ids for b in ref["blocks"])
assert WRONG_ID not in ids, f"{WRONG_ID} exists in the catalog"

# ---------------------------------------------------------------- starting flowgraph (the fault)
start = edit.set_id(ref, FAULT_BLOCK, WRONG_ID)


# ---------------------------------------------------------------- equivalent variants
def _replace_block(data: dict, name: str, new_block: dict) -> dict:
    d = copy.deepcopy(data)
    d["blocks"] = [new_block if b["name"] == name else b for b in d["blocks"]]
    return d


# v1: the taps come from a High-pass Filter Taps variable and feed a generic Decimating FIR Filter.
v1 = _replace_block(ref, FAULT_BLOCK, C.block(FAULT_BLOCK, "fir_filter_xxx", type="fff", taps="hpf_taps"))
v1["blocks"].insert(1, C.block("hpf_taps", "variable_high_pass_filter_taps",
                               cutoff_freq=HPF_CUTOFF, width=HPF_WIDTH))

# v2: a DC Blocker removes the offset instead of a windowed-sinc high-pass.
v2 = _replace_block(ref, FAULT_BLOCK, C.block(FAULT_BLOCK, "dc_blocker_xx", type="ff", length="32"))

# v3: the 2.5 gain is applied by the filter's own gain parameter; no Multiply Const block.
v3 = edit.set_param(edit.bypass(ref, "gain"), FAULT_BLOCK, "gain", GAIN)

# v4: scale first, then a band-pass (300 Hz - 6 kHz, Hann window) removes the DC.
v4 = copy.deepcopy(ref)
v4["blocks"] = [b for b in v4["blocks"] if b["name"] != FAULT_BLOCK] + [
    C.block("bpf", "band_pass_filter", type="fir_filter_fff", low_cutoff_freq=HPF_CUTOFF,
            high_cutoff_freq=6000, width=HPF_WIDTH, win="window.WIN_HANN")]
v4["connections"] = [[TONE, "0", "gain", "0"], ["gain", "0", "bpf", "0"], ["bpf", "0", "clean_sink", "0"]]

# v5 (reviewer answer b_fft_filter): an FFT Filter with firdes.high_pass taps.
v5 = _replace_block(ref, FAULT_BLOCK, C.block(FAULT_BLOCK, "fft_filter_xxx", type="fff", nthreads="1",
                                              taps=f"firdes.high_pass(1, samp_rate, {HPF_CUTOFF}, {HPF_WIDTH})"))

# v6 (reviewer answers h_kaiser, k_abs_path): the High Pass Filter with a Kaiser window, output path absolute.
v6 = edit.set_param(edit.set_param(ref, FAULT_BLOCK, "win", "window.WIN_KAISER"),
                    "clean_sink", "file", f"/tmp/{OUT_FILE}")

# v7 (reviewer answer m_iir_1st_order): a first-order IIR DC blocker y[n] = x[n] - x[n-1] + 0.98 y[n-1]
# (gain 1.006 at 1.2 kHz, time constant 50 samples).
v7 = _replace_block(ref, FAULT_BLOCK, C.block(FAULT_BLOCK, "iir_filter_xxx", type="ffd", fftaps="[1, -1]",
                                              fbtaps="[1, -0.98]", oldstyle="False"))

variants = {"v1_taps_variable_fir.grc": v1, "v2_dc_blocker.grc": v2,
            "v3_gain_in_filter.grc": v3, "v4_scale_then_band_pass.grc": v4,
            "v5_fft_filter.grc": v5, "v6_kaiser_abs_path.grc": v6, "v7_iir_first_order.grc": v7}
variants = {k: compact(v) for k, v in variants.items()}
for k, v in variants.items():
    assert all(b["id"] in ids for b in v["blocks"]), k
    assert C.signature_split(C.signature(v)) == SPLIT, (k, C.signature(v))

# ---------------------------------------------------------------- instruction
instruction = (
    "This flowgraph is meant to clean up a test tone. It runs at a sample rate of 32 kHz. The Signal "
    "Source block tone produces a 1.2 kHz cosine (float samples, amplitude 0.4) that sits on a DC "
    "offset of 0.75. A high-pass filter, designed for a 300 Hz cutoff and a 200 Hz transition width, "
    "removes the DC offset, and a Multiply Const block then scales the signal by 2.5. The File Sink "
    "whose file name is clean_tone.dat (item type float) should therefore receive a 1.2 kHz tone "
    "centred on zero that swings between -1.0 and +1.0.\n\n"
    "At the moment GNU Radio Companion cannot generate the flowgraph: it shows the filter as a Missing "
    "Block because its block id is not found.\n\n"
    "Please repair the flowgraph. The DC offset must still be removed by a filter block (the high-pass "
    "filter, or another filter that blocks DC and passes the tone), not by subtracting a fixed value. "
    "Keep the sample rate, the Signal Source tone, the gain of 2.5 and the File Sink clean_tone.dat as "
    "they are. After the first 1024 output samples, which cover the filter's start-up, the output must "
    "have a mean within 0.05 of zero and a peak amplitude of 1.0 within 5 % (an RMS of 0.707, also "
    "within 5 %), and its tone must be at 1.2 kHz within 1 %. The flowgraph should keep producing "
    "samples until it is stopped; do not limit the number of samples."
)

# ---------------------------------------------------------------- observation and oracle
observation = C.file_observation(OUT_FILE, n_items=N_ITEMS, timeout_s=10.0)
observation["points"][0]["sink"]["params"]["type"] = "float"
observation["points"].append({"name": "src", "edge": {"block": TONE, "port": "0"}})
observation["contract"] = (
    "'out': the float samples written to the File Sink whose file name is clean_tone.dat (item type "
    "float; the first 1024 samples, which cover the filter's start-up transient, are skipped); 'src': "
    "the output of the Signal Source block tone (edge tap, no extra block). V4 liveness: 8192 items at "
    "each point.")
checks = [
    # f2p: everything at the output depends on the repaired filter block
    {"point": "out", "feature": "mean", "expect": 0.0, "tol": DC_TOL, "skip": SKIP, "group": "f2p",
     "property": "dc_removed"},
    {"point": "out", "feature": "max_abs", "expect": OUT_PEAK, "rel_tol": REL_TOL, "skip": SKIP, "group": "f2p",
     "property": "output_tone"},
    {"point": "out", "feature": "rms", "expect": OUT_RMS, "rel_tol": REL_TOL, "skip": SKIP, "group": "f2p",
     "property": "output_tone"},
    {"point": "out", "feature": "peak_freq", "fs": FS, "expect": TONE_F, "tol": FREQ_TOL, "skip": SKIP,
     "group": "f2p", "property": "output_tone"},
    # p2p: the tone source, which the fault does not touch, is kept (R8)
    {"point": "src", "feature": "mean", "expect": DC_OFFSET, "tol": round(REL_TOL * DC_OFFSET, 5),
     "group": "p2p", "property": "tone_source_kept"},
    {"point": "src", "feature": "rms", "expect": SRC_RMS, "rel_tol": REL_TOL, "group": "p2p",
     "property": "tone_source_kept"},
    {"point": "src", "feature": "peak_freq", "fs": FS, "expect": TONE_F, "tol": FREQ_TOL, "group": "p2p",
     "property": "tone_source_kept"},
]
graph_checks = [
    {"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": DC_FILTERS, "to_ids": ["blocks_file_sink"],
     "group": "f2p", "property": "dc_removed_by_filter"},
    {"kind": "block_present", "ids": ["analog_sig_source_x"],
     "params": {"freq": str(TONE_F), "amp": str(TONE_AMP), "offset": str(DC_OFFSET)},
     "group": "p2p", "property": "tone_source_kept"},
]

spec_properties = [
    {"id": "output_point", "text": "float samples are written by a File Sink to clean_tone.dat",
     "checked_by": "observation 'out' (sink matched by file name and type float; a missing or mistyped point fails V5)"},
    {"id": "dc_removed", "text": "the 0.75 DC offset is removed: output mean within 0.05 of zero after 1024 samples",
     "checked_by": "f2p mean at 'out'"},
    {"id": "dc_removed_by_filter",
     "text": "the DC offset is removed by a filter block, not by subtracting a fixed value",
     "checked_by": "f2p graph check: a path Signal Source -> filter block -> File Sink (any filter block id)"},
    {"id": "output_tone",
     "text": "the output is the 1.2 kHz tone (within 1 %) scaled by the kept gain of 2.5: peak 1.0, RMS 0.707 (within 5 %)",
     "checked_by": "f2p max_abs, rms and peak_freq at 'out' (the gain is observable only here)"},
    {"id": "tone_source_kept",
     "text": "the Signal Source tone and the 32 kHz sample rate are kept: 1.2 kHz cosine, amplitude 0.4, DC offset 0.75",
     "checked_by": "p2p mean, rms and peak_freq (fs 32 kHz) at edge point 'src', plus a p2p graph check on the "
                   "source's freq/amp/offset values (a sample-rate change that is matched by a tone-frequency "
                   "change gives identical samples and is caught only there)"},
    {"id": "run_length", "text": "the flowgraph keeps producing samples until it is stopped",
     "checked_by": "V4 liveness: at least 8192 items at both observation points; nothing more is claimed"},
    {"id": "context_only",
     "text": "the 300 Hz cutoff and 200 Hz transition width describe the original design; they are not graded",
     "checked_by": "not checked by design (a cutoff anywhere that blocks DC and passes 1.2 kHz is accepted)"},
]

mutations = [
    {"op": "bypass", "block": FAULT_BLOCK, "breaks": "dc_removed", "note": "missing filter"},
    {"op": "rewire_input", "dst": "clean_sink", "dst_port": "0", "new_src": TONE, "breaks": "dc_removed",
     "note": "wrong tap: the raw source is written"},
    {"op": "set_param", "block": FAULT_BLOCK, "param": "win", "value": "window.WIN_RECTANGULAR",
     "breaks": "dc_removed", "note": "near miss: DC gain 5.2 %, output mean -0.097"},
    {"op": "set_param", "block": FAULT_BLOCK, "param": "cutoff_freq", "value": "2500", "breaks": "output_tone",
     "note": "cutoff above the tone: the tone is removed"},
    {"op": "set_param", "block": FAULT_BLOCK, "param": "gain", "value": "1.06", "breaks": "output_tone",
     "note": "near miss: output 6 % too large"},
    {"op": "set_param", "block": "gain", "param": "const", "value": "2.0", "breaks": "output_tone",
     "note": "the gain of 2.5 not kept"},
    {"op": "scale_param", "block": TONE, "param": "freq", "factor": 1.02, "breaks": "tone_source_kept",
     "note": "near miss: tone at 1224 Hz (2 %)"},
    {"op": "set_param", "block": TONE, "param": "offset", "value": "0", "breaks": "tone_source_kept",
     "note": "source changed; the output alone cannot show it"},
    {"op": "set_param", "block": TONE, "param": "amp", "value": "0.5", "breaks": "tone_source_kept"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "48000", "breaks": "tone_source_kept",
     "note": "sample rate not kept: the 1.2 kHz tone reads as 800 Hz at 32 kHz"},
    {"op": "set_param", "block": "clean_sink", "param": "file", "value": "clean.dat", "breaks": "output_point",
     "note": "output written to another file"},
]

vsigs = {Path(k).stem: C.signature(v) for k, v in variants.items()}
notes = (
    "Round-1 fixes applied 18 Oct 2026 (see the resolution file). Known limits: (1) the filter-path graph check "
    "is by block type, so a filter that passes everything combined with a subtracted constant would satisfy it "
    "(no plausible answer does this); (2) the 300 Hz / 200 Hz design is context, not graded: any filter block "
    "that blocks DC and passes the tone within the stated tolerances is accepted; (3) a High Pass Filter with "
    "the Rectangular window is rejected because it leaves 5.2 % DC gain (output mean -0.097, outside the stated "
    "0.05); (4) the item is an easy L1 repair (the wrong id is the catalog file name of the correct block); "
    "treat it as an easy control in the analysis. Scoring note (tooling T9): the start graph's unknown id "
    f"'{WRONG_ID}' (fault.wrong_id) must not count in the hallucinated-block rate when an answer keeps it; "
    "count it as 'fault not repaired'. L1 repair conventions shared with gbf-dev-04: symptom-only wording, "
    "p2p = properties of the parts the fault does not touch (R8), an edge observation point on the kept "
    "source. No hum source is added: it would move the v2 cell to combine|filter|tone (train). v2 signatures: "
    "reference " + "|".join(sig) + "; " + "; ".join(f"{k}: {'|'.join(v)}" for k, v in vsigs.items()) + "."
)

record = {
    "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
    "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
    "source": {"kind": "composed_chain", "ref": "|".join(sig)},
    "cluster_id": C.cluster_id(sig),
    "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                   "generator_version": "0.2.0"},
    "review": {"status": "pending-team-review", "reviewers": [], "notes": notes},
    "family": "repair", "level": "L1",
    "instruction": instruction,
    "start_grc": "start.grc", "reference_grc": "reference.grc",
    "observation": observation,
    "oracle": {"checks": checks, "graph_checks": graph_checks},
    "mutations": mutations,
    "spec_properties": spec_properties,
    "perturbation": {"perturbed": False, "changes": []},
    "fault": {"type": "nonexistent_block",
              "description": (f"Block '{FAULT_BLOCK}' has id '{WRONG_ID}', which does not exist in GNU Radio "
                              "3.10.12 (it is the catalog file name filter_high_pass_filter.block.yml; the "
                              "block id is 'high_pass_filter'). GRC loads it as a Missing (dummy) block, so "
                              "the flowgraph fails to load (V2). Fix: id high_pass_filter, or any other filter "
                              "block that blocks DC and passes the 1.2 kHz tone at unity gain."),
              "expected_rung": "V2", "block": FAULT_BLOCK, "wrong_id": WRONG_ID, "correct_id": "high_pass_filter"},
    "disjointness": ("Repair of a hallucinated block id in a DC-removing tone-cleanup chain. No training-side task "
                     "spec covers this: they are dial tone (t01/t24), QPSK BER, FM/WBFM receiver repair (t03/t17), "
                     "unconnected-sink repair (t04), QAM debug, resampler-ratio repair (t20), AM demod, "
                     "spectrogram, GMSK, PSK31, RSSI, Embedded-Python gating, and the GRC-Agent dial-tone edit "
                     "scenarios. Under signature v2 the reference and all seven variants share the cell "
                     "filter|tone, which hashes to the test partition (the v1 train-side variant signatures noted "
                     "in round 1 no longer exist)."),
    "notes": "v2 signature of the reference: " + "|".join(sig) + "; variants: "
             + "; ".join(f"{k}: {'|'.join(v)}" for k, v in vsigs.items()),
}

files = {"reference.grc": grcfile.dump(ref), "start.grc": grcfile.dump(start)}
files.update({f"variants/{k}": grcfile.dump(v) for k, v in variants.items()})

if __name__ == "__main__":
    errors = taskio.save(HERE, record, files)
    if errors:
        raise SystemExit("schema errors:\n" + "\n".join(errors))
    for stale in sorted((HERE / "variants").glob("*.grc")):
        if f"variants/{stale.name}" not in files:
            stale.unlink()
            print(f"removed stale {stale.relative_to(ROOT)}")
    print(f"start     v2 {'|'.join(C.signature(start))}")
    for k, v in vsigs.items():
        print(f"{k:28s} v2 {'|'.join(v)} -> {C.signature_split(v)}")
    print(f"written {HERE} (signature {'|'.join(sig)} -> {C.signature_split(sig)})")
