"""Authoring script for GRC-Bench Tier-F item gbf-dev-02 (family build, level L1, composed chain).

Re-themed on 18 Oct (reports/tier_f_reviews/fix_plan.json): the former band-limited-noise item was a
template twin of gbf-test-03 and hashes to test under signature v2. New theme: the envelope of a
Rayleigh flat-fading channel. In the empty flowgraph (sample rate 48 kHz) the user asks for complex
Gaussian noise with an RMS magnitude of 0.8, its magnitude written as float samples to envelope.dat.
v2 signature envelope|noise -> dev.

Deterministic: regenerates task.json, reference.grc and variants/*.grc (gates.json is written by
`./gb tier-f gates`). Run in WSL:

    cd grc-bench && PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python \
        items/tier_f/dev/gbf-dev-02/build.py           # write the task
    ... build.py --probes                                 # ladder verdicts of the counter-examples
    ./gb tier-f gates items/tier_f/dev/gbf-dev-02

Numbers (GNU Radio 3.10.12; fixer calibration d02_calib.py, 65536-sample captures as in the oracle):
  * the complex Noise Source divides its amplitude between I and Q (each has standard deviation
    amp / sqrt 2), so amp 0.8 gives RMS |z| = 0.800 and mean |z| = 0.709; the Fast Noise Source and the
    Channel Model's AWGN (noise voltage) use the same convention;
  * Rayleigh envelope with E|z|^2 = 0.64: mean = 0.8 * sqrt(pi) / 2 = 0.70898, RMS = 0.8, so
    mean / RMS = 0.886. Distinguishing cases: power |z|^2 instead of |z| gives mean 0.64 and RMS 0.905;
    |real Gaussian| (half-normal) scaled to RMS 0.8 has mean -10.4 %; uniform complex noise (amp 0.8)
    mean -39 %, RMS -42 % (ratio 0.937); a constant 1.0. A rectified sine of RMS 0.8 has mean +1.6 %
    (ratio 0.900), so a structural check is needed to tell noise from a deterministic shape (round-1
    finding F2 of the former item); probe rectified_sine;
  * seed 0 (GNU Radio's default) is a fixed seed in 3.10.12, not time-based: 300 runs are identical;
  * spread over 300 seeds (deviation from 0.709 / 0.8): Noise Source within 0.62 % (sd 0.2 %; seed 0:
    +0.23 / +0.15 %); Fast Noise Source with the default 8192-variate pool -1.58..+1.86 % (sd 0.6 %;
    seed 0: -1.47 / -1.14 %); Channel Model AWGN within 1.3 % (sd 0.45 %). The stated 3 % is therefore
    >= 5 sigma from every valid generator, and a 5 % amplitude error (mutant amp 0.84: +4.6..+5.4 %), the
    per-component misreading (amp 0.8 / sqrt 2: -29 %) and the half-normal fail with certainty;
  * GNU Radio's complex Laplacian / impulse noise types raise "invalid type" in the block thread, so
    the mean/RMS pair is the behavioural Gaussianity check among the noise types that run.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from grcbench.common.canary import CANARY  # noqa: E402
from grcbench.tier_f import compose as C  # noqa: E402
from grcbench.tier_f import edit, grc, grcfile, taskio  # noqa: E402

TASK_ID = "gbf-dev-02"
SPLIT = "dev"

# ---- the specification (every number here is stated in the instruction) ----
FS = 48000                          # sample rate, Hz
RMS_MAG = 0.8                       # RMS magnitude of the complex noise = RMS of the envelope
MEAN_ENV = RMS_MAG * math.sqrt(math.pi) / 2   # 0.70898: mean of a Rayleigh envelope with that RMS
TOL = 0.03                          # relative tolerance on mean and RMS
OUT_FILE = "envelope.dat"
SEED = 42                           # reference only (gate determinism); not part of the specification
N_ITEMS = 65536

INSTRUCTION = (
    "In this flowgraph, simulate the amplitude of a Rayleigh flat-fading channel at a sample rate of "
    "48 kHz. Generate complex Gaussian noise with an RMS magnitude of 0.8 (the square root of the mean of "
    "|z|^2 over its samples z is 0.8), take the magnitude of every sample, and write the magnitudes as "
    "float samples to a File Sink whose file name is envelope.dat. The flowgraph should keep producing "
    "samples until it is stopped; do not limit the number of samples. Like any Rayleigh envelope with "
    "that RMS, the samples in envelope.dat must have an RMS level of 0.8 and a mean of 0.709 (0.8 times "
    "the square root of pi, divided by 2), each within 3 %."
)

SPEC_PROPERTIES = [
    "P1 sample rate 48 kHz: a variable of the final graph holds 48000 (graph check; the envelope "
    "statistics do not depend on the rate, and the harness's empty start graph already has samp_rate = "
    "48000, so this fails only if an answer changes it)",
    "P2 RMS magnitude 0.8: the RMS of envelope.dat is 0.8 within 3 % (the RMS of |z| is the RMS "
    "magnitude of the noise)",
    "P3 Rayleigh envelope of complex Gaussian noise: the mean of envelope.dat is 0.709 within 3 % (mean / "
    "RMS = sqrt(pi)/2 = 0.886; the power |z|^2, the magnitude of one real Gaussian component, uniform "
    "complex noise and a constant all miss it)",
    "P4 the envelope is made from noise (structural, graph check): envelope.dat is reached from a noise "
    "generator (Noise Source, Fast Noise Source, or the AWGN of a Channel Model block); a deterministic "
    "shape with similar moments, such as a rectified sine (mean / RMS 0.900), would otherwise pass P2-P3",
    "P5 observation: the magnitudes are written as float samples to the File Sink whose file name is "
    f"envelope.dat (the observation point requires that file name and item type); P2-P3 use its first "
    f"{N_ITEMS} samples",
    f"P6 run condition (V5): the flowgraph keeps producing samples until it is stopped; V5 liveness needs "
    f"{N_ITEMS} items at envelope.dat, so a graph that stops itself earlier fails V5",
]

NOISE_IDS = ["analog_noise_source_x", "analog_fastnoise_source_x", "channels_channel_model",
             "channels_channel_model2", "channels_dynamic_channel_model"]

ORACLE = {
    "checks": [
        {"point": "env", "feature": "rms", "expect": RMS_MAG, "rel_tol": TOL, "group": "main", "property": "P2"},
        {"point": "env", "feature": "mean", "expect": round(MEAN_ENV, 5), "rel_tol": TOL, "group": "main",
         "property": "P3"},
    ],
    "graph_checks": [
        {"kind": "path", "from_ids": NOISE_IDS, "to_ids": ["blocks_file_sink"], "group": "main",
         "property": "P4 noise generator -> File Sink"},
        {"kind": "block_present", "ids": ["variable"], "params": {"value": str(FS)}, "group": "main",
         "property": "P1 sample rate 48 kHz"},
    ],
}

MUTATIONS = [
    {"op": "set_id", "block": "mag", "new_id": "blocks_complex_to_mag_squared",
     "breaks": "P3/P2 magnitude, not power (|z|^2: mean 0.64, RMS 0.905)"},
    {"op": "set_id", "block": "mag", "new_id": "blocks_complex_to_real",
     "breaks": "P3 magnitude (real part instead: mean about 0)"},
    {"op": "set_param", "block": "noise", "param": "amp", "value": "0.565685",
     "breaks": "P2 RMS magnitude 0.8 (amplitude read as the per-component standard deviation: RMS 0.566)"},
    {"op": "set_param", "block": "noise", "param": "amp", "value": "0.84",
     "breaks": "P2/P3 RMS magnitude 0.8 (5 % too large)"},
    {"op": "set_param", "block": "noise", "param": "noise_type", "value": "analog.GR_UNIFORM",
     "breaks": "P3 complex Gaussian noise (uniform: mean 0.433, RMS 0.462)"},
    {"op": "set_param", "block": "samp_rate", "param": "value", "value": "32000",
     "breaks": "P1 sample rate 48 kHz (GRC's default 32 kHz)"},
]


# ---- flowgraphs ----
def _cnoise(name: str = "noise", bid: str = "analog_noise_source_x", amp=RMS_MAG, seed=SEED) -> dict:
    params = {"type": "complex", "noise_type": "analog.GR_GAUSSIAN", "amp": amp}
    if seed is not None:
        params["seed"] = seed
    return C.block(name, bid, **params)


def reference() -> dict:
    """complex Gaussian noise (amp 0.8) -> throttle -> Complex To Mag -> file sink (float)."""
    return C.chain("rayleigh_envelope",
                   [_cnoise(),
                    C.block("thr", "blocks_throttle2", type="complex", samples_per_second="samp_rate"),
                    C.block("mag", "blocks_complex_to_mag"),
                    C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": str(FS)})


def variant_fastnoise() -> dict:
    """Fast Noise Source (default 8192-sample pool), no throttle, rate written as 48e3."""
    return C.chain("fading_fast",
                   [_cnoise("fading", "analog_fastnoise_source_x", seed=None),
                    C.block("envelope", "blocks_complex_to_mag"),
                    C.file_sink("out", "float", OUT_FILE)],
                   {"samp_rate": "48e3"})


def variant_two_real_sources() -> dict:
    """Two independent float noise sources (standard deviation 0.8 / sqrt 2 each) as I and Q."""
    sd = "0.8 / 2**0.5"
    blocks = [
        C.block("noise_i", "analog_noise_source_x", type="float", noise_type="analog.GR_GAUSSIAN", amp=sd, seed=1),
        C.block("noise_q", "analog_noise_source_x", type="float", noise_type="analog.GR_GAUSSIAN", amp=sd, seed=2),
        C.block("iq", "blocks_float_to_complex"),
        C.block("mag", "blocks_complex_to_mag"),
        C.block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
        C.file_sink("snk", "float", OUT_FILE),
    ]
    conns = [["noise_i", "0", "iq", "0"], ["noise_q", "0", "iq", "1"], ["iq", "0", "mag", "0"],
             ["mag", "0", "thr", "0"], ["thr", "0", "snk", "0"]]
    return C.graph("fading_iq", blocks, conns, {"samp_rate": "48000"})


def variant_mag_squared_sqrt() -> dict:
    """Unit-power noise scaled by 0.8, then |z|^2 and a square root (Transcendental sqrt)."""
    return C.chain("fading_sqrt",
                   [_cnoise("src", amp="1", seed=5),
                    C.block("scale", "blocks_multiply_const_vxx", type="complex", const="0.8"),
                    C.block("thr", "blocks_throttle2", type="complex", samples_per_second="samp_rate"),
                    C.block("power", "blocks_complex_to_mag_squared"),
                    {"name": "root", "id": "blocks_transcendental",   # its "name" parameter is the function
                     "parameters": {"type": "float", "name": "sqrt"}},
                    C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": "48000"})


def variant_channel_model_awgn() -> dict:
    """Channel Model's AWGN on an all-zero input (Null Source): noise voltage 0.8, no impairments."""
    return C.chain("fading_channel",
                   [C.block("zeros", "blocks_null_source", type="complex"),
                    C.block("chan", "channels_channel_model", noise_voltage="0.8", freq_offset="0.0",
                            epsilon="1.0", taps="[1.0]", seed="3"),
                    C.block("thr", "blocks_throttle2", type="complex", samples_per_second="samp_rate"),
                    C.block("mag", "blocks_complex_to_mag"),
                    C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": "48000"})


VARIANTS = {
    "v1_fast_noise_source.grc": variant_fastnoise,
    "v2_two_real_noise_sources_iq.grc": variant_two_real_sources,
    "v3_mag_squared_then_sqrt.grc": variant_mag_squared_sqrt,
    "v4_channel_model_awgn.grc": variant_channel_model_awgn,
}

# ---- probes: reviewer-style counter-examples and alternatives a mutant cannot express ----
def _real_to_mag(src: dict, gid: str) -> dict:
    """float source -> Float To Complex (real part only) -> Complex To Mag -> envelope.dat."""
    blocks = [src, C.block("f2c", "blocks_float_to_complex"), C.block("mag", "blocks_complex_to_mag"),
              C.file_sink("snk", "float", OUT_FILE)]
    conns = [[src["name"], "0", "f2c", "0"], ["f2c", "0", "mag", "0"], ["mag", "0", "snk", "0"]]
    return C.graph(gid, blocks, conns, {"samp_rate": str(FS)})


def _with_head(n: int) -> dict:
    g = reference()
    g["blocks"].append(C.block("hd", "blocks_head", type="float", num_items=str(n)))
    g["connections"] = [c for c in g["connections"] if c[2] != "snk"] + \
        [["mag", "0", "hd", "0"], ["hd", "0", "snk", "0"]]
    return g


def _magphase(park_phase: bool) -> dict:
    blocks = [_cnoise(seed=9), C.block("mp", "blocks_complex_to_magphase"), C.file_sink("snk", "float", OUT_FILE)]
    conns = [["noise", "0", "mp", "0"], ["mp", "0", "snk", "0"]]
    if park_phase:
        blocks.append(C.block("phase_null", "blocks_null_sink", type="float"))
        conns.append(["mp", "1", "phase_null", "0"])
    return C.graph("fading_magphase", blocks, conns, {"samp_rate": str(FS)})


def _fading_model() -> dict:
    return C.chain("fading_jakes",
                   [C.block("one", "analog_const_source_x", type="complex", const="0.8"),
                    C.block("fade", "channels_fading_model", N="8", fDTs="10.0/samp_rate", LOS="False",
                            K="4.0", seed="1"),
                    C.block("mag", "blocks_complex_to_mag"), C.file_sink("snk", "float", OUT_FILE)],
                   {"samp_rate": str(FS)})


def _rate_expression() -> dict:
    g = reference()
    for b in g["blocks"]:
        if b["name"] == "samp_rate":
            b["parameters"]["value"] = "48*1000"
    return g


def probes() -> dict[str, tuple[dict, bool, str]]:
    """name -> (graph, expected to pass V6, why)."""
    sine = C.block("tone", "analog_sig_source_x", type="float", samp_rate="samp_rate",
                   waveform="analog.GR_SIN_WAVE", freq="1000", amp=f"{RMS_MAG * 2 ** 0.5:.6f}", offset="0")
    half = C.block("noise", "analog_noise_source_x", type="float", noise_type="analog.GR_GAUSSIAN",
                   amp=str(RMS_MAG), seed="4")
    fast = _cnoise("fast", "analog_fastnoise_source_x", seed=777)
    return {
        "rectified_sine": (_real_to_mag(sine, "rectified_sine"), False,
                           "|sin| of RMS 0.8 meets both moments (mean +1.6 %) and fails the P4 path check"),
        "half_normal": (_real_to_mag(half, "half_normal"), False,
                        "|real Gaussian| of RMS 0.8 (one component only) fails the P3 mean (-10 %)"),
        "head_40000": (_with_head(40000), False, "a fixed-length file (Head 40000) fails V5 liveness"),
        "fastnoise_seed_777": (C.chain("fading_fast_seed", [fast, C.block("mag", "blocks_complex_to_mag"),
                                                            C.file_sink("snk", "float", OUT_FILE)],
                                       {"samp_rate": str(FS)}), True,
                               "Fast Noise Source with another seed passes (pool spread is 0.6 %)"),
        "complex_to_magphase": (_magphase(True), True,
                                "Complex to Mag/Phase, magnitude to the file and phase to a Null Sink, passes"),
        "magphase_phase_open": (_magphase(False), False,
                                "the same with the phase output left open fails at V5: GNU Radio's runtime "
                                "needs both outputs connected (GRC marks the port optional; not an oracle "
                                "decision)"),
        "fading_model_jakes": (_fading_model(), False,
                               "a sum-of-sinusoids Fading Model is not the complex Gaussian noise the "
                               "instruction asks for: fails the P4 path check"),
        "samp_rate_expression": (_rate_expression(), False,
                                 "documented limit: samp_rate written as 48*1000 fails the P1 value check "
                                 "(48e3 / 48000.0 pass; the harness start graph already holds 48000)"),
    }


def run_probes() -> int:
    from grcbench.tier_f.ladder import check
    task = taskio.load(HERE)
    bad = 0
    for name, (g, expect, why) in probes().items():
        cg = compact(g, KEEP)
        res = check(grcfile.dump(cg), task)
        ok = res["first_failed"] is None and res["highest_passed"] == "V6"
        v6 = res["rungs"].get("V6") or {}
        info = [f"{c['feature']}={c['value']}{'' if c['passed'] else ' FAIL'}" for c in v6.get("checks", [])] + \
               [f"graph:{g_['kind']} FAIL" for g_ in v6.get("graph_checks", []) if not g_["passed"]]
        if res["first_failed"] and res["first_failed"] != "V6":
            info = (res["rungs"].get(res["first_failed"]) or {}).get("errors", [])[:1]
        verdict = "as expected" if ok == expect else "UNEXPECTED"
        bad += ok != expect
        print(f"{name:22s} passed={ok!s:5s} first_failed={res['first_failed']} {info} -> {verdict}\n    {why}")
    return 1 if bad else 0


REVIEW_NOTES = (
    "Re-themed on 18 Oct (fix plan): the former band-limited-noise item was a template twin of gbf-test-03. "
    "Known oracle limits (drafted by Claude; for the team). (1) Only the first two moments of the envelope "
    "are checked (there is no distribution feature): the mean/RMS pair (ratio 0.886) rejects the power "
    "|z|^2, a single real component (half-normal, mean -10 %), uniform complex noise and constants, but a "
    "non-Gaussian source with the same two moments would pass. GNU Radio's complex Laplacian and impulse "
    "noise types fail at run time ('invalid type'), so no noise-type graph check is used. (2) The path "
    "check proves that the File Sink is fed from a noise generator; it accepts Noise Source, Fast Noise "
    "Source and the AWGN of Channel Model / Channel Model 2 / Dynamic Channel Model. A sum-of-sinusoids "
    "Fading Model fed by a constant is not complex Gaussian noise as the instruction asks and fails it "
    "(probe fading_model_jakes); a deterministic signal added on a noise path could pass. (3) The 48 kHz "
    "sample rate is unobservable in white-noise statistics and is checked as a variable's value "
    "(block_present variable value 48000; numeric comparison, so 48e3 passes, an expression such as "
    "48*1000 does not: probe samp_rate_expression). The harness's empty start graph already holds "
    "samp_rate = 48000, so this fails only an answer that rewrites or removes it. (4) The noise seed is "
    "not specified (the reference keeps seed 42 so gate runs are bit-identical; GNU Radio's seed 0 is also "
    "fixed). The stated 3 % is >= 5 sigma from every valid generator over 300 seeds (Noise Source within "
    "0.62 %, default Fast Noise Source within 1.9 %, Channel Model AWGN within 1.3 %); a Fast Noise Source "
    "whose pool is shrunk far below its 8192-variate default can miss it on some seeds. (5) A Head block "
    "larger than the 65536-item capture is not detected. Variant signatures (v2): v1, v2 envelope|noise "
    "(dev); v3 envelope|noise|nonlinear (hashes to train) and v4 channel_model|envelope (hashes to test; "
    "no test task uses that cell) are genuine structural alternatives (|z|^2 then a square root; a Channel "
    "Model's AWGN) and belong on the reserved list (fix_plan R1/R2); probe complex_to_magphase "
    "(other:blocks_complex_to_magphase|noise) also passes."
)


def compact(data: dict, keep: dict[str, list[str]] | None = None) -> dict:
    """Compact form (only non-default parameters, plus the `keep` ones), verified equivalent in GRC."""
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(copy.deepcopy(data), defaults)
    full = {b["name"]: b.get("parameters", {}) for b in data["blocks"]}
    for blk in out["blocks"]:
        for pid in (keep or {}).get(blk["id"], []):
            if pid in full[blk["name"]]:
                blk["parameters"][pid] = full[blk["name"]][pid]
    diffs = grc.compact_equivalent(data, out)
    if diffs:
        raise SystemExit(f"compact form differs from the full form: {diffs}")
    return out


KEEP = {"analog_noise_source_x": ["noise_type"], "analog_fastnoise_source_x": ["noise_type"]}


def main(argv: list[str]) -> int:
    if "--probes" in argv:
        return run_probes()
    ref = compact(reference(), KEEP)
    sig = C.signature(ref)
    split = C.signature_split(sig)
    if split != SPLIT:
        raise SystemExit(f"v2 signature {sig} belongs to split {split}, not {SPLIT}")
    for op in MUTATIONS:
        edit.apply(ref, op)                      # every mutation applies to the reference
    files = {"reference.grc": grcfile.dump(ref)}
    print(f"reference: v2 signature {'|'.join(sig)} -> {split}")
    variant_sigs = {}
    for path, fn in VARIANTS.items():
        v = compact(fn(), KEEP)
        vsig = C.signature(v)
        variant_sigs[path] = f"{'|'.join(vsig)} ({C.signature_split(vsig)})"
        print(f"{path}: v2 signature {variant_sigs[path]}")
        files[f"variants/{path}"] = grcfile.dump(v)

    record = {
        "id": TASK_ID, "tier": "F", "split": SPLIT, "version": 2, "canary": CANARY,
        "license": "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)",
        "source": {"kind": "composed_chain", "ref": "|".join(sig)},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "grcbench Phase 3 authoring",
                       "generator_version": "0.2.0"},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": REVIEW_NOTES},
        "family": "build", "level": "L1",
        "instruction": INSTRUCTION,
        "start_grc": None, "reference_grc": "reference.grc",
        "observation": {
            "points": [{"name": "env", "sink": {"block_id": "blocks_file_sink",
                                                "params": {"file": OUT_FILE, "type": "float"}}}],
            "run": {"n_items": N_ITEMS, "min_items": N_ITEMS, "timeout_s": 20},
            "contract": f"the float samples written to the File Sink whose file name is {OUT_FILE}",
        },
        "oracle": ORACLE,
        "mutations": MUTATIONS,
        "spec_properties": SPEC_PROPERTIES,
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (
            "Nearest training-side spec: t08_run_verify_probe (Qoherent collection and SFT session: a noise "
            "source feeding a file plus a level probe, run briefly, and a report on whether the measured "
            "level matches theory). This task also compares a noise level with theory, but its core is the "
            "Rayleigh envelope of COMPLEX Gaussian noise: GNU Radio's complex amplitude convention (RMS "
            "magnitude, not per-component deviation), the magnitude conversion and the envelope's mean/RMS "
            "ratio; it has no probe block and no run-and-report step. No other training spec (dial tone, "
            "QPSK BER chain, FM receiver repair, AM demod, spectrogram sweep) or GRC-Agent scenario "
            "(03_disable_and_enable, 05_full_rewire: toggling or replacing the dial-tone noise source) "
            "builds a fading envelope. Its v2 signature envelope|noise hashes to dev; it shares no cell "
            "with gbf-test-03 (filter|noise, band-limited noise) or gbf-dev-05 (noise|nonlinear, limiter)."
        ),
        "authoring_notes": {
            "level": "L1: 4 functional blocks (noise source, throttle, complex to mag, file sink), one rate",
            "signature_v2": "|".join(sig),
            "variant_signatures_v2": variant_sigs,
            "design": "see build.py docstring",
            "start_graph": ("build task: start_grc is null in this stage; in GRC-Agent's harness the start is "
                            "the empty graph with samp_rate = 48000 (the instruction's stated rate)"),
            "round1": ("findings of the former item are moot except F2 (deterministic shapes passing a "
                       "moments-only oracle: path graph check P4), F3 (every stated number checked) and F4 "
                       "(distinct mutants); see the resolution record"),
        },
    }
    errors = taskio.save(HERE, record, files)
    if errors:
        print("\n".join(errors))
        return 1
    for old in (HERE / "variants").glob("*.grc"):
        if f"variants/{old.name}" not in files:
            old.unlink()
    print(f"written {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
