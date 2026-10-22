"""Signature v2 of composed Tier-F flowgraphs (compose.signature): a functional signature that is
invariant to (a) inert / plumbing blocks, (b) block order and (c) type variants and equivalent blocks,
works on branching L2 graphs, and covers every block id used by the Tier-F items."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from grcbench.tier_f import compose as C
from grcbench.tier_f import edit

REPO = Path(__file__).resolve().parents[1]
ITEMS = REPO / "items" / "tier_f"
SR = {"samp_rate": "48000"}


def b(name, bid, **p):
    return C.block(name, bid, **p)


def ch(*blocks, variables=None):
    return C.chain("t", list(blocks), dict(SR, **(variables or {})))


def g(blocks, conns, variables=None):
    return C.graph("t", blocks, conns, dict(SR, **(variables or {})))


def sig(data):
    return C.signature(data)


TONE = C.sig_source("src", "float", "1250", "1")
CTONE = C.sig_source("src", "complex", "5500", "0.6")
NOISE = C.noise_source("src", "float", "1")
LPF = C.low_pass("lpf", "float", "3000", "500")
SNK = C.file_sink("snk", "float")
CSNK = C.file_sink("snk", "complex")
THR = b("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate")
MUL = C.multiply_const("mul", "float", "0.4")
ADD = C.add_const("add", "float", "0.55")
BPF = b("bpf", "band_pass_filter", type="fir_filter_fff", low_cutoff_freq="6000", high_cutoff_freq="9000", width="500")


# ---------------------------------------------------------------- v1 kept for provenance

def test_v1_is_kept_unchanged_for_provenance():
    # values reported by the round-1 reviews of gbf-test-02 (computed with the v1 code)
    assert C.signature_split_v1(("analog_sig_source_x", "blocks_multiply_conjugate_cc", "blocks_file_sink")) == "test"
    assert C.signature_split_v1(("analog_sig_source_x", "blocks_multiply_xx", "blocks_file_sink")) == "dev"
    assert C.signature_split_v1(("analog_sig_source_x", "blocks_rotator_cc", "blocks_file_sink")) == "train"
    assert C.signature_v1(ch(TONE, THR, LPF, SNK)) == ("analog_sig_source_x", "blocks_throttle2", "low_pass_filter",
                                                       "blocks_file_sink")
    assert C.SIGNATURE_SEED == "grc-bench-l1-signatures-v2-2026-10-18"
    assert C.SIGNATURE_SEED_V1 == "grc-bench-l1-signatures-v1-2026-10-14"


# ---------------------------------------------------------------- (a) inert / plumbing blocks

def test_invariant_to_inert_and_plumbing_blocks():
    base = ch(TONE, LPF, SNK)
    assert sig(base) == ("filter", "tone")
    noisy = g([TONE, THR, b("head", "blocks_head", type="float", num_items="48000"),
               b("skip", "blocks_skiphead", type="float", num_items="10"), b("cp", "blocks_copy", type="float"),
               b("s2v", "blocks_stream_to_vector", type="float", vlen="4"),
               b("v2s", "blocks_vector_to_stream", type="float", vlen="4"), LPF, SNK,
               b("dbg", "blocks_tag_debug", type="float"), b("ns", "blocks_null_sink", type="float"),
               b("probe", "blocks_probe_signal_x", type="float"), C.file_sink("tap", "float", "tap.dat"),
               b("gui", "qtgui_time_sink_x", type="float"),
               dict(b("off", "blocks_delay", type="float", delay="3"), states={"state": "disabled"})],
              [["src", "0", "thr", "0"], ["thr", "0", "head", "0"], ["head", "0", "skip", "0"],
               ["skip", "0", "cp", "0"], ["cp", "0", "s2v", "0"], ["s2v", "0", "v2s", "0"], ["v2s", "0", "lpf", "0"],
               ["lpf", "0", "snk", "0"], ["lpf", "0", "dbg", "0"], ["lpf", "0", "ns", "0"], ["lpf", "0", "probe", "0"],
               ["src", "0", "tap", "0"], ["lpf", "0", "gui", "0"]])
    assert sig(noisy) == sig(base)
    assert C.signature_v1(noisy) != C.signature_v1(base)


def test_invariant_to_type_conversions_and_bypassed_blocks():
    real = ch(TONE, LPF, SNK)
    cplx = ch(CTONE, b("c2r", "blocks_complex_to_real"), LPF, SNK)
    assert sig(real) == sig(cplx)
    byp = ch(TONE, dict(b("hpf", "high_pass_filter", type="fir_filter_fff", cutoff_freq="10", width="5"),
                        states={"state": "bypassed"}), LPF, SNK)
    assert sig(byp) == sig(real)


def test_inert_throttle_does_not_move_an_l2_graph():
    ref = yaml.safe_load((ITEMS / "test" / "gbf-test-13" / "reference.grc").read_text())
    thr = next(x["name"] for x in ref["blocks"] if x["id"] == "blocks_throttle2")
    stripped = edit.bypass(ref, thr)
    assert C.signature_v1(ref) != C.signature_v1(stripped)           # v1 sees the inert Throttle
    assert sig(ref) == sig(stripped) == ("data_source", "filter", "mix", "resample")


# ---------------------------------------------------------------- (b) block order

@pytest.mark.parametrize("first,second", [
    ((TONE, MUL, ADD, SNK), (TONE, ADD, MUL, SNK)),          # gbf-test-04 vs its reordered twin
    ((TONE, LPF, MUL, SNK), (TONE, MUL, LPF, SNK)),
    ((NOISE, THR, BPF, SNK), (NOISE, BPF, THR, SNK)),        # gbf-test-03 vs gbf-dev-02
    ((TONE, MUL, LPF, ADD, SNK), (TONE, ADD, LPF, MUL, SNK)),
])
def test_invariant_to_block_order(first, second):
    a, z = ch(*first), ch(*second)
    assert sig(a) == sig(z)
    assert C.signature_v1(a) != C.signature_v1(z)


# ---------------------------------------------------------------- (c) equivalent blocks

def _two_input(mult_id, lo_freq="8750", lo_through=None):
    blocks = [CTONE, C.sig_source("lo", "complex", lo_freq, "1"), b("m", mult_id, type="complex"), CSNK]
    conns = [["src", "0", "m", "0"], ["m", "0", "snk", "0"]]
    if lo_through:
        blocks.append(lo_through)
        conns += [["lo", "0", lo_through["name"], "0"], [lo_through["name"], "0", "m", "1"]]
    else:
        conns.append(["lo", "0", "m", "1"])
    return g(blocks, conns)


def test_frequency_shift_blocks_are_equivalent():
    graphs = [
        _two_input("blocks_multiply_conjugate_cc"),
        _two_input("blocks_multiply_xx", "-8750"),
        _two_input("blocks_multiply_xx", "-8750", lo_through=b("cj", "blocks_conjugate_cc")),
        _two_input("blocks_multiply_xx", "-8750", lo_through=C.multiply_const("lo_gain", "complex", "1")),
        ch(CTONE, b("rot", "blocks_rotator_cc", phase_inc="-2*math.pi*8750/samp_rate"), CSNK),
        ch(CTONE, b("x", "freq_xlating_fir_filter_xxx", type="ccc", taps="[1]", center_freq="8750"), CSNK),
        ch(CTONE, b("x", "freq_xlating_fft_filter_ccc", taps="[1.0]", center_freq="8750"), CSNK),
    ]
    sigs = {sig(x) for x in graphs}
    assert sigs == {("mix", "tone")}
    assert len({C.signature_split_v1(C.signature_v1(x)) for x in graphs}) > 1  # v1 scattered them


def test_gain_and_offset_are_parameters_not_functions():
    plain = ch(C.sig_source("src", "float", "1250", "0.4", offset="0.55"), SNK)
    assert sig(plain) == ("tone",)
    via_const_mult = g([TONE, b("k", "analog_const_source_x", type="float", const="0.4"),
                        b("m", "blocks_multiply_xx", type="float"), SNK],
                       [["src", "0", "m", "0"], ["k", "0", "m", "1"], ["m", "0", "snk", "0"]])
    via_const_add = g([TONE, MUL, b("k", "analog_const_source_x", type="float", const="0.55"),
                       b("a", "blocks_add_xx", type="float"), SNK],
                      [["src", "0", "mul", "0"], ["mul", "0", "a", "0"], ["k", "0", "a", "1"], ["a", "0", "snk", "0"]])
    single_tap = ch(TONE, b("fir", "fir_filter_xxx", type="fff", taps="[0.4]"), SNK)
    for x in (ch(TONE, MUL, SNK), ch(TONE, MUL, ADD, SNK), via_const_mult, via_const_add, single_tap,
              ch(TONE, b("m2", "blocks_multiply_const_ff", k="2"), SNK)):
        assert sig(x) == ("tone",)
    assert sig(ch(TONE, LPF, MUL, SNK)) == sig(ch(TONE, C.low_pass("lpf", "float", "3000", "500", gain="0.4"), SNK))


def test_sources_waveforms_and_types():
    assert sig(ch(C.sig_source("src", "float", "1", "1", waveform="analog.GR_SIN_WAVE"), SNK)) == ("tone",)
    assert sig(ch(C.sig_source("src", "float", "1", "1", waveform="analog.GR_SQR_WAVE"), LPF, SNK)) == \
        sig(ch(C.sig_source("src", "float", "1", "1", waveform="analog.GR_SAW_WAVE"), LPF, SNK)) == \
        ("filter", "harmonic_wave")
    assert sig(ch(NOISE, BPF, SNK)) == sig(ch(b("src", "analog_fastnoise_source_x", type="float", amp="1"), BPF, SNK))
    assert sig(ch(b("v", "blocks_vector_source_x", type="float", vector="[1,2]"), SNK)) == \
        sig(ch(b("k", "analog_const_source_x", type="float", const="1"), SNK)) == ("data_source",)


def test_filter_implementations_and_responses_are_equivalent():
    variants = [
        ch(NOISE, BPF, SNK),
        ch(NOISE, b("f", "fir_filter_xxx", type="fff", taps="firdes.band_pass(1, samp_rate, 6000, 9000, 500)"), SNK),
        ch(NOISE, b("f", "fft_filter_xxx", type="fff", taps="firdes.band_pass_2(1, samp_rate, 6e3, 9e3, 500, 60)"), SNK),
        ch(NOISE, b("hpf", "high_pass_filter", type="fir_filter_fff", cutoff_freq="6000", width="500"),
           b("lp", "low_pass_filter", type="fir_filter_fff", cutoff_freq="9000", width="500"), SNK),
        ch(NOISE, b("notch", "band_reject_filter", type="fir_filter_fff", low_cutoff_freq="30", high_cutoff_freq="90",
                    width="20"), SNK),
        ch(NOISE, b("dc", "dc_blocker_xx", type="ff", length="32"), SNK),
        ch(NOISE, b("iir", "iir_filter_xxx", type="ffd", fbtaps="[1, -0.9]", fftaps="[0.1]"), SNK),
        ch(NOISE, b("ma", "blocks_moving_average_xx", type="float", length="8"), SNK),
    ]
    taps_var = g([b("bp_taps", "variable_band_pass_filter_taps"), NOISE,
                  b("f", "fir_filter_xxx", type="fff", taps="bp_taps"), SNK],
                 [["src", "0", "f", "0"], ["f", "0", "snk", "0"]])
    var_firdes = g([b("t", "variable", value="firdes.low_pass(1.0, samp_rate, 1000, 100)"), NOISE,
                    b("f", "fft_filter_xxx", type="fff", taps="t"), SNK],
                   [["src", "0", "f", "0"], ["f", "0", "snk", "0"]])
    assert {sig(x) for x in variants + [taps_var, var_firdes]} == {("filter", "noise")}
    # the fine profile still tells the responses apart
    assert C.functional_profile(taps_var)["bandpass"] == 1
    assert C.functional_profile(var_firdes)["lowpass"] == 1


def test_taps_decide_the_role_of_a_fir_block():
    rrc = ch(NOISE, b("f", "fir_filter_xxx", type="fff", taps="firdes.root_raised_cosine(1, 4, 1, 0.35, 44)"), SNK)
    hil = ch(NOISE, b("f", "fir_filter_xxx", type="fff", taps="firdes.hilbert(65)"), SNK)
    lit = ch(NOISE, b("f", "fir_filter_xxx", type="fff", taps="[0.25, 0.5, 0.25]"), SNK)
    opaque = ch(NOISE, b("f", "fir_filter_xxx", type="fff", taps="my_taps"), SNK)
    assert sig(rrc) == ("noise", "pulse_shaping")
    assert sig(hil) == ("hilbert", "noise")
    assert sig(lit) == sig(opaque) == ("filter", "noise")


def test_rate_changes_are_one_class_and_ratio_one_is_none():
    lpf4 = C.low_pass("lpf", "float", "3000", "500")
    lpf4["parameters"]["decim"] = "4"
    forms = [
        ch(TONE, lpf4, SNK),
        ch(TONE, b("f", "fir_filter_xxx", type="fff", decim="decim", taps="firdes.low_pass(1, samp_rate, 3e3, 5e2)"),
           SNK, variables={"decim": "int(samp_rate / 12000)"}),
        ch(TONE, b("r", "rational_resampler_xxx", type="fff", decim="4"), SNK),
        ch(TONE, LPF, b("k", "blocks_keep_one_in_n", type="float", n="4"), SNK),
        ch(TONE, LPF, b("k", "blocks_keep_m_in_n", type="float", n="4"), SNK),
        ch(TONE, b("up", "rational_resampler_xxx", type="fff", interp="3"), b("dn", "low_pass_filter",
           type="fir_filter_fff", decim="12", cutoff_freq="3000", width="500"), SNK),
        ch(TONE, b("arb", "pfb_arb_resampler_xxx", type="fff", rrate="0.25"), SNK),
    ]
    assert {sig(x) for x in forms} == {("filter", "resample", "tone")}
    same_rate = [ch(TONE, b("r", "rational_resampler_xxx", type="fff", interp="3", decim="3"), SNK),
                 ch(TONE, LPF, b("k", "blocks_keep_one_in_n", type="float", n="1"), SNK)]
    assert {sig(x) for x in same_rate} == {("filter", "tone")}


def test_composite_blocks_expand_into_their_parts():
    xl = b("x", "freq_xlating_fir_filter_xxx", type="ccc", decim="4", center_freq="6000",
           taps="firdes.low_pass(1, samp_rate, 1000, 500)")
    built = g([CTONE, C.sig_source("lo", "complex", "-6000", "1"), b("m", "blocks_multiply_xx", type="complex"),
               b("lp", "low_pass_filter", type="fir_filter_ccf", decim="4", cutoff_freq="1000", width="500"), CSNK],
              [["src", "0", "m", "0"], ["lo", "0", "m", "1"], ["m", "0", "lp", "0"], ["lp", "0", "snk", "0"]])
    assert sig(ch(CTONE, xl, CSNK)) == sig(built) == ("filter", "mix", "resample", "tone")
    # FM Demod block vs Quadrature Demod + decimating low-pass
    src = b("src", "analog_sig_source_x", type="complex", freq="1000", amp="1")
    fmd = ch(src, b("d", "analog_fm_demod_cf", chan_rate="samp_rate", audio_decim="4", deviation="5000"), SNK)
    quad = ch(src, b("q", "analog_quadrature_demod_cf", gain="1"),
              b("lp", "low_pass_filter", type="fir_filter_fff", decim="4", cutoff_freq="3000", width="500"), SNK)
    assert sig(fmd) == sig(quad) == ("filter", "fm_demod", "resample", "tone")
    # Constellation Modulator vs Chunks to Symbols + RRC interpolating resampler
    data = b("bits", "analog_random_source_x", type="byte", min="0", max="4", num_samps="1000")
    mod = ch(data, b("mod", "digital_constellation_modulator", constellation="qpsk", samples_per_symbol="4"), CSNK)
    hand = ch(data, b("c2s", "digital_chunks_to_symbols_xx", in_type="byte", out_type="complex"),
              b("rs", "pfb_arb_resampler_xxx", type="ccf", rrate="4", taps="firdes.root_raised_cosine(32, 32, 1, 0.35, 352)"),
              CSNK)
    assert sig(mod) == sig(hand) == ("data_source", "linear_mod", "pulse_shaping", "resample")
    # a channelizer is a bank of frequency-translating decimating filters
    chan = ch(CTONE, b("pfb", "pfb_channelizer_ccf", nchans="4", taps="firdes.low_pass_2(1, samp_rate, 5e3, 2e3, 80)"), CSNK)
    bank = g([CTONE] + [b(f"x{i}", "freq_xlating_fir_filter_xxx", type="ccf", decim="4", center_freq=str(12000 * i),
                          taps="chan_taps") for i in (1, 2)] + [C.file_sink("s1", "complex"), C.file_sink("s2", "complex")],
             [["src", "0", "x1", "0"], ["src", "0", "x2", "0"], ["x1", "0", "s1", "0"], ["x2", "0", "s2", "0"]],
             {"chan_taps": "firdes.low_pass(1, samp_rate, 5e3, 2e3)"})
    assert sig(chan) == sig(bank) == ("filter", "mix", "resample", "tone")


def test_quantisation_by_integer_round_trip_equals_quantizer():
    q = ch(TONE, b("q", "channels_quantizer", bits="4"), SNK)
    rt = ch(TONE, C.multiply_const("up", "float", "7"), b("f2s", "blocks_float_to_short", scale="1"),
            b("s2f", "blocks_short_to_float", scale="7"), SNK)
    assert sig(q) == sig(rt) == ("quantize", "tone")


# ---------------------------------------------------------------- multiplicity, topology, oscillators

def test_multiplicity_is_ignored():
    def tones(n):
        srcs = [C.sig_source(f"s{i}", "float", str(1000 * (i + 1)), "0.2") for i in range(n)]
        return g(srcs + [b("sum", "blocks_add_xx", type="float", num_inputs=str(n)), LPF, SNK],
                 [[f"s{i}", "0", "sum", str(i)] for i in range(n)] + [["sum", "0", "lpf", "0"], ["lpf", "0", "snk", "0"]])
    assert sig(tones(2)) == sig(tones(4)) == ("combine", "filter", "tone")
    lpf2 = C.low_pass("lpf2", "float", "2000", "500")
    assert sig(ch(TONE, LPF, lpf2, SNK)) == sig(ch(TONE, LPF, SNK))
    assert C.functional_profile(tones(4))["tone"] == 4


def test_branching_graph_counts_every_branch():
    xl = b("x", "freq_xlating_fir_filter_xxx", type="ccc", decim="4", center_freq="6000", taps="[0.5, 0.5]")
    data = g([CTONE, xl, b("mag", "blocks_complex_to_mag"), CSNK, C.file_sink("env", "float", "env.dat")],
             [["src", "0", "x", "0"], ["x", "0", "snk", "0"], ["src", "0", "mag", "0"], ["mag", "0", "env", "0"]])
    assert sig(data) == ("envelope", "filter", "mix", "resample", "tone")
    assert len(C.signature_v1(data)) < 4  # v1 walked only one branch


def test_oscillator_is_absorbed_only_when_it_feeds_only_the_mixer():
    lo = C.sig_source("lo", "float", "6000", "1")
    only_lo = g([NOISE, lo, b("m", "blocks_multiply_xx", type="float"), SNK],
                [["src", "0", "m", "0"], ["lo", "0", "m", "1"], ["m", "0", "snk", "0"]])
    shared = g([NOISE, lo, b("m", "blocks_multiply_xx", type="float"), SNK, C.file_sink("ref", "float", "ref.dat")],
               [["src", "0", "m", "0"], ["lo", "0", "m", "1"], ["m", "0", "snk", "0"], ["lo", "0", "ref", "0"]])
    assert sig(only_lo) == ("mix", "noise")
    assert sig(shared) == ("mix", "noise", "tone")
    product = g([NOISE, b("n2", "analog_fastnoise_source_x", type="float", amp="1"), b("m", "blocks_multiply_xx", type="float"), SNK],
                [["src", "0", "m", "0"], ["n2", "0", "m", "1"], ["m", "0", "snk", "0"]])
    assert sig(product) == ("multiply", "noise")
    summed = g([TONE, C.sig_source("hum", "float", "60", "0.3"), b("a", "blocks_add_xx", type="float"), SNK],
               [["src", "0", "a", "0"], ["hum", "0", "a", "1"], ["a", "0", "snk", "0"]])
    assert sig(summed) == ("combine", "tone")


def test_distinct_functions_get_distinct_signatures():
    graphs = [ch(TONE, SNK), ch(TONE, LPF, SNK), ch(C.sig_source("src", "float", "1", "1", waveform="analog.GR_SQR_WAVE"), LPF, SNK),
              ch(NOISE, LPF, SNK), _two_input("blocks_multiply_xx"),
              ch(TONE, b("fm", "analog_frequency_modulator_fc", sensitivity="0.1"), CSNK),
              ch(TONE, b("d", "blocks_delay", type="float", delay="4"), SNK)]
    sigs = [sig(x) for x in graphs]
    assert len(set(sigs)) == len(sigs)


def test_unknown_blocks_and_hostile_expressions_stay_defined():
    weird = ch(TONE, b("w", "my_oot_block_ff"), SNK)
    assert sig(weird) == ("other:my_oot_block_ff", "tone")
    evil = ch(TONE, b("k", "blocks_keep_one_in_n", type="float", n="__import__('os').getpid()"), SNK)
    assert sig(evil) == ("resample", "tone")  # an expression that cannot be evaluated counts as a rate change
    loop = ch(TONE, b("r", "rational_resampler_xxx", type="fff", decim="a"), SNK, variables={"a": "b", "b": "a"})
    assert sig(loop) == ("filter", "resample", "tone")


# ---------------------------------------------------------------- split function

def test_split_is_deterministic_pinned_and_covers_all_splits():
    assert C.signature_split(("tone",)) == "test"
    assert C.signature_split(("filter", "harmonic_wave")) == "train"
    assert C.signature_split(("filter", "mix", "noise")) == "dev"
    sigs = [tuple(sorted(("tone", f"class_{i}"))) for i in range(300)]
    splits = [C.signature_split(s) for s in sigs]
    assert splits == [C.signature_split(s) for s in sigs]
    assert set(splits) == {"test", "dev", "train"}
    counts = Counter(C.signature_split((f"x{i}",)) for i in range(20000))
    for name, share in C.SHARES:
        assert abs(counts[name] / 20000 - share) < 0.02


def test_cluster_id_and_digest():
    assert C.cluster_id(("filter", "tone")) == "chain:filter|tone"
    d = C.signature_digest(("filter", "tone"))
    assert len(d) == 20 and d == C.signature_digest(("filter", "tone")) != C.signature_digest(("tone",))


# ---------------------------------------------------------------- coverage of the Tier-F items

def _item_graphs():
    for task in sorted(ITEMS.glob("*/*/task.json")):
        d = task.parent
        for p in [d / "reference.grc"] + sorted((d / "variants").glob("*.grc")):
            yield p, yaml.safe_load(p.read_text())


def test_class_map_covers_every_block_in_item_references_and_variants():
    missing = {}
    n = 0
    for p, data in _item_graphs():
        n += 1
        for blk in data["blocks"]:
            if C.block_rule(blk["id"]) is None:
                missing.setdefault(blk["id"], str(p.relative_to(REPO)))
        assert set(C.signature(data)) <= set(C.V2_CLASSES), p
    assert n > 0
    assert not missing, f"block ids without a class rule: {missing}"


def test_audit_reports_every_task():
    audit = C.audit_items(REPO)
    tasks = {r["task"] for r in audit["rows"]}
    assert tasks == {json_id.parent.name for json_id in ITEMS.glob("*/*/task.json")}
    for r in audit["rows"]:
        if r["source_kind"] == "composed_chain":
            assert r["rule_split"] == r["v2_split"]
            assert r["v2_cluster_id"] == C.cluster_id(tuple(r["v2_signature"]))
