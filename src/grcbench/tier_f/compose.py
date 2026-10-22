"""Programmatic chains ("simple hand-written L1 chains", eval plan §4.3; proposal §5.2) and the
functional signature that assigns every composed flowgraph to GRC-Bench test, dev or the training side.

Builders return compact .grc dicts from a few core GNU Radio 3.10 blocks.

Signature v2 (18 Oct 2026): a canonical FUNCTIONAL signature
===========================================================

A composed task's split is decided by a hash of its reference graph's signature, and training-side
generators may only emit composed graphs whose signature hashes to "train". Signature v1 (kept as
`signature_v1` for provenance) was the ordered tuple of block ids along one source-to-sink walk, so
the same task moved to another split when (a) an inert block such as a Throttle was inserted,
(b) commuting blocks were swapped, or (c) a rarely used but equivalent block did the same job
(round-1 reviews of gbf-test-01/02/03/04 and gbf-dev-06). v2 closes the three holes:

    signature(graph) = sorted SET of the functional classes of all enabled blocks

  * no path walk: every enabled block counts, so branching L2 graphs work and order is irrelevant (b);
  * plumbing is dropped (a) and equivalent blocks share a class (c), using the class map below;
  * multiplicity is dropped (a set, not a multiset): "sum of 2 tones" and "sum of 4 tones", one
    channel or two, one filter stage or a cascade are the same template with other numbers. Counting
    copies would let a generator escape a held-out cell by adding one more tone or one more stage.

How each enabled block is classified (`functional_profile` gives the fine classes; `COARSE` maps them
to the v2 classes; `block_rule` names the rule used for a block id):

  dropped, plumbing (a)   throttle, head, skiphead, copy, every sink (file, null, GUI, audio, ...),
                          null source, tag debug/gate, probes, message debug, selector, vector/stream
                          reshaping, bit (un)packing, and lossless type conversions (complex<->float,
                          integer->float, complex_to_real, ...)
  dropped, absorbed       constant scaling (Multiply Const, Multiply by a constant source, taps [g])
                          and constant offset (Add Const, Add of a constant source): a source's
                          amplitude/offset or a filter's gain does the same, so they are parameters
                          of the neighbouring block, not functions. Non-functional blocks (options,
                          variables, imports, notes, parameters) are ignored.
  tone                    Signal Source with a cosine or sine waveform (any type, amplitude, offset)
  harmonic_wave           Signal Source with a square, triangle or sawtooth waveform
  noise                   Noise Source, Fast Noise Source
  data_source             Vector Source, Random Source, File Source, a constant source not used as a
                          gain or offset
  message_source          Message Strobe (random), Tags Strobe, Random PDU
  mix                     frequency shift / mixing: Rotator; Multiply (or Multiply Conjugate) whose
                          input is a pure generator (a tone source through plumbing, constants,
                          conjugation) -- that generator (the local oscillator) is absorbed into the
                          mixer; Frequency Xlating filters with a non-zero centre; channelizers
  combine                 Add / Subtract of two or more non-constant streams
  multiply                Multiply of two non-generator streams (e.g. squaring, correlation)
  conjugate               Conjugate of a signal (a conjugated oscillator is part of the oscillator)
  filter                  every frequency-selective linear filter, whatever its response or
                          implementation: Low/High/Band Pass, Band Reject, DC Blocker, Moving
                          Average, IIR / single-pole IIR, de-/pre-emphasis, and FIR / FFT /
                          interpolating FIR / xlating / resampler taps designed by firdes or optfir
                          (low_pass, high_pass, band_pass, complex_band_pass, band_reject, window) or
                          given as a literal list of two or more taps, or a taps variable block.
                          Resamplers with default taps carry their own anti-alias low-pass.
  pulse_shaping           taps from firdes.root_raised_cosine or firdes.gaussian (RRC filter block,
                          RRC taps variable), the pulse shaping inside PSK/QAM/GMSK modulators, the
                          matched filter inside Polyphase Clock Sync / Symbol Sync with a PFB-MF
  hilbert                 Hilbert transform (block or firdes.hilbert taps)
  resample                any rate change: decimation, interpolation or resampling, wherever it sits
                          (Keep 1 in N, Keep M in N, Repeat, decim/interp of filters, Rational /
                          Polyphase Arbitrary / Fractional resamplers, PFB decimator/interpolator,
                          channelizers, demodulators with audio decimation, modulators' samples per
                          symbol). Rate parameters are evaluated with the graph's variables; a
                          ratio of exactly 1 is no rate change; an unknown ratio counts as one.
  fm_mod / fm_demod       angle modulation (Frequency Modulator, VCO, Phase Modulator, NBFM/WBFM Tx)
                          / Quadrature Demod, FM Demod, NBFM/WBFM Receive (composites also add their
                          filter and rate change)
  envelope / phase        Complex to Mag, Mag^2, Abs, AM Demod / Complex to Arg
  linear_mod / linear_demod  Chunks to Symbols, Constellation/PSK/QAM Modulator / Constellation
                          Decoder/Receiver, Binary Slicer, PSK/QAM Demod
  fsk_mod / fsk_demod     GMSK/GFSK/CPM modulators / GMSK/GFSK demodulators
  quantize                Quantizer, float -> integer conversions (they round to an integer grid)
  carrier_sync, timing_sync, frame_sync, equalizer, channel_model, agc, squelch, power_measure (RMS,
  average power probe), log, divide, peak_detect, delay, mux, demux, nonlinear (rail, transcendental,
  min/max, threshold, bitwise logic), fft, fec, custom (embedded Python): what their names say.
  other:<block id>        any block id not in the map (the signature stays defined; extend the map
                          when such a block becomes common)

The fine classes keep low/high/band-pass/band-reject, decimate/interpolate, scale and offset apart;
v2 merges them because the accepted variants of the first Tier-F set use them interchangeably for the
same task (a notch or band-pass instead of a high-pass to remove hum, an HPF+LPF cascade instead of a
band-pass, the gain in the filter instead of a Multiply Const). reports/tier_f_signature_v2.md gives
the numbers.

Split rule (`signature_split`): sha256(SIGNATURE_SEED + NUL + "|".join(signature)) mapped onto the
shares test 0.45, dev 0.20, train 0.35. A composed task's split must equal the split of its reference
graph's signature, and its cluster_id is "chain:" + "|".join(signature). Training-side generators may
only emit composed graphs with `signature_split(signature(g)) == "train"` and must not emit a
signature reserved by a test or dev task (handoff/training_team/flowgraph_composed_chains.md).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SIGNATURE_VERSION = 2
# Date of the class map (V2_RULES): a block id added to the map changes the signature of every graph
# that uses it, so the reserved list handed to the training team names it.
CLASS_MAP_DATE = "2026-10-22"
# block ids that got a class on CLASS_MAP_DATE (before: other:<id>)
CLASS_MAP_ADDED = ("blocks_vector_insert_x", "blocks_and_xx", "blocks_or_xx", "blocks_xor_xx",
                   "blocks_not_xx", "blocks_and_const_xx", "blocks_complex_to_magphase")
SIGNATURE_SEED_V1 = "grc-bench-l1-signatures-v1-2026-10-14"
SIGNATURE_SEED = "grc-bench-l1-signatures-v2-2026-10-18"
SHARES = (("test", 0.45), ("dev", 0.20), ("train", 0.35))


# --------------------------------------------------------------------------------------------------
# Signature v1 (14 Oct): ordered block ids along one walk. Kept for provenance only.
# --------------------------------------------------------------------------------------------------

def signature_v1(data: dict) -> tuple[str, ...]:
    """v1: block ids along the chain from source to sink (variables excluded)."""
    names = {b["name"]: b["id"] for b in data["blocks"] if not b["id"].startswith("variable")}
    succ = {c[0]: c[2] for c in data["connections"]}
    pred = {c[2] for c in data["connections"]}
    starts = sorted(n for n in names if n not in pred)
    out = []
    node = starts[0] if starts else None
    seen = set()
    while node and node not in seen:
        seen.add(node)
        out.append(names[node])
        node = succ.get(node)
    return tuple(out)


def _split(sig: tuple[str, ...], seed: str) -> str:
    h = int(hashlib.sha256((seed + "\0" + "|".join(sig)).encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for name, share in SHARES:
        acc += share
        if h < acc:
            return name
    return SHARES[-1][0]


def signature_split_v1(sig: tuple[str, ...]) -> str:
    return _split(sig, SIGNATURE_SEED_V1)


# --------------------------------------------------------------------------------------------------
# Signature v2: the class map
# --------------------------------------------------------------------------------------------------

NON_FUNCTIONAL = {"options", "import", "note", "parameter", "epy_module", "snippet", "virtual_sink",
                  "virtual_source", "pad_sink", "pad_source"}

PLUMBING = {
    # rate-neutral flow control and sample selection that does not change the signal
    "blocks_throttle", "blocks_throttle2", "blocks_head", "blocks_skiphead", "blocks_copy",
    "blocks_selector", "blocks_valve", "blocks_mute_xx", "blocks_tag_gate", "blocks_tag_debug",
    "blocks_tag_share", "blocks_annotator_alltoall", "blocks_annotator_1to1", "blocks_burst_tagger",
    "blocks_stream_to_tagged_stream", "blocks_tagged_stream_align", "blocks_endian_swap",
    # sinks, probes and debugging
    "blocks_file_sink", "blocks_file_meta_sink", "blocks_null_sink", "blocks_vector_sink_x",
    "blocks_wavfile_sink", "blocks_message_debug", "blocks_probe_signal_x", "blocks_probe_signal_vx",
    "blocks_probe_rate", "blocks_ctrlport_probe2_x", "blocks_ctrlport_probe2_c", "audio_sink",
    "blocks_udp_sink", "blocks_tcp_server_sink", "uhd_usrp_sink", "soapy_custom_sink",
    "blocks_null_source",
    # reshaping and lossless representation changes
    "blocks_stream_to_vector", "blocks_vector_to_stream", "blocks_complex_to_real",
    "blocks_complex_to_imag", "blocks_complex_to_float", "blocks_float_to_complex",
    "blocks_complex_to_interleaved_short", "blocks_complex_to_interleaved_char",
    "blocks_interleaved_short_to_complex", "blocks_interleaved_char_to_complex",
    "blocks_char_to_float", "blocks_short_to_float", "blocks_int_to_float", "blocks_uchar_to_float",
    "blocks_char_to_short", "blocks_short_to_char", "blocks_float_to_float",
    "blocks_unpack_k_bits_bb", "blocks_pack_k_bits_bb", "blocks_packed_to_unpacked_xx",
    "blocks_unpacked_to_packed_xx", "blocks_repack_bits_bb",
}
PLUMBING_PREFIXES = ("qtgui_", "zeromq_", "blocks_ctrlport_")

SCALE = {"blocks_multiply_const_vxx", "blocks_multiply_const_xx", "blocks_multiply_const_ff",
         "blocks_multiply_const_cc", "blocks_multiply_const_ii", "blocks_multiply_const_ss"}
OFFSET = {"blocks_add_const_vxx", "blocks_add_const_xx", "blocks_add_const_ff", "blocks_add_const_cc",
          "blocks_add_const_ii", "blocks_add_const_ss", "blocks_add_const_bb"}
ADDERS = {"blocks_add_xx", "blocks_sub_xx", "blocks_add_ff", "blocks_add_cc", "blocks_add_ii",
          "blocks_add_ss"}
MULTIPLIERS = {"blocks_multiply_xx", "blocks_multiply_conjugate_cc", "blocks_multiply_ff",
               "blocks_multiply_cc", "blocks_multiply_ii", "blocks_multiply_ss"}
CONJUGATE = {"blocks_conjugate_cc"}
SIG_SOURCES = {"analog_sig_source_x", "analog_sig_source_f", "analog_sig_source_c"}
CONST_SOURCES = {"analog_const_source_x"}
QUANTIZE_CONVERSIONS = {"blocks_float_to_char", "blocks_float_to_short", "blocks_float_to_int",
                        "blocks_float_to_uchar"}

# block id -> fine classes, for blocks whose class does not depend on parameters or topology
FIXED: dict[str, tuple[str, ...]] = {
    "analog_noise_source_x": ("noise",), "analog_fastnoise_source_x": ("noise",),
    "analog_noise_source_f": ("noise",), "analog_noise_source_c": ("noise",),
    "analog_random_source_x": ("data_source",), "analog_random_uniform_source_x": ("data_source",),
    "blocks_vector_source_x": ("data_source",), "blocks_file_source": ("data_source",),
    "blocks_wavfile_source": ("data_source",), "digital_glfsr_source_x": ("data_source",),
    "blocks_message_strobe": ("message_source",), "blocks_message_strobe_random": ("message_source",),
    "blocks_tags_strobe": ("message_source",), "pdu_random_pdu": ("message_source",),
    "blocks_rotator_cc": ("mix",),
    "analog_frequency_modulator_fc": ("fm_mod",), "analog_phase_modulator_fc": ("fm_mod",),
    "blocks_vco_f": ("fm_mod",), "blocks_vco_c": ("fm_mod",),
    "analog_quadrature_demod_cf": ("fm_demod",),
    "blocks_complex_to_mag": ("envelope",), "blocks_complex_to_mag_squared": ("envelope",),
    "blocks_abs_xx": ("envelope",), "blocks_complex_to_arg": ("phase",),
    "blocks_complex_to_magphase": ("envelope", "phase"),
    "blocks_rms_xx": ("power_measure",), "analog_probe_avg_mag_sqrd_x": ("power_measure",),
    "blocks_nlog10_ff": ("log",), "blocks_divide_xx": ("divide",),
    "blocks_peak_detector2_fb": ("peak_detect",), "blocks_peak_detector_xb": ("peak_detect",),
    "blocks_delay": ("delay",),
    "blocks_stream_mux": ("mux",), "blocks_patterned_interleaver": ("mux",),
    "blocks_interleave": ("mux",), "blocks_streams_to_stream": ("mux",),
    "blocks_tagged_stream_mux": ("mux",), "blocks_vector_insert_x": ("mux",),
    "blocks_stream_demux": ("demux",), "blocks_deinterleave": ("demux",),
    "blocks_stream_to_streams": ("demux",),
    "channels_channel_model": ("channel_model",), "channels_impairments": ("channel_model",),
    "channels_fading_model": ("channel_model",), "channels_selective_fading_model": ("channel_model",),
    "channels_dynamic_channel_model": ("channel_model",), "channels_cfo_model": ("channel_model",),
    "channels_sro_model": ("channel_model",),
    "channels_quantizer": ("quantize",),
    "digital_costas_loop_cc": ("carrier_sync",), "digital_fll_band_edge_cc": ("carrier_sync",),
    "analog_pll_carriertracking_cc": ("carrier_sync",), "analog_pll_freqdet_cf": ("carrier_sync",),
    "analog_pll_refout_cc": ("carrier_sync",),
    "digital_clock_recovery_mm_xx": ("timing_sync",),
    "digital_corr_est_cc": ("frame_sync",), "digital_correlate_access_code_xx_ts": ("frame_sync",),
    "digital_correlate_access_code_bb": ("frame_sync",),
    "digital_correlate_access_code_tag_xx": ("frame_sync",),
    "digital_linear_equalizer": ("equalizer",), "digital_cma_equalizer_cc": ("equalizer",),
    "digital_lms_dd_equalizer_cc": ("equalizer",), "digital_decision_feedback_equalizer": ("equalizer",),
    "digital_chunks_to_symbols_xx": ("linear_mod",), "digital_map_bb": ("linear_mod",),
    "digital_constellation_decoder_cb": ("linear_demod",),
    "digital_constellation_receiver_cb": ("linear_demod",),
    "digital_constellation_soft_decoder_cf": ("linear_demod",),
    "digital_binary_slicer_fb": ("linear_demod",),
    "digital_gmsk_demod": ("fsk_demod",), "digital_gfsk_demod": ("fsk_demod",),
    "digital_cpmmod_bc": ("fsk_mod", "resample"),
    "analog_agc_xx": ("agc",), "analog_agc2_xx": ("agc",), "analog_agc3_xx": ("agc",),
    "analog_feedforward_agc_cc": ("agc",),
    "analog_pwr_squelch_xx": ("squelch",), "analog_simple_squelch_cc": ("squelch",),
    "analog_rail_ff": ("nonlinear",), "blocks_transcendental": ("nonlinear",),
    "blocks_max_xx": ("nonlinear",), "blocks_min_xx": ("nonlinear",), "blocks_threshold_ff": ("nonlinear",),
    "blocks_and_xx": ("nonlinear",), "blocks_or_xx": ("nonlinear",), "blocks_xor_xx": ("nonlinear",),
    "blocks_not_xx": ("nonlinear",), "blocks_and_const_xx": ("nonlinear",),
    "hilbert_fc": ("hilbert",),
    "dc_blocker_xx": ("highpass",), "blocks_moving_average_xx": ("lowpass",),
    "iir_filter_xxx": ("filter",), "single_pole_iir_filter_xx": ("lowpass",),
    "analog_fm_deemph": ("lowpass",), "analog_fm_preemph": ("highpass",),
    "fft_vxx": ("fft",), "epy_block": ("custom",),
}
FIXED_PREFIXES = (("fec_", ("fec",)),)

# blocks whose classes depend on parameters (taps, rate, waveform, centre frequency)
NAMED_FILTERS = {"low_pass_filter": "lowpass", "high_pass_filter": "highpass",
                 "band_pass_filter": "bandpass", "band_reject_filter": "bandreject",
                 "root_raised_cosine_filter": "pulse_shaping"}
TAPS_FILTERS = {"fir_filter_xxx", "fft_filter_xxx", "interp_fir_filter_xxx"}
XLATING = {"freq_xlating_fir_filter_xxx", "freq_xlating_fft_filter_ccc"}
PARAM_RULES = {
    "rational_resampler_xxx", "pfb_arb_resampler_xxx", "mmse_resampler_xx", "pfb_decimator_ccf",
    "pfb_interpolator_ccf", "blocks_keep_one_in_n", "blocks_keep_m_in_n", "blocks_repeat",
    "blocks_integrate_xx", "pfb_channelizer_ccf", "pfb_channelizer_hier_ccf", "pfb_synthesizer_ccf",
    "analog_fm_demod_cf", "analog_am_demod_cf", "analog_nbfm_rx", "analog_wfm_rcv", "analog_wfm_rcv_pll",
    "analog_nbfm_tx", "analog_wfm_tx", "digital_constellation_modulator", "digital_psk_mod",
    "digital_qam_mod", "digital_gmsk_mod", "digital_gfsk_mod", "digital_pfb_clock_sync_xxx",
    "digital_symbol_sync_xx",
}

# fine class -> v2 class (None: dropped). Classes not listed map to themselves.
COARSE: dict[str, str | None] = {
    "lowpass": "filter", "highpass": "filter", "bandpass": "filter", "bandreject": "filter",
    "filter": "filter", "decimate": "resample", "interpolate": "resample", "resample": "resample",
    "scale": None, "offset": None, "identity": None,
}

V2_CLASSES = ("tone", "harmonic_wave", "noise", "data_source", "message_source", "mix", "combine",
              "multiply", "conjugate", "filter", "pulse_shaping", "hilbert", "resample", "fm_mod",
              "fm_demod", "envelope", "phase", "linear_mod", "linear_demod", "fsk_mod", "fsk_demod",
              "quantize", "carrier_sync", "timing_sync", "frame_sync", "equalizer", "channel_model",
              "agc", "squelch", "power_measure", "log", "divide", "peak_detect", "delay", "mux", "demux",
              "nonlinear", "fft", "fec", "custom")

FIRDES_CLASSES = {
    "low_pass": "lowpass", "low_pass_2": "lowpass", "high_pass": "highpass", "high_pass_2": "highpass",
    "band_pass": "bandpass", "band_pass_2": "bandpass", "complex_band_pass": "bandpass",
    "complex_band_pass_2": "bandpass", "band_reject": "bandreject", "band_reject_2": "bandreject",
    "root_raised_cosine": "pulse_shaping", "gaussian": "pulse_shaping", "hilbert": "hilbert",
    "window": "filter", "complex_band_reject": "bandreject",
}
TAPS_VARIABLE_CLASSES = {
    "variable_low_pass_filter_taps": "lowpass", "variable_high_pass_filter_taps": "highpass",
    "variable_band_pass_filter_taps": "bandpass", "variable_band_reject_filter_taps": "bandreject",
    "variable_rrc_filter_taps": "pulse_shaping",
}


def block_rule(bid: str) -> str | None:
    """Name of the explicit rule that classifies block id `bid`, or None (it would be other:<id>)."""
    if bid in NON_FUNCTIONAL or bid.startswith("variable"):
        return "non-functional"
    if bid in PLUMBING or bid.startswith(PLUMBING_PREFIXES):
        return "plumbing"
    for table, rule in ((SCALE, "scale"), (OFFSET, "offset"), (ADDERS, "adder"), (MULTIPLIERS, "multiplier"),
                        (CONJUGATE, "conjugate"), (SIG_SOURCES, "signal source"),
                        (CONST_SOURCES, "constant source"), (QUANTIZE_CONVERSIONS, "quantize"),
                        (NAMED_FILTERS, "named filter"), (TAPS_FILTERS, "taps filter"),
                        (XLATING, "xlating filter"), (PARAM_RULES, "parameter rule"), (FIXED, "fixed")):
        if bid in table:
            return rule
    if any(bid.startswith(p) for p, _ in FIXED_PREFIXES):
        return "fixed"
    return None


# --------------------------------------------------------------------------------------------------
# Signature v2: graph context, expression evaluation, taps and rate rules
# --------------------------------------------------------------------------------------------------

_MATH_FUNCS = {k: getattr(math, k) for k in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "pi",
                                              "ceil", "floor", "radians", "degrees", "e")}
_CALLS = {"int": int, "float": float, "round": round, "abs": abs, "min": min, "max": max, "pow": pow}


class _Ctx:
    def __init__(self, data: dict):
        blocks = [b for b in (data.get("blocks") or []) if isinstance(b, dict) and b.get("id")]
        self.all = {str(b.get("name")): b for b in blocks}
        self.blocks = {str(b.get("name")): b for b in blocks if _state(b) != "disabled"}
        self.preds: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.succs: dict[str, set[str]] = defaultdict(set)
        for c in data.get("connections") or []:
            if isinstance(c, (list, tuple)) and len(c) == 4:
                s, sp, d, dp = (str(x) for x in c)
                if s in self.blocks and d in self.blocks:
                    self.preds[d].append((dp, s))
                    self.succs[s].add(d)
        self._vals: dict[str, float | None] = {}

    def param(self, name: str, key: str, default: Any = None) -> Any:
        v = (self.blocks[name].get("parameters") or {}).get(key)
        return default if v is None or str(v).strip() == "" else v

    def value(self, expr: Any, _depth: int = 0) -> float | None:
        """Numeric value of a parameter expression, using the graph's variables; None if unknown."""
        if expr is None or _depth > 20:
            return None
        s = str(expr).strip()
        try:
            return float(s)
        except ValueError:
            pass
        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError:
            return None
        try:
            v = self._eval(tree.body, _depth)
        except Exception:
            return None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def _var(self, name: str, depth: int) -> float | None:
        if name in self._vals:
            return self._vals[name]
        b = self.all.get(name)
        val = None
        if b is not None and b.get("id") in ("variable", "parameter"):
            self._vals[name] = None  # cycle guard
            val = self.value((b.get("parameters") or {}).get("value"), depth + 1)
        self._vals[name] = val
        return val

    def _eval(self, node: ast.AST, depth: int) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._eval(node.operand, depth)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left, depth), self._eval(node.right, depth)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b,
                   ast.Pow: lambda: a ** b if abs(b) <= 64 else None}
            return ops[type(node.op)]()
        if isinstance(node, ast.Name):
            if node.id in ("pi", "e"):
                return _MATH_FUNCS[node.id]
            v = self._var(node.id, depth)
            if v is None:
                raise ValueError(node.id)
            return v
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("math", "numpy", "np"):
            v = _MATH_FUNCS[node.attr]
            if callable(v):
                raise ValueError(node.attr)
            return v
        if isinstance(node, ast.Call) and not node.keywords:
            args = [self._eval(a, depth) for a in node.args]
            f = node.func
            if isinstance(f, ast.Name) and f.id in _CALLS:
                return _CALLS[f.id](*args)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("math", "numpy", "np") \
                    and callable(_MATH_FUNCS.get(f.attr)):
                return _MATH_FUNCS[f.attr](*args)
        raise ValueError(ast.dump(node))


def _state(b: dict) -> str:
    return str((b.get("states") or {}).get("state", "enabled"))


def _rate(ratio: float | None) -> list[str]:
    """Fine class for an output/input rate ratio."""
    if ratio is None:
        return ["resample"]
    if abs(ratio - 1.0) < 1e-12:
        return []
    return ["interpolate"] if ratio > 1 else ["decimate"]


def _ratio(ctx: _Ctx, num: Any, den: Any) -> float | None:
    a, b = ctx.value(num), ctx.value(den)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _taps_class(ctx: _Ctx, expr: Any, default: str, _depth: int = 0) -> str:
    """Fine class of a filter given by its taps: firdes/optfir design, taps variable, literal list."""
    s = "" if expr is None else str(expr).strip()
    if s in ("", "[]", "()", "None") or _depth > 10:
        return default
    ref = ctx.all.get(s)
    if ref is not None and re.fullmatch(r"[A-Za-z_]\w*", s):
        rid = ref.get("id", "")
        if rid in TAPS_VARIABLE_CLASSES:
            return TAPS_VARIABLE_CLASSES[rid]
        if rid in ("variable", "parameter"):
            return _taps_class(ctx, (ref.get("parameters") or {}).get("value"), default, _depth + 1)
        return "filter"
    designs = {FIRDES_CLASSES.get(m, "filter") for m in re.findall(r"\b(?:firdes|optfir)\.(\w+)", s)}
    if len(designs) == 1:
        return designs.pop()
    if designs:
        return "filter"
    try:
        lit = ast.literal_eval(s)
    except Exception:
        lit = None
    if isinstance(lit, (list, tuple)):
        return "identity" if len(lit) == 1 else "filter"
    if isinstance(lit, (int, float)):
        return "identity"
    names = [n for n in re.findall(r"[A-Za-z_]\w*", s) if n in ctx.all]
    if len(names) == 1:
        return _taps_class(ctx, names[0], default, _depth + 1)
    return "filter"


def _waveform(ctx: _Ctx, name: str) -> str:
    w = str(ctx.param(name, "waveform", "analog.GR_COS_WAVE"))
    if "CONST" in w:
        return "constant"
    if any(k in w for k in ("SQR", "TRI", "SAW")):
        return "harmonic_wave"
    return "tone"


def _param_classes(ctx: _Ctx, name: str, bid: str) -> list[str]:
    p = lambda k, d=None: ctx.param(name, k, d)  # noqa: E731
    if bid in NAMED_FILTERS:
        typ = str(p("type", ""))
        ratio = ctx.value(p("interp", "1")) if typ.startswith("interp") else _ratio(ctx, 1, p("decim", "1"))
        return [NAMED_FILTERS[bid]] + _rate(ratio)
    if bid in TAPS_FILTERS:
        ratio = ctx.value(p("interp", "1")) if bid == "interp_fir_filter_xxx" else _ratio(ctx, 1, p("decim", "1"))
        return [_taps_class(ctx, p("taps"), "filter")] + _rate(ratio)
    if bid in XLATING:
        centre = ctx.value(p("center_freq", "0"))
        mix = [] if centre == 0 else ["mix"]
        return mix + [_taps_class(ctx, p("taps"), "filter")] + _rate(_ratio(ctx, 1, p("decim", "1")))
    if bid == "rational_resampler_xxx":
        return [_taps_class(ctx, p("taps"), "lowpass")] + _rate(_ratio(ctx, p("interp", "1"), p("decim", "1")))
    if bid == "pfb_arb_resampler_xxx":
        return [_taps_class(ctx, p("taps"), "lowpass")] + _rate(ctx.value(p("rrate")))
    if bid == "mmse_resampler_xx":
        return _rate(_ratio(ctx, 1, p("resamp_ratio")))
    if bid == "pfb_decimator_ccf":
        ch = ctx.value(p("channel", "0"))
        return ([] if ch == 0 else ["mix"]) + [_taps_class(ctx, p("taps"), "lowpass")] + \
            _rate(_ratio(ctx, 1, p("decim")))
    if bid == "pfb_interpolator_ccf":
        return [_taps_class(ctx, p("taps"), "lowpass")] + _rate(ctx.value(p("interp")))
    if bid == "blocks_keep_one_in_n":
        return _rate(_ratio(ctx, 1, p("n", "1")))
    if bid == "blocks_keep_m_in_n":
        return _rate(_ratio(ctx, p("m", "1"), p("n", "2")))
    if bid == "blocks_repeat":
        return _rate(ctx.value(p("interp")))
    if bid == "blocks_integrate_xx":
        return ["lowpass"] + _rate(_ratio(ctx, 1, p("decim")))
    if bid in ("pfb_channelizer_ccf", "pfb_channelizer_hier_ccf"):
        return ["mix", "lowpass", "decimate"]
    if bid == "pfb_synthesizer_ccf":
        return ["mix", "lowpass", "interpolate"]
    if bid in ("analog_fm_demod_cf", "analog_am_demod_cf"):
        head = "fm_demod" if bid == "analog_fm_demod_cf" else "envelope"
        return [head, "lowpass"] + _rate(_ratio(ctx, 1, p("audio_decim")))
    if bid == "analog_nbfm_rx":
        return ["fm_demod", "lowpass"] + _rate(_ratio(ctx, p("audio_rate"), p("quad_rate")))
    if bid in ("analog_wfm_rcv", "analog_wfm_rcv_pll"):
        return ["fm_demod", "lowpass"] + _rate(_ratio(ctx, 1, p("audio_decimation")))
    if bid == "analog_nbfm_tx":
        return ["fm_mod", "lowpass"] + _rate(_ratio(ctx, p("quad_rate"), p("audio_rate")))
    if bid == "analog_wfm_tx":
        return ["fm_mod", "lowpass"] + _rate(_ratio(ctx, p("quad_rate"), p("audio_rate")))
    if bid in ("digital_constellation_modulator", "digital_psk_mod", "digital_qam_mod"):
        return ["linear_mod", "pulse_shaping"] + _rate(ctx.value(p("samples_per_symbol", "2")))
    if bid in ("digital_gmsk_mod", "digital_gfsk_mod"):
        return ["fsk_mod", "pulse_shaping"] + _rate(ctx.value(p("samples_per_symbol", "2")))
    if bid == "digital_pfb_clock_sync_xxx":
        return ["timing_sync", _taps_class(ctx, p("taps"), "pulse_shaping")]
    if bid == "digital_symbol_sync_xx":
        mf = "PFB_MF" in str(p("resamp_type", ""))
        return ["timing_sync"] + ([_taps_class(ctx, p("pfb_mf_taps"), "pulse_shaping")] if mf else [])
    raise KeyError(bid)


# --------------------------------------------------------------------------------------------------
# Signature v2: graph pass (generators, local oscillators, gains and offsets)
# --------------------------------------------------------------------------------------------------

_TRANSPARENT = PLUMBING | SCALE | OFFSET | CONJUGATE | ADDERS | MULTIPLIERS


def _is_plumbing(bid: str) -> bool:
    return bid in PLUMBING or bid.startswith(PLUMBING_PREFIXES)


def _gen_kind(ctx: _Ctx, name: str, memo: dict, stack: frozenset = frozenset()) -> str | None:
    """'tone' / 'const' if the block's output is a pure generator (signal or constant sources seen
    through plumbing, gains, offsets, conjugation, sums and products of generators), else None."""
    if name in memo:
        return memo[name]
    b = ctx.blocks[name]
    bid = b["id"]
    kind = None
    if bid in SIG_SOURCES:
        kind = "const" if _waveform(ctx, name) == "constant" else "tone"
    elif bid in CONST_SOURCES:
        kind = "const"
    elif (bid in _TRANSPARENT or _is_plumbing(bid) or _state(b) == "bypassed") and name not in stack:
        ins = ctx.preds.get(name) or []
        kinds = [_gen_kind(ctx, s, memo, stack | {name}) for _, s in ins]
        if ins and all(k is not None for k in kinds):
            kind = "tone" if "tone" in kinds else "const"
    if not stack:
        memo[name] = kind
    return kind


def _upstream(ctx: _Ctx, roots: list[str]) -> set[str]:
    seen, todo = set(), list(roots)
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += [s for _, s in ctx.preds.get(n, [])]
    return seen


def _exclusive(ctx: _Ctx, closure: set[str], consumer: str) -> set[str]:
    """Members of `closure` all of whose outputs end, through the closure, only in `consumer`."""
    excl: set[str] = set()
    changed = True
    while changed:
        changed = False
        for n in closure - excl:
            if ctx.succs.get(n, set()) <= excl | {consumer}:
                excl.add(n)
                changed = True
    return excl


def _port_key(p: str) -> tuple[int, str]:
    return (int(p), p) if p.isdigit() else (10**6, p)


def functional_profile(data: dict) -> Counter:
    """Multiset of FINE functional classes of all enabled blocks (see the module docstring)."""
    ctx = _Ctx(data)
    memo: dict = {}
    absorbed: set[str] = set()
    classes: dict[str, list[str]] = {}

    def absorb(srcs: list[str], consumer: str) -> None:
        closure = _upstream(ctx, srcs)
        for n in _exclusive(ctx, closure, consumer):
            if ctx.blocks[n]["id"] in SIG_SOURCES | CONST_SOURCES:
                absorbed.add(n)

    # pass 1: adders and multipliers decide gains, offsets, mixers and absorbed oscillators
    for name, b in ctx.blocks.items():
        bid = b["id"]
        if _state(b) == "bypassed" or bid not in ADDERS | MULTIPLIERS:
            continue
        ports: dict[str, list[str]] = defaultdict(list)
        for dp, s in ctx.preds.get(name, []):
            ports[dp].append(s)
        kinds = {}
        for p, srcs in ports.items():
            ks = [_gen_kind(ctx, s, memo) for s in srcs]
            kinds[p] = None if any(k is None for k in ks) else ("tone" if "tone" in ks else "const")
        nonconst = [p for p, k in kinds.items() if k != "const"]
        consts = [s for p, k in kinds.items() if k == "const" for s in ports[p]]
        if len(nonconst) <= 1:
            classes[name] = ["offset" if bid in ADDERS else "scale"]
            absorb(consts, name)
        elif bid in ADDERS:
            classes[name] = ["combine"]
            absorb(consts, name)
        elif any(kinds[p] == "tone" for p in nonconst):
            classes[name] = ["mix"]
            tone_ports = [p for p in nonconst if kinds[p] == "tone"]

            def lo_rank(p: str) -> tuple:
                closure = _upstream(ctx, ports[p])
                n_src = sum(ctx.blocks[n]["id"] in SIG_SOURCES for n in closure)
                return (closure <= _exclusive(ctx, closure, name), -n_src, _port_key(p))
            absorb(ports[max(tone_ports, key=lo_rank)] + consts, name)
        else:
            classes[name] = ["multiply"]
            absorb(consts, name)

    # pass 2: every other block
    for name, b in ctx.blocks.items():
        if name in classes:
            continue
        bid = b["id"]
        if _state(b) == "bypassed" or name in absorbed or bid in NON_FUNCTIONAL or bid.startswith("variable") \
                or _is_plumbing(bid):
            classes[name] = []
        elif bid in SCALE:
            classes[name] = ["scale"]
        elif bid in OFFSET:
            classes[name] = ["offset"]
        elif bid in CONJUGATE:
            src = [s for _, s in ctx.preds.get(name, [])]
            gen = src and all(_gen_kind(ctx, s, memo) is not None for s in src)
            classes[name] = [] if gen else ["conjugate"]
        elif bid in SIG_SOURCES:
            w = _waveform(ctx, name)
            classes[name] = ["data_source" if w == "constant" else w]
        elif bid in CONST_SOURCES:
            classes[name] = ["data_source"]
        elif bid in QUANTIZE_CONVERSIONS:
            classes[name] = ["quantize"]
        elif bid in FIXED:
            classes[name] = list(FIXED[bid])
        elif any(bid.startswith(p) for p, _ in FIXED_PREFIXES):
            classes[name] = list(next(c for p, c in FIXED_PREFIXES if bid.startswith(p)))
        else:
            try:
                classes[name] = _param_classes(ctx, name, bid)
            except KeyError:
                classes[name] = [f"other:{bid}"]
    return Counter(c for cs in classes.values() for c in cs)


def signature(data: dict) -> tuple[str, ...]:
    """Signature v2: sorted set of the functional classes of all enabled blocks."""
    out = {COARSE.get(c, c) for c in functional_profile(data)}
    out.discard(None)
    return tuple(sorted(out))  # type: ignore[arg-type]


signature_v2 = signature


def signature_split(sig: tuple[str, ...]) -> str:
    """Split of a v2 signature (test 0.45, dev 0.20, train 0.35 by a seeded hash)."""
    return _split(tuple(sig), SIGNATURE_SEED)


def signature_digest(sig: tuple[str, ...]) -> str:
    """Opaque id of a v2 signature, used for the reserved list handed to the training team."""
    return hashlib.sha256((SIGNATURE_SEED + "\0reserved\0" + "|".join(sig)).encode()).hexdigest()[:20]


def cluster_id(sig: tuple[str, ...]) -> str:
    return "chain:" + "|".join(sig)


# --------------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------------

def graph(gid: str, blocks: list[dict], connections: list[list[str]], variables: dict[str, str]) -> dict[str, Any]:
    var_blocks = [{"name": k, "id": "variable", "parameters": {"value": str(v)}} for k, v in variables.items()]
    return {"options": {"parameters": {"id": gid, "generate_options": "no_gui", "output_language": "python"}},
            "blocks": var_blocks + blocks, "connections": connections, "metadata": {"file_format": 1}}


def block(name: str, bid: str, **params: Any) -> dict:
    return {"name": name, "id": bid, "parameters": {k: str(v) for k, v in params.items()}}


def chain(gid: str, blocks: list[dict], variables: dict[str, str]) -> dict[str, Any]:
    """Connect blocks in order (port 0 to port 0)."""
    conns = [[a["name"], "0", b["name"], "0"] for a, b in zip(blocks, blocks[1:])]
    return graph(gid, blocks, conns, variables)


# common building blocks (GNU Radio 3.10 ids and parameter ids)
def sig_source(name, typ, freq, amp, waveform="analog.GR_COS_WAVE", samp_rate="samp_rate", offset="0"):
    return block(name, "analog_sig_source_x", type=typ, samp_rate=samp_rate, waveform=waveform, freq=freq,
                 amp=amp, offset=offset)


def noise_source(name, typ, amp, seed=0, noise_type="analog.GR_GAUSSIAN"):
    return block(name, "analog_noise_source_x", type=typ, noise_type=noise_type, amp=amp, seed=seed)


def multiply_const(name, typ, const):
    return block(name, "blocks_multiply_const_vxx", type=typ, const=const)


def add_const(name, typ, const):
    return block(name, "blocks_add_const_vxx", type=typ, const=const)


def low_pass(name, typ, cutoff, width, samp_rate="samp_rate", gain="1", window="window.WIN_HAMMING"):
    ftype = {"float": "fir_filter_fff", "complex": "fir_filter_ccf"}[typ]
    return block(name, "low_pass_filter", type=ftype, decim="1", gain=gain, samp_rate=samp_rate,
                 cutoff_freq=cutoff, width=width, win=window, beta="6.76")


def high_pass(name, typ, cutoff, width, samp_rate="samp_rate", gain="1", window="window.WIN_HAMMING"):
    ftype = {"float": "fir_filter_fff", "complex": "fir_filter_ccf"}[typ]
    return block(name, "high_pass_filter", type=ftype, decim="1", gain=gain, samp_rate=samp_rate,
                 cutoff_freq=cutoff, width=width, win=window, beta="6.76")


def file_sink(name, typ, path="output.dat"):
    return block(name, "blocks_file_sink", type=typ, file=path, append="False", unbuffered="False", vlen="1")


def file_observation(path: str = "output.dat", n_items: int = 8192, timeout_s: float = 10.0) -> dict:
    """Observation contract: the signal written to the File Sink whose file is `path`."""
    return {"points": [{"name": "out", "sink": {"block_id": "blocks_file_sink", "params": {"file": path}}}],
            "run": {"n_items": n_items, "min_items": n_items, "timeout_s": timeout_s},
            "contract": f"the samples written to the File Sink whose file name is {path}"}


# --------------------------------------------------------------------------------------------------
# Re-split audit of items/tier_f (reports/tier_f_signature_v2.md)
# --------------------------------------------------------------------------------------------------

def audit_items(repo: Path) -> dict:
    """v1/v2 signatures and splits of every Tier-F task, its variants, twin groups, reserved set.

    The reserved set also holds the cells of the Tier-A graphs (start graphs and golden end states)."""
    import yaml

    seed_split = json.loads((repo / "splits" / "flowgraph_seed_split.json").read_text())["seeds"]
    rows = []
    for td in sorted((repo / "items" / "tier_f").glob("*/*/task.json")):
        task = json.loads(td.read_text())
        d = td.parent
        ref = yaml.safe_load((d / "reference.grc").read_text())
        kind = task["source"]["kind"]
        sig = signature(ref)
        row = {"task": task["id"], "split": task["split"], "family": task["family"], "level": task["level"],
               "source_kind": kind, "source_ref": task["source"].get("ref"), "cluster_id": task.get("cluster_id"),
               "v1_signature": list(signature_v1(ref)), "v1_split": signature_split_v1(signature_v1(ref)),
               "v2_signature": list(sig), "v2_split": signature_split(sig), "v2_cluster_id": cluster_id(sig),
               "fine_profile": dict(sorted(functional_profile(ref).items())), "variants": []}
        if kind == "composed_chain":
            row["rule_split"] = row["v2_split"]
        else:
            row["rule_split"] = (seed_split.get(task["source"].get("ref")) or {}).get("split")
        row["conflict"] = row["rule_split"] != task["split"]
        for vp in sorted((d / "variants").glob("*.grc")):
            v = yaml.safe_load(vp.read_text())
            vs = signature(v)
            row["variants"].append({"file": vp.name, "v1_signature": list(signature_v1(v)),
                                    "v1_split": signature_split_v1(signature_v1(v)),
                                    "v2_signature": list(vs), "v2_split": signature_split(vs),
                                    "same_signature": vs == sig, "same_split": signature_split(vs) == row["v2_split"],
                                    "fine_profile": dict(sorted(functional_profile(v).items()))})
        if (d / "start.grc").exists():
            s = signature(yaml.safe_load((d / "start.grc").read_text()))
            row["start_v2_signature"] = list(s)
        rows.append(row)

    groups: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        groups[tuple(r["v2_signature"])].append(r["task"])
    twins = [{"v2_signature": list(k), "v2_split": signature_split(k), "tasks": v,
              "splits": sorted({r["split"] for r in rows if r["task"] in v}),
              "kinds": sorted({r["source_kind"] for r in rows if r["task"] in v})}
             for k, v in sorted(groups.items()) if len(v) > 1]

    # Tier-A graphs: the start graphs and the golden end state (written by the Tier-A gates)
    tier_a_rows = []
    for td in sorted((repo / "items" / "tier_a").glob("*/*/task.json")):
        task = json.loads(td.read_text())
        graphs = sorted((td.parent / "start").glob("*.grc")) + sorted(td.parent.glob("golden_end_state.grc"))
        for gp in graphs:
            sig = signature(yaml.safe_load(gp.read_text()))
            if sig:                     # a graph of variables only has no functional cell to reserve
                tier_a_rows.append({"task": task["id"], "split": task["split"],
                                    "file": gp.relative_to(td.parent).as_posix(), "v2_signature": list(sig),
                                    "v2_split": signature_split(sig)})

    reserved: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        if r["split"] in ("test", "dev"):
            reserved[tuple(r["v2_signature"])].add(r["task"])
            for v in r["variants"]:
                reserved[tuple(v["v2_signature"])].add(r["task"])
    for r in tier_a_rows:
        if r["split"] in ("test", "dev"):
            reserved[tuple(r["v2_signature"])].add(r["task"])
    reserved_rows = [{"v2_signature": list(k), "v2_split": signature_split(k), "digest": signature_digest(k),
                      "tasks": sorted(v)} for k, v in sorted(reserved.items())]
    # tasks linked through a shared signature of any accepted solution (reference or variant)
    by_sig: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        for s in [r["v2_signature"]] + [v["v2_signature"] for v in r["variants"]]:
            by_sig[tuple(s)].add(r["task"])
    linked = [{"v2_signature": list(k), "tasks": sorted(v)} for k, v in sorted(by_sig.items()) if len(v) > 1]
    return {"signature_version": SIGNATURE_VERSION, "seed": SIGNATURE_SEED, "shares": dict(SHARES),
            "rows": rows, "twin_groups": twins, "solution_overlaps": linked, "reserved": reserved_rows,
            "tier_a_rows": tier_a_rows}
