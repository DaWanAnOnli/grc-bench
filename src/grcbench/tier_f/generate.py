"""Generated Tier-F tasks (`./gb tier-f generate`): tone chains whose expected behaviour follows exactly from
the tones they pass (pre-registration §2: 150 test tasks, 60 L1, 60 L2, 30 L3, and 45 dev tasks).

A task is a chain of stages from Signal Sources to a File Sink. Every stage maps a set of tones to a set of
tones (the model below, checked against GNU Radio 3.10.12 runs): filters keep or remove tones, mixers and the
Channel Model shift them, Rational Resamplers change the sample rate, Hilbert makes a real tone analytic,
Conjugate mirrors it, Complex To Mag and the rectifier turn a tone into a level, FM modulation and
demodulation pass a tone scaled by deviation / receiver scale, the quantizer rounds to a grid, the AGC sets the
level, the dB meter reads the power. The oracle's numbers come from the model; tolerances come from the stated
specification (frequencies 1 % or three FFT bins, levels 5 %, power shares 10 %).

Families (eval plan §4.3):
  build           the empty flowgraph; the instruction describes every stage and the output it must give
  parameter-tune  the start graph has one parameter off; the instruction describes the graph as it is and the
                  output it must give, parameter changes only (same blocks and connections as the start)
  repair          the start graph has one fault: a wrong value (with its symptom), a missing connection, or a
                  block id that does not exist; the instruction describes what the graph should do
  modify          the start graph makes one output; the instruction asks for a second output from a tap
                  point through a further stage, the first output kept (its checks are pass-to-pass)
Levels: L1 single rate, at most 5 blocks; L2 6-15 blocks; L3 a Message Strobe that retunes the first Signal
Source through its message port (cmd) while the flowgraph runs, throttled to real time.

Every task gets its own functional cell (signature v2, compose.signature): a candidate is kept only if its
cell is new (no other Tier-F or Tier-A graph has it) and hashes to the task's split, so every task is its own
cluster. L1 and L3 are the exceptions: single-rate chains of at most five blocks reach only five test cells, and
the retune template about thirty, so with `--share-cells` (attempts from SHARE_FROM on) a task of these levels may
share its cell with other tasks of its split, up to CELL_CAP[level] tasks per cell; tasks of one cell share its
cluster_id, which the clustered statistics use. Variants (further solutions) come from equivalences; mutations
are kept only where re-running the model predicts that the oracle rejects them. Everything follows from SEED and the task id.

    ./gb tier-f generate --ids gbf-test-21,gbf-test-22      write those tasks (task folders and build.py)
    ./gb tier-f generate --split test                        every new task of the split
    ./gb tier-f generate --plan                              print the id -> family / level plan
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..common.canary import CANARY
from ..common.paths import ITEMS
from . import compose as C
from . import edit

SEED = "grc-bench-tier-f-generator-v1-2026-11-04"
GENERATOR = ("grcbench.tier_f.generate", "1")
LICENSE = "CC-BY-4.0 (original GRC-Bench item; licence to be confirmed by the team)"
TF = ITEMS / "tier_f"
MAX_ATTEMPTS = 30000
EXTENDED_FROM = 400       # from this attempt on: waveform and noise sources, the phase and dB stages, closer tones
SHARE_FROM = 2000         # with --share-cells, attempts from here on may reuse a cell of their split
CELL_CAP = {"L1": 8, "L3": 2}   # tasks per cell with --share-cells; L2 keeps one task per cell
_EXTENDED = [False]


def _ext() -> bool:
    return _EXTENDED[0]

TARGETS = {
    "test": {"start": 21, "levels": {"L1": 50, "L2": 51, "L3": 29},
             "families": {"build": 30, "modify": 34, "repair": 33, "parameter-tune": 33}},
    "dev": {"start": 11, "levels": {"L1": 13, "L2": 13, "L3": 9},
            "families": {"build": 8, "modify": 10, "repair": 8, "parameter-tune": 9}},
}


# =============================================================================================== the model
@dataclass(frozen=True)
class Tone:
    f: float        # Hz; a float signal holds a*cos(2 pi f t) (f > 0), a complex one a*exp(j 2 pi f t)
    a: float


@dataclass(frozen=True)
class Sig:
    kind: str                         # "float" or "complex"
    fs: float
    tones: tuple[Tone, ...]
    dc: float = 0.0                   # a float signal's constant level
    int_scale: float | None = None    # every sample is a whole multiple of 1 / int_scale
    level_only: bool = False          # a constant level (magnitude, dB meter)
    rate: str = "samp_rate"           # the sample rate at this point, as a GRC expression
    noise: float = 0.0                # RMS of white Gaussian noise over the whole band

    def power(self) -> float:
        tp = sum(t.a ** 2 for t in self.tones) if self.kind == "complex" else sum(t.a ** 2 / 2 for t in self.tones)
        return self.dc ** 2 + tp + self.noise ** 2

    def rms(self) -> float:
        return math.sqrt(self.power())

    def strongest(self) -> Tone | None:
        """The strongest tone, if it holds at least twice the power of the next one."""
        ts = sorted((t for t in self.tones if abs(t.f) > 0), key=lambda t: -t.a)
        if not ts or (len(ts) > 1 and ts[1].a ** 2 * 2 > ts[0].a ** 2):
            return None
        return ts[0]


def tone_power(sig: Sig, t: Tone) -> float:
    return t.a ** 2 if sig.kind == "complex" else t.a ** 2 / 2


def band_share(sig: Sig, lo: float, hi: float, remove_mean: bool) -> float:
    """The share of the power within [lo, hi] Hz: the tones there plus the noise over that width."""
    total = sig.power() - (sig.dc ** 2 if remove_mean else 0)
    if total <= 0:
        return 0.0
    tones = sum(tone_power(sig, t) for t in sig.tones if lo <= (t.f if sig.kind == "complex" else abs(t.f)) <= hi)
    density = sig.noise ** 2 / (sig.fs if sig.kind == "complex" else sig.fs / 2)
    return (tones + density * (hi - lo)) / total


WAVES = {   # GNU Radio's 0..A waveforms: (waveform constant, words, harmonic k -> amplitude / A, harmonics used)
    "square": ("analog.GR_SQR_WAVE", "a square wave that switches between 0 and {a}", lambda k: 2 / (math.pi * k), "odd"),
    "saw": ("analog.GR_SAW_WAVE", "a sawtooth wave that rises from 0 to {a}", lambda k: 1 / (math.pi * k), "all"),
    "triangle": ("analog.GR_TRI_WAVE", "a triangle wave between 0 and {a}", lambda k: 4 / (math.pi ** 2 * k * k),
                 "odd"),
}


def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"


def hz(x: float) -> str:
    ax = abs(x)
    s = f"{fmt(ax / 1000)} kHz" if ax >= 1000 else f"{fmt(ax)} Hz"
    return ("-" if x < 0 else "") + s


def per(rate: str) -> str:
    """Division by a rate expression: "/samp_rate", or "/(samp_rate*2/1)"."""
    return f"/{rate}" if rate == "samp_rate" else f"/({rate})"


def _grid(rng: random.Random, lo: float, hi: float, step: float) -> float:
    return lo + step * rng.randint(0, max(0, int((hi - lo) // step)))


def _pick(rng: random.Random, options: list) -> Any:
    return options[rng.randrange(len(options))]


# =============================================================================================== graph building
class Graph:
    def __init__(self, gid: str, fs: float):
        self.gid, self.fs = gid, fs
        self.blocks: list[dict] = []
        self.conns: list[list[str]] = []
        self.used: set[str] = set()

    def add(self, base: str, bid: str, **params: Any) -> str:
        name, i = base, 1
        while name in self.used:
            i += 1
            name = f"{base}_{i}"
        self.used.add(name)
        self.blocks.append(C.block(name, bid, **params))
        return name

    def wire(self, src: str, dst: str, sport: str = "0", dport: str = "0") -> None:
        self.conns.append([src, sport, dst, dport])

    def data(self) -> dict:
        return C.graph(self.gid, copy.deepcopy(self.blocks), copy.deepcopy(self.conns), {"samp_rate": fmt(self.fs)})

    def snapshot(self) -> tuple:
        return copy.deepcopy(self.blocks), copy.deepcopy(self.conns), set(self.used)

    def restore(self, snap: tuple) -> None:
        self.blocks, self.conns, self.used = copy.deepcopy(snap[0]), copy.deepcopy(snap[1]), set(snap[2])


# =============================================================================================== stages
class Stage:
    """design(rng, sig) -> params | None; apply(sig, p) -> Sig | None; build(g, src, p) -> (names, out)."""
    kind = ""
    ids: list[str] = []
    terminal = False

    def design(self, rng: random.Random, sig: Sig) -> dict | None:
        raise NotImplementedError

    def apply(self, sig: Sig, p: dict) -> Sig | None:
        raise NotImplementedError

    def build(self, g: Graph, src: str, p: dict, sig: Sig) -> tuple[list[str], str]:
        raise NotImplementedError

    def phrase(self, p: dict, sig: Sig) -> str:
        raise NotImplementedError

    def tunables(self, p: dict, names: list[str]) -> list[dict]:
        """[{"block", "param", "text": value as GRC text, "p": params with the value off, "what"}]"""
        return []

    def mutations(self, p: dict, names: list[str]) -> list[dict]:
        """[{"op", ..., "breaks", "_p": params of the mutated stage (or "_bypass": True)}]"""
        return []


class Source(Stage):
    kind, ids = "tones", ["analog_sig_source_x"]

    def design(self, rng, sig):
        roll = rng.random() if _ext() else 1.0
        if roll < 0.2:                                # a square, sawtooth or triangle wave (float, 0..A)
            wave = _pick(rng, list(WAVES))
            f = _grid(rng, 200, sig.fs / 40, 50)
            return {"kind": "float", "wave": wave, "tones": [(f, _pick(rng, [0.5, 1.0, 2.0]))], "noise": 0.0}
        kind = _pick(rng, ["float", "complex"])
        n = _pick(rng, [1, 1, 2, 3])
        freqs: list[float] = []
        for _ in range(60):
            if len(freqs) == n:
                break
            f = _grid(rng, 400, sig.fs * 0.28, 100)
            if kind == "complex" and rng.random() < 0.3:
                f = -f
            if all(abs(f - h) >= 1000 and abs(abs(f) - abs(h)) >= 1000 for h in freqs):
                freqs.append(f)
        amps = [_pick(rng, [0.25, 0.4, 0.5, 0.6, 0.8, 1.0]) for _ in freqs]
        if not _ext():
            return {"kind": kind, "tones": [(f, a) for f, a in zip(freqs, amps)]}
        noise = _pick(rng, [0.05, 0.1, 0.2]) if roll > 0.75 else 0.0
        return {"kind": kind, "tones": [(f, a) for f, a in zip(freqs, amps)], "noise": noise}

    def apply(self, sig, p):
        if p.get("wave"):
            f, a = p["tones"][0]
            _, _, amp, which = WAVES[p["wave"]]
            ks = range(1, int(sig.fs / 2 // f) + 1)
            harm = tuple(Tone(k * f, a * amp(k)) for k in ks if which == "all" or k % 2)
            return Sig("float", sig.fs, harm, dc=a / 2)
        return Sig(p["kind"], sig.fs, tuple(Tone(f, a) for f, a in p["tones"]), noise=p.get("noise", 0.0))

    def build(self, g, src, p, sig):
        wave = WAVES[p["wave"]][0] if p.get("wave") else "analog.GR_COS_WAVE"
        names = [g.add(f"{p.get('wave') or 'tone'}_{int(abs(f))}", "analog_sig_source_x", type=p["kind"],
                       samp_rate="samp_rate", waveform=wave, freq=fmt(f), amp=fmt(a), offset="0") for f, a in p["tones"]]
        if p.get("noise"):
            names.append(g.add("noise", "analog_noise_source_x", type=p["kind"], noise_type="analog.GR_GAUSSIAN",
                               amp=fmt(p["noise"]), seed="0"))
        if len(names) == 1:
            return names, names[0]
        adder = g.add("sum_of_tones", "blocks_add_xx", type=p["kind"], num_inputs=str(len(names)))
        for i, n in enumerate(names):
            g.wire(n, adder, dport=str(i))
        return names + [adder], adder

    def phrase(self, p, sig):
        if p.get("wave"):
            f, a = p["tones"][0]
            return f"use a Signal Source to make a {hz(f)} " + WAVES[p["wave"]][1].format(a=fmt(a))[2:]
        word = "cosine" if p["kind"] == "float" else "complex tone"
        parts = [f"a {hz(f)} {word} with amplitude {fmt(a)}" for f, a in p["tones"]]
        noise = (f" and add {'complex ' if p['kind'] == 'complex' else ''}Gaussian noise of amplitude {fmt(p['noise'])} "
                 "from a Noise Source (seed 0)") if p.get("noise") else ""
        if len(parts) == 1:
            return f"use a Signal Source to make {parts[0]}" + (noise.replace(" and add", ", add") +
                                                                 " with an Add block" if noise else "")
        return "use Signal Sources to make " + ", ".join(parts[:-1]) + f" and {parts[-1]}, and add them" + \
            (noise.replace(" and add", " together with") if noise else "")

    def tunables(self, p, names):
        out = []
        for i, (f, a) in enumerate(p["tones"]):
            off = list(p["tones"])
            off[i] = (round(f * 1.3 / 100) * 100, a)
            out.append({"block": names[i], "param": "freq", "text": fmt(off[i][0]), "p": {**p, "tones": off},
                        "what": f"the frequency of the {hz(f)} tone"})
        return out

    def mutations(self, p, names):
        out = []
        for i, (f, a) in enumerate(p["tones"]):
            moved = list(p["tones"])
            moved[i] = (f * 1.25, a)
            out.append({"op": "set_param", "block": names[i], "param": "freq", "value": fmt(f * 1.25),
                        "breaks": f"the {hz(f)} tone at {hz(f * 1.25)}", "_p": {**p, "tones": moved}})
            louder = list(p["tones"])
            louder[i] = (f, a * 1.5)
            out.append({"op": "set_param", "block": names[i], "param": "amp", "value": fmt(a * 1.5),
                        "breaks": f"the {hz(f)} tone 50 % louder", "_p": {**p, "tones": louder}})
        return out


class Filter(Stage):
    kind = "filter"

    def design(self, rng, sig):
        if not _ext():
            if sig.level_only or len(sig.tones) < 2 or sig.dc:
                return None
            freqs = sorted({abs(t.f) for t in sig.tones})
            gaps = [(a, b) for a, b in zip(freqs, freqs[1:]) if b - a >= 1000]
            if not gaps:
                return None
            lo, hi = _pick(rng, gaps)
            width = _pick(rng, [200, 300, 400])
            cut = round(((lo + hi) / 2) / 50) * 50
            if cut - width / 2 - lo < 200 or hi - (cut + width / 2) < 200:
                return None
        else:
            if sig.level_only or len(sig.tones) < 2:
                return None
            freqs = sorted({abs(t.f) for t in sig.tones})
            if sig.dc:                                # a waveform: low-pass at its fundamental only (the model's
                freqs = freqs[:2]                     # harmonic series holds there; aliases sit far above)
            gaps = [(a, b) for a, b in zip(freqs, freqs[1:]) if b - a >= 500]
            if not gaps:
                return None
            lo, hi = _pick(rng, gaps)
            width = _pick(rng, [200, 200, 300]) if hi - lo < 1000 else _pick(rng, [200, 300, 400])
            cut = round(((lo + hi) / 2) / 50) * 50
            if cut - width / 2 - lo < 140 or hi - (cut + width / 2) < 140:
                return None
        ftype = "low_pass_filter" if sig.kind == "complex" or (_ext() and sig.dc) else             _pick(rng, ["low_pass_filter", "high_pass_filter"])
        return {"ftype": ftype, "cut": cut, "width": width, "gap": (lo, hi), "rate": sig.rate}

    def apply(self, sig, p):
        low = p["ftype"] == "low_pass_filter"
        keep = (lambda f: abs(f) < p["cut"]) if low else (lambda f: abs(f) > p["cut"])
        tones = tuple(t for t in sig.tones if keep(t.f))
        if not tones:
            return None
        band = (2 * p["cut"] / sig.fs) if sig.kind == "complex" else \
            ((p["cut"] if low else sig.fs / 2 - p["cut"]) / (sig.fs / 2))
        return replace(sig, tones=tones, dc=sig.dc if low else 0.0, noise=sig.noise * math.sqrt(min(1.0, band)))

    def build(self, g, src, p, sig):
        ftype = {"float": "fir_filter_fff", "complex": "fir_filter_ccf"}[sig.kind]
        name = g.add("lpf" if p["ftype"] == "low_pass_filter" else "hpf", p["ftype"], type=ftype, decim="1", gain="1",
                     samp_rate=p["rate"], cutoff_freq=fmt(p["cut"]), width=fmt(p["width"]),
                     win="window.WIN_HAMMING", beta="6.76")
        g.wire(src, name)
        self.ids = [p["ftype"], "fft_filter_xxx", "fir_filter_xxx"]
        return [name], name

    def phrase(self, p, sig):
        low = p["ftype"] == "low_pass_filter"
        return (f"{'low' if low else 'high'}-pass filter it with a {'Low' if low else 'High'} Pass Filter (cutoff "
                f"{hz(p['cut'])}, transition width {hz(p['width'])}), so that only the tones "
                f"{'below' if low else 'above'} {hz(p['cut'])} remain")

    def _other(self, p):
        lo, hi = p["gap"]
        return (hi + 2 * p["width"]) if p["ftype"] == "low_pass_filter" else max(100.0, lo - 2 * p["width"])

    def tunables(self, p, names):
        other = self._other(p)
        return [{"block": names[0], "param": "cutoff_freq", "text": fmt(other), "p": {**p, "cut": other},
                 "what": "the filter's cutoff"}]

    def mutations(self, p, names):
        other = self._other(p)
        return [{"op": "bypass", "block": names[0], "breaks": "no filter: every tone passes", "_bypass": True},
                {"op": "set_param", "block": names[0], "param": "cutoff_freq", "value": fmt(other),
                 "breaks": f"cutoff at {hz(other)}", "_p": {**p, "cut": other}}]


class Shift(Stage):
    kind, ids = "shift", ["blocks_multiply_xx"]

    def design(self, rng, sig):
        if sig.kind != "complex" or sig.level_only:
            return None
        f_lo = _pick(rng, [-1, 1]) * _grid(rng, 1000, sig.fs * 0.2, 250)
        if any(abs(t.f + f_lo) > sig.fs * 0.4 or abs(t.f + f_lo) < 300 for t in sig.tones):
            return None
        return {"lo": f_lo, "rate": sig.rate}

    def apply(self, sig, p):
        tones = tuple(Tone(t.f + p["lo"], t.a) for t in sig.tones)
        if any(abs(t.f) >= sig.fs / 2 for t in tones):
            return None
        return replace(sig, tones=tones)

    def build(self, g, src, p, sig):
        lo = g.add("lo", "analog_sig_source_x", type="complex", samp_rate=p["rate"], waveform="analog.GR_COS_WAVE",
                   freq=fmt(p["lo"]), amp="1", offset="0")
        mix = g.add("mixer", "blocks_multiply_xx", type="complex", num_inputs="2")
        g.wire(src, mix)
        g.wire(lo, mix, dport="1")
        return [lo, mix], mix

    def phrase(self, p, sig):
        return (f"shift it {'up' if p['lo'] > 0 else 'down'} by {hz(abs(p['lo']))}: multiply it with a complex "
                f"{hz(p['lo'])} tone of amplitude 1 from another Signal Source")

    def tunables(self, p, names):
        return [{"block": names[0], "param": "freq", "text": fmt(-p["lo"]), "p": {**p, "lo": -p["lo"]},
                 "what": "the local oscillator's frequency"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[0], "param": "freq", "value": fmt(-p["lo"]),
                 "breaks": "local oscillator of the wrong sign", "_p": {**p, "lo": -p["lo"]}},
                {"op": "set_param", "block": names[0], "param": "freq", "value": fmt(p["lo"] * 1.5),
                 "breaks": "local oscillator 50 % off", "_p": {**p, "lo": p["lo"] * 1.5}}]


class Resample(Stage):
    kind, ids = "resample", ["rational_resampler_xxx"]

    def design(self, rng, sig):
        if sig.level_only:
            return None
        interp, decim = _pick(rng, [(1, 2), (1, 4), (2, 1), (3, 2), (2, 3), (1, 3)])
        fs2 = sig.fs * interp / decim
        if fs2 < 8000 or fs2 > 96000 or abs(fs2 - round(fs2)) > 1e-9:
            return None
        if any(abs(t.f) > 0.3 * min(fs2, sig.fs) for t in sig.tones):
            return None
        return {"interp": interp, "decim": decim}

    def apply(self, sig, p):
        fs2 = sig.fs * p["interp"] / p["decim"]
        if any(abs(t.f) > 0.4 * min(fs2, sig.fs) for t in sig.tones):
            return None
        return replace(sig, fs=fs2, rate=f"{sig.rate}*{p['interp']}/{p['decim']}",
                       noise=sig.noise * math.sqrt(min(1.0, fs2 / sig.fs)))

    def build(self, g, src, p, sig):
        name = g.add("resampler", "rational_resampler_xxx", type={"float": "fff", "complex": "ccc"}[sig.kind],
                     interp=str(p["interp"]), decim=str(p["decim"]), taps="[]", fbw="0")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return (f"change its sample rate from {hz(sig.fs)} to {hz(sig.fs * p['interp'] / p['decim'])} with a Rational "
                f"Resampler (interpolation {p['interp']}, decimation {p['decim']})")

    def tunables(self, p, names):
        return [{"block": names[0], "param": "decim", "text": str(p["decim"] * 2), "p": {**p, "decim": p["decim"] * 2},
                 "what": "the resampler's decimation"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[0], "param": "decim", "value": str(p["decim"] * 2),
                 "breaks": "decimation doubled (read at the stated rate, every frequency doubles)",
                 "_p": {**p, "decim": p["decim"] * 2}, "_read_fs": True}]


class Conjugate(Stage):
    kind, ids = "conjugate", ["blocks_conjugate_cc"]

    def design(self, rng, sig):
        return {} if sig.kind == "complex" and not sig.level_only else None

    def apply(self, sig, p):
        return replace(sig, tones=tuple(Tone(-t.f, t.a) for t in sig.tones))

    def build(self, g, src, p, sig):
        name = g.add("conj", "blocks_conjugate_cc")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return "take its complex conjugate with a Conjugate block, which mirrors every tone to minus its frequency"

    def mutations(self, p, names):
        return [{"op": "bypass", "block": names[0], "breaks": "not conjugated", "_bypass": True}]


class Hilbert(Stage):
    kind, ids = "hilbert", ["hilbert_fc"]

    def design(self, rng, sig):
        if sig.kind != "float" or sig.level_only or sig.dc or sig.noise or any(t.f < 600 for t in sig.tones):
            return None
        if _ext() and any(not sig.fs / 20 <= t.f <= 0.45 * sig.fs for t in sig.tones):
            return None                                       # 65 taps are accurate from fs/20 up
        return {}

    def apply(self, sig, p):
        return replace(sig, kind="complex")

    def build(self, g, src, p, sig):
        name = g.add("hilbert", "hilbert_fc", num_taps="65", win="window.WIN_HAMMING", beta="6.76")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return ("turn it into a complex analytic signal with a Hilbert block (65 taps), so that each tone keeps its "
                "amplitude at its positive frequency only")


class Magnitude(Stage):
    kind, ids = "magnitude", ["blocks_complex_to_mag"]

    @property
    def terminal(self):                               # extended attempts: LevelDb may follow
        return not _ext()

    def design(self, rng, sig):
        return {} if sig.kind == "complex" and len(sig.tones) == 1 and not sig.level_only and not sig.noise else None

    def apply(self, sig, p):
        if sig.kind != "complex" or len(sig.tones) != 1:
            return None
        return Sig("float", sig.fs, (), dc=sig.tones[0].a, level_only=True)

    def build(self, g, src, p, sig):
        name = g.add("magnitude", "blocks_complex_to_mag", vlen="1")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return "take its magnitude with a Complex To Mag block"


class Rectify(Stage):
    kind, ids, terminal = "rectify", ["blocks_abs_xx"], True

    def design(self, rng, sig):
        if sig.kind != "float" or len(sig.tones) != 1 or sig.dc or sig.level_only or sig.noise or \
                sig.tones[0].f * 4 > sig.fs * 0.4:
            return None
        return {}

    def apply(self, sig, p):
        if sig.kind != "float" or len(sig.tones) != 1:
            return None
        t = sig.tones[0]
        harm = tuple(Tone(2 * k * t.f, 4 * t.a / (math.pi * (4 * k * k - 1))) for k in range(1, 8)
                     if 2 * k * t.f < sig.fs / 2)
        return Sig("float", sig.fs, harm, dc=2 * t.a / math.pi)

    def build(self, g, src, p, sig):
        name = g.add("rectifier", "blocks_abs_xx", type="float", vlen="1")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return "full-wave rectify it, so that each output sample is the absolute value of the input sample"

    def mutations(self, p, names):
        return [{"op": "bypass", "block": names[0], "breaks": "not rectified", "_bypass": True}]


class Quantize(Stage):
    kind, ids, terminal = "quantize", ["blocks_float_to_short"], True

    def design(self, rng, sig):
        if sig.kind != "float":
            return None
        scale = _pick(rng, [10, 100, 1000])
        if (sig.rms() * 3 + abs(sig.dc)) * scale > 30000:
            return None
        if _ext() and sig.rms() * scale < 5:          # a step of at most a fifth of the level
            return None
        return {"scale": scale, "back": scale}

    def apply(self, sig, p):
        k = p["scale"] / p["back"]
        return replace(sig, tones=tuple(Tone(t.f, t.a * k) for t in sig.tones), dc=sig.dc * k, int_scale=float(p["back"]))

    def build(self, g, src, p, sig):
        a = g.add("to_short", "blocks_float_to_short", vlen="1", scale=str(p["scale"]))
        b = g.add("to_float", "blocks_short_to_float", vlen="1", scale=str(p["back"]))
        g.wire(src, a)
        g.wire(a, b)
        return [a, b], b

    def phrase(self, p, sig):
        return (f"round it to steps of {fmt(1 / p['scale'])}: convert it to short integers with a Float To Short block "
                f"(scale {p['scale']}) and back with a Short To Float block (scale {p['scale']})")

    def tunables(self, p, names):
        return [{"block": names[1], "param": "scale", "text": str(p["back"] * 10), "p": {**p, "back": p["back"] * 10},
                 "what": "the Short To Float scale"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[1], "param": "scale", "value": str(p["back"] * 10),
                 "breaks": "scaled back by ten times the factor", "_p": {**p, "back": p["back"] * 10}}]


class Agc(Stage):
    kind, ids = "agc", ["analog_agc_xx", "analog_agc2_xx", "analog_agc3_xx"]

    def design(self, rng, sig):
        if sig.kind != "complex" or len(sig.tones) != 1 or sig.level_only or sig.noise:
            return None
        return {"ref": _pick(rng, [0.5, 1.0, 2.0])}

    def apply(self, sig, p):
        if len(sig.tones) != 1:
            return None
        return replace(sig, tones=(Tone(sig.tones[0].f, p["ref"]),))

    def build(self, g, src, p, sig):
        name = g.add("agc", "analog_agc_xx", type="complex", rate="1e-2", reference=fmt(p["ref"]), gain="1",
                     max_gain="65536")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return f"bring it to a magnitude of {fmt(p['ref'])} with an AGC block (rate 0.01, reference {fmt(p['ref'])})"

    def tunables(self, p, names):
        return [{"block": names[0], "param": "reference", "text": fmt(p["ref"] * 2), "p": {"ref": p["ref"] * 2},
                 "what": "the AGC's reference"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[0], "param": "reference", "value": fmt(p["ref"] * 2),
                 "breaks": "AGC reference doubled", "_p": {"ref": p["ref"] * 2}}]


class Channel(Stage):
    kind, ids = "channel", ["channels_channel_model"]

    def design(self, rng, sig):
        if sig.kind != "complex" or sig.level_only:
            return None
        off = _pick(rng, [-1, 1]) * _grid(rng, 500, 3000, 250)
        if any(abs(t.f + off) > sig.fs * 0.4 or abs(t.f + off) < 300 for t in sig.tones):
            return None
        return {"off": off, "rate": sig.rate}

    def apply(self, sig, p):
        return replace(sig, tones=tuple(Tone(t.f + p["off"], t.a) for t in sig.tones))

    def build(self, g, src, p, sig):
        name = g.add("channel", "channels_channel_model", noise_voltage="0", freq_offset=f"{fmt(p['off'])}{per(p['rate'])}",
                     epsilon="1.0", taps="[1.0]", seed="0", block_tags="False")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return (f"pass it through a Channel Model with no noise and a frequency offset of {hz(p['off'])} (freq_offset "
                f"{fmt(p['off'])}{per(p['rate'])})")

    def tunables(self, p, names):
        return [{"block": names[0], "param": "freq_offset", "text": f"{fmt(-p['off'])}{per(p['rate'])}",
                 "p": {**p, "off": -p["off"]}, "what": "the Channel Model's frequency offset"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[0], "param": "freq_offset", "value": f"{fmt(-p['off'])}{per(p['rate'])}",
                 "breaks": "offset of the wrong sign", "_p": {**p, "off": -p["off"]}}]


class FmPair(Stage):
    kind, ids = "fm", ["analog_frequency_modulator_fc", "analog_quadrature_demod_cf"]

    def design(self, rng, sig):
        if sig.kind != "float" or len(sig.tones) != 1 or sig.dc or sig.level_only or sig.noise or sig.tones[0].a > 1:
            return None
        dev = _pick(rng, [1000, 1500, 2000, 2500])
        if (dev * sig.tones[0].a + sig.tones[0].f) * 2 > sig.fs * 0.8:
            return None
        return {"dev": dev, "scale": _pick(rng, [dev, dev * 2, dev // 2]), "rate": sig.rate}

    def apply(self, sig, p):
        if len(sig.tones) != 1:
            return None
        t = sig.tones[0]
        return replace(sig, tones=(Tone(t.f, t.a * p["dev"] / p["scale"]),))

    def build(self, g, src, p, sig):
        mod = g.add("fm_mod", "analog_frequency_modulator_fc", sensitivity=f"2*math.pi*{fmt(p['dev'])}{per(p['rate'])}")
        dem = g.add("fm_demod", "analog_quadrature_demod_cf", gain=f"{p['rate']}/(2*math.pi*{fmt(p['scale'])})")
        g.wire(src, mod)
        g.wire(mod, dem)
        return [mod, dem], dem

    def phrase(self, p, sig):
        return (f"frequency-modulate it with a Frequency Mod block (sensitivity 2*pi*{fmt(p['dev'])}{per(p['rate'])}, so an "
                f"input of 1.0 deviates by {hz(p['dev'])}) and demodulate it with a Quadrature Demod block whose gain "
                f"turns {hz(p['scale'])} of deviation into an output of 1.0 (gain {p['rate']}/(2*pi*{fmt(p['scale'])}))")

    def tunables(self, p, names):
        return [{"block": names[1], "param": "gain", "text": f"{p['rate']}/(2*math.pi*{fmt(p['scale'] * 2)})",
                 "p": {**p, "scale": p["scale"] * 2}, "what": "the Quadrature Demod's gain"}]

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[1], "param": "gain",
                 "value": f"{p['rate']}/(2*math.pi*{fmt(p['scale'] * 2)})", "breaks": "receiver scale doubled",
                 "_p": {**p, "scale": p["scale"] * 2}},
                {"op": "set_param", "block": names[0], "param": "sensitivity",
                 "value": f"2*math.pi*{fmt(p['dev'] * 2)}{per(p['rate'])}", "breaks": "deviation doubled",
                 "_p": {**p, "dev": p["dev"] * 2}}]


class DbMeter(Stage):
    kind, ids, terminal = "db", ["blocks_nlog10_ff"], True

    def design(self, rng, sig):
        if sig.kind != "complex" or len(sig.tones) != 1 or sig.level_only or sig.noise:
            return None
        return {"n": _pick(rng, [500, 1000, 2000]), "k": 10}

    def apply(self, sig, p):
        if sig.kind != "complex" or len(sig.tones) != 1:
            return None
        return Sig("float", sig.fs, (), dc=p["k"] * math.log10(sig.tones[0].a ** 2), level_only=True)

    def build(self, g, src, p, sig):
        pw = g.add("power", "blocks_complex_to_mag_squared", vlen="1")
        avg = g.add("average", "blocks_moving_average_xx", type="float", length=str(p["n"]), scale=f"1/{p['n']}",
                    max_iter="4000", vlen="1")
        db = g.add("to_db", "blocks_nlog10_ff", n=str(p["k"]), k="0", vlen="1")
        g.wire(src, pw)
        g.wire(pw, avg)
        g.wire(avg, db)
        return [pw, avg, db], db

    def phrase(self, p, sig):
        return (f"measure its power in dB: a Complex To Mag^2 block, a Moving Average over {p['n']} samples (scale "
                f"1/{p['n']}) and a Log10 block with n = 10, which gives 10*log10 of the power")

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[2], "param": "n", "value": "20", "breaks": "20*log10 of a power",
                 "_p": {**p, "k": 20}}]


class Phase(Stage):
    """Complex To Arg of one complex tone: a sawtooth between -pi and pi at the tone's frequency."""
    kind, ids, terminal = "phase", ["blocks_complex_to_arg"], True

    def design(self, rng, sig):
        if sig.kind != "complex" or len(sig.tones) != 1 or sig.level_only or sig.noise or \
                not 100 <= abs(sig.tones[0].f) <= sig.fs / 40:
            return None
        return {}

    def apply(self, sig, p):
        if sig.kind != "complex" or len(sig.tones) != 1 or abs(sig.tones[0].f) < 1:
            return None
        f = abs(sig.tones[0].f)
        return Sig("float", sig.fs, tuple(Tone(k * f, 2.0 / k) for k in range(1, int(sig.fs / 2 // f) + 1)),
                   rate=sig.rate)

    def build(self, g, src, p, sig):
        name = g.add("phase", "blocks_complex_to_arg", vlen="1")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return "take its phase with a Complex To Arg block (a value between -pi and pi)"


class LevelDb(Stage):
    """20*log10 of a constant level (after Complex To Mag): the level in dB."""
    kind, ids, terminal = "level_db", ["blocks_nlog10_ff"], True

    def design(self, rng, sig):
        return {"n": 20} if sig.level_only and sig.dc > 0 else None

    def apply(self, sig, p):
        if not sig.level_only or sig.dc <= 0:
            return None
        return replace(sig, dc=p["n"] * math.log10(sig.dc))

    def build(self, g, src, p, sig):
        name = g.add("to_db", "blocks_nlog10_ff", n=str(p["n"]), k="0", vlen="1")
        g.wire(src, name)
        return [name], name

    def phrase(self, p, sig):
        return "convert the level to dB with a Log10 block with n = 20 (20*log10 of the level)"

    def mutations(self, p, names):
        return [{"op": "set_param", "block": names[0], "param": "n", "value": "10",
                 "breaks": "10*log10 of a level (half the dB value)", "_p": {"n": 10}}]


SHAPING = {"filter", "resample", "fm", "shift", "channel", "hilbert"}   # rate change, modulation or filtering
SINGLE_RATE_EXCLUDED = {"resample", "fm", "db"}
STAGES = {s.kind: s for s in (Filter(), Shift(), Resample(), Conjugate(), Hilbert(), Magnitude(), Rectify(),
                              Quantize(), Agc(), Channel(), FmPair(), DbMeter(), Phase(), LevelDb())}
BASE_POOL = 12                                        # attempts before EXTENDED_FROM draw from the first twelve


def pool() -> list[str]:
    return list(STAGES)[:BASE_POOL] if not _ext() else list(STAGES)


# =============================================================================================== a chain
@dataclass
class Step:
    stage: Stage
    p: dict
    names: list[str]
    out: str
    sig_in: Sig
    sig: Sig
    ids: list[str] = field(default_factory=list)


@dataclass
class Chain:
    graph: Graph
    steps: list[Step]
    sink: str
    out_file: str
    level: str
    msg: dict | None = None         # L3: {"strobe", "f_new", "period_ms", "key"}

    @property
    def sig(self) -> Sig:
        return self.steps[-1].sig

    def rerun(self, index: int, p: dict | None, bypass: bool = False, read_fs: float | None = None) -> Sig | None:
        """The model's output with step `index` given params `p` (or bypassed)."""
        sig = self.steps[index].sig_in
        for j, st in enumerate(self.steps):
            if j < index:
                continue
            if j == index and bypass:
                continue
            new = run_stage(st.stage, sig, p if j == index else st.p)
            if new is None:
                return None
            sig = new
        if read_fs:                                    # the oracle reads the output at the stated rate
            k = read_fs / sig.fs
            sig = replace(sig, fs=read_fs, tones=tuple(Tone(t.f * k, t.a) for t in sig.tones))
        return sig


def run_stage(stage: Stage, sig: Sig, p: dict) -> Sig | None:
    """A stage's model; a stage other than the quantizer leaves no integer grid."""
    out = stage.apply(sig, p)
    if out is not None and stage.kind != "quantize" and out.int_scale is not None:
        out = replace(out, int_scale=None)
    return out


def append(g: Graph, steps: list[Step], stage: Stage, p: dict) -> Step | None:
    sig_in = steps[-1].sig if steps else Sig("float", g.fs, ())
    out = run_stage(stage, sig_in, p)
    if out is None:
        return None
    snap = g.snapshot()
    try:
        names, o = stage.build(g, steps[-1].out if steps else "", p, sig_in)
    except Exception:                                    # noqa: BLE001 - a stage that cannot follow here
        g.restore(snap)
        return None
    st = Step(stage, p, names, o, sig_in, out, list(stage.ids))
    steps.append(st)
    return st


def make_chain(rng: random.Random, gid: str, level: str, out_file: str, msg: bool) -> Chain | None:
    fs = float(_pick(rng, [32000, 48000]))
    g = Graph(gid, fs)
    steps: list[Step] = []
    src = Source()
    p = src.design(rng, Sig("float", fs, ()))
    if msg:
        p["tones"] = p["tones"][:1]
    append(g, steps, src, p)
    want = {"L1": _pick(rng, [1, 1, 2]), "L2": _pick(rng, [2, 3, 3, 4]), "L3": _pick(rng, [1, 2, 3])}[level]
    tries = 0
    while len(steps) - 1 < want and tries < 50:
        tries += 1
        if steps[-1].stage.terminal:
            break
        stage = STAGES[_pick(rng, pool())]
        if level == "L1" and stage.kind in SINGLE_RATE_EXCLUDED:
            continue
        sp = stage.design(rng, steps[-1].sig)
        if sp is None:
            continue
        append(g, steps, stage, sp)
    if len(steps) == 1:
        return None
    info = None
    if msg:
        info = _add_strobe(rng, g, steps)
        if info is None:
            return None
    sig = steps[-1].sig
    sink = g.add("out", "blocks_file_sink", type=sig.kind, file=out_file, append="False", unbuffered="False",
                 vlen="1")
    g.wire(steps[-1].out, sink)
    count = len(g.blocks)
    if level == "L1" and (count > 5 or any(s.stage.kind == "resample" for s in steps)):
        return None
    if level == "L2" and not 6 <= count <= 15:
        return None
    if level in ("L2", "L3") and not any(s.stage.kind in SHAPING for s in steps):
        return None
    return Chain(g, steps, sink, out_file, level, info)


def _add_strobe(rng: random.Random, g: Graph, steps: list[Step]) -> dict | None:
    """L3: a Message Strobe retunes the source while the flowgraph runs (a Throttle keeps real time)."""
    first = steps[0]
    f0, a0 = first.p["tones"][0]
    f_new = round(f0 * _pick(rng, [0.7, 1.4, 1.6]) / 100) * 100
    p_new = {**first.p, "tones": [(f_new, a0)]}
    sig = first.stage.apply(first.sig_in, p_new)          # the chain's model must hold with the new frequency
    for st in steps[1:]:
        sig = st.stage.apply(sig, st.p) if sig is not None else None
    if sig is None or abs(f_new - f0) < 400:
        return None
    period = _pick(rng, [100, 200, 250])
    strobe = g.add("retune", "blocks_message_strobe", msg=_msg(f_new), period=str(period))
    g.conns.append([strobe, "strobe", first.names[0], "cmd"])
    next(b for b in g.blocks if b["name"] == first.names[0])["parameters"]["showports"] = "True"
    thr = g.add("throttle", "blocks_throttle2", type=first.sig.kind, samples_per_second="samp_rate", ignoretag="True",
                maximum="0.1", limit="auto", vlen="1")
    # the throttle goes right after the first source's output (before the adder or the first stage)
    old = [c for c in g.conns if c[0] == first.names[0] and c[1] == "0"]
    g.conns = [c for c in g.conns if not (c[0] == first.names[0] and c[1] == "0")]
    g.wire(first.names[0], thr)
    for c in old:
        g.conns.append([thr, "0", c[2], c[3]])
    # the chain's model from here on uses the new frequency (the window starts after the first message)
    steps[0] = replace(first, p=p_new, sig=first.stage.apply(first.sig_in, p_new))
    for i in range(1, len(steps)):
        steps[i] = replace(steps[i], sig_in=steps[i - 1].sig, sig=steps[i].stage.apply(steps[i - 1].sig, steps[i].p))
    return {"strobe": strobe, "f0": f0, "f_new": f_new, "period_ms": period, "throttle": thr, "p0": first.p}


# =============================================================================================== the oracle
def window(sig: Sig, msg: dict | None = None) -> dict:
    n = 16384 if sig.fs >= 24000 else 8192
    skip = 4096
    if msg:            # the first retune, then the samples the source made ahead of the Throttle before it
        skip = int(sig.fs * (msg["period_ms"] / 1000 + 0.3)) + 32768
    return {"skip": skip, "n": n}


def oracle_checks(sig: Sig, point: str, group: str, msg: dict | None = None) -> tuple[list[dict], list[str]]:
    w = window(sig, msg)
    res = sig.fs / w["n"]
    checks, props = [], []
    if sig.level_only:
        tol = 0.3 if abs(sig.dc) > 5 or sig.dc < 0 else round(0.03 * max(1.0, abs(sig.dc)), 4)
        checks += [{"point": point, "feature": "mean", "expect": round(sig.dc, 4), "tol": tol, **w, "group": group},
                   {"point": point, "feature": "min", "min": round(sig.dc - 2 * tol, 4), **w, "group": group},
                   {"point": point, "feature": "max", "max": round(sig.dc + 2 * tol, 4), **w, "group": group}]
        props.append(f"a constant level of {fmt(round(sig.dc, 3))} (within {fmt(tol)})")
        return checks, props
    strongest = sig.strongest()
    if strongest is not None:
        tol = round(max(0.01 * abs(strongest.f), 3 * res), 1)
        checks.append({"point": point, "feature": "peak_freq", "fs": sig.fs, "expect": strongest.f, "tol": tol, **w,
                       "group": group})
        props.append(f"its strongest component at {hz(strongest.f)} (within {fmt(tol)} Hz)")
    band = max(50.0, round(5 * res))
    total = sig.power()
    for t in sorted(sig.tones, key=lambda t: t.f):
        share = tone_power(sig, t) / total
        if share < 0.05:
            continue
        lo, hi = (t.f - band, t.f + band) if sig.kind == "complex" else (max(0.0, t.f - band), t.f + band)
        share_ac = band_share(sig, lo, hi, sig.kind == "float")
        checks.append({"point": point, "feature": "band_fraction", "fs": sig.fs, "lo": round(lo, 1), "hi": round(hi, 1),
                       "remove_mean": sig.kind == "float", "window": "hann",
                       "min": round(max(0.0, share_ac - 0.1 * share_ac - 0.01), 3),
                       "max": round(min(1.0, share_ac + 0.1 * share_ac + 0.01), 3), **w, "group": group})
        props.append(f"{round(100 * share_ac)} % of its {'AC ' if sig.kind == 'float' and sig.dc else ''}power within "
                     f"{fmt(band)} Hz of {hz(t.f)} (within a tenth of that share)")
    checks.append({"point": point, "feature": "rms", "expect": round(sig.rms(), 4), "rel_tol": 0.05, **w,
                   "group": group})
    props.append(f"an RMS level of {fmt(round(sig.rms(), 3))} (within 5 %)")
    if sig.dc:
        tol = round(0.03 * max(1.0, abs(sig.dc)), 4)
        checks.append({"point": point, "feature": "mean", "expect": round(sig.dc, 4), "tol": tol, **w, "group": group})
        props.append(f"an average of {fmt(round(sig.dc, 3))} (within {fmt(tol)})")
    if sig.int_scale:
        checks.append({"point": point, "feature": "max_int_dev", "scale": sig.int_scale, "max": 1e-3, **w,
                       "group": group})
        props.append(f"every sample a whole multiple of {fmt(1 / sig.int_scale)}")
    return checks, props


def model_feature(sig: Sig, chk: dict) -> float | None:
    feat = chk["feature"]
    if feat == "rms":
        return sig.rms()
    if feat == "mean":
        return sig.dc
    if feat == "peak_freq":
        s = sig.strongest()
        return s.f if s else None
    if feat == "band_fraction":
        return band_share(sig, chk["lo"], chk["hi"], bool(chk.get("remove_mean")))
    if feat in ("min", "max") and sig.level_only:
        return sig.dc
    if feat == "max_int_dev":
        return 0.0 if sig.int_scale == chk.get("scale") else 0.5
    return None


def rejected(sig: Sig | None, checks: list[dict], point: str | None = None) -> bool:
    """True if the model output fails at least one check even with twice its tolerance (a clear miss)."""
    if sig is None:
        return False
    from .oracles import _compare
    for chk in checks:
        if point is not None and chk["point"] != point:
            continue
        val = model_feature(sig, chk)
        if val is None:
            continue
        strict = dict(chk)
        for key in ("rel_tol", "tol"):
            if key in strict:
                strict[key] = strict[key] * 2
        if "min" in strict:
            strict["min"] = strict["min"] - 0.05 * abs(strict["min"]) - 0.02
        if "max" in strict:
            strict["max"] = strict["max"] + 0.05 * abs(strict["max"]) + 0.02
        if not _compare(val, strict):
            return True
    return False


# =============================================================================================== tasks
OUT_FILE = {"phase": "phase.dat", "level_db": "level_db.dat", "filter": "filtered.dat", "shift": "shifted.dat", "resample": "resampled.dat", "conjugate": "mirrored.dat",
            "hilbert": "analytic.dat", "magnitude": "level.dat", "rectify": "rectified.dat",
            "quantize": "quantized.dat", "agc": "levelled.dat", "channel": "received.dat", "fm": "recovered.dat",
            "db": "power_db.dat", "tones": "tones.dat"}
LABEL = {"filter": "the filter", "shift": "the mixer", "resample": "the Rational Resampler",
         "conjugate": "the Conjugate block", "hilbert": "the Hilbert block", "agc": "the AGC",
         "channel": "the Channel Model", "fm": "the Quadrature Demod", "quantize": "the Short To Float block",
         "tones": "the Signal Source", "magnitude": "the Complex To Mag block", "rectify": "the rectifier",
         "db": "the Log10 block", "phase": "the Complex To Arg block", "level_db": "the Log10 block"}
SAME_GRAPH = [{"kind": "same_blocks_as_start", "variables": "value_only", "group": "p2p"},
              {"kind": "same_connections_as_start", "mode": "isomorphic", "group": "p2p"}]
TAIL = ("The flowgraph should keep producing samples until it is stopped; do not limit the number of samples.")


def plan(split: str) -> dict[str, tuple[str, str]]:
    """task id -> (family, level) for every new task of the split (deterministic, interleaved)."""
    tgt = TARGETS[split]
    levels = [lv for lv, n in tgt["levels"].items() for _ in range(n)]
    fams = [fm for fm, n in tgt["families"].items() for _ in range(n)]
    r = random.Random(f"{SEED}\0plan\0{split}")
    r.shuffle(levels)
    r.shuffle(fams)
    return {f"gbf-{split}-{tgt['start'] + i:02d}": (fams[i], levels[i]) for i in range(len(levels))}


def rng_for(task_id: str, attempt: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{SEED}\0{task_id}\0{attempt}".encode()).hexdigest()[:16], 16))


def used_cells(exclude: set[str] = frozenset()) -> dict[str, str]:
    """Functional cells of every other Tier-F reference graph and every Tier-A graph -> an item id."""
    import yaml
    cells: dict[str, str] = {}
    paths = list(TF.glob("*/*/reference.grc")) + list((ITEMS / "tier_a").glob("*/*/*.grc")) + \
        list((ITEMS / "tier_a").glob("*/*/start/*.grc"))
    for p in sorted(paths):
        item = p.parent.name if p.parent.name != "start" else p.parent.parent.name
        if item in exclude:
            continue
        try:
            cells.setdefault("|".join(C.signature(yaml.safe_load(p.read_text(encoding="utf-8")))), item)
        except Exception:                       # noqa: BLE001 - a graph the classifier cannot read
            pass
    return cells


def compact(data: dict) -> dict:
    from . import grc, grcfile
    defaults = {b["id"]: grc.param_defaults(b["id"]) for b in data["blocks"]}
    out = grcfile.compact(copy.deepcopy(data), defaults)
    out["options"] = data["options"]
    diffs = grc.compact_equivalent(data, out)
    if diffs:
        raise ValueError(f"compact form differs: {diffs[:3]}")
    return out


def _clauses(chain: Chain, override: tuple[int, dict] | None = None) -> str:
    parts = []
    for i, st in enumerate(chain.steps):
        p = override[1] if override and override[0] == i else st.p
        if i == 0 and chain.msg and not (override and override[0] == 0):
            p = chain.msg["p0"]                     # the source as built; the strobe retunes it
        parts.append(st.stage.phrase(p, st.sig_in))
    text = parts[0] + "".join(f"; then {x}" for x in parts[1:])
    if chain.msg:
        m = chain.msg
        text += (f". Throttle the first Signal Source's output to real time with a Throttle block, and connect a Message "
                 f"Strobe that sends pmt.to_pmt({{'freq': {fmt(m['f_new'])}}}) every {m['period_ms']} ms to that "
                 f"Signal Source's message input cmd, so that the {hz(m['f0'])} tone is retuned to {hz(m['f_new'])} "
                 "while the flowgraph runs")
    return text


def _msg(f: float) -> str:
    return f"pmt.to_pmt({{'freq': {fmt(f)}}})"


def _observation(points: list[tuple[str, str, Sig]], msg: dict | None) -> dict:
    sig = points[0][2]
    w = window(sig, msg)
    n_items = w["skip"] + w["n"] + 1024
    timeout = 10.0 if not msg else round(max(10.0, 2 * n_items / sig.fs + 5), 1)
    return {"points": [{"name": n, "sink": {"block_id": "blocks_file_sink", "params": {"file": f, "type": s.kind}}}
                       for n, f, s in points],
            "run": {"n_items": n_items, "min_items": n_items, "timeout_s": timeout},
            "contract": "; ".join(f"'{n}': the {s.kind} samples ({hz(s.fs)}) written to the File Sink whose file name "
                                  f"is {f}" for n, f, s in points)}


def _measured(sig: Sig, msg: dict | None) -> str:
    w = window(sig, msg)
    return f"Measured over {w['n']:,} samples after the first {w['skip']:,}"


def _path_checks(chain: Chain, group: str, out_file: str, steps: list[Step] | None = None) -> list[dict]:
    checks = []
    for st in (steps if steps is not None else chain.steps[1:]):
        if st.stage.kind == "tones":
            continue
        checks.append({"kind": "path", "from_ids": ["analog_sig_source_x"], "via_ids": st.ids,
                       "to_ids": ["blocks_file_sink"], "to_params": {"file": out_file}, "group": group})
    return checks


def _mutations(chain: Chain, checks: list[dict], point: str = "out") -> list[dict]:
    """Spec-breaking edits of the reference whose model output the oracle clearly rejects."""
    kept = []
    for i, st in enumerate(chain.steps):
        for m in st.stage.mutations(st.p, st.names):
            if m.get("_bypass") and (st.sig_in.kind != st.sig.kind or len(st.names) != 1 or i == 0):
                continue
            if chain.msg and i == 0 and m.get("param") == "freq":
                continue                              # the message retunes it: its start frequency does not matter
            pred = chain.rerun(i, m.get("_p", st.p), bypass=bool(m.get("_bypass")),
                               read_fs=chain.sig.fs if m.get("_read_fs") else None)
            if rejected(pred, checks, point):
                kept.append({k: v for k, v in m.items() if not k.startswith("_")})
    if chain.msg:
        f2 = chain.msg["f_new"] * 1.25
        p2 = {**chain.steps[0].p, "tones": [(f2, chain.steps[0].p["tones"][0][1])]}
        if rejected(chain.rerun(0, p2), checks, point):
            kept.append({"op": "set_param", "block": chain.msg["strobe"], "param": "msg", "value": _msg(f2),
                         "breaks": f"the message retunes to {hz(f2)}"})
    fs = chain.graph.fs
    k = 1.25
    pred = replace(chain.sig, tones=tuple(Tone(t.f / k, t.a) for t in chain.sig.tones))
    if not chain.sig.level_only and rejected(pred, checks, point):
        kept.append({"op": "set_param", "block": "samp_rate", "param": "value", "value": fmt(fs * k),
                     "breaks": f"sample rate {hz(fs * k)} instead of {hz(fs)} (every frequency read 20 % low)"})
    return kept


def _variants_structural(ref: dict, chain: Chain, sink: str) -> dict[str, dict]:
    """Further solutions of a build / modify / structural repair task."""
    out: dict[str, dict] = {}
    v = copy.deepcopy(ref)
    feed = next(c for c in v["connections"] if c[2] == sink)
    typ = next(b for b in v["blocks"] if b["name"] == sink)["parameters"]["type"]
    v["blocks"].append(C.block("rate_limit", "blocks_throttle2", type=typ, samples_per_second="samp_rate",
                               ignoretag="True", maximum="0.1", limit="auto", vlen="1"))
    v["connections"] = [c for c in v["connections"] if c is not feed] + [[feed[0], feed[1], "rate_limit", "0"],
                                                                          ["rate_limit", "0", sink, "0"]]
    out["variants/v1_throttle.grc"] = v
    retuned = {c[2] for c in ref["connections"] if c[3] == "cmd"}
    src = next((b for b in ref["blocks"] if b["id"] == "analog_sig_source_x" and b["parameters"].get("amp") not in
                ("1", None) and (not _ext() or b["name"] not in retuned)), None)
    if src is not None:
        v = copy.deepcopy(ref)
        amp = src["parameters"]["amp"]
        next(b for b in v["blocks"] if b["name"] == src["name"])["parameters"]["amp"] = "1"
        outs = [c for c in v["connections"] if c[0] == src["name"] and c[1] == "0"]
        v["connections"] = [c for c in v["connections"] if c not in outs]
        v["blocks"].append(C.block("gain", "blocks_multiply_const_vxx", type=src["parameters"]["type"], const=amp,
                                   vlen="1"))
        v["connections"] += [[src["name"], "0", "gain", "0"]] + [["gain", "0", c[2], c[3]] for c in outs]
        out["variants/v2_gain_after_source.grc"] = v
    for b in ref["blocks"]:
        if b["id"] in ("low_pass_filter", "high_pass_filter"):
            v = copy.deepcopy(ref)
            vb = next(x for x in v["blocks"] if x["name"] == b["name"])
            prm = b["parameters"]
            kind = "low_pass" if b["id"] == "low_pass_filter" else "high_pass"
            ftype = "fff" if prm.get("type") == "fir_filter_fff" else "ccf"
            vb["id"] = "fft_filter_xxx"
            vb["parameters"] = {"type": ftype, "decim": "1", "nthreads": "1", "samp_delay": "0",
                                "taps": f"firdes.{kind}(1, {prm.get('samp_rate', 'samp_rate')}, {prm['cutoff_freq']}, "
                                        f"{prm['width']})"}
            out["variants/v3_fft_filter.grc"] = v
            break
    if len(out) < 2:
        v = copy.deepcopy(ref)
        var = next(b for b in v["blocks"] if b["name"] == "samp_rate")
        var["parameters"]["value"] = f"{fmt(chain.graph.fs / 1000)}e3"
        out["variants/v2_rate_expression.grc"] = v
    return out


def _variants_value(ref: dict, block: str, param: str) -> dict[str, dict]:
    """Further solutions of a tune / value-fault repair task: the same value written differently."""
    val = next(b for b in ref["blocks"] if b["name"] == block)["parameters"][param]
    if val.startswith("pmt.to_pmt("):                         # a message: the same dictionary written differently
        f = val.split(":", 1)[1].rstrip("})").strip()
        forms = [f"pmt.to_pmt({{'freq': {fmt(float(f))}.0}})" if "." not in f else f"pmt.to_pmt({{'freq': {f}}})",
                 f"pmt.to_pmt(dict(freq={f}))"]
    else:
        try:
            x = float(eval(val, {"__builtins__": {}}, {}))     # literal numbers only
            forms = [f"({fmt(x / 2)})*2", f"{x:.6e}"]
        except Exception:                                     # noqa: BLE001 - an expression of samp_rate etc.
            forms = [f"({val})", f"({val})*1.0"]
    out = {}
    for i, form in enumerate(forms, 1):
        v = copy.deepcopy(ref)
        next(b for b in v["blocks"] if b["name"] == block)["parameters"][param] = form
        out[f"variants/v{i}_value_written_as_{'expression' if i == 1 else 'other_form'}.grc"] = v
    return out


def _description(chain: Chain, override: tuple[int, dict] | None = None) -> str:
    return f"at a sample rate of {hz(chain.graph.fs)}, " + _clauses(chain, override)


def _steps(chain: Chain, override: tuple[int, dict] | None = None) -> str:
    """An existing graph's steps, as the imperative clauses it was built from."""
    return f"at a sample rate of {hz(chain.graph.fs)}: " + _clauses(chain, override)


def _summary(sig: Sig, want: Sig) -> str:
    """What the faulty output does differently from the required one, in words (a repair task's symptom)."""
    if sig.level_only or want.level_only:
        return f"its output is a constant level of {fmt(round(sig.dc, 3))}"
    got = sorted({round(t.f) for t in sig.tones})
    if got != sorted({round(t.f) for t in want.tones}):
        return "its output holds " + (f"components at {_and([hz(f) for f in got])}" if got else "no tone")
    if abs(sig.rms() - want.rms()) > 0.05 * want.rms():
        return f"its RMS level is {fmt(round(sig.rms(), 3))}"
    if abs(sig.dc - want.dc) > 0.03 * max(1.0, abs(want.dc)):
        return f"its average is {fmt(round(sig.dc, 3))}"
    if want.int_scale and sig.int_scale != want.int_scale:
        return f"its samples are not whole multiples of {fmt(1 / want.int_scale)}"
    shares = [f"{hz(t.f)} with amplitude {fmt(round(t.a, 3))}" for t in sorted(sig.tones, key=lambda x: x.f)]
    return "its output holds " + _and(shares)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def assemble(task_id: str, split: str, family: str, level: str, attempt: int) -> dict | None:
    """One candidate task (record, graphs, notes) from (task id, attempt), or None if it does not work out."""
    rng = rng_for(task_id, attempt)
    _EXTENDED[0] = attempt >= EXTENDED_FROM
    chain = make_chain(rng, task_id.replace("-", "_"), level, "out.dat", msg=(level == "L3"))
    if chain is None:
        return None
    if _ext() and not chain.sig.level_only and chain.sig.rms() < 0.05:
        return None                                   # no nearly silent output
    out_file = OUT_FILE.get(chain.steps[-1].stage.kind, "out.dat")
    next(b for b in chain.graph.blocks if b["name"] == chain.sink)["parameters"]["file"] = out_file
    chain.out_file = out_file
    ref = chain.graph.data()
    start, fault, extra_points, notes = None, None, [], []
    checks, props = oracle_checks(chain.sig, "out", "main" if family == "build" else "f2p", chain.msg)
    graph_checks = _path_checks(chain, "main" if family == "build" else "f2p", out_file)
    measured = _measured(chain.sig, chain.msg)
    kind_word = chain.sig.kind
    if family == "build":
        instruction = (f"In this flowgraph, {_description(chain)}. Write the result as {kind_word} samples to a File "
                       f"Sink whose file name is {out_file}. {measured}, the output should have "
                       + "; ".join(props) + f". {TAIL}")
        variants = _variants_structural(ref, chain, chain.sink)
    elif family in ("parameter-tune", "repair"):
        choices = [(i, t) for i, st in enumerate(chain.steps) for t in st.stage.tunables(st.p, st.names)
                   if not (chain.msg and i == 0 and t["param"] == "freq")]
        if chain.msg:                                 # the retune message sets the frequency that is measured
            f_off = round(chain.msg["f_new"] * 1.3 / 100) * 100
            p_off = {**chain.steps[0].p, "tones": [(f_off, chain.steps[0].p["tones"][0][1])]}
            choices.append((0, {"block": chain.msg["strobe"], "param": "msg", "text": _msg(f_off), "p": p_off,
                                "what": "the frequency the Message Strobe sends"}))
        structural = family == "repair" and rng.random() < 0.5
        if structural:
            inner = [c for c in chain.graph.conns if c[0] not in (chain.sink,) and c[2] != chain.sink
                     and c[1] == "0" and c[3] == "0"]
            if not inner:
                return None
            kind_f = _pick(rng, ["missing_connection", "nonexistent_block"])
            if kind_f == "missing_connection":
                c = _pick(rng, inner)
                start = edit.delete_connection(ref, c[0], c[2])
                fault = {"type": "missing_connection", "expected_rung": "V3", "connection": f"{c[0]}:{c[1]}->{c[2]}:{c[3]}",
                         "description": f"the connection {c[0]} -> {c[2]} is missing, so GRC reports an unconnected "
                                        "port and the flowgraph does not run"}
                symptom = "GRC reports an error and the flowgraph does not run"
            else:
                cands = [b for st in chain.steps[1:] for b in chain.graph.blocks if b["name"] in st.names]
                if not cands:
                    return None
                b = _pick(rng, cands)
                wrong = b["id"].replace("_xx", "_x").replace("filter", "filtre") if b["id"] != b["id"].replace(
                    "_xx", "_x").replace("filter", "filtre") else b["id"] + "_x"
                start = edit.set_id(ref, b["name"], wrong)
                fault = {"type": "nonexistent_block", "expected_rung": "V2", "block": b["name"], "wrong_id": wrong,
                         "correct_id": b["id"], "description": f"block {b['name']} has the id {wrong}, which does not "
                                                                "exist in GNU Radio"}
                symptom = "GRC cannot load the flowgraph"
            instruction = (f"The open flowgraph is meant to do the following, {_steps(chain)}; and write the result "
                           f"as {kind_word} samples to a File Sink whose file name is {out_file}. {measured}, the "
                           "output should have " + "; ".join(props) + f". Instead, {symptom}. Find the fault and fix "
                           f"it, keeping every block and parameter value it does not involve. {TAIL}")
            variants = _variants_structural(ref, chain, chain.sink)
        else:
            rng.shuffle(choices)
            pick = None
            for i, t in choices:
                pred = chain.rerun(i, t["p"])
                if pred is not None and rejected(pred, checks):
                    pick = (i, t, pred)
                    break
            if pick is None:
                return None
            i, t, pred = pick
            start = edit.set_param(ref, t["block"], t["param"], t["text"])
            if family == "parameter-tune":
                instruction = (f"The open flowgraph was built with these steps, {_steps(chain, (i, t['p']))}; and "
                               f"its result is written as {kind_word} samples to a File Sink whose file name is "
                               f"{out_file}. Change parameter values only (do not add, remove or reconnect any block) "
                               f"so that, {measured.lower()}, the output has " + "; ".join(props) + f". {TAIL}")
            else:
                fault = {"type": "wrong_value", "expected_rung": "V5", "block": t["block"], "param": t["param"],
                         "wrong_value": t["text"], "description": f"{t['what']} is {t['text']}"}
                instruction = (f"The open flowgraph is meant to do the following, {_steps(chain)}; and write the "
                               f"result as {kind_word} samples to a File Sink whose file name is {out_file}. "
                               f"{measured}, the output should have " + "; ".join(props) + f". Instead, "
                               f"{_summary(pred, chain.sig)}. Find the fault and fix it, keeping everything else as it is. {TAIL}")
            graph_checks = copy.deepcopy(SAME_GRAPH)
            variants = _variants_value(ref, t["block"], t["param"])
            notes.append(f"start: {t['what']} set to {t['text']}")
    else:                                                     # modify: a second output from a tap point
        taps = list(range(len(chain.steps)))
        rng.shuffle(taps)
        branch = None
        for k in taps:
            tap = chain.steps[k]
            for _ in range(12):
                stage = STAGES[_pick(rng, pool())]
                if level == "L1" and stage.kind in SINGLE_RATE_EXCLUDED:
                    continue
                sp = stage.design(rng, tap.sig)
                if sp is None:
                    continue
                g2 = Graph(chain.graph.gid, chain.graph.fs)
                g2.blocks, g2.conns, g2.used = copy.deepcopy(chain.graph.blocks), copy.deepcopy(chain.graph.conns), \
                    set(chain.graph.used)
                fake = [replace(tap, out=tap.out)]
                st = append(g2, fake, stage, sp)
                if st is None:
                    continue
                branch = (k, g2, st)
                break
            if branch:
                break
        if branch is None:
            return None
        k, g2, st = branch
        file_b = "second_" + OUT_FILE.get(st.stage.kind, "out.dat")
        sink_b = g2.add("out_b", "blocks_file_sink", type=st.sig.kind, file=file_b, append="False",
                        unbuffered="False", vlen="1")
        g2.wire(st.out, sink_b)
        start = ref
        ref = g2.data()
        if level == "L1" and len(g2.blocks) > 7:
            return None
        if level == "L2" and len(g2.blocks) > 17:
            return None
        checks_a, _ = oracle_checks(chain.sig, "out", "p2p", chain.msg)
        checks_b, props_b = oracle_checks(st.sig, "second", "f2p", chain.msg)
        checks = checks_a + checks_b
        graph_checks = _path_checks(chain, "f2p", file_b, [st])
        extra_points = [("second", file_b, st.sig)]
        tap_text = "the sum of the tones" if k == 0 and len(chain.steps[0].names) > 1 else \
            ("the Signal Source" if k == 0 else LABEL.get(chain.steps[k].stage.kind, "that stage"))
        instruction = (f"The open flowgraph was built with these steps, {_steps(chain)}; and its result is written "
                       f"as {kind_word} samples to a File Sink whose file name is {out_file}. Keep that output as it "
                       "is, and add a second one: take "
                       f"the signal at the output of {tap_text}, {st.stage.phrase(st.p, st.sig_in)}, and write it as "
                       f"{st.sig.kind} samples to a new File Sink whose file name is {file_b}. "
                       f"{_measured(st.sig, chain.msg)}, {file_b} should have " + "; ".join(props_b) + f". {TAIL}")
        variants = _variants_structural(ref, chain, sink_b)
        props = props + [f"{file_b}: " + "; ".join(props_b)]
        notes.append(f"second output tapped after step {k} ({chain.steps[k].stage.kind}) through {st.stage.kind}")
        chain_for_mut = Chain(g2, chain.steps[:k + 1] + [st], sink_b, file_b, level, chain.msg)
    try:
        ref_c = compact(ref)
    except ValueError:
        return None
    sig = C.signature(ref_c)
    if C.signature_split(sig) != split:
        return None
    muts = _mutations(chain, checks) if family != "modify" else \
        _mutations(chain, checks) + [m for m in _mutations(chain_for_mut, checks_b, "second")
                                     if m["block"] not in {n for s in chain.steps for n in s.names}]
    seen, uniq = set(), []
    for m in muts:
        key = (m["op"], m.get("block"), m.get("param"), m.get("value"))
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    if len(uniq) < 5:
        return None
    if start is not None and family != "repair" or (family == "repair" and fault["type"] == "wrong_value"):
        pass
    points = [("out", out_file, chain.sig)] + extra_points
    record = {
        "id": task_id, "tier": "F", "split": split, "version": 1, "canary": CANARY, "license": LICENSE,
        "source": {"kind": "generated_chain", "ref": "|".join(sig), "signature_version": C.SIGNATURE_VERSION,
                   "generator_seed": SEED, "attempt": attempt},
        "cluster_id": C.cluster_id(sig),
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": GENERATOR[0],
                       "generator_version": GENERATOR[1]},
        "review": {"status": "pending-team-review", "reviewers": [], "notes": ""},
        "family": family, "level": level, "instruction": instruction,
        "start_grc": "start.grc" if start is not None else None, "reference_grc": "reference.grc",
        "observation": _observation(points, chain.msg),
        "oracle": {"checks": checks, "graph_checks": graph_checks},
        "mutations": uniq,
        "spec_properties": props + ["structure: " + ("same blocks and connections as the start graph"
                                                     if graph_checks == SAME_GRAPH else
                                                     "a path from a Signal Source through each stated stage to the "
                                                     "File Sink")],
        "perturbation": {"perturbed": False, "changes": []},
        "disjointness": (f"A generated chain ({', '.join(s.stage.kind for s in chain.steps)}) whose functional "
                         f"signature {'|'.join(sig)} hashes to the {split} side and is reserved, so no training-side "
                         "generator emits it (they emit only train-side signatures and skip the reserved list); "
                         "Tier-F tasks with this signature share its cluster_id, and `./gb traces` checks the agent "
                         "traces at task level."),
        "authoring_notes": {"generator": f"{GENERATOR[0]} {GENERATOR[1]}, seed {SEED}, attempt {attempt}",
                            "chain": [s.stage.kind for s in chain.steps], "notes": notes,
                            "level": f"{level}: {len(ref_c['blocks']) - 1} blocks besides samp_rate"},
    }
    if fault:
        record["fault"] = fault
    files = {"reference.grc": ref_c}
    if start is not None:
        files["start.grc"] = compact(start) if fault is None or fault["type"] == "wrong_value" else start
    for path, v in variants.items():
        try:
            files[path] = compact(v)
        except ValueError:
            files[path] = v
    return {"record": record, "files": files, "cell": "|".join(sig)}


BUILD_PY = '''"""Authoring script for GRC-Bench Tier-F item {tid} (generated by grcbench.tier_f.generate, attempt {attempt}).

Regenerates task.json, reference.grc, start.grc and variants/*.grc from the generator's seed and this attempt
(gates.json and golden_calls.json are written by `./gb tier-f gates` and `./gb tier-a tier-f-golden`). In WSL:

    PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/grc-agent/.venv/bin/python items/tier_f/{split}/{tid}/build.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))

from grcbench.tier_f.generate import build_task  # noqa: E402

raise SystemExit(build_task("{tid}", {attempt}, HERE))
'''


def write(task_dir: Path, cand: dict) -> list[str]:
    from . import grcfile, taskio
    files = {name: grcfile.dump(data) for name, data in cand["files"].items()}
    if (task_dir / "variants").exists():
        for old in (task_dir / "variants").glob("*.grc"):
            if f"variants/{old.name}" not in files:
                old.unlink()
    return taskio.save(task_dir, cand["record"], files)


def build_task(task_id: str, attempt: int, task_dir: Path) -> int:
    """Rebuild one generated task from its attempt (its build.py)."""
    split = task_id.split("-")[1]
    family, level = plan(split)[task_id]
    cand = assemble(task_id, split, family, level, attempt)
    if cand is None:
        print(f"{task_id}: attempt {attempt} no longer gives a task", file=sys.stderr)
        return 1
    errors = write(Path(task_dir), cand)
    if errors:
        print(f"{task_id}: schema errors: {errors}", file=sys.stderr)
        return 1
    print(f"written {task_id}: {cand['cell']} ({family}, {level})")
    return 0


def cell_counts(exclude: set[str]) -> dict[str, int]:
    """How many Tier-F reference graphs have each cell (tasks in `exclude` left out)."""
    import yaml
    counts: dict[str, int] = {}
    for p in sorted(TF.glob("*/*/reference.grc")):
        if p.parent.name in exclude:
            continue
        cell = "|".join(C.signature(yaml.safe_load(p.read_text(encoding="utf-8"))))
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def generate(ids: list[str], first_attempt: int = 0, share: bool = False) -> list[str]:
    """Search each id's first attempt with a new cell of the right split (or, with `share`, a cell of its split
    used by fewer than CELL_CAP[level] tasks); write the task and its build.py."""
    done = []
    taken = used_cells(exclude=set(ids))
    counts = cell_counts(exclude=set(ids))
    for tid in ids:
        split = tid.split("-")[1]
        family, level = plan(split)[tid]
        shares = share and level in CELL_CAP
        start = max(first_attempt, SHARE_FROM) if shares else first_attempt
        stop = start + (MAX_ATTEMPTS - EXTENDED_FROM) if shares else MAX_ATTEMPTS
        for attempt in range(start, stop):
            try:
                cand = assemble(tid, split, family, level, attempt)
            except (ValueError, ZeroDivisionError, KeyError, IndexError, StopIteration) as exc:   # a candidate that breaks
                print(f"{tid} attempt {attempt}: skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
                continue
            if cand is None:
                continue
            sharing = shares and attempt >= SHARE_FROM
            if cand["cell"] in taken and not (sharing and counts.get(cand["cell"], 0) < CELL_CAP[level]):
                continue
            d = TF / split / tid
            errors = write(d, cand)
            if errors:
                print(f"{tid} attempt {attempt}: schema errors {errors}", file=sys.stderr)
                continue
            (d / "build.py").write_text(BUILD_PY.format(tid=tid, attempt=attempt, split=split), encoding="utf-8",
                                        newline="\n")
            taken[cand["cell"]] = tid
            counts[cand["cell"]] = counts.get(cand["cell"], 0) + 1
            done.append(tid)
            print(f"{tid}: {family} {level} attempt {attempt}: {cand['cell']}")
            break
        else:
            print(f"{tid}: no task in attempts {start} to {stop - 1}", file=sys.stderr)
    return done


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb tier-f generate", description=__doc__.splitlines()[0])
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--split", choices=list(TARGETS), default=None, help="every new task of the split")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--from-attempt", type=int, default=0, help="search from this attempt on")
    ap.add_argument("--missing", action="store_true", help="with --split: only the ids without a task folder")
    ap.add_argument("--share-cells", action="store_true", help="L1 and L3: allow a cell of the split already "
                    f"used (tasks per cell {CELL_CAP}; attempts from {SHARE_FROM})")
    a = ap.parse_args(argv)
    if a.plan:
        for split in TARGETS:
            for tid, (fam, lev) in plan(split).items():
                print(tid, fam, lev)
        return 0
    ids = a.ids.split(",") if a.ids else list(plan(a.split)) if a.split else []
    if a.missing:
        ids = [i for i in ids if not (TF / i.split("-")[1] / i / "task.json").exists()]
    if not ids:
        ap.error("give --ids or --split")
    done = generate(ids, a.from_attempt, a.share_cells)
    print(f"{len(done)} of {len(ids)} tasks written")
    return 0 if len(done) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
