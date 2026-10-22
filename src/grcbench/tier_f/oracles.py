"""V6 behavioural oracles (eval plan §4.3; proposal Table 5).

An oracle is a list of checks on the signals captured at a task's observation points. Each check
computes one feature and compares it with the value the task specification requires (its
tolerance comes from the specification, not from run-to-run noise). Repair, modify and
parameter-tune tasks group their checks into "f2p" (the requested behaviour now holds) and "p2p"
(behaviour that held in the starting flowgraph still holds) [46]; the task passes V6 only if every
check in every group passes.

A check is {"point", "feature", <feature options>, <comparison>, "group", "skip", "n"}. Every
feature is computed on the first `n` captured items after skipping `skip` items (vector items are
flattened). The comparison keys are "min" / "max" (value bounds) and "expect" with "tol" (absolute)
and/or "rel_tol" (relative); boolean features compare with "expect" (default true). A missing
observation, a malformed signal or a NaN value fails the check.

Features (FEATURES below; options in brackets):
  peak_freq      frequency of the strongest spectral component, in Hz [fs]. The mean is subtracted
                 and a Hann window applied before the FFT, so a DC level never counts as the peak
                 (a slow drift near 0 Hz still can). Signed for complex signals (fftshifted, -fs/2 to
                 fs/2); 0 to fs/2 for real signals. NaN (check fails) for fewer than 16 items or a
                 constant signal, which has no peak.
  band_fraction  share of the signal's power inside [lo, hi] Hz [fs, lo, hi, remove_mean, window].
                 Complex signals use the two-sided spectrum (lo/hi signed); real signals the
                 one-sided spectrum with every bin except DC and Nyquist counted twice (Parseval, 18
                 Oct fix). Defaults: the DC level counts as power at 0 Hz and no window is applied.
                   remove_mean: true  subtract the mean first, so the fraction measures the purity
                                      of the AC part despite a DC level (a constant signal gives 0)
                   window: "hann" | "blackman"  taper before the FFT (less leakage from a tone that
                                      does not sit on an FFT bin; the fraction is then that of the
                                      windowed signal's power)
  rms            root-mean-square amplitude (|x| for complex)
  mean           mean value; for complex signals the mean of the real part (as before 18 Oct)
  max_abs        largest absolute value (|x| for complex)
  min            smallest sample; for complex signals the smallest real part
  max            largest sample; for complex signals the largest real part
  corr_ref       best normalised correlation with a reference sequence over time shifts 0..max_lag
                 [ref, ref_imag, max_lag (512), signed]. The captured signal may lag the reference.
                   ref_imag: [...]    imaginary parts of a complex reference (same length as ref);
                                      the reference is ref + 1j*ref_imag
                   signed: false      (default) |<ref, x>| / (|ref| |x|): blind to a constant phase
                                      rotation and to a sign inversion
                   signed: true       Re<ref, x> / (|ref| |x|) (vdot conjugates ref): phase- and
                                      sign-sensitive. A constant rotation by theta scores cos(theta)
                                      (90 degrees -> 0, inversion -> -1); a conjugated (I/Q-mirrored)
                                      complex signal scores near 0 against a complex reference.
                                      The best lag still wins: for a periodic signal a shift
                                      within max_lag can undo an inversion or a rotation (a
                                      negated sine is the sine half a period later), so use an
                                      aperiodic reference (data) or a max_lag below that shift.
  equal_prefix   captured items equal a reference sequence exactly (digital data), after an offset
                 search 0..max_offset [ref, max_offset (256)]; boolean
  max_int_dev    integer-lattice deviation: max |s*x - round(s*x)| with s = "scale" (default 1)
                 [scale]. 0 when every sample is an integer multiple of 1/scale; at most 0.5.
                 For complex signals the larger of the real and imaginary parts' deviations.
                 Typical check: {"feature": "max_int_dev", "max": 1e-4}.
  int_fraction   share of samples whose s*x lies within "int_tol" (default 1e-4) of an integer
                 [scale, int_tol]; complex: both parts must. Typical check: {"min": 0.999}.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURES = ("peak_freq", "band_fraction", "rms", "mean", "max_abs", "min", "max", "corr_ref",
            "equal_prefix", "max_int_dev", "int_fraction")
WINDOWS = ("hann", "blackman")
CONSTANT_REL_ENERGY = 1e-20     # AC energy below this share of the total energy is rounding residue


def _prep(x: np.ndarray, skip: int, n: int | None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    x = x[skip:]
    if n:
        x = x[:n]
    return x


def _window(name: str | None, n: int) -> np.ndarray | None:
    if name in (None, "", "none"):
        return None
    if name == "hann":
        return np.hanning(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window {name!r} (use one of {WINDOWS})")


def _is_constant(x: np.ndarray) -> bool:
    """True if x has no AC energy beyond floating-point residue (relative to its total energy)."""
    y = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    total = float(np.sum(np.abs(y) ** 2))
    ac = float(np.sum(np.abs(y - y.mean()) ** 2))
    return ac <= CONSTANT_REL_ENERGY * total


def peak_freq(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x)
    if len(x) < 16 or _is_constant(x):            # a constant signal has no spectral peak
        return float("nan")
    w = np.hanning(len(x))
    if np.iscomplexobj(x):
        spec = np.fft.fftshift(np.abs(np.fft.fft((x - x.mean()) * w)))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
    else:
        spec = np.abs(np.fft.rfft((x - x.mean()) * w))
        freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(freqs[int(np.argmax(spec))])


def band_fraction(x: np.ndarray, fs: float, lo: float, hi: float, remove_mean: bool = False,
                  window: str | None = None) -> float:
    x = np.asarray(x)
    if remove_mean and len(x):
        if _is_constant(x):                       # no AC part at all
            return 0.0
        x = x - x.mean()
    w = _window(window, len(x))
    if w is not None:
        x = x * w
    if np.iscomplexobj(x):
        p = np.abs(np.fft.fft(x)) ** 2
        f = np.fft.fftfreq(len(x), 1.0 / fs)
    else:
        # one-sided spectrum of a real signal: every bin except DC (and Nyquist, for even
        # lengths) stands for two conjugate bins, so its power counts twice (Parseval)
        p = np.abs(np.fft.rfft(x)) ** 2
        p[1:len(x) // 2 + (len(x) % 2)] *= 2.0
        f = np.fft.rfftfreq(len(x), 1.0 / fs)
    total = p.sum()
    if total <= 0:
        return 0.0
    return float(p[(f >= lo) & (f <= hi)].sum() / total)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(np.asarray(x)) ** 2))) if len(x) else float("nan")


def sample_min(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.min(np.real(x))) if len(x) else float("nan")


def sample_max(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.max(np.real(x))) if len(x) else float("nan")


def _int_dev(x: np.ndarray, scale: float) -> np.ndarray:
    """Per-sample distance of scale*x from the nearest integer (complex: the worse of both parts)."""
    y = np.asarray(x) * scale
    if np.iscomplexobj(y):
        return np.maximum(np.abs(y.real - np.rint(y.real)), np.abs(y.imag - np.rint(y.imag)))
    y = y.astype(np.float64)
    return np.abs(y - np.rint(y))


def max_int_dev(x: np.ndarray, scale: float = 1.0) -> float:
    return float(np.max(_int_dev(x, scale))) if len(x) else float("nan")


def int_fraction(x: np.ndarray, scale: float = 1.0, int_tol: float = 1e-4) -> float:
    return float(np.mean(_int_dev(x, scale) <= int_tol)) if len(x) else float("nan")


def corr_ref(x: np.ndarray, ref: np.ndarray, max_lag: int = 512, signed: bool = False) -> float:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
    n = min(len(x), len(ref)) - max_lag
    if n < 16:
        return float("nan")
    r = ref[:n]
    rn = np.linalg.norm(r)
    if not signed:
        best = 0.0
        for lag in range(0, max_lag + 1):
            seg = x[lag:lag + n]
            denom = np.linalg.norm(seg) * rn
            if denom > 0:
                best = max(best, float(np.abs(np.vdot(r, seg)) / denom))
        return best
    best_s: float | None = None
    for lag in range(0, max_lag + 1):
        seg = x[lag:lag + n]
        denom = np.linalg.norm(seg) * rn
        if denom > 0:
            v = float(np.real(np.vdot(r, seg)) / denom)
            best_s = v if best_s is None else max(best_s, v)
    return 0.0 if best_s is None else best_s


def _reference(chk: dict) -> np.ndarray:
    ref = np.asarray(chk["ref"])
    if chk.get("ref_imag") is not None:
        imag = np.asarray(chk["ref_imag"], dtype=np.float64)
        if imag.shape != ref.shape:
            raise ValueError(f"ref_imag has shape {imag.shape}, ref has {ref.shape}")
        ref = ref.astype(np.float64) + 1j * imag
    return ref


def equal_prefix(x: np.ndarray, ref: np.ndarray, max_offset: int = 256) -> bool:
    x = np.asarray(x).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    for off in range(0, max_offset + 1):
        n = min(len(x) - off, len(ref))
        if n >= 16 and np.array_equal(x[off:off + n], ref[:n]):
            return True
    return False


def feature_value(feat: str, x: np.ndarray, chk: dict) -> Any:
    """Compute one feature of a prepared signal (options read from the check)."""
    if feat == "peak_freq":
        return peak_freq(x, chk["fs"])
    if feat == "band_fraction":
        return band_fraction(x, chk["fs"], chk["lo"], chk["hi"], bool(chk.get("remove_mean", False)),
                             chk.get("window"))
    if feat == "rms":
        return rms(x)
    if feat == "mean":
        return float(np.mean(np.real(x))) if len(x) else float("nan")
    if feat == "max_abs":
        return float(np.max(np.abs(x))) if len(x) else float("nan")
    if feat == "min":
        return sample_min(x)
    if feat == "max":
        return sample_max(x)
    if feat == "corr_ref":
        return corr_ref(x, _reference(chk), int(chk.get("max_lag", 512)), bool(chk.get("signed", False)))
    if feat == "equal_prefix":
        return equal_prefix(x, np.asarray(chk["ref"]), int(chk.get("max_offset", 256)))
    if feat == "max_int_dev":
        return max_int_dev(x, float(chk.get("scale", 1.0)))
    if feat == "int_fraction":
        return int_fraction(x, float(chk.get("scale", 1.0)), float(chk.get("int_tol", 1e-4)))
    raise ValueError(f"unknown feature {feat}")


def evaluate(oracle: dict[str, Any], obs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run every check; returns {"passed", "checks": [...], "groups": {...}}."""
    results = []
    for chk in oracle.get("checks", []):
        point = chk["point"]
        if point not in obs:
            results.append({**_brief(chk), "passed": False, "value": None, "error": "observation missing"})
            continue
        x = _prep(obs[point], int(chk.get("skip", 0)), chk.get("n"))
        try:
            val = feature_value(chk["feature"], x, chk)
        except Exception as exc:  # a malformed signal fails the check, never crashes the harness
            results.append({**_brief(chk), "passed": False, "value": None, "error": str(exc)})
            continue
        results.append({**_brief(chk), "passed": _compare(val, chk), "value": _num(val)})
    groups: dict[str, bool] = {}
    for r in results:
        groups[r["group"]] = groups.get(r["group"], True) and r["passed"]
    return {"passed": bool(results) and all(r["passed"] for r in results), "checks": results, "groups": groups}


def _compare(val: Any, chk: dict) -> bool:
    if isinstance(val, bool):
        return val is bool(chk.get("expect", True))
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if "min" in chk and val < chk["min"]:
        return False
    if "max" in chk and val > chk["max"]:
        return False
    if "expect" in chk:
        tol = chk.get("tol")
        rel = chk.get("rel_tol")
        if tol is not None and abs(val - chk["expect"]) > tol:
            return False
        if rel is not None and abs(val - chk["expect"]) > rel * abs(chk["expect"]):
            return False
    return True


def _brief(chk: dict) -> dict:
    return {"point": chk.get("point"), "feature": chk.get("feature"), "group": chk.get("group", "main"),
            "spec": {k: v for k, v in chk.items() if k not in ("ref", "ref_imag")}}


def _num(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        return round(float(v), 6)
    except Exception:
        return v
