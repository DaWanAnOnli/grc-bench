"""Tier E: speed and memory on the target hardware (eval plan §4.5; proposal §5.4).

Eval plan §4.5: "For each compression level (BF16, Q8_0, Q5_K_M and Q4_K_M) we measure
prompt-processing and generation speed in tokens per second, time to first token, and peak RAM and
GPU memory, at 4K, 16K, 32K and 64K tokens of context. Measurements use llama-bench with a standard
protocol (median of 5 runs, warm cache)." One `run` measures one GGUF file (one compression level)
on one hardware setup (HARDWARE):

  H1      CPU-only laptop, 8-16 cores, 32 GB RAM (team)
  H2      GPU limited to 4 GB, MoE experts kept in CPU memory: the reference hardware for H4
          ("the MoE model reaches at least 80 tokens/s on the reference hardware at Q4_K_M", §3)
  H3      desktop GPU: Qoherent's RTX 3090 (24 GB)
  server  rented A100 80GB (North-Mini-Code BF16 and Q8_0, which do not fit H1-H3)

The protocol (`./gb tier-e protocol`; the same for every model and setup, recorded in every result):

  * llama-bench of the pinned llama.cpp build (~/grc-bench-cache/llama.cpp/<LLAMA_CPP_BUILD>, run with
    LD_LIBRARY_PATH set to that directory). `--llama-dir` selects another build (H1 needs the
    Nanbeige fork build); the build number, commit and backends llama-bench reports are recorded.
  * Per context C, two llama-bench processes, so that each process's peak RAM belongs to one context:
      prompt  `-p C -n 0 -d 0`      a C-token prompt into an empty KV cache: prompt-processing
                                    tokens/s at C, and the prefill part of time to first token;
      depth   `-p 512 -n 128 -d C`  with C tokens already in the KV cache: generation tokens/s
                                    (tg128) and incremental prompt processing (pp512: a new agent
                                    turn on a cached prefix, §7.1) at C. llama-bench fills the depth
                                    once per process and restores it for every repetition.
  * Warm cache: one short warm-up invocation first (it pages the weights in; its numbers are
    discarded), and llama-bench's own warm-up run before every test (discarded). Then `-r 5`; we
    report the median of the 5 per-run samples (llama-bench itself reports only mean and SD).
  * Time to first token at C = the median prefill time of the C-token prompt + one decode step at
    depth C (1 / median generation tokens/s). llama-bench samples no tokens, so TTFT is derived; both
    parts are recorded.
  * Fixed settings: threads (default: the physical cores; fixed for the whole run), batch 2048,
    micro-batch 512, KV cache q8_0 for K and V (the 8-bit cache of every Tier-A run, §7.1), flash
    attention on (llama.cpp needs it for a quantized V cache), weights loaded by mmap. GPU layers:
    0 on H1, all on H2, H3 and the server; H2 places the MoE experts in CPU memory with
    `--n-cpu-moe` or `--override-tensor`. Any departure from these is listed as a deviation.
  * Peak RAM: the llama-bench process's peak resident set, ru_maxrss from wait4 (the kernel's VmHWM
    at exit; /proc/<pid>/status VmHWM is also sampled while it runs, as a cross-check). Mapped
    weight pages count, as they occupy RAM. Linux folds the forked child's pre-exec RSS (a copy of
    this Python process, tens of MB) into ru_maxrss as a maximum, not a sum, so the figure is exact
    whenever llama-bench's own peak is larger, as it is for any model of more than a few MB.
  * Peak GPU memory: nvidia-smi memory.used sampled every second, peak minus the value just before
    the process started (device-wide: WSL cannot attribute GPU memory to a process); "n/a" when the
    build has no GPU backend library (libggml-cuda.so etc.) or llama-bench used only the CPU backend.
  * Contention: before every llama-bench process we wait until no other llama-server / llama-bench
    process is running (poll every 30 s, up to 60 min); the 1-minute load average before and after
    each process, and any llama process that appeared during it, are recorded.
  * `--extend C --extend-if-under S` adds context C only if the previous context's whole
    measurement (both processes, warm-ups included) took under S seconds; a context that is not run
    is recorded with the reason.

`--purpose plumbing` (the default) labels the output "not a Tier-E result"; only `--purpose result`,
for a GRC-Bench model on a declared hardware setup, produces a Tier-E result. The hardware label is
checked against its definition and every mismatch is recorded. Tier E runs no benchmark item, so it
does not touch the test-run log (§4.1).

Outputs: reports/tier_e/<name>.json (protocol, settings, machine, model, per-run samples, medians),
reports/tier_e/<name>.md and the index reports/tier_e/summary.md (both rendered from the JSON files
only, so `report` is deterministic). Raw llama-bench output and logs stay in
~/grc-bench-work/tier_e/<name>/. The JSON is saved after every llama-bench process, so a long run
that is stopped keeps what it measured: SIGTERM, SIGHUP or Ctrl-C mark it "failed" with the reason
and list the context it stopped in as incomplete; a record left at "running" (the process was
killed outright, e.g. by a shutdown) is rendered as partial.

    ./gb tier-e run --model PATH --hardware {H1,H2,H3,server} [--contexts 4096,16384]
                    [--extend 32768 --extend-if-under 180] [--threads N] [--purpose plumbing|result]
                    [--name NAME] [--machine-label TEXT] [--note TEXT] [--dry-run]
    ./gb tier-e report        rebuild every reports/tier_e/*.md from the JSON files
    ./gb tier-e protocol      print the protocol as JSON
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable

from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import CACHE, LLAMA_CPP_BUILD, REPORTS, WORK

LLAMA_DIR = CACHE / "llama.cpp" / LLAMA_CPP_BUILD
OUT_DIR = REPORTS / "tier_e"
RAW_DIR = WORK / "tier_e"
SUMMARY_NAME = "summary.md"

PROTOCOL_VERSION = "tier-e/1"
CONTEXTS = (4096, 16384, 32768, 65536)
REPETITIONS = 5
GEN_TOKENS = 128
INCREMENTAL_PROMPT = 512
BATCH = 2048
UBATCH = 512
KV_CACHE = "q8_0"
FLASH_ATTN = "on"
LOAD_MODE = "mmap"
WARMUP = {"n_prompt": 512, "n_gen": 16, "n_depth": 0, "repetitions": 1}
IDLE_POLL_S = 30.0
IDLE_MAX_S = 3600.0
TIMEOUT_S = 6 * 3600.0           # per llama-bench process
CONTENDERS = ("llama-server", "llama-bench", "llama-cli", "llama-completion")
PURPOSES = ("plumbing", "result")
H4_TARGET_TS = 80.0
FA_CODES = {"on": 1, "off": 0, "auto": -1}
KV_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
GPU_BACKEND_WORDS = ("cuda", "vulkan", "hip", "sycl", "musa", "opencl", "cann", "metal")
NON_GPU_BACKENDS = {"CPU", "RPC", "BLAS", ""}
GIB = 1024 ** 3

HARDWARE: dict[str, dict[str, Any]] = {
    "H1": {"description": "CPU-only laptop, 8-16 cores, 32 GB RAM (team)", "gpu": False,
           "n_gpu_layers": 0, "cores": (8, 16), "ram_gib": (28.0, 34.0)},
    "H2": {"description": "GPU limited to 4 GB, MoE experts kept in CPU memory (the H4 reference hardware)",
           "gpu": True, "n_gpu_layers": 999, "gpu_limit_gib": 4.0},
    "H3": {"description": "desktop GPU: Qoherent's RTX 3090 (24 GB)", "gpu": True, "n_gpu_layers": 999,
           "gpu_name": "3090"},
    "server": {"description": "server reference: rented A100 80GB (North-Mini-Code BF16 and Q8_0)",
               "gpu": True, "n_gpu_layers": 999, "gpu_name": "A100"},
}


# --------------------------------------------------------------------------------------------
# Protocol and settings
# --------------------------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    """The fixed protocol, recorded verbatim in every run."""
    return {
        "version": PROTOCOL_VERSION,
        "source": "eval plan §4.5 (median of 5 runs, warm cache, 4K-64K context); §7.1 (8-bit KV cache)",
        "contexts": list(CONTEXTS),
        "repetitions": REPETITIONS,
        "statistic": "median of the per-run samples of each llama-bench test",
        "warmup": ("one short warm-up llama-bench invocation per run (pp512 + tg16, 1 repetition; "
                   "discarded) to page the weights in, plus llama-bench's own warm-up run before "
                   "every test (discarded)"),
        "tests_per_context": {
            "prompt": "llama-bench -p C -n 0 -d 0: a C-token prompt into an empty KV cache",
            "depth": (f"llama-bench -p {INCREMENTAL_PROMPT} -n {GEN_TOKENS} -d C: tg{GEN_TOKENS} and "
                      f"pp{INCREMENTAL_PROMPT} with C tokens already in the KV cache"),
        },
        "ttft": ("median prefill time of the C-token prompt + one decode step at depth C "
                 f"(1 / median tg{GEN_TOKENS} tokens/s); derived, both parts recorded"),
        "batch": BATCH,
        "ubatch": UBATCH,
        "cache_type_k": KV_CACHE,
        "cache_type_v": KV_CACHE,
        "flash_attn": FLASH_ATTN,
        "load_mode": LOAD_MODE,
        "threads": "physical cores unless set; fixed for the whole run",
        "n_gpu_layers": {hw: spec["n_gpu_layers"] for hw, spec in HARDWARE.items()},
        "peak_ram": "ru_maxrss of the llama-bench process from wait4 (= VmHWM at exit); max over a context's processes",
        "peak_gpu": ("nvidia-smi memory.used, sampled every second, peak minus the value before the "
                     "process started (device-wide); n/a without a GPU backend"),
        "contention": (f"wait until no other {'/'.join(CONTENDERS[:2])} process runs (poll {IDLE_POLL_S:.0f} s, "
                       f"up to {IDLE_MAX_S / 60:.0f} min) before every llama-bench process; record load average"),
        "extension_rule": ("--extend C --extend-if-under S: C is measured only if the previous context's "
                           "whole measurement (both processes, warm-ups included) took under S seconds"),
    }


@dataclass(frozen=True)
class Settings:
    """The llama-bench settings of one run (protocol defaults; threads and GPU layers per setup)."""

    threads: int
    n_gpu_layers: int
    batch: int = BATCH
    ubatch: int = UBATCH
    cache_type_k: str = KV_CACHE
    cache_type_v: str = KV_CACHE
    flash_attn: str = FLASH_ATTN
    load_mode: str = LOAD_MODE
    repetitions: int = REPETITIONS
    gen_tokens: int = GEN_TOKENS
    incremental_prompt: int = INCREMENTAL_PROMPT
    n_cpu_moe: int = 0
    override_tensor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def problems(self) -> list[str]:
        """Settings llama-bench would reject or that cannot give a median."""
        out = []
        if self.threads < 1:
            out.append(f"threads must be >= 1 (got {self.threads})")
        if self.repetitions < 1:
            out.append(f"repetitions must be >= 1 (got {self.repetitions})")
        if self.flash_attn not in FA_CODES:
            out.append(f"flash_attn must be one of {sorted(FA_CODES)} (got {self.flash_attn!r})")
        for name in ("cache_type_k", "cache_type_v"):
            if getattr(self, name) not in KV_TYPES:
                out.append(f"{name} must be one of {list(KV_TYPES)} (got {getattr(self, name)!r})")
        if self.cache_type_v not in ("f32", "f16", "bf16") and self.flash_attn == "off":
            out.append(f"a quantized V cache ({self.cache_type_v}) needs flash attention in llama.cpp")
        return out

    def deviations(self) -> list[str]:
        """Departures from the fixed protocol (recorded and shown with the results)."""
        fixed = {"batch": BATCH, "ubatch": UBATCH, "cache_type_k": KV_CACHE, "cache_type_v": KV_CACHE,
                 "flash_attn": FLASH_ATTN, "load_mode": LOAD_MODE, "repetitions": REPETITIONS,
                 "gen_tokens": GEN_TOKENS, "incremental_prompt": INCREMENTAL_PROMPT}
        return [f"{k} = {getattr(self, k)!r} (protocol: {v!r})" for k, v in fixed.items()
                if getattr(self, k) != v]


def llama_bench_argv(binary: Path | str, model: Path | str, s: Settings, *, n_prompt: int, n_gen: int,
                     n_depth: int, repetitions: int | None = None) -> list[str]:
    """The llama-bench command line for one test shape under the protocol settings."""
    argv = [str(binary), "-m", str(model),
            "-p", str(n_prompt), "-n", str(n_gen), "-d", str(n_depth),
            "-r", str(s.repetitions if repetitions is None else repetitions),
            "-t", str(s.threads), "-b", str(s.batch), "-ub", str(s.ubatch),
            "-ctk", s.cache_type_k, "-ctv", s.cache_type_v, "-fa", s.flash_attn,
            "-ngl", str(s.n_gpu_layers), "-lm", s.load_mode,
            "-o", "json", "--progress"]
    if s.n_cpu_moe:
        argv += ["-ncmoe", str(s.n_cpu_moe)]
    if s.override_tensor:
        argv += ["-ot", s.override_tensor]
    return argv


def context_tests(context: int, s: Settings) -> list[tuple[str, dict[str, int]]]:
    """The two llama-bench processes measured at one context (see the module docstring)."""
    return [("prompt", {"n_prompt": context, "n_gen": 0, "n_depth": 0}),
            ("depth", {"n_prompt": s.incremental_prompt, "n_gen": s.gen_tokens, "n_depth": context})]


def ctx_label(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else f"{n:,}"


def parse_contexts(text: str) -> list[int]:
    """'4096,16384' or '4K,16k' -> [4096, 16384] (order kept, duplicates dropped)."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*([kK]?)", part)
        if not m:
            raise ValueError(f"not a context length: {part!r}")
        n = int(m.group(1)) * (1024 if m.group(2) else 1)
        if n <= 0:
            raise ValueError(f"context length must be positive: {part!r}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("no context lengths given")
    return out


def quant_from_name(filename: str) -> str | None:
    """'Qwen3-4B-Q4_K_M.gguf' -> 'Q4_K_M'; BF16/F16/F32 and IQ types too; None if absent."""
    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    hits = re.findall(r"(?i)(?<![A-Za-z0-9])(BF16|F16|F32|I?Q\d(?:_[A-Za-z0-9]+)*)(?![A-Za-z0-9])", stem)
    return hits[-1].upper() if hits else None


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", text.lower()).strip("_.-") or "run"


def default_name(purpose: str, model: Path, hardware: str) -> str:
    return sanitize_name(f"{purpose}_{Path(model).stem}_{hardware}")


# --------------------------------------------------------------------------------------------
# llama-bench output: parsing, medians, per-context summary
# --------------------------------------------------------------------------------------------

def parse_llama_bench_output(text: str) -> list[dict[str, Any]]:
    """The test records of llama-bench `-o json` (a JSON array) or `-o jsonl` output.

    Anything before the first '[' or '{' (a stray log line) is ignored."""
    text = text.strip()
    if not text:
        return []
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON in llama-bench output")
    text = text[start:]
    if text.startswith("["):
        data, _ = json.JSONDecoder().raw_decode(text)
        if not isinstance(data, list):
            raise ValueError("llama-bench JSON output is not a list")
        return [dict(e) for e in data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def median(values: Iterable[float]) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("median of no values")
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def bench_kind(entry: dict[str, Any]) -> str:
    p, g = int(entry.get("n_prompt", 0)), int(entry.get("n_gen", 0))
    return "pp" if p > 0 and g == 0 else "tg" if g > 0 and p == 0 else "pg"


def bench_label(entry: dict[str, Any]) -> str:
    """llama-bench's own naming: 'pp4096 @ d0', 'tg128 @ d16384'."""
    kind = bench_kind(entry)
    p, g = int(entry.get("n_prompt", 0)), int(entry.get("n_gen", 0))
    core = f"pp{p}" if kind == "pp" else f"tg{g}" if kind == "tg" else f"pp{p}+tg{g}"
    return f"{core} @ d{int(entry.get('n_depth', 0))}"


def summarize_test(entry: dict[str, Any], expected_reps: int | None = REPETITIONS) -> dict[str, Any]:
    """Median and spread of one llama-bench test from its per-run samples.

    Tokens/s of a run = (n_prompt + n_gen) / run time, as llama-bench computes it; the median is
    taken over the runs (for an odd count it is the run with the median time)."""
    samples = [int(x) for x in entry.get("samples_ns") or []]
    if not samples:
        raise ValueError(f"{bench_label(entry)}: no samples_ns in llama-bench output")
    if expected_reps is not None and len(samples) != expected_reps:
        raise ValueError(f"{bench_label(entry)}: {len(samples)} samples, the protocol needs {expected_reps}")
    tokens = int(entry.get("n_prompt", 0)) + int(entry.get("n_gen", 0))
    ts = [tokens * 1e9 / ns for ns in samples]
    med_ns = median(samples)
    return {
        "test": bench_label(entry),
        "kind": bench_kind(entry),
        "n_prompt": int(entry.get("n_prompt", 0)),
        "n_gen": int(entry.get("n_gen", 0)),
        "n_depth": int(entry.get("n_depth", 0)),
        "repetitions": len(samples),
        "samples_ns": samples,
        "samples_ts": [round(t, 4) for t in ts],
        "median_ns": med_ns,
        "median_s": round(med_ns / 1e9, 4),
        "median_ts": round(median(ts), 3),
        "min_ts": round(min(ts), 3),
        "max_ts": round(max(ts), 3),
        "llama_bench_avg_ts": entry.get("avg_ts"),
        "llama_bench_stddev_ts": entry.get("stddev_ts"),
        "test_time": entry.get("test_time"),
    }


def find_test(tests: list[dict[str, Any]], kind: str, *, n_prompt: int | None = None,
              n_gen: int | None = None, n_depth: int) -> dict[str, Any] | None:
    for t in tests:
        if (t["kind"] == kind and t["n_depth"] == n_depth
                and (n_prompt is None or t["n_prompt"] == n_prompt)
                and (n_gen is None or t["n_gen"] == n_gen)):
            return t
    return None


def summarize_context(context: int, tests: list[dict[str, Any]], invocations: list[dict[str, Any]],
                      s: Settings, gpu_note: str) -> dict[str, Any]:
    """One context's row: prompt / generation / incremental tokens/s, TTFT, peak memory, wall time."""
    pp = find_test(tests, "pp", n_prompt=context, n_depth=0)
    tg = find_test(tests, "tg", n_gen=s.gen_tokens, n_depth=context)
    inc = find_test(tests, "pp", n_prompt=s.incremental_prompt, n_depth=context)
    ttft = None
    if pp and tg and tg["median_ts"] > 0:
        decode_s = 1.0 / tg["median_ts"]
        ttft = {"seconds": round(pp["median_ns"] / 1e9 + decode_s, 4),
                "prefill_s": round(pp["median_ns"] / 1e9, 4), "first_decode_s": round(decode_s, 5)}
    rss = [i["max_rss_bytes"] for i in invocations if i.get("max_rss_bytes")]
    gpu = [i["gpu_peak_delta_bytes"] for i in invocations if i.get("gpu_peak_delta_bytes") is not None]
    errors = [f"{i['kind']}: {i['error']}" for i in invocations if i.get("error")]
    ok = not errors and pp is not None and tg is not None and inc is not None
    return {
        "context": context,
        "label": ctx_label(context),
        "ok": ok,
        "errors": errors,
        "prompt": pp,
        "generation": tg,
        "incremental_prompt": inc,
        "ttft": ttft,
        "peak_ram_bytes": max(rss) if rss else None,
        "peak_gpu_bytes": max(gpu) if gpu else None,
        "gpu_note": gpu_note,
        "wall_s": round(sum(i.get("wall_s", 0.0) for i in invocations), 1),
        "invocations": invocations,
    }


def partial_context(partial: dict[str, Any], s: Settings, gpu_note: str, reason: str) -> dict[str, Any]:
    """The row of the context a run stopped in: what its finished processes measured, marked not ok.

    `partial` is the record's "in_progress" entry ({"context", "tests", "invocations"}), which is
    saved after every llama-bench process so that a run stopped mid-context keeps what it measured."""
    row = summarize_context(int(partial["context"]), list(partial.get("tests") or []),
                            list(partial.get("invocations") or []), s, gpu_note)
    done = [i["kind"] for i in partial.get("invocations") or []]
    planned = [kind for kind, _ in context_tests(int(partial["context"]), s)]
    missing = [k for k in planned if k not in done]
    row["ok"] = False
    row["incomplete"] = True
    row["errors"].append(f"stopped before the {' and '.join(missing) or 'last'} process "
                         f"of this context finished ({reason})")
    return row


def extension_decision(previous: dict[str, Any] | None, context: int,
                       budget_s: float | None) -> tuple[bool, str]:
    """Whether an --extend context is measured, given the previous (largest so far) context."""
    want = ctx_label(context)
    if previous is None:
        return False, f"{want} not measured: no smaller context was measured"
    prev = previous["label"]
    if not previous["ok"]:
        return False, f"{want} not measured: the {prev} context failed"
    if budget_s is None:
        return True, f"{want} measured (no time budget set)"
    wall = previous["wall_s"]
    one = previous["prompt"]["median_s"] if previous.get("prompt") else None
    detail = f"; one {prev}-token prefill took {one:.0f} s" if one is not None else ""
    if wall < budget_s:
        return True, (f"{want} measured: the {prev} context took {wall:.0f} s in all, "
                      f"under the {budget_s:.0f} s budget")
    return False, (f"{want} not measured: the {prev} context took {wall:.0f} s in all (both processes, "
                   f"warm-ups included{detail}), not under the {budget_s:.0f} s budget set for adding {want}")


def unmeasured(record: dict[str, Any], requested: list[int], reason: str) -> list[dict[str, Any]]:
    """Skip entries for the protocol and requested contexts a run neither measured nor already skipped.

    A requested context left over means the run stopped before reaching it (`reason`); a protocol
    context that was not requested is listed as such, so every one of 4K-64K is accounted for."""
    have = {r["context"] for r in record.get("contexts", [])} | {s["context"] for s in record.get("skipped", [])}
    out = []
    for ctx in list(requested) + [c for c in CONTEXTS if c not in requested]:
        if ctx in have:
            continue
        why = (f"{ctx_label(ctx)} not measured: the run stopped before it ({reason})" if ctx in requested
               else f"{ctx_label(ctx)} not requested in this run")
        out.append({"context": ctx, "label": ctx_label(ctx), "reason": why})
        have.add(ctx)
    return out


def applied_settings_problems(entry: dict[str, Any], s: Settings) -> list[str]:
    """Settings that llama-bench reports differently from what the protocol asked for."""
    want = {"n_threads": s.threads, "n_batch": s.batch, "n_ubatch": s.ubatch, "type_k": s.cache_type_k,
            "type_v": s.cache_type_v, "flash_attn": FA_CODES.get(s.flash_attn), "n_gpu_layers": s.n_gpu_layers,
            "n_cpu_moe": s.n_cpu_moe, "load_mode": s.load_mode}
    return [f"{k}: llama-bench reported {entry[k]!r}, the run asked for {v!r}"
            for k, v in want.items() if k in entry and entry[k] != v]


def build_info(entries: list[dict[str, Any]], stderr_text: str = "") -> dict[str, Any]:
    """What llama-bench says about itself (from its first test record and its log)."""
    if not entries:
        return {}
    e = entries[0]
    number = e.get("build_number")
    return {
        "build_number": number,
        "build_commit": e.get("build_commit"),
        "backends": e.get("backends"),
        "cpu_info": e.get("cpu_info"),
        "gpu_info": e.get("gpu_info"),
        "cpu_backend_variant": cpu_backend_variant(stderr_text),
        "matches_pin": number is not None and f"b{number}" == LLAMA_CPP_BUILD,
    }


def cpu_backend_variant(stderr_text: str) -> str | None:
    """'loaded CPU backend from .../libggml-cpu-alderlake.so' -> 'alderlake'."""
    m = re.search(r"loaded CPU backend from \S*libggml-cpu-([A-Za-z0-9_]+)\.so", stderr_text)
    return m.group(1) if m else None


def gpu_used(backends: str | None) -> bool:
    parts = {p.strip() for p in (backends or "").split(",")}
    return bool(parts - NON_GPU_BACKENDS)


def model_fields(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {}
    e = entries[0]
    return {"llama_bench_model_type": e.get("model_type"), "llama_bench_model_size": e.get("model_size"),
            "n_params": e.get("model_n_params")}


# --------------------------------------------------------------------------------------------
# Machine information, hardware definitions, contention
# --------------------------------------------------------------------------------------------

def parse_cpuinfo(text: str) -> dict[str, Any]:
    """CPU model, logical CPUs and physical cores (distinct (physical id, core id) pairs)."""
    model, logical, cores, phys = None, 0, set(), "0"
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
            phys = "0"
        elif key == "model name" and model is None:
            model = value
        elif key == "physical id":
            phys = value
        elif key == "core id":
            cores.add((phys, value))
    return {"model": model, "logical_cpus": logical, "physical_cores": len(cores) or logical}


def parse_meminfo(text: str) -> int | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.M)
    return int(m.group(1)) * 1024 if m else None


def parse_os_release(text: str) -> str | None:
    m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', text, re.M)
    return m.group(1) if m else None


def parse_proc_status(text: str) -> dict[str, int]:
    """VmHWM / VmRSS / VmPeak in bytes from /proc/<pid>/status."""
    out = {}
    for key in ("VmHWM", "VmRSS", "VmPeak"):
        m = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.M)
        if m:
            out[key] = int(m.group(1)) * 1024
    return out


def parse_nvidia_smi_csv(text: str, fields: list[str]) -> list[dict[str, str]]:
    """Rows of `nvidia-smi --query-...=a,b --format=csv,noheader,nounits`."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append(dict(zip(fields, parts)))
    return rows


def _read(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi") or next(
        (p for p in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi") if os.access(p, os.X_OK)), None)


def list_gpus() -> list[dict[str, Any]]:
    binary = nvidia_smi()
    if not binary:
        return []
    fields = ["name", "memory_total_mib", "driver"]
    try:
        out = subprocess.run([binary, "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = parse_nvidia_smi_csv(out.stdout, fields) if out.returncode == 0 else []
    return [{"name": r["name"], "memory_total_mib": int(float(r["memory_total_mib"])), "driver": r["driver"]}
            for r in rows if re.fullmatch(r"[\d.]+", r["memory_total_mib"])]


def backend_libraries(llama_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(llama_dir).glob("libggml-*.so"))


def is_gpu_backend_library(name: str) -> bool:
    """'libggml-cuda.so' yes; 'libggml-cpu-cannonlake.so' no (a CPU variant, not the CANN backend)."""
    m = re.fullmatch(r"libggml-([a-z0-9]+)\.so(?:\.[0-9.]+)?", name.lower())
    return bool(m) and m.group(1) in GPU_BACKEND_WORDS


def gpu_backend_libraries(llama_dir: Path) -> list[str]:
    return [n for n in backend_libraries(llama_dir) if is_gpu_backend_library(n)]


def machine_info(llama_dir: Path) -> dict[str, Any]:
    cpu = parse_cpuinfo(_read("/proc/cpuinfo"))
    kernel = platform.release()
    return {
        "cpu_model": cpu["model"],
        "physical_cores": cpu["physical_cores"],
        "logical_cpus": os.cpu_count(),
        "ram_total_bytes": parse_meminfo(_read("/proc/meminfo")),
        "os": parse_os_release(_read("/etc/os-release")) or platform.platform(),
        "kernel": kernel,
        "wsl": "microsoft" in kernel.lower(),
        "gpus": list_gpus(),
        "llama_cpp": {"dir": str(llama_dir), "pinned_build": LLAMA_CPP_BUILD,
                      "backend_libraries": backend_libraries(llama_dir),
                      "gpu_backend_libraries": gpu_backend_libraries(llama_dir)},
    }


def hardware_mismatches(label: str, machine: dict[str, Any], *, used_gpu: bool,
                        peak_gpu_bytes: int | None = None, s: Settings | None = None) -> list[str]:
    """Where this machine and run differ from the definition of the hardware label (§4.5)."""
    spec = HARDWARE[label]
    out = []
    if label == "H1":
        lo, hi = spec["cores"]
        cores = machine.get("physical_cores")
        if cores is not None and not lo <= cores <= hi:
            out.append(f"{cores} physical cores (H1: {lo}-{hi})")
        if s is not None and not lo <= s.threads <= hi:
            out.append(f"{s.threads} threads (H1 laptops have {lo}-{hi} cores)")
        ram = machine.get("ram_total_bytes")
        rlo, rhi = spec["ram_gib"]
        if ram is not None and not rlo <= ram / GIB <= rhi:
            out.append(f"{ram / GIB:.0f} GiB RAM visible (H1: 32 GB)")
        if used_gpu:
            out.append("a GPU backend was used (H1 is CPU only)")
        return out
    if not used_gpu:
        out.append(f"no GPU backend was used ({label} is a GPU setup)")
    gpus = machine.get("gpus") or []
    if label == "H2":
        if peak_gpu_bytes is not None and peak_gpu_bytes > spec["gpu_limit_gib"] * GIB:
            out.append(f"peak GPU memory {peak_gpu_bytes / GIB:.2f} GiB is over H2's 4 GB")
        if s is not None and not s.n_cpu_moe and not s.override_tensor:
            out.append("no expert placement given (--n-cpu-moe / --override-tensor); fine for a dense model")
    if "gpu_name" in spec and not any(spec["gpu_name"] in g.get("name", "") for g in gpus):
        out.append(f"no {spec['gpu_name']} GPU found ({label}: {spec['description']})")
    return out


def llama_processes(proc_root: Path = Path("/proc"), exclude: Iterable[int] = ()) -> list[dict[str, Any]]:
    """Running llama.cpp inference processes (by argv[0] basename), other than `exclude`."""
    skip = {os.getpid(), *exclude}
    found = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return []
    for d in entries:
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            argv0 = (d / "cmdline").read_bytes().split(b"\0", 1)[0].decode("utf-8", "replace")
        except OSError:
            continue
        name = os.path.basename(argv0)
        if name not in CONTENDERS and argv0:           # renamed argv[0]: fall back to the kernel's comm
            comm = _read(d / "comm").strip()           # (at most 15 characters)
            name = next((c for c in CONTENDERS if comm and comm == c[:15]), name)
        if name in CONTENDERS:
            found.append({"pid": int(d.name), "name": name})
    return sorted(found, key=lambda p: p["pid"])


def wait_until_idle(max_wait_s: float = IDLE_MAX_S, poll_s: float = IDLE_POLL_S, *,
                    probe: Callable[[], list[dict[str, Any]]] | None = None,
                    sleep: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.monotonic,
                    log: Callable[[str], None] = lambda msg: print(msg, flush=True)) -> dict[str, Any]:
    """Block until no other llama-server / llama-bench runs (or max_wait_s passes)."""
    probe = probe or llama_processes
    t0 = clock()
    seen: set[str] = set()
    while True:
        busy = probe()
        if not busy:
            return {"idle": True, "waited_s": round(clock() - t0, 1), "seen": sorted(seen)}
        seen.update(f"{p['name']} (pid {p['pid']})" for p in busy)
        waited = clock() - t0
        if waited >= max_wait_s:
            return {"idle": False, "waited_s": round(waited, 1), "seen": sorted(seen)}
        log(f"tier-e: waiting for {', '.join(p['name'] for p in busy)} to finish "
            f"({waited / 60:.1f} of {max_wait_s / 60:.0f} min)")
        sleep(poll_s)


# --------------------------------------------------------------------------------------------
# Running one llama-bench process
# --------------------------------------------------------------------------------------------

class GpuProbe:
    """Device memory in use, summed over GPUs, from nvidia-smi."""

    def __init__(self, binary: str):
        self.binary = binary

    def used_bytes(self) -> int | None:
        try:
            out = subprocess.run([self.binary, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        rows = parse_nvidia_smi_csv(out.stdout, ["used"]) if out.returncode == 0 else []
        vals = [float(r["used"]) for r in rows if re.fullmatch(r"[\d.]+", r["used"])]
        return int(sum(vals) * 1024 * 1024) if vals else None


def gpu_probe_for(llama_dir: Path) -> tuple[GpuProbe | None, str]:
    libs = gpu_backend_libraries(llama_dir)
    if not libs:
        return None, "n/a: the llama.cpp build has no GPU backend (CPU only)"
    binary = nvidia_smi()
    if not binary:
        return None, f"not measured: GPU backend {', '.join(libs)} but no nvidia-smi"
    return GpuProbe(binary), "nvidia-smi memory.used, device-wide, peak minus the value before the process started"


def _die_with_parent() -> None:          # runs in the child before exec
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:
        pass


def _load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def measure(argv: list[str], env: dict[str, str] | None, out_path: Path, err_path: Path, *,
            timeout_s: float = TIMEOUT_S, gpu: GpuProbe | None = None, poll_s: float = 1.0,
            contender_every_s: float = 10.0) -> dict[str, Any]:
    """Run one process to completion; stdout -> out_path, stderr -> err_path.

    Returns wall time, exit code, peak RSS (wait4 ru_maxrss; /proc VmHWM as sampled), the peak GPU
    memory delta (with `gpu`), the load average before and after, and any other llama process seen."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    started, load_before = utc_now(), _load1()
    gpu_base = gpu.used_bytes() if gpu else None
    gpu_peak: int | None = None
    vmhwm = 0
    others: dict[int, str] = {}
    timed_out = False
    t0 = time.monotonic()
    last_check = float("-inf")
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env,
                                preexec_fn=_die_with_parent if os.name == "posix" else None)
        try:
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid == proc.pid:
                    break
                now = time.monotonic()
                vmhwm = max(vmhwm, parse_proc_status(_read(f"/proc/{proc.pid}/status")).get("VmHWM", 0))
                if gpu is not None:
                    used = gpu.used_bytes()
                    if used is not None:
                        gpu_peak = used if gpu_peak is None else max(gpu_peak, used)
                if now - last_check >= contender_every_s:
                    for p in llama_processes(exclude=(proc.pid,)):
                        others[p["pid"]] = p["name"]
                    last_check = now
                if now - t0 > timeout_s:
                    timed_out = True
                    proc.kill()
                    _, status, usage = os.wait4(proc.pid, 0)
                    break
                time.sleep(poll_s)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    rc = proc.returncode
    error = (f"timed out after {timeout_s:.0f} s" if timed_out
             else f"exit code {rc}" if rc != 0 else None)
    return {
        "argv": list(argv),
        "started_utc": started,
        "wall_s": round(wall, 2),
        "returncode": rc,
        "error": error,
        "max_rss_bytes": int(usage.ru_maxrss) * 1024,
        "vmhwm_sampled_bytes": vmhwm or None,
        "gpu_peak_delta_bytes": (max(0, gpu_peak - gpu_base) if gpu_peak is not None and gpu_base is not None
                                 else None),
        "load1_before": load_before,
        "load1_after": _load1(),
        "other_llama_processes": sorted(f"{name} (pid {pid})" for pid, name in others.items()),
        "stdout": str(out_path),
        "stderr": str(err_path),
    }


def llama_env(llama_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(llama_dir) + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    return env


# --------------------------------------------------------------------------------------------
# One run of the protocol
# --------------------------------------------------------------------------------------------

def _json_path(name: str, out_dir: Path) -> Path:
    return out_dir / f"{name}.json"


def run_protocol(*, model: Path, hardware: str, settings: Settings, contexts: list[int], extend: list[int],
                 extend_budget_s: float | None, purpose: str, name: str, machine_label: str,
                 notes: list[str], llama_dir: Path = LLAMA_DIR, out_dir: Path = OUT_DIR,
                 raw_dir: Path = RAW_DIR, idle_max_s: float = IDLE_MAX_S, timeout_s: float = TIMEOUT_S,
                 cli_args: list[str] | None = None) -> dict[str, Any]:
    """Measure `model` at every requested context and write reports/tier_e/<name>.{json,md}."""
    binary = llama_dir / "llama-bench"
    env = llama_env(llama_dir)
    run_raw = raw_dir / name
    probe, gpu_note = gpu_probe_for(llama_dir)
    print(f"tier-e: hashing {model.name} ...", flush=True)
    record: dict[str, Any] = {
        "name": name,
        "purpose": purpose,
        "machine_label": machine_label,
        "label": (f"Plumbing test of the protocol on {machine_label}, not a Tier-E result for any "
                  "GRC-Bench model or hardware tier." if purpose == "plumbing" else
                  f"Tier-E result: {model.name} on {hardware} ({machine_label})."),
        "status": "running",
        "protocol": protocol(),
        "settings": settings.to_dict(),
        "deviations": settings.deviations(),
        "hardware": {"label": hardware, "definition": HARDWARE[hardware]["description"]},
        "model": {"path": str(model), "file": model.name, "size_bytes": model.stat().st_size,
                  "sha256": sha256_file(model), "quant": quant_from_name(model.name)},
        "machine": machine_info(llama_dir),
        "notes": list(notes),
        "cli_args": list(cli_args or []),
        "requested": {"contexts": contexts, "extend": extend, "extend_if_under_s": extend_budget_s},
        "raw_dir": str(run_raw),
        "started_utc": utc_now(),
        "contexts": [],
        "skipped": [],
        "applied_settings_problems": [],
    }
    from ..runlog.log import git_commit
    record["code"] = {"grc_bench_commit": git_commit(), "module": "grcbench.tier_e.bench",
                      "module_sha256": sha256_file(Path(__file__))}     # names the code when nothing is committed
    json_path = _json_path(name, out_dir)

    def save() -> None:
        write_json(json_path, record)

    def wait() -> dict[str, Any]:
        idle = wait_until_idle(idle_max_s)
        if not idle["idle"]:
            raise RuntimeError(f"other llama.cpp processes still running after {idle_max_s / 60:.0f} min: "
                               f"{', '.join(idle['seen'])}")
        return idle

    all_entries: list[dict[str, Any]] = []
    first_log = ""
    try:
        # warm-up invocation: pages the weights in; its numbers are discarded
        idle = wait()
        w = dict(WARMUP)
        argv = llama_bench_argv(binary, model, settings, n_prompt=w["n_prompt"], n_gen=w["n_gen"],
                                n_depth=w["n_depth"], repetitions=w["repetitions"])
        print("tier-e: warm-up invocation (discarded)", flush=True)
        m = measure(argv, env, run_raw / "warmup.json", run_raw / "warmup.log", timeout_s=timeout_s)
        m["idle_wait"] = idle
        record["warmup"] = {k: m[k] for k in ("argv", "wall_s", "returncode", "error", "max_rss_bytes",
                                              "idle_wait", "load1_before", "load1_after")}
        first_log = _read(run_raw / "warmup.log")
        if m["error"]:
            raise RuntimeError(f"warm-up invocation failed ({m['error']}); see {run_raw / 'warmup.log'}")
        save()

        def measure_context(ctx: int) -> dict[str, Any]:
            tests: list[dict[str, Any]] = []
            invocations: list[dict[str, Any]] = []
            record["in_progress"] = {"context": ctx, "tests": tests, "invocations": invocations}
            for kind, shape in context_tests(ctx, settings):
                idle = wait()
                argv = llama_bench_argv(binary, model, settings, **shape)
                print(f"tier-e: {ctx_label(ctx)} {kind}: {' '.join(argv[1:])}", flush=True)
                m = measure(argv, env, run_raw / f"ctx{ctx}_{kind}.json", run_raw / f"ctx{ctx}_{kind}.log",
                            timeout_s=timeout_s, gpu=probe)
                m["kind"], m["idle_wait"] = kind, idle
                if not m["error"]:
                    try:
                        entries = parse_llama_bench_output(_read(m["stdout"]))
                        all_entries.extend(entries)
                        tests.extend(summarize_test(e, settings.repetitions) for e in entries)
                        record["applied_settings_problems"] = sorted(set(record["applied_settings_problems"]) | {
                            p for e in entries for p in applied_settings_problems(e, settings)})
                    except ValueError as exc:
                        m["error"] = f"unparsable llama-bench output: {exc}"
                invocations.append(m)
                save()                  # each process's result survives a run stopped mid-context
                print(f"tier-e: {ctx_label(ctx)} {kind}: {m['wall_s']:.0f} s, peak RSS "
                      f"{m['max_rss_bytes'] / GIB:.2f} GiB{'; ' + m['error'] if m['error'] else ''}", flush=True)
            row = summarize_context(ctx, tests, invocations, settings, gpu_note)
            record.pop("in_progress", None)
            record["contexts"].append(row)
            save()
            return row

        failed: str | None = None
        for ctx in contexts:
            if failed:
                record["skipped"].append({"context": ctx, "label": ctx_label(ctx),
                                          "reason": f"{ctx_label(ctx)} not measured: the {failed} context failed"})
                continue
            row = measure_context(ctx)
            if not row["ok"]:
                failed = row["label"]
        for ctx in extend:
            previous = record["contexts"][-1] if record["contexts"] else None
            go, reason = extension_decision(previous, ctx, extend_budget_s)
            record.setdefault("extension_decisions", []).append({"context": ctx, "measured": go, "reason": reason})
            if go:
                measure_context(ctx)
            else:
                record["skipped"].append({"context": ctx, "label": ctx_label(ctx), "reason": reason})
        record["status"] = "complete" if all(r["ok"] for r in record["contexts"]) and record["contexts"] else "failed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_reason = record.get("error") or "stopped"
        partial = record.pop("in_progress", None)
        if partial is not None and partial.get("invocations"):    # keep what a stopped context measured
            record["contexts"].append(partial_context(partial, settings, gpu_note, stop_reason))
        record["skipped"] += unmeasured(record, list(contexts) + list(extend), stop_reason)
        log_text = first_log or next((_read(i["stderr"]) for r in record["contexts"] for i in r["invocations"]), "")
        record["llama_cpp"] = build_info(all_entries, log_text)
        record["model"].update(model_fields(all_entries))
        used = gpu_used(record["llama_cpp"].get("backends"))
        if not used:
            for r in record["contexts"]:
                r["gpu_note"] = "n/a: llama-bench used only the CPU backend"
                r["peak_gpu_bytes"] = None
        peaks = [r["peak_gpu_bytes"] for r in record["contexts"] if r.get("peak_gpu_bytes") is not None]
        record["hardware"]["gpu_used"] = used
        record["hardware"]["mismatches"] = hardware_mismatches(
            hardware, record["machine"], used_gpu=used, peak_gpu_bytes=max(peaks) if peaks else None, s=settings)
        record["finished_utc"] = utc_now()
        save()
        write_text(out_dir / f"{name}.md", render_run(record))
        write_text(out_dir / SUMMARY_NAME, render_summary(load_records(out_dir)))
    return record


# --------------------------------------------------------------------------------------------
# Markdown (rendered from the JSON records only)
# --------------------------------------------------------------------------------------------

def _f(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{x:,.{nd}f}"


def _ts(x: float | None) -> str:
    return "–" if x is None else f"{x:,.2f}" if x < 10 else f"{x:,.1f}"


def _gib(b: int | None) -> str:
    return "–" if b is None else f"{b / GIB:.2f}"


def _get(d: dict[str, Any] | None, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _rss_crosscheck(rec: dict[str, Any]) -> str:
    """' The /proc VmHWM sampled during each process agrees with ru_maxrss within x%.' (or '')."""
    diffs = [abs(i["vmhwm_sampled_bytes"] - i["max_rss_bytes"]) / i["max_rss_bytes"]
             for r in rec["contexts"] for i in r["invocations"]
             if i.get("vmhwm_sampled_bytes") and i.get("max_rss_bytes")]
    if not diffs:
        return ""
    return (f" Cross-check: the /proc VmHWM sampled while each process ran is within {max(diffs) * 100:.2f}% "
            "of its ru_maxrss.")


def _status_text(rec: dict[str, Any]) -> str:
    status = f"**{rec['status']}**" + (f" ({rec['error']})" if rec.get("error") else "")
    if rec["status"] == "running" and not rec.get("finished_utc"):
        status += (" (no finish time recorded: unless a `tier-e run` is still writing this record, its process "
                   "was killed, e.g. by a shutdown, and the record is partial)")
    return status


def _context_rows(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """The measured contexts, plus (for a record whose process was killed) the one it was in."""
    rows = list(rec["contexts"])
    partial = rec.get("in_progress")
    if partial and partial.get("invocations"):
        known = {f.name for f in fields(Settings)}
        s = Settings(**{k: v for k, v in rec["settings"].items() if k in known})
        rows.append(partial_context(partial, s, "–", "the process was killed"))
    return rows


def h4b_line(rec: dict[str, Any]) -> str:
    """The H4b verdict of a record (stats.functions.h4b_decision, pre-registration H4b-context): a
    context that did not run all REPETITIONS warm runs does not hold."""
    from grcbench.stats.functions import h4b_decision
    speeds = {r["context"]: _get(r, "generation", "samples_ts") for r in rec["contexts"]
              if r["context"] in CONTEXTS and r.get("ok")}
    d = h4b_decision({c: s for c, s in speeds.items() if s and len(s) == REPETITIONS})

    def k(c: int) -> str:
        return f"{c // 1024}K"
    verdict = f"holds up to {k(d['holds_up_to'])}" if d["holds_up_to"] else f"does not hold at {k(min(CONTEXTS))}"
    medians = "; ".join(f"{k(c)} {v['median']:.1f}" if v["median"] is not None else f"{k(c)} not run"
                        for c, v in d["contexts"].items())
    return (f"H4b (at least {H4_TARGET_TS:.0f} generation tokens/s at Q4_K_M on H2): {verdict} "
            f"(median generation tokens/s: {medians}).")


def render_run(rec: dict[str, Any]) -> str:
    s, m, mach, hw = rec["settings"], rec["model"], rec["machine"], rec["hardware"]
    build = rec.get("llama_cpp") or {}
    plumbing = rec["purpose"] == "plumbing"
    contexts = _context_rows(rec)
    title = (f"# Tier E plumbing test: {m['file']} on {rec['machine_label']}" if plumbing
             else f"# Tier E: {m['file']} on {hw['label']}")
    L = [title, ""]
    if plumbing:
        L += [f"> **{rec['label']}**", ">",
              "> The numbers below show that the protocol runs end to end and records every field. "
              "They are not a measurement of any GRC-Bench model, of any compression level of one, or "
              f"of hardware tier {hw['label']}.", ""]
    L += [f"Generated by `./gb tier-e report` from `reports/tier_e/{rec['name']}.json` "
          f"(eval plan §4.5; protocol `{rec['protocol']['version']}`). Status: {_status_text(rec)}.", ""]
    L += [f"Model `{m['file']}` ({m.get('llama_bench_model_type') or m.get('quant') or 'unknown type'}), "
          f"hardware label **{hw['label']}** ({hw['definition']}), {s['threads']} threads, "
          f"KV cache {s['cache_type_k']}/{s['cache_type_v']}, flash attention {s['flash_attn']}, "
          f"median of {s['repetitions']} runs.", ""]
    if hw.get("mismatches"):
        what = ("The label names the protocol setting exercised; this machine does not meet its definition"
                if plumbing else "This machine or run does not meet the hardware definition")
        L += [f"{what}: " + "; ".join(hw["mismatches"]) + ".", ""]

    L += [f"## Results (median of {s['repetitions']} runs)", "",
          "| Context | Prompt tokens/s (C-token prompt) | Time to first token (s) | "
          f"Generation tokens/s (tg{s['gen_tokens']} at C) | Incremental prompt tokens/s "
          f"(pp{s['incremental_prompt']} at C) | Peak RAM (GiB) | Peak GPU memory (GiB) | Wall time (s) |",
          "|---|--:|--:|--:|--:|--:|---|--:|"]
    for r in contexts:
        gpu = _gib(r["peak_gpu_bytes"]) if r.get("peak_gpu_bytes") is not None else (r.get("gpu_note") or "n/a")
        flag = "" if r["ok"] else " incomplete" if r.get("incomplete") else " FAILED"
        L.append(f"| {r['label']} ({r['context']:,}){flag} | "
                 f"{_ts(_get(r, 'prompt', 'median_ts'))} | {_f(_get(r, 'ttft', 'seconds'), 2)} | "
                 f"{_ts(_get(r, 'generation', 'median_ts'))} | {_ts(_get(r, 'incremental_prompt', 'median_ts'))} | "
                 f"{_gib(r.get('peak_ram_bytes'))} | {gpu} | {_f(r.get('wall_s'), 0)} |")
    if not contexts:
        L.append("| (none measured) | | | | | | | |")
    L.append("")
    problems = [f"{r['label']}: {e}" for r in contexts for e in r.get("errors") or []]
    if problems:
        L += ["Problems: " + "; ".join(problems) + ".", ""]
    if hw["label"] == "H2" and m.get("quant") == "Q4_K_M" and not plumbing and rec["contexts"]:
        L += [h4b_line(rec), ""]
    L += ["Time to first token = median prefill time of the C-token prompt + one decode step at depth C. "
          "Peak RAM is the larger of the context's two llama-bench processes (mapped weights included)."
          + _rss_crosscheck(rec), ""]

    L += ["### Run-to-run spread (min – max of the runs)", "",
          "| Context | Prompt tokens/s | Generation tokens/s | Incremental prompt tokens/s | "
          "Load average (1 min) before → after | Other llama processes seen |", "|---|--:|--:|--:|---|---|"]
    for r in contexts:
        def rng(key: str) -> str:
            t = r.get(key)
            return f"{_ts(t['min_ts'])} – {_ts(t['max_ts'])}" if t else "–"
        loads = "; ".join(f"{i['kind']} {_f(i.get('load1_before'), 2)} → {_f(i.get('load1_after'), 2)}"
                          for i in r["invocations"])
        others = sorted({o for i in r["invocations"] for o in i.get("other_llama_processes", [])})
        L.append(f"| {r['label']} | {rng('prompt')} | {rng('generation')} | {rng('incremental_prompt')} | "
                 f"{loads} | {', '.join(others) or 'none'} |")
    L += ["", f"The load average before a process shows what else was running; after it, it also counts "
          f"llama-bench's own {s['threads']} threads.", ""]

    L += ["## Contexts not measured", ""]
    L += [f"- {sk['reason']}." for sk in rec.get("skipped", [])] or ["- none"]
    L.append("")

    L += ["## Protocol and settings", "", "| Setting | Value |", "|---|---|"]
    proto = rec["protocol"]
    rows = [("Protocol", f"`{proto['version']}`: {proto['source']}"),
            ("Statistic", f"{proto['statistic']}; {s['repetitions']} runs per test"),
            ("Warm-up", proto["warmup"]),
            ("Per context: prompt", proto["tests_per_context"]["prompt"]),
            ("Per context: depth", proto["tests_per_context"]["depth"]),
            ("Time to first token", proto["ttft"]),
            ("Threads", f"{s['threads']} (fixed for the run; machine has {mach.get('physical_cores')} physical cores)"),
            ("Batch / micro-batch", f"{s['batch']} / {s['ubatch']}"),
            ("KV cache (K / V)", f"{s['cache_type_k']} / {s['cache_type_v']}"),
            ("Flash attention", s["flash_attn"]),
            ("Weights", f"load mode {s['load_mode']}"),
            ("GPU layers", f"{s['n_gpu_layers']}" + (f"; --n-cpu-moe {s['n_cpu_moe']}" if s.get("n_cpu_moe") else "")
             + (f"; --override-tensor {s['override_tensor']}" if s.get("override_tensor") else "")),
            ("Peak RAM", proto["peak_ram"]),
            ("Peak GPU memory", proto["peak_gpu"]),
            ("Contention", proto["contention"]),
            ("Extension rule", proto["extension_rule"]
             + (f"; this run: extend {', '.join(ctx_label(c) for c in rec['requested']['extend'])} "
                f"if under {rec['requested']['extend_if_under_s']:.0f} s"
                if rec["requested"].get("extend") and rec["requested"].get("extend_if_under_s") is not None else "")),
            ("Deviations from the protocol", "; ".join(rec.get("deviations") or []) or "none"),
            ("Settings llama-bench applied differently", "; ".join(rec.get("applied_settings_problems") or []) or "none")]
    L += [f"| {k} | {v} |" for k, v in rows]
    L.append("")

    gpus = mach.get("gpus") or []
    gpu_text = ("; ".join(f"{g['name']} ({g['memory_total_mib']:,} MiB, driver {g['driver']})" for g in gpus)
                if gpus else "none detected")
    if gpus and not hw.get("gpu_used"):
        gpu_text += " — not used: llama-bench ran on the CPU backend only"
    L += ["## Machine", "", "| | |", "|---|---|",
          f"| Machine | {rec['machine_label']} |",
          f"| CPU | {mach.get('cpu_model')} |",
          f"| Cores | {mach.get('physical_cores')} physical, {mach.get('logical_cpus')} logical |",
          f"| RAM | {_gib(mach.get('ram_total_bytes'))} GiB visible to the OS |",
          f"| OS | {mach.get('os')}, kernel {mach.get('kernel')}{' (WSL2)' if mach.get('wsl') else ''} |",
          f"| GPUs | {gpu_text} |",
          (f"| llama.cpp | build {build.get('build_number')} (commit {build.get('build_commit')}; pin "
           f"{mach['llama_cpp']['pinned_build']}: {'match' if build.get('matches_pin') else 'DIFFERENT'}), "
           f"backends {build.get('backends')}, CPU variant {build.get('cpu_backend_variant') or '–'}, "
           f"from `{mach['llama_cpp']['dir']}` |" if build else
           f"| llama.cpp | not recorded (no llama-bench output parsed), from `{mach['llama_cpp']['dir']}` |"),
          f"| GRC-Bench code | {rec['code']['grc_bench_commit']} |", ""]

    L += ["## Model", "", "| | |", "|---|---|",
          f"| File | `{m['file']}` ({m['size_bytes']:,} bytes) |",
          f"| SHA-256 | `{m['sha256']}` |",
          f"| Type (llama-bench) | {m.get('llama_bench_model_type') or '–'} |",
          f"| Parameters | {m['n_params']:,} |" if m.get("n_params") else "| Parameters | – |",
          f"| Quantization (file name) | {m.get('quant') or '–'} |", ""]

    L += ["## Timing and raw output", "",
          f"Started {rec['started_utc']}, finished {rec.get('finished_utc', '–')} (UTC). Warm-up invocation: "
          f"{_f(_get(rec, 'warmup', 'wall_s'), 0)} s (discarded). Raw llama-bench JSON and logs: "
          f"`{rec['raw_dir']}`.", ""]
    if rec.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in rec["notes"]] + [""]
    return "\n".join(L)


def load_records(out_dir: Path = OUT_DIR) -> list[dict[str, Any]]:
    recs = []
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            rec = read_json(p)
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and str(_get(rec, "protocol", "version") or "").startswith("tier-e/"):
            recs.append(rec)
    return recs


def _by_context(rec: dict[str, Any], key: str) -> str:
    parts = [f"{r['label']}: {_ts(_get(r, key, 'median_ts'))}" for r in _context_rows(rec) if r.get(key)]
    return "; ".join(parts) or "–"


def render_summary(recs: list[dict[str, Any]]) -> str:
    L = ["# Tier E: speed and memory runs", "",
         "Generated by `./gb tier-e report` from the JSON files in this folder (eval plan §4.5). Each run "
         "measures one GGUF file on one hardware setup with the fixed llama-bench protocol "
         f"(`./gb tier-e protocol`, `{PROTOCOL_VERSION}`): median of {REPETITIONS} runs after discarded "
         "warm-ups, at 4K, 16K, 32K and 64K tokens of context, KV cache q8_0, flash attention on.", ""]
    header = ["| Run | Model | Quant | Hardware | Threads | Status | Generation tokens/s | "
              "Prompt tokens/s | Not measured |", "|---|---|---|---|--:|---|---|---|---|"]

    def row(rec: dict[str, Any]) -> str:
        skipped = ", ".join(sk["label"] for sk in rec.get("skipped", [])) or "–"
        return (f"| [{rec['name']}]({rec['name']}.md) | {rec['model']['file']} | {rec['model'].get('quant') or '–'} | "
                f"{rec['hardware']['label']} | {rec['settings']['threads']} | {rec['status']} | "
                f"{_by_context(rec, 'generation')} | {_by_context(rec, 'prompt')} | {skipped} |")

    results = [r for r in recs if r["purpose"] == "result"]
    plumbing = [r for r in recs if r["purpose"] != "result"]
    L += ["## Tier-E results", ""]
    if results:
        L += header + [row(r) for r in results]
    else:
        L.append("None yet. Tier E needs the GRC-Bench models' GGUF files (W7) and the hardware: "
                 + "; ".join(f"{k} = {v['description']}" for k, v in HARDWARE.items()) + ".")
    L += ["", "## Plumbing tests (not Tier-E results)", ""]
    if plumbing:
        L += ["Each of these exercises the protocol on a machine that is not a hardware tier, or with a model "
              "that is not a GRC-Bench model; the hardware column names the protocol setting exercised.", ""]
        L += header + [row(r) for r in plumbing]
    else:
        L.append("None.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="./gb tier-e", description="Tier-E llama-bench speed/memory protocol (eval plan §4.5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure one GGUF file on this machine")
    r.add_argument("--model", required=True, help="GGUF file")
    r.add_argument("--hardware", required=True, choices=sorted(HARDWARE), help="hardware setup (eval plan §4.5)")
    r.add_argument("--contexts", default=",".join(str(c) for c in CONTEXTS),
                   help="context lengths, e.g. 4096,16384 or 4K,16K (default: all four)")
    r.add_argument("--extend", default="", help="contexts added only under --extend-if-under")
    r.add_argument("--extend-if-under", type=float, default=None, metavar="SECONDS",
                   help="add each --extend context only if the previous context took under SECONDS")
    r.add_argument("--purpose", choices=PURPOSES, default="plumbing",
                   help="'result' only for a GRC-Bench model on a declared hardware setup (default: plumbing)")
    r.add_argument("--name", default=None, help="output name (default: <purpose>_<model>_<hardware>)")
    r.add_argument("--machine-label", default="the build workstation", help="how the reports name this machine")
    r.add_argument("--note", action="append", default=[], help="free-text note recorded with the run")
    r.add_argument("--threads", type=int, default=None, help="threads (default: physical cores)")
    r.add_argument("--ngl", type=int, default=None, help="GPU layers (default: 0 on H1, all elsewhere)")
    r.add_argument("--n-cpu-moe", type=int, default=0, help="MoE expert layers kept on the CPU (H2)")
    r.add_argument("--override-tensor", default=None, help="llama-bench -ot pattern, e.g. 'exps=CPU' (H2)")
    r.add_argument("--cache-type", default=KV_CACHE, help=f"KV cache type for K and V (protocol: {KV_CACHE})")
    r.add_argument("--flash-attn", default=FLASH_ATTN, choices=sorted(FA_CODES), help="(protocol: on)")
    r.add_argument("--llama-dir", default=None, help=f"llama.cpp build directory (default: {LLAMA_DIR})")
    r.add_argument("--wait-idle-min", type=float, default=IDLE_MAX_S / 60,
                   help="max minutes to wait for other llama.cpp processes before each measurement")
    r.add_argument("--timeout-min", type=float, default=TIMEOUT_S / 60, help="per llama-bench process")
    r.add_argument("--dry-run", action="store_true", help="print the llama-bench commands and exit")
    sub.add_parser("report", help="rebuild reports/tier_e/*.md from the JSON files")
    sub.add_parser("protocol", help="print the protocol as JSON")
    return ap


def _settings_from_args(args: argparse.Namespace) -> Settings:
    cores = parse_cpuinfo(_read("/proc/cpuinfo"))["physical_cores"] or os.cpu_count() or 1
    ngl = args.ngl if args.ngl is not None else HARDWARE[args.hardware]["n_gpu_layers"]
    return Settings(threads=args.threads or cores, n_gpu_layers=ngl, cache_type_k=args.cache_type,
                    cache_type_v=args.cache_type, flash_attn=args.flash_attn, n_cpu_moe=args.n_cpu_moe,
                    override_tensor=args.override_tensor)


class Stopped(BaseException):
    """SIGTERM or SIGHUP during a run: raised so that the partial record is still written."""


def _stop_on_signals() -> dict[int, Any]:
    """Turn SIGTERM / SIGHUP into Stopped; returns the previous handlers (for restore_signals)."""
    def handler(signum: int, _frame: Any) -> None:
        raise Stopped(f"stopped by {signal.Signals(signum).name}")
    previous = {}
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            previous[sig] = signal.signal(sig, handler)
    return previous


def _restore_signals(previous: dict[int, Any]) -> None:
    for sig, old in previous.items():
        signal.signal(sig, old)


def cmd_run(args: argparse.Namespace, argv: list[str]) -> int:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        print(f"tier-e: no such model file: {model}")
        return 2
    llama_dir = Path(args.llama_dir).expanduser().resolve() if args.llama_dir else LLAMA_DIR
    settings = _settings_from_args(args)
    problems = settings.problems()
    try:
        contexts = parse_contexts(args.contexts)
        extend = [c for c in (parse_contexts(args.extend) if args.extend.strip() else []) if c not in contexts]
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        print("tier-e: " + "; ".join(problems))
        return 2
    name = sanitize_name(args.name) if args.name else default_name(args.purpose, model, args.hardware)
    if args.dry_run:
        binary = llama_dir / "llama-bench"
        print(f"# {name}: {args.purpose}, {args.hardware}, settings {json.dumps(settings.to_dict(), sort_keys=True)}")
        print(f"LD_LIBRARY_PATH={llama_dir} " + " ".join(llama_bench_argv(binary, model, settings, **WARMUP)))
        for ctx in contexts + extend:
            if ctx in extend:
                budget = (f"under {args.extend_if_under:.0f} s" if args.extend_if_under is not None
                          else "(no time budget set)")
                print(f"# {ctx_label(ctx)} only if the previous context's measurement took {budget}")
            for _, shape in context_tests(ctx, settings):
                print(f"LD_LIBRARY_PATH={llama_dir} " + " ".join(llama_bench_argv(binary, model, settings, **shape)))
        return 0
    if not (llama_dir / "llama-bench").is_file():
        print(f"tier-e: no llama-bench in {llama_dir} (run ./gb fetch)")
        return 2
    previous = _stop_on_signals()
    try:
        rec = run_protocol(model=model, hardware=args.hardware, settings=settings, contexts=contexts,
                           extend=extend, extend_budget_s=args.extend_if_under, purpose=args.purpose, name=name,
                           machine_label=args.machine_label, notes=args.note, llama_dir=llama_dir,
                           idle_max_s=args.wait_idle_min * 60, timeout_s=args.timeout_min * 60,
                           cli_args=["tier-e", *argv])
    except (RuntimeError, Stopped, KeyboardInterrupt) as exc:   # not idle in time, warm-up failed, or a signal
        print(f"tier-e: stopped: {type(exc).__name__}: {exc} (partial record in {OUT_DIR / (name + '.json')})")
        return 1
    finally:
        _restore_signals(previous)
    print(f"tier-e: {rec['status']}; wrote {OUT_DIR / (name + '.json')}, {OUT_DIR / (name + '.md')}, "
          f"{OUT_DIR / SUMMARY_NAME}")
    return 0 if rec["status"] == "complete" else 1


def cmd_report(out_dir: Path = OUT_DIR) -> int:
    recs = load_records(out_dir)
    for rec in recs:
        write_text(out_dir / f"{rec['name']}.md", render_run(rec))
    write_text(out_dir / SUMMARY_NAME, render_summary(recs))
    print(f"tier-e: rendered {len(recs)} run(s) and {out_dir / SUMMARY_NAME}")
    return 0


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "protocol":
        print(json.dumps(protocol(), indent=1, sort_keys=True, ensure_ascii=False))
        return 0
    if args.cmd == "report":
        return cmd_report()
    return cmd_run(args, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
