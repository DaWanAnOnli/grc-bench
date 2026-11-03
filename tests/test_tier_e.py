"""Tests for the Tier-E llama-bench protocol (grcbench.tier_e.bench; eval plan §4.5).

tests/fixtures/tier_e/ holds llama-bench output recorded from llama.cpp b10333 (Qwen3-4B Q4_K_M,
CPU, 16 threads, KV cache q8_0, flash attention on) during the first, interrupted attempt of the
plumbing run on 18 Oct 2026 (raw files kept in ~/grc-bench-work/tier_e/plumbing_qwen3_4b_cpu.attempt1_*):
ctx4096_prompt.json is `-p 4096 -n 0 -d 0 -r 5`, ctx4096_depth.json is `-p 512 -n 128 -d 4096 -r 5`.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from grcbench.common.paths import FIXTURES
from grcbench.tier_e import bench

FIX = FIXTURES / "tier_e"
PROMPT = FIX / "ctx4096_prompt.json"
DEPTH = FIX / "ctx4096_depth.json"
BANNER = ("Plumbing test of the protocol on the build workstation, not a Tier-E result for any GRC-Bench "
          "model or hardware tier.")


def _settings(**kw) -> bench.Settings:
    base = {"threads": 16, "n_gpu_layers": 0}
    base.update(kw)
    return bench.Settings(**base)


def _entries(path: Path) -> list[dict]:
    return bench.parse_llama_bench_output(path.read_text(encoding="utf-8"))


def _inv(kind: str, rss: int, wall: float, **kw) -> dict:
    inv = {"kind": kind, "max_rss_bytes": rss, "wall_s": wall, "error": None, "gpu_peak_delta_bytes": None,
           "load1_before": 0.5, "load1_after": 0.7, "other_llama_processes": [], "stderr": "", "argv": []}
    inv.update(kw)
    return inv


# --------------------------------------------------------------------------------------------
# Parsing recorded llama-bench output
# --------------------------------------------------------------------------------------------

def test_parse_recorded_prompt_output():
    (e,) = _entries(PROMPT)
    assert (e["n_prompt"], e["n_gen"], e["n_depth"]) == (4096, 0, 0)
    assert e["build_number"] == 10333 and e["backends"] == "CPU"
    assert len(e["samples_ns"]) == 5 and len(e["samples_ts"]) == 5
    assert bench.bench_kind(e) == "pp" and bench.bench_label(e) == "pp4096 @ d0"


def test_parse_recorded_depth_output_has_pp512_and_tg128_at_depth():
    entries = _entries(DEPTH)
    assert [bench.bench_label(e) for e in entries] == ["pp512 @ d4096", "tg128 @ d4096"]
    assert [bench.bench_kind(e) for e in entries] == ["pp", "tg"]


def test_parse_jsonl_noise_and_errors():
    rows = [{"n_prompt": 8, "n_gen": 0, "n_depth": 0, "samples_ns": [1]},
            {"n_prompt": 0, "n_gen": 4, "n_depth": 0, "samples_ns": [2]}]
    jsonl = "\n".join(json.dumps(r) for r in rows)
    assert bench.parse_llama_bench_output(jsonl) == rows
    assert bench.parse_llama_bench_output("load_backend: something\n" + json.dumps(rows)) == rows
    assert bench.parse_llama_bench_output("  \n") == []
    with pytest.raises(ValueError):
        bench.parse_llama_bench_output("no json here")
    assert bench.bench_label({"n_prompt": 16, "n_gen": 4, "n_depth": 0}) == "pp16+tg4 @ d0"


# --------------------------------------------------------------------------------------------
# Median logic
# --------------------------------------------------------------------------------------------

def test_median_odd_even_and_empty():
    assert bench.median([3, 1, 2]) == 2
    assert bench.median([5, 1, 4, 2, 3]) == 3
    assert bench.median([4, 1, 3, 2]) == 2.5
    with pytest.raises(ValueError):
        bench.median([])


def test_summarize_recorded_test_takes_the_median_run():
    (e,) = _entries(PROMPT)
    t = bench.summarize_test(e)
    ordered = sorted(e["samples_ns"])
    assert t["median_ns"] == ordered[2]
    assert t["median_ts"] == round(4096e9 / ordered[2], 3)
    assert t["min_ts"] == round(4096e9 / ordered[-1], 3) and t["max_ts"] == round(4096e9 / ordered[0], 3)
    # our per-run tokens/s agree with llama-bench's own (rounded) samples_ts
    for ours, theirs in zip(t["samples_ts"], e["samples_ts"]):
        assert ours == pytest.approx(theirs, rel=1e-3)
    # the median is robust to one slow run, unlike llama-bench's mean
    assert t["median_ts"] >= t["min_ts"]


def test_recorded_generation_outlier_moves_the_mean_not_the_median():
    tg = next(e for e in _entries(DEPTH) if e["n_gen"] == 128)
    t = bench.summarize_test(tg)
    # one of the five recorded tg128 runs was ~20% slower (the first, right after the depth fill)
    assert t["min_ts"] < 0.9 * t["median_ts"]
    assert t["median_ts"] > tg["avg_ts"]                       # llama-bench's mean is pulled down
    assert t["median_ts"] == round(128e9 / sorted(tg["samples_ns"])[2], 3)


def test_median_tokens_per_second_ignores_an_outlier():
    e = {"n_prompt": 0, "n_gen": 128, "n_depth": 4096,
         "samples_ns": [1_000_000_000, 1_100_000_000, 5_000_000_000, 900_000_000, 1_050_000_000]}
    t = bench.summarize_test(e)
    assert t["median_ns"] == 1_050_000_000
    assert t["median_ts"] == round(128 / 1.05, 3)          # odd count: tokens / median time
    assert t["min_ts"] == round(128 / 5.0, 3)


def test_summarize_rejects_wrong_repetition_count_and_missing_samples():
    (e,) = _entries(PROMPT)
    with pytest.raises(ValueError, match="protocol needs 3"):
        bench.summarize_test(e, expected_reps=3)
    assert bench.summarize_test(e, expected_reps=None)["repetitions"] == 5
    with pytest.raises(ValueError, match="no samples_ns"):
        bench.summarize_test({"n_prompt": 1, "n_gen": 0, "n_depth": 0})


def test_summarize_context_from_recorded_runs():
    s = _settings()
    tests = [bench.summarize_test(e) for e in _entries(PROMPT) + _entries(DEPTH)]
    invs = [_inv("prompt", 4_000_000_000, 60.0), _inv("depth", 4_500_000_000, 40.0)]
    row = bench.summarize_context(4096, tests, invs, s, "n/a: CPU")
    assert row["ok"] and row["label"] == "4K" and row["errors"] == []
    assert row["prompt"]["test"] == "pp4096 @ d0"
    assert row["generation"]["test"] == "tg128 @ d4096"
    assert row["incremental_prompt"]["test"] == "pp512 @ d4096"
    expect = row["prompt"]["median_ns"] / 1e9 + 1 / row["generation"]["median_ts"]
    assert row["ttft"]["seconds"] == pytest.approx(expect, abs=1e-3)
    assert row["ttft"]["prefill_s"] == pytest.approx(row["prompt"]["median_ns"] / 1e9, abs=1e-4)
    assert row["peak_ram_bytes"] == 4_500_000_000 and row["peak_gpu_bytes"] is None
    assert row["wall_s"] == 100.0


def test_summarize_context_without_generation_is_not_ok():
    tests = [bench.summarize_test(e) for e in _entries(PROMPT)]
    row = bench.summarize_context(4096, tests, [_inv("prompt", 1, 1.0), _inv("depth", 1, 1.0, error="exit code 1")],
                                  _settings(), "n/a")
    assert not row["ok"] and row["ttft"] is None and row["generation"] is None
    assert row["errors"] == ["depth: exit code 1"]


# --------------------------------------------------------------------------------------------
# Protocol fields and settings
# --------------------------------------------------------------------------------------------

def test_protocol_fixes_the_plan_settings():
    p = bench.protocol()
    assert p["contexts"] == [4096, 16384, 32768, 65536]
    assert p["repetitions"] == 5 and "median" in p["statistic"]
    assert p["cache_type_k"] == p["cache_type_v"] == "q8_0"
    assert p["flash_attn"] == "on" and p["batch"] == 2048 and p["ubatch"] == 512
    assert p["n_gpu_layers"]["H1"] == 0
    assert "discarded" in p["warmup"]
    assert set(bench.HARDWARE) == {"H1", "H2", "H3", "server"}


def test_llama_bench_argv_carries_every_protocol_setting():
    s = _settings()
    argv = bench.llama_bench_argv("/b/llama-bench", "/m/x.gguf", s, n_prompt=512, n_gen=128, n_depth=16384)
    joined = " ".join(argv)
    for flag in ("-m /m/x.gguf", "-p 512", "-n 128", "-d 16384", "-r 5", "-t 16", "-b 2048", "-ub 512",
                 "-ctk q8_0", "-ctv q8_0", "-fa on", "-ngl 0", "-lm mmap", "-o json"):
        assert flag in joined, flag
    assert "-ncmoe" not in argv and "-ot" not in argv
    moe = bench.llama_bench_argv("b", "m", _settings(n_gpu_layers=999, n_cpu_moe=30, override_tensor="exps=CPU"),
                                 n_prompt=1, n_gen=0, n_depth=0, repetitions=1)
    assert moe[moe.index("-ncmoe") + 1] == "30" and moe[moe.index("-ot") + 1] == "exps=CPU"
    assert moe[moe.index("-r") + 1] == "1"


def test_context_tests_are_prompt_from_empty_and_generation_at_depth():
    assert bench.context_tests(16384, _settings()) == [
        ("prompt", {"n_prompt": 16384, "n_gen": 0, "n_depth": 0}),
        ("depth", {"n_prompt": 512, "n_gen": 128, "n_depth": 16384})]


def test_settings_problems_and_deviations():
    assert _settings().problems() == [] and _settings().deviations() == []
    assert any("flash attention" in p for p in _settings(flash_attn="off").problems())
    assert _settings(flash_attn="off", cache_type_k="f16", cache_type_v="f16").problems() == []
    assert any("cache_type_k" in p for p in _settings(cache_type_k="q3_x").problems())
    assert any("threads" in p for p in _settings(threads=0).problems())
    dev = _settings(cache_type_k="f16", cache_type_v="f16", repetitions=3).deviations()
    assert len(dev) == 3 and any("protocol: 'q8_0'" in d for d in dev)


def test_applied_settings_checked_against_llama_bench_report():
    entries = _entries(PROMPT) + _entries(DEPTH)
    assert [p for e in entries for p in bench.applied_settings_problems(e, _settings())] == []
    probs = bench.applied_settings_problems(entries[0], _settings(threads=24))
    assert len(probs) == 1 and probs[0].startswith("n_threads")


def test_build_info_and_model_fields_from_recorded_output():
    entries = _entries(PROMPT)
    log = "load_backend: loaded CPU backend from /x/b10333/libggml-cpu-alderlake.so\n"
    info = bench.build_info(entries, log)
    assert info["build_number"] == 10333 and info["matches_pin"] is False       # recorded from b10333; pin b12020
    assert bench.build_info([{**entries[0], "build_number": 12020}], log)["matches_pin"] is True
    assert info["cpu_backend_variant"] == "alderlake" and info["backends"] == "CPU"
    assert bench.model_fields(entries)["n_params"] == entries[0]["model_n_params"]
    assert bench.build_info([], log) == {}
    assert not bench.gpu_used("CPU") and bench.gpu_used("CUDA") and not bench.gpu_used("CPU,RPC")


def test_parse_contexts_labels_and_quant():
    assert bench.parse_contexts("4096,16384") == [4096, 16384]
    assert bench.parse_contexts("4K, 16k,4096,32768") == [4096, 16384, 32768]
    for bad in ("", "abc", "0", "4M"):
        with pytest.raises(ValueError):
            bench.parse_contexts(bad)
    assert [bench.ctx_label(c) for c in (4096, 16384, 65536, 5000)] == ["4K", "16K", "64K", "5,000"]
    assert bench.quant_from_name("Qwen3-4B-Q4_K_M.gguf") == "Q4_K_M"
    assert bench.quant_from_name("north-mini-code-BF16.gguf") == "BF16"
    assert bench.quant_from_name("nanbeige4.2-3b-q8_0.gguf") == "Q8_0"
    assert bench.quant_from_name("model-IQ4_XS.gguf") == "IQ4_XS"
    assert bench.quant_from_name("model.gguf") is None


def test_extension_decision_follows_the_time_budget():
    prev = {"label": "16K", "ok": True, "wall_s": 812.0, "prompt": {"median_s": 98.0}}
    go, why = bench.extension_decision(prev, 32768, 180)
    assert not go and "not under the 180 s budget" in why and "812 s" in why and "98 s" in why
    go, why = bench.extension_decision(dict(prev, wall_s=150.0), 32768, 180)
    assert go and "under the 180 s budget" in why
    assert bench.extension_decision(dict(prev, ok=False), 32768, 180)[0] is False
    assert bench.extension_decision(None, 32768, 180)[0] is False
    assert bench.extension_decision(prev, 32768, None)[0] is True


# --------------------------------------------------------------------------------------------
# Machine information, hardware definitions, contention
# --------------------------------------------------------------------------------------------

def test_machine_parsers():
    cpuinfo = "".join(f"processor\t: {i}\nmodel name\t: Test CPU 9000\nphysical id\t: {i // 4}\n"
                      f"core id\t\t: {(i // 2) % 2}\n\n" for i in range(8))      # 2 sockets x 2 cores x 2 threads
    info = bench.parse_cpuinfo(cpuinfo)
    assert info == {"model": "Test CPU 9000", "logical_cpus": 8, "physical_cores": 4}
    assert bench.parse_cpuinfo("processor : 0\nprocessor : 1\n")["physical_cores"] == 2
    assert bench.parse_meminfo("MemTotal:       32768000 kB\nMemFree: 1 kB\n") == 32768000 * 1024
    assert bench.parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 26.04.1 LTS"\n') == "Ubuntu 26.04.1 LTS"
    assert bench.parse_proc_status("VmPeak:\t 10 kB\nVmHWM:\t 4323200 kB\nVmRSS:\t 100 kB\n") == {
        "VmHWM": 4323200 * 1024, "VmRSS": 100 * 1024, "VmPeak": 10 * 1024}
    rows = bench.parse_nvidia_smi_csv("NVIDIA GeForce RTX 3090, 24576, 550.1\n\nbad line\n",
                                      ["name", "memory_total_mib", "driver"])
    assert rows == [{"name": "NVIDIA GeForce RTX 3090", "memory_total_mib": "24576", "driver": "550.1"}]
    assert bench.cpu_backend_variant("nothing") is None


B10333_LIBS = ["libggml-base.so", "libggml-cpu-alderlake.so", "libggml-cpu-cannonlake.so",
               "libggml-cpu-cascadelake.so", "libggml-cpu-cooperlake.so", "libggml-cpu-haswell.so",
               "libggml-cpu-icelake.so", "libggml-cpu-ivybridge.so", "libggml-cpu-piledriver.so",
               "libggml-cpu-sandybridge.so", "libggml-cpu-sapphirerapids.so", "libggml-cpu-skylakex.so",
               "libggml-cpu-sse42.so", "libggml-cpu-x64.so", "libggml-cpu-zen4.so", "libggml-rpc.so"]


def test_cpu_only_build_has_no_gpu_backend(tmp_path):
    # regression: 'libggml-cpu-cannonlake.so' once matched the CANN (GPU/NPU) backend by substring
    assert not any(bench.is_gpu_backend_library(n) for n in B10333_LIBS)
    for name in ("libggml-cuda.so", "libggml-vulkan.so", "libggml-hip.so", "libggml-cann.so", "libggml-cuda.so.0"):
        assert bench.is_gpu_backend_library(name), name
    for name in B10333_LIBS:
        (tmp_path / name).write_bytes(b"")
    probe, note = bench.gpu_probe_for(tmp_path)
    assert probe is None and note.startswith("n/a")
    (tmp_path / "libggml-cuda.so").write_bytes(b"")
    assert bench.gpu_backend_libraries(tmp_path) == ["libggml-cuda.so"]


def test_hardware_mismatches_against_definitions():
    work = {"physical_cores": 24, "ram_total_bytes": 140 * bench.GIB, "gpus": []}
    out = bench.hardware_mismatches("H1", work, used_gpu=False, s=_settings())
    assert len(out) == 2 and "24 physical cores" in out[0] and "140 GiB" in out[1]
    laptop = {"physical_cores": 12, "ram_total_bytes": int(31.2 * bench.GIB), "gpus": []}
    assert bench.hardware_mismatches("H1", laptop, used_gpu=False, s=_settings(threads=12)) == []
    assert any("GPU backend" in m for m in bench.hardware_mismatches("H1", laptop, used_gpu=True))
    desk = {"gpus": [{"name": "NVIDIA GeForce RTX 3090"}]}
    assert bench.hardware_mismatches("H3", desk, used_gpu=True) == []
    assert any("3090" in m for m in bench.hardware_mismatches("H3", {"gpus": []}, used_gpu=True))
    h2 = bench.hardware_mismatches("H2", {"gpus": []}, used_gpu=True, peak_gpu_bytes=5 * bench.GIB,
                                   s=_settings(n_gpu_layers=999, n_cpu_moe=20))
    assert h2 == ["peak GPU memory 5.00 GiB is over H2's 4 GB"]


def test_llama_processes_scans_proc(tmp_path):
    for pid, argv in {"123": [b"/x/llama-server", b"-m", b"m.gguf"], "124": [b"python3", b"llama-bench"],
                      "125": [b"llama-bench", b"-p", b"1"], "126": []}.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(b"\0".join(argv) + (b"\0" if argv else b""))
    (tmp_path / "self").mkdir()
    found = bench.llama_processes(tmp_path)
    assert found == [{"pid": 123, "name": "llama-server"}, {"pid": 125, "name": "llama-bench"}]
    assert bench.llama_processes(tmp_path, exclude=(123, 125)) == []


def test_wait_until_idle_polls_until_free_or_gives_up():
    clock = {"t": 0.0}
    busy = [[{"pid": 7, "name": "llama-server"}]] * 2

    def probe():
        return busy.pop(0) if busy else []

    def sleep(s):
        clock["t"] += s

    res = bench.wait_until_idle(3600, 30, probe=probe, sleep=sleep, clock=lambda: clock["t"], log=lambda m: None)
    assert res == {"idle": True, "waited_s": 60.0, "seen": ["llama-server (pid 7)"]}
    clock["t"] = 0.0
    res = bench.wait_until_idle(90, 30, probe=lambda: [{"pid": 8, "name": "llama-bench"}], sleep=sleep,
                                clock=lambda: clock["t"], log=lambda m: None)
    assert res["idle"] is False and res["waited_s"] == 90.0


def test_measure_records_exit_code_and_peak_rss(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "llama_processes", lambda *a, **k: [])
    code = "b = bytearray(96 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); print('[]')"
    m = bench.measure([sys.executable, "-c", code], None, tmp_path / "o.json", tmp_path / "e.log", poll_s=0.02)
    assert m["returncode"] == 0 and m["error"] is None
    assert m["max_rss_bytes"] >= 96 * 1024 * 1024
    assert (tmp_path / "o.json").read_text().strip() == "[]"
    bad = bench.measure([sys.executable, "-c", "import sys; sys.exit(3)"], None, tmp_path / "b.json",
                        tmp_path / "b.log", poll_s=0.02)
    assert bad["returncode"] == 3 and bad["error"] == "exit code 3"
    slow = bench.measure([sys.executable, "-c", "import time; time.sleep(30)"], None, tmp_path / "s.json",
                         tmp_path / "s.log", timeout_s=0.3, poll_s=0.05)
    assert slow["error"].startswith("timed out") and slow["wall_s"] < 10


# --------------------------------------------------------------------------------------------
# A whole run against a fake llama-bench (orchestration, extension rule, JSON and Markdown)
# --------------------------------------------------------------------------------------------

FAKE_LLAMA_BENCH = r'''#!{python}
"""Prints llama-bench-shaped JSON for the -p/-n/-d it is given (test double)."""
import json, sys
a = sys.argv[1:]
def opt(flag, default):
    return a[a.index(flag) + 1] if flag in a else default
p, n, d, r = int(opt("-p", 512)), int(opt("-n", 128)), int(opt("-d", 0)), int(opt("-r", 5))
sys.stderr.write("load_backend: loaded CPU backend from /fake/libggml-cpu-haswell.so\n")
base = {{"build_commit": "0ec4027a2", "build_number": 12020, "cpu_info": "Fake CPU", "gpu_info": "",
        "backends": "CPU", "model_filename": opt("-m", ""), "model_type": "fake 1B Q4_K - Medium",
        "model_size": 1000, "model_n_params": 1000000, "n_batch": int(opt("-b", 2048)),
        "n_ubatch": int(opt("-ub", 512)), "n_threads": int(opt("-t", 1)), "type_k": opt("-ctk", "f16"),
        "type_v": opt("-ctv", "f16"), "n_gpu_layers": int(opt("-ngl", -1)), "n_cpu_moe": 0,
        "flash_attn": {{"on": 1, "off": 0, "auto": -1}}[opt("-fa", "auto")], "load_mode": opt("-lm", "mmap")}}
out = []
for kind, (np_, ng) in (("pp", (p, 0)), ("tg", (0, n))):
    if np_ + ng == 0:
        continue
    per_tok = 1_000_000 if kind == "pp" else 50_000_000
    samples = [(np_ + ng) * per_tok * (1 + d / 65536) * (1 + 0.01 * i) for i in range(r)]
    out.append(dict(base, n_prompt=np_, n_gen=ng, n_depth=d, test_time="2026-10-18T00:00:00Z",
                    samples_ns=[int(x) for x in samples], samples_ts=[(np_ + ng) * 1e9 / x for x in samples],
                    avg_ns=int(sum(samples) / r), stddev_ns=0, avg_ts=0.0, stddev_ts=0.0))
print(json.dumps(out, indent=2))
'''


def _fake_llama_dir(tmp_path: Path) -> Path:
    d = tmp_path / "llama"
    d.mkdir()
    exe = d / "llama-bench"
    exe.write_text(FAKE_LLAMA_BENCH.format(python=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (d / "libggml-cpu-haswell.so").write_bytes(b"")
    return d


def _run(tmp_path: Path, monkeypatch, *, budget: float, purpose: str = "plumbing", name: str = "fake_run") -> dict:
    monkeypatch.setattr(bench, "llama_processes", lambda *a, **k: [])
    model = tmp_path / "Fake-1B-Q4_K_M.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    return bench.run_protocol(model=model, hardware="H1", settings=_settings(threads=2), contexts=[4096, 16384],
                              extend=[32768], extend_budget_s=budget, purpose=purpose, name=name,
                              machine_label="the build workstation", notes=["a note"],
                              llama_dir=_fake_llama_dir(tmp_path), out_dir=tmp_path / "out",
                              raw_dir=tmp_path / "raw", idle_max_s=1, cli_args=["tier-e", "run"])


def test_run_protocol_end_to_end_with_fake_llama_bench(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, budget=0.0)
    assert rec["status"] == "complete"
    assert [r["context"] for r in rec["contexts"]] == [4096, 16384]
    assert all(r["ok"] for r in rec["contexts"])
    skipped = {s["context"]: s["reason"] for s in rec["skipped"]}
    assert "not under the 0 s budget" in skipped[32768] and "not requested" in skipped[65536]
    row = rec["contexts"][0]
    assert row["prompt"]["test"] == "pp4096 @ d0" and row["generation"]["test"] == "tg128 @ d4096"
    assert row["prompt"]["repetitions"] == 5
    assert row["gpu_note"].startswith("n/a") and row["peak_gpu_bytes"] is None
    assert row["peak_ram_bytes"] > 0 and len(row["invocations"]) == 2
    assert rec["llama_cpp"]["matches_pin"] and rec["llama_cpp"]["cpu_backend_variant"] == "haswell"
    assert rec["applied_settings_problems"] == [] and rec["deviations"] == []
    assert rec["model"]["quant"] == "Q4_K_M" and len(rec["model"]["sha256"]) == 64
    assert rec["warmup"]["argv"][rec["warmup"]["argv"].index("-r") + 1] == "1"
    out = tmp_path / "out"
    assert bench.read_json(out / "fake_run.json")["status"] == "complete"
    md = (out / "fake_run.md").read_text(encoding="utf-8")
    assert BANNER in md and "| 4K (4,096) |" in md and "| 16K (16,384) |" in md
    assert "32K not measured" in md
    summary = (out / bench.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "## Plumbing tests (not Tier-E results)" in summary and "fake_run" in summary
    assert "None yet." in summary


def test_run_protocol_extends_when_under_budget(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, budget=1e6)
    assert [r["context"] for r in rec["contexts"]] == [4096, 16384, 32768]
    assert rec["extension_decisions"][0]["measured"] is True
    assert [s["context"] for s in rec["skipped"]] == [65536]


def test_report_is_deterministic_and_separates_results(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, budget=0.0, purpose="result", name="fake_result")
    out = tmp_path / "out"
    md1 = (out / "fake_result.md").read_bytes()
    assert BANNER.encode() not in md1 and b"# Tier E: Fake-1B-Q4_K_M.gguf on H1" in md1
    assert bench.cmd_report(out) == 0
    assert (out / "fake_result.md").read_bytes() == md1
    summary = (out / bench.SUMMARY_NAME).read_text(encoding="utf-8")
    results = summary.split("## Plumbing tests")[0]
    assert "fake_result" in results and "None yet." not in results
    assert bench.render_run(rec) == bench.render_run(json.loads(json.dumps(rec)))


def test_the_h4_line_states_the_h4b_verdict(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, budget=0.0, purpose="result", name="fake_h2")
    rec["hardware"]["label"] = "H2"
    base = rec["contexts"][0]
    # holds at 4K and 32K but not at 16K; 64K not run: the verdict is the unbroken range from 4K
    speeds = {4096: [90.0] * 5, 16384: [70.0] * 5, 32768: [85.0] * 5}
    rec["contexts"] = [json.loads(json.dumps({**base, "context": c, "label": f"{c // 1024}K",
                                              "generation": {**base["generation"], "samples_ts": s,
                                                             "median_ts": s[0]}}))
                       for c, s in speeds.items()]
    md = bench.render_run(rec)
    assert "H4b (at least 80 generation tokens/s at Q4_K_M on H2): holds up to 4K" in md
    assert "64K not run" in md and "met at" not in md
    rec["contexts"][0]["generation"]["samples_ts"] = [70.0] * 5
    assert "H4b (at least 80 generation tokens/s at Q4_K_M on H2): does not hold at 4K" in bench.render_run(rec)


def test_rss_crosscheck_sentence():
    rec = {"contexts": [{"invocations": [_inv("prompt", 1000, 1.0, vmhwm_sampled_bytes=1002),
                                         _inv("depth", 2000, 1.0, vmhwm_sampled_bytes=None)]}]}
    assert "within 0.20% of its ru_maxrss" in bench._rss_crosscheck(rec)
    assert bench._rss_crosscheck({"contexts": []}) == ""


def test_cli_protocol_and_dry_run(tmp_path, capsys):
    assert bench.main(["protocol"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == bench.PROTOCOL_VERSION
    model = tmp_path / "m-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    assert bench.main(["run", "--model", str(model), "--hardware", "H1", "--contexts", "4K",
                       "--threads", "4", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "-p 4096 -n 0 -d 0 -r 5" in out and "-p 512 -n 128 -d 4096 -r 5" in out and "-r 1" in out
    assert bench.main(["run", "--model", str(tmp_path / "missing.gguf"), "--hardware", "H1", "--dry-run"]) == 2
    assert bench.main(["run", "--model", str(model), "--hardware", "H1", "--flash-attn", "off", "--dry-run"]) == 2


# --------------------------------------------------------------------------------------------
# A run that is stopped keeps what it measured
# --------------------------------------------------------------------------------------------

def test_run_stopped_mid_context_keeps_the_finished_process(tmp_path, monkeypatch):
    real = bench.measure

    def measure(argv, *a, **k):                    # SIGTERM arrives during the 16K depth process
        if argv[argv.index("-d") + 1] == "16384":
            raise bench.Stopped("stopped by SIGTERM")
        return real(argv, *a, **k)

    monkeypatch.setattr(bench, "measure", measure)
    with pytest.raises(bench.Stopped):
        _run(tmp_path, monkeypatch, budget=1e6)
    rec = bench.read_json(tmp_path / "out" / "fake_run.json")
    assert rec["status"] == "failed" and rec["error"] == "Stopped: stopped by SIGTERM"
    assert "in_progress" not in rec and rec["finished_utc"]
    first, stopped = rec["contexts"]
    assert first["ok"] and first["context"] == 4096
    assert stopped["context"] == 16384 and stopped["incomplete"] and not stopped["ok"]
    assert stopped["prompt"]["test"] == "pp16384 @ d0" and stopped["generation"] is None
    assert "before the depth process" in stopped["errors"][-1]
    skipped = {s["context"]: s["reason"] for s in rec["skipped"]}
    assert "run stopped before it (Stopped" in skipped[32768] and "not requested" in skipped[65536]
    md = (tmp_path / "out" / "fake_run.md").read_text(encoding="utf-8")
    assert "| 16K (16,384) incomplete |" in md and "Status: **failed** (Stopped: stopped by SIGTERM)" in md


def test_record_of_a_killed_process_renders_as_partial(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, budget=0.0)
    killed = json.loads(json.dumps(rec))
    last = killed["contexts"].pop()                # as saved after the 16K prompt process, then SIGKILL
    killed["in_progress"] = {"context": last["context"], "tests": [last["prompt"]],
                             "invocations": last["invocations"][:1]}
    killed["status"] = "running"
    for key in ("finished_utc", "skipped"):
        killed.pop(key)
    killed["skipped"] = []
    md = bench.render_run(killed)
    assert "Status: **running** (no finish time recorded" in md
    assert "| 16K (16,384) incomplete |" in md and "the process was killed" in md
    assert "16K: " in bench.render_summary([killed])


def test_unmeasured_accounts_for_every_protocol_context():
    rec = {"contexts": [{"context": 4096}], "skipped": [{"context": 32768, "label": "32K", "reason": "x"}]}
    out = bench.unmeasured(rec, [4096, 16384, 32768], "KeyboardInterrupt: ")
    assert [(s["context"], s["reason"]) for s in out] == [
        (16384, "16K not measured: the run stopped before it (KeyboardInterrupt: )"),
        (65536, "64K not requested in this run")]
    full = {"contexts": [{"context": c} for c in bench.CONTEXTS], "skipped": []}
    assert bench.unmeasured(full, list(bench.CONTEXTS), "x") == []


def test_llama_processes_falls_back_to_comm(tmp_path):
    (tmp_path / "200").mkdir()
    (tmp_path / "200" / "cmdline").write_bytes(b"renamed-by-wrapper\0-m\0x.gguf\0")
    (tmp_path / "200" / "comm").write_text("llama-server\n")
    (tmp_path / "201").mkdir()
    (tmp_path / "201" / "cmdline").write_bytes(b"/opt/x/run\0")
    (tmp_path / "201" / "comm").write_text("llama-completio\n")     # 15-character comm of llama-completion
    (tmp_path / "202").mkdir()
    (tmp_path / "202" / "cmdline").write_bytes(b"python3\0")
    (tmp_path / "202" / "comm").write_text("python3\n")
    assert bench.llama_processes(tmp_path) == [{"pid": 200, "name": "llama-server"},
                                               {"pid": 201, "name": "llama-completion"}]


def test_sigterm_becomes_stopped_and_handlers_are_restored():
    import signal
    before = signal.getsignal(signal.SIGTERM)
    previous = bench._stop_on_signals()
    try:
        with pytest.raises(bench.Stopped, match="SIGTERM"):
            os.kill(os.getpid(), signal.SIGTERM)
            import time
            time.sleep(5)                          # the handler raises before this returns
    finally:
        bench._restore_signals(previous)
    assert signal.getsignal(signal.SIGTERM) == before


def test_cli_drops_extend_contexts_already_requested(tmp_path, capsys):
    model = tmp_path / "m-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    assert bench.main(["run", "--model", str(model), "--hardware", "H1", "--contexts", "4K,16K",
                       "--extend", "16K,32K", "--threads", "4", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("-d 16384 ") == 1 and out.count("-d 32768 ") == 1
    assert "# 32K only if the previous context's measurement took (no time budget set)" in out
    assert "# 16K only if" not in out
