"""`./gb bfcl`: the BFCL runs of the run list (eval plan Table 6; pre-registration §6).

    ./gb bfcl setup
    ./gb bfcl generate --config-id ID --base-url URL --model NAME --out DIR [--threads N] [--run-ids FILE]
    ./gb bfcl evaluate --config-id ID --out DIR

The Berkeley Function-Calling Leaderboard runs through its own package, bfcl-eval BFCL_VERSION (BFCL v4), in
its own environment (BFCL_ENV, made by `setup` with uv), never in GRC-Agent's. `./gb` starts this module
with GRC-Agent's Python; `main` then runs it again with BFCL_ENV's Python.

generate  the configuration (`--config-id`) must be on docs/preregistration/run_list.yaml with the BFCL mark.
          The model is the llama.cpp server that serves the configuration (`--base-url`, `--model` its alias),
          called through BFCL's OpenAI-compatible chat handler in function-calling mode: the tools go to the
          server as native tools, which the server's default tool grammar holds to their schemas, as in B0.
          The configuration is registered for this run only, as REGISTRY_PREFIX + config id, with that
          handler (`model_config_fields`); bfcl-eval itself is not edited. Categories: CATEGORIES (BFCL v4's
          single-turn sets, which need no web service), at BFCL's own temperature (TEMPERATURE). Every
          request asks for at most REPLY_TOKENS tokens (max_tokens, reasoning included), as Tier K limits an
          answer; the handler is BFCL's own with only that added (`handler_class`).
          DIR/manifest.json records the configuration, the server's /props (build, model file, context,
          slots), the BFCL version, the categories, the temperature and the GRC-Bench commit.
          --run-ids FILE: only the BFCL test ids FILE lists (BFCL's test_case_ids_to_generate.json): a
          plumbing check, never a result.
evaluate  BFCL's own checker scores DIR's responses; DIR/summary.json holds its non-live and live overall
          accuracies (BFCL's "Overall Acc" counts the categories that were not run as 0 and is not used).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .common.paths import CACHE, REPO

BFCL_VERSION = "2026.3.23"
BFCL_ENV = CACHE / "bfcl" / ".venv"
BFCL_PY = BFCL_ENV / "bin" / "python"
CATEGORIES = ("non_live", "live")
TEMPERATURE = 0.001
REPLY_TOKENS = 16384
REGISTRY_PREFIX = "grcbench-"
RUN_LIST = REPO / "docs" / "preregistration" / "run_list.yaml"


class BfclError(RuntimeError):
    """A BFCL run that may not or cannot run."""


def bfcl_configurations(run_list: Path = RUN_LIST) -> dict[str, dict[str, Any]]:
    """The run list's configurations that carry the BFCL mark, by id."""
    import yaml
    data = yaml.safe_load(Path(run_list).read_text(encoding="utf-8"))
    return {str(c["id"]): c for c in data.get("configurations", [])
            if isinstance(c, dict) and (c.get("tiers") or {}).get("BFCL")}


def check_configuration(config_id: str, run_list: Path = RUN_LIST) -> dict[str, Any]:
    marked = bfcl_configurations(run_list)
    if config_id not in marked:
        raise BfclError(f"{config_id!r} is not a configuration with the BFCL mark on {run_list.name}")
    return marked[config_id]


def registry_name(config_id: str) -> str:
    return f"{REGISTRY_PREFIX}{config_id}-FC"


def model_config_fields(config_id: str, model: str) -> dict[str, Any]:
    """The bfcl-eval ModelConfig of a configuration, without its handler class (added by `register`)."""
    return {"model_name": model, "display_name": f"GRC-Bench {config_id} (FC)", "url": "",
            "org": "GRC-Bench run list", "license": "", "input_price": None, "output_price": None,
            "is_fc_model": True, "underscore_to_dot": True}


def handler_class() -> type:
    """BFCL's OpenAI-compatible chat handler, with every request limited to REPLY_TOKENS tokens."""
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    class GrcBenchFCHandler(OpenAICompletionsHandler):
        def generate_with_backoff(self, **kwargs: Any) -> Any:
            kwargs.setdefault("max_tokens", REPLY_TOKENS)
            return super().generate_with_backoff(**kwargs)
    return GrcBenchFCHandler


def register(config_id: str, model: str) -> str:
    """Add the configuration to bfcl-eval's model table for this process; returns its registry name."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    name = registry_name(config_id)
    MODEL_CONFIG_MAPPING[name] = ModelConfig(model_handler=handler_class(), **model_config_fields(config_id, model))
    return name


def server_props(base_url: str) -> dict[str, Any]:
    root = base_url.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(f"{root}/props", timeout=10) as r:
        props = json.loads(r.read())
    gen = props.get("default_generation_settings") or {}
    return {"build": props.get("build_info"), "model_path": props.get("model_path"), "n_ctx": gen.get("n_ctx"),
            "total_slots": props.get("total_slots"), "n_predict": (gen.get("params") or {}).get("n_predict")}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
        return out.stdout.strip() + ("+uncommitted" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _project_root(out: Path) -> None:
    """BFCL reads its project root (results, scores, the id file) when it is first imported: set it before."""
    if "bfcl_eval.constants.eval_config" in sys.modules and os.environ.get("BFCL_PROJECT_ROOT") != str(out):
        raise BfclError("bfcl_eval was imported before its project root was set")
    os.environ["BFCL_PROJECT_ROOT"] = str(out)


def _bfcl(args: list[str], out: Path, base_url: str | None = None) -> None:
    # BFCL builds the handler's OpenAI client for scoring too; a local llama-server needs no real key
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "local"
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    from bfcl_eval.__main__ import cli
    cli(args, standalone_mode=False)


def check_version() -> None:
    from importlib.metadata import version
    have = version("bfcl-eval")
    if have != BFCL_VERSION:
        raise BfclError(f"bfcl-eval {have} is installed, the pre-registration pins {BFCL_VERSION}: ./gb bfcl setup")


def generate(config_id: str, base_url: str, model: str, out: Path, threads: int = 1,
             run_ids: Path | None = None) -> dict[str, Any]:
    check_configuration(config_id)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _project_root(out)
    check_version()
    props = server_props(base_url)
    manifest = {"config_id": config_id, "registry_name": registry_name(config_id), "base_url": base_url,
                "model": model, "server": props, "bfcl_eval": BFCL_VERSION, "categories": list(CATEGORIES),
                "temperature": TEMPERATURE, "reply_tokens": REPLY_TOKENS, "threads": threads,
                "plumbing_check": run_ids is not None, "grc_bench_commit": git_commit(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    name = register(config_id, model)
    args = ["generate", "--model", name, "--test-category", ",".join(CATEGORIES), "--temperature", str(TEMPERATURE),
            "--num-threads", str(threads)]
    if run_ids is not None:
        shutil.copyfile(run_ids, out / "test_case_ids_to_generate.json")
        args.append("--run-ids")
    t0 = time.monotonic()
    _bfcl(args, out, base_url)
    manifest["generate_seconds"] = round(time.monotonic() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _accuracies(csv_path: Path, name: str) -> dict[str, str]:
    if not csv_path.is_file():
        return {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Model", "").startswith("GRC-Bench") or name in json.dumps(row):
                return row
    return {}


def evaluate(config_id: str, out: Path) -> dict[str, Any]:
    check_configuration(config_id)
    out = out.resolve()
    _project_root(out)
    check_version()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    name = register(config_id, manifest["model"])
    args = ["evaluate", "--model", name, "--test-category", ",".join(CATEGORIES)]
    if manifest.get("plumbing_check"):
        args.append("--partial-eval")
    _bfcl(args, out)
    score = out / "score"
    non_live, live = _accuracies(score / "data_non_live.csv", name), _accuracies(score / "data_live.csv", name)
    summary = {"config_id": config_id, "bfcl_eval": BFCL_VERSION, "plumbing_check": manifest.get("plumbing_check"),
               "non_live_overall_acc": non_live.get("Non-Live Overall Acc") or non_live.get("Non_Live Overall Acc"),
               "live_overall_acc": live.get("Live Overall Acc"), "non_live": non_live, "live": live}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def setup() -> int:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not BFCL_PY.is_file():
        subprocess.run([uv, "venv", "--quiet", "--python", "3.12", str(BFCL_ENV)], check=True)
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(BFCL_PY), f"bfcl-eval=={BFCL_VERSION}", "pyyaml", "soundfile"],
                   check=True)
    print(f"ok    bfcl-eval {BFCL_VERSION} in {BFCL_ENV}")
    return 0


def _in_bfcl_env() -> bool:
    import importlib.util
    return importlib.util.find_spec("bfcl_eval") is not None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["setup"]:
        return setup()
    if not _in_bfcl_env():
        if not BFCL_PY.is_file():
            print("refused: the BFCL environment is missing: ./gb bfcl setup")
            return 2
        env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run([str(BFCL_PY), "-m", "grcbench.bfcl_run", *argv], env=env).returncode
    p = argparse.ArgumentParser(prog="gb bfcl")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config-id", required=True)
    g.add_argument("--base-url", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--run-ids", type=Path, default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--config-id", required=True)
    e.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "generate":
            m = generate(a.config_id, a.base_url, a.model, a.out, a.threads, a.run_ids)
            print(f"generated: {m['registry_name']} in {m['generate_seconds']} s -> {a.out}")
        else:
            s = evaluate(a.config_id, a.out)
            print(json.dumps({k: s[k] for k in ("config_id", "non_live_overall_acc", "live_overall_acc",
                                                 "plumbing_check")}, indent=1))
    except BfclError as exc:
        print(f"refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
