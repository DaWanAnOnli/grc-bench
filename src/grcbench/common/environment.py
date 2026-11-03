"""Version pins and the environment record (`reports/environment.json`).

The benchmark is only meaningful on the pinned GNU Radio and GRC-Agent versions (eval plan §4.4:
"One pinned GRC-Agent version"; proposal §5: "GNU Radio 3.10.x pinned"). Docker is not available on
this machine yet, so the pin is enforced here instead of by an image; docker/Dockerfile records the
intended image.
"""

from __future__ import annotations

import platform
import subprocess
import sys

from .io import read_json, write_json
from .paths import (CACHE, GNURADIO_VERSION, GRC_AGENT, GRC_AGENT_COMMIT, LLAMA_CPP_BUILD,
                    REPORTS)


def _git_head(repo) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _gnuradio_version() -> str:
    try:
        from gnuradio import gr
        return gr.version()
    except Exception:
        return "unavailable"


def check_pins() -> None:
    head = _git_head(GRC_AGENT)
    gr_version = _gnuradio_version()
    problems = []
    if not head.startswith(GRC_AGENT_COMMIT):
        problems.append(f"GRC-Agent at {head[:12]}, expected {GRC_AGENT_COMMIT}")
    if gr_version != GNURADIO_VERSION:
        problems.append(f"GNU Radio {gr_version}, expected {GNURADIO_VERSION}")
    if problems:
        sys.exit("pin check failed: " + "; ".join(problems))


def _run(cmd: list[str], env: dict | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout + out.stderr).strip() if out.returncode == 0 else None


def _gpu() -> dict:
    """The GPU (nvidia-smi) and the CUDA build of the pinned llama.cpp tag (`./gb fetch --cuda`)."""
    import os
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"])
    cuda_dir = CACHE / "llama.cpp" / f"{LLAMA_CPP_BUILD}-cuda"
    server = cuda_dir / "llama-server"
    version = None
    if server.exists():
        version = _run([str(server), "--version"],
                       env={**os.environ, "LD_LIBRARY_PATH": f"{cuda_dir}:/usr/lib/wsl/lib"})
    return {"nvidia_smi": smi, "llama_cpp_cuda_dir": str(cuda_dir) if server.exists() else None,
            "llama_cpp_cuda_version": version.splitlines()[-2:] if version else None}


def main(argv: list[str]) -> int:
    path = REPORTS / "environment.json"
    env = read_json(path) if path.exists() else {}
    env.update({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gnuradio_version": _gnuradio_version(),
        "grc_agent_commit": _git_head(GRC_AGENT),
        "pins": {"grc_agent_commit": GRC_AGENT_COMMIT, "gnuradio": GNURADIO_VERSION,
                 "llama_cpp_build": LLAMA_CPP_BUILD},
        "cache_dir": str(CACHE),
    })
    env["gpu"] = _gpu()
    for mod in ("numpy", "scipy", "yaml", "jsonschema", "tiktoken", "pydantic_ai", "pytest"):
        try:
            m = __import__(mod)
            env.setdefault("packages", {})[mod] = getattr(m, "__version__", "present")
        except Exception:
            env.setdefault("packages", {})[mod] = "missing"
    write_json(path, env)
    print(f"wrote {path}")
    check_pins()
    print("pins ok")
    return 0
