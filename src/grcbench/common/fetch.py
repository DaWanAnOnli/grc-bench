"""`./gb fetch` — download the pinned third-party files into ~/grc-bench-cache (outside the repo).

Every file is pinned by URL and SHA-256 and verified after download. None of these is a model under
test: Qwen3-4B is only for plumbing (smoke) tests of the runner, EmbeddingGemma only for the
contamination audit, and llama.cpp b10333 is the build GRC-Agent itself pins (embed_runtime.py).
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .io import read_json, sha256_file, write_json
from .paths import CACHE, REPORTS, ensure

FILES = {
    "llama_cpp": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10333/llama-b10333-bin-ubuntu-x64.tar.gz",
        "sha256": "936ce04d98abe2a977e9dd2ff92659bb96947e136acee8f2bc3e21d8eaebbf23",
        "dest": "llama.cpp/llama-b10333-bin-ubuntu-x64.tar.gz",
    },
    "qwen3_4b": {
        "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        "dest": "models/Qwen3-4B-Q4_K_M.gguf",
    },
    "embeddinggemma": {
        "url": "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf",
        "sha256": "6fa0c02a9c302be6f977521d399b4de3a46310a4f2621ee0063747881b673f67",
        "dest": "models/embeddinggemma-300m-qat-Q8_0.gguf",
    },
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    ensure(dest.parent)
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def main(argv: list[str]) -> int:
    if "--cuda" in argv:            # the pinned tag built with CUDA for this machine's GPU (no sudo)
        return subprocess.run(["bash", str(Path(__file__).with_name("build_llama_cuda.sh"))]).returncode
    record = {}
    for name, spec in FILES.items():
        dest = CACHE / spec["dest"]
        if not dest.exists() or sha256_file(dest) != spec["sha256"]:
            print(f"downloading {name} ...", flush=True)
            _download(spec["url"], dest)
        digest = sha256_file(dest)
        ok = digest == spec["sha256"]
        print(f"{name}: {'ok' if ok else 'SHA-256 MISMATCH'}  {dest}", flush=True)
        if not ok:
            return 1
        record[name] = {"url": spec["url"], "sha256": digest, "path": str(dest)}
    # unpack llama.cpp
    tarball = CACHE / FILES["llama_cpp"]["dest"]
    target = CACHE / "llama.cpp" / "b10333"
    if not (target / "llama-server").exists():
        ensure(target)
        with tarfile.open(tarball) as tf:
            for member in tf.getmembers():
                if member.isfile() or member.issym():
                    member.name = Path(member.name).name
                    tf.extract(member, target, filter="data")
    server = target / "llama-server"
    ver = subprocess.run([str(server), "--version"], capture_output=True, text=True,
                         env={"LD_LIBRARY_PATH": str(target)})
    record["llama_cpp"]["unpacked"] = str(target)
    record["llama_cpp"]["version_output"] = (ver.stdout + ver.stderr).strip().splitlines()[:3]
    # tiktoken cl100k, cached for offline (sandboxed) use
    import tiktoken
    tiktoken.get_encoding("cl100k_base")
    record["tiktoken_cache"] = str(CACHE / "tiktoken")
    env_path = REPORTS / "environment.json"
    env = read_json(env_path) if env_path.exists() else {}
    env["third_party"] = record
    write_json(env_path, env)
    print("recorded in reports/environment.json")
    return 0
