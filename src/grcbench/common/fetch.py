"""`./gb fetch` — download the pinned third-party files into ~/grc-bench-cache (outside the repo).

Every file is pinned by URL and SHA-256 and verified after download. None of these is a model under
test: Qwen3-4B is only for plumbing (smoke) tests of the runner, EmbeddingGemma only for the
contamination audit. llama.cpp LLAMA_CPP_BUILD serves every model; the contamination audit's embedding
server runs EMBED_LLAMA_CPP_BUILD, the build GRC-Agent's own embed_runtime.py pins.

    ./gb fetch          the CPU builds, the smoke model and the embedding model
    ./gb fetch --cuda   llama.cpp's own CUDA 12.8 build of the pinned tag and its CUDA runtime
                        (~/grc-bench-cache/llama.cpp/<tag>-cuda; it needs an NVIDIA driver for CUDA 12.8 or newer)
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .io import read_json, sha256_file, write_json
from .paths import CACHE, EMBED_LLAMA_CPP_BUILD, LLAMA_CPP_BUILD, REPORTS, ensure

RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
FILES = {
    "llama_cpp": {
        "url": f"{RELEASES}/{LLAMA_CPP_BUILD}/llama-{LLAMA_CPP_BUILD}-bin-ubuntu-x64.tar.gz",
        "sha256": "1b69032f9ab60dc6aa909689fda941fa1b9162f6ab46b863582d5649d96ac6be",
        "dest": f"llama.cpp/llama-{LLAMA_CPP_BUILD}-bin-ubuntu-x64.tar.gz",
    },
    "llama_cpp_embed": {
        "url": f"{RELEASES}/{EMBED_LLAMA_CPP_BUILD}/llama-{EMBED_LLAMA_CPP_BUILD}-bin-ubuntu-x64.tar.gz",
        "sha256": "936ce04d98abe2a977e9dd2ff92659bb96947e136acee8f2bc3e21d8eaebbf23",
        "dest": f"llama.cpp/llama-{EMBED_LLAMA_CPP_BUILD}-bin-ubuntu-x64.tar.gz",
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

CUDA_FILES = {
    "llama_cpp_cuda": {
        "url": f"{RELEASES}/{LLAMA_CPP_BUILD}/llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "sha256": "0170c61c0ddf6208215a68b36f78f45b1641631317736c093f3faa40f05f8e85",
        "dest": f"llama.cpp/llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    },
    "cuda_runtime": {
        "url": f"{RELEASES}/{LLAMA_CPP_BUILD}/cudart-llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "sha256": "a17d59ed20873a82cbf2ff6c651d5bb432689d8569b4af0b6b1c7e35aeb001ba",
        "dest": f"llama.cpp/cudart-llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    },
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    ensure(dest.parent)
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def download_checked(files: dict[str, dict[str, str]]) -> dict[str, dict[str, str]] | None:
    """Download each file unless the cache already holds it with its SHA-256; None if one does not match."""
    record = {}
    for name, spec in files.items():
        dest = CACHE / spec["dest"]
        if not dest.exists() or sha256_file(dest) != spec["sha256"]:
            print(f"downloading {name} ...", flush=True)
            _download(spec["url"], dest)
        digest = sha256_file(dest)
        ok = digest == spec["sha256"]
        print(f"{name}: {'ok' if ok else 'SHA-256 MISMATCH'}  {dest}", flush=True)
        if not ok:
            return None
        record[name] = {"url": spec["url"], "sha256": digest, "path": str(dest)}
    return record


def unpack(archives: list[Path], target: Path) -> None:
    """Every file of the archives, flat in `target` (a release archive holds one folder, llama-<tag>/)."""
    ensure(target)
    for archive in archives:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.isfile() or member.issym():
                    member.name = Path(member.name).name
                    tf.extract(member, target, filter="data")


def version_lines(llama_dir: Path) -> list[str]:
    ver = subprocess.run([str(llama_dir / "llama-server"), "--version"], capture_output=True, text=True,
                         env={"LD_LIBRARY_PATH": f"{llama_dir}:/usr/lib/wsl/lib"})
    return (ver.stdout + ver.stderr).strip().splitlines()[-2:]


def _record(update: dict) -> None:
    env_path = REPORTS / "environment.json"
    env = read_json(env_path) if env_path.exists() else {}
    env["third_party"] = {**(env.get("third_party") or {}), **update}
    write_json(env_path, env)
    print("recorded in reports/environment.json")


def main(argv: list[str]) -> int:
    if "--cuda" in argv:
        record = download_checked(CUDA_FILES)
        if record is None:
            return 1
        target = CACHE / "llama.cpp" / f"{LLAMA_CPP_BUILD}-cuda"
        if not (target / "llama-server").exists():
            unpack([CACHE / spec["dest"] for spec in CUDA_FILES.values()], target)
        record["llama_cpp_cuda"].update(unpacked=str(target), version_output=version_lines(target))
        print("\n".join(record["llama_cpp_cuda"]["version_output"]))
        _record(record)
        return 0
    record = download_checked(FILES)
    if record is None:
        return 1
    for name, build in (("llama_cpp", LLAMA_CPP_BUILD), ("llama_cpp_embed", EMBED_LLAMA_CPP_BUILD)):
        target = CACHE / "llama.cpp" / build
        if not (target / "llama-server").exists():
            unpack([CACHE / FILES[name]["dest"]], target)
        record[name].update(unpacked=str(target), version_output=version_lines(target))
    # tiktoken cl100k, cached for offline (sandboxed) use
    import tiktoken
    tiktoken.get_encoding("cl100k_base")
    record["tiktoken_cache"] = str(CACHE / "tiktoken")
    _record(record)
    return 0
