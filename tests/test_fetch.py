"""Tests for `./gb fetch`: what is pinned, and how a llama.cpp release archive is unpacked (no network)."""

from __future__ import annotations

import io
import re
import tarfile

from grcbench.common import fetch, paths


def test_the_model_server_builds_are_the_pinned_tag():
    """The CPU build and the CUDA build (with its CUDA runtime) are llama.cpp's own release downloads of the
    pinned tag, each pinned by SHA-256."""
    assert paths.LLAMA_CPP_BUILD == "b12020"
    serving = {"llama_cpp": fetch.FILES["llama_cpp"], **fetch.CUDA_FILES}
    assert len(fetch.CUDA_FILES) == 2
    for spec in serving.values():
        assert f"/releases/download/{paths.LLAMA_CPP_BUILD}/" in spec["url"]
        assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
    assert all("ubuntu-cuda-12.8-x64" in spec["url"] for spec in fetch.CUDA_FILES.values())


def test_the_embedding_build_is_grc_agents_pin():
    """The contamination audit's embedding server keeps the build GRC-Agent's embed_runtime.py pins."""
    from grcbench.contamination import embed
    assert paths.EMBED_LLAMA_CPP_BUILD == "b10333"
    assert f"/releases/download/{paths.EMBED_LLAMA_CPP_BUILD}/" in fetch.FILES["llama_cpp_embed"]["url"]
    assert embed.LLAMA_DIR.name == paths.EMBED_LLAMA_CPP_BUILD


def _archive(path, files):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_unpack_puts_every_file_of_the_archives_in_one_folder(tmp_path):
    """A release archive holds one folder (llama-bXXXX/); the CUDA runtime archive its own. Both land flat in
    the target folder, where LD_LIBRARY_PATH finds the libraries."""
    a, b = tmp_path / "llama.tar.gz", tmp_path / "cudart.tar.gz"
    _archive(a, {"llama-b12020/llama-server": b"server", "llama-b12020/libggml-cuda.so": b"lib"})
    _archive(b, {"cudart/libcudart.so.12": b"rt"})
    target = tmp_path / "out"
    fetch.unpack([a, b], target)
    assert sorted(p.name for p in target.iterdir()) == ["libcudart.so.12", "libggml-cuda.so", "llama-server"]
    assert (target / "llama-server").read_bytes() == b"server"


def test_fetch_cuda_downloads_the_release_instead_of_compiling(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(fetch, "CACHE", tmp_path)
    monkeypatch.setattr(fetch, "download_checked", lambda files: got.setdefault("files", files) and
                        {n: {"path": str(tmp_path / s["dest"])} for n, s in files.items()})
    monkeypatch.setattr(fetch, "unpack", lambda archives, target: got.setdefault("unpack", (archives, target)))
    monkeypatch.setattr(fetch, "version_lines", lambda d: ["version: 12020 (0ec4027a2)"])
    monkeypatch.setattr(fetch, "_record", lambda update: got.setdefault("record", update))
    assert fetch.main(["--cuda"]) == 0
    assert got["files"] == fetch.CUDA_FILES
    assert got["unpack"][1] == tmp_path / "llama.cpp" / f"{paths.LLAMA_CPP_BUILD}-cuda"
    assert got["record"]["llama_cpp_cuda"]["version_output"] == ["version: 12020 (0ec4027a2)"]
    assert not (fetch.Path(fetch.__file__).with_name("build_llama_cuda.sh")).exists()
