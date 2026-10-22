"""Embedding similarity with EmbeddingGemma on llama.cpp b10333 (eval plan §6 audit, plan WS2).

`LlamaEmbedServer` starts llama-server from ~/grc-bench-cache/llama.cpp/b10333 in embedding mode
(--embeddings, mean pooling: EmbeddingGemma is a mean-pooling model) on a free 127.0.0.1 port, with
LD_LIBRARY_PATH set to the build directory, waits for /health, and always stops the server when the
`with` block ends (and, on Linux, when the parent process dies). Context and both batch sizes are
2,048 tokens, EmbeddingGemma's trained context, so any single input up to that length fits in one
physical batch (as GRC-Agent's embed_runtime.py sets them); SLOTS slots share that context through
a unified KV cache, so several short texts are embedded in one forward pass. The GGUF is the one
pinned by `./gb fetch` (and by GRC-Agent) and its SHA-256 is checked before use.

`EmbedClient` sends texts in small batches to the OpenAI-compatible /v1/embeddings endpoint,
L2-normalises the vectors, and caches each one in ~/grc-bench-work/embed_cache/<model>/ keyed by
sha256 of the exact string sent (prompt included), so re-runs only embed new text. The cache holds
vectors only, never text.

Every text gets EmbeddingGemma's symmetric "sentence similarity" prompt on both sides, because the
audit asks whether two texts say the same thing, not whether one answers the other. Whitespace is
collapsed and texts are cut at MAX_CHARS characters (corpus passages are ~64 words, far below).
The embedding model only maps text to vectors; it never generates text, and nothing it sees is used
to create training data.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from grcbench.common.fetch import FILES
from grcbench.common.io import sha256_file, sha256_text
from grcbench.common.paths import CACHE, LLAMA_CPP_BUILD, WORK

from .corpora import clean

MODEL_PATH = CACHE / FILES["embeddinggemma"]["dest"]
MODEL_SHA256 = FILES["embeddinggemma"]["sha256"]
MODEL_TAG = Path(FILES["embeddinggemma"]["dest"]).stem          # embeddinggemma-300m-qat-Q8_0
LLAMA_DIR = CACHE / "llama.cpp" / LLAMA_CPP_BUILD
CACHE_DIR = WORK / "embed_cache"
LOG_PATH = WORK / "logs" / "embed_server.log"

PROMPT = "task: sentence similarity | query: "
CONTEXT = 2048
MAX_CHARS = 2000
BATCH = 16              # texts per request
SLOTS = 4               # server slots sharing one unified KV cache: short texts are batched together
DEFAULT_THREADS = 4     # measured best on the 4P+8E laptop CPU in WSL (more threads were slower)


class EmbedError(RuntimeError):
    pass


def normalise(m: np.ndarray) -> np.ndarray:
    """L2-normalise rows; an all-zero row stays zero."""
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine matrix of two sets of L2-normalised rows."""
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def verify_model(path: Path = MODEL_PATH) -> str:
    if not path.exists():
        raise EmbedError(f"embedding model missing: {path} (run ./gb fetch)")
    digest = sha256_file(path)
    if digest != MODEL_SHA256:
        raise EmbedError(f"embedding model SHA-256 mismatch: {digest} != pinned {MODEL_SHA256}")
    return digest


class EmbedClient:
    """Batching, caching, normalising client for an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, cache_dir: Path | None = CACHE_DIR, model_tag: str = MODEL_TAG,
                 batch: int = BATCH, timeout: float = 900.0, api_key: str | None = None,
                 prompt: str = PROMPT, max_chars: int = MAX_CHARS):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = (cache_dir / model_tag) if cache_dir is not None else None
        self.batch = batch
        self.timeout = timeout
        self._api_key = api_key
        self.prompt = prompt
        self.max_chars = max_chars
        self.requests = 0
        self.computed = 0
        self.cached = 0
        self.tokens = 0
        self.seconds = 0.0
        self.dim: int | None = None

    # -- cache ------------------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / key[:2] / f"{key}.f32"

    def _load(self, key: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        v = np.frombuffer(p.read_bytes(), dtype="<f4")
        return v if v.size and (self.dim is None or v.size == self.dim) else None

    def _save(self, key: str, vec: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(np.asarray(vec, dtype="<f4").tobytes())
        os.replace(tmp, p)

    # -- HTTP -------------------------------------------------------------------------------
    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"input": inputs, "model": MODEL_TAG, "encoding_format": "float"}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/embeddings", data=body, headers=headers)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise EmbedError(f"embedding request failed: HTTP {exc.code} {exc.read()[:300]!r}") from exc
        self.seconds += time.monotonic() - t0
        self.requests += 1
        self.tokens += int((out.get("usage") or {}).get("prompt_tokens", 0))
        data = sorted(out["data"], key=lambda d: d["index"])
        if len(data) != len(inputs):
            raise EmbedError(f"server returned {len(data)} embeddings for {len(inputs)} inputs")
        return [d["embedding"] for d in data]

    # -- public -----------------------------------------------------------------------------
    def prepare(self, text: str) -> str:
        return self.prompt + clean(text)[: self.max_chars]

    def embed(self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """L2-normalised embeddings, one row per text, in input order."""
        prepared = [self.prepare(t) for t in texts]
        keys = [sha256_text(s) for s in prepared]
        vecs: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for k, s in zip(keys, prepared):
            if k in vecs or k in missing:
                continue
            v = self._load(k)
            if v is not None:
                vecs[k] = v
                self.cached += 1
            else:
                missing[k] = s
        todo = list(missing.items())
        for start in range(0, len(todo), self.batch):
            chunk = todo[start:start + self.batch]
            for (k, _), raw in zip(chunk, self._post([s for _, s in chunk])):
                v = normalise(np.asarray([raw], dtype=np.float32))[0]
                if self.dim is None:
                    self.dim = v.size
                vecs[k] = v
                self._save(k, v)
                self.computed += 1
            if progress:
                progress(min(start + self.batch, len(todo)), len(todo))
        if not keys:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        return normalise(np.stack([vecs[k] for k in keys]))


def _die_with_parent() -> None:          # runs in the child before exec (Linux only)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:
        pass


class LlamaEmbedServer:
    """`with LlamaEmbedServer() as client:` starts llama-server and always stops it afterwards."""

    def __init__(self, model: Path = MODEL_PATH, llama_dir: Path = LLAMA_DIR, threads: int | None = None,
                 ctx: int = CONTEXT, port: int | None = None, log_path: Path = LOG_PATH,
                 ready_timeout: float = 300.0, binary: Path | None = None, cache_dir: Path | None = CACHE_DIR,
                 batch: int = BATCH):
        self.model = model
        self.llama_dir = llama_dir
        self.binary = binary or llama_dir / "llama-server"
        self.threads = threads or int(os.environ.get("GRC_BENCH_EMBED_THREADS", DEFAULT_THREADS))
        self.ctx = ctx
        self.port = port
        self.log_path = log_path
        self.ready_timeout = ready_timeout
        self.cache_dir = cache_dir
        self.batch = batch
        self.proc: subprocess.Popen | None = None
        self._api_key = secrets.token_urlsafe(18)

    def command(self) -> list[str]:
        return [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-t", str(self.threads), "-c", str(self.ctx), "-b", str(self.ctx), "-ub", str(self.ctx),
                "-np", str(SLOTS), "-kvu", "--embeddings", "--pooling", "mean", "--no-webui",
                "--api-key", self._api_key]

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def __enter__(self) -> EmbedClient:
        self.port = self.port or free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(self.llama_dir) + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
        cmd = self.command()
        with open(self.log_path, "ab") as log:
            shown = " ".join("***" if a == self._api_key else a for a in cmd)       # never log the key
            log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')}: {shown}\n".encode())
            log.flush()
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env,
                                         preexec_fn=_die_with_parent if os.name == "posix" else None)
        deadline = time.monotonic() + self.ready_timeout
        try:
            while not self._healthy():
                if self.proc.poll() is not None:
                    raise EmbedError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
                if time.monotonic() > deadline:
                    raise EmbedError(f"llama-server not ready after {self.ready_timeout:.0f}s; see {self.log_path}")
                time.sleep(0.25)
        except BaseException:
            self.stop()
            raise
        return EmbedClient(f"http://127.0.0.1:{self.port}", cache_dir=self.cache_dir, batch=self.batch,
                           api_key=self._api_key)

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)

    def __exit__(self, *exc) -> None:
        self.stop()
