"""GRC-Bench's Colab steps 1-11 as one script: the code cells of grcbench_colab_test.ipynb (all but its Settings
form), written by make_notebook.py; change them there. Another notebook runs it with its own settings, e.g. the
team's evaluation notebook (grc-model-eval, colab/eval.ipynb) with the model server it already runs:

    import runpy
    runpy.run_path("grcbench_colab_steps.py", init_globals={
        "DRIVE_FOLDER": "grc-bench-colab", "PACKAGE": "grc-bench-colab.zip", "GR_SETUP": "ubuntu-26.04",
        "TIERS": "K,F,A,E", "LIMIT_K": 5, "LIMIT_F": 2, "LIMIT_A": 1, "JUDGE_IN_COLAB": False, "MODEL_STEPS": True,
        "SERVER_URL": "http://127.0.0.1:8181/v1", "SERVER_MODEL": "nanbeige42",
        "MODEL_FILE": "/content/models/nanbeige42/Nanbeige4.2-3B-Q8_0.gguf",
        "RESULTS_DIR": "/content/drive/MyDrive/grc-eval/runs/<run>/grc-bench",
        "STOP_SERVER": lambda: h.stop_server(SERVER)})
"""
import json, os, re, shlex, shutil, subprocess, sys, time, zipfile
from pathlib import Path

# paths as ./gb sees them, in both setups (/root is hidden inside GRC-Bench's sandbox)
PKG, WORK, CACHE, AGENT = (Path("/root") / n for n in ("grc-bench-colab", "grc-bench-work", "grc-bench-cache", "grc-agent"))
HOST_PKG = PKG                                       # the package unzipped on Colab itself (u2604.py runs from here)
ROOTFS = Path("/opt/u2604")                          # the Ubuntu 26.04 root filesystem
GR = Path("/opt/gr")                                 # conda-forge's GNU Radio (fallback setup)
MM = Path("/opt/micromamba/bin/micromamba")
UBUNTU = GR_SETUP == "ubuntu-26.04"
BASE_URL = "https://cdimage.ubuntu.com/ubuntu-base/releases/26.04/release/ubuntu-base-26.04.1-base-amd64.tar.gz"
BASE_SHA256 = "a496a960472ce474a59590b8987d3a1135d3cbef1991f3b1abe8cacfea8bf85a"
APT_PACKAGES = ("gnuradio=3.10.12.0-6 bubblewrap git ca-certificates curl python3 python3-yaml python3-numpy "
                "python3-scipy")                     # as docker/Dockerfile
INNER_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "QT_QPA_PLATFORM": "offscreen", "GRC_BENCH_CACHE": str(CACHE),
             "GRC_BENCH_WORK": str(WORK), "GRC_AGENT_REPO": str(AGENT),
             "GRC_BENCH_LLAMA_BACKEND": "cuda"}          # "auto" looks for WSL's GPU library, which Colab does not have
os.environ.update({**INNER_ENV, "MAMBA_ROOT_PREFIX": "/opt/mamba"})
LLAMA_BUILD = "b12020"                               # the pinned llama.cpp tag (src/grcbench/common/paths.py)
# Set by a notebook that runs these steps as a script (grcbench_colab_steps.py) with a model server of its own:
# that server answers Tiers K, F and A, Tier E measures its model file, and the results go to its run folder.
SERVER_URL = globals().get("SERVER_URL")              # e.g. "http://127.0.0.1:8181/v1"; None: the smoke model's own
SERVER_MODEL = globals().get("SERVER_MODEL")          # the model name that server answers to
MODEL_FILE = Path(globals().get("MODEL_FILE") or CACHE / "models/Qwen3-4B-Q4_K_M.gguf")
RESULTS_DIR = globals().get("RESULTS_DIR")            # a Drive folder; None: My Drive/<DRIVE_FOLDER>/results
STOP_SERVER = globals().get("STOP_SERVER")            # stops that server before Tier E, which measures alone
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
RESULT = {"notebook": "grcbench_colab_test 5", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "settings": {"gr_setup": GR_SETUP, "tiers": TIERS, "limit_k": LIMIT_K, "limit_f": LIMIT_F,
                       "limit_a": LIMIT_A, "judge_in_colab": JUDGE_IN_COLAB,
                       **({"server_url": SERVER_URL, "server_model": SERVER_MODEL, "model_file": str(MODEL_FILE)}
                          if SERVER_URL else {})}, "steps": {}}
OUT = Path("/content/grcbench-results") / STAMP       # copied to Drive at each save
OUT.mkdir(parents=True, exist_ok=True)
DRIVE_OUT = None
QUIET = re.compile(r"^\s*$|^(  Downloading|Downloading|Collecting|Building|\[\d+/\d+\] |-- |Resolved |Prepared |Installed |"
                   r"Uninstalled |Audited | \+ | - |Transaction|Linking|Preparing|Verifying|Executing|info|warning libmamba|"
                   r"Get:|Selecting|Unpacking|Setting up|Processing triggers|\(Reading database|sysctl: permission)")


class StopHere(Exception):
    """Stops Run all with a message saying what to do."""


def log(step, msg):
    print(f"[{time.strftime('%H:%M:%S')} | {step}] {msg}", flush=True)


def save():
    RESULT["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "feasibility.json").write_text(json.dumps(RESULT, indent=1))
    if DRIVE_OUT is not None:
        shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True)


def record(step, **kw):
    RESULT["steps"].setdefault(step, {}).update(kw)
    save()


def sh(cmd, step, *, name, cwd=None, env=None, show=60, timeout=None):
    """Run a bash command on Colab itself; full output to OUT/logs/<name>.log; prints at most `show` informative
    lines and the last 15 lines. Returns (exit code, output lines, seconds)."""
    (OUT / "logs").mkdir(exist_ok=True)
    path = OUT / "logs" / f"{name}.log"
    t0, lines, shown = time.time(), [], 0
    with open(path, "w") as fh:
        proc = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, env={**os.environ, **(env or {})}, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, errors="replace")
        for line in proc.stdout:
            fh.write(line)
            lines.append(line.rstrip("\n"))
            if shown < show and not QUIET.search(line):
                print("    " + line.rstrip()[:220], flush=True)
                shown += 1
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -9
    if shown >= show:
        print("    ... (full output in " + str(path) + "); last lines:")
        for line in lines[-15:]:
            print("    " + line[:220])
    secs = round(time.time() - t0, 1)
    log(step, f"exit {code} after {secs} s: {cmd[-100:]}")
    return code, lines, secs


def host(p):
    """Where a path that ./gb sees lies on Colab itself."""
    return ROOTFS / str(p).lstrip("/") if UBUNTU else Path(p)


def driver_dir():
    """The folder of Colab's NVIDIA driver libraries (libcuda.so.1)."""
    out = subprocess.run(["bash", "-c", "ldconfig -p | grep -m1 'libcuda.so.1 '"], capture_output=True, text=True).stdout
    m = re.search(r"=> (\S+)/libcuda\.so\.1", out)
    return m.group(1) if m else "/usr/lib64-nvidia"


def nvidia_binds():
    """Colab's NVIDIA driver libraries, one file at a time, into /opt/nvidia-host inside the container (never a whole
    system library folder, which would replace the container's own libraries)."""
    d = Path(driver_dir())
    files = sorted({*d.glob("libcuda.so*"), *d.glob("libnvidia-*.so*"), *d.glob("libnvcuvid.so*"),
                    *d.glob("libdxcore.so*")})                # libdxcore: WSL's GPU driver (not on Colab)
    wsl = ["/usr/lib/wsl/drivers"] if Path("/usr/lib/wsl/drivers").is_dir() else []   # WSL: where libcuda's driver lives
    return [f"{f}:/opt/nvidia-host/{f.name}" for f in files] + wsl


def run_in(cmd, step, *, name, show=60, timeout=None, env=None):
    """Run a bash command where ./gb lives: inside the Ubuntu 26.04 container, or on Colab itself."""
    env = {**INNER_ENV, **(env or {})}
    if not UBUNTU:
        return sh(cmd, step, name=name, env=env, show=show, timeout=timeout)
    env["LD_LIBRARY_PATH"] = "/opt/nvidia-host"
    binds = ["/content", *nvidia_binds()]
    smi = shutil.which("nvidia-smi")
    if smi:
        binds.append(f"{smi}:/usr/local/bin/nvidia-smi")
    args = " ".join([f"--bind {shlex.quote(b)}" for b in binds] + [f"--env {shlex.quote(k + '=' + v)}" for k, v in env.items()])
    return sh(f"python3 {HOST_PKG}/colab/u2604.py {ROOTFS} {args} -- {shlex.quote(cmd)}", step, name=name, show=show,
              timeout=timeout)


def gb(args, step_key, *, name, show=60, timeout=None, env=None):
    return run_in(f"cd {PKG} && ./gb {args}", step_key, name=name, show=show, timeout=timeout, env=env)


def step(key, title, fn):
    """Run one step; record ok / failed (with the error) / skipped and its time. Later steps still run. A step that
    returns {"failed": reason, ...} is recorded as failed with its details, {"skipped": reason} as skipped."""
    print("=" * 100 + f"\n{key}: {title}\n" + "=" * 100, flush=True)
    t0 = time.time()
    try:
        detail = fn() or {}
        failed, skipped = detail.pop("failed", None), detail.pop("skipped", None)
        detail = {("result_" + k if k in ("title", "status", "seconds", "error") else k): v for k, v in detail.items()}
        if failed:
            record(key, title=title, status="failed", seconds=round(time.time() - t0, 1), error=failed, **detail)
            log(key, f"FAILED: {failed}  (recorded; the notebook goes on)")
            return
        if skipped:
            record(key, title=title, status="skipped", seconds=round(time.time() - t0, 1), error=skipped, **detail)
            log(key, f"skipped: {skipped}")
            return
        record(key, title=title, status="ok", seconds=round(time.time() - t0, 1), **detail)
    except StopHere:
        record(key, title=title, status="failed", seconds=round(time.time() - t0, 1))
        raise
    except Exception as exc:
        record(key, title=title, status="failed", seconds=round(time.time() - t0, 1), error=f"{type(exc).__name__}: {exc}")
        log(key, f"FAILED: {type(exc).__name__}: {exc}  (recorded; the notebook goes on)")

print("helpers ready; results go to", OUT)

def machine():
    if "google.colab" not in sys.modules:
        raise StopHere("This notebook is written for Google Colab.")
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip().split(", ")
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        cuda = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)     # newer drivers: "CUDA UMD Version"
    elif MODEL_STEPS:
        raise StopHere("No GPU: Runtime > Change runtime type > T4 GPU > Save, then Runtime > Run all "
                       "(or untick MODEL_STEPS in Settings: the checks alone run on a CPU runtime).")
    else:                                                # the checks (steps 1-7b) need no GPU
        q, cuda = [None] * 4, None
        log("step 1", "no GPU: a CPU runtime, enough for the checks (MODEL_STEPS is off)")
    osr = dict(l.rstrip().split("=", 1) for l in open("/etc/os-release") if "=" in l)
    caps = re.search(r"CapEff:\s*(\w+)", open("/proc/self/status").read())
    info = {"gpu": q[0], "driver": q[1], "gpu_memory": q[2], "compute_cap": q[3],
            "driver_cuda": cuda.group(1) if cuda else None, "os": osr["PRETTY_NAME"].strip('"'),
            "python": sys.version.split()[0], "cpus": os.cpu_count(),
            "memory_gb": round(int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 1024 ** 2, 1),
            "free_disk_gb": round(shutil.disk_usage("/content").free / 1e9), "uid": os.getuid(),
            "capabilities": caps.group(1) if caps else None, "kernel": os.uname().release}
    for k, v in info.items():
        log("step 1", f"{k:13s} {v}")
    return info

step("1_machine", "the machine", machine)
GPU = RESULT["steps"]["1_machine"]

def package():
    global DRIVE_OUT
    root, drive_error = None, None
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive") / DRIVE_FOLDER
    except Exception as exc:                            # e.g. "credential propagation was unsuccessful"
        drive_error = f"{type(exc).__name__}: {exc}"
        log("step 2", f"Google Drive did not mount ({drive_error}); looking for /content/{PACKAGE} (the Files pane)")
    zpath = next((p for p in ([root / PACKAGE] if root else []) + [Path("/content") / PACKAGE] if p.exists()), None)
    if zpath is None:
        raise StopHere(f"{PACKAGE} not found: put it in My Drive/{DRIVE_FOLDER}/, or upload it to the Files pane, "
                       "and Run all again.")
    gpu = re.sub(r"[^A-Za-z0-9]+", "-", GPU["gpu"] or "CPU").strip("-")
    if RESULTS_DIR:
        DRIVE_OUT = Path(RESULTS_DIR) / f"{STAMP}_{gpu}"
    elif root is not None:
        DRIVE_OUT = root / "results" / f"{STAMP}_{gpu}"
    if HOST_PKG.exists():
        shutil.rmtree(HOST_PKG)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(HOST_PKG.parent)
    for p in [HOST_PKG / "gb", *HOST_PKG.rglob("*.sh")]:
        p.chmod(0o755)
    version = (HOST_PKG / "VERSION.txt").read_text().splitlines()[:3]
    for line in version:
        log("step 2", line)
    return {"package": str(zpath), "package_bytes": zpath.stat().st_size, "version": version, "drive": root is not None,
            "drive_error": drive_error, "drive_results": str(DRIVE_OUT) if DRIVE_OUT else None}

step("2_package", "Google Drive and the package", package)

def namespaces():
    probes = {}
    sh("apt-get -qq update >/dev/null && apt-get -qq install -y bubblewrap >/dev/null && bwrap --version", "step 3",
       name="03_bubblewrap_install")
    bw = "bwrap --ro-bind / / --dev /dev --tmpfs /tmp --unshare-pid --unshare-net --die-with-parent --new-session"
    for label, cmd in (("user namespace", "unshare --user --map-root-user true"),
                       ("user + mount ns, Colab's /proc bound",
                        "unshare --user --map-root-user --mount sh -c 'mkdir -p /tmp/pb && mount --rbind /proc /tmp/pb'"),
                       ("bwrap, own user ns, Colab's /proc", f"{bw} --unshare-user -- true"),
                       ("bwrap, fresh /proc (not needed)", f"{bw} --proc /proc --unshare-user -- true")):
        c, out, _ = sh(cmd, "step 3", name="03_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"))
        probes[label] = {"exit": c, "output": out[-2:]}
    for k, v in probes.items():
        if v["exit"] == 0:
            log("step 3", f"{k:40s} works")
        elif k.endswith("(not needed)"):
            log("step 3", f"{k:40s} refused, as expected on Colab")
        else:
            log("step 3", f"{k:40s} fails: {' '.join(v['output'])[:120]}")
    needed = [k for k in list(probes)[:3] if probes[k]["exit"] != 0]
    return {"probes": probes, **({"failed": "not allowed here: " + ", ".join(needed)} if needed else {})}

step("3_namespaces", "Linux namespaces and bubblewrap", namespaces)

def ubuntu_container():
    if not (HOST_PKG / "colab" / "u2604.py").exists():
        raise RuntimeError(f"the package is not unpacked in {HOST_PKG} (step 2)")
    tarball = Path("/content") / BASE_URL.rsplit("/", 1)[1]
    c, _, _ = sh(f"curl -Ls -o {tarball} {BASE_URL} && echo '{BASE_SHA256}  {tarball}' | sha256sum -c -", "step 4",
                 name="04_ubuntu_base")
    if c:
        raise RuntimeError("the Ubuntu 26.04 base system did not download or its SHA-256 differs")
    # processes left in the container by an earlier, interrupted run (their root is ROOTFS): stopped first
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if os.readlink(proc / "root") == str(ROOTFS):
                os.kill(int(proc.name), 9)
        except OSError:
            pass
    time.sleep(2)
    if ROOTFS.exists():
        shutil.rmtree(ROOTFS)
    ROOTFS.mkdir(parents=True)
    sh(f"tar -xzpf {tarball} -C {ROOTFS} --numeric-owner --exclude='dev/*'", "step 4", name="04_ubuntu_extract")
    c, _, secs = run_in("export DEBIAN_FRONTEND=noninteractive; apt-get -qq update && "
                        f"apt-get install -y -qq --no-install-recommends {APT_PACKAGES}", "step 4", name="04_apt_install",
                        show=30)
    if c:
        raise RuntimeError("apt-get could not install GNU Radio inside the container (log 04_apt_install.log)")
    c, out, _ = run_in("dpkg-query -W gnuradio bubblewrap python3; echo audit: $(dpkg --audit | wc -l) packages not configured; "
                       "python3 -c 'from gnuradio import gr; print(\"gnuradio\", gr.version())'", "step 4", name="04_check")
    if c or not any(l.startswith("gnuradio\t3.10.12.0-6") for l in out):
        raise RuntimeError("the container's GNU Radio is not 3.10.12.0-6 (log 04_check.log)")
    shutil.copytree(HOST_PKG, host(PKG), dirs_exist_ok=True)
    return {"setup": "ubuntu-26.04", "apt_seconds": secs, "check": out[-6:]}


def conda_gnuradio():
    if not MM.exists():
        sh("mkdir -p /opt/micromamba && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /opt/micromamba bin/micromamba",
           "step 4", name="04_micromamba")
    code, _, secs = sh(f"{MM} create -y -q -p {GR} -c conda-forge python=3.14 gnuradio=3.10.12.0", "step 4",
                       name="04_gnuradio_create", show=20)
    if code:
        raise RuntimeError("micromamba could not install gnuradio=3.10.12.0 (log 04_gnuradio_create.log)")
    out = subprocess.run([str(GR / "bin/python"), "-c", "from gnuradio import gr; print(gr.version())"], capture_output=True, text=True)
    version = out.stdout.strip() or out.stderr.strip()[-300:]
    # GRC-Bench reads the block catalog and examples from /usr/share/gnuradio and runs grcc from PATH
    links = {}
    share = Path("/usr/share/gnuradio")
    if share.exists() and not share.is_symlink():
        share.rename("/usr/share/gnuradio.ubuntu")
    if not share.exists():
        share.symlink_to(GR / "share/gnuradio")
    links["/usr/share/gnuradio"] = str(share.resolve())
    for tool in ("grcc", "gnuradio-companion", "gnuradio-config-info"):
        target, link = GR / "bin" / tool, Path("/usr/local/bin") / tool
        if target.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(target)
            links[str(link)] = str(target)
    log("step 4", f"GNU Radio {version} in {secs} s")
    return {"setup": "conda-forge", "gnuradio_version": version, "install_seconds": secs, "links": links}


def gnuradio():
    global UBUNTU
    if UBUNTU:
        try:
            return ubuntu_container()
        except Exception as exc:
            log("step 4", f"the Ubuntu 26.04 container did not work ({exc}); FALLBACK: conda-forge's GNU Radio on Colab, "
                          "for Tier K and E")
            UBUNTU = False
            return {**conda_gnuradio(), "ubuntu_container_error": f"{type(exc).__name__}: {exc}"}
    return conda_gnuradio()

step("4_gnuradio", "GNU Radio", gnuradio)

def grc_agent():
    py = AGENT / ".venv/bin/python"
    if UBUNTU:
        uv = "[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh; export PATH=/root/.local/bin:$PATH"
        base_python = "/usr/bin/python3"
    else:
        uv = "pip install -q uv==0.12.21"
        base_python = f"{GR}/bin/python"
    c, out, secs = run_in(f"set -e; {uv}; uv --version; "
                          f"[ -d {AGENT}/.git ] || git clone -q https://github.com/qoherent/grc-agent.git {AGENT}; "
                          f"cd {AGENT}; git checkout -q 6c98128; git rev-parse HEAD; "
                          f"uv venv -q --clear --system-site-packages --python {base_python} .venv; "
                          "uv sync -q --extra dev --locked --python .venv/bin/python", "step 5", name="05_grc_agent", show=30)
    if c:
        raise RuntimeError("GRC-Agent could not be installed (log 05_grc_agent.log)")
    head = next((l for l in reversed(out) if re.fullmatch(r"[0-9a-f]{40}", l.strip())), "")
    _, imp, _ = run_in(f"{py} -c 'import grc_agent, gnuradio, pydantic_ai; from gnuradio import gr; print(\"imports ok\", gr.version())'",
                       "step 5", name="05_imports")
    c_db, db_out, _ = run_in(f"cd {AGENT} && GRC_EMBED_BACKEND=lexical {py} -c \"from grc_agent.adapter import rag\n"
                             "for d in ('catalog', 'docs'):\n    p, m = rag.get_db_and_model(d)\n    rag._ensure_db_built(d, p, m)\"",
                             "step 5", name="05_knowledge_dbs")
    c_tok, _, _ = run_in(f"mkdir -p {CACHE}/tiktoken && TIKTOKEN_CACHE_DIR={CACHE}/tiktoken {py} -c "
                         "\"import tiktoken; tiktoken.get_encoding('cl100k_base')\"", "step 5", name="05_tiktoken")
    vec = host(AGENT) / "src/grc_agent/vectors"
    dbs = c_db == 0 and (vec / "catalog_lexical.db").exists() and (vec / "docs_lexical.db").exists()
    bad = [name for name, ok in (("knowledge databases", dbs), ("tokenizer", c_tok == 0)) if not ok]
    return {"head": head, "install_seconds": secs, "imports": imp[-1:], "knowledge_dbs": dbs,
            "knowledge_dbs_error": None if c_db == 0 else db_out[-1:], "tokenizer": c_tok == 0,
            **({"failed": " and ".join(bad) + " did not work (logs 05_*)"} if bad else {})}

step("5_grc_agent", "GRC-Agent 6c98128", grc_agent)

def llama():
    fetch_s = None
    if not SERVER_URL:                                   # the smoke model's own server needs the smoke model
        c, _, fetch_s = gb("fetch", "step 6", name="06_fetch")
        if c:
            raise RuntimeError("./gb fetch failed (log 06_fetch.log)")
    if not MODEL_STEPS:                                  # the checks of steps 7 and 7b need no CUDA build
        return {"fetch_seconds": fetch_s, "cuda": "not downloaded: MODEL_STEPS is off"}
    c, out, cuda_s = gb("fetch --cuda", "step 6", name="06_fetch_cuda")
    if c:
        raise RuntimeError("./gb fetch --cuda failed (log 06_fetch_cuda.log)")
    version = [l for l in out if l.startswith(("version:", "built with"))]
    log("step 6", f"llama.cpp {LLAMA_BUILD} with CUDA: {version}")
    return {"fetch_seconds": fetch_s, "cuda_fetch_seconds": cuda_s, "version": version}

step("6_llama_cpp", f"llama.cpp {LLAMA_BUILD} (CUDA)", llama)

SANDBOX_PROBE = ("from pathlib import Path; import tempfile; from grcbench.common import sandbox; "
                 "w = Path('/root/grc-bench-work'); w.mkdir(parents=True, exist_ok=True); d = Path(tempfile.mkdtemp(dir=w)); "
                 "r = sandbox.run(['bash', '-c', 'echo hi > x; echo y > /etc/nope 2>/dev/null || echo ro; "
                 "(exec 3<>/dev/tcp/1.1.1.1/53) 2>/dev/null && echo net || echo nonet; "
                 f"cat /proc/{os.getpid()}/environ >/dev/null 2>&1 && echo environ || echo noenviron'], d, timeout=60); "
                 "print('SANDBOX', r.returncode, ' '.join(r.stdout.split()), r.stderr.strip()[-200:])")
SANDBOX_MODES = (("plain", {}), ("user namespace", {"GRC_BENCH_BWRAP_USERNS": "1"}),
                 ("user namespace, Colab's /proc", {"GRC_BENCH_BWRAP_USERNS": "1", "GRC_BENCH_BWRAP_PROC": "host"}))
SANDBOX_MODE = None

def pins():
    global SANDBOX_MODE
    c, out, _ = gb("env", "step 7", name="07_env")
    env = json.loads(host(PKG / "reports/environment.json").read_text())
    py = AGENT / ".venv/bin/python"
    _, cat, _ = run_in(f"cd {PKG} && PYTHONPATH=src {py} -c 'from grcbench.tier_f.ladder import pinned_catalog_hash, "
                       "catalog_matches_pin; print(\"CATALOG\", pinned_catalog_hash(), catalog_matches_pin())'", "step 7",
                       name="07_catalog")
    parts = next((l.split()[1:] for l in cat if l.startswith("CATALOG")), [])
    sandbox = {}
    for mode, extra in SANDBOX_MODES:
        _, sb, _ = run_in(f"cd {PKG} && PYTHONPATH=src {py} -c \"{SANDBOX_PROBE}\"", "step 7",
                          name="07_sandbox_" + re.sub(r"[^a-z0-9]+", "_", mode.lower()).strip("_"), env=extra)
        line = next((l for l in sb if l.startswith("SANDBOX")), "SANDBOX none")
        sandbox[mode] = line
        # ran; the root is read-only; no network; the notebook's own process cannot be inspected
        if line.startswith("SANDBOX 0 ro nonet noenviron"):
            INNER_ENV.update(extra)
            SANDBOX_MODE = mode
            break
    shutil.copy(host(PKG / "reports/environment.json"), OUT / "environment.json")
    pins_ok, catalog_ok = c == 0 and any("pins ok" in l for l in out), parts[1:] == ["True"]
    log("step 7", f"pins: {out[-1] if out else ''}; catalog matches pin: {catalog_ok}; sandbox: {SANDBOX_MODE} {sandbox}")
    bad = [name for name, ok in (("version pins", pins_ok), ("block catalog hash", catalog_ok),
                                 ("sandbox", SANDBOX_MODE is not None)) if not ok]
    return {"pins_ok": pins_ok, "pin_message": out[-1] if out else "",
            "gnuradio_version": env.get("gnuradio_version"), "grc_agent_commit": env.get("grc_agent_commit"),
            "python": env.get("python"), "packages": env.get("packages"),
            "catalog_hash": parts[0] if parts else None, "catalog_matches_pin": catalog_ok,
            "sandbox": sandbox, "sandbox_ok": SANDBOX_MODE is not None, "sandbox_mode": SANDBOX_MODE,
            "sandbox_env": {k: v for k, v in INNER_ENV.items() if k.startswith("GRC_BENCH_BWRAP")},
            **({"failed": " / ".join(bad) + " not as required"} if bad else {})}

step("7_pins", "./gb env, pins, catalog and sandbox", pins)

SANDBOX_TESTS = ("tests/test_sandbox_masks.py::test_the_outer_sandbox_hides_the_users_files",
                 "tests/test_sandbox_masks.py::test_agent_shell_commands_see_only_the_project_folder",
                 "tests/test_sandbox_masks.py::test_flowgraph_runs_see_only_their_own_paths",
                 "tests/test_common.py::test_sandbox_blocks_network_and_writes_outside",
                 "tests/test_common.py::test_sandbox_timeout",
                 "tests/test_common.py::test_a_sandbox_inside_the_sandbox_gets_the_same_namespace_settings",
                 "tests/test_tier_a_runner.py::test_agent_shell_cannot_write_outside_the_project_or_reach_the_network",
                 "tests/test_tier_a_runner.py::test_flowgraph_run_time_cap")

def episode_sandbox():
    if SANDBOX_MODE is None:
        return {"failed": "not run: no sandbox setting passed step 7"}
    if RESULT["steps"].get("5_grc_agent", {}).get("knowledge_dbs") is not True:
        return {"failed": "not run: GRC-Agent did not install (step 5)"}
    c1, replays, s1 = gb("tier-a smoke --no-probe --no-report", "step 7b", name="07b_golden_replays", show=40)
    c2, tests, s2 = gb("test " + " ".join(SANDBOX_TESTS), "step 7b", name="07b_sandbox_tests", show=40)
    summary = next((l.strip() for l in reversed(tests) if re.search(r"\d+ (passed|failed|error)", l)), "")
    log("step 7b", f"golden replays: exit {c1}; sandbox tests: {summary or 'no summary'}")
    bad = [name for name, code in (("golden replays", c1), ("sandbox tests", c2)) if code]
    return {"sandbox_mode": SANDBOX_MODE, "replays_exit": c1, "replays_seconds": s1, "replays_tail": replays[-12:],
            "tests": list(SANDBOX_TESTS), "tests_exit": c2, "tests_seconds": s2, "tests_summary": summary,
            **({"failed": " and ".join(bad) + " did not pass (logs 07b_*)"} if bad else {})}

step("7b_episode_sandbox", "the sandbox inside a Tier-A episode", episode_sandbox)

E_NAME = "colab_" + re.sub(r"[^a-z0-9]+", "_", (GPU["gpu"] or "cpu").lower()).strip("_")
E2E = WORK / "colab_e2e"

def e2e():
    if not MODEL_STEPS:
        return {"skipped": "MODEL_STEPS is off"}
    tiers = [t.strip().upper() for t in TIERS.split(",") if t.strip()]
    agent_ok = RESULT["steps"].get("5_grc_agent", {}).get("knowledge_dbs") is True
    sandbox_ok = RESULT["steps"].get("7_pins", {}).get("sandbox_ok") is True
    skipped = [t for t in tiers if t in "FA" and not (agent_ok and sandbox_ok)]
    if skipped:
        log("step 8", f"Tier {', '.join(skipped)} not run: GRC-Agent (step 5) or the sandbox (step 7) did not work here")
    tiers = [t for t in tiers if t not in skipped]
    setup = "Ubuntu 26.04 container" if UBUNTU else "conda-forge GNU Radio"
    label = ("Google Colab (" + GPU["gpu"] + ", driver " + GPU["driver"] + "; " + setup + "; sandbox: "
             + str(SANDBOX_MODE) + ")")
    args = (f"e2e --out {E2E} --limit-k {LIMIT_K} --limit-f {LIMIT_F} --limit-a {LIMIT_A} "
            f"--rubrics colab/dev_placeholder.jsonl --report {OUT}/e2e_colab.md --e-name {E_NAME} "
            f"--machine-label {shlex.quote(label)}")
    if SERVER_URL:                                       # the other notebook's server; Tier E on its model file
        args += f" --base-url {shlex.quote(SERVER_URL)} --model {shlex.quote(SERVER_MODEL)} --gguf {shlex.quote(str(MODEL_FILE))}"
    served = [t for t in tiers if t != "E"]
    # Tier E measures with no other llama.cpp process running: the other notebook's server stops first
    if SERVER_URL and "E" in tiers and not STOP_SERVER:
        log("step 8", "Tier E not run: the model server stays up (no STOP_SERVER), and Tier E measures alone")
        tiers, skipped = served, skipped + ["E"]
    if SERVER_URL and "E" in tiers:
        c1, s1 = 0, 0.0
        if served:
            c1, _, s1 = gb(f"{args} --only {','.join(served)}", "step 8", name="08_e2e", show=80)
        STOP_SERVER()
        log("step 8", "the model server is stopped: Tier E runs alone on the GPU")
        c2, _, s2 = gb(f"{args} --only E", "step 8", name="08_e2e_tier_e", show=40)
        c, secs = c1 or c2, round(s1 + s2, 1)
    else:
        c, _, secs = gb(f"{args} --only {','.join(tiers)}", "step 8", name="08_e2e", show=80)
    s = json.loads(host(E2E / "summary.json").read_text())
    for k, v in s.get("tiers", {}).items():
        log("step 8", f"Tier {k}: " + json.dumps({x: v.get(x) for x in ("items", "answers", "n_rules", "n_judge", "tasks",
                                                                     "samples", "episodes", "passed", "funnel",
                                                                     "generation_tps", "error", "seconds") if x in v}))
    return {"exit": c, "seconds": secs, "setup": setup, "tiers_run": tiers, "tiers_not_run": skipped,
            "tiers": s.get("tiers"), "server": s.get("server"),
            **({"failed": f"./gb e2e exited {c} (log 08_e2e.log)"} if c else {})}

step("8_e2e", "./gb e2e on the dev split", e2e)

def smi():
    q = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def speeds():
    if not MODEL_STEPS:
        return {"skipped": "MODEL_STEPS is off"}
    if "E" not in RESULT["steps"].get("8_e2e", {}).get("tiers_run", []):
        return {"skipped": "Tier E did not run in step 8"}
    model, llama_dir = MODEL_FILE, CACHE / f"llama.cpp/{LLAMA_BUILD}-cuda"
    before = smi()
    c, _, secs = gb(f"tier-e run --model {model} --hardware H3 --llama-dir {llama_dir} --contexts 4096 --purpose plumbing "
                    f"--name {E_NAME}_repeat --machine-label {shlex.quote('Google Colab (' + GPU['gpu'] + ')')} "
                    "--note 'Colab pipeline test: repeat of the e2e Tier-E run, to check that speeds repeat'",
                    "step 9", name="09_tier_e_repeat")
    after = smi()
    runs = {}
    for name in (E_NAME, E_NAME + "_repeat"):
        p = host(PKG / "reports/tier_e" / f"{name}.json")
        if p.exists():
            rec = json.loads(p.read_text())
            shutil.copy(p, OUT / p.name)
            ctx = next((x for x in rec.get("contexts", []) if x.get("context") == 4096), {})
            runs[name] = {k: {f: (ctx.get(k) or {}).get(f) for f in ("median_ts", "min_ts", "max_ts", "samples_ts",
                                                                     "llama_bench_stddev_ts")}
                          for k in ("generation", "prompt") if ctx.get(k)}
    gen = [r["generation"]["median_ts"] for r in runs.values() if r.get("generation")]
    answers = host(E2E / "tier_k/answers.jsonl")
    k_rows = [json.loads(l) for l in open(answers)] if answers.exists() else []
    k_tps = [round(r["usage"]["completion_tokens"] / r["seconds"], 1) for r in k_rows
             if r.get("usage") and r.get("seconds") and r["usage"].get("completion_tokens")]
    out = {"exit": c, "runs": runs, "smi_before": before, "smi_after": after, "tier_k_answer_tps": k_tps}
    if len(gen) == 2:
        out["generation_tps_two_runs"] = gen
        out["difference_pct"] = round(100 * abs(gen[0] - gen[1]) / max(gen), 2)
    log("step 9", f"generation tokens/s: {gen}; Tier-K answers: {k_tps}")
    log("step 9", f"GPU before: {before}")
    log("step 9", f"GPU after:  {after}")
    if c or len(gen) != 2:
        out["failed"] = f"the repeat Tier-E run exited {c}; {len(gen)} of 2 speeds recorded (log 09_tier_e_repeat.log)"
    return out

step("9_speeds", "speeds: Tier E twice, GPU clocks", speeds)

JUDGE = r"""
import json, sys
from pathlib import Path
from grcbench.tier_a import smoke
from grcbench.tier_k import judge_run
out = Path(sys.argv[1])
ctx = smoke.server_context(smoke.native_context(smoke.SMOKE_MODEL))
with smoke.LocalServer(out / "judge-server.log", ctx=ctx) as url:
    ep = judge_run.Endpoint(base_url=url, model=smoke.SERVER_ALIAS)
    s = judge_run.judge(out, primary=ep, fallback=ep, timeout=1800)
print(json.dumps({k: s.get(k) for k in ("judge", "n_judged", "n_judge_failed", "n_judge_request_failed", "report_counts")}))
"""

def judge():
    if not MODEL_STEPS or not JUDGE_IN_COLAB or SERVER_URL:
        return {"skipped": "MODEL_STEPS is off" if not MODEL_STEPS else "JUDGE_IN_COLAB is off" if not JUDGE_IN_COLAB
                else "the judge plumbing starts the smoke model's own server; not with another notebook's server"}
    (OUT / "judge_plumbing.py").write_text(JUDGE)           # /content is visible inside the container too
    c, out, secs = run_in(f"cd {PKG} && PYTHONPATH={PKG}/src {AGENT}/.venv/bin/python {OUT}/judge_plumbing.py {E2E}/tier_k",
                          "step 10", name="10_judge")
    last = next((l for l in reversed(out) if l.startswith("{")), None)
    return {"exit": c, "seconds": secs, "summary": json.loads(last) if last else None,
            **({"failed": f"the judge runner exited {c} (log 10_judge.log)"} if c or not last else {})}

step("10_judge", "the judge runner (plumbing)", judge)

def results():
    for src, dst in ((host(E2E), "e2e_run"), (host(PKG / "runs"), "runs")):
        if Path(src).exists():
            shutil.copytree(src, OUT / dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.gguf", "home"))
    RESULT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RESULT["setup_used"] = "ubuntu-26.04" if UBUNTU else "conda-forge"
    save()
    z = OUT.parent / f"results_{OUT.name}.zip"
    shutil.make_archive(str(z.with_suffix("")), "zip", OUT)
    if DRIVE_OUT is not None:
        shutil.copy(z, DRIVE_OUT / z.name)
    else:                                               # no Drive: the browser downloads it
        from google.colab import files
        files.download(str(z))
    return {"zip": str(DRIVE_OUT / z.name if DRIVE_OUT is not None else z)}

step("11_results", "results", results)
print("\n" + "=" * 100)
for k, v in RESULT["steps"].items():
    print(f"{k:14s} {v.get('status', '?'):7s} {v.get('seconds', '')!s:>8} s  {v.get('error', '')[:120]}")
print("=" * 100)
print("GNU Radio setup used:", RESULT.get("setup_used"))
print("Download this file and put it in your Downloads folder:", RESULT["steps"].get("11_results", {}).get("zip"))
