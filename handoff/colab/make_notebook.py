"""Writes grcbench_colab_test.ipynb next to this file: the Colab pipeline test of GRC-Bench.

    python make_notebook.py [OUT.ipynb]

The notebook runs `./gb` on a Colab GPU in one of two setups (GR_SETUP):
  ubuntu-26.04  (default) an Ubuntu 26.04 root filesystem with Ubuntu's own gnuradio 3.10.12.0-6, entered with
                u2604.py (Linux namespaces, no Docker): the pinned GNU Radio, so Tier F and A can run;
  conda-forge   GNU Radio 3.10.12.0 from conda-forge on Colab itself: GRC-Agent 6c98128 does not run on it, so
                Tier K and E only. It is also the fallback when the Ubuntu container cannot be made.
In both, ./gb sees the same paths (/root/grc-bench-colab, /root/grc-bench-work, /root/grc-bench-cache,
/root/grc-agent). Every code cell must compile (checked here).
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("grcbench_colab_test.ipynb")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    src = text.strip("\n")
    compile(src, "<cell>", "exec")                      # every code cell is valid Python
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})


md("""
# GRC-Bench in Colab: pipeline test on the dev split

This notebook checks whether GRC-Bench's own command, `./gb`, runs end to end on a Colab GPU. It is a
**feasibility test, not an evaluation**: dev-split items only, the pinned smoke model (Qwen3-4B Q4_K_M),
placeholder rubrics, and a few items per tier.

GRC-Bench pins Ubuntu 26.04's GNU Radio package (3.10.12.0-6), which Colab's Ubuntu 22.04 cannot install. So by
default (`GR_SETUP = "ubuntu-26.04"`) the notebook downloads Ubuntu 26.04's base system (about 30 MB), installs
Ubuntu's own GNU Radio into it and runs `./gb` inside it, the way a container would (`colab/u2604.py`: Linux
namespaces, no Docker). If Colab does not allow that, it falls back to conda-forge's GNU Radio on Colab itself,
where Tier K and E run but Tier F and A cannot (GRC-Agent does not run on that GNU Radio).

| Step | What it checks |
|---|---|
| 1 | the machine: GPU, driver, CUDA version, OS, Python, CPUs, memory, disk |
| 2 | Google Drive, and the GRC-Bench package (`grc-bench-colab.zip`) from your Drive |
| 3 | Linux namespaces and bubblewrap on Colab itself |
| 4 | **GNU Radio**: the Ubuntu 26.04 container with gnuradio 3.10.12.0-6 (or conda-forge's 3.10.12.0) |
| 5 | **GRC-Agent** at the pinned commit `6c98128`, its knowledge databases and the tokenizer |
| 6 | **llama.cpp b10333** (the pinned build): no prebuilt Linux CUDA download exists, so it is compiled once and kept in your Drive, unless a build is already there |
| 7 | `./gb env` (the version pins), the GNU Radio block catalog hash, and GRC-Bench's sandbox |
| 8 | `./gb e2e`: Tier K (answers scored with placeholder rubrics), Tier F, Tier A, Tier E on a few dev items |
| 9 | speeds: Tier E twice, the GPU's clocks and temperature, Tier-K tokens per second |
| 10 | the judge runner: the same local model grades the Tier-K answers (plumbing only) |
| 11 | results: one folder and one zip in your Drive |

**How to run**
1. Put `grc-bench-colab.zip` in **My Drive/grc-bench-colab/** (no need to unzip it). Optional: put
   `llama-b10333-cuda-sm75-cu12.4.tar.gz` in **My Drive/grc-bench-colab/cache/** to skip compiling llama.cpp on a T4.
2. *Runtime → Change runtime type → **T4 GPU***.
3. *Runtime → **Run all***, and allow Google Drive access when asked.
4. Keep the tab open. **First run: about 45 minutes with the llama.cpp archive in Drive, 1.5–2 hours without.**

**What comes back:** `My Drive/grc-bench-colab/results/<run>/results_<run>.zip`, which holds `feasibility.json` (one
entry per step: ok or failed, with times and details), the logs, the e2e report and the run folders. A step that
fails is recorded and the notebook goes on, so one run gives the whole picture.

**What leaves this machine:** nothing is uploaded anywhere except your own Drive. Only dev-split items are in the
package; the test split is never in it. Everything else lives in this Colab session and goes with it.
""")

code('''
#@title Settings
DRIVE_FOLDER = "grc-bench-colab"  #@param {type:"string"}
PACKAGE = "grc-bench-colab.zip"  #@param {type:"string"}
GR_SETUP = "ubuntu-26.04"  #@param ["ubuntu-26.04", "conda-forge"]
TIERS = "K,F,A,E"  #@param {type:"string"}
LIMIT_K = 5  #@param {type:"integer"}
LIMIT_F = 2  #@param {type:"integer"}
LIMIT_A = 1  #@param {type:"integer"}
JUDGE_IN_COLAB = True  #@param {type:"boolean"}
REUSE_LLAMA_BUILD = True  #@param {type:"boolean"}
print(f"GNU Radio setup {GR_SETUP}; tiers {TIERS}: {LIMIT_K} Tier-K questions, {LIMIT_F} Tier-F tasks, "
      f"{LIMIT_A} Tier-A tasks; judge in Colab: {JUDGE_IN_COLAB}; package My Drive/{DRIVE_FOLDER}/{PACKAGE}")
''')

md("""
## Helpers
Every command's full output goes to a log file in the results folder; the cell shows the important lines.
`run_in` runs a command where `./gb` lives: inside the Ubuntu 26.04 container, or on Colab itself.
`feasibility.json` is written after every step.
""")

code(r'''
import json, os, re, shlex, shutil, subprocess, sys, tarfile, time, zipfile
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
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
RESULT = {"notebook": "grcbench_colab_test 2", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "settings": {"gr_setup": GR_SETUP, "tiers": TIERS, "limit_k": LIMIT_K, "limit_f": LIMIT_F,
                       "limit_a": LIMIT_A, "judge_in_colab": JUDGE_IN_COLAB}, "steps": {}}
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
    """Run one step; record ok / failed (with the error) and its time. Later steps still run."""
    print("=" * 100 + f"\n{key}: {title}\n" + "=" * 100, flush=True)
    t0 = time.time()
    try:
        detail = fn() or {}
        detail = {("result_" + k if k in ("title", "status", "seconds") else k): v for k, v in detail.items()}
        record(key, title=title, status="ok", seconds=round(time.time() - t0, 1), **detail)
    except StopHere:
        record(key, title=title, status="failed", seconds=round(time.time() - t0, 1))
        raise
    except Exception as exc:
        record(key, title=title, status="failed", seconds=round(time.time() - t0, 1), error=f"{type(exc).__name__}: {exc}")
        log(key, f"FAILED: {type(exc).__name__}: {exc}  (recorded; the notebook goes on)")

print("helpers ready; results go to", OUT)
''')

md("## Step 1: the machine")
code(r'''
def machine():
    if "google.colab" not in sys.modules:
        raise StopHere("This notebook is written for Google Colab.")
    if not shutil.which("nvidia-smi"):
        raise StopHere("No GPU: Runtime > Change runtime type > T4 GPU > Save, then Runtime > Run all.")
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip().split(", ")
    head = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)     # newer drivers: "CUDA UMD Version"
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
''')

md("## Step 2: Google Drive and the GRC-Bench package")
code(r'''
def package():
    global DRIVE_OUT
    from google.colab import drive
    drive.mount("/content/drive")
    root = Path("/content/drive/MyDrive") / DRIVE_FOLDER
    zpath = root / PACKAGE
    if not zpath.exists():
        raise StopHere(f"{zpath} not found: put {PACKAGE} in My Drive/{DRIVE_FOLDER}/ and Run all again.")
    gpu = re.sub(r"[^A-Za-z0-9]+", "-", GPU["gpu"]).strip("-")
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
    return {"package": str(zpath), "package_bytes": zpath.stat().st_size, "version": version, "drive_results": str(DRIVE_OUT)}

step("2_package", "Google Drive and the package", package)
''')

md("""
## Step 3: Linux namespaces and bubblewrap on Colab itself
The Ubuntu container (step 4) needs Linux user namespaces; GRC-Bench's sandbox (bubblewrap) needs them too.
""")
code(r'''
def namespaces():
    probes = {}
    for label, cmd in (("user namespace", "unshare --user --map-root-user true"),
                       ("user + mount namespaces", "unshare --user --map-root-user --mount true"),
                       ("user + mount + pid namespaces", "unshare --user --map-root-user --mount --pid --fork --mount-proc true")):
        c, out, _ = sh(cmd, "step 3", name="03_" + label.replace(" ", "_").replace("+", ""))
        probes[label] = {"exit": c, "output": out[-2:]}
    sh("apt-get -qq update >/dev/null && apt-get -qq install -y bubblewrap >/dev/null && bwrap --version", "step 3",
       name="03_bubblewrap_install")
    base = "bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid --unshare-net --die-with-parent --new-session"
    for label, extra in (("bwrap plain", ""), ("bwrap with --unshare-user", " --unshare-user")):
        c, out, _ = sh(f"{base}{extra} -- true", "step 3", name="03_" + label.replace(" ", "_").replace("--", ""))
        probes[label] = {"exit": c, "output": out[-2:]}
    for k, v in probes.items():
        log("step 3", f"{k:32s} {'works' if v['exit'] == 0 else 'fails: ' + ' '.join(v['output'])[:120]}")
    return {"probes": probes}

step("3_namespaces", "Linux namespaces and bubblewrap", namespaces)
''')

md("""
## Step 4: GNU Radio
**ubuntu-26.04:** Ubuntu 26.04.1's base system (checked by SHA-256) in `/opt/u2604`, then `apt-get install
gnuradio=3.10.12.0-6` and the other packages of `docker/Dockerfile` inside it. **conda-forge** (or the fallback):
GNU Radio 3.10.12.0 from conda-forge on Colab itself.
""")
code(r'''
def ubuntu_container():
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
''')

md("## Step 5: GRC-Agent at `6c98128`")
code(r'''
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
    return {"head": head, "install_seconds": secs, "imports": imp[-1:],
            "knowledge_dbs": c_db == 0 and (vec / "catalog_lexical.db").exists() and (vec / "docs_lexical.db").exists(),
            "knowledge_dbs_error": None if c_db == 0 else db_out[-1:], "tokenizer": c_tok == 0}

step("5_grc_agent", "GRC-Agent 6c98128", grc_agent)
''')

md("""
## Step 6: llama.cpp b10333 with CUDA
`./gb fetch` downloads the pinned CPU build, the smoke model and the embedding model (each checked by SHA-256).
llama.cpp publishes no Linux CUDA build of b10333, so `./gb fetch --cuda` compiles it for this GPU (the first
time: up to an hour or more on Colab's 2 CPUs) and the result is kept in your Drive for the next run. A build
placed in `My Drive/grc-bench-colab/cache/` beforehand (`llama-b10333-cuda-sm75-cu12.4.tar.gz` for a T4) is used
instead.
""")
code(r'''
def llama():
    c, _, fetch_s = gb("fetch", "step 6", name="06_fetch")
    if c:
        raise RuntimeError("./gb fetch failed (log 06_fetch.log)")
    cc = GPU["compute_cap"].replace(".", "")
    drv = tuple(int(x) for x in (GPU["driver_cuda"] or "12.4").split("."))
    cuda = "12.9" if drv >= (12, 9) else f"{drv[0]}.{drv[1]}"
    out_dir = host(CACHE / "llama.cpp" / "b10333-cuda")
    cache = Path("/content/drive/MyDrive") / DRIVE_FOLDER / "cache"
    saved = cache / f"llama-b10333-cuda-sm{cc}-cu{cuda}.tar.gz"
    # a build kept earlier (or put there by hand) for this GPU, made with a CUDA version this driver supports
    cu_of = lambda p: tuple(int(x) for x in re.search(r"-cu(\d+)\.(\d+)\.tar\.gz$", p.name).groups())
    usable = [p for p in cache.glob(f"llama-b10333-cuda-sm{cc}-cu*.tar.gz") if cu_of(p) <= drv[:2]]
    if usable:
        saved = max(usable, key=cu_of)
    how, build_s = "built", None
    if REUSE_LLAMA_BUILD and saved.exists():
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(saved) as t:
            t.extractall(out_dir.parent)
        how = f"reused from Drive ({saved.name})"
    else:
        gcc = "13" if drv >= (12, 6) else "12"           # conda-forge's nvcc below 12.6 accepts GCC 12 at most
        c, _, build_s = gb("fetch --cuda", "step 6", name="06_build_cuda", show=40,
                           env={"CUDA_ARCH": cc, "CUDA_VERSION": cuda, "GCC_VERSION": gcc, "SYSROOT": "2.28",
                                "JOBS": str(os.cpu_count() or 2)})
        if c:
            raise RuntimeError("the CUDA build failed (log 06_build_cuda.log)")
        saved.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(saved, "w:gz") as t:
            t.add(out_dir, arcname=out_dir.name)
    d = CACHE / "llama.cpp" / "b10333-cuda"
    _, version, _ = run_in(f"LD_LIBRARY_PATH={d}:$LD_LIBRARY_PATH {d}/llama-server --version", "step 6", name="06_version")
    log("step 6", f"llama.cpp {how}: {version[-2:]}")
    return {"fetch_seconds": fetch_s, "cuda_arch": cc, "cuda_version": cuda, "how": how, "build_seconds": build_s,
            "version": version[-2:], "build_cache": str(saved)}

step("6_llama_cpp", "llama.cpp b10333 (CUDA)", llama)
''')

md("## Step 7: `./gb env`, the pins, the block catalog and GRC-Bench's sandbox")
code(r'''
SANDBOX_PROBE = ("from pathlib import Path; import tempfile; from grcbench.common import sandbox; "
                 "w = Path('/root/grc-bench-work'); w.mkdir(parents=True, exist_ok=True); d = Path(tempfile.mkdtemp(dir=w)); "
                 "r = sandbox.run(['bash', '-c', 'echo hi > x; echo y > /etc/nope 2>/dev/null || echo ro; "
                 "(exec 3<>/dev/tcp/1.1.1.1/53) 2>/dev/null && echo net || echo nonet'], d, timeout=60); "
                 "print('SANDBOX', r.returncode, ' '.join(r.stdout.split()), r.stderr.strip()[-200:])")

def pins():
    c, out, _ = gb("env", "step 7", name="07_env")
    env = json.loads(host(PKG / "reports/environment.json").read_text())
    py = AGENT / ".venv/bin/python"
    _, cat, _ = run_in(f"cd {PKG} && PYTHONPATH=src {py} -c 'from grcbench.tier_f.ladder import pinned_catalog_hash, "
                       "catalog_matches_pin; print(\"CATALOG\", pinned_catalog_hash(), catalog_matches_pin())'", "step 7",
                       name="07_catalog")
    parts = next((l.split()[1:] for l in cat if l.startswith("CATALOG")), [])
    sandbox = {}
    for mode, extra in (("plain", {}), ("user namespace", {"GRC_BENCH_BWRAP_USERNS": "1"})):
        _, sb, _ = run_in(f"cd {PKG} && PYTHONPATH=src {py} -c \"{SANDBOX_PROBE}\"", "step 7",
                          name="07_sandbox_" + mode.replace(" ", "_"), env=extra)
        line = next((l for l in sb if l.startswith("SANDBOX")), "SANDBOX none")
        sandbox[mode] = line
        if line.startswith("SANDBOX 0 ro nonet"):      # ran, the root is read-only, no network
            INNER_ENV.update(extra)
            break
    sandbox_ok = any(v.startswith("SANDBOX 0 ro nonet") for v in sandbox.values())
    shutil.copy(host(PKG / "reports/environment.json"), OUT / "environment.json")
    log("step 7", f"pins: {out[-1] if out else ''}; catalog matches pin: {parts[1:] == ['True']}; sandbox: {sandbox}")
    return {"pins_ok": c == 0 and any("pins ok" in l for l in out), "pin_message": out[-1] if out else "",
            "gnuradio_version": env.get("gnuradio_version"), "grc_agent_commit": env.get("grc_agent_commit"),
            "python": env.get("python"), "packages": env.get("packages"),
            "catalog_hash": parts[0] if parts else None, "catalog_matches_pin": parts[1:] == ["True"],
            "sandbox": sandbox, "sandbox_ok": sandbox_ok}

step("7_pins", "./gb env, pins, catalog and sandbox", pins)
''')

md("""
## Step 8: `./gb e2e` on a few dev items
One llama-server (the smoke model, every layer on the GPU) serves Tier K, F and A in turn; then Tier E runs
llama-bench. Tier-K answers are scored with the **placeholder rubrics** (the training release's reference answer as
the only point): that tests the routing, not the model. Tier F and A run only where GRC-Agent (step 5) and the
sandbox (step 7) work.
""")
code(r'''
E_NAME = "colab_" + re.sub(r"[^a-z0-9]+", "_", GPU["gpu"].lower()).strip("_")
E2E = WORK / "colab_e2e"

def e2e():
    tiers = [t.strip().upper() for t in TIERS.split(",") if t.strip()]
    agent_ok = RESULT["steps"].get("5_grc_agent", {}).get("knowledge_dbs") is True
    sandbox_ok = RESULT["steps"].get("7_pins", {}).get("sandbox_ok") is True
    skipped = [t for t in tiers if t in "FA" and not (agent_ok and sandbox_ok)]
    if skipped:
        log("step 8", f"Tier {', '.join(skipped)} not run: GRC-Agent (step 5) or the sandbox (step 7) did not work here")
    tiers = [t for t in tiers if t not in skipped]
    setup = "Ubuntu 26.04 container" if UBUNTU else "conda-forge GNU Radio"
    label = "Google Colab (" + GPU["gpu"] + ", driver " + GPU["driver"] + "; " + setup + ")"
    args = (f"e2e --out {E2E} --only {','.join(tiers)} --limit-k {LIMIT_K} --limit-f {LIMIT_F} --limit-a {LIMIT_A} "
            f"--rubrics colab/dev_placeholder.jsonl --report {OUT}/e2e_colab.md --e-name {E_NAME} "
            f"--machine-label {shlex.quote(label)}")
    c, _, secs = gb(args, "step 8", name="08_e2e", show=80)
    s = json.loads(host(E2E / "summary.json").read_text())
    for k, v in s.get("tiers", {}).items():
        log("step 8", f"Tier {k}: " + json.dumps({x: v.get(x) for x in ("items", "answers", "n_rules", "n_judge", "tasks",
                                                                     "samples", "episodes", "passed", "funnel",
                                                                     "generation_tps", "error", "seconds") if x in v}))
    return {"exit": c, "seconds": secs, "setup": setup, "tiers_run": tiers, "tiers_not_run": skipped,
            "tiers": s.get("tiers"), "server": s.get("server")}

step("8_e2e", "./gb e2e on the dev split", e2e)
''')

md("""
## Step 9: can speeds be recorded accurately here?
Tier E runs a second time with the same settings. Two runs that agree, a small spread inside each run, and steady
GPU clocks mean the recorded speed is a property of the setup, not of the moment. Colab GPUs are shared and can be
throttled, so this is worth checking.
""")
code(r'''
def smi():
    q = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def speeds():
    model, llama_dir = CACHE / "models/Qwen3-4B-Q4_K_M.gguf", CACHE / "llama.cpp/b10333-cuda"
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
    return out

step("9_speeds", "speeds: Tier E twice, GPU clocks", speeds)
''')

md("""
## Step 10: the judge runner (plumbing)
The real judge is a hosted model (not available yet). Here the smoke model grades its own Tier-K answers through
`./gb tier-k judge`, only to show that the requests, verdicts and scores flow. These scores mean nothing.
""")
code(r'''
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
    if not JUDGE_IN_COLAB:
        return {"skipped": "JUDGE_IN_COLAB is off"}
    (OUT / "judge_plumbing.py").write_text(JUDGE)           # /content is visible inside the container too
    c, out, secs = run_in(f"cd {PKG} && PYTHONPATH={PKG}/src {AGENT}/.venv/bin/python {OUT}/judge_plumbing.py {E2E}/tier_k",
                          "step 10", name="10_judge")
    last = next((l for l in reversed(out) if l.startswith("{")), None)
    return {"exit": c, "seconds": secs, "summary": json.loads(last) if last else None}

step("10_judge", "the judge runner (plumbing)", judge)
''')

md("## Step 11: results")
code(r'''
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
    return {"zip": str((DRIVE_OUT or OUT) / z.name)}

step("11_results", "results", results)
print("\n" + "=" * 100)
for k, v in RESULT["steps"].items():
    print(f"{k:14s} {v.get('status', '?'):7s} {v.get('seconds', '')!s:>8} s  {v.get('error', '')[:120]}")
print("=" * 100)
print("GNU Radio setup used:", RESULT.get("setup_used"))
print("Download this file and put it in your Downloads folder:", RESULT["steps"].get("11_results", {}).get("zip"))
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {OUT} ({len(cells)} cells)")
