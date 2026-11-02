# GRC-Bench in Colab: pipeline test

`grcbench_colab_test.ipynb` runs GRC-Bench's own command, `./gb`, on a Google Colab GPU, on a few
dev-split items. It answers one question: does the whole pipeline work there? It does not evaluate a model.

## The GNU Radio setup

GRC-Bench pins Ubuntu 26.04's GNU Radio package (3.10.12.0-6: upstream `v3.10.12.0-80-g5b6b9ac46b` with
Debian's patches). Colab's own Ubuntu (24.04 in November 2026) cannot install it, and conda-forge's GNU Radio
3.10.12.0 is a different build that GRC-Agent `6c98128` does not run on. So the notebook offers two setups
(`GR_SETUP`):

- **`ubuntu-26.04`** (default): Ubuntu 26.04.1's base system (about 30 MB, checked by SHA-256) is unpacked in
  `/opt/u2604`, Ubuntu's own `gnuradio=3.10.12.0-6` and the other packages of `docker/Dockerfile` are installed
  into it with apt, and every `./gb` command runs inside it through `u2604.py`. That script uses Linux user and
  mount namespaces and enters the root filesystem with `pivot_root`, as bubblewrap does; there is no Docker.
  Colab's `/proc`, its NVIDIA driver files and `/content` are made visible inside. The block catalog hash
  matches the pin, and Tier F and A run there.
- **`conda-forge`**: GNU Radio 3.10.12.0 from conda-forge on Colab itself: Tier K and E only. The notebook also
  falls back to it when the Ubuntu container cannot be made, and says so in `feasibility.json`.

Everything the notebook installs lives in the Colab session and goes with it; only `My Drive/grc-bench-colab/`
(results and the llama.cpp build) stays.

## The sandbox on Colab

A Colab runtime is a container whose root may create namespaces but may not mount a fresh `/proc` or configure
network interfaces. GRC-Bench's sandbox therefore runs there with two settings (step 7 picks them only after
checking them):

- `GRC_BENCH_BWRAP_USERNS=1`: bubblewrap makes a user namespace of its own, in which it may bring up the
  loopback of the sandbox's empty network namespace;
- `GRC_BENCH_BWRAP_PROC=host`: the sandbox binds Colab's `/proc` instead of a fresh one. Code inside sees the
  numbers and command lines of the processes outside, but cannot inspect them (`/proc/<pid>/root`, `environ`:
  refused from a child user namespace). The setting is refused without the first one. Tier A's episode process,
  which runs GRC-Agent's harness and starts sandboxes itself, then shares Colab's PID namespace (bubblewrap inside
  must find its own processes under that `/proc`); the sandboxes it starts for the agent's shell commands and
  flowgraph runs have their own.

Checked on Colab (3 Nov 2026, a T4 runtime) with a probe: loopback works, the internet is unreachable, and
reading a hidden file or writing a read-only folder through `/proc/1/root`, and reading another process's
environment, are all refused. The run's machine label names the sandbox setting.

## What you need

- `grc-bench-colab.zip`, made by `./gb colab-package` (it goes to `~/grc-bench-work/colab/`). It holds `./gb`,
  the code, the dev split, placeholder rubrics for the dev questions, `u2604.py` and this notebook. It never holds
  a test item: the build refuses any file that contains one.
- A Google account with Colab (the free T4 is enough) and about 10 GB free in Google Drive.
- Optional, for a T4: `llama-b10333-cuda-sm75-cu12.4.tar.gz` (made on the build machine with
  `CUDA_ARCH=75 CUDA_VERSION=12.4 GCC_VERSION=12 SYSROOT=2.28 ./gb fetch --cuda`), which saves compiling llama.cpp.

## How to run it

1. Put `grc-bench-colab.zip` in **My Drive/grc-bench-colab/** and, if you have it, the llama.cpp archive in
   **My Drive/grc-bench-colab/cache/**.
2. Open the notebook in Colab (*File → Upload notebook*; it is also inside the zip, under `colab/`).
3. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*, and allow Drive access (choose the Google
   account the notebook runs under and tick every box). Without Drive: upload the zip (and the llama.cpp
   archive) to the Files pane before *Run all*; the results zip is then offered as a download.
4. About 45 minutes with the llama.cpp archive in Drive; 1.5–2 hours without (compiling llama.cpp b10333 on
   Colab's 2 CPUs). A build made here is kept in `My Drive/grc-bench-colab/cache/` and reused next time.
5. Download `My Drive/grc-bench-colab/results/<run>/results_<run>.zip`.

## What it checks

| Step | Check |
|---|---|
| 1 | GPU, driver and its CUDA version, OS, Python, CPUs, memory, disk, capabilities |
| 2 | Drive, and the package |
| 3 | Linux namespaces, a bind of Colab's `/proc`, and bubblewrap with its own user namespace |
| 4 | GNU Radio: the Ubuntu 26.04 container with gnuradio 3.10.12.0-6 (or conda-forge's, as the fallback) |
| 5 | GRC-Agent at `6c98128`, its knowledge databases, the tokenizer |
| 6 | llama.cpp b10333 with CUDA, compiled for this GPU or taken from Drive |
| 7 | `./gb env` (version pins), the GNU Radio block catalog hash, GRC-Bench's sandbox (the setting it uses here) |
| 8 | `./gb e2e`: Tier K (scored with placeholder rubrics), Tier F, Tier A, Tier E on a few dev items |
| 9 | speeds: Tier E twice, the GPU's clocks and temperature, Tier-K tokens per second |
| 10 | the judge runner, with the same local model as a stand-in judge (plumbing only) |
| 11 | everything into one results folder and zip |

Each step is recorded in `feasibility.json` as ok or failed, with its time and details; a step whose command
fails is recorded as failed. A failed step does not stop the later ones; a missing package stops the run.

## Changing the notebook

`make_notebook.py` writes the notebook (`python make_notebook.py`); edit it there, then rebuild the package with
`./gb colab-package`.

## Placeholder rubrics

The team's dev rubrics are not written yet. `colab/dev_placeholder.jsonl` gives each dev question one required
point: the training release's reference answer, copied word for word (`./gb placeholder-rubrics`). With it, an
answer goes through the whole route (rules, judge request, verdict, score), but the score says nothing about the
model. Once the team's rubrics are approved in `items/tier_k/dev/rubrics.jsonl`, score the same answers without
the file: `./gb tier-k score DIR`.
