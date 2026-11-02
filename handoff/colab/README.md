# GRC-Bench in Colab: pipeline test

`grcbench_colab_test.ipynb` runs GRC-Bench's own command, `./gb`, on a Google Colab GPU, on a few
dev-split items. It answers one question: does the whole pipeline work there? It does not evaluate a model.

## The GNU Radio setup

GRC-Bench pins Ubuntu 26.04's GNU Radio package (3.10.12.0-6: upstream `v3.10.12.0-80-g5b6b9ac46b` with
Debian's patches). Colab runs Ubuntu 22.04, which cannot install it, and conda-forge's GNU Radio 3.10.12.0 is
a different build that GRC-Agent `6c98128` does not run on. So the notebook offers two setups (`GR_SETUP`):

- **`ubuntu-26.04`** (default): Ubuntu 26.04.1's base system (about 30 MB, checked by SHA-256) is unpacked in
  `/opt/u2604`, Ubuntu's own `gnuradio=3.10.12.0-6` and the other packages of `docker/Dockerfile` are installed
  into it with apt, and every `./gb` command runs inside it through `u2604.py`. That script uses Linux user,
  mount and PID namespaces, as bubblewrap does; there is no Docker. Colab's NVIDIA driver files and `/content`
  are made visible inside. The block catalog hash matches the pin, and Tier F and A run there.
- **`conda-forge`**: GNU Radio 3.10.12.0 from conda-forge on Colab itself: Tier K and E only. The notebook also
  falls back to it when the Ubuntu container cannot be made, and says so in `feasibility.json`.

Everything the notebook installs lives in the Colab session and goes with it; only `My Drive/grc-bench-colab/`
(results and the llama.cpp build) stays.

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
3. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*, and allow Drive access.
4. About 45 minutes with the llama.cpp archive in Drive; 1.5–2 hours without (compiling llama.cpp b10333 on
   Colab's 2 CPUs). A build made here is kept in `My Drive/grc-bench-colab/cache/` and reused next time.
5. Download `My Drive/grc-bench-colab/results/<run>/results_<run>.zip`.

## What it checks

| Step | Check |
|---|---|
| 1 | GPU, driver and its CUDA version, OS, Python, CPUs, memory, disk, capabilities |
| 2 | Drive, and the package |
| 3 | Linux namespaces and bubblewrap on Colab itself |
| 4 | GNU Radio: the Ubuntu 26.04 container with gnuradio 3.10.12.0-6 (or conda-forge's, as the fallback) |
| 5 | GRC-Agent at `6c98128`, its knowledge databases, the tokenizer |
| 6 | llama.cpp b10333 with CUDA, compiled for this GPU or taken from Drive |
| 7 | `./gb env` (version pins), the GNU Radio block catalog hash, GRC-Bench's sandbox |
| 8 | `./gb e2e`: Tier K (scored with placeholder rubrics), Tier F, Tier A, Tier E on a few dev items |
| 9 | speeds: Tier E twice, the GPU's clocks and temperature, Tier-K tokens per second |
| 10 | the judge runner, with the same local model as a stand-in judge (plumbing only) |
| 11 | everything into one results folder and zip |

Each step is recorded in `feasibility.json` as ok or failed, with its time and details. A failed step does not
stop the later ones.

## Changing the notebook

`make_notebook.py` writes the notebook (`python make_notebook.py`); edit it there, then rebuild the package with
`./gb colab-package`.

## Placeholder rubrics

The team's dev rubrics are not written yet. `colab/dev_placeholder.jsonl` gives each dev question one required
point: the training release's reference answer, copied word for word (`./gb placeholder-rubrics`). With it, an
answer goes through the whole route (rules, judge request, verdict, score), but the score says nothing about the
model. Once the team's rubrics are approved in `items/tier_k/dev/rubrics.jsonl`, score the same answers without
the file: `./gb tier-k score DIR`.
