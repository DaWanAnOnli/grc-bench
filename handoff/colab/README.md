# GRC-Bench in Colab: pipeline test

`grcbench_colab_test.ipynb` runs GRC-Bench's own command, `./gb`, on a Google Colab GPU, on a few
dev-split items. It answers one question: does the whole pipeline work there? It does not evaluate a model.

## What you need

- `grc-bench-colab.zip`, made by `./gb colab-package` (it goes to `~/grc-bench-work/colab/`). It holds `./gb`,
  the code, the dev split, placeholder rubrics for the dev questions and this notebook. It never holds a
  test item: the build refuses any file that contains one.
- A Google account with Colab (the free T4 is enough) and about 10 GB free in Google Drive.

## How to run it

1. Put `grc-bench-colab.zip` in **My Drive/grc-bench-colab/**.
2. Open the notebook in Colab (*File → Upload notebook*; it is also inside the zip, under `colab/`).
3. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*, and allow Drive access.
4. The first run takes about 1.5–2 hours: compiling llama.cpp b10333 for the GPU is most of it. The build
   is kept in `My Drive/grc-bench-colab/cache/` and reused next time.
5. Download `My Drive/grc-bench-colab/results/<run>/results_<run>.zip`.

## What it checks

| Step | Check |
|---|---|
| 1 | GPU, driver and its CUDA version, OS, Python, CPUs, memory, disk |
| 2 | Drive, and the package |
| 3 | bubblewrap (the sandbox for model-written code): plain, or with its own user namespace |
| 4 | GNU Radio 3.10.12 from conda-forge, linked where GRC-Bench looks for it |
| 5 | GRC-Agent at `6c98128`, its knowledge databases, the tokenizer |
| 6 | llama.cpp b10333 with CUDA, compiled for this GPU |
| 7 | `./gb env` (version pins) and the GNU Radio block catalog hash |
| 8 | `./gb e2e`: Tier K (scored with placeholder rubrics), Tier F, Tier A, Tier E on a few dev items |
| 9 | speeds: Tier E twice, the GPU's clocks and temperature, Tier-K tokens per second |
| 10 | the judge runner, with the same local model as a stand-in judge (plumbing only) |
| 11 | everything into one results folder and zip |

Each step is recorded in `feasibility.json` as ok or failed, with its time and details. A failed step does not
stop the later ones.

## Placeholder rubrics

The team's dev rubrics are not written yet. `colab/dev_placeholder.jsonl` gives each dev question one required
point: the training release's reference answer, copied word for word (`./gb placeholder-rubrics`). With it, an
answer goes through the whole route (rules, judge request, verdict, score), but the score says nothing about the
model. Once the team's rubrics are approved in `items/tier_k/dev/rubrics.jsonl`, score the same answers without
the file: `./gb tier-k score DIR`.
