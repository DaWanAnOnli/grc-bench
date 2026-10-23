# Running GRC-Agent headless: `./gb agent`

Inside GRC-Bench, `./gb agent` runs GRC-Agent headless on any request and writes every output as files:
summary, transcript, trace, the flowgraph as GRC holds it, and the project folder. A later call continues the
same conversation (`./gb agent continue`). It is the Tier-A runner (`src/grcbench/tier_a/`) used outside the
benchmark: every call is a training-side request in collection mode, so the run-log guard refuses benchmark
items.

**The guide** is `handoff/agent_package/TUTORIAL.md`. Inside GRC-Bench, read `./gb agent` wherever it says
`./agent`; everything else is the same. It covers setup on a fresh machine, inputs, outputs, conversations,
local and hosted models, limits, the sandbox and troubleshooting.

**For a teammate's machine**, `./gb agent-package` builds the zip `grc-agent-headless-<commit>.zip` (in
`../share/`). It contains only the runner's code, the launcher `agent`, `setup.sh`, the tutorial and examples:
no benchmark items, answer keys, solutions or reports. The build refuses to package any module that holds
benchmark content, or any file that names a benchmark item. `--verify` unzips the package into a fresh folder,
runs its `setup.sh` (which ends with the self-test) and removes the folder again.

**Self-test:** `./gb agent selftest` checks GRC-Agent in the sandbox, a real flowgraph run, the output files,
`continue` and the hosted-model relay, all with scripted models.
