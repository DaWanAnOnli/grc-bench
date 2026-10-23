# grc-agent-headless: tutorial

`grc-agent-headless` runs GRC-Agent without its window. You give it a message as text, and it writes back
everything that happened, as files:
- the agent's replies;
- every tool call with its arguments and result;
- the full trace in Qoherent's training-row format;
- the flowgraph as GNU Radio Companion holds it, plus a readable view of it;
- the agent's project folder.

A later call can continue the same conversation on the same flowgraph.

It is GRC-Agent's real executor at commit `6c98128`: the same system prompt, the same 22 tools and real
flowgraph runs. Only the GTK window is replaced, by a headless copy of the object behind GRC-Agent's canvas.
It comes from GRC-Bench (Team JJJP), where the same runner drives the Tier-A benchmark.

Contents:
1. [What you need](#1-what-you-need)
2. [Setup (once, about 20 minutes)](#2-setup-once-about-20-minutes)
3. [First run](#3-first-run)
4. [Inputs](#4-inputs)
5. [Outputs](#5-outputs)
6. [Conversations](#6-conversations)
7. [Models](#7-models)
8. [Limits](#8-limits)
9. [What the agent can and cannot do](#9-what-the-agent-can-and-cannot-do)
10. [Using it from another program](#10-using-it-from-another-program)
11. [Training data and the benchmark](#11-training-data-and-the-benchmark)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What you need

| | |
|---|---|
| Operating system | Ubuntu 26.04 LTS: native, or under WSL 2 on Windows 10/11 |
| GNU Radio | 3.10.12, the `gnuradio` package of Ubuntu 26.04 |
| GRC-Agent | Qoherent's repository (`github.com/qoherent/grc-agent`, access needed) at commit `6c98128` |
| Sandbox tools | `bubblewrap` and `prlimit` |
| A model | a local OpenAI-compatible server (llama.cpp's `llama-server`, ...), or an API key for a hosted OpenAI-compatible API |
| Disk | about 2 GB, plus the model files if you serve one locally |

The versions are pinned: `./agent` refuses to run with another GRC-Agent commit or GNU Radio version.

## 2. Setup (once, about 20 minutes)

### Step 1 (Windows only): WSL 2 with Ubuntu 26.04

In PowerShell:

```powershell
wsl --list --online          # find the Ubuntu 26.04 entry
wsl --install -d <the Ubuntu 26.04 name from the list>
```

Open the new Ubuntu terminal and create your Linux user when asked. From here on, every command runs **inside
Ubuntu**.

### Step 2: system packages

```bash
sudo apt update
sudo apt install -y gnuradio bubblewrap util-linux git curl unzip python3-gi python3-gi-cairo
/usr/bin/python3 -c "from gnuradio import gr; print(gr.version())"     # expect 3.10.12...
```

### Step 3: uv (Python package manager used by GRC-Agent)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash                                                               # reload the shell so `uv` is found
```

### Step 4: GRC-Agent at the pinned commit

```bash
git clone https://github.com/qoherent/grc-agent.git ~/grc-agent
cd ~/grc-agent
git checkout 6c98128
uv venv --system-site-packages --python /usr/bin/python3
uv sync --extra dev --locked --python .venv/bin/python
```

`--system-site-packages` lets GRC-Agent's environment see the system GNU Radio. Use `/usr/bin/python3`: GNU Radio's
bindings are built for the system interpreter. To keep GRC-Agent somewhere else, set
`export GRC_AGENT_REPO=/path/to/grc-agent` in your `~/.bashrc`.

### Step 5: unzip this package in your Linux home folder

```bash
cd ~
unzip /mnt/c/Users/<your Windows user>/Downloads/grc-agent-headless-<id>.zip     # or wherever the zip is
cd ~/grc-agent-headless
```

It must live in your Linux home folder, not in `/tmp` or under `/mnt/c`, because the sandbox hides both.

### Step 6: run the setup check

```bash
bash setup.sh
```

It checks the steps above and prints a fix for anything missing. It then does two one-time jobs:
- it builds GRC-Agent's knowledge databases (block catalog and GNU Radio wiki, lexical search, a few seconds);
- it downloads the tokenizer file the harness uses to measure tool outputs.

Finally it runs `./agent selftest`, which uses scripted models, so it needs no model server. A good run ends
like this:

```
ok    GRC-Agent ran headless in the sandbox (status completed)
ok    change_graph, save_graph and run_flowgraph executed
ok    the flowgraph ran in GNU Radio, sandboxed (return code 0)
ok    the flowgraph wrote project/selftest.dat
ok    graph.grc / graph_view.json: valid, the saved file matches GRC's memory
ok    summary.json, transcript.md, trace.jsonl, graph.grc and graph_view.json written
ok    continue: the next call has the whole conversation
ok    hosted-model relay: the request went out with the key and the reply came back
ok    hosted-model relay: the key is in no output file
self-test: 9/9 checks passed
```

### Step 7: a model

Choose one:
- **A local model.** llama.cpp's `llama-server` serves any GGUF model on an OpenAI-compatible endpoint. The
  settings GRC-Bench uses, for example with Qwen3-4B:

  ```bash
  llama-server -m ~/models/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 65536 \
      -ctk q8_0 -ctv q8_0 -fa on --jinja --reasoning-format deepseek --no-context-shift \
      --sse-ping-interval 30 -np 1 -ngl 999
  ```

  Leave out `-ngl 999` without a GPU. `--jinja` is needed for tool calls, and `-c 65536` gives a 64K context.
  Any other OpenAI-compatible server works too; point `--base-url` at it.
- **A hosted API.** See [section 7](#7-models).

## 3. First run

With `llama-server` running on port 8080:

```bash
cd ~/grc-agent-headless

# a new flowgraph from scratch
./agent run "Build a flowgraph: a 1 kHz Signal Source into a Throttle and a QT GUI Time Sink. Save it as tone.grc."

# a flowgraph of your own, in a named session
./agent run "Change the tone to 2 kHz and save the flowgraph." --graph ~/flowgraphs/tone.grc --name tone-demo

# the next message of that session: full history, same flowgraph
./agent continue ~/grc-bench-work/agent/tone-demo/call-01 "Now add a Low Pass Filter with a 5 kHz cutoff."

# the results
./agent show ~/grc-bench-work/agent/tone-demo/call-02                 # summary.json
./agent show ~/grc-bench-work/agent/tone-demo/call-02 --transcript    # the conversation as text
```

Each call prints a short report:

```
session tone-demo, call 2: completed (agent_finished)
run folder: /home/you/grc-bench-work/agent/tone-demo/call-02
graph: valid, project/tone.grc (graph.grc, graph_view.json)
files: changed tone.grc
requests 3, tokens in 18240 / out 512, tool calls 4

final reply:
I added a Low Pass Filter (cutoff 5 kHz) between the Signal Source and the Throttle and saved tone.grc.
```

## 4. Inputs

| Option | Meaning |
|---|---|
| `MESSAGE` | the user message, of any length |
| `--graph FILE` | a `.grc` file, copied into the project folder and opened as the active flowgraph. Without it the agent starts from GRC's File > New. |
| `--files DIR` | a folder copied into the project folder, for data files. A flowgraph inside it can be named with `--graph`. |
| `--name NAME` | the session name (default: date and time) |
| `--then MESSAGE` | a follow-up message sent after the agent's turn, whatever it said (repeatable, sent in order) |
| `--reply REGEX=TEXT` | an answer sent when the agent asks something that matches REGEX (repeatable) |
| `--auto-reply TEXT` | the answer to any other question. Without it the call stops when the agent asks ([section 6](#6-conversations)). |
| `--max-user-turns N` | the most scripted messages one call sends (default: the number of `--then` and `--reply` messages, plus 3 with `--auto-reply`) |
| `--request FILE` | a JSON file with the same fields; options given on the command line take precedence |
| `--json` | print `summary.json` instead of the short report |

Your input files are copied, never edited. Model options are in [section 7](#7-models) and limits in
[section 8](#8-limits).

**Request files** (`examples/request.json`, `examples/continue.json`). The keys are the option names: `message`,
`graph`, `files`, `name`, `then` (a list), `replies` (a list of `{"pattern", "reply"}`), `auto_reply`,
`max_user_turns`, `base_url`, `uds`, `model`, `api_base`, `api_key_env`, `preset`, and `caps` (an object with
`requests`, `context_tokens`, `tool_output_tokens`, `episode_seconds`, `flowgraph_run_seconds`). For `continue`,
add `run_dir`.

```bash
./agent run --request examples/request.json --json
./agent continue --request examples/continue.json
```

## 5. Outputs

Each session is a folder `~/grc-bench-work/agent/<name>/`:

| Path | Content |
|---|---|
| `session.json` | every call in order: run folder, parent call, message, status, end reason, usage, model and limits |
| `tasks/call-NN/` | the request as written for call NN (`task.json` + `start/`) |
| `call-NN/summary.json` | the call's results (below) |
| `call-NN/transcript.md` | the whole conversation so far, as plain text: messages, thinking, tool calls with arguments, tool results |
| `call-NN/trace.jsonl` | the same conversation as one row in Qoherent's training format (system prompt, tools, messages, usage) |
| `call-NN/messages.json` | the full message history in pydantic-ai's own format |
| `call-NN/graph.grc` | the flowgraph as GRC holds it at the end, saved or not (GRC's file format) |
| `call-NN/graph_view.json` | what GNU Radio makes of it: blocks, parameters with their evaluated values, connections, validation errors |
| `call-NN/project/` | the agent's project folder at the end: saved flowgraphs, generated Python, files it wrote, files a flowgraph wrote |
| `call-NN/result.json` | the complete record: every tool call's validation and execution, every model request with tokens and seconds, flowgraph runs, limits |
| `call-NN/episode.log` | the log of the process that ran GRC-Agent |

**`summary.json`:**
- `status`: `completed`, `cap_failure` (a limit was hit), `agent_error` (the agent run failed, for example retries
  ran out) or `infrastructure_error` (the model server or the harness failed).
- `end_reason`: `agent_finished`, `agent_asked`, `user_turns_exhausted` or `cap:<limit>`.
- `waiting_for_reply`, `final_reply`, and `turns` (each user message with the agent's reply).
- `tool_calls`: tool, arguments as the model wrote them, validation, execution outcome and the result text the
  model saw.
- `flowgraph_runs`: command, return code, seconds. The console output is in the `run_flowgraph` and `get_run_log`
  results.
- `graph`:
  - `file_in_project`: the flowgraph's file in `project/`, or null for an untitled graph;
  - `page_saved_flag`: GRC's saved flag;
  - `file_matches_graph`: whether that file holds the same graph as GRC's memory;
  - `valid` and `errors`.

  GRC-Agent's `change_graph` writes every edit straight to the flowgraph's file when it has one, and GRC then
  marks the page modified (the asterisk in the window title). So `page_saved_flag` can be false while
  `file_matches_graph` is true, as in the GUI.
- `files`: added, changed and removed against the start of the call. GRC-Agent's own `.grc_agent/` lock and backup
  files are left out.
- `usage` for the call and `session_totals` for the session; `model`, `setting`, `caps`, `caps_hit`; `error`.

**From Windows**, the session folders are at `\\wsl$\<distro name>\home\<linux user>\grc-bench-work\agent\` in
File Explorer.

## 6. Conversations

Within one call, the user's side is fixed in advance:
- `--then` messages are sent in order after each agent turn.
- `--reply` answers questions that match their pattern.
- `--auto-reply` answers any other question.

Without `--auto-reply`, a question with no scripted answer ends the call. The end reason is `agent_asked` and
`waiting_for_reply` is true. Read `final_reply`, decide the answer, and send it:

```bash
./agent continue ~/grc-bench-work/agent/<name>/call-01 "Yes, use 48 kHz."
```

`continue` carries over:
- the full message history and the conversation id;
- the project folder;
- the flowgraph exactly as GRC held it, including edits to an untitled graph that were never saved;
- GRC-Agent's own state.

Flowgraphs still running at the end of a call are stopped; the next call starts with none running. Any earlier
call can be continued, so a session can branch. Each call gets its own limits; `session_totals` adds them up.

## 7. Models

**Local server (the default).** `--base-url URL` gives the server address (default `http://127.0.0.1:8080/v1`),
`--uds PATH` a server on a UNIX socket, and `--model NAME` the model name sent to the server.

**Hosted API.** Put the key in an environment variable, then name the variable (not the key):

```bash
export OPENROUTER_API_KEY=...          # in your shell, or in ~/.bashrc; never in a file you share
./agent run "Build an FM receiver for 98.1 MHz with an RTL-SDR source." \
    --api-base https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model <provider model id>
```

- **Which APIs work.** Any OpenAI-compatible endpoint works, for example `https://api.openai.com/v1`,
  `https://openrouter.ai/api/v1` or `https://api.deepseek.com/v1`. Providers' own non-OpenAI APIs are not used:
  take their OpenAI-compatible endpoint.
- **How the key is kept safe.** The key is read once, outside the sandbox. A small relay adds it to each request,
  and the agent's process talks to the relay over a UNIX socket and has no network at all. The key never enters
  the sandbox or any output file; the self-test checks this.
- **Continuing a session.** `continue` reuses the model of the call it continues, so the variable must still be
  set.
- **Compaction.** From inside the sandbox GRC-Agent cannot ask the provider for the model's context window. With
  compaction on, it therefore uses its fallback rule for self-hosted servers, which compacts earlier.

## 8. Limits

| Option | `--preset gui` (default) | `--preset benchmark` |
|---|---|---|
| compaction (dropping old tool results from long conversations) | GRC-Agent's own | off |
| `--max-requests N\|none` | 40 | 40 |
| `--context-tokens N\|none` (input tokens per model request) | none | 65,536 |
| `--tool-output-tokens N\|none` | none | 4,096 |
| `--episode-seconds S` (one call) | 1,800 | 1,800 |
| `--flowgraph-seconds S` (one flowgraph run) | 300 | 300 |

Hitting a limit ends the call with status `cap_failure`. `continue` reuses the previous call's preset and limits
unless you give new ones. `--preset benchmark` matches GRC-Bench's Tier-A settings.

## 9. What the agent can and cannot do

- **Approvals:** every change and command is approved automatically, as in GRC-Agent's "yolo" mode, but
  everything runs inside a bubblewrap sandbox.
- **Files:** the agent writes only in its project folder.
- **Shell commands** (`run_command`, `start_command`):
  - they have no network;
  - they see system files, GNU Radio and GRC-Agent's code, plus their own project folder;
  - they cannot see your home folder, the Windows drives or other sessions.
- **Flowgraphs** really run, headless (QT GUI windows are drawn off-screen), sandboxed, with a time limit.
  SDR hardware is not available inside the sandbox.
- **Web search and fetch** answer "offline".

## 10. Using it from another program

**From any language.** Call `./agent ... --json` and parse standard output (`summary.json`), or read the session
folder. From Windows:

```
wsl -d <distro name> -- bash -lc "cd ~/grc-agent-headless && ./agent run 'Add a throttle after the source.' --graph ~/flowgraphs/tone.grc --json"
```

The exit code is 0 when the call ran, 1 for an infrastructure error and 2 when the request was refused (a bad
option, a missing file, a missing API-key variable).

**From Python**, with GRC-Agent's interpreter (`~/grc-agent/.venv/bin/python`) and `PYTHONPATH=~/grc-agent-headless/src`.
Set the same variables the `agent` script sets.

```python
from grcbench.tier_a import adhoc, models

first = adhoc.run("Change the tone to 2 kHz.", graph="/home/you/flowgraphs/tone.grc", name="demo")
if first["waiting_for_reply"]:
    second = adhoc.continue_run(first["run_dir"], "Yes, and save it.")
hosted = models.hosted("https://openrouter.ai/api/v1", "<provider model id>", "OPENROUTER_API_KEY")
third = adhoc.run("Build a QPSK modulator.", model=hosted, preset="benchmark", caps={"requests": 60})
```

**One call at a time per machine** is the safe default. Calls run in separate sandboxed processes, but a local
`llama-server` with `-np 1` serves one request at a time.

## 11. Training data and the benchmark

- This package contains no GRC-Bench benchmark items, answer keys or solutions.
- If your sessions may become training data, never paste GRC-Bench benchmark questions or tasks into a message:
  GRC-Bench's test items are never sent to a model before its registered runs.
- The task records written here are training-side records and never carry GRC-Bench's canary string.
- Whether sessions become training data is the training team's decision. `trace.jsonl` is already in Qoherent's
  row format.

## 12. Troubleshooting

| Message | Cause and fix |
|---|---|
| `pin check failed: GRC-Agent at ..., expected 6c98128` | `git -C ~/grc-agent checkout 6c98128` |
| `pin check failed: GNU Radio ..., expected 3.10.12.0` | use Ubuntu 26.04 and its `gnuradio` package |
| `GRC-Agent's Python was not found` | step 4: create GRC-Agent's virtual environment; or set `GRC_AGENT_REPO` |
| `GRC-Agent knowledge database missing` | run `bash setup.sh` (step 6 builds them) |
| `bwrap: setting up uid map: Permission denied` or `bwrap: No permissions to create a new namespace` | native Ubuntu 24.04+ restricts unprivileged user namespaces: `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`, or add `kernel.apparmor_restrict_unprivileged_userns=0` to `/etc/sysctl.d/60-userns.conf` to keep it |
| `must live in your Linux home folder` | move the package folder to `~/` |
| status `infrastructure_error`, end reason `model server error` | the model server is not reachable: check that it runs and `--base-url`; `episode.log` has the details |
| `refused: environment variable ... is not set` | export the API-key variable in the shell that runs `./agent` |
| status `cap_failure` | a limit was hit (`caps_hit` names it): raise it ([section 8](#8-limits)) |
| `run_flowgraph` says the flowgraph was never saved | GRC's rule: a flowgraph must be saved once before it runs; the agent normally saves first |

**Questions:** Joel Rompis (GRC-Bench evaluation lead).
