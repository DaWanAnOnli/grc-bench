# Running GRC-Agent headless: `./gb agent`

`./gb agent` gives GRC-Agent a text message and writes back everything that happened, as files:
- the agent's replies and every tool call with its result;
- the full trace in Qoherent's training-row format;
- the flowgraph as GRC holds it, plus a readable view of it;
- the project folder.

A later call can continue the same conversation on the same flowgraph.

It runs GRC-Agent's real executor (commit `6c98128`) without the GTK window: the same system prompt, the
same 22 tools, real flowgraph runs. This is the harness GRC-Bench's Tier A uses (`src/grcbench/tier_a/`).

## Requirements

- **Linux or WSL Ubuntu** with GNU Radio 3.10.12, bubblewrap (`bwrap`) and `prlimit`.
- **GRC-Agent** at `~/grc-agent`, commit `6c98128`, with its `.venv`.
- **This repository.** `./gb` refuses to run if the pins don't match. All of this is set up on Joel's machine
  (`GRC-Bench-Ubuntu`).
- **A model:** a local OpenAI-compatible server (llama.cpp `llama-server`, default `http://127.0.0.1:8080/v1`),
  or a hosted OpenAI-compatible API with its key in an environment variable.

Do not copy the WSL package of Joel's machine to set up another computer: it contains API keys.

## Quick start

From the repository folder inside WSL:

```bash
# a new flowgraph from scratch
./gb agent run "Build a flowgraph: a 1 kHz Signal Source into a Throttle and a QT GUI Time Sink."

# work on an existing flowgraph, in a named session
./gb agent run "Change the tone to 2 kHz and save the flowgraph." --graph ~/flowgraphs/tone.grc --name tone-demo

# the next message of that session (full history, same flowgraph)
./gb agent continue ~/grc-bench-work/agent/tone-demo/call-01 "Now add a Low Pass Filter with a 5 kHz cutoff."

# read a call's results
./gb agent show ~/grc-bench-work/agent/tone-demo/call-02              # summary.json
./gb agent show ~/grc-bench-work/agent/tone-demo/call-02 --transcript # the conversation as text
```

From Windows (PowerShell or cmd), add `--json` to get the summary as JSON on standard output:

```
wsl -d GRC-Bench-Ubuntu -- bash -lc "cd '/mnt/c/Users/POPA/Documents/!MScAC/2701/ml_ops_project/project/grc-bench' && ./gb agent run 'Add a throttle after the source.' --graph /home/lenovo/flowgraphs/tone.grc --json"
```

## Inputs

| Option | Meaning |
|---|---|
| `MESSAGE` | the user message (any length) |
| `--graph FILE` | a `.grc` file, copied into the project folder and opened as the active flowgraph; without it the agent starts from GRC's File > New |
| `--files DIR` | a folder copied into the project folder (data files, a flowgraph inside it can be named with `--graph`) |
| `--name NAME` | the session name (default: date and time) |
| `--then MESSAGE` | a follow-up message sent after the agent's turn, whatever it said (repeatable, sent in order) |
| `--reply REGEX=TEXT` | an answer sent when the agent asks something that matches REGEX (repeatable) |
| `--auto-reply TEXT` | the answer to any other question; without it the call stops when the agent asks (see "Conversations") |
| `--max-user-turns N` | the most scripted messages one call sends (default: the number of `--then` and `--reply` messages, plus 3 with `--auto-reply`) |
| `--request FILE` | a JSON file with the same fields (below); options given on the command line take precedence |
| `--json` | print `summary.json` instead of the short report |

The input files are copied, never edited.

### Request file (for automation)

```json
{
  "message": "Add a Low Pass Filter after the source, cutoff 5 kHz, and save.",
  "graph": "/home/lenovo/flowgraphs/tone.grc",
  "name": "lpf-001",
  "then": ["Now run the flowgraph for two seconds and tell me whether it worked."],
  "replies": [{"pattern": "which (window|filter type)", "reply": "Use a Hamming window."}],
  "auto_reply": null,
  "base_url": "http://127.0.0.1:8080/v1",
  "preset": "gui",
  "caps": {"requests": 60}
}
```

`./gb agent run --request lpf-001.json --json` runs it; `./gb agent continue --request next.json` takes
`run_dir` and `message` from the file.

## Outputs

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
| `call-NN/project/` | the agent's project folder at the end (saved flowgraphs, generated Python, files it wrote) |
| `call-NN/result.json` | the complete record: every tool call's validation and execution, every model request with tokens and seconds, flowgraph runs, caps |
| `call-NN/episode.log` | the episode process's log |

**`summary.json` fields:**
- `status`: `completed`, `cap_failure` (a limit was hit), `agent_error` or `infrastructure_error` (the model server
  or the harness failed).
- `end_reason`: `agent_finished`, `agent_asked` (see below), `user_turns_exhausted` or `cap:<limit>`.
- `waiting_for_reply`, `final_reply`, and `turns` (each user message with the agent's reply).
- `tool_calls`: tool, arguments as the model wrote them, validation, execution outcome and the result text.
- `flowgraph_runs`: command, return code, seconds. The console output is in the `run_flowgraph` and
  `get_run_log` results.
- `graph`:
  - `file_in_project`: the flowgraph's file in `project/`, or null for an untitled graph;
  - `page_saved_flag`: GRC's saved flag;
  - `file_matches_graph`: whether that file holds the same graph as GRC's memory;
  - `valid` and `errors`.

  GRC-Agent's `change_graph` writes each edit through to the flowgraph's file when it has one, and GRC then marks
  the page modified. So `page_saved_flag` can be false while `file_matches_graph` is true.
- `files`: added, changed and removed against the start of the call. GRC-Agent's own `.grc_agent/` lock and
  backup files are left out.
- `usage` for the call and `session_totals` for the session; `model`, `setting`, `caps`, `caps_hit`; `error`.

## Conversations

Within one call, the user's side is fixed in advance:
- `--then` messages are sent in order after each agent turn.
- `--reply` answers questions that match their pattern.
- `--auto-reply` answers any other question.

Without `--auto-reply`, a question with no scripted answer ends the call. The end reason is `agent_asked` and
`waiting_for_reply` is true. Read `final_reply`, decide the answer, and send it with `./gb agent continue`.

`continue` carries over:
- the full message history and the conversation id;
- the project folder;
- the flowgraph exactly as GRC held it, including edits that were never saved;
- GRC-Agent's own state.

Flowgraphs still running at the end of a call are stopped; the next call starts with none running. Any earlier
call can be continued, so a session can branch. Each call has its own limits; `session_totals` adds them up.

## Models

**Local server (default).** Any OpenAI-compatible server, for example llama.cpp's `llama-server` with `--jinja`
on port 8080. Use `--base-url URL` for another address and `--uds PATH` for a server on a UNIX socket.
`--model NAME` is the name sent to the server.

**Hosted API.** Put the key in an environment variable, then name the variable (not the key):

```bash
export OPENROUTER_API_KEY=...            # in your own shell; never in a file inside the repository
./gb agent run "Build an FM receiver for 98.1 MHz with an RTL-SDR source." \
    --api-base https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model <provider model id>
```

- **How the key is kept safe.** The key is read on the host. A small relay (`tier_a/relay.py`) adds it to each
  request, and the agent's process talks to the relay over a UNIX socket and has no network at all. The key
  never enters the sandbox or any output file.
- **Which APIs work.** Any OpenAI-compatible endpoint works, for example `https://api.openai.com/v1`,
  `https://openrouter.ai/api/v1` or `https://api.deepseek.com/v1`. Providers' own non-OpenAI APIs are not used:
  use their OpenAI-compatible endpoint.
- **Compaction.** From inside the sandbox GRC-Agent cannot ask the provider for the model's context window. With
  compaction on, it therefore uses its fallback rule for self-hosted servers, which compacts earlier.

## Limits

| Option | Default (`--preset gui`) | `--preset benchmark` |
|---|---|---|
| compaction | GRC-Agent's own | off |
| `--max-requests N\|none` | 40 | 40 |
| `--context-tokens N\|none` (input tokens per request) | none | 65,536 |
| `--tool-output-tokens N\|none` | none | 4,096 |
| `--episode-seconds S` | 1,800 | 1,800 |
| `--flowgraph-seconds S` (one flowgraph run) | 300 | 300 |

Hitting a limit ends the call with status `cap_failure`. `continue` reuses the previous call's preset and limits
unless new ones are given.

## What the agent can and cannot do

- **Approvals:** every change and command is approved automatically, as in GRC-Agent's "yolo" mode, but inside a
  bubblewrap sandbox.
- **Files:** the agent writes only in its project folder.
- **Shell commands** (`run_command`, `start_command`):
  - they have no network;
  - they see system files, GNU Radio and GRC-Agent's code, plus their own project folder;
  - they cannot see your home folder, the Windows drives, other runs or this repository.
- **Flowgraphs** really run, headless and sandboxed, with a time limit.
- **Web search and fetch** answer "offline".

## Keeping the benchmark separate

- Every call runs as a training-side request (split `train`, collection mode). The run-log guard refuses the
  benchmark's own items, and nothing is written to `items/` or the benchmark's run log.
- The guard checks task ids, not text. Never paste a benchmark question or task (Tier K, F or A, test or dev)
  into a message: test questions are never sent to any model before the registered runs.
- Whether any of these sessions becomes training data is the training team's decision.

## From Python

Inside WSL, with `./gb`'s interpreter (`~/grc-agent/.venv/bin/python`, `PYTHONPATH=src`):

```python
from grcbench.tier_a import adhoc, models

first = adhoc.run("Change the tone to 2 kHz.", graph="/home/lenovo/flowgraphs/tone.grc", name="demo")
if first["waiting_for_reply"]:
    second = adhoc.continue_run(first["run_dir"], "Yes, and save it.")
hosted = models.hosted("https://openrouter.ai/api/v1", "<provider model id>", "OPENROUTER_API_KEY")
third = adhoc.run("Build a QPSK modulator.", model=hosted, preset="benchmark", caps={"requests": 60})
```

From another program, calling `./gb agent ... --json` and reading `summary.json` works just as well.
