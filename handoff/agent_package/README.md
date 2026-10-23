# grc-agent-headless

GRC-Agent (commit `6c98128`) without its window. A text message goes in; the agent's replies, every tool call,
the trace, the flowgraph and the project folder come out as files. A later call continues the same conversation.

```bash
bash setup.sh                                   # once: checks the setup, builds what is missing, self-test
./agent run "Build a flowgraph with a 1 kHz Signal Source into a QT GUI Time Sink."
./agent continue ~/grc-bench-work/agent/<session>/call-01 "Now save it as tone.grc."
./agent show ~/grc-bench-work/agent/<session>/call-02 --transcript
```

**Start with `TUTORIAL.md`**: it covers setup from a fresh machine, the inputs and outputs, conversations, local
and hosted models, limits and troubleshooting.

| File | |
|---|---|
| `TUTORIAL.md` | the full guide |
| `setup.sh` | the setup check (safe to re-run) |
| `agent` | the command |
| `examples/` | request files for `./agent run --request` and `./agent continue --request` |
| `src/grcbench/` | the runner (from GRC-Bench, Team JJJP); no benchmark items, answer keys or solutions |
| `VERSION.txt` | the GRC-Bench commit this package was built from, the pins, and a checksum of every file |

For Team JJJP's use. GRC-Agent is Qoherent's and comes with its own licence.
