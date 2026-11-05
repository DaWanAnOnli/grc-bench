#!/usr/bin/env bash
# GRC-Bench setup check for a new Linux or WSL machine (README, "Set up a new machine"). Safe to run
# again at any time. Checks every prerequisite of ./gb and says how to fix what is missing; builds,
# once, the two things a fresh machine lacks (GRC-Agent's lexical knowledge databases, which GRC-Agent
# keeps in its git-ignored vectors/ folder and builds on first use, and the tokenizer file); ends with
# ./gb's own pin check and the read-only copies of the databases that every GRC-Agent run uses.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRC_AGENT_REPO="${GRC_AGENT_REPO:-$HOME/grc-agent}"
CACHE="${GRC_BENCH_CACHE:-$HOME/grc-bench-cache}"
PIN_AGENT=6c98128
PIN_GR=3.10.12
PY="$GRC_AGENT_REPO/.venv/bin/python"
failed=0

ok()   { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; failed=1; }
stop_if_failed() {
  if [ "$failed" -ne 0 ]; then
    echo; echo "setup: fix the FAIL lines above, then run ./setup.sh again"; exit 1
  fi
}

echo "== 1. system tools"
for cmd in bwrap prlimit git python3; do
  if command -v "$cmd" >/dev/null 2>&1; then ok "$cmd"; else
    fail "$cmd not found" "sudo apt install -y bubblewrap util-linux git python3"; fi
done

echo "== 2. GNU Radio $PIN_GR"
gr=$(/usr/bin/python3 -c "from gnuradio import gr; print(gr.version())" 2>/dev/null || true)
case "$gr" in
  "$PIN_GR"*) ok "GNU Radio $gr" ;;
  "") fail "GNU Radio's Python bindings not found" "sudo apt install -y gnuradio (Ubuntu 26.04 ships $PIN_GR)" ;;
  *) fail "GNU Radio $gr (GRC-Bench is pinned to $PIN_GR)" "use Ubuntu 26.04, whose gnuradio package is $PIN_GR" ;;
esac

echo "== 3. GRC-Agent at $PIN_AGENT in $GRC_AGENT_REPO"
if [ ! -d "$GRC_AGENT_REPO/.git" ]; then
  fail "no GRC-Agent checkout at $GRC_AGENT_REPO" \
       "git clone https://github.com/qoherent/grc-agent.git $GRC_AGENT_REPO && git -C $GRC_AGENT_REPO checkout $PIN_AGENT"
else
  head=$(git -C "$GRC_AGENT_REPO" rev-parse --short=7 HEAD 2>/dev/null)
  if [ "$head" = "$PIN_AGENT" ]; then ok "GRC-Agent commit $head"; else
    fail "GRC-Agent is at $head" "git -C $GRC_AGENT_REPO checkout $PIN_AGENT"; fi
  if [ -x "$PY" ] && "$PY" -c "import grc_agent, gnuradio, pydantic_ai" 2>/dev/null; then
    ok "GRC-Agent's virtual environment ($PY) imports grc_agent and gnuradio"
  else
    fail "GRC-Agent's virtual environment is missing or incomplete" \
         "install uv (https://docs.astral.sh/uv/), then: cd $GRC_AGENT_REPO && uv venv --system-site-packages --python /usr/bin/python3 && uv sync --extra dev --locked --python .venv/bin/python"
  fi
fi

echo "== 4. the sandbox (bubblewrap)"
if bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid --unshare-net --die-with-parent true 2>/tmp/grc-bench-setup-bwrap.txt; then
  ok "bubblewrap can create a sandbox"
else
  fail "bubblewrap cannot create a sandbox: $(head -1 /tmp/grc-bench-setup-bwrap.txt)" \
       "on native Ubuntu 24.04+: sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0 (handoff/agent_package/TUTORIAL.md, Troubleshooting)"
fi
rm -f /tmp/grc-bench-setup-bwrap.txt
stop_if_failed

echo "== 5. GRC-Agent's knowledge databases (built once)"
VEC="$GRC_AGENT_REPO/src/grc_agent/vectors"
if [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
  ok "catalog_lexical.db and docs_lexical.db present"
else
  echo "      building them in $VEC (lexical search, a few seconds) ..."
  if (cd "$GRC_AGENT_REPO" && GRC_EMBED_BACKEND=lexical QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 "$PY" -c "
from grc_agent.adapter import rag
for domain in ('catalog', 'docs'):
    path, model = rag.get_db_and_model(domain)
    rag._ensure_db_built(domain, path, model)
" >/tmp/grc-bench-setup-dbs.txt 2>&1) && [ -f "$VEC/catalog_lexical.db" ] && [ -f "$VEC/docs_lexical.db" ]; then
    ok "built catalog_lexical.db and docs_lexical.db"
  else
    fail "could not build the knowledge databases: $(tail -1 /tmp/grc-bench-setup-dbs.txt)" \
         "check that GRC_AGENT_VECTORS_DIR is not set, then run ./setup.sh again"
  fi
  rm -f /tmp/grc-bench-setup-dbs.txt
fi

echo "== 6. tokenizer file (downloaded once)"
mkdir -p "$CACHE/tiktoken"
if TIKTOKEN_CACHE_DIR="$CACHE/tiktoken" "$PY" -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" 2>/dev/null; then
  ok "cl100k_base in $CACHE/tiktoken"
else
  fail "could not get the cl100k_base tokenizer file" "check the internet connection, then run ./setup.sh again"
fi
stop_if_failed

echo "== 7. ./gb's pin check and the read-only database copies"
if out=$(bash "$HERE/gb" tier-a vectors 2>&1); then
  ok "./gb runs; database copies in $(printf '%s\n' "$out" | tail -1)"
else
  fail "./gb tier-a vectors failed: $(printf '%s\n' "$out" | tail -1)" "fix what the message names, then run ./setup.sh again"
fi
stop_if_failed

echo
echo "setup complete. Next: ./gb help (the commands), ./gb test (the unit tests)."
echo "Model runs also need llama.cpp and the smoke-test model: ./gb fetch (add --cuda on an NVIDIA GPU)."
