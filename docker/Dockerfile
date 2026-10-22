# GRC-Bench harness image (proposal §5: "GNU Radio 3.10.x pinned in Docker").
#
# NOT BUILT YET: Docker is not installed on the build machine (needs admin rights; see
# handoff/joel_actions.md). Until it is, the same pins are enforced by ./gb on the WSL install:
# GNU Radio 3.10.12.0 (Ubuntu 26.04 package), GRC-Agent commit 6c98128, catalog SHA-256 in
# src/grcbench/common/paths.py. Build with:  docker build -t grc-bench:0.1 -f docker/Dockerfile .
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
        gnuradio=3.10.12.0* python3 python3-yaml python3-numpy python3-scipy \
        bubblewrap git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned) and GRC-Agent at the pinned commit, installed exactly as on the build machine
RUN curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN git clone https://github.com/qoherent/grc-agent.git /opt/grc-agent \
    && cd /opt/grc-agent && git checkout 6c98128 \
    && uv venv --system-site-packages .venv && uv sync --frozen

COPY . /opt/grc-bench
WORKDIR /opt/grc-bench
ENV GRC_AGENT_REPO=/opt/grc-agent
ENTRYPOINT ["./gb"]
CMD ["help"]
