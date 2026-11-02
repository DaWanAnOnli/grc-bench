"""Command-line entry point (`./gb <command>`).

Commands are looked up lazily in COMMANDS ("module:function"), so each workstream registers its own
entry without importing GNU Radio or GRC-Agent for unrelated commands. Every command except `env`
and `help` first checks the pinned GNU Radio and GRC-Agent versions.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    # setup and housekeeping
    "env": ("grcbench.common.environment:main", "record versions and pins in reports/environment.json"),
    "baseline": ("grcbench.common.baseline:main", "take | check manifests of read-only inputs"),
    "fetch": ("grcbench.common.fetch:main", "download pinned llama.cpp, smoke model, embedding model"),
    "reference": ("grcbench.common.reference:main", "index training-side task specs (disjointness)"),
    "test": ("grcbench.common.testing:main", "run the pytest suite"),
    "audit": ("grcbench.common.audit:main", "schema, canary, licence and disjointness checks on all items"),
    "build-all": ("grcbench.common.build_all:main", "regenerate every generated artefact (determinism check)"),
    # splits, labels, contamination
    "split-dev": ("grcbench.splits.dev:main", "draw the Tier-K dev split, anchor and CI subsets"),
    "labels": ("grcbench.labels.task_types:main", "map raw task-type spellings to the 6 canonical types"),
    "tier-k-items": ("grcbench.tier_k.items:main", "write Tier-K test/dev item records"),
    "contamination": ("grcbench.contamination.audit:main", "8-gram, TF-IDF and embedding overlap audit"),
    "traces": ("grcbench.contamination.agent_traces:main", "audit Qoherent's agent traces against the benchmark"),
    # Tier K
    "blocks": ("grcbench.tier_k.catalog:main", "load the GNU Radio block catalog: modules, families, eligibility"),
    "blockq": ("grcbench.tier_k.blockgen:main", "GNU Radio behaviour questions: the reference set, fresh sets from any seed"),
    "calc": ("grcbench.tier_k.calc:main", "compute calculation-question answers by script"),
    "rubrics": ("grcbench.tier_k.rubrics:main", "write rubric templates and review sheets"),
    "placeholder-rubrics": ("grcbench.tier_k.placeholder_rubrics:main",
                            "dev placeholder rubrics from the release's reference answers (pipeline tests only)"),
    "score-k": ("grcbench.tier_k.rules:main", "rule-cascade scorer for Tier-K answers"),
    "tier-k": ("grcbench.tier_k.run:main", "answer a split's Tier-K questions with one model; score them"),
    "retriever": ("grcbench.tier_k.retriever:main", "B1 documentation-search retriever"),
    # Tier F
    "ladder": ("grcbench.tier_f.ladder:main", "run the V1-V5 validity ladder on a .grc file"),
    "seeds": ("grcbench.tier_f.seeds:main", "convert GNU Radio examples to headless seeds; seed split"),
    "tier-f": ("grcbench.tier_f.cli:main", "build and gate Tier-F tasks"),
    # Tier A
    "tier-a": ("grcbench.tier_a.cli:main", "Tier-A runner, gates and smoke test"),
    "agent": ("grcbench.tier_a.adhoc:main", "run GRC-Agent headless on any request; outputs as files"),
    "colab-package": ("grcbench.common.colab_package:main", "zip of ./gb and the dev split for a pipeline test on a Colab GPU"),
    "agent-package": ("grcbench.tier_a.agent_package:main", "zip of the headless runner for a teammate (no benchmark content)"),
    # Tier E, analysis, run log
    "tier-e": ("grcbench.tier_e.bench:main", "llama-bench speed/memory protocol"),
    "stats": ("grcbench.stats.cli:main", "analysis functions (self-check)"),
    "prereg": ("grcbench.stats.prereg:main", "fill the pre-registration's generated appendices"),
    "runlog": ("grcbench.runlog.log:main", "append-only test-run log"),
    "e2e": ("grcbench.e2e:main", "benchmark v0.9 end to end on the dev split with the smoke model (G2)"),
}


def _help() -> None:
    print("usage: ./gb <command> [args]\n\ncommands:")
    for name in sorted(COMMANDS):
        print(f"  {name:22s} {COMMANDS[name][1]}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"help", "-h", "--help"}:
        _help()
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n")
        _help()
        return 2
    if name not in {"env", "help"}:
        from grcbench.common.environment import check_pins
        check_pins()
    target, _ = COMMANDS[name]
    module, func = target.split(":")
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        print(f"command '{name}' is not implemented yet ({exc.name})")
        return 2
    result = getattr(mod, func)(rest)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
