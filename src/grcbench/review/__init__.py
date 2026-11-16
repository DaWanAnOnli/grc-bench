"""The review protocol for every review sheet (`./gb review`).

Two Claude (Anthropic) review agents, each in its own fresh context and blind to the other, fill in
every sheet in `handoff/review_sheets/` by the sheet's guide. Rubrics have a writer agent and two checker
agents. An expert agent decides every disagreement and audits a seeded sample of the agreements. The
decision rules are code (`decide.py`), not judgement, and every agent run is declared in
`reviews/review_log.jsonl`. Raw agent outputs live outside the repository (`<project>/review_work/`);
the decided records are `reviews/<sheet>.jsonl` and the filled sheets.
"""
