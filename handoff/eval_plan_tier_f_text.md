# Tier F in the evaluation plan: replacement text

For `Benchmark_Evaluation_Plan_to_7Dec.docx` (and its CAD 300 companion). It describes Tier F as
it is built: a short episode inside GRC-Agent's own harness, because GRC-Agent never lets a model
write a `.grc` file and edits a flowgraph only through its `change_graph` tool. Three places change:
§4.3, the B1 row of Table 1, and the V1 rung wherever the ladder is listed. The documents are
rebuilt only on request.

## §4.3 Tier F: building, editing and repairing flowgraphs

Tier F has four task families (build, modify, repair, parameter-tune) and three difficulty levels
(L1–L3). A task is a short episode in GRC-Agent's harness, with GRC-Agent's real system prompt and
tool definitions, so that the model meets the same inputs as in real use and in Tier A. The episode
starts with the task's flowgraph open (an empty flowgraph with a sample-rate variable for build
tasks) and with one `inspect_graph` call and its real result already in the history. The model may
then look up the documentation once (`query_knowledge`, with a query of its choice) and must make
its edit in one `change_graph` call, within three turns. GRC-Agent applies the edit as one batch; an
edit it rejects is not retried, and any other tool answers that it is not available in this task.

- **Validity ladder.** The edited flowgraph is checked in five steps: V1 a well-formed `change_graph`
  call; V2 the flowgraph loads with only real blocks; V3 it compiles (it validates and its Python code
  is generated); V4 it runs and every observation point receives data; V5 it behaves as the task asks. Each check runs only
  if the previous one passed, so the ladder shows where a model fails.
- **Sources.** The example flowgraphs that ship with GNU Radio, converted to run without a display,
  and short hand-composed chains. All flowgraphs from one source file stay in one split, and the
  functional signature of every test and dev flowgraph is withheld from training.
- **Perturbed variants.** For each task built from a public example we change its numbers (sample
  rates, frequencies, filter settings) and record where it came from, so we can report scores on
  original and perturbed tasks separately.
- **Oracles.** Every task has a reference ("golden") flowgraph and a stored golden edit that turns the
  start flowgraph into it. The V5 oracle checks the signals at the task's observation points and,
  where the instruction states a structure ("filter the signal", "change parameter values only"), the
  structure of the flowgraph. Repair and modify tasks are checked both ways: the requested edit works,
  and everything that worked in the starting flowgraph still works.
- **Quality gates.** The golden edit passes every check in 10 of 10 runs through the real harness; at
  least two other valid solutions pass; the oracle rejects at least 90 % of mutants that break the
  specification; and the start flowgraph of a repair, modify or tune task fails the oracle.
- **Coverage.** By 7 December at least 50 tasks have V5 oracles. Tasks without an oracle are scored
  up to V4.
- **Metrics.** The pass rate at each rung, shown as a funnel; Exec@1 and Func@1 with the unbiased
  estimator; the hallucinated-block rate over repeated samples; and a partial-credit score over
  blocks, parameters and connections. We also report whether the model used the lookup, whether the
  lookup's results contained the reference's blocks, and whether GRC-Agent accepted the edit.
- **Limits.** Each model turn is limited to 8K tokens, reasoning included.

## Table 1, row "Baselines B1 and B2"

| Topic | This plan | Why |
|---|---|---|
| Baselines B1 and B2 | B1 (documentation search) runs on Tier K with the retrieved pages added to the prompt. In Tier F the model already has one documentation lookup, so B1 gives it the catalog entries of every block of the reference solution instead. In Tier A every configuration already has the search tool. B2 (grammar-constrained tool calls) runs on Tier A. | Each baseline is run where it can make a difference. |

## Wherever the ladder is listed (Key terms, §1.2, Table 2, §4.3)

"V1 parse" reads "V1 a well-formed edit call".
