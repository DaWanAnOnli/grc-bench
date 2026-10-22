# runs/

`test_run_log.jsonl` is the append-only test-run log (eval plan §4.1): one line per run on the test
split, written *before* the run starts, with the model, settings, code version, date and reason.
Runs on the test split are refused unless their configuration id is on the declared run list
(`docs/preregistration/run_list.yaml`). Everything else in this folder is run output and is not
versioned.
