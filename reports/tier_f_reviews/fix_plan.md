# Tier F first set: set-level fix plan

Drafted by Claude (Anthropic) on 18 Oct 2026, from the round-1 reviews and the signature-v2 re-split. **Status: pending team review.** The machine-readable plan, including the full theme briefs, is `reports/tier_f_reviews/fix_plan.json`. No task directory was edited in this stage.

## 1. Decisions

* **15 tasks get a new theme.**
  * 14 of them conflict under signature v2: gbf-dev-01 to -06 and gbf-test-01, -02, -05, -07, -08, -12, -13, -20.
  * The 15th, gbf-test-11, has no conflict. But it shares the cell `filter|tone` with gbf-test-10 and overlaps gbf-test-04 and gbf-test-12 in theme, so it is re-themed too.
* **The other 15 tasks are fixed** within their current theme.
* No task moves between test and dev. Every re-themed composed task gets a new function whose v2 signature hashes to its own split.
* **Two tasks are relabelled:**
  * gbf-test-06 becomes parameter-tune at level L3. Its edit changes one value, and the task is about controlling a block through a message port. The L3 level is pending team confirmation.
  * gbf-test-11 becomes modify, so the test split keeps two modify L1 items.
* **Every old twin group is resolved:**

| v2 cell (old) | split of the cell | members | who keeps it |
|---|---|---|---|
| tone | test | dev-01, dev-03, test-04 | test-04 |
| filter&#124;noise | test | dev-02, test-03 | test-03 |
| combine&#124;filter&#124;mix&#124;resample&#124;tone | test | dev-06, test-18 | test-18 |
| filter&#124;tone | test | test-10, test-11 | test-10 |
| combine&#124;filter&#124;tone | train | dev-04, test-05 | nobody (both re-themed) |
| filter&#124;harmonic_wave (square/sawtooth to sine) | train | dev-05, test-01, -07, -08, -12 | nobody (all re-themed) |
| combine&#124;filter&#124;resample&#124;tone | train | test-13, test-20 | nobody (both re-themed) |

### How the new themes were checked

* I built every new reference, and its most likely alternative structures, as a concrete graph. Each one was classified with `compose.signature` and `compose.signature_split`.
* All 30 references now have distinct cells, and none of them equals a signature reserved by another task.
* Fixers must re-check their real files and report the v2 signature of each variant.

### Themes considered and rejected

| theme | reason |
|---|---|
| OOK keying (tone × square wave) | v2 classifies it by port order: `mix|tone` (train) or `harmonic_wave|mix` (test) |
| Spectral inversion of tones | The conjugate of a generator collapses to `combine|tone` |
| Squaring frequency doubler | Collapses to `filter|mix` |
| Duobinary encoding with Delay + Add | Hashes to train |
| PWM comparator | Its Threshold form hashes to train |
| 16-bit PCM tone | Near-duplicate of test-04 |
| AFSK retune | Near-duplicate of test-19 (GMSK retune) |
| Averaged FFT spectrum | Close to training spec t06 (spectrogram) |

## 2. New themes

### Dev

| task | slot | new theme | v2 cell |
|---|---|---|---|
| gbf-dev-01 | build L1 | sync-word (access-code) detector on a bit stream, byte output | data_source&#124;frame_sync |
| gbf-dev-02 | build L1 | Rayleigh fading envelope: magnitude of complex Gaussian noise | envelope&#124;noise |
| gbf-dev-03 | modify L1 | add a rectangular-pulse matched filter to an NRZ stream | data_source&#124;filter |
| gbf-dev-04 | repair L1, undefined variable | heterodyne mixer to a 2 kHz IF; the LO frequency variable was deleted | filter&#124;mix&#124;tone |
| gbf-dev-05 | tune L1 | Gaussian-noise limiter: set the clip level so that 5 % of samples clip (1.96 σ) | noise&#124;nonlinear |
| gbf-dev-06 | build L2 | USB SSB transmitter, phasing method: two-tone audio at 8 kHz, Hilbert transformer, 10 kHz carrier at 48 kHz | combine&#124;filter&#124;hilbert&#124;mix&#124;resample&#124;tone |

### Test

| task | slot | new theme | v2 cell |
|---|---|---|---|
| gbf-test-01 | build L1 | full-wave rectifier of a 50 Hz waveform (mean 2A/π, ripple at 100 Hz) | envelope&#124;tone |
| gbf-test-02 | build L1 | PLL carrier recovery from a noisy complex carrier | carrier_sync&#124;combine&#124;noise&#124;tone |
| gbf-test-05 | build L1 | BPSK hard-decision demapper with MSB-first bit packing (inverted slicer polarity) | data_source&#124;linear_demod |
| gbf-test-07 | modify L1 | delay a reference stream by a 41-tap FIR's group delay (20 samples) | delay&#124;filter&#124;tone |
| gbf-test-08 | repair L1, deleted connection | phase comparator of two oscillators 5 Hz apart (wrapped phase ramp) | mix&#124;phase&#124;tone |
| gbf-test-11 | modify L1 (relabelled) | add AGC to a weak/strong burst recording | agc&#124;data_source&#124;mix |
| gbf-test-12 | tune L1 | set Channel Model impairments to a 600 Hz offset and 20 dB SNR | channel_model&#124;tone |
| gbf-test-13 | build L2 | baseband-to-IF data transmitter: NRZ data, low-pass interpolation ×8, +8 kHz complex IF | data_source&#124;filter&#124;mix&#124;resample |
| gbf-test-20 | tune L2 | retune a 2-PAM transmitter to Gray-coded 4-PAM at 2.4 kBd | data_source&#124;linear_mod&#124;resample |

Each brief in the JSON gives:
* the numbers to use;
* the observation point;
* the graph checks needed where a shortcut generator would otherwise pass;
* the variants and their signatures;
* the mutants;
* the nearest training-side spec.

## 3. Coverage after the plan

| | build L1 | modify L1 | repair L1 | tune L1 | build L2 | modify L2 | repair L2 | tune L2 | other |
|---|---|---|---|---|---|---|---|---|---|
| test (20) | 01, 02, 03, 04, 05 | 07, 11 | 08, 09, 10 | 12 | 13, 14 | 15, 16 | 17, 18 | 19, 20 | tune L3: 06 |
| dev (10) | 01, 02 | 03 | 04 | 05 | 06 | 07 | 08, 10 | 09 | |

* The test split covers all five repair fault types: deleted connection (08), type mismatch (09), nonexistent block (10), undefined variable (17) and rate mismatch (18).
* The dev split covers three: undefined variable (04), nonexistent block (08) and type mismatch (10).

## 4. Fixes to kept tasks that would change their cell

I checked these risks with concrete graphs:

* **gbf-test-10:** do not add a hum source to make the cutoff measurable. With it, the cell becomes `combine|filter|tone`, which hashes to train. Use an edge observation point on the tone source instead.
* **gbf-test-14:** to make demodulation observable, use a different receiver scale, or a receive carrier offset through a Rotator. The Rotator keeps the cell in test. Do not add a noise source: that moves the cell to train.
* **gbf-test-18:** key the beacon with a square-wave Signal Source, which stays in test. Do not use a Vector Source: that moves the cell to dev, which is a split conflict.

## 5. Rules for every fixer (R1-R14 in the JSON)

1. **Split and cluster:** use the v2 cluster_id. Each build.py asserts the v2 split and prints the signatures of the reference and every variant. Fixers never edit the reserved-signature hand-off; a final step regenerates it.
2. **Instruction format (16 Oct):** the instruction is a request to GRC-Agent about the open flowgraph.
   * For build tasks, the open graph is empty apart from `samp_rate`, so the instruction states the rate to set.
   * Instructions never mention .grc or YAML files, benchmarks or oracles.
   * `start_grc` stays null for build tasks until the edit-format conversion stage.
3. **Oracle policy:**
   * Every stated requirement is checked, and its tolerance is stated in the instruction.
   * Thresholds come from the spec.
   * Where structure is the point, graph checks accept every legitimate block.
   * Every "keep X" clause is either checked or removed.
4. **Run length:** use the set sentence, and record it as a V5 liveness run condition.
5. **Family boundary:**
   * Modify is a structural or functional change.
   * Parameter-tune changes values only. The instruction says so, and `same_blocks_as_start` enforces it, plus the connection check from tooling T1 once it exists. Variants must obey the same rule.
6. **Level rule (proposed, for the team):**
   * L1: at most 5 functional blocks, single rate, no modulator+demodulator chain.
   * L2: 6-15 blocks, or any rate change, or modulation/demodulation.
   * L3: message ports, tags or PDUs.
7. **p2p meaning:** for repair starts that fail at V2 or V3, p2p means the properties of the parts of the graph the fault does not touch.
8. **Observation points:** each one names the file and the item type, and checks the type. Use a second observation point, or an edge point, wherever an output-only oracle could be satisfied by a shortcut.
9. **Mutants and variants:**
   * At least 5 mutants, all distinct, with correct "breaks" labels.
   * Near-miss mutants on the stated numbers.
   * A mutant for every "keep" clause.
   * Documented survivors stay in the set.
   * At least 2 literal variants, plus every alternative a reviewer showed.
10. **Records:**
    * Known limits go into review.notes.
    * Disjointness names the nearest training specs.
    * Seed-derived tasks record every conversion change and the GPL attribution.
11. **band_fraction:** re-calibrate every band_fraction check after the 18 Oct correction.
12. **Edits and gates:** fixers do not edit `src/` or `tests/`. They rebuild with build.py and re-run gates, and re-run gates again after tooling T7 for any task with several observation points.

## 6. Tooling to do first (one owner)

| id | file | need | for |
|---|---|---|---|
| T1 | graph_checks.py | connection check for value-only tasks; variable handling in `same_blocks_as_start` | all tune tasks |
| T2 | oracles.py | `corr_ref` with complex references and a signed (phase-sensitive) mode | test-17, test-19, test-08, test-20 |
| T3 | oracles.py | integer-lattice feature | dev-08 |
| T4 | oracles.py | `min` feature | test-09, dev-05, test-01, test-11 |
| T5 | oracles.py | `band_fraction` with `remove_mean` | test-04, test-01, dev-05 |
| T6 | ladder.py | CLI sets the task dir, so `same_blocks_as_start` works from `./gb ladder` | tune tasks |
| T7 | _worker.py | observation taps never back-pressure the graph; then re-gate every multi-point task | 13 tasks |
| T8 | seeds.py | message-port hints count as L3 | test-06 |
| T9 | scoring stage | the hallucinated-block rate excludes a fault's own wrong id | dev-08, test-10 |

## 7. Decisions for the team

1. **Level rule:** confirm the rule above, and in particular L3 for gbf-test-06.
2. **Family boundary:** confirm modify vs parameter-tune as defined above.
3. **Composed L2 items:** dev-06, test-13, test-14, test-18 and test-20 are hand-composed L2 graphs. The plans say L2 items come from converted examples, so either amend the plans or record the deviation. Dev has only L3 seeds left.
4. **Role table:** Claude also drafted the golden references and oracles, not just the instructions. This needs the role-table declaration the plan requires.
5. **Canary policy:** at present only task.json carries the canary.
6. **GPL attribution:** decide how seed-derived items keep it.
7. **Output-only limits:** accept or close the documented limits, e.g. hard-coded outputs in dev-07, test-05 and dev-01.
8. **dev-10 vs test-17:** both fault the Polyphase Clock Sync. This plan prefers moving dev-10's fault to another block.

## 8. Fixer groups (10 × 3)

| group | tasks | why together |
|---|---|---|
| G1 | test-04, dev-01, dev-03 | old twin group {tone}; two new dev digital items |
| G2 | test-03, dev-02, dev-05 | noise items: test-03 keeps its cell, plus two distinct dev noise themes |
| G3 | dev-04, test-05, test-10 | old pair dev-04/test-05; consistent L1 repair conventions with test-10 |
| G4 | test-01, test-08, test-12 | former square-to-sine test items, now three distinct L1 themes |
| G5 | test-06, test-07, test-11 | modify/tune boundary and the two relabels |
| G6 | test-13, test-20, test-19 | test L2 transmit/modulation items kept distinct |
| G7 | dev-06, test-18, test-14 | old twin dev-06/test-18; composed-L2 observability fixes |
| G8 | dev-10, test-17, test-02 | synchronisation receivers; same faulted block in dev-10 and test-17 |
| G9 | dev-08, test-09, test-15 | seed items with under-checked stated numbers; p2p for V2/V3 starts |
| G10 | dev-07, dev-09, test-16 | seed L2 items with many observation points; value-only rule; liveness |
