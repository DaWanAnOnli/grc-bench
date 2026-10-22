# Tier F: composed-chain signature v2 (functional signature) and the re-split

Drafted by Claude (Anthropic), 18 Oct 2026, for team review. Status: **pending team review**. Code: `src/grcbench/tier_f/compose.py` (`signature`, `signature_split`, `functional_profile`, `audit_items`); tests: `tests/test_tier_f_signature.py`. Machine-readable version of every table: `reports/tier_f_signature_v2.json` (regenerate the table data with `python -c "from grcbench.tier_f import compose; import json, pathlib; print(json.dumps(compose.audit_items(pathlib.Path('.'))))"` from the repo root with `PYTHONPATH=src`).

## 1. Why v1 had to go

Signature v1 was the ordered tuple of block ids along one source-to-sink walk. The round-1 reviews (gbf-test-01, -02, -03, -04, gbf-dev-06) showed three ways to move the same functional task to another split: (a) insert an inert block such as a Throttle; (b) swap commuting blocks (Add Const / Multiply Const); (c) use a rare but equivalent block (Multiply Conjugate vs Rotator vs Multiply by a second complex exponential). Several first-set tasks reached test or dev only that way, a training-side generator obeying `signature_split == "train"` could legally emit functional twins of test items, and template twins received different cluster_ids. v1 also followed only one branch of an L2 graph.

The same cases under v1 and v2 (pairs of graphs that do the same job):

| case | v1 splits | v2 signature | v2 splits |
|---|---|---|---|
| gbf-test-04: inert Throttle inserted | test / train | tone | test / test |
| gbf-test-04: Add Const and Multiply Const swapped | test / train | tone | test / test |
| gbf-test-03 / gbf-dev-02: Throttle before or after the band-pass | test / dev | filter&#124;noise | test / test |
| gbf-test-03: Fast Noise Source instead of Noise Source | test / train | filter&#124;noise | test / test |
| gbf-test-02: Multiply Conjugate vs Multiply with a negative LO | test / dev | mix&#124;tone | train / train |
| gbf-test-02: Multiply Conjugate vs Rotator | test / train | mix&#124;tone | train / train |
| gbf-test-02: Multiply Conjugate vs Frequency Xlating FIR (taps [1]) | test / train | mix&#124;tone | train / train |
| gbf-test-02: Multiply Conjugate vs Conjugate + Multiply + Throttle | test / train | mix&#124;tone | train / train |
| gbf-dev-06 (L2, two branches): Throttle removed | dev / train | combine&#124;filter&#124;mix&#124;resample&#124;tone | test / test |

## 2. The rule

* `signature(graph)` = the sorted **set** of functional classes of all enabled blocks (no path walk, so branching L2 graphs work and block order is irrelevant).
* `signature_split(sig)` = sha256(`grc-bench-l1-signatures-v2-2026-10-18` + NUL + `"|".join(sig)`) mapped onto test 0.45, dev 0.20, train 0.35 (shares unchanged; see section 5).
* A composed task's split must equal `signature_split(signature(reference))`; its cluster_id is `"chain:" + "|".join(signature(reference))`. Seed-derived tasks keep the seed-file split.
* Training-side generators may only emit composed graphs whose v2 split is `train` **and** whose signature is not reserved by a test/dev task (section 7).
* `signature_v1` / `signature_split_v1` (seed `grc-bench-l1-signatures-v1-2026-10-14`) are kept for provenance only.

## 3. The class map

Every enabled block is mapped to zero or more fine classes (`functional_profile`), then fine classes are coarsened to the v2 classes. The module docstring of `compose.py` is the normative text; summary:

| v2 class | blocks (GNU Radio 3.10 ids, incl. legacy type variants) |
|---|---|
| *(dropped: plumbing)* | throttle/throttle2, head, skiphead, copy, selector, valve, every sink (file, null, vector, wav, audio, GUI `qtgui_*`, network), null source, tag debug/gate/share, probes, message debug, stream/vector reshaping, bit (un)packing, lossless type conversions (complex to real/imag/float, float to complex, integer to float, char to short ...) |
| *(dropped: absorbed)* | constant scaling: Multiply Const (all types), Multiply by a constant source, single-tap filters `[g]`; constant offset: Add Const (all types), Add of a constant source. A source's amplitude/offset or a filter's gain does the same, and accepted variants move the gain freely. |
| tone | Signal Source, cosine or sine waveform, any type/amplitude/offset (float and complex+Complex-to-Real are the same) |
| harmonic_wave | Signal Source, square/triangle/sawtooth waveform |
| noise | Noise Source, Fast Noise Source |
| data_source | Vector Source, Random Source, File Source, Random Uniform, GLFSR, constant source used as data |
| message_source | Message Strobe (random), Tags Strobe, Random PDU |
| mix | Rotator; Multiply / Multiply Conjugate with a pure-generator input (that local oscillator, incl. conjugated or scaled ones, is absorbed); Frequency Xlating FIR/FFT with a non-zero centre; channelizers/synthesizers; PFB decimator on a non-zero channel |
| combine | Add / Subtract of two or more non-constant streams |
| multiply | Multiply of two non-generator streams |
| conjugate | Conjugate of a signal (not of an oscillator) |
| filter | Low Pass, High Pass, Band Pass, Band Reject, DC Blocker, Moving Average, IIR, Single-Pole IIR, FM de-/pre-emphasis, Integrate; FIR / FFT / interpolating FIR / xlating / resampler taps from firdes or optfir low_pass(_2), high_pass(_2), band_pass(_2), complex_band_pass(_2), band_reject(_2), window, a taps variable block (LPF/HPF/BPF/BRF taps), a literal list of >= 2 taps; default taps of resamplers |
| pulse_shaping | RRC Filter, RRC taps variable, firdes.root_raised_cosine / gaussian taps; inside Constellation/PSK/QAM/GMSK/GFSK modulators; matched filter of Polyphase Clock Sync and of Symbol Sync with a PFB-MF interpolator |
| hilbert | Hilbert block, firdes.hilbert taps |
| resample | any rate change, wherever it sits: Keep 1 in N, Keep M in N, Repeat, decim/interp of any filter, Rational / Polyphase Arbitrary / Fractional resamplers, PFB decimator/interpolator, channelizers, FM/AM/NBFM/WBFM demod audio decimation, modulators' samples per symbol. Rate parameters are evaluated with the graph's variables; ratio 1 is no rate change; an unknown ratio counts as one |
| fm_mod / fm_demod | Frequency Modulator, Phase Modulator, VCO, NBFM/WBFM Tx / Quadrature Demod, FM Demod, NBFM/WBFM Receive (composites also add filter + resample) |
| envelope / phase | Complex to Mag, Mag^2, Abs, AM Demod / Complex to Arg |
| linear_mod / linear_demod | Chunks to Symbols, Map, Constellation/PSK/QAM Modulator / Constellation Decoder, Receiver, Soft Decoder, Binary Slicer |
| fsk_mod / fsk_demod | GMSK, GFSK, CPM modulators / GMSK, GFSK demodulators |
| quantize | Quantizer (channels), float to char/short/int/uchar (they round onto an integer grid) |
| carrier_sync, timing_sync, frame_sync, equalizer, channel_model, agc, squelch, power_measure, log, divide, peak_detect, delay, mux, demux, nonlinear, fft, fec, custom | Costas, FLL band-edge, PLLs; Clock Recovery MM, Polyphase Clock Sync, Symbol Sync; Corr Est, Correlate Access Code; linear/CMA/LMS/DFE equalizers; channel/impairment/fading/CFO/SRO models; AGC 1-3, feed-forward AGC; power/simple squelch; RMS, average-power probe; Log10; Divide; Peak Detector; Delay; Stream Mux, Patterned Interleaver, Interleave; Stream Demux, Deinterleave; Rail, Transcendental, Min/Max, Threshold; FFT; `fec_*`; embedded Python block |
| other:&lt;id&gt; | fallback for an id not in the map (the signature stays defined; extend the map when needed) |

Local-oscillator rule: an input of a Multiply is a *pure generator* when everything upstream of it is a signal or constant source seen through plumbing, gains, offsets, conjugation, or sums/products of generators. If at most one input is non-constant the Multiply is a gain (Add: an offset). Otherwise, if one input is a pure tone generator the block is a mixer, and the oscillator input (preferring an input that feeds only this mixer, then the one with fewer sources, then the higher port) is absorbed. Composite blocks expand into their parts (Frequency Xlating = mix + filter + resample; FM Demod = fm_demod + filter + resample; Constellation Modulator = linear_mod + pulse_shaping + resample; channelizer = mix + filter + resample), which is why a composite and its hand-built equivalent get one signature.

Every block id used in `items/tier_f/**/reference.grc` and `variants/*.grc` is covered by an explicit rule (`compose.block_rule`), enforced by `tests/test_tier_f_signature.py`. The only `other:` ids in the item set are the two nonexistent-block faults in repair start graphs (`filter_high_pass_filter`, `blocks_quantizer_ff`).

## 4. Why these merges (numbers)

The brief for this change suggested separate classes for constant scaling, constant offset, low-pass, high-pass, band-pass/band-reject and a multiset. The accepted variants of the 20 composed tasks (each is a correct, gate-checked alternative solution of its task) show that this is too fine: the same task is solved with the gain in the filter, a notch or band-pass instead of a high-pass, an HPF+LPF cascade instead of a band-pass, two decimation stages instead of one. A variant whose split differs from its reference's is exactly the loophole being closed.

| signature definition | composed-task variants | signature differs from reference | split differs from reference | distinct reference cells (of 20 tasks) |
|---|---|---|---|---|
| v1: ordered block ids along one walk | 65 | 51 | 38 | 19 |
| fine classes, multiset (scale/offset kept, LP/HP/BP/BR and decim/interp apart) | 65 | 20 | 10 | 15 |
| fine classes, set | 65 | 19 | 14 | 15 |
| v2 but scale/offset kept as classes | 65 | 15 | 6 | 14 |
| v2 but filter responses kept apart | 65 | 6 | 5 | 11 |
| v2 but decimation/interpolation kept apart | 65 | 2 | 2 | 9 |
| v2 as a multiset (counts kept) | 65 | 4 | 3 | 11 |
| **v2 (adopted)** | 65 | 2 | 2 | 9 |

The two variants that v2 still separates from their reference are structural alternatives, not class-map gaps; merging them would merge functionally different tasks:

* gbf-dev-02 `v2_lowpass_then_frequency_shift.grc`: filter&#124;mix&#124;noise (dev) vs reference filter&#124;noise (test). It builds the band-limited noise as low-pass noise shifted up by a mixer. Folding mix+filter into filter would also fold every down-converter (gbf-dev-06, gbf-test-18) into plain filtering, so the map keeps them apart. The variant's signature hashes to dev (not train-emittable) and is on the reserved list.
* gbf-test-12 `v2_cosine_source_no_filter.grc`: tone (test) vs reference filter&#124;harmonic_wave (train). It drops the filter and switches the source to a cosine. The oracle accepts a plain tone, so the item as phrased is functionally also a tone-generation task (the item-quality reviews found the same shortcut in gbf-test-01). That is an item issue for the fix stage, not a class-map gap; the {tone} cell hashes to test and is reserved.

Decimation and interpolation are merged into `resample` although no current variant needs it: the function is the net rate change, and the same net rate can be built as one decimator or as an interpolate-then-decimate pair (rational resampler, interpolating front end).

Multiplicity is dropped because counts are template numbers: with counts, "sum of 2 tones" and "sum of 4 tones", one channel or two, one decimation stage or two would be different cells, and a generator could leave a held-out cell by adding a tone or a stage (gbf-dev-06 v2 and gbf-test-16 v2 differ from their references only in counts).

## 5. Is the class space too coarse for the shares?

Shares are kept (test 0.45, dev 0.20, train 0.35). Over a plausible composed vocabulary (4 source classes, 16 processing classes; 1-2 source classes and 0-3 processing classes) there are 6970 distinct v2 signatures; they split test 3165 (45.4%), dev 1345 (19.3%), train 2460 (35.3%). On 20,000 arbitrary signatures the hash gives test 45.2%, dev 20.3%, train 34.5%. The space is large enough for the shares. What *is* coarse is the handful of simplest cells (one source class with at most one processing class: 68 cells, split test 24, dev 21, train 23): every task built on such a cell lands in one split, so authors must vary the function, not the plumbing, to place a task. That is the intended effect, so no share adjustment is proposed.

## 6. Re-split table (every Tier-F task)

`rule split` is the split the rules assign: v2 split of the reference for composed tasks, the seed-file split for seed-derived tasks. **conflict** = rule split differs from the task's current split. Variant column: number of variants with the same v2 signature / total.

| task | split now | family / level | source | v1 split | v2 signature | v2 split | rule split | conflict | variants same v2 |
|---|---|---|---|---|---|---|---|---|---|
| gbf-dev-01 | dev | build L1 | composed | dev | tone | test | test | **CONFLICT** | 4/4 |
| gbf-dev-02 | dev | build L1 | composed | dev | filter&#124;noise | test | test | **CONFLICT** | 2/3 |
| gbf-dev-03 | dev | modify L1 | composed | dev | tone | test | test | **CONFLICT** | 3/3 |
| gbf-dev-04 | dev | repair L1 | composed | dev | combine&#124;filter&#124;tone | train | train | **CONFLICT** | 4/4 |
| gbf-dev-05 | dev | parameter-tune L1 | composed | dev | filter&#124;harmonic_wave | train | train | **CONFLICT** | 3/3 |
| gbf-dev-06 | dev | build L2 | composed | dev | combine&#124;filter&#124;mix&#124;resample&#124;tone | test | test | **CONFLICT** | 3/3 |
| gbf-dev-07 | dev | modify L2 | blocks/stream_mux_demo.grc | train | data_source&#124;mux | test | dev | ok | 4/4 |
| gbf-dev-08 | dev | repair L2 | channels/demo_quantization.grc | dev | combine&#124;filter&#124;noise&#124;quantize&#124;tone | test | dev | ok | 4/4 |
| gbf-dev-09 | dev | parameter-tune L2 | filter/filter_taps.grc | train | filter&#124;noise&#124;pulse_shaping | test | dev | ok | 3/3 |
| gbf-dev-10 | dev | repair L2 | digital/demod/test_corr_est.grc | train | carrier_sync&#124;channel_model&#124;data_source&#124;frame_sync&#124;linear_mod&#124;mux&#124;pulse_shaping&#124;resample&#124;timing_sync | test | dev | ok | 3/3 |
| gbf-test-01 | test | build L1 | composed | test | filter&#124;harmonic_wave | train | train | **CONFLICT** | 3/3 |
| gbf-test-02 | test | build L1 | composed | test | mix&#124;tone | train | train | **CONFLICT** | 3/3 |
| gbf-test-03 | test | build L1 | composed | test | filter&#124;noise | test | test | ok | 3/3 |
| gbf-test-04 | test | build L1 | composed | test | tone | test | test | ok | 3/3 |
| gbf-test-05 | test | build L1 | composed | test | combine&#124;filter&#124;tone | train | train | **CONFLICT** | 3/3 |
| gbf-test-06 | test | modify L1 | analog/sig_source_msg_ports.grc | train | message_source&#124;tone | dev | test | ok | 3/3 |
| gbf-test-07 | test | modify L1 | composed | test | filter&#124;harmonic_wave | train | train | **CONFLICT** | 3/3 |
| gbf-test-08 | test | repair L1 | composed | test | filter&#124;harmonic_wave | train | train | **CONFLICT** | 3/3 |
| gbf-test-09 | test | repair L1 | blocks/peak_detector2.grc | test | harmonic_wave&#124;peak_detect | test | test | ok | 3/3 |
| gbf-test-10 | test | repair L1 | composed | test | filter&#124;tone | test | test | ok | 4/4 |
| gbf-test-11 | test | parameter-tune L1 | composed | test | filter&#124;tone | test | test | ok | 3/3 |
| gbf-test-12 | test | parameter-tune L1 | composed | test | filter&#124;harmonic_wave | train | train | **CONFLICT** | 2/3 |
| gbf-test-13 | test | build L2 | composed | test | combine&#124;filter&#124;resample&#124;tone | train | train | **CONFLICT** | 3/3 |
| gbf-test-14 | test | build L2 | composed | test | filter&#124;fm_demod&#124;fm_mod&#124;resample&#124;tone | test | test | ok | 3/3 |
| gbf-test-15 | test | modify L2 | channels/demo_spec_an.grc | train | combine&#124;divide&#124;filter&#124;log&#124;noise&#124;power_measure&#124;resample&#124;tone | train | test | ok | 4/4 |
| gbf-test-16 | test | modify L2 | filter/polyphase_channelizer_demo.grc | test | combine&#124;filter&#124;mix&#124;noise&#124;resample&#124;tone | train | test | ok | 3/3 |
| gbf-test-17 | test | repair L2 | digital/demod/pam_timing.grc | train | channel_model&#124;data_source&#124;linear_mod&#124;pulse_shaping&#124;resample&#124;timing_sync | test | test | ok | 4/4 |
| gbf-test-18 | test | repair L2 | composed | test | combine&#124;filter&#124;mix&#124;resample&#124;tone | test | test | ok | 3/3 |
| gbf-test-19 | test | parameter-tune L2 | channels/demo_gmsk.grc | train | channel_model&#124;data_source&#124;fm_demod&#124;fsk_mod&#124;pulse_shaping&#124;resample | test | test | ok | 3/3 |
| gbf-test-20 | test | parameter-tune L2 | composed | test | combine&#124;filter&#124;resample&#124;tone | train | train | **CONFLICT** | 5/5 |

Composed tasks: 20; conflicts: 14. Under v2 the composed references fall in test 10, dev 0, train 10. No composed reference hashes to dev, so every composed dev task conflicts, and the ten tasks whose cell is `train` cannot stay in GRC-Bench as they are (re-theme to a different function, or replace). Seed-derived tasks have no conflict (their split comes from the seed file), but their functional signatures are reserved (section 7): gbf-test-15 and gbf-test-16 hash to train, so without the reserved list a training generator could emit their functional twins.

New cluster_id for each composed task (`chain:` + v2 signature):

* gbf-dev-01: `chain:tone` (was `chain:analog_sig_source_x|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-dev-02: `chain:filter|noise` (was `chain:analog_noise_source_x|band_pass_filter|blocks_throttle2|blocks_file_sink`)
* gbf-dev-03: `chain:tone` (was `chain:analog_sig_source_x|blocks_add_const_vxx|blocks_multiply_const_vxx|blocks_throttle2|blocks_file_sink`)
* gbf-dev-04: `chain:combine|filter|tone` (was `chain:analog_sig_source_x|blocks_add_xx|high_pass_filter|blocks_file_sink`)
* gbf-dev-05: `chain:filter|harmonic_wave` (was `chain:analog_sig_source_x|band_pass_filter|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-dev-06: `chain:combine|filter|mix|resample|tone` (was `chain:analog_sig_source_x|blocks_add_xx|blocks_throttle2|freq_xlating_fir_filter_xxx|blocks_file_sink`)
* gbf-test-01: `chain:filter|harmonic_wave` (was `chain:analog_sig_source_x|blocks_multiply_const_vxx|low_pass_filter|blocks_file_sink`)
* gbf-test-02: `chain:mix|tone` (was `chain:analog_sig_source_x|blocks_multiply_conjugate_cc|blocks_file_sink`)
* gbf-test-03: `chain:filter|noise` (was `chain:analog_noise_source_x|blocks_throttle2|band_pass_filter|blocks_file_sink`)
* gbf-test-04: `chain:tone` (was `chain:analog_sig_source_x|blocks_multiply_const_vxx|blocks_add_const_vxx|blocks_file_sink`)
* gbf-test-05: `chain:combine|filter|tone` (was `chain:analog_sig_source_x|blocks_add_xx|fft_filter_xxx|blocks_file_sink`)
* gbf-test-07: `chain:filter|harmonic_wave` (was `chain:analog_sig_source_x|blocks_multiply_const_vxx|low_pass_filter|blocks_add_const_vxx|blocks_file_sink`)
* gbf-test-08: `chain:filter|harmonic_wave` (was `chain:analog_sig_source_x|blocks_add_const_vxx|low_pass_filter|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-test-10: `chain:filter|tone` (was `chain:analog_sig_source_x|high_pass_filter|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-test-11: `chain:filter|tone` (was `chain:analog_sig_source_x|blocks_multiply_const_vxx|low_pass_filter|blocks_add_const_vxx|blocks_file_sink`)
* gbf-test-12: `chain:filter|harmonic_wave` (was `chain:analog_sig_source_x|low_pass_filter|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-test-13: `chain:combine|filter|resample|tone` (was `chain:analog_sig_source_x|blocks_add_xx|low_pass_filter|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-test-14: `chain:filter|fm_demod|fm_mod|resample|tone` (was `chain:analog_sig_source_x|rational_resampler_xxx|analog_frequency_modulator_fc|analog_quadrature_demod_cf|low_pass_filter|blocks_file_sink`)
* gbf-test-18: `chain:combine|filter|mix|resample|tone` (was `chain:analog_sig_source_x|blocks_add_xx|blocks_multiply_xx|low_pass_filter|blocks_multiply_xx|blocks_complex_to_real|blocks_multiply_const_vxx|blocks_file_sink`)
* gbf-test-20: `chain:combine|filter|resample|tone` (was `chain:analog_sig_source_x|blocks_add_xx|low_pass_filter|low_pass_filter|blocks_file_sink`)

## 7. Twin groups

Tasks (any split) whose references share a v2 signature. A group that spans test and dev is a cross-split template twin; any group is one cluster for clustered standard errors.

| v2 signature | cell split | tasks | splits now |
|---|---|---|---|
| combine&#124;filter&#124;mix&#124;resample&#124;tone | test | gbf-dev-06, gbf-test-18 | dev, test |
| combine&#124;filter&#124;resample&#124;tone | train | gbf-test-13, gbf-test-20 | test |
| combine&#124;filter&#124;tone | train | gbf-dev-04, gbf-test-05 | dev, test |
| filter&#124;harmonic_wave | train | gbf-dev-05, gbf-test-01, gbf-test-07, gbf-test-08, gbf-test-12 | dev, test |
| filter&#124;noise | test | gbf-dev-02, gbf-test-03 | dev, test |
| filter&#124;tone | test | gbf-test-10, gbf-test-11 | test |
| tone | test | gbf-dev-01, gbf-dev-03, gbf-test-04 | dev, test |

Seed-derived tasks whose functional signature equals a composed task's: none.

Links through accepted alternative solutions (a variant of one task has the signature of another task's reference or variant), beyond the groups above:

* tone: gbf-dev-01, gbf-dev-03, gbf-test-04, gbf-test-12

### Reserved signatures

The v2 signatures of every test/dev task's reference and accepted variants are reserved: 20 signatures, 6 of which hash to train (these are the ones the split rule alone would not protect). They are handed to the training team as opaque digests (`compose.signature_digest`) in `handoff/training_team/flowgraph_reserved_signatures.txt`; regenerate that file whenever a test/dev task is added, re-themed or re-split.

| v2 signature | split | reserved by |
|---|---|---|
| carrier_sync&#124;channel_model&#124;data_source&#124;frame_sync&#124;linear_mod&#124;mux&#124;pulse_shaping&#124;resample&#124;timing_sync | test | gbf-dev-10 |
| channel_model&#124;data_source&#124;fm_demod&#124;fsk_mod&#124;pulse_shaping&#124;resample | test | gbf-test-19 |
| channel_model&#124;data_source&#124;linear_mod&#124;pulse_shaping&#124;resample&#124;timing_sync | test | gbf-test-17 |
| combine&#124;divide&#124;filter&#124;log&#124;noise&#124;power_measure&#124;resample&#124;tone | train | gbf-test-15 |
| combine&#124;filter&#124;mix&#124;noise&#124;resample&#124;tone | train | gbf-test-16 |
| combine&#124;filter&#124;mix&#124;resample&#124;tone | test | gbf-dev-06, gbf-test-18 |
| combine&#124;filter&#124;noise&#124;quantize&#124;tone | test | gbf-dev-08 |
| combine&#124;filter&#124;resample&#124;tone | train | gbf-test-13, gbf-test-20 |
| combine&#124;filter&#124;tone | train | gbf-dev-04, gbf-test-05 |
| data_source&#124;mux | test | gbf-dev-07 |
| filter&#124;fm_demod&#124;fm_mod&#124;resample&#124;tone | test | gbf-test-14 |
| filter&#124;harmonic_wave | train | gbf-dev-05, gbf-test-01, gbf-test-07, gbf-test-08, gbf-test-12 |
| filter&#124;mix&#124;noise | dev | gbf-dev-02 |
| filter&#124;noise | test | gbf-dev-02, gbf-test-03 |
| filter&#124;noise&#124;pulse_shaping | test | gbf-dev-09 |
| filter&#124;tone | test | gbf-test-10, gbf-test-11 |
| harmonic_wave&#124;peak_detect | test | gbf-test-09 |
| message_source&#124;tone | dev | gbf-test-06 |
| mix&#124;tone | train | gbf-test-02 |
| tone | test | gbf-dev-01, gbf-dev-03, gbf-test-04, gbf-test-12 |

## 8. Consequences for the fix stage (not done here; task directories untouched)

* Every composed task's `build.py` asserts `compose.signature_split(compose.signature(ref)) == SPLIT`; after this change the 14 conflicting builds stop with that assertion until the task is re-split or re-themed. Their `cluster_id` should become the v2 cluster id above.
* Within a twin group, the team decides which item keeps the cell; the others need a different function (not a different block arrangement, which v2 ignores).
* Variants should be checked with `compose.signature` too: a variant with a different signature is fine if it is a structural alternative, but its signature is then reserved for the task.
* Seed-derived tasks: the v2 split column above is informational (their split comes from the seed file); their signatures are reserved.

Where the simplest cells fall (one source class plus at most one processing class), to help pick replacement themes; larger cells are spread by the hash as in section 5:

| processing class | tone | harmonic_wave | noise | data_source |
|---|---|---|---|---|
| (none) | test | train | dev | train |
| filter | test | train | test | dev |
| mix | train | test | dev | test |
| combine | test | train | test | dev |
| multiply | train | dev | train | test |
| resample | dev | test | test | train |
| fm_mod | train | test | dev | test |
| fm_demod | dev | train | dev | dev |
| envelope | test | test | dev | dev |
| quantize | test | train | dev | dev |
| delay | test | test | train | dev |
| hilbert | train | test | train | dev |
| conjugate | dev | train | dev | train |
| nonlinear | train | test | dev | test |
| agc | train | dev | dev | test |
| pulse_shaping | test | train | test | train |
| linear_mod | test | train | train | train |
