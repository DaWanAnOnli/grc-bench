# Flowgraph seed split (for the training team)

GRC-Bench Tier-F and Tier-A tasks are built from the GNU Radio example flowgraphs listed as
test or dev below. Training flowgraph tasks and programmatic agent sessions (proposal §4.4) must
come from **other** seed files: the train seeds below, permissively licensed community flowgraphs,
or hand-written chains. Never derive training items from a test or dev seed file, including by
editing or fault-injecting it. Full detail: splits/flowgraph_seed_split.json, reports/seed_inventory.md.

## Held out for GRC-Bench (31 files): do not use

- analog/sig_source_msg_ports.grc
- blocks/burst_to_stream.grc
- blocks/matrix_multiplexer.grc
- blocks/peak_detector2.grc
- blocks/stream_mux_demo.grc
- channels/demo_gmsk.grc
- channels/demo_quantization.grc
- channels/demo_spec_an.grc
- digital/demod/constellation_soft_decoder.grc
- digital/demod/pam_timing.grc
- digital/demod/test_corr_est.grc
- digital/equalizers/le_vs_dfe.grc
- digital/packet/example_corr_est.grc
- digital/packet/simple_bpsk_tx.grc
- digital/packet/tx_stage4.grc
- fec/fecapi_async_decoders.grc
- fec/fecapi_async_ldpc_decoders.grc
- fec/fecapi_async_packed_decoders.grc
- fec/fecapi_cc_decoders.grc
- fec/fecapi_decoders.grc
- fec/fecapi_polar_decoders.grc
- fec/fecapi_polar_encoders.grc
- fec/polar_code_example.grc
- filter/filter_taps.grc
- filter/polyphase_channelizer_demo.grc
- pdu/pdu_lambda_chirp_demo.grc
- qt-gui/qtgui_multi_input.grc
- qt-gui/qtgui_tags_viewing.grc
- qt-gui/test_compass_stream.grc
- trellis/sccc.grc
- trellis/tcm.grc

## Train seeds (32 files)

- analog/noise_power.grc
- audio/dial_tone.grc
- blocks/matrix_interleaver.grc
- blocks/stream_demux_demo.grc
- blocks/vector_source_with_tags.grc
- channels/channel_tone_response.grc
- channels/demo_ofdm.grc
- channels/demo_qam.grc
- ctrlport/pfb_sync_test.grc
- digital/demod/symbol_sync_test_complex.grc
- digital/demod/symbol_sync_test_float.grc
- digital/equalizers/linear_equalizer_compare.grc
- digital/mpsk_stage6.grc
- digital/ofdm/tx_ofdm.grc
- digital/packet/example_corr_est_and_clock_sync.grc
- digital/packet/example_corr_est_and_phase_sync.grc
- fec/ber_test.grc
- fec/fecapi_async_encoders.grc
- fec/fecapi_async_ldpc_encoders.grc
- fec/fecapi_encoders.grc
- fec/fecapi_polar_async_packed_decoders.grc
- fec/fecapi_tagged_decoders.grc
- fec/fecapi_tagged_encoders.grc
- filter/filter_taps_loader.grc
- filter/resampler_demo.grc
- pdu/simple_pdu_to_stream_example.grc
- qt-gui/qtgui_vector_sink_example.grc
- qt-gui/test_autocorrelator.grc
- qt-gui/test_qtgui_msg.grc
- tags/test_tag_prop.grc
- trellis/pccc.grc
- trellis/viterbi_equalization.grc
