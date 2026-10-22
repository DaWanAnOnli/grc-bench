# Block partition for the training team

From the GRC-Bench evaluation team. Drafted by Claude (Anthropic) with `./gb blocks`; status: pending GRC-Bench team review. Not yet sent.

## What this is

GRC-Bench asks about 125 GNU Radio blocks (the *test* blocks) in its script-generated GNU Radio questions, which ask what a block does to a stated input (evaluation plan §4.2: "Training questions of this kind come from a different set of blocks"). To keep the benchmark clean, every GNU Radio block in GNU Radio 3.10.12.0 is marked `test`, `train` or excluded in `splits/block_partition.json` (field `split`; `null` means excluded).

## Rules for GNU Radio question expansion in training

1. **Use train-side blocks only.** A generated training question (and its answer) may be about a block only if that block's `split` is `train`. Test-side blocks must not be the subject of any training question, answer, or repair pair produced by the expansion.
2. **Excluded blocks** (`split` is `null`: GRC entries such as variables, deprecated blocks, and the uhd, soapy, iio, zeromq, network and audio modules) are outside the partition. Do not use them for the expansion unless the GRC-Bench team agrees.
3. **Whole families.** Blocks that differ only in type or version (field `family`, for example `blocks_multiply_const_xx` and `blocks_multiply_const_vxx`) are always on the same side.
4. **Different wording.** Write your own question wording and templates. Do not copy or imitate GRC-Bench's. You will not receive benchmark questions or answers; only this partition is shared.
5. **Scope.** This partition is for generated block Q&A: block properties (types, ports, parameters, rates, constraints) and block behaviour (what a block outputs for a given input). Flowgraph and agent training data follow their own split files, handed off separately; Qoherent's agent traces of 15-18 October are used in full (see START_HERE.md).

## How to check a generated set

List the block ids each generated item is about, then:

```python
import json
part = json.load(open('splits/block_partition.json'))
side = {b['id']: b['split'] for b in part['blocks']}
bad = [bid for bid in used_block_ids if side.get(bid) != 'train']   # must be empty
```

## Pins

- GNU Radio 3.10.12.0; catalog hash `ee88541933fc92b0061a3a6b900d7964b7cd17f61f01831b560e94d4bb08a277` (580 `.block.yml` files). If your GNU Radio install gives a different hash, tell us before generating.
- Seed `grc-bench-blocks-v1-2026-10-14`; rule: stratified-largest-remainder-sha256-v1 over block families; strata = module; test = round(0.3 * eligible families).
- Counts: 415 eligible blocks, 125 test, 290 train, 170 excluded.

## Train-side blocks you may use (290)

| Module | Blocks |
|---|---|
| analog | analog_agc2_xx, analog_agc3_xx, analog_agc_xx, analog_const_source_x, analog_dpll_bb, analog_fastnoise_source_x, analog_feedforward_agc_cc, analog_fm_demod_cf, analog_fm_preemph, analog_fmdet_cf, analog_frequency_modulator_fc, analog_nbfm_tx, analog_phase_modulator_fc, analog_pll_carriertracking_cc, analog_pll_freqdet_cf, analog_probe_avg_mag_sqrd_x, analog_pwr_squelch_xx, analog_quadrature_demod_cf, analog_random_source_x, analog_sig_source_x, analog_simple_squelch_cc, analog_standard_squelch, analog_wfm_rcv_pll, analog_wfm_tx |
| blocks | blocks_abs_xx, blocks_and_const_xx, blocks_and_xx, blocks_burst_tagger, blocks_burst_to_stream, blocks_char_to_float, blocks_char_to_short, blocks_complex_to_arg, blocks_complex_to_float, blocks_complex_to_imag, blocks_complex_to_interleaved_char, blocks_complex_to_interleaved_short, blocks_complex_to_mag, blocks_complex_to_mag_squared, blocks_conjugate_cc, blocks_correctiq_man, blocks_ctrlport_monitor, blocks_ctrlport_monitor_performance, blocks_ctrlport_probe2_c, blocks_ctrlport_probe2_x, blocks_ctrlport_probe_c, blocks_deinterleave, blocks_delay, blocks_endian_swap, blocks_exponentiate_const_cci, blocks_file_descriptor_sink, blocks_file_meta_sink, blocks_file_meta_source, blocks_file_sink, blocks_float_to_char, blocks_float_to_uchar, blocks_freqshift_cc, blocks_head, blocks_int_to_float, blocks_integrate_xx, blocks_interleave, blocks_interleaved_char_to_complex, blocks_interleaved_short_to_complex, blocks_magphase_to_complex, blocks_matrix_interleaver, blocks_message_debug, blocks_message_strobe, blocks_message_strobe_random, blocks_min_xx, blocks_moving_average_xx, blocks_msg_meta_to_pair, blocks_msgpair_to_var, blocks_multiply_conjugate_cc, blocks_multiply_const_vxx, blocks_multiply_const_xx, blocks_multiply_xx, blocks_mute_xx, blocks_nlog10_ff, blocks_null_source, blocks_pack_k_bits_bb, blocks_packed_to_unpacked_xx, blocks_peak_detector2_fb, blocks_plateau_detector_fb, blocks_probe_signal_vx, blocks_probe_signal_x, blocks_regenerate_bb, blocks_repack_bits_bb, blocks_repeat, blocks_rms_xx, blocks_rotator_cc, blocks_sample_and_hold_xx, blocks_selector, blocks_skiphead, blocks_stream_demux, blocks_stream_mux, blocks_stream_to_streams, blocks_stream_to_tagged_stream, blocks_stream_to_vector_decimator, blocks_streams_to_stream, blocks_swapiq, blocks_tag_debug, blocks_tag_share, blocks_tagged_file_sink, blocks_tagged_stream_align, blocks_tagged_stream_multiply_length, blocks_tagged_stream_mux, blocks_tags_strobe, blocks_test_tag_variable_rate_ff, blocks_threshold_ff, blocks_throttle2, blocks_uchar_to_float, blocks_unpack_k_bits_bb, blocks_var_to_msg, blocks_vco_c, blocks_vco_f, blocks_vector_insert_x, blocks_vector_sink_x, blocks_vector_source_x, blocks_xor_xx, xmlrpc_client, xmlrpc_server |
| channels | channels_cfo_model, channels_conj_fs_iqcorr, channels_distortion_2_gen, channels_dynamic_channel_model, channels_fading_model, channels_impairments, channels_iqbal_gen, channels_phase_bal, channels_phase_noise_gen, channels_quantizer |
| digital | digital_additive_scrambler_xx, digital_binary_slicer_fb, digital_burst_shaper_xx, digital_chunks_to_symbols_xx, digital_constellation_decoder_cb, digital_constellation_encoder_bc, digital_constellation_modulator, digital_constellation_soft_decoder_cf, digital_corr_est_cc, digital_cpmmod_bc, digital_crc_append, digital_crc_check, digital_diff_decoder_bb, digital_diff_encoder_bb, digital_diff_phasor_cc, digital_framer_sink_1, digital_gfsk_demod, digital_gfsk_mod, digital_glfsr_source_x, digital_gmsk_demod, digital_gmsk_mod, digital_header_payload_demux, digital_map_bb, digital_meas_evm_cc, digital_mpsk_snr_est_cc, digital_ofdm_cyclic_prefixer, digital_ofdm_frame_equalizer_vcvc, digital_ofdm_rx, digital_ofdm_tx, digital_packet_headergenerator_bb, digital_packet_headergenerator_bb_default, digital_packet_headerparser_b, digital_packet_headerparser_b_default, digital_packet_sink, digital_pfb_clock_sync_xxx, digital_probe_mpsk_snr_est_c, digital_protocol_formatter_async, digital_protocol_formatter_bb, digital_protocol_parser_b, digital_symbol_sync_xx |
| dtv | dtv_atsc_deinterleaver, dtv_atsc_depad, dtv_atsc_derandomizer, dtv_atsc_equalizer, dtv_atsc_fpll, dtv_atsc_fs_checker, dtv_atsc_interleaver, dtv_atsc_pad, dtv_atsc_randomizer, dtv_atsc_rs_decoder, dtv_atsc_rs_encoder, dtv_atsc_trellis_encoder, dtv_atsc_viterbi_decoder, dtv_catv_frame_sync_enc_bb, dtv_catv_randomizer_bb, dtv_dvb_bbheader_bb, dtv_dvb_bbscrambler_bb, dtv_dvb_bch_bb, dtv_dvbs2_interleaver_bb, dtv_dvbs2_modulator_bc, dtv_dvbt2_freqinterleaver_cc, dtv_dvbt2_interleaver_bb, dtv_dvbt2_miso_cc, dtv_dvbt2_modulator_bc, dtv_dvbt2_paprtr_cc, dtv_dvbt2_pilotgenerator_cc, dtv_dvbt_bit_inner_deinterleaver, dtv_dvbt_bit_inner_interleaver, dtv_dvbt_convolutional_deinterleaver, dtv_dvbt_demap, dtv_dvbt_demod_reference_signals, dtv_dvbt_energy_descramble, dtv_dvbt_energy_dispersal, dtv_dvbt_inner_coder, dtv_dvbt_ofdm_sym_acquisition, dtv_dvbt_reed_solomon_dec, dtv_dvbt_reference_signals, dtv_dvbt_symbol_inner_interleaver |
| fec | fec_async_decoder, fec_async_encoder, fec_bercurve_generator, fec_depuncture_bb, fec_extended_decoder, fec_extended_encoder, fec_extended_tagged_decoder, fec_extended_tagged_encoder, fec_generic_decoder, fec_generic_encoder, fec_tagged_decoder |
| fft | fft_ctrlport_probe_psd, fft_vxx, logpwrfft_x |
| filter | band_pass_filter, dc_blocker_xx, filter_delay_fc, filter_fft_low_pass_filter, filter_fft_rrc_filter, fir_filter_xxx, freq_xlating_fft_filter_ccc, high_pass_filter, hilbert_fc, interp_fir_filter_xxx, ival_decimator, low_pass_filter, mmse_resampler_xx, pfb_arb_resampler_xxx, pfb_decimator_ccf, pfb_interpolator_ccf, pfb_synthesizer_ccf, root_raised_cosine_filter |
| pdu | pdu_add_system_time, pdu_pdu_filter, pdu_pdu_lambda, pdu_pdu_remove, pdu_pdu_to_stream_x, pdu_pdu_to_tagged_stream, pdu_tagged_stream_to_pdu, pdu_take_skip_to_pdu, pdu_time_delta |
| qtgui | qtgui_bercurve_sink, qtgui_compass, qtgui_eye_sink_x, qtgui_freq_sink_x, qtgui_graphicitem, qtgui_graphicoverlay, qtgui_grbackground, qtgui_histogram_sink_x, qtgui_ledindicator, qtgui_levelgauge, qtgui_matrix_sink, qtgui_msgdigitalnumbercontrol, qtgui_number_sink, qtgui_time_raster_sink_x, qtgui_time_sink_x, qtgui_vector_sink_f |
| trellis | trellis_encoder_xx, trellis_metrics_x, trellis_pccc_decoder_combined_xx, trellis_pccc_encoder_xx, trellis_permutation, trellis_sccc_encoder_xx, trellis_siso_combined_f, trellis_viterbi_combined_xx, trellis_viterbi_x |
| video | video_sdl_sink |
| vocoder | vocoder_alaw_decode_bs, vocoder_codec2_decode_ps, vocoder_codec2_encode_sp, vocoder_cvsd_decode_bf, vocoder_cvsd_decode_bs, vocoder_cvsd_encode_fb, vocoder_cvsd_encode_sb, vocoder_freedv_rx_ss, vocoder_g721_encode_sb, vocoder_g723_24_decode_bs, vocoder_g723_24_encode_sb, vocoder_g723_40_decode_bs, vocoder_g723_40_encode_sb, vocoder_gsm_fr_encode_sp, vocoder_ulaw_decode_bs |
