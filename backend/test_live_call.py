"""
Live Full-Duplex Multilingual Voice Call Translation Tool.
Connects ALL 8 PHASES end-to-end:
1. Live Microphone Capture (Zero disk writes, 16kHz float32 in RAM)
2. Streaming Silero VAD (Real-time speech onset & offset detection)
3. Spoken Acoustic Language Identification (Whisper INT8 acoustic logits + unbiasing)
4. Streaming Speech-to-Text ASR (Faster-Whisper INT8 with anti-hallucination penalties)
5. Contextual Neural Translation (Named entity masking + Google/MyMemory NMT)
6. Multilingual Neural Speech Synthesis (Edge-TTS across Indian & global languages)
7. Voice Prosody Adaptation (Pitch & tempo matching)
8. Speaker Audio Playback with Acoustic Echo Feedback Prevention (Auto-mutes mic during playback)
"""

import sys
import time
import argparse
import numpy as np

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker
from app.services.asr_service import StreamingAsrService
from app.services.translation_service import TranslationService
from app.services.tts_service import StreamingTtsService
from app.services.voice_cloner import VoiceCloningService
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code


def print_banner():
    print("\n" + "=" * 74)
    print("  LIVE FULL-DUPLEX VOICE CALL TRANSLATOR (END-TO-END PIPELINE)")
    print("  Zero Disk Writes • Real-Time VAD • LID • ASR • NMT • TTS • Playback")
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(description="Live End-to-End Voice Call Translator")
    parser.add_argument("--source", type=str, default="auto", help="Speaker source language (default: auto for 107-lang detection, or e.g. te, hi, ta, en)")
    parser.add_argument("--target", type=str, default="en", help="Language code or name for receiver (default: en / English)")
    parser.add_argument("--gender", type=str, default="female", choices=["female", "male"], help="TTS Voice gender")
    parser.add_argument("--echo", action="store_true", help="Mirror/Echo mode: speaks back exact spoken words in original language (skips translation)")
    parser.add_argument("--duration", type=float, default=None, help="Stop listening after N seconds (default: runs continuously until Ctrl+C)")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech playback (text subtitles only)")
    parser.add_argument("--text", action="store_true", help="Interactive text input mode instead of live microphone")
    parser.add_argument("--model-size", type=str, default="base", choices=["tiny", "base", "small"], help="Whisper fallback model size")
    args = parser.parse_args()

    print_banner()

    target_lang = normalize_language_code(args.target, default="en")
    target_info = SUPPORTED_LANGUAGES.get(target_lang, LanguageInfo(target_lang, target_lang.title(), target_lang))

    source_mode = args.source.strip().lower()
    forced_source = None if source_mode == "auto" else normalize_language_code(source_mode, default="te")
    source_name = "Auto-Detect (107 Languages via VoxLingua107 ECAPA)" if forced_source is None else SUPPORTED_LANGUAGES.get(forced_source, LanguageInfo(forced_source, forced_source.title(), forced_source)).name

    print(f"\n[Configuration]")
    print(f"  • Speaker Source Language  : {source_name} ({'auto' if forced_source is None else forced_source})")
    if args.echo:
        print(f"  • Operating Mode           : 🔁 VOICE ECHO / MIRROR (Repeats what you speak in same language)")
    else:
        print(f"  • Receiver Target Language : {target_info.name} ({target_lang})")
    print(f"  • Audio Playback           : {'DISABLED (--no-tts)' if args.no_tts else 'ENABLED (🔊 Live Speakers)'}")
    print(f"  • Voice Gender             : {args.gender.title()}")
    print(f"  • LID + ASR Engine         : VoxLingua107 ECAPA (107 langs) + Native Neural STT (Whisper offline fallback)")
    print(f"  • Acoustic Echo Prevention : ACTIVE (Mic auto-muted during synthetic speech playback)")
    if args.duration:
        print(f"  • Max Duration             : {args.duration:.0f} seconds")
    print("-" * 74)

    # 1. Initialize Acoustic Models (VoxLingua107 ECAPA + Shared Faster-Whisper fallback)
    print("\n[1/4] Loading VoxLingua107 ECAPA LID (107 languages) & Whisper fallback...")
    LanguageDetectionEngine.get_ecapa_model()
    from faster_whisper import WhisperModel
    shared_whisper = WhisperModel(args.model_size, device="cpu", compute_type="int8", cpu_threads=4)
    print("  -> Acoustic models initialized.")

    # 2. Initialize Core Services
    print("[2/4] Initializing Language Detection, ASR, NMT, and TTS Services...")
    lid_engine = LanguageDetectionEngine(whisper_model=shared_whisper, model_size=args.model_size)
    tracker = DynamicLanguageTracker(
        engine=lid_engine,
        initial_language=forced_source or "auto",
        switch_confidence_threshold=0.80,
        consecutive_turns_for_switch=2
    )
    asr_service = StreamingAsrService(model_size=args.model_size, shared_model=shared_whisper)
    translation_service = TranslationService()
    tts_service = StreamingTtsService()
    playback_engine = AudioPlaybackEngine()
    print("  -> All AI engines ready.")

    # 3. Setup Hardware Microphone Streamer & Echo Loop Safety Switch
    print("[3/4] Linking Acoustic Echo Prevention Safety Switch...")
    streamer = LiveAudioStreamer(
        vad_start_threshold=0.35,
        vad_end_threshold=0.25,
        hangover_ms=1100,
        min_speech_ms=300,
        max_speech_ms=9000
    )

    # Prevent Acoustic Echo Feedback Loops:
    # When synthetic speech plays out of the speakers, mute the mic input so it doesn't re-translate itself!
    link_playback_to_streamer(playback_engine, streamer, duck_only=False)
    print("  -> Feedback switch linked: mic automatically mutes during TTS playback.")

    turn_counter = 0

    # 4. End-to-End Processing Callback
    def on_speech_complete(segment: SpeechSegment):
        nonlocal turn_counter
        turn_counter += 1

        t_pipeline_start = time.perf_counter()
        audio = segment.audio
        sr = segment.sample_rate

        sys.stdout.write(f"\n\n{'=' * 74}\n")
        sys.stdout.write(f"📞 [Turn #{turn_counter}] Spoken Utterance Captured ({len(audio)/sr:.2f}s audio in RAM)\n")
        sys.stdout.write(f"{'=' * 74}\n")

        # Peak Normalization (Software AGC) for quiet microphones
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak > 0.003:
            audio = (audio / peak * 0.90).astype(np.float32)

        # --------------------------------------------------------------
        # Phase 3 & 4: Unified Acoustic LID + Streaming ASR
        # --------------------------------------------------------------
        if forced_source is not None:
            target_decode_lang = forced_source
        elif tracker.is_locked:
            lid_dec = tracker.process_turn(audio=audio, sample_rate=16000)
            target_decode_lang = tracker.current_language
        else:
            lid_dec = lid_engine.identify_spoken(audio, sample_rate=16000)
            target_decode_lang = lid_dec.language_code

        t_asr_0 = time.perf_counter()
        asr_res = asr_service.transcribe(
            audio_or_path=audio,
            language=target_decode_lang,
            beam_size=1
        )
        t_asr_ms = (time.perf_counter() - t_asr_0) * 1000.0

        spoken_text = asr_res.text.strip()

        # Drop empty noise or lone punctuation artifacts
        clean_words = "".join(c for c in spoken_text if c.isalnum())
        if not clean_words or len(clean_words) < 2:
            sys.stdout.write(f"  ⚠️ Pure acoustic noise/breath detected (raw: '{spoken_text}'), skipping turn.\n")
            return

        if forced_source is None:
            if not tracker.is_locked:
                script_dec = lid_engine.identify_script(spoken_text)
                locked_code = script_dec.language_code if (script_dec and script_dec.confidence >= 0.70) else target_decode_lang
                tracker.lock_language(locked_code)
                detected_code = tracker.current_language
            else:
                final_dec = tracker.process_turn(transcript=spoken_text)
                detected_code = tracker.current_language
        else:
            detected_code = forced_source

        lang_name = SUPPORTED_LANGUAGES.get(detected_code, LanguageInfo(detected_code, detected_code.title(), detected_code)).name
        conf_str = f" [{asr_res.language_probability:.0%}]" if forced_source is None else " [locked]"
        sys.stdout.write(f"  🔍 [Phase 3] Detected Language : {lang_name} ({detected_code}){conf_str}\n")
        sys.stdout.write(f"  🗣️ [Phase 4] Original Spoken   : \"{spoken_text}\" | Latency: {t_asr_ms:.1f}ms\n")

        # --------------------------------------------------------------
        # Phase 5: Contextual Translation with Entity Preservation
        # --------------------------------------------------------------
        if args.echo or detected_code == target_lang:
            tts_text = spoken_text
            tts_lang = detected_code
            t_trans_ms = 0.0
            sys.stdout.write(f"  🔁 [Echo/Mirror Mode] Speaking back in {lang_name} ({detected_code})\n")
        else:
            t_trans_0 = time.perf_counter()
            trans_res = translation_service.translate(
                text=spoken_text,
                source_lang=detected_code,
                target_lang=target_lang,
                preserve_entities=True
            )
            t_trans_ms = (time.perf_counter() - t_trans_0) * 1000.0
            tts_text = trans_res.translated_text
            tts_lang = target_lang

            sys.stdout.write(f"  🌐 [Phase 5] Translation ({target_lang.upper()}): \"{trans_res.translated_text}\"\n")
            sys.stdout.write(f"     Engine: {trans_res.backend} | Latency: {t_trans_ms:.1f}ms\n")
            if trans_res.preserved_entities:
                sys.stdout.write(f"     🔒 Preserved Entities: {trans_res.preserved_entities}\n")

            if target_lang != "en" and detected_code != "en":
                try:
                    en_trans = translation_service.translate(spoken_text, source_lang=detected_code, target_lang="en")
                    sys.stdout.write(f"     💡 English Meaning : \"{en_trans.translated_text}\"\n")
                except Exception:
                    pass

        # --------------------------------------------------------------
        # Phase 6 & 7: Streaming Neural TTS & Real-Time Voice Cloning Adaptation
        # --------------------------------------------------------------
        tts_result = None
        t_tts_ms = 0.0

        # Automatic live acoustic profiling for voice cloning:
        # Extract Fundamental Pitch (F0), Vocal Gender, and Speech Rate from caller's real voice
        try:
            acoustic_profile = VoiceCloningService.extract_acoustic_features(audio, sr)
            pitch_adj, rate_adj = VoiceCloningService.calculate_prosody_adjustments(acoustic_profile)
            cloned_gender = acoustic_profile.estimated_gender
        except Exception:
            acoustic_profile = None
            pitch_adj, rate_adj = "+0Hz", "+0%"
            cloned_gender = args.gender

        if not args.no_tts and tts_text:
            t_tts_0 = time.perf_counter()
            try:
                tts_result = tts_service.synthesize(
                    text=tts_text,
                    language_code=tts_lang,
                    gender=cloned_gender,
                    pitch=pitch_adj,
                    rate=rate_adj
                )
                t_tts_ms = (time.perf_counter() - t_tts_0) * 1000.0
                f0_info = f"{acoustic_profile.f0_median_hz:.1f}Hz" if acoustic_profile else "N/A"
                sys.stdout.write(
                    f"  🎙️ [Phase 6] Cloned Voice Synthesized: {tts_result.voice_id}\n"
                    f"     Adaptive Prosody: Gender={cloned_gender.upper()} | Pitch={f0_info} ({pitch_adj}) | "
                    f"Tempo={rate_adj} | Latency: {t_tts_ms:.1f}ms\n"
                )
            except Exception as e:
                sys.stdout.write(f"  ❌ TTS Synthesis failed: {e}\n")

        # Total Roundtrip Time
        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        sys.stdout.write(f"\n  ⏱️ [Full Roundtrip Latency]: {t_total_ms:.0f} ms ")
        sys.stdout.write(f"(Acoustic LID+ASR: {t_asr_ms:.0f}ms | NMT: {t_trans_ms:.0f}ms | TTS: {t_tts_ms:.0f}ms)\n")

        # --------------------------------------------------------------
        # Phase 8: Real-Time Speaker Playback (Echo-Protected)
        # --------------------------------------------------------------
        if tts_result is not None:
            mode_label = "echo voice" if (args.echo or detected_code == target_lang) else "translated voice"
            sys.stdout.write(f"  🔊 [Phase 8] Playing {mode_label} aloud through speakers (Mic safely muted)...\n")
            sys.stdout.flush()
            playback_engine.play(tts_result.audio, sample_rate=tts_result.sample_rate, blocking=True)
            sys.stdout.write(f"  ✅ Playback finished. Microphone unmuted and listening...\n")

        sys.stdout.write(f"{'-' * 74}\n")
        sys.stdout.flush()

    # 4b. Interactive Keyboard Text Mode (Bypasses Microphone)
    if args.text:
        print("\n" + "=" * 74)
        print("⌨️  INTERACTIVE TERMINAL TRANSLATION & LIVE AUDIO PLAYBACK")
        print("   Type ANY sentence in Telugu, Hindi, Tamil, Kannada, or English.")
        print("   -> Language will be auto-detected")
        print(f"   -> Translated into {target_info.name} ({target_lang})")
        print(f"   -> Spoken ALOUD through your speakers in natural neural voice!")
        print("   Type 'exit' or press Ctrl+C to stop.")
        print("=" * 74 + "\n")

        while True:
            try:
                user_input = input(f"\n[Your Input] > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    break

                t0 = time.perf_counter()

                # Phase 3: Language Detection
                t_lid_0 = time.perf_counter()
                lid_dec = lid_engine.identify(user_input)
                t_lid_ms = (time.perf_counter() - t_lid_0) * 1000.0

                # Phase 5: Translation
                t_trans_0 = time.perf_counter()
                trans_res = translation_service.translate(
                    text=user_input,
                    source_lang=lid_dec.language_code,
                    target_lang=target_lang,
                    preserve_entities=True
                )
                t_trans_ms = (time.perf_counter() - t_trans_0) * 1000.0

                print(f"  🔍 Detected Language : {lid_dec.language_name} ({lid_dec.language_code}) [{lid_dec.confidence*100:.1f}%]")
                print(f"  🌐 Translation ({target_lang.upper()}): \"{trans_res.translated_text}\" (Latency: {t_trans_ms:.1f}ms)")

                # Phase 6: TTS & Playback
                if not args.no_tts and trans_res.translated_text:
                    t_tts_0 = time.perf_counter()
                    tts_res = tts_service.synthesize(
                        text=trans_res.translated_text,
                        language_code=target_lang,
                        gender=args.gender
                    )
                    t_tts_ms = (time.perf_counter() - t_tts_0) * 1000.0
                    print(f"  🎙️ Synthesized: {tts_res.voice_id} | Roundtrip Latency: {(time.perf_counter() - t0)*1000.0:.0f}ms")
                    print(f"  🔊 Playing aloud through speakers...")
                    playback_engine.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)

            except (KeyboardInterrupt, EOFError):
                break

        print("\n\nExiting interactive text mode.")
        return

    streamer.register_on_utterance_complete(on_speech_complete)

    # 5. Start Live Capture Loop
    print("\n[4/4] Starting live microphone capture...")
    streamer.start()

    print("\n" + "=" * 74)
    print("🟢 CALL ACTIVE! Microphone is live.")
    if args.echo:
        print("👉 Speak a sentence in ANY language (Telugu, Tamil, Hindi, English, etc.).")
        print("   The system will capture your words, transcribe them, and immediately")
        print("   SPEAK THEM BACK to you out loud through your speakers!")
    else:
        print("👉 Speak a sentence in Telugu, Hindi, Tamil, or English, then pause naturally.")
        print("   The system will automatically detect the language, translate it,")
        print(f"   and speak it aloud in {target_info.name} ({target_lang}) through your speakers!")
    if args.duration:
        print(f"   Listening for {args.duration:.0f} seconds (or press Ctrl+C to stop).")
    else:
        print("   Press Ctrl+C at any time to end the call.")
    print("=" * 74 + "\n")

    t_start = time.time()
    try:
        spinner = ["|", "/", "-", "\\"]
        s_idx = 0
        while True:
            if args.duration and (time.time() - t_start >= args.duration):
                sys.stdout.write(f"\n\n⏱️ Duration limit ({args.duration:.0f}s) reached.\n")
                break
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            is_speaking = prob >= 0.45
            status_text = "🗣️ SPEAKING" if is_speaking else "👂 LISTENING"

            # Visual VU Meter bar
            bar_len = int(max(0, min(15, (rms + 60) / 4)))
            vu_bar = "█" * bar_len + "░" * (15 - bar_len)

            muted_tag = " [MIC MUTED - ECHO PREVENTED]" if playback_engine.is_playing else ""
            mode_tag = "ECHO" if args.echo else target_lang.upper()

            sys.stdout.write(
                f"\r{spinner[s_idx % 4]} [{status_text}] [{vu_bar}] {rms:5.1f} dBFS | "
                f"VAD Prob: {prob:4.2f} | Out: {mode_tag}{muted_tag}    "
            )
            sys.stdout.flush()
            s_idx += 1
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\n\nEnding call...")
    finally:
        streamer.stop()
        playback_engine.stop()

    print("\n" + "=" * 74)
    print(f"📞 Call Session Ended. Total Turns Processed: {turn_counter}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
