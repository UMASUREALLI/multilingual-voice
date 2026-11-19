"""
Interactive Live Translation Diagnostic & Demonstration Tool.
Supports:
1. Live Interactive Text Translation (Type ANY custom sentence in Telugu, Hindi, Tamil, English, etc.)
2. Live Microphone Speech Translation (Speak naturally -> VAD -> LID -> ASR -> Translation)
3. Automated Benchmark Validation (--self-test)
"""

import sys
import time
import argparse
import numpy as np

from app.services.translation_service import TranslationService, BidirectionalCallTranslator
from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker
from app.services.asr_service import StreamingAsrService
from app.services.tts_service import StreamingTtsService
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code
from app.services.entity_preserver import EntityPreserver


def print_banner(title: str):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def run_interactive_text_mode(translator: TranslationService, play_audio: bool = True):
    """Allows user to type ANY custom sentence, see real-time translation, and hear it spoken aloud."""
    print_banner("INTERACTIVE TEXT-TO-SPEECH & TRANSLATION (TYPE & HEAR ALOUD)")
    print("Languages supported: te (Telugu), hi (Hindi), ta (Tamil), en (English), kn (Kannada), etc.")
    print("Tip: You can enter 2-letter codes ('en', 'ta', 'te') or full names ('english', 'tamil', 'telugu').")
    print("Type 'exit' or press Ctrl+C to return to main menu.\n")

    raw_src = input("Enter Source Language (e.g. en / ta / te) [default: en] > ").strip().lower() or "en"
    raw_tgt = input("Enter Target Language (e.g. te / telugu / hi) [default: te] > ").strip().lower() or "te"

    src = normalize_language_code(raw_src, default="en")
    tgt = normalize_language_code(raw_tgt, default="te")

    src_name = SUPPORTED_LANGUAGES.get(src, LanguageInfo(src, src.title(), src)).name
    tgt_name = SUPPORTED_LANGUAGES.get(tgt, LanguageInfo(tgt, tgt.title(), tgt)).name

    tts = StreamingTtsService()
    player = AudioPlaybackEngine()

    print(f"\n🟢 Ready! Translating & Speaking: [{src_name} ({src}) ➔ {tgt_name} ({tgt})]")
    print(f"🔊 Audio Playback: ACTIVE (The translation will speak aloud through your speakers!)")
    print("-" * 74)

    while True:
        try:
            user_text = input(f"\n[{src_name}] Type text > ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit", "q"}:
                break

            t0 = time.time()
            res = translator.translate(user_text, source_lang=src, target_lang=tgt, preserve_entities=True)
            trans_ms = (time.time() - t0) * 1000.0

            print(f"[{tgt_name}] 🌐 Translation: \"{res.translated_text}\"")

            if play_audio and res.translated_text:
                try:
                    t_tts = time.time()
                    tts_res = tts.synthesize(text=res.translated_text, language_code=tgt, gender="female")
                    tts_ms = (time.time() - t_tts) * 1000.0
                    print(f"     🎙️ Voice: {tts_res.voice_id} | Latency: NMT {trans_ms:.0f}ms + TTS {tts_ms:.0f}ms")
                    print(f"     🔊 Speaking aloud through speakers...")
                    player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)
                except Exception as e:
                    print(f"     ⚠️ Audio playback error: {e}")

            if res.preserved_entities:
                print(f"     🔒 Preserved Entities: {res.preserved_entities}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive text mode.")
            break


def run_live_mic_mode(translator: TranslationService, src_lang: str = "te", tgt_lang: str = "en", duration: float = 30.0, play_tts: bool = False, echo: bool = False, gender: str = "female"):
    """Captures microphone speech, transcribes, translates live, and optionally speaks out loud."""
    src_mode = src_lang.strip().lower()
    forced_lang = None if src_mode == "auto" else normalize_language_code(src_mode, default="te")
    tgt_lang = normalize_language_code(tgt_lang, default="en")
    mode_title = "VOICE ECHO / MIRROR" if echo else f"[{src_lang.upper()} ➔ {tgt_lang.upper()}]"
    print_banner(f"PHASE 5: LIVE MICROPHONE TRANSLATION {mode_title}")

    print("\nInitializing VoxLingua107 ECAPA-TDNN LID (107 langs) & Neural ASR...")
    asr = StreamingAsrService.get_instance(model_size="base")
    LanguageDetectionEngine.get_ecapa_model()
    
    lid_engine = LanguageDetectionEngine(whisper_model=asr.model, model_size=asr.model_size)
    tracker = DynamicLanguageTracker(
        engine=lid_engine,
        initial_language=forced_lang or "auto",
        switch_confidence_threshold=0.80,
        consecutive_turns_for_switch=2
    )

    tts = StreamingTtsService() if play_tts else None
    player = AudioPlaybackEngine() if play_tts else None

    active_lang_name = "Auto-Detect" if forced_lang is None else SUPPORTED_LANGUAGES.get(forced_lang, LanguageInfo(forced_lang, forced_lang.title(), forced_lang)).name

    turn_counter = 0

    def on_utterance_complete(segment: SpeechSegment):
        nonlocal turn_counter, active_lang_name
        turn_counter += 1

        audio = segment.audio
        # Software AGC (Peak Normalization) for quiet microphones
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak > 0.003:
            audio = (audio / peak * 0.90).astype(np.float32)

        # 1. Unified 107-Language Acoustic LID with Session Locking + Native-Script Neural ASR
        t_asr = time.time()
        if forced_lang is not None:
            decode_lang = forced_lang
        elif tracker.is_locked:
            # Session is established: hold locked language with anti-jitter hysteresis
            lid_dec = tracker.process_turn(audio=audio, sample_rate=16000)
            decode_lang = tracker.current_language
        else:
            # Session not yet locked: run acoustic LID to find tentative candidate
            lid_dec = lid_engine.identify_spoken(audio, sample_rate=16000)
            decode_lang = lid_dec.language_code

        asr_res = asr.transcribe(
            audio,
            language=decode_lang,
            word_timestamps=False,
            trim_audio=False
        )
        asr_ms = (time.time() - t_asr) * 1000.0

        spoken_text = asr_res.text.strip()

        clean_words = "".join(c for c in spoken_text if c.isalnum())
        if not clean_words or len(clean_words) < 2:
            sys.stdout.write(f"\n⚠️ [Turn #{turn_counter}] Pure noise/breath detected, skipped.\n")
            sys.stdout.flush()
            return

        # 2. Post-ASR Verification & Session Locking
        if forced_lang is None:
            if not tracker.is_locked:
                # First valid speech turn: verify script and LOCK session language!
                script_dec = lid_engine.identify_script(spoken_text)
                locked_code = script_dec.language_code if (script_dec and script_dec.confidence >= 0.70) else decode_lang
                tracker.lock_language(locked_code)
                detected_code = tracker.current_language
                active_lang_name = tracker.current_decision.language_name
                sys.stdout.write(f"\n🔒 [Session Locked] Spoken language confirmed: {active_lang_name} [{detected_code}]\n")
            else:
                final_dec = tracker.process_turn(transcript=spoken_text)
                detected_code = tracker.current_language
                active_lang_name = tracker.current_decision.language_name
        else:
            detected_code = forced_lang
            active_lang_name = SUPPORTED_LANGUAGES.get(detected_code, LanguageInfo(detected_code, detected_code.title(), detected_code)).name

        src_info = SUPPORTED_LANGUAGES.get(detected_code, LanguageInfo(detected_code, detected_code.title(), detected_code))

        # 2. Dynamic Real-Time Translation with Entity Preservation
        t_trans = time.time()
        if echo or detected_code == tgt_lang:
            trans_text = spoken_text
            trans_ms = 0.0
            tgt_display_name = src_info.name
            out_code = detected_code
        else:
            trans_res = translator.translate(
                text=spoken_text,
                source_lang=detected_code,
                target_lang=tgt_lang,
                preserve_entities=True
            )
            trans_text = trans_res.translated_text
            trans_ms = (time.time() - t_trans) * 1000.0
            tgt_display_name = trans_res.target_language_name
            out_code = tgt_lang

        total_pipeline_ms = asr_ms + trans_ms

        en_ref = ""
        if not echo and tgt_lang != "en" and detected_code != "en":
            try:
                en_trans = translator.translate(spoken_text, source_lang=detected_code, target_lang="en")
                en_ref = f"   💡 English Meaning : \"{en_trans.translated_text}\"\n"
            except Exception:
                pass

        header_label = f"💬 [Turn #{turn_counter}] {src_info.name} [{detected_code}] ➔ {tgt_display_name} [{out_code}]:"
        sys.stdout.write(
            f"\n\n{header_label}\n"
            f"   🗣️  Original Spoken : \"{spoken_text}\"\n"
            f"   🌐 Live Translation: \"{trans_text}\"\n"
        )
        if en_ref:
            sys.stdout.write(en_ref)
        sys.stdout.write(
            f"   ⏱️  Total Latency   : {total_pipeline_ms:.0f}ms (LID+ASR: {asr_ms:.0f}ms, NMT: {trans_ms:.0f}ms)\n"
        )
        if not echo and 'trans_res' in locals() and trans_res.preserved_entities:
            sys.stdout.write(f"   🔒 Preserved Entities: {trans_res.preserved_entities}\n")

        # 3. Neural TTS Playback (with Acoustic Echo Feedback Loop Prevention)
        if play_tts and tts and player and trans_text:
            try:
                t_tts = time.time()
                tts_res = tts.synthesize(text=trans_text, language_code=out_code, gender=gender)
                tts_ms = (time.time() - t_tts) * 1000.0
                action_word = "Voice Echo" if echo else "Translation"
                sys.stdout.write(f"   🎙️ Neural TTS ({tts_res.voice_id}): Synthesized in {tts_ms:.0f}ms\n")
                sys.stdout.write(f"   🔊 Speaking {action_word} aloud (Mic safely muted)...\n")
                sys.stdout.flush()
                player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)
                sys.stdout.write(f"   ✅ Finished speaking. Listening again...\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"   ⚠️ TTS error: {e}\n")
                sys.stdout.flush()

        sys.stdout.write("-" * 68 + "\n")
        sys.stdout.flush()

    streamer = LiveAudioStreamer(
        vad_start_threshold=0.50,
        vad_end_threshold=0.35,
        hangover_ms=1100,
        min_speech_ms=400,
        max_speech_ms=9000
    )
    streamer.register_on_utterance_complete(on_utterance_complete)

    if play_tts and player:
        link_playback_to_streamer(player, streamer, duck_only=False)

    tgt_info = SUPPORTED_LANGUAGES.get(tgt_lang, LanguageInfo(tgt_lang, tgt_lang.title(), tgt_lang))
    print(f"\nMicrophone active for {duration:.0f} seconds.")
    if echo:
        print(f"Echo Mode: Speak in ANY language, and the system will speak your exact words back out loud!")
    else:
        print(f"Speak clearly in {active_lang_name}. Receiver language: {tgt_info.name} ({tgt_lang}).")
    if play_tts:
        print(f"🔊 Live TTS Playback: ENABLED (Natural neural speech will play through your speakers!)")
    print("Press Ctrl+C to stop listening at any time.\n")

    streamer.start()
    t_start = time.time()
    try:
        while time.time() - t_start < duration:
            elapsed = time.time() - t_start
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            playing_tag = " [MIC MUTED]" if (player and player.is_playing) else ""
            sys.stdout.write(f"\r[{elapsed:4.1f}s] {active_lang_name:<11}➔{tgt_lang.upper()} | {rms:5.1f}dB | VAD:{prob:.2f}{playing_tag}  ")
            sys.stdout.flush()
            time.sleep(0.10)
    except KeyboardInterrupt:
        print("\nStopping live capture...")
    finally:
        streamer.stop()
        if player:
            player.stop()

    print(f"\n\nDone: Processed {turn_counter} conversational turns live.\n")


def run_benchmark(translator: TranslationService):
    """Automated benchmark test suite."""
    print_banner("RUNNING MULTILINGUAL TRANSLATION BENCHMARK")
    test_pairs = [
        ("te", "en", "నమస్కారం, ఈ ప్రాజెక్ట్ చాలా బాగుంది."),
        ("en", "te", "Hello, I am very excited to work on this real-time AI project."),
        ("hi", "en", "नमस्ते, आज का मौसम बहुत अच्छा है।"),
        ("en", "hi", "We are developing an intelligent call translation system."),
        ("en", "te", "Please transfer ₹5000 to user@gmail.com and call 9848012345."),
    ]
    for src, tgt, text in test_pairs:
        t0 = time.time()
        res = translator.translate(text, source_lang=src, target_lang=tgt, preserve_entities=True)
        dt = (time.time() - t0) * 1000.0
        entities_badge = f" [Preserved: {', '.join(res.preserved_entities)}]" if res.preserved_entities else ""
        print(f"\n[{res.source_language_name} ➔ {res.target_language_name}]:")
        print(f"  Source : \"{res.source_text}\"")
        print(f"  Target : \"{res.translated_text}\"{entities_badge}")
        print(f"  Latency: {dt:.1f}ms | Backend: {res.backend}")
    print("\n" + "=" * 74)
    print(" ✅ BENCHMARK COMPLETED")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Translation Diagnostic & Live Testing Tool")
    parser.add_argument("--self-test", action="store_true", help="Run automated static benchmark")
    parser.add_argument("--text", action="store_true", help="Launch interactive text translation CLI")
    parser.add_argument("--live", action="store_true", help="Launch live microphone-to-translation stream")
    parser.add_argument("--src", type=str, default="te", help="Source language (default: te)")
    parser.add_argument("--tgt", type=str, default="en", help="Target language (default: en)")
    parser.add_argument("--duration", type=float, default=30.0, help="Live microphone capture duration in seconds")
    parser.add_argument("--tts", action="store_true", help="Synthesize and speak translation aloud through speakers")
    parser.add_argument("--echo", action="store_true", help="Mirror/Echo mode: speaks back exact spoken words in original language")
    parser.add_argument("--gender", type=str, default="female", choices=["female", "male"], help="TTS voice gender")
    args = parser.parse_args()

    translator = TranslationService()

    if args.self_test:
        run_benchmark(translator)
        return

    if args.live:
        run_live_mic_mode(
            translator,
            src_lang=args.src,
            tgt_lang=args.tgt,
            duration=args.duration,
            play_tts=args.tts or args.echo,
            echo=args.echo,
            gender=args.gender
        )
        return

    if args.text:
        run_interactive_text_mode(translator)
        return

    # If no flags passed, prompt user to choose mode
    print_banner("REAL-TIME CALL TRANSLATION — LIVE TESTING SUITE")
    print("Choose how you want to test live translation:")
    print("  [1] Interactive Live Text Translation (Type ANY sentence in Telugu/Hindi/English)")
    print("  [2] Live Microphone Translation (Speak into mic -> Transcribes & Translates)")
    print("  [3] Automated Benchmark Self-Test")
    print("-" * 74)

    try:
        choice = input("Enter choice (1, 2, or 3) [default: 1] > ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    if choice == "2":
        run_live_mic_mode(
            translator,
            src_lang=args.src,
            tgt_lang=args.tgt,
            duration=args.duration,
            play_tts=args.tts or args.echo,
            echo=args.echo,
            gender=args.gender
        )
    elif choice == "3":
        run_benchmark(translator)
    else:
        run_interactive_text_mode(translator)


if __name__ == "__main__":
    main()
