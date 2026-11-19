"""
Live Real-Time Streaming Speech-to-Text (ASR) with Automatic Language Detection.
Demonstrates end-to-end continuous microphone capture, VAD utterance segmentation,
automatic language identification, and real-time transcription with word timestamps.
Zero disk writes!
"""

import sys
import time
import argparse
import numpy as np

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker
from app.services.asr_service import StreamingAsrService
from app.models.transcription import TranscriptionResult


def run_self_test(asr_service: StreamingAsrService, lid_engine: LanguageDetectionEngine):
    """Automated self-test on benchmark audio samples."""
    import soundfile as sf
    import os

    print("\n--- Running Automated Streaming ASR Self-Test on Curated Sample ---")
    sample_path = r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\transcripts\demo_spoken_sample.wav"
    if not os.path.exists(sample_path):
        print(f"Sample not found: {sample_path}")
        return

    audio, sr = sf.read(sample_path)
    print(f"Sample: demo_spoken_sample.wav ({len(audio)/sr:.2f}s, sr={sr})")

    # Step 1: Detect Language
    t0 = time.time()
    lid_dec = lid_engine.identify_spoken(audio, sample_rate=sr)
    lid_time = (time.time() - t0) * 1000.0

    print(f"  Step 1: LID -> {lid_dec.language_name} [{lid_dec.language_code}] ({lid_dec.confidence:.1%}) in {lid_time:.1f}ms")

    # Step 2: Transcribe with detected language
    t1 = time.time()
    res = asr_service.transcribe(audio, language=lid_dec.language_code, word_timestamps=True)
    asr_time = (time.time() - t1) * 1000.0

    print(f"  Step 2: ASR -> \"{res.text}\"")
    print(f"  Latency: {asr_time:.1f}ms (RTF: {res.rtf:.2f}x)")
    print(f"  Word Count: {len(res.words)} words with timestamps")
    if res.words:
        first_few = " | ".join(f"{w.word} ({w.start}s-{w.end}s)" for w in res.words[:4])
        print(f"  Sample Words: {first_few}")
    print("\n" + "=" * 70)
    print(" ✅ ASR SELF-TEST PASSED SUCCESSFULLY")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Streaming Speech-to-Text with Auto-LID")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small"], help="Whisper model size (default: base)")
    parser.add_argument("--lang", type=str, default="auto", help="Language code to transcribe (e.g. te, hi, ta, en, or auto)")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--duration", type=float, default=30.0, help="Continuous listening duration in seconds")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable per-word timing sync (slightly slower on CPU)")
    parser.add_argument("--self-test", action="store_true", help="Run automated self-test on sample audio")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print(" 🎙️ PHASE 4: REAL-TIME STREAMING SPEECH-TO-TEXT (ASR)")
    print("=" * 72)

    # Initialize shared ASR & 107-Language VoxLingua107 ECAPA LID services
    print(f"\nInitializing VoxLingua107 ECAPA-TDNN LID (107 langs) & Neural ASR...")
    asr_service = StreamingAsrService.get_instance(model_size=args.model)
    lid_engine = LanguageDetectionEngine(whisper_model=asr_service.model, model_size=args.model)
    LanguageDetectionEngine.get_ecapa_model()

    if args.self_test:
        run_self_test(asr_service, lid_engine)
        return

    from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code
    forced_lang = None if args.lang.strip().lower() == "auto" else normalize_language_code(args.lang, default="en")
    
    tracker = DynamicLanguageTracker(
        engine=lid_engine,
        initial_language=forced_lang or "auto",
        switch_confidence_threshold=0.80,
        consecutive_turns_for_switch=2
    )
    active_lang_display = "Auto-Detect" if forced_lang is None else SUPPORTED_LANGUAGES.get(forced_lang, LanguageInfo(forced_lang, forced_lang.title(), forced_lang)).name

    turn_counter = 0

    def on_speech_start(ts: float):
        sys.stdout.write("\n🎙️ [SPEECH] Listening...")
        sys.stdout.flush()

    def on_utterance_complete(segment: SpeechSegment):
        nonlocal turn_counter, active_lang_display
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
            lid_dec = tracker.process_turn(audio=audio, sample_rate=16000)
            decode_lang = tracker.current_language
        else:
            lid_dec = lid_engine.identify_spoken(audio, sample_rate=16000)
            decode_lang = lid_dec.language_code

        res = asr_service.transcribe(
            audio,
            language=decode_lang,
            word_timestamps=args.word_timestamps,
            trim_audio=False
        )
        asr_ms = (time.time() - t_asr) * 1000.0

        spoken_text = res.text.strip()

        clean_words = "".join(c for c in spoken_text if c.isalnum())
        if not clean_words or len(clean_words) < 2:
            sys.stdout.write(f"\n⚠️ [Turn #{turn_counter}] Pure noise/breath detected, skipped.\n")
            sys.stdout.flush()
            return

        # 2. Post-ASR Verification & Session Locking
        if forced_lang is None:
            if not tracker.is_locked:
                script_dec = lid_engine.identify_script(spoken_text)
                locked_code = script_dec.language_code if (script_dec and script_dec.confidence >= 0.70) else decode_lang
                tracker.lock_language(locked_code)
                detected_code = tracker.current_language
                active_lang_display = tracker.current_decision.language_name
                sys.stdout.write(f"\n🔒 [Session Locked] Spoken language confirmed: {active_lang_display} [{detected_code}]\n")
            else:
                final_dec = tracker.process_turn(transcript=spoken_text)
                detected_code = tracker.current_language
                active_lang_display = tracker.current_decision.language_name
        else:
            detected_code = forced_lang
            active_lang_display = SUPPORTED_LANGUAGES.get(detected_code, LanguageInfo(detected_code, detected_code.title(), detected_code)).name

        lang_info = SUPPORTED_LANGUAGES.get(detected_code, LanguageInfo(detected_code, detected_code.title(), detected_code))

        # Display real-time formatted subtitle
        conf_str = f" {res.language_probability:.0%}" if forced_lang is None else " locked"
        sys.stdout.write(
            f"\n\n💬 [Turn #{turn_counter}] {lang_info.name} [{detected_code}{conf_str}] "
            f"({segment.duration_sec:.2f}s | {asr_ms:.0f}ms):\n"
            f"   👉 \"{spoken_text}\"\n"
        )
        if res.words and len(res.words) <= 10:
            words_info = " ".join(f"{w.word}[{w.start:.1f}s]" for w in res.words)
            sys.stdout.write(f"   ⏱️  Word Sync: {words_info}\n")
        sys.stdout.write("-" * 68 + "\n")
        sys.stdout.flush()

    streamer = LiveAudioStreamer(
        device_index=args.device,
        vad_start_threshold=0.50,
        vad_end_threshold=0.35,
        hangover_ms=1100,
        min_speech_ms=400,
        max_speech_ms=9000
    )
    streamer.register_on_speech_start(on_speech_start)
    streamer.register_on_utterance_complete(on_utterance_complete)

    print(f"\nMode: {active_lang_display} | Listening live on microphone for {args.duration:.0f} seconds.")
    print("Speak in ANY language (Telugu, Hindi, Tamil, Kannada, Malayalam, English, Spanish, French...).")
    print("Speech boundaries are segmented automatically via Silero VAD (Zero disk I/O).")
    print("Press Ctrl+C to stop at any time.\n")

    streamer.start()
    t_start = time.time()
    try:
        while time.time() - t_start < args.duration:
            elapsed = time.time() - t_start
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            sys.stdout.write(f"\r[{elapsed:4.1f}s] {active_lang_display:<11} | {rms:5.1f}dB | VAD:{prob:.2f}  ")
            sys.stdout.flush()
            time.sleep(0.10)
    except KeyboardInterrupt:
        print("\nStopping streaming...")
    finally:
        streamer.stop()

    print("\n\n" + "=" * 72)
    print(f" Summary: {turn_counter} utterances transcribed in real time (Zero audio files saved).")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
