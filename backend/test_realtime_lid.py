"""
Interactive and Automated Verification Tool for Real-Time Automatic Language Detection (LID).
Demonstrates zero-disk streaming microphone capture, spoken language identification,
confidence ranking, and dynamic language switching during conversation.
"""

import sys
import time
import argparse
import numpy as np

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker
from app.models.language import SUPPORTED_LANGUAGES


def run_self_test(engine: LanguageDetectionEngine):
    """Automated self-test on curated audio samples."""
    import soundfile as sf
    import os

    print("\n--- Running Automated LID Self-Test on Curated Samples ---")
    samples = [
        ("English Spoken Sample", r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\transcripts\demo_spoken_sample.wav"),
    ]

    tracker = DynamicLanguageTracker(engine=engine, initial_language="en")

    for label, path in samples:
        if os.path.exists(path):
            audio, sr = sf.read(path)
            t0 = time.time()
            decision = tracker.process_turn(audio=audio, sample_rate=sr)
            dt = time.time() - t0
            print(f"\n[Test Case: {label}] ({len(audio)/sr:.2f}s)")
            print(f"  Detected Language : {decision.language_name} ({decision.language_code})")
            print(f"  Confidence Score  : {decision.confidence:.1%}")
            print(f"  Detection Source  : {decision.detection_source}")
            print(f"  Latency           : {decision.latency_ms:.1f}ms (Total elapsed: {dt:.3f}s)")
            print(f"  Top Candidates    :")
            for c_code, c_name, c_prob in decision.top_candidates[:3]:
                print(f"    - {c_name} ({c_code}): {c_prob:.1%}")
        else:
            print(f"  File not found: {path}")

    # Test Script LID
    print("\n--- Testing Multi-Script Unicode Identification ---")
    phrases = [
        ("Telugu", "నమస్కారం, మీరు ఎలా ఉన్నారు?"),
        ("Hindi", "नमस्ते, आप आज कैसे हैं?"),
        ("Tamil", "வணக்கம், நீங்கள் நலமா?"),
        ("English", "Hello, we are testing the multilingual real-time translation system."),
    ]
    for lang_name, text in phrases:
        dec = engine.identify_script(text)
        print(f"  [{lang_name} Script]: '{text}' -> Detected '{dec.language_name}' ({dec.language_code}) at {dec.confidence:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Spoken Language Identification (LID)")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device index")
    parser.add_argument("--duration", type=float, default=20.0, help="Listening duration in seconds")
    parser.add_argument("--self-test", action="store_true", help="Run automated self-test on benchmark samples")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🌐 REAL-TIME AUTOMATIC LANGUAGE DETECTION (LID) ENGINE")
    print("=" * 70)

    engine = LanguageDetectionEngine()

    if args.self_test:
        run_self_test(engine)
        print("\n" + "=" * 70)
        print(" ✅ LID SELF-TEST COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
        return

    tracker = DynamicLanguageTracker(
        engine=engine,
        initial_language="en",
        switch_confidence_threshold=0.72,
        consecutive_turns_for_switch=2
    )

    def on_switch(old_lang, new_lang, conf):
        old_info = SUPPORTED_LANGUAGES.get(old_lang)
        new_info = SUPPORTED_LANGUAGES.get(new_lang)
        old_name = old_info.name if old_info else old_lang
        new_name = new_info.name if new_info else new_lang
        sys.stdout.write(
            f"\n\n🔄 [LANGUAGE SWITCH DETECTED]! {old_name} ({old_lang}) -> {new_name} ({new_lang}) "
            f"with confidence {conf:.1%}\n\n"
        )
        sys.stdout.flush()

    tracker.register_switch_listener(on_switch)

    turn_count = 0

    def on_utterance_complete(segment: SpeechSegment):
        nonlocal turn_count
        turn_count += 1
        t0 = time.time()
        decision = tracker.process_turn(audio=segment.audio, sample_rate=segment.sample_rate)
        dt = (time.time() - t0) * 1000.0

        candidates_str = ", ".join(f"{c_name}: {c_prob:.1%}" for _, c_name, c_prob in decision.top_candidates[:3])
        switch_marker = " ⚡ [SWITCH]" if decision.is_language_switch else ""

        sys.stdout.write(
            f"\n🗣️  [Turn #{turn_count}]: Speech ({segment.duration_sec:.2f}s, RMS={segment.rms_dbfs:.1f} dBFS)\n"
            f"    Language   : {decision.language_name} [{decision.language_code}]{switch_marker}\n"
            f"    Confidence : {decision.confidence:.1%}\n"
            f"    Top Rank   : {candidates_str}\n"
            f"    Latency    : {dt:.1f}ms (Acoustic LID: {decision.latency_ms:.1f}ms) | Zero Disk I/O\n"
        )
        sys.stdout.flush()

    streamer = LiveAudioStreamer(device_index=args.device)
    streamer.register_on_utterance_complete(on_utterance_complete)

    print(f"\nMicrophone active (Listening for {args.duration:.1f}s). Speak in English, Hindi, Telugu, etc...")
    print("Language switching is handled automatically. Press Ctrl+C to stop.\n")

    streamer.start()
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            curr_lang = tracker.current_decision.language_name
            sys.stdout.write(f"\r[{elapsed:4.1f}s] Active Lang: {curr_lang:<10} | Mic Level: {rms:5.1f} dBFS | VAD: {prob:4.2f}")
            sys.stdout.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        streamer.stop()

    print("\n\n" + "=" * 70)
    print(f" Total conversational turns processed: {turn_count}")
    print(f" Final locked language: {tracker.current_decision.language_name} ({tracker.current_decision.language_code})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
