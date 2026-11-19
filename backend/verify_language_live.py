"""
Enhanced Live Language Detection Diagnostic & Verification Tool.
- Extended listening window (10-15s default or press ENTER to finish early).
- Automatic silence stripping (removes room noise and delays before speech).
- Whisper 'small' model (244M parameters) for superior Telugu/Hindi/Tamil discernment.
- Consensus resolution combining Acoustic LID, Full-Sequence Decoding, and Unicode Script Matching.
"""

import sys
import time
import argparse
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo
from app.services.lid_service import LanguageDetectionEngine, trim_silence


def print_banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def render_vu(rms: float, width: int = 18) -> str:
    db = 20.0 * np.log10(max(rms, 1e-6))
    clamped_db = max(-60.0, min(0.0, db))
    ratio = (clamped_db + 60.0) / 60.0
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {db:5.1f} dBFS"


def record_with_feedback(duration: float, sample_rate: int = 16000, device: int = None) -> np.ndarray:
    """Records audio continuously with a live VU meter and early ENTER stop."""
    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    def wait_for_enter():
        try:
            input()
            stop_event.set()
        except EOFError:
            pass

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=512,
        device=device,
        callback=callback
    )

    t_thread = threading.Thread(target=wait_for_enter, daemon=True)
    t_thread.start()

    print(f"\n🔴 RECORDING for up to {duration:.0f} seconds (Press ENTER at any time to finish early):")
    stream.start()
    t0 = time.time()

    try:
        while time.time() - t0 < duration and not stop_event.is_set():
            remaining = duration - (time.time() - t0)
            if frames:
                last_frame = frames[-1]
                rms = float(np.sqrt(np.mean(last_frame ** 2)))
                vu = render_vu(rms)
                sys.stdout.write(f"\r  ⏱️ [{remaining:4.1f}s remaining] {vu} | Speak clearly...")
                sys.stdout.flush()
            time.sleep(0.05)
    finally:
        stop_event.set()
        stream.stop()
        stream.close()

    sys.stdout.write("\n\n⏹️ Recording finished. Processing audio...\n")
    sys.stdout.flush()

    if not frames:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(frames, axis=0).flatten()


def analyze_audio(audio: np.ndarray, sample_rate: int, engine: LanguageDetectionEngine):
    if len(audio) == 0:
        print("❌ No audio captured!")
        return

    raw_duration = len(audio) / sample_rate
    raw_rms = float(np.sqrt(np.mean(audio ** 2)))
    raw_peak = float(np.max(np.abs(audio)))

    print(f"\n📊 [Raw Audio Diagnostics]:")
    print(f"  Captured Duration : {raw_duration:.2f} seconds ({len(audio)} samples)")
    print(f"  Signal RMS Level  : {raw_rms:.5f}")
    print(f"  Peak Amplitude    : {raw_peak:.5f}")

    if raw_rms < 0.002:
        print("\n⚠️  [WARNING]: Signal volume is extremely low!")
        print("    Please check if your microphone is selected properly or speak louder.")

    # 1. Silence Trimming
    trimmed_audio, start_offset, speech_duration = trim_silence(audio, sample_rate=sample_rate)
    print(f"\n✂️ [Smart Silence Trimming]:")
    print(f"  Detected Speech Start : {start_offset:.2f}s (stripped leading room silence)")
    print(f"  Active Speech Length  : {speech_duration:.2f}s of human vocalization")

    # 2. Spoken Acoustic LID on Pure Speech
    print(f"\n🔍 [1. Acoustic Spoken Language Identification (Whisper {engine.model_size})]:")
    t0 = time.time()
    spoken_decision = engine.identify_spoken(trimmed_audio, sample_rate=sample_rate, trim_audio=False)
    lid_latency = (time.time() - t0) * 1000.0

    print(f"  Acoustic Prediction : {spoken_decision.language_name} ({spoken_decision.native_name}) [{spoken_decision.language_code}]")
    print(f"  Acoustic Confidence : {spoken_decision.confidence:.1%}")
    print(f"  Inference Latency   : {lid_latency:.1f} ms")

    print("\n  Top Language Probability Distribution:")
    for code, name, prob in spoken_decision.top_candidates[:6]:
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    - {name:<12} [{code}]: {prob:6.1%} |{bar}|")

    # 3. Full Sequence Transcription
    print(f"\n📝 [2. Multilingual Speech Recognition (Transcription)]:")
    t1 = time.time()
    final_decision, transcript = engine.full_speech_and_language_analysis(audio, sample_rate=sample_rate)
    asr_latency = (time.time() - t1) * 1000.0

    print(f"  Transcribed Text    : \"{transcript}\"")
    print(f"  ASR Decode Latency  : {asr_latency:.1f} ms")

    # 4. Stage-2 Script Analysis
    print(f"\n🔤 [3. Stage-2 Unicode Script Verification]:")
    script_dec = engine.identify_script(transcript)
    if script_dec:
        print(f"  Script Detected     : {script_dec.language_name} ({script_dec.native_name}) [{script_dec.language_code}]")
        print(f"  Script Confidence   : {script_dec.confidence:.1%}")
        if script_dec.language_code == spoken_decision.language_code:
            print("  Alignment Status    : ✅ Acoustic LID and Script LID MATCH PERFECTLY!")
        else:
            print(f"  Alignment Status    : ℹ️ Script suggests [{script_dec.language_code}], Acoustic was [{spoken_decision.language_code}]")
    else:
        print("  Script Detected     : (Insufficient text tokens)")

    # Final Decision
    print("\n" + "=" * 72)
    print(f"  🎯 FINAL RESOLVED LANGUAGE: {final_decision.language_name.upper()} ({final_decision.native_name}) [{final_decision.language_code}]")
    print(f"     Confidence: {final_decision.confidence:.1%}  |  Detection Source: {final_decision.detection_source}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Language Detection Diagnostic Tool")
    parser.add_argument("--model", type=str, default="base", choices=["small", "base", "tiny"], help="Whisper model size (default: base)")
    parser.add_argument("--duration", type=float, default=12.0, help="Listening window duration in seconds (default: 12.0s)")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--file", type=str, default=None, help="Path to audio file to test")
    args = parser.parse_args()

    print_banner(f"MULTILINGUAL LANGUAGE DETECTION DIAGNOSTIC (MODEL: {args.model.upper()})")

    print(f"\nLoading faster-whisper '{args.model}' (INT8 CPU)...")
    engine = LanguageDetectionEngine(model_size=args.model)
    _ = engine._get_model()  # Pre-warm model in memory
    print("Model initialized and ready.")

    if args.file:
        import soundfile as sf
        print(f"\nTesting File: {args.file}")
        audio, sr = sf.read(args.file)
        if audio.ndim > 1:
            audio = audio[:, 0]
        analyze_audio(audio, sr, engine)
        return

    sample_rate = 16000
    print(f"\nInput Device: {args.device or 'System Default'}")
    print(f"Listening Window: {args.duration:.0f} seconds (Speak a full sentence, then press ENTER or wait)")
    print("-" * 72)
    print("Example Sentences to Speak:")
    print("  - Telugu: 'నమస్కారం, ఈరోజు మనం ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ గురించి మాట్లాడుకుంటున్నాము.'")
    print("  - Hindi : 'नमस्ते, आज हम रियल टाइम ट्रांसलेशन सिस्टम का टेस्ट कर रहे हैं.'")
    print("  - Tamil : 'வணக்கம், நாம் இப்போது இந்த புதிய அமைப்பை சோதிக்கிறோம்.'")
    print("  - English: 'Hello, I am testing real-time multilingual language detection.'")
    print("-" * 72)

    try:
        input("\nPress ENTER when you are ready to start speaking...")
    except EOFError:
        pass

    audio = record_with_feedback(duration=args.duration, sample_rate=sample_rate, device=args.device)
    analyze_audio(audio, sample_rate, engine)


if __name__ == "__main__":
    main()
