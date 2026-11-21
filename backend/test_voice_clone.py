"""
Interactive Voice Cloning & Acoustic Profiling Testbench.
Captures live speaker audio from microphone, extracts F0 fundamental pitch,
vocal tract gender, and timbre brightness in RAM (zero disk writes),
and synthesizes cloned multilingual neural speech (English, Telugu, Hindi, Tamil)
adapted to the speaker's real vocal characteristics.
"""

import sys
import time
import argparse
import numpy as np
import sounddevice as sd

from app.services.voice_cloner import VoiceCloningService
from app.services.tts_service import StreamingTtsService
from app.audio.player import AudioPlaybackEngine
from app.models.speaker import AcousticFeatures


def print_banner():
    print("\n" + "=" * 74)
    print("  🎙️ ETHICAL AI VOICE CLONING & PROSODY ADAPTATION TESTBENCH")
    print("  Real-Time F0 Tracking • Zero Disk Writes • Multilingual Neural Voices")
    print("=" * 74)


def record_voice_sample(duration_sec: float = 4.5, sample_rate: int = 16000) -> np.ndarray:
    """Record high-fidelity voice sample from default microphone."""
    print(f"\n🎙️  GET READY TO SPEAK...")
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print("   🔴 RECORDING NOW! Speak naturally (e.g. 'Hello, this is my voice for AI cloning')...", flush=True)

    audio_chunks = []

    def audio_callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback):
        start_t = time.time()
        while time.time() - start_t < duration_sec:
            elapsed = time.time() - start_t
            remaining = duration_sec - elapsed
            bars = int((elapsed / duration_sec) * 30)
            sys.stdout.write(f"\r   [{'█' * bars}{'░' * (30 - bars)}] {elapsed:.1f}s / {duration_sec:.1f}s")
            sys.stdout.flush()
            time.sleep(0.1)

    print("\n   ⏹️ Recording complete! Analyzing vocal biometric profile in RAM...")

    if not audio_chunks:
        return np.zeros(sample_rate, dtype=np.float32)

    recorded = np.concatenate(audio_chunks, axis=0).flatten()
    # Normalize peak
    peak = float(np.max(np.abs(recorded)))
    if peak > 0.001:
        recorded = (recorded / peak * 0.90).astype(np.float32)
    return recorded


def display_acoustic_profile(features: AcousticFeatures, pitch_adj: str, rate_adj: str):
    print("\n" + "-" * 74)
    print("  🧬 EXTRACTED ACOUSTIC VOICE PROFILE (IN-MEMORY)")
    print("-" * 74)
    print(f"  • Estimated Vocal Gender : {features.estimated_gender.upper()} "
          f"({'Human male baseline: 85-155 Hz' if features.estimated_gender == 'male' else 'Human female baseline: 165-260 Hz'})")
    print(f"  • Median Pitch (F0)      : {features.f0_median_hz:.1f} Hz (Variability: ±{features.f0_std_hz:.1f} Hz)")
    print(f"  • Timbre Brightness      : {features.spectral_centroid_hz:.1f} Hz (Spectral Centroid)")
    print(f"  • Energy Level           : {features.energy_rms_dbfs:.1f} dBFS")
    print(f"  • Speech Duration        : {features.audio_duration_sec:.2f} seconds")
    print(f"  • Dynamic Pitch Shift    : {pitch_adj}")
    print(f"  • Dynamic Tempo Shift    : {rate_adj}")
    print("-" * 74)


def main():
    parser = argparse.ArgumentParser(description="Voice Cloning & Acoustic Adaptation Testbench")
    parser.add_argument("--gender", type=str, choices=["male", "female"], default=None,
                        help="Override detected vocal gender")
    parser.add_argument("--duration", type=float, default=4.5,
                        help="Duration of voice enrollment sample in seconds (default: 4.5)")
    args = parser.parse_args()

    print_banner()

    cloner = VoiceCloningService()
    tts = StreamingTtsService()
    player = AudioPlaybackEngine()

    # Step 1: Capture or generate voice sample
    audio = record_voice_sample(duration_sec=args.duration)

    if len(audio) < 16000:
        print("❌ Audio was too short. Please run again and speak clearly.")
        return

    # Step 2: Extract acoustic features
    features = cloner.extract_acoustic_features(audio, sample_rate=16000)
    if args.gender:
        features.estimated_gender = args.gender

    pitch_adj, rate_adj = cloner.calculate_prosody_adjustments(features)
    display_acoustic_profile(features, pitch_adj, rate_adj)

    gender = features.estimated_gender

    # Step 3: Multilingual Voice Showcase using cloned profile
    test_phrases = [
        ("en", "English", f"Hello! This is my cloned voice speaking in English, adapted to my pitch and vocal characteristics."),
        ("te", "Telugu", f"నమస్కారం! ఇది నా గొంతును పోలిన కృత్రిమ మేధస్సు క్లోన్ వాయిస్."),
        ("hi", "Hindi", f"नमस्ते! यह मेरी पिच और टोन में तैयार की गई क्लोन की गई आवाज है।")
    ]

    print("\n🔊 Synthesizing and playing cloned speech across languages...\n")

    for lang_code, lang_name, phrase in test_phrases:
        print(f"  🗣️ [{lang_name}] Synthesizing ({gender.title()} • Pitch: {pitch_adj}):")
        print(f"     \"{phrase}\"")

        t0 = time.perf_counter()
        res = tts.synthesize(
            text=phrase,
            language_code=lang_code,
            gender=gender,
            pitch=pitch_adj,
            rate=rate_adj
        )
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"     -> Neural Voice: {res.voice_id} | Latency: {latency:.1f}ms | Playing aloud...")

        player.play(res.audio, sample_rate=res.sample_rate, blocking=True)
        time.sleep(0.6)

    # Step 4: Interactive Typing Mode in Cloned Voice
    print("\n" + "=" * 74)
    print("⌨️  INTERACTIVE CLONED VOICE TYPING MODE")
    print(f"   Active Cloned Profile: {gender.upper()} • Pitch: {pitch_adj} • Tempo: {rate_adj}")
    print("   Type any sentence to hear it spoken aloud in your cloned voice.")
    print("   Prefix with language code to switch: 'te: <text>', 'hi: <text>', or default English.")
    print("   Type 'exit' or 'quit' to end.")
    print("=" * 74 + "\n")

    while True:
        try:
            line = input("Your sentence > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                print("Exiting voice clone testbench.")
                break

            target_lang = "en"
            synth_text = line
            if line.startswith("te:") or line.startswith("te "):
                target_lang = "te"
                synth_text = line[3:].strip()
            elif line.startswith("hi:") or line.startswith("hi "):
                target_lang = "hi"
                synth_text = line[3:].strip()
            elif line.startswith("ta:") or line.startswith("ta "):
                target_lang = "ta"
                synth_text = line[3:].strip()
            elif line.startswith("en:") or line.startswith("en "):
                target_lang = "en"
                synth_text = line[3:].strip()

            print(f"   🔊 Synthesizing in {target_lang.upper()} ({gender} {pitch_adj})...")
            res = tts.synthesize(
                text=synth_text,
                language_code=target_lang,
                gender=gender,
                pitch=pitch_adj,
                rate=rate_adj
            )
            player.play(res.audio, sample_rate=res.sample_rate, blocking=True)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting voice clone testbench.")
            break


if __name__ == "__main__":
    main()
