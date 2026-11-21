"""
Interactive ElevenLabs Instant Voice Cloning & Translation Testbench.

Records 10 seconds of user speech, uploads to ElevenLabs Instant Voice Clone,
and synthesizes test sentences in English, Telugu, and Hindi in the user's authentic cloned voice.

Usage:
    python test_elevenlabs_clone.py [--api-key YOUR_API_KEY] [--duration 10]
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import io

from app.services.elevenlabs_service import ElevenLabsVoiceService


def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Instant Voice Cloning Testbench")
    parser.add_argument("--api-key", default=None, help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--device", type=int, default=None, help="Microphone input device index")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("\n" + "="*70)
        print("🔑 ELEVENLABS API KEY REQUIRED")
        print("="*70)
        print("To use ElevenLabs Instant Voice Cloning:")
        print("1. Go to https://elevenlabs.io")
        print("2. Sign in, click your profile icon (bottom left) -> 'API Keys'")
        print("3. Copy your API Key.")
        print("="*70)
        api_key = input("\n👉 Paste your ElevenLabs API Key here and press Enter: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            sys.exit(1)

    print("\n[1/3] Initializing ElevenLabs Service...")
    service = ElevenLabsVoiceService(api_key=api_key)

    sample_rate = 16000
    duration = args.duration

    print("\n" + "="*70)
    print("🎙️ STEP 1: VOICE ENROLLMENT RECORDING")
    print("="*70)
    print(f"Please speak naturally in your normal voice for {duration:.0f} seconds.")
    print("Say a sentence like:")
    print("  'Hello! I am testing the multilingual real-time voice call translator.'")
    print("  'నమస్కారం! నేను రియల్ టైమ్ వాయిస్ అనువాద వ్యవస్థను పరీక్షిస్తున్నాను.'")
    input("\nPress ENTER when you are ready to speak...")

    print(f"\n🔴 RECORDING for {duration:.0f} seconds... SPEAK NOW!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=args.device)
    sd.wait()
    audio = audio.flatten()
    print("✅ Recording complete!")

    # Normalize audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.90

    print("\n" + "="*70)
    print("🚀 STEP 2: CREATING INSTANT VOICE CLONE IN ELEVENLABS")
    print("="*70)
    voice_name = f"User_Voice_{int(time.time())}"
    try:
        voice_id = service.clone_voice_from_audio(audio, sample_rate=sample_rate, voice_name=voice_name)
        print(f"🎉 SUCCESS! Cloned Voice ID: {voice_id}")
    except Exception as e:
        print(f"\n❌ Error creating voice clone: {e}")
        print("\nNote: ElevenLabs requires an active subscription or Starter plan ($1 trial) for Instant Voice Cloning.")
        sys.exit(1)

    print("\n" + "="*70)
    print("🔊 STEP 3: TESTING CLONED MULTILINGUAL SYNTHESIS")
    print("="*70)

    test_sentences = [
        ("English", "Hello, this is my real voice translated seamlessly in real time.", "en"),
        ("Telugu", "నమస్కారం! ఇది రియల్ టైమ్‌లో అనువదించబడిన నా నిజమైన స్వరం.", "te"),
        ("Hindi", "नमस्ते! यह वास्तविक समय में अनुवादित मेरी अपनी आवाज़ है।", "hi"),
    ]

    for lang_name, text, lang_code in test_sentences:
        print(f"\nSpeaking in {lang_name} using YOUR cloned voice:")
        print(f"  \"{text}\"")
        try:
            mp3_bytes = service.synthesize_cloned(
                text=text,
                voice_id=voice_id,
                language_code=lang_code,
                output_format="mp3_44100_128"
            )
            data, fs = sf.read(io.BytesIO(mp3_bytes))
            sd.play(data, fs)
            sd.wait()
            print("  ✅ Playback finished!")
        except Exception as e:
            print(f"  ❌ Synthesis failed for {lang_name}: {e}")

    print("\n" + "="*70)
    print("💬 INTERACTIVE MODE: Type any text to hear it in YOUR cloned voice")
    print("   Type 'exit' to quit.")
    print("="*70)

    while True:
        try:
            line = input("\n👉 Enter text to speak: ").strip()
            if not line or line.lower() in ("exit", "quit"):
                break
            mp3_bytes = service.synthesize_cloned(
                text=line,
                voice_id=voice_id,
                output_format="mp3_44100_128"
            )
            data, fs = sf.read(io.BytesIO(mp3_bytes))
            sd.play(data, fs)
            sd.wait()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\n👋 Done!")


if __name__ == "__main__":
    main()
