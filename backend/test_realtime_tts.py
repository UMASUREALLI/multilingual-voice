"""
Interactive Real-Time Multilingual Text-to-Speech (TTS) Verification Tool.
Zero disk writes: synthesizes directly into memory, measures latency,
and plays through speakers via AudioPlaybackEngine.
"""

import sys
import argparse
import time
from app.services.tts_service import StreamingTtsService
from app.audio.player import AudioPlaybackEngine


BENCHMARK_SENTENCES = [
    ("te", "female", "నమస్కారం! రియల్ టైమ్ వాయిస్ అనువాద వ్యవస్థను పరీక్షిస్తున్నాము."),
    ("hi", "female", "नमस्ते! हम इस नए रीयल-टाइम अनुवाद सिस्टम का परीक्षण कर रहे हैं."),
    ("ta", "male", "வணக்கம்! நிகழ்நேர மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."),
    ("en", "female", "Welcome to the real-time multilingual voice translation system."),
]


def run_benchmark(tts: StreamingTtsService, player: AudioPlaybackEngine, play_audio: bool = False):
    print("=" * 72)
    print("  MULTILINGUAL STREAMING TTS BENCHMARK (ZERO DISK WRITES)")
    print("=" * 72)

    for lang, gender, text in BENCHMARK_SENTENCES:
        voice = tts.get_voice(lang, gender)
        print(f"\n[Testing] Language: {lang.upper()} | Voice: {voice.voice_id} ({gender})")
        print(f"  Text: '{text}'")

        t0 = time.perf_counter()
        result = tts.synthesize(text=text, language_code=lang, gender=gender)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  -> Generated: {result.num_samples:,} samples @ {result.sample_rate} Hz")
        print(f"  -> Audio Duration: {result.duration_sec:.2f} seconds")
        print(f"  -> Synthesis Latency: {result.latency_ms:.1f} ms (Total: {elapsed_ms:.1f} ms)")
        print(f"  -> Streaming Chunks: {result.num_chunks}")

        if play_audio:
            print("  -> 🔊 Playing audio through speakers...")
            player.play(result.audio, sample_rate=result.sample_rate, blocking=True)
            print("  -> Playback complete.")

    print("\n" + "=" * 72)
    print("  TTS BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 72)


def run_interactive_tts(tts: StreamingTtsService, player: AudioPlaybackEngine, default_lang: str = "te", default_gender: str = "female"):
    print("\n" + "=" * 72)
    print("  INTERACTIVE TEXT-TO-SPEECH (TYPE ANY SENTENCE ➔ HEAR IT ALOUD)")
    print("=" * 72)
    print("Supported languages: te (Telugu), ta (Tamil), hi (Hindi), en (English), kn (Kannada), ml (Malayalam)")
    print("Type 'exit' or press Ctrl+C to stop.\n")

    lang = input(f"Enter Language code (te/ta/hi/en/kn) [default: {default_lang}] > ").strip().lower() or default_lang
    gender = input(f"Enter Voice Gender (female/male) [default: {default_gender}] > ").strip().lower() or default_gender
    voice = tts.get_voice(lang, gender)
    print(f"\n🟢 Ready! Using Neural Voice: {voice.voice_id} ({voice.name})")
    print("Type any text below and press Enter to hear it spoken aloud through your speakers!\n" + "-" * 72)

    while True:
        try:
            text = input(f"\n[{lang.upper()}] Type text > ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                break
            t0 = time.perf_counter()
            res = tts.synthesize(text=text, language_code=lang, gender=gender)
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  🎙️ Synthesized: {res.duration_sec:.2f}s audio in {ms:.0f}ms")
            print("  🔊 Playing aloud through speakers...")
            player.play(res.audio, sample_rate=res.sample_rate, blocking=True)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive TTS.")
            break


def main():
    parser = argparse.ArgumentParser(description="Multilingual TTS Diagnostic Tool")
    parser.add_argument("--text", type=str, default=None, help="Custom text to synthesize")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive type-and-hear CLI")
    parser.add_argument("--lang", type=str, default="te", help="Language code (te, hi, ta, en, kn, etc.)")
    parser.add_argument("--gender", type=str, default="female", choices=["female", "male"])
    parser.add_argument("--rate", type=str, default="+0%", help="Speaking rate adjustment (e.g. +10%, -5%)")
    parser.add_argument("--pitch", type=str, default="+0Hz", help="Pitch adjustment (e.g. +20Hz, -15Hz)")
    parser.add_argument("--play", action="store_true", help="Play synthesized audio out loud through speakers")

    args = parser.parse_args()

    tts = StreamingTtsService()
    player = AudioPlaybackEngine()

    if args.interactive:
        run_interactive_tts(tts, player, default_lang=args.lang, default_gender=args.gender)
        return

    if args.text:
        voice = tts.get_voice(args.lang, args.gender)
        print(f"\nSynthesizing with {voice.voice_id} ({voice.name})...")
        print(f"Text: '{args.text}'")
        res = tts.synthesize(
            text=args.text,
            language_code=args.lang,
            gender=args.gender,
            rate=args.rate,
            pitch=args.pitch
        )
        print(f"Success! Duration: {res.duration_sec:.2f}s | Latency: {res.latency_ms:.1f}ms | Sample Rate: {res.sample_rate}Hz")
        if args.play:
            print("🔊 Playing audio through speakers...")
            player.play(res.audio, sample_rate=res.sample_rate, blocking=True)
            print("Done.")
    else:
        run_benchmark(tts, player, play_audio=args.play)


if __name__ == "__main__":
    main()
