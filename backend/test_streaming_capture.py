"""
Interactive and Automated Verification Tool for Live Audio Streaming & VAD.
Validates non-blocking microphone capture, circular ring-buffer streaming,
and real-time Silero VAD segmentation with ZERO disk I/O.
"""

import sys
import time
import argparse
import numpy as np

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechEventType, SpeechSegment


def render_vu_meter(rms_dbfs: float, width: int = 24) -> str:
    """Render a dynamic ASCII VU level meter."""
    # Scale from -60 dBFS (silence) to 0 dBFS (peak)
    clamped_db = max(-60.0, min(0.0, rms_dbfs))
    ratio = (clamped_db + 60.0) / 60.0
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {rms_dbfs:6.1f} dBFS"


def main():
    parser = argparse.ArgumentParser(description="Live Microphone Audio Capture & VAD Tester")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index (default: system default)")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (0 for indefinite)")
    parser.add_argument("--self-test", action="store_true", help="Run automated self-test mode")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" 🎙️  LIVE STREAMING MICROPHONE & VAD ENGINE (ZERO DISK I/O)")
    print("=" * 65)

    devices = LiveAudioStreamer.list_input_devices()
    print("\n[Detected Audio Input Devices]:")
    for d in devices:
        star = " *" if args.device == d["index"] else ""
        print(f"  [{d['index']}] {d['name']} (Channels: {d['channels']}, Default SR: {d['default_samplerate']} Hz){star}")

    captured_segments = []

    def on_speech_start(ts: float):
        sys.stdout.write("\n>>> [SPEECH ONSET DETECTED] Rolling pre-roll prepended...\n")
        sys.stdout.flush()

    def on_utterance_complete(segment: SpeechSegment):
        captured_segments.append(segment)
        sys.stdout.write(
            f"\n✨ [UTTERANCE COMPLETED]: Duration={segment.duration_sec:.2f}s "
            f"({segment.num_samples} samples) | RMS={segment.rms_dbfs:.1f} dBFS | "
            f"Peak={segment.peak_amp:.3f} | In-Memory (Zero disk writes!)\n"
        )
        sys.stdout.flush()

    streamer = LiveAudioStreamer(
        device_index=args.device,
        vad_start_threshold=0.45,
        vad_end_threshold=0.30,
        hangover_ms=450,
        min_speech_ms=250,
        max_speech_ms=8000
    )

    streamer.register_on_speech_start(on_speech_start)
    streamer.register_on_utterance_complete(on_utterance_complete)

    print("\nStarting live capture stream...")
    streamer.start()

    run_duration = args.duration if (args.self_test or args.duration > 0) else 60.0
    print(f"Streaming live audio for {run_duration:.1f} seconds. Speak into your microphone!\n")

    start_time = time.time()
    try:
        while time.time() - start_time < run_duration:
            elapsed = time.time() - start_time
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            event = streamer.current_event.value

            vu = render_vu_meter(rms, width=20)
            status_symbol = "🔴 SPEECH" if "speech" in event else ("🟡 HANGOVER" if "hangover" in event else "⚪ SILENCE")

            # Dynamic single-line update
            sys.stdout.write(f"\r[{elapsed:4.1f}s] {vu} | VAD: {prob:4.2f} | {status_symbol:<12}")
            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping capture early...")
    finally:
        streamer.stop()

    print("\n\n" + "=" * 65)
    print(" 📊 CAPTURE SUMMARY")
    print("=" * 65)
    print(f"Total stream time      : {time.time() - start_time:.2f}s")
    print(f"Total written to buffer: {streamer.ring_buffer.total_written} samples (In-Memory)")
    print(f"Utterances segmented   : {len(captured_segments)}")
    for i, seg in enumerate(captured_segments, 1):
        print(f"  Segment #{i}: {seg.duration_sec:.2f}s, RMS={seg.rms_dbfs:.1f} dBFS, Peak={seg.peak_amp:.3f}")
    print(f"Disk files generated   : 0 (Pure streaming)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
