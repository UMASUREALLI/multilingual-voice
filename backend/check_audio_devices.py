"""
Interactive Audio Diagnostic & Sound Check Tool.
Tests both your microphone and your speakers/earbuds to guarantee audio works before starting calls.
"""

import time
import sounddevice as sd
import numpy as np
from app.audio.streamer import LiveAudioStreamer
from app.audio.player import AudioPlaybackEngine


def main():
    print("\n" + "=" * 70)
    print("  AUDIO HARDWARE DIAGNOSTIC & SOUND CHECK TOOL")
    print("=" * 70)

    # 1. Detect active hardware
    in_idx = LiveAudioStreamer.auto_detect_input_device()
    out_idx = AudioPlaybackEngine.auto_detect_output_device(in_idx)
    speaker_idx = AudioPlaybackEngine.get_laptop_speakers()

    dev_list = sd.query_devices()
    def_in = sd.default.device[0]
    def_out = sd.default.device[1]

    print("\n[Detected Audio Hardware]")
    print(f"  • Recommended Earbuds Mic  : [{in_idx}] {dev_list[in_idx]['name'] if in_idx is not None else 'None'}")
    print(f"  • Recommended Earbuds Spk  : [{out_idx}] {dev_list[out_idx]['name'] if out_idx is not None else 'None'}")
    print(f"  • Laptop Physical Speakers : [{speaker_idx}] {dev_list[speaker_idx]['name'] if speaker_idx is not None else 'None'}")
    print(f"  • Windows Default In / Out : [{def_in}] / [{def_out}]")
    print("-" * 70)

    # 2. Test Earbuds / Headset Audio Playback
    test_device = out_idx if out_idx is not None else def_out
    print(f"\n[Test 1/3] Playing a test chime to YOUR EARBUDS (Device {test_device}: {dev_list[test_device]['name']})...")
    sr = 24000
    t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
    # Pleasant two-tone chime
    chime = (0.25 * (np.sin(2 * np.pi * 523.25 * t) + np.sin(2 * np.pi * 659.25 * t)) * np.exp(-t * 4)).astype(np.float32)
    try:
        sd.play(chime, samplerate=sr, device=test_device, blocking=True)
        print("  -> Finished playing. Did you hear the chime in your earbuds?")
    except Exception as e:
        print(f"  ❌ Playback error on device {test_device}: {e}")

    # 3. Test Laptop Speakers Playback
    if speaker_idx is not None:
        print(f"\n[Test 2/3] Playing a test chime to LAPTOP SPEAKERS (Device {speaker_idx}: {dev_list[speaker_idx]['name']})...")
        try:
            sd.play(chime, samplerate=sr, device=speaker_idx, blocking=True)
            print("  -> Finished playing. Did you hear the chime from your laptop speakers?")
        except Exception as e:
            print(f"  ❌ Playback error on device {speaker_idx}: {e}")

    # 4. Test Microphone Capture & Immediate Loopback
    mic_test = in_idx if in_idx is not None else def_in
    print(f"\n[Test 3/3] Testing Microphone recording for 3 seconds...")
    print(f"  👉 Speak something into your microphone now (Device [{mic_test}])...")
    try:
        recording = sd.rec(int(sr * 3.0), samplerate=sr, channels=1, dtype="float32", device=mic_test)
        sd.wait()
        rms = float(np.sqrt(np.mean(recording**2)))
        peak = float(np.max(np.abs(recording)))
        print(f"  -> Recording complete! (RMS: {rms:.4f}, Peak: {peak:.4f})")

        if peak > 0.005:
            print("  -> Normalizing volume and playing your recorded voice back to you...")
            norm_rec = (recording / peak * 0.90).astype(np.float32)
            sd.play(norm_rec, samplerate=sr, device=test_device, blocking=True)
            print("  ✅ Voice loopback complete!")
        else:
            print("  ⚠️ Audio was very quiet. Make sure your microphone is not muted in Windows!")
    except Exception as e:
        print(f"  ❌ Microphone recording error on device {mic_test}: {e}")

    print("\n" + "=" * 70)
    print("  AUDIO CHECK COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
