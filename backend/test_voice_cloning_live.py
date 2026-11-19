"""
Interactive Voice Cloning & Consent Verification Diagnostic Tool.
Tests verbal authorization check, acoustic pitch/tempo profiling,
and voice adaptation with Edge-TTS in memory.
"""

import sys
import argparse
import time
import numpy as np

from app.models.speaker import MANDATORY_CONSENT_PHRASES, ConsentStatus
from app.services.voice_cloner import VoiceCloningService
from app.services.tts_service import StreamingTtsService
from app.audio.player import AudioPlaybackEngine


def main():
    parser = argparse.ArgumentParser(description="Consent-Based Voice Cloning Live Diagnostic")
    parser.add_argument("--consent-text", type=str, default=None, help="Spoken consent phrase to test")
    parser.add_argument("--lang", type=str, default="en", help="Language code (en, te, hi, ta)")
    parser.add_argument("--speaker-id", type=str, default="speaker_01", help="Speaker identifier")
    parser.add_argument("--test-pitch", type=float, default=125.0, help="Simulated speaker F0 in Hz (e.g. 120 for male, 215 for female)")
    parser.add_argument("--synthesize-test", action="store_true", help="Synthesize adapted voice sample")
    parser.add_argument("--play", action="store_true", help="Play synthesized adapted voice out loud")

    args = parser.parse_args()

    cloner = VoiceCloningService()
    tts = StreamingTtsService()
    player = AudioPlaybackEngine()

    print("=" * 72)
    print("  CONSENT-BASED VOICE CLONING DIAGNOSTIC & VERIFICATION")
    print("=" * 72)

    # 1. Spoken Consent Verification
    consent_text = args.consent_text or (
        MANDATORY_CONSENT_PHRASES.get(args.lang, MANDATORY_CONSENT_PHRASES["en"])[0]
    )
    print(f"\n[1] Verifying Spoken Consent Statement:")
    print(f"  Speaker ID : {args.speaker_id}")
    print(f"  Language   : {args.lang.upper()}")
    print(f"  Transcript : '{consent_text}'")

    verification = cloner.verify_consent(consent_text, language_code=args.lang)
    print(f"  -> Valid Consent? : {'✅ GRANTED' if verification.is_valid else '❌ REJECTED'}")
    print(f"  -> Similarity     : {verification.similarity_score * 100:.1f}%")
    print(f"  -> Status Msg     : {verification.message}")

    if not verification.is_valid:
        print("\n❌ Enrollment aborted: Ethical AI consent policy strictly forbids voice modeling without verbal consent.")
        return

    # 2. Acoustic Feature Profiling
    print(f"\n[2] Extracting Acoustic Profiling Features in Memory:")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate test vocal harmonic signal based on --test-pitch
    f0 = args.test_pitch
    test_audio = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.25 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)

    profile = cloner.enroll_speaker(
        speaker_id=args.speaker_id,
        spoken_consent_text=consent_text,
        audio_samples=test_audio,
        sample_rate=sr,
        consent_language=args.lang
    )

    feat = profile.acoustic_features
    print(f"  -> Estimated F0 (Pitch)   : {feat.f0_median_hz:.1f} Hz")
    print(f"  -> Inferred Gender        : {feat.estimated_gender.upper()}")
    print(f"  -> Timbre Spectral Center : {feat.spectral_centroid_hz:.1f} Hz")
    print(f"  -> Prosody Pitch Mod      : {profile.pitch_adjustment}")
    print(f"  -> Prosody Rate Mod       : {profile.rate_adjustment}")

    # 3. Voice Adaptation Synthesis
    if args.synthesize_test or args.play:
        print(f"\n[3] Synthesizing Adapted Voice Sample across Languages:")
        test_samples = [
            ("te", "నమస్కారం, ఇది మీ స్వరాన్ని పోలిన అనువాదం."),
            ("en", "Hello, this is a translated voice with your cloned pitch profile.")
        ]
        for t_lang, sample_text in test_samples:
            base_voice = tts.get_voice(t_lang, feat.estimated_gender)
            print(f"  Target: {t_lang.upper()} | Base Voice: {base_voice.voice_id} | Pitch: {profile.pitch_adjustment}")
            res = tts.synthesize(
                text=sample_text,
                language_code=t_lang,
                gender=feat.estimated_gender,
                pitch=profile.pitch_adjustment,
                rate=profile.rate_adjustment
            )
            print(f"  -> Generated: {res.duration_sec:.2f}s audio in {res.latency_ms:.1f}ms")
            if args.play:
                print("  -> 🔊 Playing...")
                player.play(res.audio, sample_rate=res.sample_rate, blocking=True)

    # 4. Ephemeral Vault Cleanup
    print(f"\n[4] Testing Session Teardown & RAM Clearance:")
    cloner.clear_session()
    print("  -> Enrolled profiles cleared from RAM. Zero disk artifacts created.")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
