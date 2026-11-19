"""
Unit Tests for Consent-Based Voice Cloning Service & Acoustic Profiling.
Verifies ethical authorization check, acoustic F0 pitch estimation,
ephemeral in-memory security, and prosody mapping.
"""

import unittest
import numpy as np

from app.models.speaker import ConsentStatus
from app.services.voice_cloner import VoiceCloningService


class TestVoiceCloningService(unittest.TestCase):
    def setUp(self):
        self.cloner = VoiceCloningService()

    def test_english_consent_granted(self):
        """Verify explicit English consent authorization statement is accepted."""
        text = "I authorize this AI system to clone and use my voice for real-time translation."
        res = self.cloner.verify_consent(text, language_code="en")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ConsentStatus.GRANTED)
        self.assertGreaterEqual(res.similarity_score, 0.8)

    def test_telugu_consent_granted(self):
        """Verify Telugu verbal consent statement is accepted."""
        text = "నా స్వరాన్ని అనువాదం కోసం ఉపయోగించడానికి నేను అనుమతిస్తున్నాను"
        res = self.cloner.verify_consent(text, language_code="te")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ConsentStatus.GRANTED)

    def test_hindi_consent_granted(self):
        """Verify Hindi verbal consent statement is accepted."""
        text = "मैं इस अनुवाद के लिए अपनी आवाज का उपयोग करने की अनुमति देता हूं"
        res = self.cloner.verify_consent(text, language_code="hi")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ConsentStatus.GRANTED)

    def test_unauthorized_speech_rejected(self):
        """Verify random or non-consensual utterance is strictly rejected."""
        text = "Hello, what is the weather like today in Hyderabad?"
        res = self.cloner.verify_consent(text, language_code="en")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, ConsentStatus.REJECTED)

    def test_acoustic_feature_extraction_male_pitch(self):
        """Verify F0 pitch estimation accurately identifies 120 Hz male tone."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 120 Hz fundamental frequency with harmonics
        signal = 0.5 * np.sin(2 * np.pi * 120.0 * t) + 0.2 * np.sin(2 * np.pi * 240.0 * t)

        features = self.cloner.extract_acoustic_features(signal, sample_rate=sr)
        self.assertAlmostEqual(features.f0_median_hz, 120.0, delta=5.0)
        self.assertEqual(features.estimated_gender, "male")

    def test_acoustic_feature_extraction_female_pitch(self):
        """Verify F0 pitch estimation accurately identifies 220 Hz female tone."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 220 Hz fundamental frequency with harmonics
        signal = 0.5 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)

        features = self.cloner.extract_acoustic_features(signal, sample_rate=sr)
        self.assertAlmostEqual(features.f0_median_hz, 220.0, delta=8.0)
        self.assertEqual(features.estimated_gender, "female")

    def test_enrollment_with_valid_consent(self):
        """Verify speaker enrollment stores ephemeral profile in memory."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 130.0 * t)

        consent_text = "I authorize this AI system to clone and use my voice for real-time translation."
        profile = self.cloner.enroll_speaker(
            speaker_id="caller_alice",
            spoken_consent_text=consent_text,
            audio_samples=audio,
            sample_rate=sr,
            consent_language="en"
        )

        self.assertEqual(profile.speaker_id, "caller_alice")
        self.assertEqual(profile.consent_status, ConsentStatus.GRANTED)
        self.assertIsNotNone(profile.acoustic_features)

        # Retrieval check
        retrieved = self.cloner.get_profile("caller_alice")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.speaker_id, "caller_alice")

    def test_enrollment_without_consent_raises_permission_error(self):
        """Verify attempting to enroll without spoken consent raises PermissionError."""
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        with self.assertRaises(PermissionError):
            self.cloner.enroll_speaker(
                speaker_id="caller_mallory",
                spoken_consent_text="Just start recording my call now.",
                audio_samples=audio,
                sample_rate=sr,
                consent_language="en"
            )

    def test_revoke_consent_and_clear_session(self):
        """Verify revoking consent and clearing session leaves zero profile residue in RAM."""
        sr = 16000
        audio = 0.3 * np.sin(2 * np.pi * 200.0 * np.linspace(0, 1.0, sr))
        consent = "I authorize this AI system to clone and use my voice for real-time translation."
        self.cloner.enroll_speaker("caller_bob", consent, audio, sr, "en")

        # Revoke
        self.cloner.revoke_consent("caller_bob")
        prof = self.cloner.get_profile("caller_bob")
        self.assertEqual(prof.consent_status, ConsentStatus.REVOKED)
        self.assertIsNone(prof.acoustic_features)

        # Clear session
        self.cloner.clear_session()
        self.assertIsNone(self.cloner.get_profile("caller_bob"))


if __name__ == "__main__":
    unittest.main()
