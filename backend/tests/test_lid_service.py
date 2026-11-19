"""
Automated unit tests for Language Identification (LID) Engine and DynamicLanguageTracker.
"""

import sys
import os
import unittest
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.language import SUPPORTED_LANGUAGES, LidDecision
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker


class TestLanguageMetadata(unittest.TestCase):

    def test_supported_languages_presence(self):
        for code in ["te", "hi", "ta", "en", "kn", "ml", "mr", "bn"]:
            self.assertIn(code, SUPPORTED_LANGUAGES)
            lang = SUPPORTED_LANGUAGES[code]
            self.assertEqual(lang.code, code)
            self.assertTrue(len(lang.name) > 0)
            self.assertTrue(len(lang.native_name) > 0)


class TestScriptLID(unittest.TestCase):

    def setUp(self):
        self.engine = LanguageDetectionEngine(whisper_model=None)

    def test_telugu_script_identification(self):
        text = "నమస్కారం, మీరు ఎలా ఉన్నారు?"
        decision = self.engine.identify_script(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.language_code, "te")
        self.assertEqual(decision.language_name, "Telugu")
        self.assertGreaterEqual(decision.confidence, 0.85)
        self.assertEqual(decision.detection_source, "unicode_script")

    def test_hindi_script_identification(self):
        text = "नमस्ते, आप कैसे हैं?"
        decision = self.engine.identify_script(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.language_code, "hi")
        self.assertEqual(decision.language_name, "Hindi")
        self.assertGreaterEqual(decision.confidence, 0.85)

    def test_tamil_script_identification(self):
        text = "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
        decision = self.engine.identify_script(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.language_code, "ta")
        self.assertEqual(decision.language_name, "Tamil")
        self.assertGreaterEqual(decision.confidence, 0.85)

    def test_marathi_disambiguation(self):
        # Marathi unique retroflex lateral approximant 'ळ'
        text = "नमस्कार, बाळ कसे आहे?"
        decision = self.engine.identify_script(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.language_code, "mr")
        self.assertEqual(decision.language_name, "Marathi")

    def test_english_latin_identification(self):
        text = "Hello team, real-time call translation is working."
        decision = self.engine.identify_script(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.language_code, "en")
        self.assertEqual(decision.language_name, "English")

    def test_empty_or_symbolic_text(self):
        self.assertIsNone(self.engine.identify_script(""))
        self.assertIsNone(self.engine.identify_script("12345 !@#$%^"))


class TestDynamicLanguageTracker(unittest.TestCase):

    def setUp(self):
        self.engine = LanguageDetectionEngine(whisper_model=None)
        self.tracker = DynamicLanguageTracker(
            engine=self.engine,
            initial_language="te",
            switch_confidence_threshold=0.72,
            consecutive_turns_for_switch=2
        )

    def test_initial_state(self):
        self.assertEqual(self.tracker.current_language, "te")

    def test_high_confidence_immediate_switch(self):
        switches = []
        self.tracker.register_switch_listener(lambda o, n, c: switches.append((o, n, c)))

        # High confidence English text (>0.82)
        dec = self.tracker.process_turn(transcript="Hello everyone, how are you today?")
        self.assertEqual(dec.language_code, "en")
        self.assertTrue(dec.is_language_switch)
        self.assertEqual(self.tracker.current_language, "en")
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0][0], "te")
        self.assertEqual(switches[0][1], "en")

    def test_hysteresis_prevents_single_moderate_noise(self):
        # Current language is 'te'
        self.assertEqual(self.tracker.current_language, "te")

        # Simulate a mock spoken decision with moderate confidence 0.60 for 'hi'
        mock_spoken = self.engine._build_decision("hi", 0.60, "spoken_audio", 50.0)
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_spoken

        # Turn 1: Candidate is 'hi' at 0.60 -> Should NOT switch yet (holds 'te')
        dummy_audio = np.zeros(1600, dtype=np.float32)
        dec1 = self.tracker.process_turn(audio=dummy_audio)
        self.assertEqual(dec1.language_code, "te")
        self.assertEqual(self.tracker.current_language, "te")

        # Turn 2: Second consecutive turn with 'hi' at 0.60 -> NOW switches to 'hi'
        dec2 = self.tracker.process_turn(audio=dummy_audio)
        self.assertEqual(dec2.language_code, "hi")
        self.assertTrue(dec2.is_language_switch)
        self.assertEqual(self.tracker.current_language, "hi")

    def test_auto_initial_locks_and_prevents_turn_flapping(self):
        """Simulates the user speaking Tamil: locks on Turn 1, resists Kannada/Hindi flapping on quiet turns."""
        tracker = DynamicLanguageTracker(
            engine=self.engine,
            initial_language="auto",
            switch_confidence_threshold=0.80,
            consecutive_turns_for_switch=2
        )
        self.assertFalse(tracker.is_locked)
        self.assertEqual(tracker.current_language, "auto")

        dummy_audio = np.zeros(16000, dtype=np.float32)

        # Turn 1: Speaker speaks Tamil clearly -> locks session to 'ta'
        mock_ta = self.engine._build_decision("ta", 0.92, "spoken_audio", 50.0)
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_ta
        dec1 = tracker.process_turn(audio=dummy_audio)
        self.assertTrue(tracker.is_locked)
        self.assertEqual(dec1.language_code, "ta")
        self.assertEqual(tracker.current_language, "ta")

        # Turn 2: Faint turn where acoustic noise gives 'kn' (0.38) and 'ta' (0.30)
        # Continuity bonus (+0.25) keeps 'ta' locked (0.30 + 0.25 = 0.55 > 0.38)
        mock_kn = self.engine._build_decision(
            "kn", 0.38, "spoken_audio", 50.0,
            candidates=[("kn", "Kannada", 0.38), ("ta", "Tamil", 0.30)]
        )
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_kn
        dec2 = tracker.process_turn(audio=dummy_audio)
        self.assertEqual(dec2.language_code, "ta")
        self.assertEqual(tracker.current_language, "ta")

        # Turn 3: Single turn with Hindi noise (0.45)
        # Hysteresis requires 2 consecutive turns -> 'ta' remains locked!
        mock_hi = self.engine._build_decision(
            "hi", 0.45, "spoken_audio", 50.0,
            candidates=[("hi", "Hindi", 0.45)]
        )
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_hi
        dec3 = tracker.process_turn(audio=dummy_audio)
        self.assertEqual(dec3.language_code, "ta")
        self.assertEqual(tracker.current_language, "ta")

        # Turn 4: User continues speaking Tamil
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_ta
        dec4 = tracker.process_turn(audio=dummy_audio)
        self.assertEqual(dec4.language_code, "ta")
        self.assertEqual(tracker.current_language, "ta")

    def test_noise_turn_does_not_lock_session(self):
        """Pure noise/breath must NOT lock the session to an erroneous language."""
        tracker = DynamicLanguageTracker(
            engine=self.engine,
            initial_language="auto",
        )
        self.assertFalse(tracker.is_locked)
        self.assertEqual(tracker.current_language, "auto")

        # Noise / breath: low confidence (0.40) and short (0.3s)
        mock_noise = self.engine._build_decision("te", 0.40, "spoken_audio", 50.0)
        self.engine.identify_spoken = lambda audio, sample_rate=16000: mock_noise
        dec = tracker.process_turn(audio=np.zeros(4800, dtype=np.float32))

        # Must NOT lock on noise/breath!
        self.assertFalse(tracker.is_locked)
        self.assertEqual(tracker.current_language, "auto")


class TestSpokenLIDRealAudio(unittest.TestCase):

    def test_spoken_english_sample(self):
        import soundfile as sf
        sample_path = r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\transcripts\demo_spoken_sample.wav"
        if not os.path.exists(sample_path):
            self.skipTest("demo_spoken_sample.wav not found")

        audio, sr = sf.read(sample_path)
        engine = LanguageDetectionEngine()
        decision = engine.identify_spoken(audio, sample_rate=sr)

        self.assertEqual(decision.language_code, "en")
        self.assertEqual(decision.language_name, "English")
        self.assertGreaterEqual(decision.confidence, 0.80)
        self.assertTrue(decision.latency_ms > 0)
