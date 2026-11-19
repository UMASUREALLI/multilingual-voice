"""
Unit Tests for Multilingual Streaming Text-to-Speech (TTS) Service & Playback Engine.
Zero disk writes, in-memory audio verification, voice registry lookup, and echo prevention.
"""

import unittest
import numpy as np

from app.models.tts import SpeechSynthesisRequest, SpeechSynthesisResult
from app.services.tts_service import StreamingTtsService, DEFAULT_VOICE_REGISTRY
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer


class TestStreamingTtsService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_service = StreamingTtsService()

    def test_voice_registry_resolution(self):
        """Verify proper voice mapping for Indian and international languages."""
        # Telugu Female
        te_female = self.tts_service.get_voice("te", "female")
        self.assertEqual(te_female.voice_id, "te-IN-ShrutiNeural")
        self.assertEqual(te_female.gender, "female")

        # Telugu Male
        te_male = self.tts_service.get_voice("te", "male")
        self.assertEqual(te_male.voice_id, "te-IN-MohanNeural")
        self.assertEqual(te_male.gender, "male")

        # Hindi
        hi_female = self.tts_service.get_voice("hi", "female")
        self.assertEqual(hi_female.voice_id, "hi-IN-SwaraNeural")

        # Tamil
        ta_male = self.tts_service.get_voice("ta", "male")
        self.assertEqual(ta_male.voice_id, "ta-IN-ValluvarNeural")

        # Fallback for unknown language
        fallback = self.tts_service.get_voice("xx", "female")
        self.assertIn("en", fallback.locale)

    def test_synthesize_english_to_numpy(self):
        """Verify in-memory synthesis of English text produces valid float32 NumPy audio."""
        result = self.tts_service.synthesize(
            text="Hello, testing real-time voice translation.",
            language_code="en",
            gender="female"
        )
        self.assertIsInstance(result, SpeechSynthesisResult)
        self.assertIsInstance(result.audio, np.ndarray)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertGreater(len(result.audio), 0)
        self.assertGreater(result.sample_rate, 8000)
        self.assertGreater(result.duration_sec, 0.5)
        self.assertGreater(result.num_chunks, 0)

    def test_synthesize_telugu_to_numpy(self):
        """Verify in-memory synthesis of Telugu text."""
        result = self.tts_service.synthesize(
            text="నమస్కారం, మీరు ఎలా ఉన్నారు?",
            language_code="te",
            gender="female"
        )
        self.assertIsInstance(result, SpeechSynthesisResult)
        self.assertIsInstance(result.audio, np.ndarray)
        self.assertGreater(len(result.audio), 0)
        self.assertEqual(result.voice_id, "te-IN-ShrutiNeural")

    def test_lru_cache_hit(self):
        """Verify LRU cache returns instant result for repeated phrase."""
        text = "ధన్యవాదాలు"
        # First call: network synthesis
        r1 = self.tts_service.synthesize(text=text, language_code="te", gender="female")
        # Second call: cache hit
        r2 = self.tts_service.synthesize(text=text, language_code="te", gender="female")

        self.assertLess(r2.latency_ms, 5.0)  # Should be < 5ms (cache hit)
        np.testing.assert_array_equal(r1.audio, r2.audio)


class MockStreamer:
    def __init__(self):
        self.muted = False
        self.ducking = 1.0

    def set_muted(self, val: bool):
        self.muted = val

    def set_ducking(self, val: float):
        self.ducking = val


class TestAudioPlaybackEngine(unittest.TestCase):
    def setUp(self):
        self.player = AudioPlaybackEngine()
        self.streamer = MockStreamer()

    def test_echo_prevention_muting_link(self):
        """Verify playback engine triggers streamer muting to prevent feedback loops."""
        link_playback_to_streamer(self.player, self.streamer, duck_only=False)

        # Before playback
        self.assertFalse(self.player.is_playing)
        self.assertFalse(self.streamer.muted)

        # Simulate playback start
        self.player._notify_playback_start()
        self.assertTrue(self.player.is_playing)
        self.assertTrue(self.streamer.muted)

        # Simulate playback end
        self.player._notify_playback_end()
        self.assertFalse(self.player.is_playing)
        self.assertFalse(self.streamer.muted)

    def test_echo_prevention_ducking_link(self):
        """Verify ducking mode reduces streamer volume during playback."""
        link_playback_to_streamer(self.player, self.streamer, duck_only=True)

        self.player._notify_playback_start()
        self.assertAlmostEqual(self.streamer.ducking, 0.10)

        self.player._notify_playback_end()
        self.assertAlmostEqual(self.streamer.ducking, 1.0)


if __name__ == "__main__":
    unittest.main()
