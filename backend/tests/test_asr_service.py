"""
Automated unit tests for Streaming Multilingual Speech Recognition (ASR) Service.
"""

import sys
import os
import unittest
import numpy as np
import soundfile as sf

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.transcription import TranscriptionResult, WordTiming
from app.services.asr_service import StreamingAsrService, ConversationalContextManager, transcribe_audio


class TestConversationalContext(unittest.TestCase):

    def test_context_buffering(self):
        ctx = ConversationalContextManager(max_history_turns=3)
        self.assertIsNone(ctx.get_context_prompt())

        ctx.add_turn("Hello, how are you?")
        self.assertIn("Hello, how are you?", ctx.get_context_prompt())

        ctx.add_turn("I am doing great.")
        ctx.add_turn("Let us test speech recognition.")
        ctx.add_turn("This is the fourth sentence.")

        # Should keep the last 3
        prompt = ctx.get_context_prompt()
        self.assertNotIn("Hello, how are you?", prompt)
        self.assertIn("fourth sentence", prompt)

        ctx.clear()
        self.assertIsNone(ctx.get_context_prompt())


class TestStreamingAsrService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = StreamingAsrService.get_instance(model_size="base")
        cls.sample_path = r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\transcripts\demo_spoken_sample.wav"

    def test_legacy_wrapper_backward_compatibility(self):
        if not os.path.exists(self.sample_path):
            self.skipTest("Sample wav not found")

        result = transcribe_audio(self.sample_path)
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("language", result)
        self.assertIn("language_probability", result)
        self.assertEqual(result["language"], "en")
        self.assertIn("multilingual", result["text"].lower())

    def test_in_memory_numpy_transcription(self):
        if not os.path.exists(self.sample_path):
            self.skipTest("Sample wav not found")

        audio, sr = sf.read(self.sample_path)
        res = self.service.transcribe(audio, language="en", word_timestamps=True)

        self.assertIsInstance(res, TranscriptionResult)
        self.assertEqual(res.language, "en")
        self.assertEqual(res.language_name, "English")
        self.assertGreater(res.duration_sec, 4.0)
        self.assertGreater(res.latency_ms, 0)
        self.assertGreater(res.rtf, 0)
        self.assertIn("multilingual", res.text.lower())

        # Test word timings
        self.assertTrue(len(res.words) > 5)
        first_word = res.words[0]
        self.assertIsInstance(first_word, WordTiming)
        self.assertGreaterEqual(first_word.end, first_word.start)
        self.assertTrue(len(first_word.word) > 0)

    def test_empty_audio_handling(self):
        res = self.service.transcribe(np.zeros(0, dtype=np.float32), language="en")
        self.assertEqual(res.text, "")
        self.assertEqual(res.duration_sec, 0.0)
