"""
Automated unit tests for Intelligent Translation Service and Entity Preservation.
"""

import sys
import os
import unittest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.translation import TranslationResult
from app.services.entity_preserver import EntityPreserver
from app.services.translation_service import (
    TranslationService,
    ClauseSegmentationBuffer,
    BidirectionalCallTranslator
)


class TestEntityPreserver(unittest.TestCase):

    def setUp(self):
        self.preserver = EntityPreserver(registered_names=["Uma Surepalli"])

    def test_mask_and_unmask_technical_terms_and_names(self):
        text = "Hello Uma Surepalli, our FastAPI server is running with Whisper AI."
        masked, placeholder_map = self.preserver.mask(text)

        self.assertNotIn("Uma Surepalli", masked)
        self.assertNotIn("FastAPI", masked)
        self.assertNotIn("Whisper", masked)
        self.assertNotIn("AI", masked)
        self.assertTrue(len(placeholder_map) >= 4)

        # Unmasking test
        unmasked = self.preserver.unmask(masked, placeholder_map)
        self.assertEqual(unmasked, text)

    def test_mask_numbers_and_emails(self):
        text = "Contact me at user@test.com or call 9848012345 regarding ₹5000 invoice."
        masked, placeholder_map = self.preserver.mask(text)

        self.assertNotIn("user@test.com", masked)
        self.assertNotIn("9848012345", masked)
        self.assertNotIn("₹5000", masked)

        unmasked = self.preserver.unmask(masked, placeholder_map)
        self.assertEqual(unmasked, text)


class TestClauseSegmentationBuffer(unittest.TestCase):

    def test_negation_guard_telugu(self):
        buf = ClauseSegmentationBuffer()

        # Incomplete sentence ending with negation particle 'కాదు'
        out = buf.push("నేను చెప్పేది అది కాదు", lang="te")
        self.assertEqual(out, [])  # Should hold and wait for clause completion!

        # Completes with punctuation
        out2 = buf.push(".", lang="te")
        self.assertEqual(len(out2), 1)
        self.assertIn("కాదు .", out2[0])


class TestTranslationService(unittest.TestCase):

    def setUp(self):
        self.service = TranslationService()

    def test_identity_passthrough(self):
        res = self.service.translate("Hello world", source_lang="en", target_lang="en")
        self.assertEqual(res.translated_text, "Hello world")
        self.assertEqual(res.backend, "identity_passthrough")

    def test_telugu_to_english_translation(self):
        res = self.service.translate("నమస్కారం, మీరు ఎలా ఉన్నారు?", source_lang="te", target_lang="en")
        self.assertIsInstance(res, TranslationResult)
        self.assertEqual(res.source_language, "te")
        self.assertEqual(res.target_language, "en")
        self.assertTrue(len(res.translated_text) > 0)
        self.assertTrue(any(w in res.translated_text.lower() for w in ["hello", "how", "are", "you"]))

    def test_english_to_telugu_translation(self):
        res = self.service.translate("Hello, how are you today?", source_lang="en", target_lang="te")
        self.assertEqual(res.source_language, "en")
        self.assertEqual(res.target_language, "te")
        self.assertTrue(len(res.translated_text) > 0)
        # Should contain Telugu script Unicode glyphs
        has_telugu = any(0x0C00 <= ord(ch) <= 0x0C7F for ch in res.translated_text)
        self.assertTrue(has_telugu)

    def test_entity_preservation_during_translation(self):
        text = "Please send 5000 dollars to user@gmail.com today."
        res = self.service.translate(text, source_lang="en", target_lang="te", preserve_entities=True)
        # Email address must NOT be translated into Telugu!
        self.assertIn("user@gmail.com", res.translated_text)


class TestBidirectionalCallTranslator(unittest.TestCase):

    def test_two_way_session(self):
        session = BidirectionalCallTranslator(
            party_a_lang="te",
            party_b_lang="en",
            party_a_name="Uma",
            party_b_name="John"
        )

        # Direction A -> B (Telugu to English)
        a_to_b = session.translate_a_to_b("నమస్కారం, మీరు ఎలా ఉన్నారు?")
        self.assertEqual(a_to_b.target_language, "en")
        self.assertTrue(len(a_to_b.translated_text) > 0)

        # Direction B -> A (English to Telugu)
        b_to_a = session.translate_b_to_a("I am doing well, thank you.")
        self.assertEqual(b_to_a.target_language, "te")
        self.assertTrue(len(b_to_a.translated_text) > 0)
