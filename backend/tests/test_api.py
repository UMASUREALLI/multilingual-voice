"""
Integration Tests for FastAPI Endpoints and Full-Stack Call Gateway.
Verifies REST API, session management, verbal consent verification, and turn processing.
"""

import unittest
from fastapi.testclient import TestClient

from main import app


class TestFastApiGateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Verify /health returns 200 OK and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("whisper_model", data)

    def test_list_voices_endpoint(self):
        """Verify /api/voices returns registered neural voices."""
        response = self.client.get("/api/voices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("te", data)
        self.assertIn("en", data)
        self.assertIn("hi", data)

    def test_call_lifecycle_api(self):
        """Verify creating, inspecting, processing a turn, and terminating a call session via REST."""
        # 1. Create Call
        create_res = self.client.post("/api/call/create", json={
            "caller_a_lang": "te",
            "caller_b_lang": "en",
            "caller_a_name": "Ravi",
            "caller_b_name": "Sarah"
        })
        self.assertEqual(create_res.status_code, 200)
        session = create_res.json()
        sid = session["session_id"]
        self.assertEqual(session["status"], "active")

        # 2. Get Call Status
        get_res = self.client.get(f"/api/call/{sid}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["session_id"], sid)

        # 3. Grant Consent for Caller A
        consent_res = self.client.post(f"/api/call/{sid}/consent", json={
            "speaker_id": "caller_a",
            "consent_text": "I authorize this AI system to clone and use my voice for real-time translation."
        })
        self.assertEqual(consent_res.status_code, 200)
        self.assertEqual(consent_res.json()["status"], "success")

        # 4. Process Conversational Turn (Telugu -> English)
        turn_res = self.client.post(f"/api/call/{sid}/turn", json={
            "speaker_id": "caller_a",
            "text": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "language": "te"
        })
        self.assertEqual(turn_res.status_code, 200)
        turn_data = turn_res.json()
        self.assertEqual(turn_data["status"], "success")
        self.assertIn("subtitle", turn_data)
        self.assertIn("audio_base64", turn_data)
        self.assertGreater(len(turn_data["audio_base64"]), 100)

        # 5. Terminate Call
        term_res = self.client.post(f"/api/call/{sid}/terminate")
        self.assertEqual(term_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
