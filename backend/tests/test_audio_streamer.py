"""
Automated unit tests for in-memory streaming audio capture, ring buffer, and VAD.
"""

import sys
import os
import unittest
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.audio.buffer import AudioRingBuffer, RollingPreRollBuffer
from app.audio.vad import StreamingSileroVAD, SpeechEventType, SpeechSegment, calculate_rms_dbfs
from app.audio.streamer import LiveAudioStreamer


class TestAudioRingBuffer(unittest.TestCase):

    def test_basic_write_and_read(self):
        buf = AudioRingBuffer(capacity_samples=1000)
        data = np.arange(100, dtype=np.float32)
        buf.write(data)

        assert buf.current_size == 100
        assert buf.total_written == 100

        recent = buf.read_recent(50)
        assert len(recent) == 50
        np.testing.assert_array_equal(recent, np.arange(50, 100, dtype=np.float32))

    def test_circular_overflow_wrap(self):
        capacity = 100
        buf = AudioRingBuffer(capacity_samples=capacity)

        # Write 150 samples (overflowing capacity by 50)
        data = np.arange(150, dtype=np.float32)
        buf.write(data)

        assert buf.current_size == capacity
        assert buf.total_written == 150

        # Should retrieve the last 100 samples (50 to 149)
        recent = buf.read_recent(100)
        np.testing.assert_array_equal(recent, np.arange(50, 150, dtype=np.float32))

    def test_clear_buffer(self):
        buf = AudioRingBuffer(capacity_samples=500)
        buf.write(np.ones(200, dtype=np.float32))
        assert buf.current_size == 200

        buf.clear()
        assert buf.current_size == 0
        assert buf.total_written == 0
        recent = buf.read_recent(100)
        assert len(recent) == 0


class TestRollingPreRollBuffer(unittest.TestCase):

    def test_preroll_retrieval(self):
        # 16kHz, 100ms = 1600 samples
        preroll = RollingPreRollBuffer(sample_rate=16000, duration_ms=100)
        assert preroll.capacity_samples == 1600

        frame = np.ones(512, dtype=np.float32) * 0.5
        preroll.push(frame)
        audio = preroll.get_preroll()
        assert len(audio) == 512
        np.testing.assert_array_almost_equal(audio, frame)


class TestVADAndRMS(unittest.TestCase):

    def test_calculate_rms_dbfs(self):
        silence = np.zeros(1000, dtype=np.float32)
        db_silence = calculate_rms_dbfs(silence)
        assert db_silence <= -90.0

        full_scale = np.ones(1000, dtype=np.float32)
        db_full = calculate_rms_dbfs(full_scale)
        assert -0.1 <= db_full <= 0.1

    def test_silero_vad_silence_detection(self):
        vad = StreamingSileroVAD(start_threshold=0.5, end_threshold=0.3)
        silence_frame = np.zeros(512, dtype=np.float32)

        # Feed 10 silent frames
        for _ in range(10):
            prob, event, seg = vad.process_frame(silence_frame)
            assert prob < 0.20
            assert event == SpeechEventType.SILENCE
            assert seg is None

    def test_vad_utterance_emission_with_audio(self):
        import soundfile as sf
        wav_path = r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\curated_clean\utt_hi_01.wav"
        if not os.path.exists(wav_path):
            self.skipTest("Sample wav not found")

        audio, sr = sf.read(wav_path)
        assert sr == 16000

        vad = StreamingSileroVAD(
            start_threshold=0.40,
            end_threshold=0.25,
            hangover_ms=300,
            min_speech_ms=200
        )

        segments = []
        # Stream 512-sample chunks through VAD
        for i in range(0, len(audio) - 512, 512):
            frame = audio[i:i + 512].astype(np.float32)
            prob, event, seg = vad.process_frame(frame)
            if seg is not None:
                segments.append(seg)

        # Feed trailing silence to trigger hangover completion
        silence_frame = np.zeros(512, dtype=np.float32)
        for _ in range(15):
            prob, event, seg = vad.process_frame(silence_frame)
            if seg is not None:
                segments.append(seg)

        # Should have captured at least 1 segmented speech utterance
        assert len(segments) >= 1
        seg = segments[0]
        assert seg.duration_sec > 0.25
        assert seg.num_samples > 4000
        assert seg.peak_amp > 0.05
        assert isinstance(seg.audio, np.ndarray)


class TestLiveAudioStreamerInterface(unittest.TestCase):

    def test_list_input_devices(self):
        devices = LiveAudioStreamer.list_input_devices()
        assert isinstance(devices, list)
        if len(devices) > 0:
            dev = devices[0]
            assert "index" in dev
            assert "name" in dev
            assert "channels" in dev

    def test_mute_and_ducking_controls(self):
        streamer = LiveAudioStreamer()
        assert streamer._ducking_factor == 1.0
        assert streamer._muted is False

        streamer.set_ducking(0.5)
        assert streamer._ducking_factor == 0.5

        streamer.set_ducking(1.5)  # Should clip to 1.0
        assert streamer._ducking_factor == 1.0

        streamer.set_muted(True)
        assert streamer._muted is True
