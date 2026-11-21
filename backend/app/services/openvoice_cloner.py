"""
OpenVoice v2 Zero-Shot Neural Voice Cloning Service.
Performs in-memory speaker tone-color extraction and neural voice conversion.
Transforms base TTS speech to match the caller's acoustic vocal identity and timbre.
"""

import os
import sys
import tempfile
import threading
import numpy as np
import torch
import soundfile as sf

# Add OpenVoice repository to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
OPENVOICE_DIR = os.path.join(BACKEND_DIR, "OpenVoice")
CHECKPOINTS_DIR = os.path.join(BACKEND_DIR, "checkpoints_v2")

if OPENVOICE_DIR not in sys.path:
    sys.path.insert(0, OPENVOICE_DIR)

_INSTANCE = None
_INIT_LOCK = threading.Lock()


class OpenVoiceCloner:
    """Zero-shot neural voice cloner powered by OpenVoice v2 ToneColorConverter."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.converter = None
        self.default_src_se = None
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        try:
            from openvoice.api import ToneColorConverter

            config_path = os.path.join(CHECKPOINTS_DIR, "converter", "config.json")
            ckpt_path = os.path.join(CHECKPOINTS_DIR, "converter", "checkpoint.pth")
            src_se_path = os.path.join(CHECKPOINTS_DIR, "base_speakers", "ses", "en-india.pth")

            if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
                print(f"[OpenVoice] Checkpoints not found in {CHECKPOINTS_DIR}. Neural cloning disabled.")
                return

            # Set PyTorch to utilize multi-core CPU efficiently
            cpu_threads = max(2, min(os.cpu_count() or 4, 8))
            torch.set_num_threads(cpu_threads)

            self.converter = ToneColorConverter(config_path, device=self.device, enable_watermark=False)
            self.converter.load_ckpt(ckpt_path)

            if os.path.exists(src_se_path):
                self.default_src_se = torch.load(src_se_path, map_location=self.device)
            else:
                self.default_src_se = None

            self.is_ready = True
            print(f"[OpenVoice] Neural Voice Cloner initialized successfully on {self.device} (threads={cpu_threads}).")
        except Exception as e:
            print(f"[OpenVoice] Failed to initialize OpenVoice: {e}")
            self.converter = None
            self.is_ready = False

    def extract_speaker_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract 256-dimensional tone color embedding vector from caller's speech.
        Operates entirely in RAM with a fast temporary buffer.
        """
        if not self.is_ready or self.converter is None:
            return None

        if len(audio) < sample_rate * 0.8:
            return None  # Sample too short for reliable tone extraction

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio.astype(np.float32), sample_rate)
            with torch.inference_mode():
                target_se = self.converter.extract_se(tmp_path)
            return target_se
        except Exception as e:
            print(f"[OpenVoice] Error extracting speaker embedding: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def convert_voice(
        self,
        base_audio: np.ndarray,
        src_sample_rate: int = 24000,
        target_se: torch.Tensor = None,
        tau: float = 0.3
    ) -> tuple[np.ndarray, int]:
        """
        Morph base TTS audio into the target speaker's vocal timbre and formants.
        Returns:
            (converted_audio_float32, sample_rate=22050)
            or original audio if conversion fails/disabled.
        """
        if not self.is_ready or target_se is None or self.converter is None:
            return base_audio, src_sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            in_path = tmp_in.name

        try:
            # Write base audio to temp file
            sf.write(in_path, base_audio.astype(np.float32), src_sample_rate)

            # Use base speaker embedding
            src_se = self.default_src_se
            if src_se is None:
                src_se = self.converter.extract_se(in_path)

            with torch.inference_mode():
                converted_np = self.converter.convert(
                    in_path,
                    src_se=src_se,
                    tgt_se=target_se,
                    tau=tau
                )

            # OpenVoice v2 output sample rate is 22050 Hz
            target_sr = self.converter.hps.data.sampling_rate
            return converted_np.astype(np.float32), target_sr

        except Exception as e:
            print(f"[OpenVoice] Voice conversion warning: {e}. Falling back to base TTS.")
            return base_audio, src_sample_rate
        finally:
            if os.path.exists(in_path):
                try:
                    os.remove(in_path)
                except OSError:
                    pass


def get_openvoice_cloner() -> OpenVoiceCloner:
    """Get or create singleton instance of OpenVoiceCloner."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INIT_LOCK:
            if _INSTANCE is None:
                _INSTANCE = OpenVoiceCloner(device="cpu")
    return _INSTANCE
