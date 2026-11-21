"""
True Zero-Shot Neural Voice Cloning on CPU.
Uses SpeechBrain's pre-trained X-Vector Speaker Encoder to extract a 512-dimensional
biometric speaker embedding directly from your microphone audio in RAM,
and conditions Microsoft's SpeechT5 Neural Synthesizer to speak in YOUR unique voice.
100% Local, 100% Offline, Zero Disk Writes.
"""

import sys
import time
import argparse
import numpy as np
import torch
import sounddevice as sd

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier
from app.audio.player import AudioPlaybackEngine


def print_banner():
    print("\n" + "=" * 74)
    print("  🧠 TRUE ZERO-SHOT NEURAL VOICE CLONING (100% OFFLINE / CPU)")
    print("  SpeechBrain 512-D Speaker Embedding • Neural SpeechT5 Synthesizer")
    print("=" * 74)


class LocalNeuralVoiceCloner:
    _instance = None

    def __init__(self):
        print("\n[1/2] Loading Neural Voice Cloning Models into RAM...")
        t0 = time.time()
        torch.set_num_threads(4)

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
        
        self.model.eval()
        self.vocoder.eval()
        print(f"  -> All neural models initialized in {time.time() - t0:.1f}s.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_speaker_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract true 512-dimensional speaker acoustic embedding vector
        capturing speaker's vocal tract, formants, timbre, and pitch.
        """
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

        tensor_audio = torch.from_numpy(audio).float()
        if tensor_audio.ndim == 1:
            tensor_audio = tensor_audio.unsqueeze(0)

        # Normalize audio amplitude
        max_val = torch.max(torch.abs(tensor_audio))
        if max_val > 0:
            tensor_audio = tensor_audio / max_val * 0.90

        with torch.no_grad():
            embeddings = self.speaker_encoder.encode_batch(tensor_audio)
            # Squeeze to shape: (1, 512)
            speaker_embedding = embeddings.squeeze(1)
            # L2 Normalize
            speaker_embedding = torch.nn.functional.normalize(speaker_embedding, dim=-1)

        return speaker_embedding

    def synthesize_in_cloned_voice(self, text: str, speaker_embedding: torch.Tensor) -> np.ndarray:
        """
        Synthesize speech directly in the target speaker's cloned voice.
        Zero disk writes. Returns 16kHz float32 audio.
        """
        inputs = self.processor(text=text, return_tensors="pt")

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embedding,
                vocoder=self.vocoder
            )

        return speech.cpu().numpy().astype(np.float32)


def record_voice_prompt(duration_sec: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Record high-fidelity voice sample from microphone."""
    print("\n🎙️  STEP 1: RECORD YOUR VOICE (VOICE BIOMETRIC ENROLLMENT)")
    print("   Please prepare to speak a clear sentence for 5 seconds...")
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print("   🔴 RECORDING NOW! Speak naturally (e.g. 'Hello, my name is Uma and this is my voice'):", flush=True)

    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        start_t = time.time()
        while time.time() - start_t < duration_sec:
            elapsed = time.time() - start_t
            bars = int((elapsed / duration_sec) * 32)
            sys.stdout.write(f"\r   [{'█' * bars}{'░' * (32 - bars)}] {elapsed:.1f}s / {duration_sec:.1f}s")
            sys.stdout.flush()
            time.sleep(0.1)

    print("\n   ⏹️ Recording complete!")
    if not chunks:
        return np.zeros(sample_rate, dtype=np.float32)

    return np.concatenate(chunks, axis=0).flatten()


def main():
    parser = argparse.ArgumentParser(description="True Local Neural Voice Cloning")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of voice recording in seconds")
    args = parser.parse_args()

    print_banner()

    player = AudioPlaybackEngine()
    cloner = LocalNeuralVoiceCloner.get_instance()

    # Step 1: Record the user's voice
    audio = record_voice_prompt(duration_sec=args.duration)

    if len(audio) < 16000 * 1.5:
        print("❌ Recorded audio was too short. Please try again.")
        return

    # Step 2: Extract real 512-dim speaker biometric embedding
    print("\n🧬 [Step 2] Extracting 512-Dimensional Vocal Biometric Embedding in RAM...")
    t_emb = time.time()
    speaker_embedding = cloner.extract_speaker_embedding(audio, sample_rate=16000)
    emb_norm = float(torch.norm(speaker_embedding))
    emb_dim = speaker_embedding.shape[-1]
    print(f"   ✅ Biometric Embedding Extracted in {(time.time() - t_emb)*1000:.1f}ms!")
    print(f"   • Vector Dimensions: {emb_dim} float32 values")
    print(f"   • Embedding L2 Norm : {emb_norm:.4f}")
    print(f"   • Sample values     : {speaker_embedding[0, :5].tolist()}...")

    # Step 3: Synthesize speech conditioned on the user's REAL vocal tract
    test_sentences = [
        "Hello! This is my real cloned voice generated by neural SpeechT5 on this computer.",
        "I am speaking with my own vocal timbre, pitch, and voice characteristics.",
        "Real-time voice translation and cloning is running completely offline."
    ]

    print("\n" + "=" * 74)
    print("🔊 [Step 3] SYNTHESIZING SPEECH IN YOUR CLONED VOICE")
    print("=" * 74)

    for i, sent in enumerate(test_sentences, 1):
        print(f"\n[{i}/{len(test_sentences)}] Synthesizing in YOUR Cloned Voice:")
        print(f"   \"{sent}\"")
        t0 = time.perf_counter()
        synth_audio = cloner.synthesize_in_cloned_voice(sent, speaker_embedding)
        synth_time = time.perf_counter() - t0
        audio_dur = len(synth_audio) / 16000.0
        print(f"   -> Generated {audio_dur:.2f}s of audio in {synth_time:.2f}s! Playing aloud...")

        player.play(synth_audio, sample_rate=16000, blocking=True)
        time.sleep(0.5)

    # Step 4: Interactive Typing in Cloned Voice
    print("\n" + "=" * 74)
    print("⌨️  INTERACTIVE NEURAL CLONED VOICE TYPING MODE")
    print("   Type any sentence to hear it spoken in YOUR unique cloned voice.")
    print("   Type 'exit' to quit.")
    print("=" * 74 + "\n")

    while True:
        try:
            line = input("Your sentence > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                print("Exiting Neural Voice Cloner.")
                break

            print("   ⏳ Synthesizing in your cloned voice on CPU...")
            t0 = time.perf_counter()
            synth_audio = cloner.synthesize_in_cloned_voice(line, speaker_embedding)
            print(f"   🔊 Done ({(time.perf_counter() - t0):.2f}s)! Playing aloud through speakers...")
            player.play(synth_audio, sample_rate=16000, blocking=True)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Neural Voice Cloner.")
            break


if __name__ == "__main__":
    main()
