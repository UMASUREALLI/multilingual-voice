
import sounddevice as sd
import soundfile as sf
import numpy as np

from app.services.asr_service import transcribe_audio

DURATION = 15
SAMPLE_RATE = 16000
AUDIO_FILE = "microphone_test.wav"
MICROPHONE_DEVICE = 1

LANGUAGES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "en": "English",
    "th": "Thai"
}

print("\n========== MULTILINGUAL VOICE AI ==========")

input("Press ENTER and start speaking...")

print("\nListening for 15 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_DEVICE
)

sd.wait()

rms = float(np.sqrt(np.mean(audio ** 2)))

print(f"\nMicrophone audio level: {rms:.5f}")

if rms < 0.001:
    print("WARNING: Microphone volume is very low.")

sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

print("\nIdentifying language and transcribing speech...")

result = transcribe_audio(AUDIO_FILE)

language_code = result["language"]
language_name = LANGUAGES.get(
    language_code, "Other / Unsupported"
)

print("\n============== AI RESULT ==============")
print("Detected language:", language_name)
print("Language code:", language_code)
print("Confidence:", result["language_probability"])
print("Transcribed speech:", result["text"])
print("=======================================")
