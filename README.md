# Real-Time Bidirectional Multilingual Voice Call Translator

> **A full-duplex AI voice translator with consent-based prosody matching, acoustic echo suppression, and zero audio-file persistence.**

---

## 🌟 Key Architecture & Capabilities

1. **Zero Disk Writes**: All audio buffers and intermediate states remain strictly in RAM as 16 kHz float32 NumPy arrays and PyTorch tensors.
2. **Unified Acoustic Language Identification (107+ Languages)**: SpeechBrain VoxLingua107 ECAPA-TDNN identifies spoken languages acoustics-first, fortified with multi-turn hysteresis and script verification.
3. **Native-Script Neural Speech-to-Text (ASR)**: Native script decoding across Indic languages (Telugu, Tamil, Hindi, Kannada, Malayalam, Bengali, Marathi, etc.) and global languages.
4. **Contextual Translation with Named Entity Preservation**: Protects phone numbers, emails, addresses, currencies, and names from translation distortion.
5. **Natural Neural Text-to-Speech (TTS)**: Ultra-low latency streaming synthesis with regional voice personas (e.g., `te-IN-ShrutiNeural`, `ta-IN-PallaviNeural`, `hi-IN-SwaraNeural`, `en-IN-NeerjaNeural`).
6. **Acoustic Echo Feedback Loop Prevention**: Dynamically mutes the microphone input during speaker playback so synthesized speech never re-translates itself.
7. **Consent-Based Voice Profiles & Prosody Matching**: Pitch (F0) and tempo adaptation are held only for the active call and wiped at termination. This adapts a neural voice's prosody; it does not reproduce a speaker's biometric timbre.

---

## 🚀 Setup Guide (New Laptop Installation)

Follow these steps to set up and run the project on a new laptop.

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **Hardware**: Working microphone and speakers / headphones.
- **Git**: Installed on your system.

### 2. Clone the Repository
```powershell
git clone https://github.com/UMASUREALLI/multilingual-voice.git
cd multilingual-voice
```

### 3. Create & Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch (CPU-Optimized)
Install the official CPU build of PyTorch:
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

For API/UI development and automated tests without downloading optional speech
models, install the lightweight dependency set instead:

```powershell
pip install -r req.txt
```

The server loads local Whisper/Silero models lazily on the first audio request.
When optional neural dependencies or cloud TTS are unavailable, the API remains
usable and returns an offline audio cue rather than failing at startup.

---

## 🧪 Running Automated Tests

Run the complete automated suite from the repository root:
```powershell
python -m pytest
```

Four real-audio checks are skipped when their optional WAV fixtures are not present.

---

## 🎙️ Live Interactive Tools

Navigate to the `backend/` directory:
```powershell
cd backend
```

### 1. Live Voice Call Translation (Speak ➔ Speaks Back Translation)
Speak into your microphone in one language, and the system speaks the live translation out loud through your speakers:

- **Tamil ➔ Telugu**:
  ```powershell
  python test_live_call.py --source ta --target tel
  ```
- **English ➔ Telugu**:
  ```powershell
  python test_live_call.py --source en --target tel
  ```
- **Telugu ➔ English**:
  ```powershell
  python test_live_call.py --source tel --target en
  ```

### 2. Voice Echo / Mirror Mode (Speak ➔ Speaks Back Exact Words)
Test pure Text-to-Speech (TTS) voice fidelity and clarity without translation:
```powershell
python test_live_call.py --echo
```

### 3. Interactive Keyboard Translation & Audio Playback
Type any sentence in your terminal and hear the translation spoken aloud:
```powershell
python test_realtime_translation.py --text
```

### 4. Interactive Neural TTS Tester
Test any individual language and gender voice:
```powershell
python test_realtime_tts.py --interactive
```

### 5. Bidirectional AI Call Partner (Live Conversational Voice Call)
Engage in a live, full-duplex conversational voice call with an AI partner powered by a local neural Small Language Model (`Qwen2.5-0.5B-Instruct` on CPU) or Google Gemini:
```powershell
# English <-> Telugu full-duplex call with speaker output and keyboard input
python test_bidirectional_call.py --party-a en --party-b te --speaker --text

# Full live microphone mode (speak aloud into your mic)
python test_bidirectional_call.py --party-a en --party-b te --speaker

# 6-Language Neural Voice Showcase Demo (plays spoken greetings across EN, TE, HI, TA, KN, ES)
python test_bidirectional_call.py --demo
```

### 6. Interactive Visual Web Call Dashboard (FastAPI Server)
Launch the visual real-time call translator dashboard with live microphone capture, dual synchronized subtitles, and neural voice synthesis:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- **🌐 Visual Web Dashboard**: Open [http://localhost:8000/](http://localhost:8000/) in your browser
  - 🎯 **Auto-Detect Language (LID)**: Speak in any tongue—the model auto-identifies it.
  - ⚡ **1-Click Simulation**: Click *"Simulate Bidirectional Turn"* to test without a microphone.
  - 🎙️ **Live Mic Translation**: Click *"Speak as Speaker A"* or *"Speaker B"*, speak naturally, and hear the translation with the selected neural voice profile.
- **📖 Developer OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Structure

```
multilingual-voice/
├── backend/
│   ├── app/
│   │   ├── audio/              # Live capture, VAD, playback & echo suppression
│   │   ├── models/             # Pydantic schemas (call, language, audio, tts)
│   │   ├── services/           # LID, ASR, NMT, TTS, voice profiles, LLM dialogue
│   │   │   └── llm_dialogue.py # Dynamic neural SLM (Qwen2.5) + Gemini fallback
│   │   ├── static/             # Interactive Call Dashboard (HTML5, WebAudio, CSS3)
│   ├── tests/                  # 56 automated unit and integration tests
│   ├── test_bidirectional_call.py # Bidirectional full-duplex AI voice call CLI
│   ├── test_live_call.py       # Full-duplex live call translation CLI
│   ├── test_realtime_translation.py # Live mic translation testing
│   ├── test_realtime_tts.py    # Neural TTS benchmark & interactive CLI
│   └── main.py                 # FastAPI application entrypoint
├── datasets/                   # Audio & transcript test sets
├── requirements.txt            # Dependency manifest
├── .gitignore                  # Git exclusion rules
└── README.md                   # Project documentation
```

---

## 🔒 Security & Privacy
- **Zero Disk Persistence**: Raw audio streams are never saved to disk.
- **Voice Profile Privacy**: Acoustic pitch/rate features exist only in memory and are removed when their call terminates.
- **Acoustic Feedback Safety**: Active muting prevents acoustic echo loops during loudspeaker playback.
