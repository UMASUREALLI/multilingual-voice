# Real-Time Bidirectional Multilingual Voice Call Translator

> **An ultra-low-latency, full-duplex AI voice translator with voice cloning, acoustic echo loop suppression, and zero disk writes.**

---

## 🌟 Key Architecture & Capabilities

1. **Zero Disk Writes**: All audio buffers and intermediate states remain strictly in RAM as 16 kHz float32 NumPy arrays and PyTorch tensors.
2. **Unified Acoustic Language Identification (107+ Languages)**: SpeechBrain VoxLingua107 ECAPA-TDNN identifies spoken languages acoustics-first, fortified with multi-turn hysteresis and script verification.
3. **Native-Script Neural Speech-to-Text (ASR)**: Native script decoding across Indic languages (Telugu, Tamil, Hindi, Kannada, Malayalam, Bengali, Marathi, etc.) and global languages.
4. **Contextual Translation with Named Entity Preservation**: Protects phone numbers, emails, addresses, currencies, and names from translation distortion.
5. **Natural Neural Text-to-Speech (TTS)**: Ultra-low latency streaming synthesis with regional voice personas (e.g., `te-IN-ShrutiNeural`, `ta-IN-PallaviNeural`, `hi-IN-SwaraNeural`, `en-IN-NeerjaNeural`).
6. **Acoustic Echo Feedback Loop Prevention**: Dynamically mutes the microphone input during speaker playback so synthesized speech never re-translates itself.
7. **Ethical Voice Cloning & Prosody Matching**: Pitch (F0) tracking and tempo adaptation, governed by strict biometric consent statements and ephemeral session cleanup.

---

## 🚀 Setup Guide (New Laptop Installation)

Follow these steps to set up and run the project on a new laptop.

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **Hardware**: Working microphone and speakers / headphones.
- **Git**: Installed on your system.

### 2. Clone the Repository
```powershell
git clone https://github.com/<YOUR_GITHUB_USERNAME>/multilingual-voice.git
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

---

## 🧪 Running Automated Tests

Verify that all 56 end-to-end tests pass on your machine:
```powershell
cd backend
python -m unittest discover -s tests -p "test_*.py"
```
*(All 56 tests should complete with `OK` in ~20-30 seconds).*

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

### 5. FastAPI Backend Server
Launch the REST & WebSocket call coordination service:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger API docs: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
multilingual-voice/
├── backend/
│   ├── app/
│   │   ├── audio/              # Live capture, VAD, playback & echo prevention
│   │   ├── models/             # Pydantic schemas (call, language, audio, tts)
│   │   ├── services/           # LID, ASR, translation, TTS & voice cloning
│   │   └── api/                # FastAPI routes & WebSocket coordinators
│   ├── tests/                  # 56 automated unit and integration tests
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
- **Biometric Voice Privacy**: Voice embeddings exist solely in ephemeral memory and are immediately wiped upon call termination.
- **Acoustic Feedback Safety**: Active muting prevents acoustic echo loops during loudspeaker playback.
