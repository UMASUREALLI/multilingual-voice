# 📱 Flutter Developer Integration Guide: Real-Time Multilingual Voice Call Translator

This document provides everything the **Flutter mobile development team** needs to integrate the real-time call translation backend into Android & iOS apps.

---

## 🌐 1. Backend Server Endpoints

| Environment | Base URL |
| :--- | :--- |
| **Local Wi-Fi (Same Network)** | `http://<LAPTOP_LAN_IP>:8000` |
| **Android Emulator (Local)** | `http://10.0.2.2:8000` |
| **Live Remote Cloudflare Tunnel** | `https://neighbor-feb-june-lifestyle.trycloudflare.com` |

> [!NOTE]
> To test on physical Android/iOS devices without configuring LAN IP ports, use the **Live Cloudflare Tunnel URL** directly.

---

## 📡 2. Core API Specifications

### 2.1 Health Check
Verify the server is up and AI models are loaded.

- **Method / Endpoint:** `GET /health`
- **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "service": "Multilingual Voice AI",
    "whisper_model": "base",
    "active_sessions": 0
  }
  ```

---

### 2.2 Create Call Session
Initializes a new translated call session between two participants with their chosen languages.

- **Method / Endpoint:** `POST /api/call/create`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "caller_a_lang": "te",
    "caller_b_lang": "en",
    "caller_a_name": "Ravi (Telugu)",
    "caller_b_name": "Sarah (English)",
    "caller_a_gender": "male",
    "caller_b_gender": "female"
  }
  ```
  *(Supported language codes: `te` [Telugu], `hi` [Hindi], `en` [English], `ta` [Tamil], `kn` [Kannada], `ml` [Malayalam], `mr` [Marathi], `bn` [Bengali], `gu` [Gujarati], `auto` [Auto-Detect])*

- **Response `200 OK`:**
  ```json
  {
    "session_id": "c8a14f2e",
    "created_at": 1732341200.5,
    "status": "active",
    "participants": {
      "caller_a": {
        "id": "caller_a",
        "name": "Ravi (Telugu)",
        "language": "te",
        "gender": "male"
      },
      "caller_b": {
        "id": "caller_b",
        "name": "Sarah (English)",
        "language": "en",
        "gender": "female"
      }
    }
  }
  ```

---

### 2.3 Send Audio Turn (Speak & Translate)
The core real-time endpoint: the mobile app records audio when the user speaks, encodes it as **Base64 WAV**, and sends it to the server. The server transcribes, translates with colloquial precision, and returns both text subtitles and the translated speech audio.

- **Method / Endpoint:** `POST /api/call/{session_id}/audio_turn`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "speaker_id": "caller_a",
    "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "client_duration_sec": 2.45
  }
  ```

- **Response `200 OK` (Speech Detected & Translated):**
  ```json
  {
    "status": "success",
    "subtitle": {
      "speaker_id": "caller_a",
      "speaker_name": "Ravi (Telugu)",
      "original_language": "te",
      "original_text": "ఏం చేయట్లేదు బ్రో",
      "target_language": "en",
      "translated_text": "Not doing anything, bro",
      "total_latency_ms": 844.0
    },
    "audio_base64": "UklGRtb+AABXQVZFZm10IBAAAAABAAEA..."
  }
  ```

- **Response `200 OK` (Empty / Silence):**
  ```json
  {
    "status": "empty",
    "message": "Microphone audio was silent or too quiet."
  }
  ```

---

### 2.4 End / Terminate Call
Cleans up the session and wipes temporary memory buffers.

- **Method / Endpoint:** `POST /api/call/{session_id}/terminate`
- **Response `200 OK`:**
  ```json
  {
    "status": "success",
    "message": "Session c8a14f2e terminated."
  }
  ```

---

## 🛠️ 3. Complete Dart / Flutter Implementation

Recommended Flutter packages for `pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
  record: ^5.1.2         # Audio recording from microphone
  audioplayers: ^6.0.0   # Audio playback of translated voice
  path_provider: ^2.1.2
```

### 3.1 Call Translation Service (`lib/services/call_translation_service.dart`)

```dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class CallTranslationService {
  final String baseUrl;
  String? currentSessionId;

  CallTranslationService({required this.baseUrl});

  /// 1. Initialize a new translated call session
  Future<String> createCallSession({
    required String myLanguage,      // e.g. 'te'
    required String peerLanguage,    // e.g. 'en'
    String myName = "User A",
    String peerName = "User B",
  }) async {
    final url = Uri.parse('$baseUrl/api/call/create');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'caller_a_lang': myLanguage,
        'caller_b_lang': peerLanguage,
        'caller_a_name': myName,
        'caller_b_name': peerName,
        'caller_a_gender': 'female',
        'caller_b_gender': 'male',
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      currentSessionId = data['session_id'];
      return currentSessionId!;
    } else {
      throw Exception('Failed to create call session: ${response.body}');
    }
  }

  /// 2. Send recorded speech WAV audio file for translation & TTS
  Future<Map<String, dynamic>?> sendAudioTurn({
    required File audioFile,
    required double durationSeconds,
    String speakerId = 'caller_a',
  }) async {
    if (currentSessionId == null) {
      throw Exception('No active call session. Call createCallSession() first.');
    }

    // Read recorded audio bytes and encode to Base64
    final bytes = await audioFile.readAsBytes();
    final base64Audio = base64Encode(bytes);

    final url = Uri.parse('$baseUrl/api/call/$currentSessionId/audio_turn');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'speaker_id': speakerId,
        'audio_base64': base64Audio,
        'client_duration_sec': durationSeconds,
      }),
    );

    if (response.statusCode == 200) {
      final result = jsonDecode(response.body);
      if (result['status'] == 'success') {
        return {
          'original_text': result['subtitle']['original_text'],
          'translated_text': result['subtitle']['translated_text'],
          'latency_ms': result['subtitle']['total_latency_ms'],
          'audio_bytes': base64Decode(result['audio_base64']),
        };
      } else {
        // Status is 'empty' (silence/no speech)
        return null;
      }
    } else {
      throw Exception('Turn processing failed: ${response.body}');
    }
  }

  /// 3. Helper to write received audio bytes to temporary file for playback
  Future<File> saveAudioBytesToTempFile(Uint8List audioBytes) async {
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/translated_${DateTime.now().millisecondsSinceEpoch}.wav');
    await file.writeAsBytes(audioBytes);
    return file;
  }

  /// 4. Terminate call session
  Future<void> endCall() async {
    if (currentSessionId == null) return;
    final url = Uri.parse('$baseUrl/api/call/$currentSessionId/terminate');
    await http.post(url);
    currentSessionId = null;
  }
}
```

---

### 3.2 Flutter Call Screen Audio Workflow

```dart
// Example: In your Call Screen State
final translator = CallTranslationService(
  baseUrl: 'https://neighbor-feb-june-lifestyle.trycloudflare.com',
);
final AudioRecorder _recorder = AudioRecorder();
final AudioPlayer _player = AudioPlayer();

// When user starts speaking (or button pressed):
void startRecordingTurn() async {
  if (await _recorder.hasPermission()) {
    final tempDir = await getTemporaryDirectory();
    final path = '${tempDir.path}/turn_audio.wav';
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000),
      path: path,
    );
  }
}

// When user finishes speaking (or silence detected):
void stopAndTranslateTurn(double durationSec) async {
  final path = await _recorder.stop();
  if (path == null) return;

  final audioFile = File(path);
  final result = await translator.sendAudioTurn(
    audioFile: audioFile,
    durationSeconds: durationSec,
  );

  if (result != null) {
    // 1. Update UI with subtitles
    print("User said: ${result['original_text']}");
    print("Translated: ${result['translated_text']}");
    print("Latency: ${result['latency_ms']} ms");

    // 2. Play translated audio aloud to caller
    final playFile = await translator.saveAudioBytesToTempFile(result['audio_bytes']);
    await _player.play(DeviceFileSource(playFile.path));
  }
}
```

---

## 🧪 4. Testing & Verification Checklist for Flutter Team

1. **Verify Backend Health:**
   ```bash
   curl https://neighbor-feb-june-lifestyle.trycloudflare.com/health
   ```
   *Expected: `{"status":"healthy","service":"Multilingual Voice AI",...}`*

2. **Verify Call Creation:**
   ```bash
   curl -X POST https://neighbor-feb-june-lifestyle.trycloudflare.com/api/call/create \
     -H "Content-Type: application/json" \
     -d '{"caller_a_lang":"te","caller_b_lang":"en"}'
   ```
   *Expected: Returns a JSON object with `"session_id"` and `"status":"active"`.*

3. **Verify Audio Turn:**
   Record any 2-second WAV file on the phone, convert to Base64, and POST to `/api/call/{session_id}/audio_turn`.
   *Expected: Returns transcription, translation, and `audio_base64` synthesized speech within 1–1.5 seconds.*
