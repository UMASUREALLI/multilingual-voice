"""
FastAPI Server & WebSocket Gateway for Real-Time Multilingual Voice Call Translation.
Hosts the interactive web dashboard and full-duplex WebSocket call pipelines.
Zero disk writes: all audio capture, ASR, translation, and TTS flow in memory.
"""

import os
import io
import time
import base64
import binascii
import logging
from typing import Dict, Literal, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.models.call import CallSubtitle
from app.models.language import SUPPORTED_LANGUAGES, normalize_language_code
from app.services.call_coordinator import CallCoordinator
from app.services.tts_service import DEFAULT_VOICE_REGISTRY
from app.services.lid_service import trim_silence
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("call_translator.main")

app = FastAPI(
    title="Multilingual Voice AI Call Translator",
    description=(
        "### 🎙️ **[👉 CLICK HERE TO OPEN INTERACTIVE CALL DASHBOARD (/)](/)** 🎙️\n\n"
        "Real-Time Full-Duplex Bidirectional Voice Translation with Consent-Based Prosody Matching.\n\n"
        "*(Note: You are currently viewing the backend developer API documentation at `/docs`. "
        "To experience the interactive dashboard with live microphone speech, dual subtitles, and neural voice playback, open **[http://localhost:8000/](/)**)*"
    ),
    version="2.0.0"
)

# Mount static web assets
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

coordinator = CallCoordinator.get_instance()

# Active WebSocket connections: {session_id: {speaker_id: WebSocket}}
active_ws_connections: Dict[str, Dict[str, WebSocket]] = {}


# ----------------------------------------------------------------------
# Request/Response Schemas
# ----------------------------------------------------------------------

class CreateCallRequest(BaseModel):
    caller_a_lang: str = Field(default="te", min_length=2, max_length=32)
    caller_b_lang: str = Field(default="en", min_length=2, max_length=32)
    caller_a_name: str = Field(default="Caller A (Telugu)", min_length=1, max_length=80)
    caller_b_name: str = Field(default="Caller B (English)", min_length=1, max_length=80)
    caller_a_gender: Literal["female", "male"] = "female"
    caller_b_gender: Literal["female", "male"] = "male"


class ConsentRequest(BaseModel):
    speaker_id: Literal["caller_a", "caller_b"]
    consent_text: str = Field(min_length=1, max_length=1000)
    audio_base64: Optional[str] = Field(default=None, max_length=25_000_000)


class TurnRequest(BaseModel):
    speaker_id: Literal["caller_a", "caller_b"]
    text: str = Field(min_length=1, max_length=5000)
    language: Optional[str] = None


class AudioTurnRequest(BaseModel):
    speaker_id: Literal["caller_a", "caller_b"] = "caller_a"
    audio_base64: str = Field(min_length=1, max_length=25_000_000)
    client_duration_sec: Optional[float] = Field(default=None, gt=0.0, le=120.0)


# ----------------------------------------------------------------------
# Core REST Endpoints
# ----------------------------------------------------------------------

@app.get("/")
def home():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "Multilingual Voice AI Backend is Running!",
        "status": "success",
        "version": "2.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Multilingual Voice AI",
        "whisper_model": coordinator.model_size,
        "active_sessions": len(coordinator._sessions)
    }


@app.get("/api/voices")
def list_voices():
    """List available neural voices across supported languages."""
    return DEFAULT_VOICE_REGISTRY


@app.post("/api/call/create")
def create_call(req: CreateCallRequest):
    """Initialize a new bidirectional translated call session."""
    languages = []
    for raw_language in (req.caller_a_lang, req.caller_b_lang):
        normalized = "auto" if raw_language.strip().lower() == "auto" else normalize_language_code(raw_language)
        if normalized != "auto" and normalized not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=422, detail=f"Unsupported language: {raw_language}")
        languages.append(normalized)
    session = coordinator.create_session(
        caller_a_lang=languages[0],
        caller_b_lang=languages[1],
        caller_a_name=req.caller_a_name,
        caller_b_name=req.caller_b_name,
        caller_a_gender=req.caller_a_gender,
        caller_b_gender=req.caller_b_gender
    )
    return session.to_dict()


@app.get("/api/call/{session_id}")
def get_call(session_id: str):
    """Retrieve call session status and dual subtitle history."""
    session = coordinator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Call session not found")
    data = session.to_dict()
    data["subtitles"] = [s.to_dict() for s in session.subtitles]
    return data


@app.post("/api/call/{session_id}/consent")
def enroll_consent(session_id: str, req: ConsentRequest):
    """Enroll speaker's verbal authorization for ethical voice modeling."""
    try:
        audio_samples = None
        sample_rate = 16000
        if req.audio_base64:
            try:
                raw_audio = base64.b64decode(req.audio_base64, validate=True)
                audio_samples, sample_rate = sf.read(io.BytesIO(raw_audio), dtype="float32")
                if audio_samples.ndim > 1:
                    audio_samples = audio_samples.mean(axis=1)
                if sample_rate != 16000 and len(audio_samples) > 0:
                    import scipy.signal
                    target_length = int(len(audio_samples) * 16000 / sample_rate)
                    audio_samples = scipy.signal.resample(audio_samples, target_length).astype(np.float32)
                    sample_rate = 16000
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Consent audio is invalid") from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Consent audio is not a supported audio file") from exc

        result = coordinator.enroll_consent(
            session_id=session_id,
            speaker_id=req.speaker_id,
            spoken_consent_text=req.consent_text,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
        )
        return {"status": "success", **result}
    except PermissionError as pe:
        return JSONResponse(status_code=403, content={"status": "error", "message": str(pe)})
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))


@app.post("/api/call/{session_id}/turn")
def process_turn(session_id: str, req: TurnRequest):
    """
    Process a conversational turn through translation and neural TTS.
    Returns synchronized subtitle metadata and base64-encoded audio. Zero disk writes.
    """
    session = coordinator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Call session not found")

    speaker = session.get_participant(req.speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found in session")

    recipient = session.get_recipient(req.speaker_id)
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Turn text cannot be blank")
    src_lang = req.language or speaker.native_language
    if src_lang == "auto":
        script_decision = coordinator.lid.identify_script(req.text)
        src_lang = script_decision.language_code if script_decision else "en"
    else:
        src_lang = normalize_language_code(src_lang)
        if src_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=422, detail=f"Unsupported language: {req.language}")
    target_lang = recipient.native_language
    if target_lang == "auto":
        target_lang = "en" if src_lang != "en" else "te"

    t0 = time.perf_counter()
    trans_res = coordinator.translation.translate(
        text=req.text,
        source_lang=src_lang,
        target_lang=target_lang,
        speaker_name=speaker.name,
        receiver_name=recipient.name
    )
    trans_latency = (time.perf_counter() - t0) * 1000.0

    # TTS Synthesis
    t_tts_0 = time.perf_counter()
    pitch_adj = speaker.pitch_mod if speaker.voice_cloned else "+0Hz"
    rate_adj = speaker.rate_mod if speaker.voice_cloned else "+0%"

    tts_res = coordinator.tts.synthesize(
        text=trans_res.translated_text,
        language_code=target_lang,
        gender=speaker.gender,
        pitch=pitch_adj,
        rate=rate_adj
    )
    tts_latency = (time.perf_counter() - t_tts_0) * 1000.0

    # Encode audio array to base64 MP3/WAV in memory for browser playback
    bio = io.BytesIO()
    sf.write(bio, tts_res.audio, tts_res.sample_rate, format="WAV")
    audio_base64 = base64.b64encode(bio.getvalue()).decode("utf-8")

    total_latency = trans_latency + tts_latency

    subtitle = CallSubtitle(
        speaker_id=speaker.participant_id,
        speaker_name=speaker.name,
        original_language=src_lang,
        original_text=req.text,
        target_language=target_lang,
        translated_text=trans_res.translated_text,
        asr_latency_ms=0.0,
        translation_latency_ms=trans_latency,
        tts_latency_ms=tts_latency,
        total_latency_ms=total_latency,
        has_audio=True
    )
    session.subtitles.append(subtitle)

    return {
        "status": "success",
        "subtitle": subtitle.to_dict(),
        "audio_base64": audio_base64
    }


@app.post("/api/call/{session_id}/audio_turn")
def process_audio_turn(session_id: str, req: AudioTurnRequest):
    """
    Process raw microphone audio sent from the browser.
    Runs Acoustic LID -> Whisper ASR -> NMT Translation -> Neural TTS in cloned voice.
    Zero disk writes.
    """
    session = coordinator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Call session not found")
    if not session.get_participant(req.speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found in session")

    try:
        try:
            raw_bytes = base64.b64decode(req.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="audio_base64 is not valid base64") from exc
        bio = io.BytesIO(raw_bytes)
        try:
            audio_data, sr = sf.read(bio, dtype="float32")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Decoded audio is not a supported audio file") from exc
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sr != 16000 and len(audio_data) > 0:
            import scipy.signal
            num_target_samples = int(len(audio_data) * 16000 / sr)
            audio_data = scipy.signal.resample(audio_data, num_target_samples).astype(np.float32)
            sr = 16000

        # Physical Wall-Clock Timing Guard:
        # Detect browser hardware 48kHz/44.1kHz capture tagged as 16kHz
        if req.client_duration_sec and req.client_duration_sec > 0.2 and len(audio_data) > 0:
            wav_dur = len(audio_data) / sr
            ratio = wav_dur / req.client_duration_sec
            if ratio > 1.3:
                import scipy.signal
                target_samples = int(req.client_duration_sec * 16000)
                logger.info(
                    f"Hardware sample rate mismatch detected: WAV claims {wav_dur:.2f}s, "
                    f"client recorded for {req.client_duration_sec:.2f}s (ratio {ratio:.2f}). "
                    f"Auto-resampling from {len(audio_data)} to {target_samples} samples."
                )
                audio_data = scipy.signal.resample(audio_data, target_samples).astype(np.float32)
                sr = 16000

        peak = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
        dur_sec = len(audio_data) / sr if sr > 0 else 0.0
        logger.info(f"Incoming audio turn: {len(audio_data)} samples @ {sr}Hz (Duration: {dur_sec:.2f}s, Peak: {peak:.4f})")

        if peak < 0.012 or dur_sec < 0.25:
            return {"status": "empty", "message": "Microphone audio was silent or too quiet. Please speak clearly into your mic."}

        # Software AGC (Peak Normalization): Scale quiet microphone audio to -1.0 dBFS (0.90)
        # This prevents Whisper from hallucinating faint foreign phonemes (like Urdu/Russian) on quiet inputs.
        if peak > 0.003:
            audio_data = (audio_data / peak * 0.90).astype(np.float32)

        # Pre-trim silence to ensure fast Whisper response
        trimmed, _, _ = trim_silence(audio_data, sample_rate=sr)
        if len(trimmed) >= int(sr * 0.25):
            audio_data = trimmed

        subtitle, tts_result = coordinator.process_audio_turn(
            session_id=session_id,
            speaker_id=req.speaker_id,
            audio_samples=audio_data,
            sample_rate=sr,
            synthesize_audio=True
        )

        if not subtitle:
            return {"status": "empty", "message": "No speech detected in audio"}

        # Encode synthetic audio to base64
        audio_b64 = ""
        if tts_result is not None:
            out_bio = io.BytesIO()
            sf.write(out_bio, tts_result.audio, tts_result.sample_rate, format="WAV")
            audio_b64 = base64.b64encode(out_bio.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "subtitle": subtitle.to_dict(),
            "audio_base64": audio_b64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio turn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/call/{session_id}/terminate")
async def end_call(session_id: str):
    """End call and wipe ephemeral voice profiles from memory."""
    if coordinator.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Call session not found")
    connections = active_ws_connections.pop(session_id, {})
    for websocket in connections.values():
        try:
            await websocket.send_json({"type": "call_ended", "session_id": session_id})
            await websocket.close()
        except Exception:
            pass
    coordinator.end_session(session_id)
    return {"status": "success", "message": f"Session {session_id} terminated."}


# ----------------------------------------------------------------------
# Real-Time WebSocket Endpoint
# ----------------------------------------------------------------------

@app.websocket("/ws/call/{session_id}/{speaker_id}")
async def call_websocket_endpoint(websocket: WebSocket, session_id: str, speaker_id: str):
    await websocket.accept()

    session = coordinator.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Invalid session_id"})
        await websocket.close()
        return
    if session.get_participant(speaker_id) is None:
        await websocket.send_json({"type": "error", "message": "Invalid speaker_id"})
        await websocket.close()
        return

    if session_id not in active_ws_connections:
        active_ws_connections[session_id] = {}
    active_ws_connections[session_id][speaker_id] = websocket

    logger.info(f"WebSocket connected for session '{session_id}', speaker '{speaker_id}'")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "time": time.time()})

            elif msg_type == "barge_in":
                coordinator.handle_barge_in()
                # Broadcast barge in event
                for spk, ws in active_ws_connections.get(session_id, {}).items():
                    await ws.send_json({"type": "barge_in_event", "speaker_id": speaker_id})

            elif msg_type == "text_turn":
                text = str(data.get("text", "")).strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Text turn cannot be blank"})
                    continue
                src_lang = data.get("language")
                speaker = session.get_participant(speaker_id)
                recipient = session.get_recipient(speaker_id)
                resolved_src_lang = src_lang or speaker.native_language
                if resolved_src_lang == "auto":
                    script_decision = coordinator.lid.identify_script(text)
                    resolved_src_lang = script_decision.language_code if script_decision else "en"
                else:
                    resolved_src_lang = normalize_language_code(resolved_src_lang)
                if resolved_src_lang not in SUPPORTED_LANGUAGES:
                    await websocket.send_json({"type": "error", "message": "Unsupported source language"})
                    continue
                resolved_target_lang = recipient.native_language
                if resolved_target_lang == "auto":
                    resolved_target_lang = "en" if resolved_src_lang != "en" else "te"

                trans_res = coordinator.translation.translate(
                    text=text,
                    source_lang=resolved_src_lang,
                    target_lang=resolved_target_lang
                )

                tts_res = coordinator.tts.synthesize(
                    text=trans_res.translated_text,
                    language_code=resolved_target_lang,
                    gender=speaker.gender,
                    pitch=speaker.pitch_mod if speaker.voice_cloned else "+0Hz",
                    rate=speaker.rate_mod if speaker.voice_cloned else "+0%"
                )

                bio = io.BytesIO()
                sf.write(bio, tts_res.audio, tts_res.sample_rate, format="WAV")
                audio_b64 = base64.b64encode(bio.getvalue()).decode("utf-8")

                subtitle = CallSubtitle(
                    speaker_id=speaker.participant_id,
                    speaker_name=speaker.name,
                    original_language=resolved_src_lang,
                    original_text=text,
                    target_language=resolved_target_lang,
                    translated_text=trans_res.translated_text,
                    translation_latency_ms=trans_res.latency_ms,
                    tts_latency_ms=tts_res.latency_ms,
                    total_latency_ms=trans_res.latency_ms + tts_res.latency_ms,
                    has_audio=True
                )
                session.subtitles.append(subtitle)

                # Broadcast to both parties
                payload = {
                    "type": "subtitle",
                    "data": subtitle.to_dict(),
                    "audio_base64": audio_b64
                }
                for spk, ws in active_ws_connections.get(session_id, {}).items():
                    await ws.send_json(payload)

            elif msg_type == "audio_turn":
                raw_bytes = base64.b64decode(data.get("audio_base64", ""))
                bio = io.BytesIO(raw_bytes)
                audio_data, sr = sf.read(bio, dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                if sr != 16000 and len(audio_data) > 0:
                    import scipy.signal
                    num_target_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_target_samples).astype(np.float32)
                    sr = 16000

                subtitle, tts_result = coordinator.process_audio_turn(
                    session_id=session_id,
                    speaker_id=speaker_id,
                    audio_samples=audio_data,
                    sample_rate=sr,
                    synthesize_audio=True
                )

                if subtitle:
                    audio_b64 = ""
                    if tts_result is not None:
                        out_bio = io.BytesIO()
                        sf.write(out_bio, tts_result.audio, tts_result.sample_rate, format="WAV")
                        audio_b64 = base64.b64encode(out_bio.getvalue()).decode("utf-8")

                    payload = {
                        "type": "subtitle",
                        "data": subtitle.to_dict(),
                        "audio_base64": audio_b64
                    }
                    for spk, ws in active_ws_connections.get(session_id, {}).items():
                        await ws.send_json(payload)

            else:
                await websocket.send_json({"type": "error", "message": f"Unsupported message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for '{session_id}', '{speaker_id}'")
    except Exception as exc:
        logger.exception("WebSocket error for '%s', '%s': %s", session_id, speaker_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": "Unable to process message"})
        except Exception:
            pass
    finally:
        if session_id in active_ws_connections and speaker_id in active_ws_connections[session_id]:
            del active_ws_connections[session_id][speaker_id]
            if not active_ws_connections[session_id]:
                del active_ws_connections[session_id]
