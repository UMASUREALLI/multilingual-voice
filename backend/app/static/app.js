// Multilingual Voice AI - Real-Time Bidirectional Call Translator Client

let sessionId = null;
let ws = null;
let audioContext = null;
let mediaStream = null;
let analyserNode = null;
let recorderNode = null;
let isCalling = false;
let isMuted = false;
let vuInterval = null;

let isRecording = false;
let activeRecordingSpeaker = null;
let recordedAudioChunks = [];
let recordingStartTime = 0;

// DOM Elements
const connStatusBadge = document.getElementById("connStatusBadge");
const connStatusText = document.getElementById("connStatusText");
const statusDot = document.getElementById("statusDot");

const selectLangA = document.getElementById("selectLangA");
const selectLangB = document.getElementById("selectLangB");

const hudVad = document.getElementById("hudVad");
const hudAsr = document.getElementById("hudAsr");
const hudNmt = document.getElementById("hudNmt");
const hudTts = document.getElementById("hudTts");
const hudTotal = document.getElementById("hudTotal");

const cardCallerA = document.getElementById("cardCallerA");
const cardCallerB = document.getElementById("cardCallerB");
const nameCallerA = document.getElementById("nameCallerA");
const nameCallerB = document.getElementById("nameCallerB");
const langTagA = document.getElementById("langTagA");
const langTagB = document.getElementById("langTagB");
const vuBarA = document.getElementById("vuBarA");
const vuBarB = document.getElementById("vuBarB");
const subsBoxA = document.getElementById("subsBoxA");
const subsBoxB = document.getElementById("subsBoxB");
const cloneBadgeA = document.getElementById("cloneBadgeA");
const cloneBadgeB = document.getElementById("cloneBadgeB");

const btnRecordSpeakerA = document.getElementById("btnRecordSpeakerA");
const btnRecordSpeakerB = document.getElementById("btnRecordSpeakerB");
const recordingStatus = document.getElementById("recordingStatus");

const btnStartCall = document.getElementById("btnStartCall");
const btnEndCall = document.getElementById("btnEndCall");
const btnMuteMic = document.getElementById("btnMuteMic");
const btnVoiceClone = document.getElementById("btnVoiceClone");
const btnSimulateTurn = document.getElementById("btnSimulateTurn");

const consentModal = document.getElementById("consentModal");
const btnCancelConsent = document.getElementById("btnCancelConsent");
const btnGrantConsent = document.getElementById("btnGrantConsent");


// ---------------------------------------------------------------------
// Initialize & Setup Event Handlers
// ---------------------------------------------------------------------

btnStartCall.addEventListener("click", startCall);
btnEndCall.addEventListener("click", endCall);
btnMuteMic.addEventListener("click", toggleMute);
btnVoiceClone.addEventListener("click", () => consentModal.classList.add("open"));
btnCancelConsent.addEventListener("click", () => consentModal.classList.remove("open"));
btnGrantConsent.addEventListener("click", grantConsent);
btnSimulateTurn.addEventListener("click", simulateTurn);

btnRecordSpeakerA.addEventListener("click", () => toggleRecording("caller_a"));
btnRecordSpeakerB.addEventListener("click", () => toggleRecording("caller_b"));

function syncLanguageTags() {
  if (selectLangA && langTagA) {
    langTagA.innerText = selectLangA.options[selectLangA.selectedIndex].text;
  }
  if (selectLangB && langTagB) {
    langTagB.innerText = selectLangB.options[selectLangB.selectedIndex].text;
  }
}
syncLanguageTags();
selectLangA.addEventListener("change", syncLanguageTags);
selectLangB.addEventListener("change", syncLanguageTags);


async function startCall() {
  try {
    syncLanguageTags();
    const langA = selectLangA.value;
    const langB = selectLangB.value;

    const callerAName = (langA === "auto") ? "Speaker A (You)" : `Speaker A (${selectLangA.options[selectLangA.selectedIndex].text.split(" ")[1] || langA})`;
    const callerBName = `Speaker B (${selectLangB.options[selectLangB.selectedIndex].text.split(" ")[1] || langB})`;

    connStatusText.innerText = "Initializing Bidirectional Session...";
    const res = await fetch("/api/call/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        caller_a_lang: langA,
        caller_b_lang: langB,
        caller_a_name: callerAName,
        caller_b_name: callerBName,
        caller_a_gender: "male",
        caller_b_gender: "female"
      })
    });

    const data = await res.json();
    sessionId = data.session_id;

    // Connect WebSocket
    connectWebSocket(sessionId);

    // Setup Local Microphone for Live Audio Recording
    await setupLocalMicrophone();

    isCalling = true;
    btnStartCall.disabled = true;
    btnEndCall.disabled = false;
    btnMuteMic.disabled = false;
    btnRecordSpeakerA.disabled = false;
    btnRecordSpeakerB.disabled = false;
    selectLangA.disabled = true;
    selectLangB.disabled = true;

    connStatusBadge.style.color = "#10b981";
    connStatusText.innerText = `Call Active (${sessionId})`;
    recordingStatus.innerText = "🟢 Call Live! Click 'Speak as Speaker A' or 'Speaker B' to speak into your microphone.";
  } catch (err) {
    console.error("Start call failed:", err);
    alert("Failed to start call session: " + err.message);
  }
}


function connectWebSocket(sid) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/call/${sid}/caller_a`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket connected to session:", sid);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {
      console.warn("Non-JSON WS message:", event.data);
    }
  };

  ws.onclose = () => {
    console.log("WebSocket closed");
  };
}


async function setupLocalMicrophone() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(mediaStream);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);

    // Recorder processor node to capture 16kHz float32 audio
    recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
    recorderNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      recordedAudioChunks.push(new Float32Array(channelData));
    };
    source.connect(recorderNode);
    recorderNode.connect(audioContext.destination);

    // Drive local VU Meter
    const buffer = new Uint8Array(analyserNode.frequencyBinCount);
    vuInterval = setInterval(() => {
      if (!isCalling || isMuted) {
        vuBarA.style.width = "0%";
        return;
      }
      analyserNode.getByteFrequencyData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) sum += buffer[i];
      let avg = sum / buffer.length;
      let pct = Math.min(100, Math.round((avg / 128) * 100));
      if (isRecording && activeRecordingSpeaker === "caller_b") {
        vuBarB.style.width = pct + "%";
      } else {
        vuBarA.style.width = pct + "%";
      }
    }, 50);

  } catch (err) {
    console.warn("Microphone access not available in browser:", err);
    alert("Microphone permission required. Please allow microphone access in your browser.");
  }
}


function toggleRecording(speakerId) {
  if (!isCalling || !sessionId) {
    alert("Please click 'Start Live Call' first.");
    return;
  }

  const activeBtn = (speakerId === "caller_a") ? btnRecordSpeakerA : btnRecordSpeakerB;
  const otherBtn = (speakerId === "caller_a") ? btnRecordSpeakerB : btnRecordSpeakerA;

  if (!isRecording) {
    // START RECORDING
    isRecording = true;
    activeRecordingSpeaker = speakerId;
    recordedAudioChunks = [];
    recordingStartTime = performance.now();

    activeBtn.style.background = "#ef4444";
    activeBtn.innerHTML = "⏹️ Recording... Click to Stop & Send";
    otherBtn.disabled = true;

    const spkName = (speakerId === "caller_a") ? "Speaker A" : "Speaker B";
    recordingStatus.innerText = `🔴 Recording ${spkName}... Speak your sentence now, then click the button again to stop.`;
  } else {
    // STOP RECORDING & SEND TO BACKEND
    isRecording = false;
    const elapsedSec = Math.max(0.1, (performance.now() - recordingStartTime) / 1000.0);

    activeBtn.style.background = (speakerId === "caller_a") ? "#10b981" : "#8b5cf6";
    activeBtn.innerHTML = (speakerId === "caller_a") ? "🎙️ Speak as Speaker A (Click to Speak)" : "🎙️ Speak as Speaker B (Click to Speak)";
    otherBtn.disabled = false;

    recordingStatus.innerText = "⏳ Processing speech (VAD -> Acoustic LID -> Whisper ASR -> NMT -> Neural Voice Clone)...";

    // Flatten audio chunks
    let totalLen = 0;
    for (let chunk of recordedAudioChunks) totalLen += chunk.length;

    if (totalLen < 1000 || elapsedSec < 0.3) {
      recordingStatus.innerText = "⚠️ Audio was too short. Speak a full sentence and try again.";
      return;
    }

    let combined = new Float32Array(totalLen);
    let offset = 0;
    for (let chunk of recordedAudioChunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Determine true physical sample rate from wall-clock duration
    const physicalRate = Math.round(totalLen / elapsedSec);
    console.log(`Audio stats: ${totalLen} samples captured over ${elapsedSec.toFixed(2)}s -> Physical Rate = ${physicalRate}Hz`);

    // Downsample to 16,000 Hz if recorded at hardware 48kHz or 44.1kHz
    const downsampled = downsampleTo16k(combined, physicalRate);
    console.log(`Resampled: ${combined.length} -> ${downsampled.length} samples @ 16000Hz (Duration: ${(downsampled.length / 16000).toFixed(2)}s)`);

    // Encode to 16kHz WAV Blob in memory
    const wavBlob = encodeWAV(downsampled, 16000);

    // Read as Base64 and send to server
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Audio = reader.result.split(",")[1];
      try {
        const res = await fetch(`/api/call/${sessionId}/audio_turn`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            speaker_id: activeRecordingSpeaker,
            audio_base64: base64Audio,
            client_duration_sec: elapsedSec
          })
        });

        const data = await res.json();
        if (data.status === "success" && data.subtitle) {
          appendSubtitle(data.subtitle);
          updateLatencyHud(data.subtitle);

          // Dynamically show detected language on Speaker card!
          if (activeRecordingSpeaker === "caller_a" && data.subtitle.original_language) {
            const code = data.subtitle.original_language.toLowerCase();
            const langNames = {
              te: "🇮🇳 Telugu (తెలుగు)",
              hi: "🇮🇳 Hindi (हिन्दी)",
              ta: "🇮🇳 Tamil (தமிழ்)",
              kn: "🇮🇳 Kannada (ಕನ್ನಡ)",
              ml: "🇮🇳 Malayalam (മലയാളം)",
              en: "🇬🇧 English",
              mr: "🇮🇳 Marathi",
              bn: "🇮🇳 Bengali",
              gu: "🇮🇳 Gujarati",
              pa: "🇮🇳 Punjabi"
            };
            const label = langNames[code] || `Detected: ${code.toUpperCase()}`;
            langTagA.innerText = `🎯 ${label}`;
            langTagA.style.borderColor = "#10b981";
            langTagA.style.color = "#10b981";
          }

          if (data.audio_base64) {
            playSyntheticAudio(data.audio_base64, activeRecordingSpeaker);
          }
          recordingStatus.innerText = `✅ Processed turn! Target party heard translation in cloned voice.`;
        } else {
          recordingStatus.innerText = `⚠️ ${data.message || "No words detected. Please speak louder."}`;
        }
      } catch (err) {
        console.error("Error processing audio turn:", err);
        recordingStatus.innerText = "❌ Network error processing audio: " + err.message;
      }
    };
    reader.readAsDataURL(wavBlob);
  }
}


// Pure JavaScript High-Fidelity Downsampler (Box Filter Decimation)
function downsampleTo16k(buffer, sampleRate) {
  if (!sampleRate || Math.abs(sampleRate - 16000) < 500) return buffer;
  const ratio = sampleRate / 16000;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? (accum / count) : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}


// Pure JavaScript in-memory 16kHz Mono WAV encoder
function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate = 16000) {
  let buffer = new ArrayBuffer(44 + samples.length * 2);
  let view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: 'audio/wav' });
}


function toggleMute() {
  isMuted = !isMuted;
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach(track => track.enabled = !isMuted);
  }
  btnMuteMic.innerHTML = isMuted ? "🔇 Unmute Mic" : "🎙️ Mute Mic";
  btnMuteMic.style.background = isMuted ? "rgba(239, 68, 68, 0.2)" : "";
}


async function endCall() {
  if (sessionId) {
    await fetch(`/api/call/${sessionId}/terminate`, { method: "POST" });
  }

  if (ws) ws.close();
  if (vuInterval) clearInterval(vuInterval);
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
  }

  isCalling = false;
  isRecording = false;
  btnStartCall.disabled = false;
  btnEndCall.disabled = true;
  btnMuteMic.disabled = true;
  btnRecordSpeakerA.disabled = true;
  btnRecordSpeakerB.disabled = true;
  selectLangA.disabled = false;
  selectLangB.disabled = false;
  vuBarA.style.width = "0%";
  vuBarB.style.width = "0%";

  connStatusText.innerText = "Call Ended (Memory Wiped)";
  connStatusBadge.style.color = "#94a3b8";
  recordingStatus.innerText = "Call ended. Voice profiles cleared from RAM.";
}


async function grantConsent() {
  if (!sessionId) {
    alert("Please click 'Start Live Call' first.");
    consentModal.classList.remove("open");
    return;
  }

  try {
    const res = await fetch(`/api/call/${sessionId}/consent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        speaker_id: "caller_a",
        consent_text: "I authorize this AI system to clone and use my voice for real-time translation."
      })
    });

    const data = await res.json();
    if (data.status === "success") {
      cloneBadgeA.style.display = "inline-flex";
      cloneBadgeA.innerText = `🎙️ Cloned (${data.pitch_adjustment} ${data.estimated_gender})`;
      consentModal.classList.remove("open");
      alert(`Speaker A voice profile enrolled! Adaptations: ${data.pitch_adjustment}, gender: ${data.estimated_gender}`);
    } else {
      alert("Consent verification rejected: " + data.message);
    }
  } catch (e) {
    alert("Error granting consent: " + e.message);
  }
}


// Sample bidirectional turns for Telugu <-> Tamil, Telugu <-> Hindi, etc.
const DIALOGUE_PAIRS = {
  "te_ta": [
    { speaker: "caller_a", lang: "te", text: "నమస్కారం, ప్రాజెక్ట్ పని ఎలా జరుగుతోంది?" },
    { speaker: "caller_b", lang: "ta", text: "வணக்கம்! வேலை மிகச் சிறப்பாகவும் விரைவாகவும் நடக்கிறது." },
    { speaker: "caller_a", lang: "te", text: "చాలా సంతోషం, రేపు ఉదయం 10 గంటలకు మీటింగ్ పెట్టుకుందాం." },
    { speaker: "caller_b", lang: "ta", text: "நிச்சயமாக, நாளை காலை பத்து மணிக்கு சந்திப்போம்." }
  ],
  "te_hi": [
    { speaker: "caller_a", lang: "te", text: "నమస్కారం, ఈరోజు మన ప్రాజెక్ట్ పురోగతి ఎలా ఉంది?" },
    { speaker: "caller_b", lang: "hi", text: "नमस्ते! सब कुछ बहुत अच्छा चल रहा है, सारे टेस्ट पास हो गए हैं." },
    { speaker: "caller_a", lang: "te", text: "చాలా బాగుంది, రేపు ఉదయం కాల్ చేద్దాం." },
    { speaker: "caller_b", lang: "hi", text: "ज़रूर, कल सुबह दस बजे बात करते हैं." }
  ],
  "default": [
    { speaker: "caller_a", lang: "te", text: "నమస్కారం, మీరు ఎలా ఉన్నారు?" },
    { speaker: "caller_b", lang: "en", text: "Hello! I am doing well, how is everything going with you?" },
    { speaker: "caller_a", lang: "te", text: "అంతా చాలా బాగుంది, ధన్యవాదాలు." },
    { speaker: "caller_b", lang: "en", text: "Great to hear, looking forward to working together." }
  ]
};

let turnIdx = 0;


async function simulateTurn() {
  if (!sessionId) {
    await startCall();
  }

  const langA = selectLangA.value;
  const langB = selectLangB.value;
  const key = `${langA}_${langB}`;
  const dialogue = DIALOGUE_PAIRS[key] || DIALOGUE_PAIRS["default"];

  const turn = dialogue[turnIdx % dialogue.length];
  turnIdx++;

  connStatusText.innerText = `Processing turn from ${turn.speaker}...`;

  try {
    const res = await fetch(`/api/call/${sessionId}/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        speaker_id: turn.speaker,
        text: turn.text,
        language: turn.lang
      })
    });

    const data = await res.json();
    if (data.subtitle) {
      appendSubtitle(data.subtitle);
      updateLatencyHud(data.subtitle);
    }

    if (data.audio_base64) {
      playSyntheticAudio(data.audio_base64, turn.speaker);
    }

    connStatusText.innerText = `Call Active (${sessionId})`;
  } catch (err) {
    console.error("Error in turn simulation:", err);
  }
}


function handleWsMessage(msg) {
  if (msg.type === "subtitle") {
    appendSubtitle(msg.data);
    updateLatencyHud(msg.data);
  } else if (msg.type === "audio_playback") {
    playSyntheticAudio(msg.audio_base64, "caller_b");
  }
}


function appendSubtitle(sub) {
  const isA = (sub.speaker_id === "caller_a");
  const senderBox = isA ? subsBoxA : subsBoxB;
  const receiverBox = isA ? subsBoxB : subsBoxA;
  const cloneVoiceUsed = isA ? "Speaker A's Cloned Voice" : "Speaker B's Cloned Voice";

  // 1. Sender Card: Show what the sender spoke
  const senderItem = document.createElement("div");
  senderItem.className = "sub-item" + (isA ? "" : " right");
  senderItem.innerHTML = `
    <span class="sub-orig">🗣️ <strong>Outgoing (${sub.original_language.toUpperCase()}):</strong> "${escapeHtml(sub.original_text)}"</span>
    <span class="sub-meta">Sent by ${escapeHtml(sub.speaker_name)}</span>
  `;
  senderBox.appendChild(senderItem);
  senderBox.scrollTop = senderBox.scrollHeight;

  // 2. Receiver Card: Show what the receiver hears in their native language in the other person's voice
  const receiverItem = document.createElement("div");
  receiverItem.className = "sub-item" + (isA ? "" : " right");
  receiverItem.style.borderLeftColor = isA ? "#10b981" : "#a855f7";
  receiverItem.innerHTML = `
    <span class="sub-trans" style="font-weight: 600; font-size: 0.95rem; color: #00f2fe;">🔊 <strong>Incoming Translated (${sub.target_language.toUpperCase()}):</strong> "${escapeHtml(sub.translated_text)}"</span>
    <span class="sub-meta">🎙️ Heard in <strong>${cloneVoiceUsed}</strong> • Latency: ${Math.round(sub.total_latency_ms)}ms</span>
  `;
  receiverBox.appendChild(receiverItem);
  receiverBox.scrollTop = receiverBox.scrollHeight;
}


function updateLatencyHud(sub) {
  if (sub.asr_latency_ms) hudAsr.innerText = `${Math.round(sub.asr_latency_ms)} ms`;
  if (sub.translation_latency_ms) hudNmt.innerText = `${Math.round(sub.translation_latency_ms)} ms`;
  if (sub.tts_latency_ms) hudTts.innerText = `${Math.round(sub.tts_latency_ms)} ms`;
  if (sub.total_latency_ms) hudTotal.innerText = `${Math.round(sub.total_latency_ms)} ms`;
}


function playSyntheticAudio(base64Audio, speakerId) {
  try {
    const snd = new Audio("data:audio/wav;base64," + base64Audio);
    const activeVu = (speakerId === "caller_a") ? vuBarB : vuBarA;
    activeVu.style.width = "80%";
    snd.onended = () => {
      activeVu.style.width = "0%";
    };
    snd.play();
  } catch (e) {
    console.warn("Playback error:", e);
  }
}


function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"']/g, (m) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[m]);
}
