"""Audio transcription and speech synthesis utilities for Live Voice Mode."""

import base64
import io
import re
import streamlit.components.v1 as components
import httpx
from src.config import settings

def clean_text_for_speech(text: str) -> str:
    """Cleans markdown, links, code blocks, and symbols for natural voice synthesis."""
    if not text:
        return ""
    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Replace markdown links [label](url) with just label
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    # Remove raw URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove headers, bullets, bold, italics
    cleaned = re.sub(r"^[#\s*>-]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~]{1,3}", "", cleaned)
    # Collapse multiple whitespaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def transcribe_audio_groq(audio_bytes: bytes) -> str:
    """Transcribes audio using Groq Whisper API (ultra-low latency)."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")
    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"model": "whisper-large-v3-turbo"}
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    r = httpx.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        files=files,
        data=data,
        headers=headers,
        timeout=15.0,
    )
    r.raise_for_status()
    result = r.json()
    return result.get("text", "").strip()

def transcribe_audio_gemini(audio_bytes: bytes) -> str:
    """Transcribes audio using Google Gemini multimodal audio API."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not configured")
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GOOGLE_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                {"text": "Transcribe the user's spoken question accurately in plain text. If only silence or noise, reply with EMPTY."}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 200,
        }
    }
    r = httpx.post(url, json=payload, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts:
            text = parts[0].get("text", "").strip()
            if text.upper() in ["EMPTY", "SILENCE", "NO AUDIO", "NO_SPEECH"]:
                return ""
            return text
    return ""

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribes audio with automatic fallback between Groq Whisper and Gemini."""
    if not audio_bytes or len(audio_bytes) < 200:
        return ""
    # 1. Try Groq Whisper (ultra-fast, ~200ms)
    if settings.GROQ_API_KEY:
        try:
            text = transcribe_audio_groq(audio_bytes)
            if text and text.lower() not in ["thank you.", "thank you", "silence", "you"]:
                return text
        except Exception:
            pass

    # 2. Try Gemini audio transcription
    if settings.GOOGLE_API_KEY:
        try:
            return transcribe_audio_gemini(audio_bytes)
        except Exception:
            pass

    return ""

def render_speech_synthesis(text: str, auto_play: bool = True):
    """Renders HTML/JS using Web Speech API for natural browser-based voice responses."""
    clean = clean_text_for_speech(text)
    if not clean:
        return
    import json
    json_text = json.dumps(clean)
    auto_code = "speak();" if auto_play else ""
    html = f"""
    <div style="margin-top: 6px;">
        <button id="tts-play-btn" onclick="speak()" style="
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: inherit;
            padding: 4px 10px;
            border-radius: 14px;
            font-size: 0.8rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.2s ease;
        ">
            🔊 Listen to response
        </button>
        <span id="tts-status" style="font-size: 0.75rem; margin-left: 8px; opacity: 0.7;"></span>
    </div>
    <script>
    function speak() {{
        if (!('speechSynthesis' in window)) {{
            document.getElementById('tts-status').innerText = 'TTS not supported';
            return;
        }}
        window.speechSynthesis.cancel();
        const text = {json_text};
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        const setVoiceAndSpeak = () => {{
            const voices = window.speechSynthesis.getVoices();
            // Choose a clear English voice
            const voice = voices.find(v => v.lang.startsWith('en') && (
                v.name.includes('Google') || 
                v.name.includes('Natural') || 
                v.name.includes('Samantha') || 
                v.name.includes('Daniel')
            )) || voices.find(v => v.lang.startsWith('en'));
            if (voice) utterance.voice = voice;

            const btn = document.getElementById('tts-play-btn');
            const status = document.getElementById('tts-status');

            utterance.onstart = () => {{
                if (btn) btn.innerText = '⏹️ Stop speaking';
                if (status) status.innerText = 'Speaking...';
            }};
            utterance.onend = () => {{
                if (btn) btn.innerText = '🔊 Listen again';
                if (status) status.innerText = '';
            }};
            utterance.onerror = () => {{
                if (btn) btn.innerText = '🔊 Listen';
                if (status) status.innerText = '';
            }};

            window.speechSynthesis.speak(utterance);
        }};

        if (window.speechSynthesis.getVoices().length > 0) {{
            setVoiceAndSpeak();
        }} else {{
            window.speechSynthesis.onvoiceschanged = setVoiceAndSpeak;
        }}
    }}
    {auto_code}
    </script>
    """
    components.html(html, height=45)
