import io
import os
import time
from typing import Dict, Any, Union
from dotenv import load_dotenv
from groq import Groq
from app.services.logger import app_logger

load_dotenv(override=True)

def get_clean_key(env_name: str) -> str:
    raw = os.getenv(env_name, "")
    return raw.strip().strip('"').strip("'").strip()

def get_groq_client():
    key = get_clean_key("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key, timeout=15.0, max_retries=2)

groq_client = get_groq_client()


def transcribe_audio_file(
    audio_file_or_bytes: Union[io.BytesIO, bytes, Any],
    filename: str = "audio.wav",
    provider: str = None,
) -> Dict[str, Any]:
    """
    Transcribe audio using Groq Whisper (ultra-fast, recommended) or ElevenLabs Scribe.
    Returns:
        {
            "transcript": str,
            "language": str,
            "latency_ms": float,
            "provider": str
        }
    """
    target_provider = (provider or STT_PROVIDER or "groq").lower()
    start_time = time.perf_counter()

    # Read audio bytes
    if isinstance(audio_file_or_bytes, bytes):
        audio_bytes = audio_file_or_bytes
    elif hasattr(audio_file_or_bytes, "read"):
        audio_bytes = audio_file_or_bytes.read()
    else:
        raise ValueError("Unsupported audio data format")

    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")

    app_logger.info(f"Starting STT with provider '{target_provider}', audio size: {len(audio_bytes)} bytes")

    if target_provider == "groq":
        return _transcribe_groq(audio_bytes, filename, start_time)
    elif target_provider in ["elevenlabs", "11labs"]:
        return _transcribe_elevenlabs(audio_bytes, filename, start_time)
    else:
        app_logger.warning(f"Unknown provider '{target_provider}', falling back to Groq")
        return _transcribe_groq(audio_bytes, filename, start_time)


def _transcribe_groq(audio_bytes: bytes, filename: str, start_time: float) -> Dict[str, Any]:
    client = get_groq_client() or groq_client
    if not client:
        raise RuntimeError("GROQ_API_KEY is missing in environment variables.")

    # In Groq API, file must be a tuple: (filename, bytes_data)
    file_payload = (filename or "recording.wav", audio_bytes)
    
    # Use whisper-large-v3 for highest accuracy with multilingual/Hinglish speech
    stt_model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()
    
    try:
        response = client.audio.transcriptions.create(
            file=file_payload,
            model=stt_model,
            response_format="json",
            temperature=0.0,
            prompt="Namaste, kaise hain aap? Swasthya aur medical baatcheet in Hinglish and Hindi.",
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        text = getattr(response, "text", "") or ""
        detected_lang = getattr(response, "language", "unknown")

        app_logger.info(f"Groq Whisper STT completed in {latency_ms:.1f}ms. Detected language: {detected_lang}")

        return {
            "transcript": text.strip(),
            "language": detected_lang,
            "latency_ms": round(latency_ms, 2),
            "provider": "groq",
        }
    except Exception as e:
        err_msg = str(e).lower()
        if any(w in err_msg for w in ["audio_too_short", "too short", "invalid_media_file", "could not process file"]):
            app_logger.warning(f"Groq Whisper rejected audio snippet ({err_msg[:80]}), treating as empty transcript.")
            return {
                "transcript": "",
                "language": "unknown",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "provider": "groq",
            }
        app_logger.error(f"Groq STT transcription failed: {e}", exc_info=True)
        raise RuntimeError(f"Speech transcription failed: {e}") from e


def _transcribe_elevenlabs(audio_bytes: bytes, filename: str, start_time: float) -> Dict[str, Any]:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing in .env")

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.speech_to_text.convert(
            file=audio_bytes,
            model_id="scribe_v2",
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "transcript": response.text.strip(),
            "language": getattr(response, "language_code", "unknown"),
            "latency_ms": round(latency_ms, 2),
            "provider": "elevenlabs",
        }
    except Exception as e:
        app_logger.error(f"ElevenLabs STT failed: {e}", exc_info=True)
        raise RuntimeError(f"ElevenLabs speech-to-text failed: {e}") from e
