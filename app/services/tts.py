import os
import time
import subprocess
import tempfile
from typing import Tuple
from dotenv import load_dotenv
import httpx
from app.services.logger import app_logger

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb").strip()
DEFAULT_TTS_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()


# Persistent HTTP client to reuse connections
_http_client = httpx.Client(timeout=10.0)
_elevenlabs_permission_denied = True  # Key lacks text_to_speech permission; bypass to eliminate network latency


def generate_audio(
    text: str,
    voice_id: str = None,
    model_id: str = None,
) -> Tuple[bytes, float, str]:
    """
    Synthesize text into speech audio.
    Uses ultra-fast macOS native Indian speech (Lekha/Rishi) directly to WAV for sub-200ms latency.
    Returns:
        (audio_bytes, latency_ms, provider_used)
    """
    global _elevenlabs_permission_denied
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text for audio synthesis cannot be empty")

    target_voice_id = voice_id or DEFAULT_VOICE_ID
    target_model_id = model_id or DEFAULT_TTS_MODEL

    start_time = time.perf_counter()

    # Only attempt ElevenLabs if explicitly enabled with working key
    if ELEVENLABS_API_KEY and not _elevenlabs_permission_denied:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{target_voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": cleaned_text,
            "model_id": target_model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            response = _http_client.post(url, headers=headers, json=payload)
            if response.status_code == 200 and len(response.content) > 0:
                latency_ms = (time.perf_counter() - start_time) * 1000
                app_logger.info(
                    f"[Step 3/3: ElevenLabs Complete] Audio synthesized in {latency_ms:.1f}ms ({len(response.content)} bytes)"
                )
                return response.content, round(latency_ms, 2), "elevenlabs"

            if response.status_code == 401 and "missing_permissions" in response.text:
                _elevenlabs_permission_denied = True
                app_logger.warning(
                    "⚠️ ElevenLabs API Key is missing 'text_to_speech' permission! "
                    "Cached status -> Bypassing ElevenLabs to save network delay. "
                    "Using ultra-fast macOS native Indian speech (Lekha/Rishi)..."
                )
        except Exception as e:
            app_logger.warning(f"ElevenLabs call failed ({e}). Attempting local voice...")

    import sys

    # On macOS: Ultra-fast native high-quality speech directly to WAV (~150ms)
    if sys.platform == "darwin":
        try:
            backup_bytes, backup_latency = _generate_macos_tts(cleaned_text, start_time)
            return backup_bytes, backup_latency, "macos_native"
        except Exception as e:
            app_logger.warning(f"macOS TTS failed ({e}), falling back to Edge-TTS...")

    # On Linux / Cloud / Vercel: Microsoft Neural TTS (Edge-TTS)
    try:
        edge_bytes, edge_latency = _generate_edge_tts(cleaned_text, start_time)
        return edge_bytes, edge_latency, "edge_tts"
    except Exception as e:
        app_logger.error(f"All TTS engines failed: {e}", exc_info=True)
        raise RuntimeError(f"Text-to-speech failed: {e}") from e


def _generate_edge_tts(text: str, start_time: float) -> Tuple[bytes, float]:
    """Fallback high-quality Microsoft Neural TTS for Linux/Vercel serverless environments."""
    import asyncio
    import edge_tts

    has_hindi = any("\u0900" <= ch <= "\u097F" for ch in text)
    voice = "hi-IN-MadhurNeural" if has_hindi else "en-IN-PrabhatNeural"

    async def _synth():
        communicate = edge_tts.Communicate(text, voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    try:
        # If running inside an existing event loop, create task or new loop in thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    audio_bytes = pool.submit(lambda: asyncio.run(_synth())).result()
            else:
                audio_bytes = loop.run_until_complete(_synth())
        except RuntimeError:
            audio_bytes = asyncio.run(_synth())

        latency_ms = (time.perf_counter() - start_time) * 1000
        app_logger.info(f"[Edge-TTS ({voice})] Generated {len(audio_bytes)} bytes in {latency_ms:.1f}ms")
        return audio_bytes, round(latency_ms, 2)
    except Exception as e:
        app_logger.error(f"Edge-TTS synthesis error: {e}")
        raise


def _generate_macos_tts(text: str, start_time: float) -> Tuple[bytes, float]:
    """
    Generates audio using macOS built-in speech engine directly to WAV:
    - Uses 'Lekha' (hi_IN) for pure Devanagari Hindi text
    - Uses 'Rishi' (en_IN) for Indian English / Hinglish
    - Direct WAV output in 1 process without intermediate AIFF or afconvert
    """
    has_hindi = any("\u0900" <= ch <= "\u097F" for ch in text)
    voice = "Lekha" if has_hindi else "Rishi"

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as txt_tmp:
        txt_tmp.write(text)
        txt_path = txt_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name

    try:
        cmd_say = [
            "say",
            "-r", "195",
            "-v", voice,
            "-f", txt_path,
            "-o", wav_path,
            "--data-format=LEI16@22050",
        ]
        res = subprocess.run(cmd_say, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            subprocess.run(
                ["say", "-r", "195", "-f", txt_path, "-o", wav_path, "--data-format=LEI16@22050"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        latency_ms = (time.perf_counter() - start_time) * 1000
        app_logger.info(
            f"[macOS Voice: {voice}] Full response spoken cleanly ({len(audio_bytes)} bytes) in {latency_ms:.1f}ms"
        )
        return audio_bytes, round(latency_ms, 2)

    finally:
        for p in [txt_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
