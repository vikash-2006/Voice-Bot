import os
import json
import urllib.parse
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import Response, FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.stt import transcribe_audio_file
from app.services.llm import generate_response, session_memory
from app.services.tts import generate_audio
from app.services.logger import (
    app_logger,
    log_interaction,
    get_recent_interactions,
    get_log_stats,
    list_chat_sessions,
    get_session_details,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="Voice Assistant API (Groq + ElevenLabs)",
    description="Full-duplex Voice Assistant using Groq Whisper STT, Groq LLaMA 3.3 LLM, and ElevenLabs TTS with structured logging.",
    version="2.0.0",
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Transcript",
        "X-Assistant-Response",
        "X-Latency-STT",
        "X-Latency-LLM",
        "X-Latency-TTS",
        "X-Latency-Total",
        "X-Session-ID",
        "X-TTS-Provider",
        "X-TTFT",
        "X-Chunks-Count",
        "X-Tools-Called",
        "X-Tools-Payload",
        "X-Tokens-Used",
        "X-Prompt-Tokens",
        "X-Completion-Tokens",
    ],
)

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse, summary="Voice Assistant Web Interface")
def home():
    possible_paths = [
        os.path.join(STATIC_DIR, "index.html"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "index.html")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "index.html")),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse("<h1>Voice Assistant Backend is active.</h1>", status_code=200)


@app.get("/health", summary="Health check")
@app.get("/api/health", summary="API Health check")
def health():
    raw_groq = os.getenv("GROQ_API_KEY", "")
    clean_groq = raw_groq.strip().strip('"').strip("'").strip()
    raw_eleven = os.getenv("ELEVENLABS_API_KEY", "")
    clean_eleven = raw_eleven.strip().strip('"').strip("'").strip()

    return {
        "status": "healthy" if clean_groq else "degraded",
        "groq": {
            "configured": bool(clean_groq),
            "starts_with": clean_groq[:7] if clean_groq else "",
            "ends_with": clean_groq[-4:] if len(clean_groq) > 10 else "",
            "length": len(clean_groq),
            "had_quotes": (raw_groq.startswith('"') or raw_groq.startswith("'")),
        },
        "elevenlabs": {
            "configured": bool(clean_eleven),
            "starts_with": clean_eleven[:7] if clean_eleven else "",
            "length": len(clean_eleven),
        },
        "server_environment": "vercel" if os.environ.get("VERCEL") else "local",
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = "default"


@app.post("/api/chat", summary="Text-only Chat")
def text_chat(request: ChatRequest):
    try:
        result = generate_response(
            user_text=request.message, session_id=request.session_id or "default"
        )
        return {
            "user_message": request.message,
            "assistant_response": result["response"],
            "latency_ms": result["latency_ms"],
            "tokens_used": result["tokens_used"],
            "tools_called": result.get("tools_called", []),
        }
    except Exception as e:
        app_logger.exception("Text chat processing failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/api/stt", summary="Speech-to-Text")
def transcribe_audio(
    audio: UploadFile = File(...),
    provider: Optional[str] = None,
):
    try:
        audio_bytes = audio.file.read()
        result = transcribe_audio_file(
            audio_bytes, filename=audio.filename or "speech.wav", provider=provider
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        app_logger.exception("STT request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        audio.file.close()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: Optional[str] = None


@app.post("/api/tts", summary="Text-to-Speech")
def synthesize_speech(request: TTSRequest):
    try:
        audio_bytes, latency_ms, provider = generate_audio(request.text, voice_id=request.voice_id)
        media_type = "audio/mpeg" if provider == "elevenlabs" else "audio/wav"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={"X-Latency-TTS": str(latency_ms), "X-TTS-Provider": provider},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        app_logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/api/voice-chat", summary="Complete Voice Chat Pipeline")
def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form("default"),
    format: Optional[str] = Query("audio", description="'audio' for stream, 'json' for JSON payload"),
):
    """
    End-to-End Voice Pipeline:
    1. Audio -> Groq Whisper Speech-To-Text
    2. Transcribed Text -> Groq LLaMA 3.3 LLM (with memory)
    3. LLM Answer -> ElevenLabs Text-To-Speech
    4. Persists structured log in logs/interactions.jsonl
    5. Returns synthesized audio with metadata headers
    """
    audio_bytes = audio.file.read()
    audio.file.close()

    input_size = len(audio_bytes)
    if input_size < 350:
        raise HTTPException(status_code=400, detail="Audio recording too short. Please speak again.")

    stt_latency = 0.0
    llm_latency = 0.0
    tts_latency = 0.0
    user_text = ""
    assistant_reply = ""
    tokens = None

    try:
        # Step 1: STT
        stt_result = transcribe_audio_file(
            audio_bytes, filename=audio.filename or "recording.webm"
        )
        user_text = stt_result["transcript"]
        stt_latency = stt_result["latency_ms"]

        if not user_text.strip():
            log_interaction(
                session_id=session_id or "default",
                user_transcript="[No speech detected]",
                assistant_response="[Ignored]",
                stt_latency_ms=stt_latency,
                audio_input_size_bytes=input_size,
                status="no_speech",
            )
            raise HTTPException(status_code=400, detail="No speech detected in audio.")

        # Step 2: LLM
        llm_result = generate_response(
            user_text=user_text, session_id=session_id or "default"
        )
        assistant_reply = llm_result["response"]
        llm_latency = llm_result["latency_ms"]
        llm_ttft = llm_result.get("ttft_ms")
        chunks_count = llm_result.get("chunks_count", 1)
        tokens = llm_result["tokens_used"]

        # Step 3: TTS
        audio_output_bytes, tts_latency, tts_provider = generate_audio(assistant_reply)
        total_latency = round(stt_latency + llm_latency + tts_latency, 2)

        # Step 4: Persist structured log
        log_interaction(
            session_id=session_id or "default",
            user_transcript=user_text,
            assistant_response=assistant_reply,
            stt_latency_ms=stt_latency,
            llm_latency_ms=llm_latency,
            tts_latency_ms=tts_latency,
            ttft_ms=llm_ttft,
            chunks_count=chunks_count,
            audio_input_size_bytes=input_size,
            audio_output_size_bytes=len(audio_output_bytes),
            tokens_used=tokens,
            status="success",
        )

        tools_called = llm_result.get("tools_called", [])
        tools_str = ",".join([t["name"] for t in tools_called]) if tools_called else "none"

        if format == "json":
            return {
                "session_id": session_id,
                "transcript": user_text,
                "assistant_response": assistant_reply,
                "tools_called": tools_called,
                "audio_base64": base64.b64encode(audio_output_bytes).decode("utf-8"),
                "tts_provider": tts_provider,
                "metrics": {
                    "stt_ms": stt_latency,
                    "llm_ms": llm_latency,
                    "ttft_ms": llm_ttft,
                    "tts_ms": tts_latency,
                    "total_ms": total_latency,
                    "chunks_count": chunks_count,
                    "tokens_used": tokens,
                    "prompt_tokens": llm_result.get("prompt_tokens", 0),
                    "completion_tokens": llm_result.get("completion_tokens", 0),
                },
            }

        headers = {
            "X-Transcript": urllib.parse.quote(user_text),
            "X-Assistant-Response": urllib.parse.quote(assistant_reply),
            "X-Latency-STT": str(stt_latency),
            "X-Latency-LLM": str(llm_latency),
            "X-Latency-TTS": str(tts_latency),
            "X-Latency-Total": str(total_latency),
            "X-TTFT": str(llm_ttft or 0),
            "X-Chunks-Count": str(chunks_count),
            "X-Session-ID": str(session_id),
            "X-TTS-Provider": str(tts_provider),
            "X-Tools-Called": urllib.parse.quote(tools_str),
            "X-Tools-Payload": urllib.parse.quote(json.dumps(tools_called)),
            "X-Tokens-Used": str(tokens or 0),
            "X-Prompt-Tokens": str(llm_result.get("prompt_tokens", 0)),
            "X-Completion-Tokens": str(llm_result.get("completion_tokens", 0)),
        }

        media_type = "audio/mpeg" if tts_provider in ["elevenlabs", "edge_tts"] else "audio/wav"

        return Response(
            content=audio_output_bytes,
            media_type=media_type,
            headers=headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception("Voice chat pipeline failed")
        log_interaction(
            session_id=session_id or "default",
            user_transcript=user_text or "[Failed before transcription]",
            assistant_response="",
            stt_latency_ms=stt_latency,
            llm_latency_ms=llm_latency,
            tts_latency_ms=tts_latency,
            audio_input_size_bytes=input_size,
            status="error",
            error_message=str(e),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Voice pipeline failed: {e}",
        ) from e


@app.get("/api/logs", summary="Get recent interaction logs and stats")
def get_logs(limit: int = Query(50, ge=1, le=200)):
    return {
        "stats": get_log_stats(),
        "records": get_recent_interactions(limit=limit),
    }


@app.get("/api/sessions", summary="List all individual chat session log files")
def get_sessions():
    return {"sessions": list_chat_sessions()}


@app.get("/api/sessions/{session_id}", summary="Get logs for a specific chat session")
def get_session(session_id: str):
    return get_session_details(session_id)


class InterruptRequest(BaseModel):
    session_id: str


@app.post("/api/interrupt", summary="Signal client-side barge-in interruption")
def interrupt_session(request: InterruptRequest):
    session_memory.interrupt_current_turn(request.session_id)
    app_logger.info(f"Barge-In interrupt received for session: {request.session_id}")
    return {"status": "interrupted", "session_id": request.session_id}


# ==========================================
# Model Context Protocol (MCP) Endpoints
# ==========================================
from app.mcp.medical_server import get_mcp_tools_list
from app.services.medical_tools import execute_tool


@app.get("/api/mcp/tools", summary="List all Model Context Protocol (MCP) Tools")
def list_mcp_tools():
    """Returns the list of all active MCP medical and live search tools."""
    tools = get_mcp_tools_list()
    return {
        "status": "active",
        "count": len(tools),
        "protocol": "Model Context Protocol (MCP)",
        "tools": tools,
    }


class MCPCallRequest(BaseModel):
    tool: str
    arguments: Optional[Dict[str, Any]] = {}


@app.post("/api/mcp/call", summary="Directly execute an MCP Tool")
def call_mcp_tool(request: MCPCallRequest):
    """Directly invokes an MCP tool by name with arguments."""
    result = execute_tool(request.tool, request.arguments or {})
    return {
        "tool": request.tool,
        "arguments": request.arguments,
        "result": result,
    }


# Backward compatibility routes
@app.post("/transcribe", include_in_schema=False)
def legacy_transcribe(audio: UploadFile = File(...)):
    return transcribe_audio(audio)


@app.post("/chat", include_in_schema=False)
def legacy_chat(request: ChatRequest):
    return text_chat(request)


@app.post("/voice-chat", include_in_schema=False)
def legacy_voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form("default"),
):
    return voice_chat(audio, session_id=session_id)
