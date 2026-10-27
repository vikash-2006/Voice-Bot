import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, List, Optional

# Base logs directories (use /tmp/logs on Vercel serverless where root is read-only)
if os.environ.get("VERCEL"):
    LOGS_DIR = "/tmp/logs"
else:
    LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))

SESSIONS_LOG_DIR = os.path.join(LOGS_DIR, "sessions")

try:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SESSIONS_LOG_DIR, exist_ok=True)
except Exception as e:
    pass

APP_LOG_FILE = os.path.join(LOGS_DIR, "app.log")
INTERACTIONS_LOG_FILE = os.path.join(LOGS_DIR, "interactions.jsonl")

# Setup application logger
app_logger = logging.getLogger("voice_assistant")
app_logger.setLevel(logging.INFO)

# Console Handler
if not any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    app_logger.addHandler(console_handler)

# App Rotating File Handler (Max 10MB, keep 5 backups)
if not any(isinstance(h, RotatingFileHandler) for h in app_logger.handlers):
    file_handler = RotatingFileHandler(
        APP_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    app_logger.addHandler(file_handler)


def log_interaction(
    session_id: str,
    user_transcript: str,
    assistant_response: str,
    stt_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    tts_latency_ms: float = 0.0,
    audio_input_size_bytes: int = 0,
    audio_output_size_bytes: int = 0,
    tokens_used: Optional[int] = None,
    ttft_ms: Optional[float] = None,
    chunks_count: int = 1,
    status: str = "success",
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist structured interaction log to:
    1. Master interactions.jsonl
    2. Dedicated session-specific log file: logs/sessions/{session_id}.jsonl
    3. Dedicated session-specific readable chat log: logs/sessions/{session_id}.log
    """
    total_latency_ms = round(stt_latency_ms + llm_latency_ms + tts_latency_ms, 2)
    timestamp = datetime.now().isoformat()
    clean_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")) or "default"

    record: Dict[str, Any] = {
        "timestamp": timestamp,
        "session_id": clean_session_id,
        "status": status,
        "user_transcript": user_transcript,
        "assistant_response": assistant_response,
        "metrics": {
            "stt_latency_ms": round(stt_latency_ms, 2),
            "llm_latency_ms": round(llm_latency_ms, 2),
            "tts_latency_ms": round(tts_latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
            "total_latency_ms": total_latency_ms,
            "chunks_count": chunks_count,
            "audio_input_size_bytes": audio_input_size_bytes,
            "audio_output_size_bytes": audio_output_size_bytes,
            "tokens_used": tokens_used,
        },
    }

    if error_message:
        record["error"] = error_message

    record_json = json.dumps(record, ensure_ascii=False) + "\n"

    # 1. Write to Master JSONL
    try:
        with open(INTERACTIONS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(record_json)
    except Exception as e:
        app_logger.error(f"Failed to write to master interaction log: {e}")

    # 2. Write to Dedicated Session JSONL
    session_jsonl_file = os.path.join(SESSIONS_LOG_DIR, f"{clean_session_id}.jsonl")
    try:
        with open(session_jsonl_file, "a", encoding="utf-8") as f:
            f.write(record_json)
    except Exception as e:
        app_logger.error(f"Failed to write to session JSONL {session_jsonl_file}: {e}")

    # 3. Write to Dedicated Session Human-Readable Log
    session_text_log_file = os.path.join(SESSIONS_LOG_DIR, f"{clean_session_id}.log")
    try:
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_text = (
            f"[{formatted_time}] Status: {status.upper()}\n"
            f"👤 USER ({chunks_count} chunk{'s' if chunks_count > 1 else ''}):\n{user_transcript}\n"
            f"🤖 ASSISTANT:\n{assistant_response or '[No response / Error]'}\n"
            f"⚡ LATENCY: Total {total_latency_ms}ms (STT: {stt_latency_ms}ms, LLM: {llm_latency_ms}ms, TTFT: {ttft_ms or 0}ms, TTS: {tts_latency_ms}ms) | Tokens: {tokens_used or 'N/A'}\n"
        )
        if error_message:
            entry_text += f"⚠️ ERROR: {error_message}\n"
        entry_text += "-" * 75 + "\n"

        with open(session_text_log_file, "a", encoding="utf-8") as f:
            f.write(entry_text)
    except Exception as e:
        app_logger.error(f"Failed to write to session log {session_text_log_file}: {e}")

    # Log to main app log
    if status == "success":
        app_logger.info(
            f"[Chat: {clean_session_id[:8]}] User: '{user_transcript[:60]}...' -> "
            f"Bot: '{assistant_response[:60]}...' | "
            f"Latency: {total_latency_ms}ms (STT: {stt_latency_ms:.0f}ms, LLM: {llm_latency_ms:.0f}ms, TTS: {tts_latency_ms:.0f}ms) | "
            f"Tokens: {tokens_used} [Session Log: sessions/{clean_session_id}.log]"
        )
    else:
        app_logger.error(
            f"[Chat: {clean_session_id[:8]}] Interaction failed: {error_message} | "
            f"User: '{user_transcript[:60]}...'"
        )

    return record


def get_recent_interactions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve most recent interactions in reverse chronological order."""
    if not os.path.exists(INTERACTIONS_LOG_FILE):
        return []

    lines: List[str] = []
    try:
        with open(INTERACTIONS_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        app_logger.error(f"Error reading interactions file: {e}")
        return []

    recent_lines = lines[-limit:]
    records = []
    for line in reversed(recent_lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def list_chat_sessions() -> List[Dict[str, Any]]:
    """List all individual chat session log files created."""
    if not os.path.exists(SESSIONS_LOG_DIR):
        return []

    sessions = []
    for fname in os.listdir(SESSIONS_LOG_DIR):
        if fname.endswith(".jsonl"):
            session_id = fname[:-6]
            filepath = os.path.join(SESSIONS_LOG_DIR, fname)
            mtime = os.path.getmtime(filepath)
            
            # Count turns
            turns = 0
            with open(filepath, "r", encoding="utf-8") as f:
                turns = sum(1 for line in f if line.strip())

            sessions.append({
                "session_id": session_id,
                "turns_count": turns,
                "last_updated": datetime.fromtimestamp(mtime).isoformat(),
                "log_file": f"logs/sessions/{session_id}.log",
                "jsonl_file": f"logs/sessions/{session_id}.jsonl",
            })

    sessions.sort(key=lambda s: s["last_updated"], reverse=True)
    return sessions


def get_session_details(session_id: str) -> Dict[str, Any]:
    """Retrieve all turns for a specific chat session."""
    clean_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
    jsonl_file = os.path.join(SESSIONS_LOG_DIR, f"{clean_session_id}.jsonl")
    log_file = os.path.join(SESSIONS_LOG_DIR, f"{clean_session_id}.log")

    records = []
    if os.path.exists(jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    log_text = ""
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log_text = f.read()

    return {
        "session_id": clean_session_id,
        "turns": records,
        "text_transcript": log_text,
    }


def get_log_stats() -> Dict[str, Any]:
    """Compute summary metrics from stored interaction logs."""
    interactions = get_recent_interactions(limit=500)
    total_count = len(interactions)
    if total_count == 0:
        return {
            "total_conversations": 0,
            "avg_latency_ms": 0,
            "avg_stt_ms": 0,
            "avg_llm_ms": 0,
            "avg_tts_ms": 0,
            "success_rate": 100.0,
        }

    successful = [item for item in interactions if item.get("status") == "success"]
    success_count = len(successful)

    if success_count > 0:
        avg_total = sum(item["metrics"]["total_latency_ms"] for item in successful) / success_count
        avg_stt = sum(item["metrics"]["stt_latency_ms"] for item in successful) / success_count
        avg_llm = sum(item["metrics"]["llm_latency_ms"] for item in successful) / success_count
        avg_tts = sum(item["metrics"]["tts_latency_ms"] for item in successful) / success_count
    else:
        avg_total = avg_stt = avg_llm = avg_tts = 0.0

    return {
        "total_conversations": total_count,
        "successful_conversations": success_count,
        "success_rate": round((success_count / total_count) * 100, 1),
        "avg_latency_ms": round(avg_total, 1),
        "avg_stt_ms": round(avg_stt, 1),
        "avg_llm_ms": round(avg_llm, 1),
        "avg_tts_ms": round(avg_tts, 1),
    }
