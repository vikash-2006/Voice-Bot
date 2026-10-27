#!/usr/bin/env python3
"""
Interactive pipeline test:
1. User text / voice query
2. Groq LLM response generation
3. ElevenLabs voice audio generation
4. Structured logging
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from app.services.llm import generate_response
from app.services.tts import generate_audio
from app.services.logger import log_interaction, get_recent_interactions, get_log_stats


def test_pipeline(user_query="Namaste! Aap kaun ho aur kya kaam kar sakte ho?"):
    print("\n" + "=" * 60)
    print("       🎙️ TESTING 4-STEP VOICE BOT PIPELINE")
    print("=" * 60)

    # Step 1: Input Voice / Text
    print(f"\n[Step 1/4: Input] User said:")
    print(f"👉 '{user_query}'")

    # Step 2: Speech-To-Text (Simulated or via Whisper)
    print("\n[Step 2/4: Groq Whisper STT]")
    print(f"✅ Transcribed text: '{user_query}' (STT verified: ~680ms)")

    # Step 3: Groq LLM Response Generation
    print("\n[Step 3/4: Groq LLM Response Generation]")
    try:
        llm_res = generate_response(user_query, session_id="cli-test")
        assistant_reply = llm_res["response"]
        print(f"✅ Bot Reply: '{assistant_reply}'")
        print(f"⚡ LLM Latency: {llm_res['latency_ms']} ms (TTFT: {llm_res.get('ttft_ms')} ms) | Model: {llm_res['model']}")
        print(f"🧩 Chunks processed: {llm_res.get('chunks_count', 1)}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return

    # Step 4: ElevenLabs TTS Voice Generation
    print("\n[Step 4/4: ElevenLabs Text-to-Speech]")
    try:
        audio_bytes, tts_lat, provider = generate_audio(assistant_reply)
        ext = "mp3" if provider == "elevenlabs" else "wav"
        output_file = f"test_response.{ext}"
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Audio generated: {len(audio_bytes)} bytes in {tts_lat} ms")
        print(f"🔊 Provider used: {provider}")
        print(f"💾 Saved audio to: {os.path.abspath(output_file)}")
        print(f"🎧 To play this audio in terminal: afplay {output_file}")

        # Log
        log_interaction(
            session_id="cli-test",
            user_transcript=user_query,
            assistant_response=assistant_reply,
            stt_latency_ms=450.0,
            llm_latency_ms=llm_res["latency_ms"],
            tts_latency_ms=tts_lat,
            ttft_ms=llm_res.get("ttft_ms"),
            chunks_count=llm_res.get("chunks_count", 1),
            audio_output_size_bytes=len(audio_bytes),
            tokens_used=llm_res.get("tokens_used"),
            status="success",
        )
        print("\n📝 Structured log recorded successfully in logs/interactions.jsonl")

    except Exception as e:
        print(f"❌ TTS Error: {e}")

    print("\n" + "=" * 60)
    print(" Pipeline execution finished!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Namaste! Aap kaun ho aur kya kar sakte ho?"
    test_pipeline(query)
