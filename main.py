import sys
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def check_environment():
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

    print("\n" + "=" * 55)
    print("      🎙️  AI VOICE ASSISTANT (Groq + ElevenLabs)  ")
    print("=" * 55)
    print(f" Groq API Key       : {'[CONFIGURED]' if groq_key else '[MISSING]'}")
    print(f" ElevenLabs API Key : {'[CONFIGURED]' if eleven_key else '[MISSING]'}")
    print(f" ElevenLabs Voice ID: {voice_id if voice_id else '[DEFAULT]'}")
    print(f" Logs Directory     : {os.path.abspath('logs')}")
    print("=" * 55 + "\n")

    if not groq_key or not eleven_key:
        print("⚠️ Warning: One or more API keys are missing in your .env file.")
        print("Please ensure GROQ_API_KEY and ELEVENLABS_API_KEY are configured.\n")


def main():
    check_environment()
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🚀 Starting server at: http://localhost:{port}")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    print(f"📊 Live Logs endpoint: http://localhost:{port}/api/logs\n")

    uvicorn.run("app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
