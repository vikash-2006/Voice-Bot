# 🎙️ AI Medical & Health Voice Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq LPU](https://img.shields.io/badge/Groq-LPU_Accelerated-F05A28?style=flat-square)](https://groq.com)
[![Qwen 27B](https://img.shields.io/badge/Model-Qwen_27B-7C3AED?style=flat-square)](https://groq.com)
[![Whisper Turbo](https://img.shields.io/badge/STT-Whisper_Large_v3_Turbo-00A67E?style=flat-square)](https://groq.com)
[![Deployed on Vercel](https://img.shields.io/badge/Deploy-Vercel-black?style=flat-square&logo=vercel)](https://vercel.com)

A real-time, low-latency, full-duplex AI Medical Voice Assistant built with **Groq LPUs**, **FastAPI**, **Voice Activity Detection (VAD)**, and **Instant Barge-In (Interruption)** support.

Optimized to deliver natural, empathetic doctor-like guidance strictly in **Hinglish (Roman Hindi)** or **Hindi (Devanagari)** with **sub-second voice turnarounds (~900ms)** and **ultra-low token consumption**.

---

## 🌟 Key Features

- **⚡ Real-Time Voice Activity Detection (Dual-Threshold VAD):**
  - Listens only when you speak.
  - Automatically cuts off after **500–800ms of natural silence** without awkward delays.
  - Automatically adapts to ambient room noise floor via hysteresis.

- **⏹️ Instant Full-Duplex Barge-In (Interruption):**
  - If the assistant is speaking and you start speaking or click the Stop button, speech synthesis and playback **abort instantly within 10–20ms** to listen to your new query.

- **🩺 Clinical Medical Doctor Persona & Rich Action Cards:**
  - Provides empathetic reassurance, practical home-care tips (hydration, rest, steam), and clear red-flag warnings with a medical caveat.
  - Generates rich interactive UI cards for **Rx Medicine Dosage**, **Emergency Hospitals**, **Specialist Doctors**, and **First-Aid Guides**.

- **🪙 Ultra-Low Token Architecture & Zero-Token Intents:**
  - **Zero-Token Fast Intent Router:** Greetings (*"Hello"*, *"Namaste"*), gratitude (*"Thank you"*), and status inquiries (*"Aap kaise ho"*) are resolved in **1ms with 0 LLM tokens**.
  - **Dynamic Tool Schema Attachment:** Tools definitions (~450 tokens) are loaded only when medicines, hospitals, or specialists are explicitly asked, cutting prompt tokens by **80%+ on general health queries**.
  - **Live Token Telemetry Pill:** Real-time token breakdown (`Total`, `Prompt In`, `Completion Out`) visible on screen after every turn.

- **🗣️ Hybrid Cross-Platform TTS:**
  - **macOS Local:** Uses native Indian voice (*Rishi* / *Lekha*) with **0 API cost** and **~150ms latency**.
  - **Linux / Cloud / Vercel Serverless:** Automatically falls back to free **Microsoft Neural TTS (Edge-TTS)** (*hi-IN-MadhurNeural* / *en-IN-PrabhatNeural*) or **ElevenLabs**.

- **🔒 Production Safety & Guardrails:**
  - Zero-token rejection for non-medical requests (coding, math, politics, trivia).
  - Urdu script interceptor & post-generation filter preventing Arabic/Urdu alphabet leakage.
  - Safe, non-corrupting WebM/WAV container streaming.

---

## 🏗️ Architecture & Pipeline

```text
[User Speaks into Mic]
         │ (Web Audio API + Dual-Threshold VAD)
         ▼
[500-800ms Silence Detected] ──► Captures clean audio slice
         │
         ▼
[Step 1: Groq Whisper STT] (~250-450ms)
         │ (Biased prompt: Hinglish/Hindi transcription)
         ▼
[Step 2: Intent / Guardrail Router]
   ├── [Greeting / Thanks] ──► Instant Hinglish reply (0 tokens, 1ms)
   ├── [Non-Medical Query] ──► Polite Hinglish refusal (0 tokens, 1ms)
   └── [Medical Health Ask] ──► Groq Qwen 27B / GPT-OSS
         │ (Dynamic tools schema: lean prompt ~150 tokens)
         ▼
[Step 3: TTS Engine] (~150-400ms)
   ├── [macOS] ─────────────► Native Rishi (en-IN) / Lekha (hi-IN)
   ├── [Vercel / Linux] ────► Edge-TTS Neural Voice / ElevenLabs
   └── [Client Barge-In] ───► Instant Abort Signal (10ms)
         │
         ▼
[Natural Spoken Voice Response + Visual Action Card in UI]
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/voice-bot.git
cd voice-bot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_LLM_MODEL=qwen/qwen3.8-27b
GROQ_WHISPER_MODEL=whisper-large-v3-turbo
STT_PROVIDER=groq

# Optional: ElevenLabs (falls back to free Edge-TTS / macOS Rishi if omitted)
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
```

### 3. Run the Server

```bash
python3 main.py
```

Or using Uvicorn directly:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000** in your browser, allow microphone access, and speak!

---

## 🌐 Deploy to Vercel (Production)

Deploying to Vercel takes less than 3 minutes:

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Deploy AI Voice Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voice-bot.git
git push -u origin main
```

### 2. Import Project in Vercel
1. Go to [Vercel Dashboard](https://vercel.com/new).
2. Choose **"Import Git Repository"** and select `voice-bot`.
3. Under **Environment Variables**, add:
   - `GROQ_API_KEY` = `your_groq_api_key`
   - `GROQ_LLM_MODEL` = `qwen/qwen3.8-27b`
   - `GROQ_WHISPER_MODEL` = `whisper-large-v3-turbo`
   - `STT_PROVIDER` = `groq`
4. Click **Deploy**!

> **Note on Vercel Serverless:**
> `vercel.json` and `api/index.py` are pre-configured. On Vercel, the app automatically runs on serverless Python and uses Microsoft Neural Edge-TTS with `/tmp/logs` scratch storage.

---

## 📊 Token Usage & Cost Efficiency

| Query Type | Tokens Used | Latency | Estimated Cost (Groq) |
| :--- | :--- | :--- | :--- |
| **Greetings ("Hello", "Namaste")** | **0 tokens** | **~1ms** | **$0.00** |
| **Courtesies ("Thank you", "Bye")** | **0 tokens** | **~1ms** | **$0.00** |
| **Symptom Triage ("Sir dard ho raha hai")** | **~150–180 tokens** | **~220ms** | **~$0.0001 (~0.008 paise)** |
| **Medicine / Rx Lookup ("Dolo 650 dosage")** | **~750 tokens (with tools)** | **~550ms** | **~$0.0005 (~0.04 paise)** |

- **Average Cost per Voice Turn:** **~0.03 to 0.05 paise** (₹0.0005).
- **1,000 conversational queries cost less than ₹4 to ₹5!**

---

## 📂 Project Structure

```text
voice bot/
├── api/
│   └── index.py             # Vercel serverless entrypoint
├── app/
│   ├── main.py              # FastAPI application & audio endpoints
│   ├── static/
│   │   └── index.html       # Audio visualizer, VAD engine & telemetry UI
│   └── services/
│       ├── stt.py           # Groq Whisper transcription engine
│       ├── llm.py           # Medical reasoning, zero-token intent router & guardrails
│       ├── tts.py           # Cross-platform TTS (macOS Native, Edge-TTS, ElevenLabs)
│       ├── medical_tools.py # Clinical tools (Rx guide, triage, hospital locator)
│       ├── chunker.py       # Query chunker for long multi-line speech
│       └── logger.py        # Structured JSONL session interaction logger
├── logs/                    # Local session logs & telemetry
├── main.py                  # Local CLI entrypoint
├── vercel.json              # Vercel deployment & rewrite configuration
├── steps.txt                # Plain-text step-by-step GitHub & Vercel guide
├── requirements.txt         # Production dependencies
└── README.md                # Documentation
```

---

## 📡 API Endpoints

- `GET /` — Serves the interactive Web Voice UI.
- `POST /api/voice-chat` — Full-duplex voice pipeline (Receives audio binary, returns spoken audio + telemetry headers).
- `GET /api/logs` — Fetches real-time structured telemetry & latency history.
- `GET /health` — Health check & configured providers verification.

---

## 📄 License
MIT License. Open-source for developers and healthcare builders.
