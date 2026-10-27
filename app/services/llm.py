import os
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq
from app.services.logger import app_logger
from app.services.chunker import optimize_long_user_query, SentenceStreamBuffer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("GROQ_LLM_MODEL", "qwen/qwen3.8-27b").strip()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

import json
from app.services.medical_tools import MEDICAL_TOOLS_DEFINITIONS, execute_tool

SYSTEM_PROMPT = (
    "Aap ek experienced, empathetic AI Medical Voice Doctor hain. "
    "Aapka dhyan patient ko natural, warm conversational Hinglish (Roman Hindi) me clear aur helpful clinical guidance dena hai.\n"
    "Rules:\n"
    "1. Language: Strictly speak natural conversational Hinglish (Roman Hindi) or clean Hindi. NEVER speak Urdu/Arabic script or English sentences.\n"
    "2. Clinical Response: Mariz ko comforting reassurance dein, fir 1-2 practical home-care tips batayein (jaise aaram, gunguna paani, steam, ya hydration). Anth me hamesha 'Doctor se consult zaroor karein' jodein.\n"
    "3. Length: Strictly 2 short spoken sentences (maximum 25-35 words). Keep it crisp, empathetic, and spoken-friendly.\n"
    "4. Voice Style: Bilkul natural insaan jaisa bolein. Bullet points, stars (*), hashtags (#), ya JSON format bilkul use na karein.\n"
    "5. Domain: Sirf medical, health aur wellness sawaal. Non-medical topics par vinamrata se Hinglish me mana karein."
)

FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
FALLBACK_MODELS = list(dict.fromkeys(FALLBACK_MODELS))

# Fast Zero-Cost Intent Interceptor for common greetings & courtesies
FAST_GREETINGS = ["hello", "hi", "hey", "namaste", "pranam", "kaun ho", "who are you", "kya kar sakte ho", "help", "suno", "sun rahe ho"]
FAST_THANKS = ["thank you", "thanks", "dhanyawad", "shukriya", "bahut accha", "theek hai thank you", "shukriya doctor", "dhanyavaad"]
FAST_STATUS = ["aap kaise ho", "kaise ho", "how are you", "kya haal hai", "sab theek", "kya haal"]
FAST_BYE = ["bye", "alvida", "chalo bye", "goodbye", "good night", "shubh ratri"]

# Specific triggers for attaching heavy tool definitions (saving ~450 tokens on general medical queries)
TOOL_TRIGGERS = [
    "dawa", "dawai", "tablet", "capsule", "syrup", "paracetamol", "dolo", "crocin", "aspirin", 
    "antibiotic", "dosage", "dose", "mg", "side effect", "medicine", "drug", "khurak", "goli",
    "hospital", "aspatal", "ambulance", "emergency", "casualty", "admit", "trauma", "icu", "helpline", "108", "102",
    "specialist", "dermatologist", "cardiologist", "orthopedic", "neurologist", "gynecologist",
    "pediatrician", "ent", "dentist", "psychiatrist", "doctor dikhana", "kisko dikhayein", "visheshagya",
    "first aid", "jal gaya", "burn", "choking", "saanp", "snake", "cut gaya", "khoon beh", "bleeding", "chot lag",
    "guidelines", "outbreak", "cases", "news", "vaccine update", "latest study", "research", "search karo"
]

def should_attach_tools(text: str) -> bool:
    """Conditionally attach tools schema only when query explicitly relates to medicines, hospitals, specialists, or first-aid."""
    t = text.lower()
    return any(trig in t for trig in TOOL_TRIGGERS)

# Fast Zero-Cost Guardrail for Non-Medical Queries
NON_MEDICAL_KEYWORDS = [
    "python", "javascript", "java ", "c++", "html", "css", "sql", "coding", "program", "function",
    "variable", "algorithm", "debug", "github", "react", "fastapi", "linux", "terminal",
    "calculate", "solve equation", "derivative", "integral", "2+2", "square root", "algebra",
    "write an essay", "write a poem", "tell a joke", "sing a song", "movie", "cricket score",
    "capital of", "who won the match", "president of", "prime minister", "politics", "bitcoin",
    "crypto", "stock market", "weather forecast"
]

MEDICAL_INDICATORS = [
    "doctor", "hospital", "medicine", "dawa", "dawain", "tablet", "capsule", "pain", "dard", "fever", "bukhar",
    "cough", "khansi", "sardi", "cold", "flu", "allergy", "infection", "headache", "sar dard", "stomach", "pet",
    "vomit", "ulti", "loose motion", "blood", "khoon", "heart", "dil", "chest", "skin", "rash", "burn", "jal",
    "bite", "wound", "chot", "first aid", "emergency", "ambulance", "health", "bimari", "sehat", "diet", "swasthya",
    "symptom", "treatment", "clinic", "specialist", "test", "bp", "sugar", "diabetes", "cancer", "dengue", "malaria",
    "peeth", "gala", "throat", "eye", "aankh", "ear", "kaan", "bone", "haddi", "dr."
]

def check_fast_conversational_intent(text: str) -> Optional[str]:
    """Zero-token instant conversational intent router for common courtesies."""
    t = text.lower().strip()
    if any(m in t for m in MEDICAL_INDICATORS):
        return None

    words = t.split()
    if len(words) <= 6:
        if any(th in t for th in FAST_THANKS):
            return "Aapka swagat hai! Apna khayal rakhein aur koi bhi dikkat ho toh doctor se zaroor consult karein."
        if any(st in t for st in FAST_STATUS):
            return "Main bilkul theek hoon, poochne ke liye shukriya! Aap bataiye, aapki sehat kaisi hai aur main kya madad karu?"
        if any(by in t for by in FAST_BYE):
            return "Alvida! Apna aur parivar ki sehat ka dhyan rakhein."
        if any(gr in t for gr in FAST_GREETINGS):
            return "Namaste! Main aapka AI Health Assistant hoon. Kripya bataiye, main aapki sehat ke liye kya madad kar sakta hoon?"

    return None

def check_non_medical_guardrail(text: str) -> Optional[str]:
    """Fast, zero-token guardrail to reject obvious non-medical topics in 0ms without consuming LLM API cost."""
    t = text.lower().strip()
    if any(m in t for m in MEDICAL_INDICATORS):
        return None
    if any(non_med in t for non_med in NON_MEDICAL_KEYWORDS):
        return "Maaf kijiye, main sirf medical aur health sambandhi sawaalon ke jawaab de sakta hoon. Kripya swasthya se juda sawaal poochein."
    return None


class SessionMemory:
    """Manages chat history per session ID with a sliding window (2 turns max) for minimal token cost."""

    def __init__(self, max_history: int = 2):
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.interrupted_sessions: set = set()
        self.max_history = max_history

    def get_messages(self, session_id: str, new_user_text: str) -> List[Dict[str, str]]:
        history = list(self.sessions[session_id])
        if session_id in self.interrupted_sessions:
            self.interrupted_sessions.discard(session_id)
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = "[Interrupted by user query - aborted]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": new_user_text})
        return messages

    def add_turn(self, session_id: str, user_text: str, assistant_text: str):
        if session_id in self.interrupted_sessions:
            self.interrupted_sessions.discard(session_id)
            app_logger.info(f"[Barge-In] Discarding response for interrupted session '{session_id[:8]}'")
            return

        self.sessions[session_id].append({"role": "user", "content": user_text})
        self.sessions[session_id].append({"role": "assistant", "content": assistant_text})

        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-(self.max_history * 2) :]

    def interrupt_current_turn(self, session_id: str):
        self.interrupted_sessions.add(session_id)
        if session_id in self.sessions and self.sessions[session_id]:
            last_msg = self.sessions[session_id][-1]
            if last_msg.get("role") == "assistant":
                last_msg["content"] = "[Interrupted by user query - aborted]"
            elif last_msg.get("role") == "user":
                self.sessions[session_id].pop()
        app_logger.info(f"[Barge-In] Turn in session '{session_id[:8]}' marked as interrupted.")

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
        self.interrupted_sessions.discard(session_id)


session_memory = SessionMemory(max_history=2)


def generate_response(
    user_text: str,
    session_id: str = "default",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate conversational LLM response using Groq with query chunking for long speech.
    Returns:
        {
            "response": str,
            "latency_ms": float,
            "ttft_ms": float,
            "tokens_used": int,
            "model": str,
            "chunks_count": int
        }
    """
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing in .env")

    cleaned_text = user_text.strip()[:250]
    if not cleaned_text:
        raise ValueError("User message cannot be empty")

    # Step 1: Fast Zero-Cost Conversational Intent Interceptor (Greetings, Thanks, Status) -> 0 tokens, 1ms
    quick_reply = check_fast_conversational_intent(cleaned_text)
    if quick_reply:
        app_logger.info(f"[Quick Intent] Instant conversational reply with 0 tokens: '{cleaned_text[:50]}'")
        session_memory.add_turn(session_id, cleaned_text, quick_reply)
        return {
            "response": quick_reply,
            "latency_ms": 1.0,
            "ttft_ms": 1.0,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": "quick-intent",
            "chunks_count": 1,
            "tools_called": [],
        }

    # Step 2: Fast Zero-Cost Guardrail: Intercept obvious non-medical requests instantly (0 tokens, 0 cost)
    guardrail_refusal = check_non_medical_guardrail(cleaned_text)
    if guardrail_refusal:
        app_logger.info(f"[Guardrail Intercept] Non-medical query declined with 0 tokens: '{cleaned_text[:50]}'")
        session_memory.add_turn(session_id, cleaned_text, guardrail_refusal)
        return {
            "response": guardrail_refusal,
            "latency_ms": 1.0,
            "ttft_ms": 1.0,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": "medical-guardrail",
            "chunks_count": 1,
            "tools_called": [],
        }

    # Step: Query Chunking optimization for long multi-line speech
    chunk_meta = optimize_long_user_query(cleaned_text)
    prompt_to_send = chunk_meta["optimized_prompt"]

    # If user transcript contains Arabic/Urdu characters, strictly guide LLM to reply in Roman Hinglish
    if any('\u0600' <= ch <= '\u06FF' for ch in prompt_to_send):
        prompt_to_send = f"[User spoke in Hindi/Urdu. Reply strictly in conversational Roman Hinglish]: {prompt_to_send}"

    if chunk_meta["is_chunked"]:
        app_logger.info(
            f"[Query Chunking] Long user input detected ({chunk_meta['chunks_count']} chunks). "
            f"Focused on primary ask: '{chunk_meta['primary_chunk'][:50]}...'"
        )

    messages = session_memory.get_messages(session_id, prompt_to_send)
    models_to_try = [model] if model else FALLBACK_MODELS

    # Dynamic Tool Selection: Only attach 450-token tool definitions if query explicitly triggers tools
    attach_tools = should_attach_tools(prompt_to_send)

    last_error = None
    start_time = time.perf_counter()

    for candidate_model in models_to_try:
        try:
            tools_status = "with clinical tools" if attach_tools else "lean prompt (no tools)"
            app_logger.info(
                f"[Step 2/3: LLM] Requesting answer with '{candidate_model}' ({tools_status}) for session '{session_id[:8]}'"
            )

            extra_params = {}
            if "gpt-oss" in candidate_model.lower():
                extra_params["reasoning_format"] = "hidden"

            call_kwargs = {
                "model": candidate_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 70,
                **extra_params,
            }
            if attach_tools:
                call_kwargs["tools"] = MEDICAL_TOOLS_DEFINITIONS
                call_kwargs["tool_choice"] = "auto"

            initial_res = groq_client.chat.completions.create(**call_kwargs)

            msg = initial_res.choices[0].message
            tools_called = []
            ttft_ms = round((time.perf_counter() - start_time) * 1000, 2)
            followup_usage = None

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception:
                        fn_args = {}

                    app_logger.info(f"[Tool Execution] Running '{fn_name}' with {fn_args}")
                    tool_output = execute_tool(fn_name, fn_args)
                    tools_called.append({
                        "id": tc.id,
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": tool_output
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_output, ensure_ascii=False)
                    })

                # Follow-up completion with tool clinical facts (stream=False for exact token usage)
                followup_res = groq_client.chat.completions.create(
                    model=candidate_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=70,
                    **extra_params,
                )
                answer = (followup_res.choices[0].message.content or "").strip()
                followup_usage = getattr(followup_res, "usage", None)

            else:
                answer = (msg.content or "").strip()

            if not answer:
                app_logger.warning(f"Model '{candidate_model}' returned empty answer, trying next...")
                continue

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Strict post-generation safeguard: NEVER allow Arabic/Urdu script to pass through
            if any('\u0600' <= ch <= '\u06FF' for ch in answer):
                app_logger.warning(f"Urdu/Arabic script detected in answer ('{answer[:30]}...'), replacing with clean Hinglish.")
                answer = "Kripya thoda aaram karein aur paani piyein. Zaroorat lagne par doctor se consult zaroor karein."

            # Record turn in session memory
            session_memory.add_turn(session_id, cleaned_text, answer)

            tools_summary = f" [Tools: {[t['name'] for t in tools_called]}]" if tools_called else ""
            app_logger.info(
                f"[Step 2/3: LLM Complete] in {latency_ms:.1f}ms (TTFT: {ttft_ms}ms) with '{candidate_model}'{tools_summary}: '{answer[:50]}...'"
            )

            usage = getattr(initial_res, "usage", None)
            init_prompt = getattr(usage, "prompt_tokens", 0) or 0
            init_comp = getattr(usage, "completion_tokens", 0) or len(answer.split())

            if followup_usage:
                follow_prompt = getattr(followup_usage, "prompt_tokens", 0) or 0
                follow_comp = getattr(followup_usage, "completion_tokens", 0) or 0
                prompt_tokens = init_prompt + follow_prompt
                completion_tokens = init_comp + follow_comp
                total_tokens = getattr(usage, "total_tokens", 0) + getattr(followup_usage, "total_tokens", 0)
            else:
                prompt_tokens = init_prompt
                completion_tokens = init_comp
                total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

            return {
                "response": answer,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms or latency_ms,
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": candidate_model,
                "chunks_count": chunk_meta["chunks_count"],
                "tools_called": tools_called,
            }

        except Exception as e:
            err_str = str(e)
            last_error = e

            fallback_indicators = [
                "model_not_found",
                "model_decommissioned",
                "decommissioned",
                "does not exist",
                "not have access",
                "deprecated",
                "404",
                "400",
            ]
            if any(ind in err_str.lower() for ind in fallback_indicators):
                app_logger.warning(
                    f"Model '{candidate_model}' unavailable ({err_str[:90]}...). Trying next candidate..."
                )
                continue
            else:
                app_logger.error(f"Groq LLM call failed with model '{candidate_model}': {e}")
                raise RuntimeError(f"LLM generation failed: {e}") from e

    raise RuntimeError(f"LLM generation failed on all models: {last_error}") from last_error
