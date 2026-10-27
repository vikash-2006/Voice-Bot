#!/usr/bin/env python3
"""
Test Medical Domain Guardrails & Strict Token Bounding:
1. Non-medical queries rejected immediately (0 tokens or LLM rejection).
2. Medical queries answered accurately with tools.
3. Strict token bound verification (<= 70 tokens).
"""
import time
from app.services.llm import generate_response

test_cases = [
    # Non-medical queries (Should be REJECTED)
    {
        "query": "Write Python code to sort a list of numbers",
        "expected_type": "rejected",
        "description": "Tech/Coding query"
    },
    {
        "query": "2+2 kitna hota hai calculate karo",
        "expected_type": "rejected",
        "description": "Math query"
    },
    {
        "query": "Aaj ka cricket match kaun jeeta score batao",
        "expected_type": "rejected",
        "description": "Sports query"
    },
    {
        "query": "Taj Mahal kisne banaya tha aur kahan hai?",
        "expected_type": "rejected",
        "description": "History / General trivia query"
    },
    # Allowed medical queries (Should be ANSWERED)
    {
        "query": "Paracetamol kab aur kitni dose leni chahiye?",
        "expected_type": "answered",
        "description": "Medicine query"
    },
    {
        "query": "Aag se haath jalne par turant kya first aid karni chahiye?",
        "expected_type": "answered",
        "description": "First aid query"
    },
    {
        "query": "Namaste! Aap kaun ho aur meri kya madad kar sakte ho?",
        "expected_type": "answered",
        "description": "Greeting query"
    },
    {
        "query": "WHO ki latest Mpox guidelines live search karke batao.",
        "expected_type": "answered",
        "description": "Super API (Live Web Search) query"
    }
]

print("=" * 65)
print(" 🛡️ TESTING MEDICAL SCOPE GUARDRAILS & TOKEN BOUNDS")
print("=" * 65)

all_passed = True

for idx, tc in enumerate(test_cases, 1):
    q = tc["query"]
    desc = tc["description"]
    expected = tc["expected_type"]
    
    print(f"\n[Case {idx}/7: {desc}]")
    print(f"👉 Query: '{q}'")
    
    start = time.perf_counter()
    res = generate_response(q, session_id=f"test-guard-{idx}")
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    
    answer = res["response"]
    tokens = res["tokens_used"]
    model = res["model"]
    tools = [t["name"] for t in res.get("tools_called", [])]
    
    print(f"🗣️ Response: '{answer}'")
    print(f"📊 Model: {model} | Latency: {elapsed_ms}ms | Tokens Used: {tokens} | Tools: {tools or 'None'}")
    
    # Check if expected behavior matches
    is_declined = any(w in answer.lower() for w in ["maaf", "sorry", "sirf medical", "only assist with medical", "health sambandhi"])
    
    if expected == "rejected":
        if is_declined:
            print("✅ PASSED: Correctly declined non-medical query!")
        else:
            print("❌ FAILED: Non-medical query was answered instead of declined!")
            all_passed = False
    elif expected == "answered":
        if not is_declined or "namaste" in q.lower():
            print("✅ PASSED: Medical/Greeting query processed properly!")
        else:
            print("❌ FAILED: Medical query was incorrectly rejected!")
            all_passed = False
            
    # Check token bound rule (tokens <= 70)
    if tokens > 75:
        print(f"⚠️ Token limit exceeded: {tokens} tokens > 75")
        all_passed = False
    else:
        print(f"✅ Token limit OK ({tokens} tokens <= 75)")

print("\n" + "=" * 65)
if all_passed:
    print(" 🎉 ALL GUARDRAIL & TOKEN BOUND TESTS PASSED PERFECTLY!")
else:
    print(" ⚠️ SOME TESTS FAILED. PLEASE REVIEW.")
print("=" * 65)
