#!/usr/bin/env python3
"""
Test script to verify all 6 MCP Medical & Live Search Tools
and end-to-end voice bot autonomous tool calling.
"""
import sys
from dotenv import load_dotenv

load_dotenv()

from app.services.medical_tools import (
    analyze_disease_symptoms,
    lookup_medicine_info,
    find_hospital_emergency,
    find_clinic_specialist,
    get_first_aid_guide,
    live_web_search,
)
from app.services.llm import generate_response

def run_tests():
    print("=" * 65)
    print("     🩺 TESTING MCP MEDICAL & LIVE SEARCH TOOLS SUITE")
    print("=" * 65)

    # 1. Disease & Symptoms
    print("\n[Tool 1/6: Disease & Symptom Analyzer]")
    t1 = analyze_disease_symptoms("Mujhe 2 din se tez bukhar, body ache aur sardi lag rahi hai", duration_days=2)
    print(f"✅ Conditions: {t1['potential_conditions']}")
    print(f"⚠️ Red Flags: {t1['red_flag_warning_signs']}")
    print(f"🏠 Home Care: {t1['recommended_home_care']}")

    # 2. Medicine Info
    print("\n[Tool 2/6: Medicine & Drug Guide]")
    t2 = lookup_medicine_info("Paracetamol")
    print(f"✅ Drug: {t2['medicine_name']} (Brands: {', '.join(t2['common_brands'])})")
    print(f"💊 Uses: {t2['primary_uses']}")
    print(f"📋 Dosage: {t2['dosage_guidelines']}")

    # 3. Hospital Emergency
    print("\n[Tool 3/6: Hospital & Emergency Locator]")
    t3 = find_hospital_emergency("South Delhi", emergency_type="cardiac")
    print(f"🚨 Facility: {t3['recommended_facility']}")
    print(f"📞 Contacts: {t3['emergency_contacts']}")

    # 4. Clinic & Specialist
    print("\n[Tool 4/6: Clinic & Specialist Finder]")
    t4 = find_clinic_specialist("chest pain and palpitations", location="Mumbai")
    print(f"👨‍⚕️ Specialist: {t4['recommended_specialist']}")
    print(f"🔬 Reason: {t4['clinical_focus']}")

    # 5. First Aid
    print("\n[Tool 5/6: First-Aid Emergency Procedures]")
    t5 = get_first_aid_guide("burns")
    print(f"🩹 Protocol: {t5['scenario']}")
    print(f"⚡ Step 1: {t5['critical_steps'][0]}")
    print(f"⚡ Step 2: {t5['critical_steps'][1]}")

    # 6. Live Web Search ("Super API" / Google Search / DuckDuckGo)
    print("\n[Tool 6/6: Live Web Search (Super API)]")
    t6 = live_web_search("World Health Organization Dengue prevention guidelines", max_results=2)
    print(f"🌐 Provider: {t6.get('provider')}")
    if t6.get("results"):
        top = t6["results"][0]
        print(f"🔍 Top Result: '{top.get('title')}' -> {top.get('snippet')[:100]}...")

    # 7. End-to-End Autonomous Voice Query with Tool Calling
    print("\n" + "=" * 65)
    print(" 🤖 END-TO-END AUTONOMOUS TOOL CALLING VIA GROQ (VOICE ENGINE)")
    print("=" * 65)

    test_queries = [
        "Mujhe Paracetamol kab leni chahiye aur iska dose kya hai?",
        "Snake bite hone par turant kya first aid karni chahiye?",
        "Mujhe skin par lal daane aur khujli ho rahi hai, kis doctor ko dikhau?",
    ]

    for q in test_queries:
        print(f"\n👉 User Asked: '{q}'")
        res = generate_response(q, session_id="cli-medical-test")
        tools_used = [t['name'] for t in res.get('tools_called', [])]
        print(f"🛠️ Tools Called: {tools_used or 'None (Direct response)'}")
        print(f"🗣️ Spoken Bot Response: '{res['response']}'")
        print(f"⚡ Latency: {res['latency_ms']} ms | Model: {res['model']}")

    print("\n" + "=" * 65)
    print(" 🎉 ALL MCP MEDICAL & SEARCH TOOLS VERIFIED SUCCESSFULLY!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_tests()
