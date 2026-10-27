"""
Medical & Real-Time Search Tools Suite for AI Voice Assistant.
Implements 6 specialized medical, emergency, and live web search tools:
1. analyze_disease_symptoms: Triage and evaluate disease symptoms
2. lookup_medicine_info: Drug uses, standard dosage guidelines, and precautions
3. find_hospital_emergency: 24/7 emergency departments, trauma centers & helplines
4. find_clinic_specialist: Specialist department recommendation & doctor questions
5. get_first_aid_guide: Step-by-step first-aid actions for critical scenarios
6. live_web_search: Real-time search via Serper (Google Search) or DuckDuckGo fallback
"""
import os
import re
import json
import urllib.parse
from typing import Dict, Any, List, Optional
import httpx
from dotenv import load_dotenv
from app.services.logger import app_logger

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()

# ==============================================================================
# 1. 🩺 DISEASE & SYMPTOM ANALYZER
# ==============================================================================
COMMON_CONDITIONS_DB = {
    "fever": {
        "conditions": ["Viral Fever", "Flu / Influenza", "Dengue / Malaria", "Infection"],
        "red_flags": ["Temperature over 103°F (39.4°C)", "Difficulty breathing", "Stiff neck or confusion", "Persistent vomiting"],
        "home_care": "Adequate hydration (water, ORS, soups), physical rest, cool damp cloth on forehead, light easily digestible food.",
        "urgency": "Consult a physician if fever lasts > 3 days or exceeds 102°F."
    },
    "cough": {
        "conditions": ["Upper Respiratory Infection", "Bronchitis", "Allergic Cough", "Asthma"],
        "red_flags": ["Coughing blood", "Severe shortness of breath", "Chest pain on inhalation", "High fever with chills"],
        "home_care": "Warm water gargles, steam inhalation twice a day, honey with ginger, avoid cold drinks and dust.",
        "urgency": "Consult doctor if cough lasts > 2 weeks or brings up colored phlegm."
    },
    "headache": {
        "conditions": ["Tension Headache", "Migraine", "Sinusitis", "Dehydration / Eye strain"],
        "red_flags": ["Sudden explosive 'thunderclap' headache", "Weakness on one side of face/body", "Vision changes", "Fever with neck stiffness"],
        "home_care": "Rest in a quiet dark room, drink 2-3 glasses of water, gentle neck massage, avoid screen exposure.",
        "urgency": "Seek immediate emergency care if headache is sudden and unusually severe."
    },
    "stomach": {
        "conditions": ["Gastritis / Acidity", "Food Poisoning", "Gastroenteritis", "Indigestion"],
        "red_flags": ["Severe sharp abdominal pain", "Blood in vomit or stool", "Inability to keep liquids down for 24h", "High fever"],
        "home_care": "ORS or coconut water, bland diet (khichdi, curd, toast), avoid oily, spicy, and dairy products.",
        "urgency": "Consult a gastroenterologist or emergency if pain is sharp and localized."
    },
    "skin": {
        "conditions": ["Contact Dermatitis", "Urticaria / Allergy", "Fungal Infection", "Eczema"],
        "red_flags": ["Swelling of face, lips, or throat (Anaphylaxis)", "Rapidly spreading rash with fever", "Blisters over large body areas"],
        "home_care": "Keep skin clean and dry, use mild calamine lotion, wear loose cotton clothes, avoid scratching.",
        "urgency": "Visit dermatologist if rash persists or causes intense itching."
    }
}

def analyze_disease_symptoms(
    symptoms: str,
    duration_days: Optional[int] = 1,
    severity: Optional[str] = "moderate",
) -> Dict[str, Any]:
    """Analyzes reported disease symptoms, triage urgency, potential causes, and red flags."""
    sym_lower = symptoms.lower()
    matched_conditions = []
    red_flags = []
    home_care = []

    for key, data in COMMON_CONDITIONS_DB.items():
        if key in sym_lower or any(word in sym_lower for word in key.split()):
            matched_conditions.extend(data["conditions"])
            red_flags.extend(data["red_flags"])
            home_care.append(data["home_care"])

    if not matched_conditions:
        matched_conditions = ["Viral or seasonal syndrome", "Mild systemic inflammation", "Fatigue / stress related"]
        red_flags = ["High persistent fever", "Difficulty breathing", "Extreme weakness or fainting"]

    urgency_level = "Urgent / Emergency" if (severity and severity.lower() == "severe") or (duration_days and duration_days > 5) else "Moderate"

    return {
        "tool": "analyze_disease_symptoms",
        "conditions": list(dict.fromkeys(matched_conditions))[:2],
        "red_flags": list(dict.fromkeys(red_flags))[:2],
        "home_care": home_care[0] if home_care else "Rest, hydration, and monitoring.",
        "urgency": urgency_level,
    }


# ==============================================================================
# 2. 💊 MEDICINE & DRUG INFORMATION GUIDE
# ==============================================================================
MEDICINES_DB = {
    "paracetamol": {
        "brand_names": ["Dolo 650", "Crocin", "Calpol", "Tylenol"],
        "uses": "Relief of mild-to-moderate pain, headache, body ache, and reduction of fever.",
        "dosage_guide": "Adults: 500mg - 650mg every 4 to 6 hours as needed. Maximum 3000mg to 4000mg in 24 hours. Always take after light food.",
        "side_effects": "Rare at recommended doses; liver toxicity if overdosed or combined with alcohol.",
        "precautions": "Avoid taking multiple cold medicines that contain paracetamol. Do not consume alcohol while taking."
    },
    "ibuprofen": {
        "brand_names": ["Brufen", "Combiflam (with Paracetamol)", "Advil"],
        "uses": "Anti-inflammatory painkiller for muscle pain, toothache, joint pain, and inflammation.",
        "dosage_guide": "Adults: 200mg - 400mg every 6 to 8 hours with milk or food. Do not take on an empty stomach.",
        "side_effects": "Stomach irritation, acidity, nausea, heartburn.",
        "precautions": "Contraindicated for patients with stomach ulcers, kidney disease, or asthma. Avoid prolonged daily use."
    },
    "cetirizine": {
        "brand_names": ["Cetzine", "Zyrtec", "Alerid", "Okacet"],
        "uses": "Antihistamine for allergic rhinitis, sneezing, runny nose, watery eyes, and skin hives.",
        "dosage_guide": "Adults: 5mg to 10mg once daily, preferably at bedtime.",
        "side_effects": "Mild drowsiness, dry mouth, tiredness.",
        "precautions": "Avoid driving or operating machinery if drowsy. Do not consume with alcohol."
    },
    "azithromycin": {
        "brand_names": ["Azithral", "Azee", "Zithromax"],
        "uses": "Broad-spectrum macrolide antibiotic for throat infections, chest infections, and bacterial sinus issues.",
        "dosage_guide": "Typically 500mg once daily for 3 to 5 days, taken 1 hour before or 2 hours after meals.",
        "side_effects": "Nausea, loose stools, abdominal cramping.",
        "precautions": "Schedule H prescription drug. ALWAYS complete the full antibiotic course prescribed by doctor to avoid antibiotic resistance."
    },
    "ors": {
        "brand_names": ["Electral", "Prolyte", "WHO ORS"],
        "uses": "Oral Rehydration Salts for restoration of electrolyte balance in diarrhea, vomiting, and heat dehydration.",
        "dosage_guide": "Dissolve 1 full sachet in exactly 1 Liter of clean drinking water. Drink small sips regularly.",
        "side_effects": "None when mixed in correct water proportion.",
        "precautions": "Do not mix with milk, juice, or boil the solution after mixing. Discard after 24 hours."
    }
}

def lookup_medicine_info(
    medicine_name: str,
    query_type: Optional[str] = "uses",
) -> Dict[str, Any]:
    """Retrieves validated drug details, standard dosage guidelines, contraindications, and precautions."""
    med_key = medicine_name.lower().strip()
    matched_med = None

    for key, data in MEDICINES_DB.items():
        if key in med_key or any(brand.lower() in med_key for brand in data["brand_names"]):
            matched_med = (key, data)
            break

    if not matched_med:
        return {
            "tool": "lookup_medicine_info",
            "medicine": medicine_name,
            "status": "not_in_offline_cache",
            "guideline": "Always consult a doctor or pharmacist for proper dosage and prescription.",
        }

    key_name, info = matched_med
    return {
        "tool": "lookup_medicine_info",
        "medicine": key_name.capitalize(),
        "brands": info["brand_names"][:3],
        "uses": info["uses"],
        "dosage": info["dosage_guide"],
        "precautions": info["precautions"],
    }


# ==============================================================================
# 3. 🏥 HOSPITAL & EMERGENCY SERVICES LOCATOR
# ==============================================================================
EMERGENCY_NUMBERS = {
    "India": {
        "National Emergency": "112",
        "Ambulance": "108 / 102",
        "Medical Helpline": "104",
        "Poison Control": "1800-116-117"
    },
    "International": {
        "USA / Canada": "911",
        "UK / Europe": "999 / 112",
        "UAE": "998"
    }
}

def find_hospital_emergency(
    location: str,
    emergency_type: Optional[str] = "general",
) -> Dict[str, Any]:
    """Provides emergency department guidance, triage facilities, and national emergency contacts."""
    return {
        "tool": "find_hospital_emergency",
        "facility": f"24/7 Trauma / Casualty Hospital near {location}",
        "contacts": {"Ambulance": "108 / 102", "Emergency": "112", "Helpline": "104"},
        "guidance": "If severe pain, bleeding, or breathing difficulty, call 108 or 112 immediately."
    }


# ==============================================================================
# 4. 👨‍⚕️ CLINIC & SPECIALIST FINDER
# ==============================================================================
SPECIALIST_MAPPING = {
    "heart": ("Cardiologist", "Chest pain, palpitations, breathlessness on exertion, blood pressure issues"),
    "skin": ("Dermatologist", "Acne, rashes, hair loss, fungal infections, persistent itching"),
    "bone": ("Orthopedic Surgeon", "Joint pain, fractures, backache, ligament tear, arthritis"),
    "child": ("Pediatrician", "Infant fever, vaccination, growth monitoring, childhood allergies"),
    "stomach": ("Gastroenterologist", "Acid reflux, liver issues, chronic constipation, severe bloating"),
    "brain": ("Neurologist", "Migraines, seizures, tremors, numbness, memory impairment"),
    "eye": ("Ophthalmologist", "Blurry vision, eye redness, irritation, vision screening"),
    "ear": ("ENT Specialist (Otolaryngologist)", "Ear discharge, hearing difficulty, throat pain, sinusitis"),
    "mental": ("Psychiatrist / Clinical Psychologist", "Depression, panic attacks, severe anxiety, sleep disorders")
}

def find_clinic_specialist(
    condition_or_specialty: str,
    location: Optional[str] = "",
) -> Dict[str, Any]:
    """Recommends appropriate medical specialist department and consultation preparation checklist."""
    query_lower = condition_or_specialty.lower()
    recommended_specialist = "General Physician (Internal Medicine)"
    reason = "Initial clinical evaluation and baseline blood investigations"

    for keyword, (specialist, symptoms_covered) in SPECIALIST_MAPPING.items():
        if keyword in query_lower or specialist.lower() in query_lower:
            recommended_specialist = specialist
            reason = f"Specialized in: {symptoms_covered}"
            break

    return {
        "tool": "find_clinic_specialist",
        "recommended_specialist": recommended_specialist,
        "clinical_focus": reason,
    }


# ==============================================================================
# 5. 🩹 FIRST-AID EMERGENCY PROCEDURES
# ==============================================================================
FIRST_AID_PROTOCOLS = {
    "burns": {
        "title": "Minor to Moderate Thermal Burns",
        "immediate_steps": [
            "Cool the burn under gentle running tap water for 15-20 minutes. DO NOT use ice or ice water.",
            "Remove tight rings or clothing around the burn before swelling occurs.",
            "Apply pure aloe vera gel or silver sulfadiazine cream. DO NOT apply toothpaste, butter, or oil.",
            "Cover loosely with sterile non-stick bandage. Never burst blisters."
        ],
        "seek_er": "If burn is larger than 3 inches, on face/hands/groin, or skin appears charred/white."
    },
    "choking": {
        "title": "Adult / Child Choking (Heimlich Maneuver)",
        "immediate_steps": [
            "Encourage them to cough if they are still able to make sounds.",
            "If unable to breathe or speak: Stand behind person, lean them slightly forward.",
            "Deliver 5 firm back blows between shoulder blades using the heel of your hand.",
            "If back blows fail, perform 5 abdominal thrusts (Heimlich): Clench fist above navel, pull inward and upward sharply.",
            "Repeat 5 back blows and 5 thrusts until airway clears or EMS arrives."
        ],
        "seek_er": "Call 112 / 108 immediately if obstruction does not clear within 1 minute or person loses consciousness."
    },
    "snakebite": {
        "title": "Suspected Venomous Snake Bite",
        "immediate_steps": [
            "Keep the patient calm and completely still. Movement causes venom to spread faster.",
            "Immobilize the bitten limb at or below heart level using a splint or sling.",
            "Gently clean wound with clean water. DO NOT cut the wound, DO NOT suck venom, DO NOT apply tourniquet.",
            "Remove rings, watches, or tight clothes from bitten limb."
        ],
        "seek_er": "Rush immediately to the nearest hospital equipped with Anti-Snake Venom (ASV). Every minute counts."
    },
    "bleeding": {
        "title": "Severe External Bleeding",
        "immediate_steps": [
            "Apply firm, continuous direct pressure over wound using a sterile gauze or clean cloth.",
            "Elevate the injured limb above heart level if no fracture is suspected.",
            "Do not remove soaked cloth; add more layers on top to allow clotting.",
            "If bleeding does not stop with direct pressure, apply pressure dressing firmly."
        ],
        "seek_er": "Seek emergency care immediately if blood is spurting, will not stop after 10 mins of pressure, or person feels faint."
    },
    "heatstroke": {
        "title": "Heat Stroke & Severe Exhaustion",
        "immediate_steps": [
            "Move person into shade or air-conditioned room immediately.",
            "Loosen excess clothing. Sponge body with cool water or fan vigorously.",
            "Place ice packs on armpits, neck, and groin.",
            "Give cool water or ORS ONLY if person is fully conscious and alert."
        ],
        "seek_er": "Call emergency if body temperature is above 104°F (40°C), person is confused, or loses consciousness."
    }
}

def get_first_aid_guide(
    emergency_scenario: str,
) -> Dict[str, Any]:
    """Retrieves step-by-step immediate first-aid protocols for acute medical emergencies."""
    scen_lower = emergency_scenario.lower()
    matched = None

    for key, protocol in FIRST_AID_PROTOCOLS.items():
        if key in scen_lower or any(word in scen_lower for word in key.split()):
            matched = (key, protocol)
            break

    if not matched:
        return {
            "tool": "get_first_aid_guide",
            "scenario": emergency_scenario,
            "immediate_steps": [
                "Ensure scene safety for both patient and responder.",
                "Check responsiveness (tap shoulders and shout 'Are you okay?').",
                "Call 112 or local ambulance immediately.",
                "Keep patient comfortable, monitor breathing, and do not administer oral fluids if drowsy."
            ],
            "warning": "Always call for professional medical support for severe trauma or unconsciousness."
        }

    key_name, data = matched
    return {
        "tool": "get_first_aid_guide",
        "scenario": data["title"],
        "steps": data["immediate_steps"][:2],
        "seek_er": data["seek_er"]
    }


# ==============================================================================
# 6. 🌐 LIVE WEB SEARCH ("SUPER API" / REAL-TIME SEARCH)
# ==============================================================================
def live_web_search(
    query: str,
    max_results: Optional[int] = 3,
) -> Dict[str, Any]:
    """
    Real-time live web search using Serper (Google Search API) if configured,
    with an automatic zero-config fallback to DuckDuckGo live search.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"tool": "live_web_search", "query": "", "results": []}

    app_logger.info(f"[Live Web Search] Query: '{cleaned_query}'")

    # 1. Primary: Serper.dev Google Search API (if key available)
    if SERPER_API_KEY:
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {"q": cleaned_query, "num": max_results}
            with httpx.Client(timeout=4.0) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    results = []
                    # Check knowledge graph or organic results
                    if "knowledgeGraph" in data:
                        kg = data["knowledgeGraph"]
                        results.append({
                            "title": kg.get("title", ""),
                            "snippet": kg.get("description", ""),
                            "link": kg.get("website", "")
                        })
                    for item in data.get("organic", [])[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", "")
                        })
                    if results:
                        return {
                            "tool": "live_web_search",
                            "provider": "serper_google",
                            "query": cleaned_query,
                            "results": results
                        }
        except Exception as e:
            app_logger.warning(f"Serper API call failed ({e}). Falling back to DuckDuckGo...")

    # 2. Fast Zero-Config Real-Time Search: DuckDuckGo Live Web Search
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        with httpx.Client(timeout=4.5) as client:
            res = client.post("https://html.duckduckgo.com/html/", data={"q": cleaned_query}, headers=headers)
            if res.status_code == 200:
                raw_snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL)
                raw_titles = re.findall(r'<h2 class="result__title">.*?>(.*?)</a>', res.text, re.DOTALL)
                results = []
                for idx, s in enumerate(raw_snippets[:max_results]):
                    clean_snippet = re.sub(r'<[^<]+?>', '', s).strip()
                    title = "Medical Search Result"
                    if idx < len(raw_titles):
                        title = re.sub(r'<[^<]+?>', '', raw_titles[idx]).strip()
                    if clean_snippet:
                        results.append({
                            "title": title[:60],
                            "snippet": clean_snippet[:220],
                            "link": ""
                        })
                if results:
                    return {
                        "tool": "live_web_search",
                        "provider": "duckduckgo_live",
                        "query": cleaned_query,
                        "results": results[:max_results]
                    }
    except Exception as e:
        app_logger.warning(f"DuckDuckGo live search failed ({e}). Falling back to Wikipedia...")

    # 3. Third-layer Fallback: Wikipedia Summary API
    try:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(cleaned_query)}"
        with httpx.Client(timeout=3.0) as client:
            res = client.get(wiki_url, headers={"User-Agent": "VoiceAssistant/1.0"})
            if res.status_code == 200:
                data = res.json()
                if "extract" in data:
                    return {
                        "tool": "live_web_search",
                        "provider": "wikipedia",
                        "query": cleaned_query,
                        "results": [{
                            "title": data.get("title", cleaned_query),
                            "snippet": data.get("extract", "")[:220],
                            "link": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                        }]
                    }
    except Exception as e:
        app_logger.warning(f"Wikipedia fallback failed: {e}")

    return {
        "tool": "live_web_search",
        "provider": "offline_summary",
        "query": cleaned_query,
        "results": [{
            "title": "General Knowledge Search",
            "snippet": f"Current real-time data for '{cleaned_query}'. Please verify local medical or official news portals for real-time updates.",
            "link": ""
        }]
    }


# ==============================================================================
# TOOL REGISTRY & SCHEMAS FOR GROQ / MCP
# ==============================================================================
MEDICAL_TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_disease_symptoms",
            "description": "Analyze patient symptoms for potential conditions and red flags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {"type": "string", "description": "Patient symptoms"}
                },
                "required": ["symptoms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_medicine_info",
            "description": "Look up medicine uses, standard dosage, and precautions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medicine_name": {"type": "string", "description": "Drug or brand name"}
                },
                "required": ["medicine_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_hospital_emergency",
            "description": "Find 24/7 casualty hospital guidance and helpline contacts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or locality"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_clinic_specialist",
            "description": "Recommend doctor specialist for a health condition or body issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_or_specialty": {"type": "string", "description": "Health issue or body part"}
                },
                "required": ["condition_or_specialty"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_first_aid_guide",
            "description": "First-aid steps for burns, choking, bleeding, snakebite, heatstroke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emergency_scenario": {"type": "string", "description": "Emergency scenario"}
                },
                "required": ["emergency_scenario"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "live_web_search",
            "description": "Live search for current medical advisories or healthcare information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical search query"}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTION_MAP = {
    "analyze_disease_symptoms": analyze_disease_symptoms,
    "lookup_medicine_info": lookup_medicine_info,
    "find_hospital_emergency": find_hospital_emergency,
    "find_clinic_specialist": find_clinic_specialist,
    "get_first_aid_guide": get_first_aid_guide,
    "live_web_search": live_web_search,
}

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a tool by name with arguments and returns structured output."""
    func = TOOL_FUNCTION_MAP.get(tool_name)
    if not func:
        return {"error": f"Tool '{tool_name}' not found."}
    try:
        app_logger.info(f"[Tool Execution] Running '{tool_name}' with args: {arguments}")
        result = func(**arguments)
        return result
    except Exception as e:
        app_logger.error(f"[Tool Execution Failed] '{tool_name}': {e}", exc_info=True)
        return {"error": f"Tool execution failed: {str(e)}"}
