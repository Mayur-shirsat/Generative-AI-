import random
import streamlit as st
import requests
import json
import io
import re
import uuid
import csv
import os
import hashlib
from typing import Optional, List, Dict
from datetime import datetime
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================
# CONFIG
# ==============================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "qwen/qwen3-coder:free"  # Model name
API_TIMEOUT = 6000  # Increased timeout to 6000 seconds

# Country codes for dropdown
COUNTRY_CODES = [
    "+91 (India)", "+1 (USA)", "+44 (UK)", "+61 (Australia)", "+81 (Japan)",
    "+86 (China)", "+49 (Germany)", "+33 (France)", "+55 (Brazil)", "+27 (South Africa)"
]

# Supported languages
LANGUAGES = ["English", "Spanish", "French", "German", "Hindi", "Chinese"]
LANG_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese": "zh"
}

# Initialize API key in session state
if "openrouter_api_key" not in st.session_state:
    # Try loading from .env file
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        st.session_state.openrouter_api_key = env_key
    # Try Streamlit secrets as fallback
    elif hasattr(st.secrets, "OPENROUTER_API_KEY"):
        st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        st.session_state.openrouter_api_key = ""

# ==============================
# Utilities - resume parsing
# ==============================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_pages = []
        for p in reader.pages:
            try:
                text_pages.append(p.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(text_pages)
        return text.strip()
    except Exception as e:
        st.warning(f"PDF extraction issue: {e}")
        return ""

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        st.warning(f"DOCX extraction issue: {e}")
        return ""

def extract_text_from_txt_bytes(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        return text.strip()
    except Exception:
        try:
            return file_bytes.decode("latin-1", errors="ignore")
        except Exception:
            return ""

def parse_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    bytes_data = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(bytes_data)
    elif name.endswith(".docx"):
        return extract_text_from_docx_bytes(bytes_data)
    elif name.endswith(".doc"):
        return extract_text_from_txt_bytes(bytes_data)
    else:
        return extract_text_from_txt_bytes(bytes_data)

# ==============================
# LLM prompt & API helper
# ==============================
def get_api_key() -> str:
    """Get or prompt for OpenRouter API key using session state."""
    if not st.session_state.openrouter_api_key:
        st.error("OpenRouter API key is not set. Please provide it below or set the 'OPENROUTER_API_KEY' in your .env file.")
        api_key_input = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_input")
        if api_key_input:
            st.session_state.openrouter_api_key = api_key_input
            st.success("API key set successfully!")
            st.rerun()
    return st.session_state.openrouter_api_key

def check_openrouter_key() -> bool:
    """Check if the OpenRouter API key is set."""
    api_key = get_api_key()
    return bool(api_key)

def translate_text(text: str, target_lang: str, source_lang: str = "English") -> str:
    target_code = LANG_CODES.get(target_lang, "en")
    source_code = LANG_CODES.get(source_lang, "en")
    if target_code == source_code:
        return text
    system = "You are a translator. Translate the following text from English to the target language. Provide only the translated text without any additional comments or explanations."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Translate from {source_lang} to {target_lang}:\n{text}"}
    ]
    resp = call_openrouter(messages)
    return resp.strip() if resp else text

def get_sentiment(text: str) -> str:
    system = "You are a sentiment analyzer. Analyze the sentiment of the following text and return only one word: 'positive', 'negative', or 'neutral'."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text}
    ]
    resp = call_openrouter(messages)
    return resp.strip().lower() if resp else "neutral"

def build_question_generation_prompt(candidate_info: Dict, tech_stack: List[str], resume_text: str, application_id: str) -> List[Dict[str, str]]:
    user_lang = candidate_info.get("language", "English")
    resume_excerpt = translate_text(resume_text.strip(), "English", user_lang)
    if len(resume_excerpt) > 2000:
        resume_excerpt = resume_excerpt[:2000] + " [...] (truncated)"
    techs = ", ".join(tech_stack) if tech_stack else "General"

    sys = (
        "You are TalentScout — a professional hiring assistant. "
        "Given candidate information, a resume excerpt, a list of technologies, and an application ID, "
        "generate exactly five (5) unique technical screening questions tailored to the candidate. "
        "To ensure uniqueness, use the application ID as a seed to vary the question phrasing, focus, or specific challenges, "
        "even if the tech stack or resume is similar to other candidates. Introduce randomness in question selection while "
        "ensuring relevance to the tech stack and resume. "
        "Output **only valid JSON** (no explanatory text). "
        "The JSON MUST have the top-level key 'questions' whose value is an array of 5 objects."
        " Each question object must include: id (Q1..Q5), title, type (coding|concept|design|experience), "
        "difficulty (basic|intermediate|advanced), tech (main tech this question is about), "
        "prompt (the exact question to show the candidate), and guidelines (expected points/keywords for evaluation)."
        " For Q1, also include 'expected_output' (example input and expected output for the coding challenge)."
        "Additionally include a 'notes' field at top-level with short instructions for the candidate (one string)."
        "Q1 requirements: a basic programming challenge solvable in any language, varied based on the application ID."
        "Q2-Q4: intermediate conceptual / project / problem-solving questions (not coding), with varied focus."
        "Q5: an advanced, experience-focused technical question related to candidate's stated techs and resume, unique to this session."
        "If resume text is empty, generate general but stack-specific questions with variation. "
        "Avoid asking for any sensitive personal data. Keep each prompt short but specific."
        "Return only JSON. Do not wrap in markdown blocks."
    )

    payload = {
        "system_instructions": sys,
        "candidate_info": {
            "full_name": candidate_info.get("full_name", ""),
            "email": candidate_info.get("email", ""),
            "phone": candidate_info.get("phone", ""),
            "years_experience": candidate_info.get("years_experience", ""),
            "desired_position": candidate_info.get("desired_position", ""),
            "location": candidate_info.get("location", ""),
            "language": candidate_info.get("language", "English")
        },
        "tech_stack": tech_stack,
        "resume_excerpt": resume_excerpt,
        "application_id": application_id,
        "format_instructions": (
            "Return JSON exactly like:\n"
            '{\n'
            '  "notes":"...",\n'
            '  "questions":[\n'
            '    {"id":"Q1","title":"...","type":"coding","difficulty":"basic","tech":"Python","prompt":"...","guidelines":"...","expected_output":"..."},\n'
            '    ... (Q2..Q5 without expected_output)\n'
            '  ]\n'
            '}'
        )
    }

    prompt_text = (
        f"{sys}\n\n"
        "Candidate Info (JSON):\n"
        f"{json.dumps(payload['candidate_info'], ensure_ascii=False)}\n\n"
        f"Tech Stack: {techs}\n\n"
        f"Resume Excerpt:\n"
        f"{resume_excerpt}\n\n"
        f"Application ID (use as seed for question variation): {application_id}\n\n"
        "Formatting instructions:\n"
        f"{payload['format_instructions']}\n\n"
        "Generate JSON now."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt_text}
    ]
    return messages

def call_openrouter(messages: List[Dict[str, str]], timeout: int = API_TIMEOUT) -> Optional[str]:
    if not check_openrouter_key():
        return None
    api_key = get_api_key()
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://talentscout.example.com",  # Optional: for attribution
            "X-Title": "TalentScout"  # Optional: for attribution
        }
        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": 0.7
        }
        resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        st.warning("Unexpected response format from OpenRouter API.")
        return None
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            st.error("API key authentication failed. Please check your OpenRouter API key.")
        elif e.response.status_code == 404:
            st.error("API endpoint not found. Please check the URL.")
        else:
            st.error(f"HTTP error occurred: {e}")
        return None
    except requests.ConnectionError:
        st.error("Failed to connect to OpenRouter API. Please check your internet connection.")
        return None
    except requests.Timeout:
        st.error(f"Request to OpenRouter timed out after {timeout} seconds.")
        return None
    except requests.RequestException as e:
        st.error(f"Failed to reach OpenRouter API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during LLM call: {e}")
        return None

def extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    start_idx = None
    for i, ch in enumerate(text):
        if ch in ["{", "["]:
            start_idx = i
            break
    if start_idx is None:
        return None
    stack = []
    for j in range(start_idx, len(text)):
        ch = text[j]
        if ch == "{" or ch == "[":
            stack.append(ch)
        elif ch == "}" or ch == "]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                continue
            if not stack:
                candidate = text[start_idx:j + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    try:
                        fixed = candidate.replace("'", '"').replace("\n", " ")
                        json.loads(fixed)
                        return fixed
                    except Exception:
                        continue
    return None

def get_fallback_questions(tech_stack: List[str], application_id: str) -> Dict:
    tech = tech_stack[0] if tech_stack else "General"
    # Use application_id to select a random coding challenge for Q1
    coding_challenges = [
        {
            "prompt": f"Write a function in {tech} to reverse a string.",
            "guidelines": "Correct implementation, clear explanation, efficient algorithm.",
            "expected_output": "For input 'hello', expected output 'olleh'."
        },
        {
            "prompt": f"Write a function in {tech} to check if a number is prime.",
            "guidelines": "Correct logic, optimized for performance, handles edge cases.",
            "expected_output": "For input 7, expected output true; for input 4, expected output false."
        },
        {
            "prompt": f"Write a function in {tech} to find the factorial of a number.",
            "guidelines": "Correct implementation, handles non-negative integers, considers recursion or iteration.",
            "expected_output": "For input 5, expected output 120."
        }
    ]
    seed = int(hashlib.md5(application_id.encode()).hexdigest(), 16) % len(coding_challenges)
    selected_coding = coding_challenges[seed]
    
    return {
        "notes": "Please answer the following questions to the best of your ability.",
        "questions": [
            {
                "id": "Q1",
                "title": "Basic Programming Challenge",
                "type": "coding",
                "difficulty": "basic",
                "tech": tech,
                "prompt": selected_coding["prompt"],
                "guidelines": selected_coding["guidelines"],
                "expected_output": selected_coding["expected_output"]
            },
            {
                "id": "Q2",
                "title": "System Design Concept",
                "type": "design",
                "difficulty": "intermediate",
                "tech": tech,
                "prompt": f"Describe how you would design a scalable {tech} application with unique considerations for this role.",
                "guidelines": "Mention architecture, scalability, and unique components."
            },
            {
                "id": "Q3",
                "title": "Debugging Scenario",
                "type": "concept",
                "difficulty": "intermediate",
                "tech": tech,
                "prompt": f"Explain how you would debug a unique performance issue in a {tech} application.",
                "guidelines": "Include specific tools, steps, and potential issues."
            },
            {
                "id": "Q4",
                "title": "Project Experience",
                "type": "experience",
                "difficulty": "intermediate",
                "tech": tech,
                "prompt": f"Describe a unique {tech} project you worked on and its specific challenges.",
                "guidelines": "Focus on unique technical challenges and solutions."
            },
            {
                "id": "Q5",
                "title": "Advanced Technical Concept",
                "type": "experience",
                "difficulty": "advanced",
                "tech": tech,
                "prompt": f"Explain a specific advanced {tech} concept you’ve applied in a unique project scenario.",
                "guidelines": "Depth of understanding, practical application, uniqueness."
            }
        ]
    }

def get_llm_answer(question: str, history: List[Dict], candidate_info: Dict) -> str:
    context = "\nCandidate info: " + json.dumps(candidate_info) + "\nRecent chat history: " + "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    sys_prompt = (
        "You are a helpful hiring assistant. Answer the user's question concisely and professionally. "
        "Personalize the response based on the candidate's information and recent chat history if relevant. "
        "If the question is outside the scope of hiring or you don't have the information, respond appropriately, "
        "e.g., 'I'm sorry, but as a hiring assistant, I may not have details on that topic.' "
        "If you don't know the answer, say so honestly."
    ) + context
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    llm_output = call_openrouter(messages)
    if llm_output:
        return llm_output
    return "Sorry, I couldn't generate an answer at this time."

def generate_feedback(generated_questions: Dict, candidate_answers: Dict) -> str:
    sys_prompt = (
        "You are TalentScout, a professional hiring assistant. "
        "Given the technical questions, their guidelines, and your answers, "
        "evaluate each answer based on the guidelines. "
        "Provide detailed feedback for each question, assessing how well your answer meets the guidelines. "
        "Then, summarize your overall strengths, weaknesses, areas for improvement, and specific suggestions on how to improve. "
        "Use 'you have provided' or 'your strengths are' instead of referring to 'the candidate'. "
        "Be professional, encouraging, and constructive. "
        "If questions were skipped, note that as a weakness. "
        "Output a detailed feedback message, not in JSON format."
    )
    questions_data = [
        {
            "id": q["id"],
            "prompt": q["prompt"],
            "guidelines": q["guidelines"],
            "answer": candidate_answers.get(q["id"], "Skipped")
        }
        for q in generated_questions["questions"]
    ]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Questions and Answers (JSON):\n{json.dumps(questions_data, ensure_ascii=False)}\n\nGenerate detailed feedback now."}
    ]
    llm_output = call_openrouter(messages)
    if llm_output:
        try:
            parsed = json.loads(llm_output)
            return parsed.get("feedback", llm_output)
        except Exception:
            return llm_output
    return (
        "Sorry, I couldn't generate feedback at this time. Please ensure all questions are reviewed and try again.\n\n"
        "**General Advice for Improvement:**\n"
        "- Ensure you attempt all questions, as skipping them may suggest gaps in knowledge.\n"
        "- For coding questions, provide clear, well-commented code and explain your thought process.\n"
        "- For conceptual questions, include specific examples and technical details to demonstrate expertise.\n"
        "- Practice using the STAR method (Situation, Task, Action, Result) for experience-based questions.\n"
        "- Review fundamental and advanced concepts in your chosen tech stack to strengthen your responses."
    )

# ==============================
# Helpers - validation and candidate management
# ==============================
def validate_email(email: str) -> bool:
    if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False
    valid_domains = ["gmail.com", "outlook.com", "hotmail.com", "live.com"]
    domain = email.lower().split("@")[-1]
    return domain in valid_domains

def validate_phone(phone: str, country_code: str) -> bool:
    phone = phone.strip()
    if not phone:
        return False
    cleaned_phone = re.sub(r"[^0-9]", "", phone)
    return 7 <= len(cleaned_phone) <= 15

def save_candidate_info(candidate_info: dict, application_id: str) -> bool:
    try:
        csv_file = "registered_candidates.csv"
        candidate_name = candidate_info.get("full_name", "candidate").replace(" ", "_")
        json_file = f"{candidate_name}_with_{application_id}.json"
        headers = ["full_name", "email", "phone", "application_id", "json_file"]
        data = {
            "full_name": candidate_info.get("full_name", ""),
            "email": candidate_info.get("email", ""),
            "phone": candidate_info.get("phone", ""),
            "application_id": application_id,
            "json_file": json_file
        }
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    except Exception as e:
        st.error(f"Failed to save candidate info to CSV: {e}")
        return False

def check_past_candidate(email: str, phone: str) -> Optional[Dict]:
    csv_file = "registered_candidates.csv"
    if not os.path.exists(csv_file):
        return None
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row["email"].lower() == email.lower() and
                    row["phone"] == phone):
                    json_file = row["json_file"]
                    if os.path.exists(json_file):
                        with open(json_file, "r", encoding="utf-8") as jf:
                            return json.load(jf)
        return None
    except Exception as e:
        st.error(f"Error checking past candidate: {e}")
        return None

def save_candidate_responses(candidate_info: dict, generated: dict, answers: dict, application_id: str) -> bool:
    try:
        candidate_name = candidate_info.get("full_name", "candidate").replace(" ", "_")
        filename = f"{candidate_name}_with_{application_id}.json"
        data = {
            "application_id": application_id,
            "candidate_info": {
                "full_name": candidate_info.get("full_name", ""),
                "email": candidate_info.get("email", ""),
                "phone": candidate_info.get("phone", ""),
                "years_experience": candidate_info.get("years_experience", ""),
                "desired_position": candidate_info.get("desired_position", ""),
                "location": candidate_info.get("location", ""),
                "language": candidate_info.get("language", "English")
            },
            "notes": generated.get("notes", ""),
            "questions": [
                {
                    **q,
                    "answer": answers.get(q["id"], "Skipped")
                }
                for q in generated["questions"]
            ],
            "chat_history": st.session_state.get("chat_history", []),
            "tech_stack": st.session_state.get("selected_techs", []),
            "resume_text": st.session_state.get("resume_text", ""),
            "current_question_index": st.session_state.get("current_question_index", 0),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        save_candidate_info(candidate_info, application_id)
        return True
    except Exception as e:
        st.error(f"Failed to save candidate responses: {e}")
        return False

def save_question_answer_pair(question: str, answer: str, question_id: str, application_id: str) -> bool:
    try:
        candidate_name = st.session_state.candidate_info.get("full_name", "candidate").replace(" ", "_")
        filename = f"{candidate_name}_with_{application_id}_answers.json"
        answer_data = {"question": question, "answer": answer}
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(answer_data, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        st.error(f"Failed to save question-answer pair for {question_id}: {e}")
        return False

def download_button_for_json(data: dict, filename: str = "questions.json", key: str = "download_json"):
    b = json.dumps(data, indent=2, ensure_ascii=False)
    st.download_button(label="Download questions (JSON)", data=b, file_name=filename, mime="application/json", key=key)

def download_button_for_text(text: str, filename: str = "questions.txt", key: str = "download_txt"):
    st.download_button(label="Download questions (TXT)", data=text, file_name=filename, mime="text/plain", key=key)

# Endings
ENDINGS = [
    'It was a pleasure chatting with you today. Remember: "The only way to do great work is to love what you do." We\'ll be in touch soon! If you\'re selected for the next round, our HR team will contact you via email, WhatsApp, or phone. Best of luck!',
    'Our conversation is now complete. I hope I was able to help you today. Keep this in mind: "Believe you can and you\'re halfway there." Our HR team will reach out to you via email, WhatsApp, or phone if you are shortlisted. Wishing you all the best!'
]

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="TalentScout - Hiring Assistant", layout="wide", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def apply_theme():
    if st.session_state.theme == "dark":
        css = """
        <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stChatMessage.assistant {
                background-color: #2a3b4c;
                border-radius: 10px;
                padding: 10px;
            }
            .stChatMessage.user {
                background-color: #3a3a3a;
                border-radius: 10px;
                padding: 10px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
            }
            .stDownloadButton > button {
                background-color: #008CBA;
                color: white;
            }
            .stTextInput > div > input, .stSelectbox > div > select, .stMultiSelect > div > div {
                background-color: #333;
                color: #ffffff;
                border: 1px solid #555;
            }
            .stTextArea > div > textarea {
                background-color: #333;
                color: #ffffff;
                border: 1px solid #555;
            }
            .stSidebar {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff;
            }
            .stMarkdown p {
                color: #ffffff;
            }
            .stChatMessage.assistant::before {
                content: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M8 8h8v8H8z"/><circle cx="9" cy="11" r="1"/><circle cx="15" cy="11" r="1"/><path d="M9 15c1.5 1 3.5 1 5 0"/></svg>');
                display: inline-block;
                vertical-align: middle;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
            /* Disable paste in text areas */
            textarea {
                pointer-events: auto;
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const textAreas = document.querySelectorAll('textarea');
                textAreas.forEach(textArea => {
                    textArea.addEventListener('paste', function(e) {
                        e.preventDefault();
                        alert('Pasting is disabled. Please type your answer.');
                    });
                });
            });
        </script>
        """
    else:
        css = """
        <style>
            .stApp {
                background-color: #f0f2f6;
                color: #000000;
            }
            .stChatMessage.assistant {
                background-color: #e6f3ff;
                border-radius: 10px;
                padding: 10px;
            }
            .stChatMessage.user {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
            }
            .stDownloadButton > button {
                background-color: #008CBA;
                color: white;
            }
            .stTextInput > div > input, .stSelectbox > div > select, .stMultiSelect > div > div {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ccc;
            }
            .stTextArea > div > textarea {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ccc;
            }
            .stSidebar {
                background-color: #ffffff;
                color: #000000;
            }
            .stChatMessage.assistant::before {
                content: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M8 8h8v8H8z"/><circle cx="9" cy="11" r="1"/><circle cx="15" cy="11" r="1"/><path d="M9 15c1.5 1 3.5 1 5 0"/></svg>');
                display: inline-block;
                vertical-align: middle;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
            /* Disable paste in text areas */
            textarea {
                pointer-events: auto;
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const textAreas = document.querySelectorAll('textarea');
                textAreas.forEach(textArea => {
                    textArea.addEventListener('paste', function(e) {
                        e.preventDefault();
                        alert('Pasting is disabled. Please type your answer.');
                    });
                });
            });
        </script>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme()

with st.sidebar:
    st.header("About TalentScout")
    st.markdown("""
    TalentScout is an AI-powered hiring assistant that guides you through the application process,
    generates tailored technical questions, and answers your queries.
    
    - Powered by OpenRouter API
    - Privacy-focused
    - Easy to use chatbot interface
    """)
    st.markdown("---")
    st.caption(f"API: {OPENROUTER_BASE_URL}")
    st.caption(f"Model: {MODEL_NAME}")
    st.markdown("---")
    if st.button("Toggle Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_theme()
        st.rerun()

st.title("TalentScout - Your Hiring Assistant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {}
if "selected_techs" not in st.session_state:
    st.session_state.selected_techs = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = None
if "job_role" not in st.session_state:
    st.session_state.job_role = ""
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "candidate_answers" not in st.session_state:
    st.session_state.candidate_answers = {}
if "application_id" not in st.session_state:
    st.session_state.application_id = str(uuid.uuid4())
if "last_input_time" not in st.session_state:
    st.session_state.last_input_time = None

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def scroll_to_bottom():
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# Chatbot steps
if st.session_state.step == 0:
    original_greeting = (
        "Hello! I'm TalentScout, your personal guide to finding the perfect job opportunity. "
        "Let's get started by collecting your personal details to check if you've applied before."
    )
    language = "English"  # Default
    st.session_state.chat_history.append({"role": "assistant", "content": original_greeting})
    with st.chat_message("assistant"):
        st.markdown(original_greeting)
        with st.form("initial_candidate_form"):
            full_name = st.text_input("Full name", value=st.session_state.candidate_info.get("full_name", ""))
            email = st.text_input("Email (Gmail, Outlook, Hotmail, or Live only)", value=st.session_state.candidate_info.get("email", ""))
            country_code = st.selectbox("Country code", options=COUNTRY_CODES, index=0, key="initial_country_code_select")
            phone = st.text_input("Phone number (e.g., 12345 67890)", value=st.session_state.candidate_info.get("phone", "").split(" ", 1)[-1] if st.session_state.candidate_info.get("phone") else "")
            language = st.selectbox("Preferred language", options=LANGUAGES, index=0)
            submit_initial = st.form_submit_button("Submit")
        if submit_initial:
            if not full_name:
                st.error("Please provide your full name.")
            elif not email or not validate_email(email):
                st.error("Please provide a valid email address (Gmail, Outlook, Hotmail, or Live).")
            elif not phone or not validate_phone(phone, country_code):
                st.error("Please provide a valid phone number (7-15 digits).")
            else:
                full_phone = f"{country_code.split(' ')[0]} {phone.strip()}"
                st.session_state.candidate_info.update({
                    "full_name": full_name,
                    "email": email,
                    "phone": full_phone,
                    "language": language
                })
                user_lang = language
                past_data = check_past_candidate(email, full_phone)
                if past_data:
                    st.session_state.application_id = past_data.get("application_id", str(uuid.uuid4()))
                    st.session_state.candidate_info = past_data.get("candidate_info", {})
                    st.session_state.candidate_info["language"] = user_lang  # Update language if changed
                    st.session_state.generated_questions = {
                        "notes": past_data.get("notes", ""),
                        "questions": past_data.get("questions", [])
                    }
                    st.session_state.candidate_answers = {
                        q["id"]: q.get("answer", "Skipped") for q in past_data.get("questions", [])
                    }
                    st.session_state.chat_history = past_data.get("chat_history", [])
                    st.session_state.job_role = st.session_state.candidate_info.get("desired_position", "")
                    st.session_state.selected_techs = past_data.get("tech_stack", [])
                    st.session_state.resume_text = past_data.get("resume_text", "")
                    st.session_state.current_question_index = past_data.get("current_question_index", 0)
                    original_welcome = f"Hello {full_name}, welcome back, how may I help you?"
                    welcome_trans = translate_text(original_welcome, user_lang)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": welcome_trans
                    })
                    st.session_state.step = 6
                else:
                    original_user_msg = f"Personal info submitted: Name: {full_name}, Email: {email}, Phone: {full_phone}, Language: {user_lang}"
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": original_user_msg
                    })
                    original_assistant_msg = "Thank you! Let's continue by selecting your tech stack."
                    assistant_trans = translate_text(original_assistant_msg, user_lang)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_trans
                    })
                    st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    user_lang = st.session_state.candidate_info.get("language", "English")
    original_msg = "Please select the technologies you're skilled in (up to 6):"
    msg_trans = translate_text(original_msg, user_lang)
    with st.chat_message("assistant"):
        st.markdown(msg_trans)
        DEFAULT_TECHS = [
            "Python", "Data Science", "AI", "Machine Learning", "Deep Learning",
            "SQL / Database", "NoSQL / MongoDB", "Power BI", "BI Developer",
            "Web Development", "JavaScript", "React", "Node.js", "Django", "Flask",
            "DevOps", "AWS", "GCP", "Azure", "Docker", "Kubernetes", "NLP",
            "Computer Vision", "Pytorch", "TensorFlow", "Spark", "Hadoop"
        ]
        selected_techs = st.multiselect(
            "Choose technologies:", options=DEFAULT_TECHS, default=st.session_state.get("selected_techs", []), max_selections=6, key="tech_select"
        )
        custom_tech = st.text_input("Add custom tech (comma-separated):", value="")
        if custom_tech:
            extras = [c.strip() for c in custom_tech.split(",") if c.strip()]
            for e in extras:
                if e not in selected_techs:
                    selected_techs.append(e)
        if st.button("Submit Tech Stack"):
            if not selected_techs:
                st.error(translate_text("Please select at least one technology.", user_lang))
            else:
                st.session_state.selected_techs = selected_techs
                original_user_msg = f"Selected techs: {', '.join(selected_techs)}"
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": original_user_msg
                })
                original_assistant_msg = "Great! Now, please select the job role you're interested in."
                assistant_trans = translate_text(original_assistant_msg, user_lang)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_trans
                })
                st.session_state.step = 2
                st.rerun()

elif st.session_state.step == 2:
    user_lang = st.session_state.candidate_info.get("language", "English")
    original_msg = "Please select the job role you're applying for:"
    msg_trans = translate_text(original_msg, user_lang)
    with st.chat_message("assistant"):
        st.markdown(msg_trans)
        JOB_ROLES = [
            "AI/ML Engineer", "Data Scientist", "Software Engineer", "Web Developer",
            "DevOps Engineer", "Data Engineer", "BI Developer", "Cloud Architect"
        ]
        job_role = st.selectbox("Choose a job role:", options=JOB_ROLES, key="job_role_select")
        if st.button("Submit Job Role"):
            st.session_state.job_role = job_role
            st.session_state.candidate_info["desired_position"] = job_role
            original_user_msg = f"Selected job role: {job_role}"
            st.session_state.chat_history.append({
                "role": "user",
                "content": original_user_msg
            })
            original_assistant_msg = "Awesome! Now, please provide additional personal information."
            assistant_trans = translate_text(original_assistant_msg, user_lang)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_trans
            })
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    user_lang = st.session_state.candidate_info.get("language", "English")
    original_msg = "Please provide additional personal details:"
    msg_trans = translate_text(original_msg, user_lang)
    with st.chat_message("assistant"):
        st.markdown(msg_trans)
        with st.form("candidate_form"):
            experience_options = ["Fresher", "1 - 3 yrs", "4 - 6 yrs", "7 - 10 yrs"]
            years_experience = st.selectbox(
                "Years of experience",
                options=experience_options,
                index=experience_options.index(st.session_state.candidate_info.get("years_experience", "Fresher"))
                if st.session_state.candidate_info.get("years_experience") in experience_options else 0,
                key="experience_select"
            )
            location = st.text_input("Current location", value=st.session_state.candidate_info.get("location", ""))
            consent_store = st.checkbox(
                "I consent to store anonymized metadata for demo/testing (no raw PII stored)",
                value=True
            )
            submit_info = st.form_submit_button("Submit Personal Info")
        if submit_info:
            st.session_state.candidate_info.update({
                "years_experience": years_experience,
                "location": location,
                "consent_store": consent_store
            })
            original_user_msg = f"Additional info submitted: Years of Experience: {years_experience}, Location: {location}"
            st.session_state.chat_history.append({
                "role": "user",
                "content": original_user_msg
            })
            original_assistant_msg = "Thank you! Now, please upload your resume (PDF, DOCX, or TXT)."
            assistant_trans = translate_text(original_assistant_msg, user_lang)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_trans
            })
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
    user_lang = st.session_state.candidate_info.get("language", "English")
    original_msg = "Please upload your resume:"
    msg_trans = translate_text(original_msg, user_lang)
    with st.chat_message("assistant"):
        st.markdown(msg_trans)
        uploaded_file = st.file_uploader("Upload resume (PDF/DOCX/TXT)", type=["pdf", "docx", "doc", "txt"], key="resume_upload")
        if st.button("Submit Resume"):
            if uploaded_file is None:
                st.error(translate_text("Please upload a resume file.", user_lang))
            else:
                try:
                    resume_text = parse_resume(uploaded_file)
                    st.session_state.resume_text = resume_text
                    original_user_msg = f"Resume uploaded: {uploaded_file.name}"
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": original_user_msg
                    })
                    if not resume_text:
                        original_assistant_msg = "Could not extract text from resume, but I'll generate questions based on the info provided. Let's proceed to the technical questions!"
                        assistant_trans = translate_text(original_assistant_msg, user_lang)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_trans
                        })
                    else:
                        original_assistant_msg = "Resume processed successfully! Generating your tailored technical questions..."
                        assistant_trans = translate_text(original_assistant_msg, user_lang)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_trans
                        })
                    st.session_state.step = 5
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")
                    original_assistant_msg = f"Error parsing resume: {e}. Please try uploading again or continue without resume text."
                    assistant_trans = translate_text(original_assistant_msg, user_lang)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_trans
                    })

elif st.session_state.step == 5:
    user_lang = st.session_state.candidate_info.get("language", "English")
    with st.spinner(translate_text("Be prepared for your assessment! Generating unique questions for you...", user_lang)):
        if not st.session_state.get("generated_questions"):
            if not check_openrouter_key():
                st.rerun()
            messages = build_question_generation_prompt(
                st.session_state.candidate_info,
                st.session_state.selected_techs,
                st.session_state.resume_text,
                st.session_state.application_id
            )
            llm_output = call_openrouter(messages)
            if not llm_output:
                original_msg = "Sorry, couldn't generate questions due to API issues. Using fallback questions."
                msg_trans = translate_text(original_msg, user_lang)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": msg_trans
                })
                parsed = get_fallback_questions(st.session_state.selected_techs, st.session_state.application_id)
            else:
                parsed = None
                try:
                    parsed = json.loads(llm_output)
                except Exception:
                    js = extract_json_from_text(llm_output)
                    if js:
                        try:
                            parsed = json.loads(js)
                        except:
                            pass
                if not parsed or not isinstance(parsed, dict) or "questions" not in parsed or len(parsed["questions"]) != 5:
                    st.warning("Unexpected LLM output format. Using fallback questions.")
                    original_msg = f"Unexpected format in LLM output. Using fallback questions. Raw output (truncated):\n{llm_output[:1000]}"
                    msg_trans = translate_text(original_msg, user_lang)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": msg_trans
                    })
                    parsed = get_fallback_questions(st.session_state.selected_techs, st.session_state.application_id)
            st.session_state.generated_questions = parsed
            st.session_state.candidate_answers = {q["id"]: "" for q in parsed["questions"]}
            st.session_state.current_question_index = 0

    if st.session_state.current_question_index < len(st.session_state.generated_questions["questions"]):
        question = st.session_state.generated_questions["questions"][st.session_state.current_question_index]
        question_id = question["id"]
        prompt_trans = translate_text(question["prompt"], user_lang)
        with st.chat_message("assistant"):
            st.markdown(prompt_trans)
            if question_id == "Q1" and "expected_output" in question:
                exp_trans = translate_text(question["expected_output"], user_lang)
                exp_label_trans = translate_text("Expected output:", user_lang)
                st.markdown(f"**{exp_label_trans}** {exp_trans}")
            with st.form(f"answer_form_{question_id}"):
                answer = st.text_area(
                    translate_text("Your answer:" if question_id != "Q1" else "Your code:", user_lang),
                    value=st.session_state.candidate_answers.get(question_id, ""),
                    height=200,
                    key=f"answer_{question_id}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    submit_answer = st.form_submit_button(translate_text("Submit Answer", user_lang))
                with col2:
                    skip_question = st.form_submit_button(translate_text("Skip Question", user_lang))
                
                if submit_answer:
                    if answer.strip():
                        answer_trans = translate_text(answer.strip(), "English", user_lang)
                        sentiment = get_sentiment(answer_trans)
                        st.session_state.candidate_answers[question_id] = answer_trans
                        # Save question-answer pair to JSON Lines file
                        save_question_answer_pair(question["prompt"], answer_trans, question_id, st.session_state.application_id)
                        # Add user answer to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"Answer to {question_id}: {answer}"
                        })
                        # Add confirmation message with the user's answer to chat history
                        confirmation_msg = f"You provided the following answer for {question_id}: {answer}"
                        confirmation_trans = translate_text(confirmation_msg, user_lang)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": confirmation_trans
                        })
                        st.session_state.current_question_index += 1
                        st.rerun()
                    else:
                        st.error(translate_text("Please provide an answer before submitting.", user_lang))
                
                if skip_question:
                    st.session_state.candidate_answers[question_id] = "Skipped"
                    skipped_trans = translate_text("Skipped", user_lang)
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"{skipped_trans} {question_id}"
                    })
                    st.session_state.current_question_index += 1
                    st.rerun()
    else:
        save_candidate_responses(
            st.session_state.candidate_info,
            st.session_state.generated_questions,
            st.session_state.candidate_answers,
            st.session_state.application_id
        )
        st.session_state.step = 6
        st.session_state.last_input_time = datetime.now().timestamp()
        st.rerun()

elif st.session_state.step == 6:
    user_lang = st.session_state.candidate_info.get("language", "English")
    st.markdown("""
    <script>
        let lastInputTime = Date.now();
        document.addEventListener('input', function() {
            lastInputTime = Date.now();
        });
        setInterval(function() {
            if ((Date.now() - lastInputTime) / 1000 > 10) {
                document.getElementById('auto_end').value = 'no';
                document.getElementById('chat_form').submit();
            }
        }, 1000);
    </script>
    """, unsafe_allow_html=True)
    
    with st.form("chat_form"):
        prompt_label_trans = translate_text("Great! Do you have any questions for me?", user_lang)
        prompt = st.text_input(
            prompt_label_trans,
            key="chat_input",
            value=""
        )
        st.markdown('<input type="hidden" id="auto_end" name="auto_end">', unsafe_allow_html=True)
        submit_prompt = st.form_submit_button(translate_text("Submit", user_lang))
    
    if submit_prompt and prompt:
        st.session_state.last_input_time = datetime.now().timestamp()
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        prompt_trans = translate_text(prompt, "English", user_lang)
        sentiment = get_sentiment(prompt_trans)
        prompt_lower = prompt_trans.lower().strip()
        if prompt_lower in ['no', 'end', 'thankyou']:
            end_message = random.choice(ENDINGS)
            end_trans = translate_text(end_message, user_lang)
            st.session_state.chat_history.append({"role": "assistant", "content": end_trans})
            with st.chat_message("assistant"):
                st.markdown(end_trans)
            st.session_state.step = 7
            st.rerun()
        elif any(keyword in prompt_lower.replace(" ", "") for keyword in ["feedback", "review", "evaluate", "feedback"]):
            feedback = generate_feedback(st.session_state.generated_questions, st.session_state.candidate_answers)
            feedback_trans = translate_text(feedback, user_lang)
            st.session_state.chat_history.append({"role": "assistant", "content": feedback_trans})
            with st.chat_message("assistant"):
                st.markdown(feedback_trans)
            st.session_state.last_input_time = datetime.now().timestamp()
            st.rerun()
        else:
            answer = get_llm_answer(prompt_trans, st.session_state.chat_history, st.session_state.candidate_info)
            if sentiment == "negative":
                encourage = "Don't worry, I'm here to help!"
                answer += " " + encourage
            answer_trans = translate_text(answer, user_lang)
            st.session_state.chat_history.append({"role": "assistant", "content": answer_trans})
            with st.chat_message("assistant"):
                st.markdown(answer_trans)
            st.session_state.last_input_time = datetime.now().timestamp()
            st.rerun()

elif st.session_state.step == 7:
    user_lang = st.session_state.candidate_info.get("language", "English")
    if st.session_state.generated_questions:
        with st.sidebar:
            st.markdown("---")
            st.header(translate_text("Download Questions and Answers", user_lang))
            download_button_for_json(
                {
                    "application_id": st.session_state.application_id,
                    "notes": st.session_state.generated_questions.get("notes", ""),
                    "questions": [
                        {
                            **q,
                            "answer": st.session_state.candidate_answers.get(q["id"], "Skipped")
                        }
                        for q in st.session_state.generated_questions["questions"]
                    ],
                    "tech_stack": st.session_state.selected_techs,
                    "resume_text": st.session_state.resume_text,
                    "current_question_index": st.session_state.current_question_index
                },
                filename=f"{st.session_state.candidate_info.get('full_name', 'candidate').replace(' ', '_')}_with_{st.session_state.application_id}.json",
                key="download_json_end"
            )
            download_button_for_text(
                "\n".join([
                    f"**Question {q['id']}**: {q['prompt']}\n"
                    f"Your Answer: {st.session_state.candidate_answers.get(q['id'], 'Skipped')}\n"
                    for q in st.session_state.generated_questions["questions"]
                ]),
                filename=f"{st.session_state.candidate_info.get('full_name', 'candidate').replace(' ', '_')}_with_{st.session_state.application_id}.txt",
                key="download_txt_end"
            )
    if st.session_state.candidate_info.get("consent_store", False):
        if st.sidebar.button(translate_text("Save anonymized submission", user_lang)):
            success = save_submission_anonymized(
                st.session_state.candidate_info,
                st.session_state.selected_techs,
                {
                    "notes": st.session_state.generated_questions.get("notes", ""),
                    "questions": [
                        {
                            **q,
                            "answer": st.session_state.candidate_answers.get(q["id"], "Skipped")
                        }
                        for q in st.session_state.generated_questions["questions"]
                    ]
                },
                st.session_state.candidate_info.get("consent_store", False)
            )
            if success:
                st.sidebar.success(translate_text("Anonymized metadata saved.", user_lang))
            else:
                st.sidebar.error(translate_text("Failed to save.", user_lang))
    scroll_to_bottom()

# ==============================
# Save anonymized submission
# ==============================
def save_submission_anonymized(candidate_info: dict, techs: List[str], generated: dict, consent: bool):
    if not consent:
        return False
    try:
        safe = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "name_hash": candidate_info.get("full_name", "")[:1] + "***",
            "years_experience": candidate_info.get("years_experience", ""),
            "tech_stack": techs,
            "job_role": candidate_info.get("desired_position", ""),
            "questions_count": len(generated.get("questions", [])) if isinstance(generated, dict) else 0
        }
        with open("submissions_anonymized.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False

scroll_to_bottom()