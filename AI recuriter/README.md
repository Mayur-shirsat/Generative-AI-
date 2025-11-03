# 🤖 TalentScout – AI Hiring Assistant

**TalentScout** is an **AI-powered hiring assistant** built with **Streamlit** that automates the **initial technical screening process**.  
It guides candidates through an interactive application flow, parses their resumes, and dynamically generates **personalized technical quizzes** based on their job role and experience.

---

## ✨ Features

- 🧭 **Guided Application** — A multi-step wizard that collects candidate information in an intuitive way.  
- 📄 **Resume Parsing** — Extracts and processes content from `.pdf`, `.docx`, and `.txt` resumes.  
- 💡 **Dynamic Questions** — Leverages an AI model (via **OpenRouter**) to generate unique, role-specific technical questions.  
- 🌐 **Multi-Lingual Support** — Automatically translates the app interface and chat responses into the candidate’s chosen language.  
- 💬 **Interactive Q&A** — After completing the quiz, candidates can interact with the AI for clarifications and instant feedback.  
- 📁 **Data Export** — Saves the entire screening session (questions and answers) to a JSON file for future analysis.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / Framework** | Streamlit |
| **Language** | Python |
| **LLM Integration** | OpenRouter API |
| **Document Parsing** | PyPDF2, python-docx |
| **Environment Management** | python-dotenv |

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- Python **3.8+**
- An **OpenRouter API Key**

---

### 2️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/mayur-shirsat/talentscout.git
cd talentscout

# Install dependencies
pip install -r requirements.txt
