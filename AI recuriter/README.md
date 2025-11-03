Here is a professional README.md file for your GitHub repository, based on the code and documentation you've provided.

TalentScout - AI Hiring Assistant
TalentScout is a sophisticated, AI-powered hiring assistant built with Streamlit. It automates the initial technical screening process by guiding candidates through a multi-step application, parsing their resume, and dynamically generating a tailored technical quiz.

🤖 Project Purpose
This tool is designed to solve several key challenges in technical recruiting:

Scalability: Efficiently screen a high volume of candidates.

Consistency: Provide a standardized screening experience for all applicants.

Engagement: Create an interactive and modern application process.

Uniqueness: Generate unique, role-specific questions for each candidate to prevent cheating and get a true measure of their skills.

Core Features
Guided Application: A multi-step wizard collects candidate PII, preferred language, tech stack, and job role.

Resume Parsing: Automatically extracts text from .pdf, .docx, and .txt resumes to be used as context.

Dynamic Question Generation: Connects to the OpenRouter API (using qwen/qwen3-coder:free) to generate 5 unique technical questions based on the candidate's profile.

Robust Fallback: If the API fails, a set of pre-defined fallback questions is used to ensure the screening is not interrupted.

Multi-Lingual Interface: The entire UI and all AI interactions can be translated in real-time to the candidate's preferred language.

Interactive Screening: Presents questions one by one and records the candidate's answers.

Paste Prevention: Includes custom JavaScript to disable pasting into answer fields, encouraging original thought.

AI-Powered Q&A: After the screening, candidates can ask the AI questions about the role or request performance feedback on their answers.

Session Persistence: Saves all interview data to a local JSON file and can recognize returning candidates to allow them to resume.

Data Export: Allows for the download of the complete screening (questions and answers) as a JSON or TXT file.

🛠 Tech Stack
Core Framework: Python, Streamlit

LLM API: OpenRouter

Document Parsing: PyPDF2 (for PDF), python-docx (for DOCX)

Configuration: python-dotenv

Data Handling: json, csv, hashlib, re

🚀 Getting Started
Follow these instructions to get a local copy up and running.

1. Prerequisites
Python 3.8 or newer

An OpenRouter API Key

2. Installation & Setup
Clone the repository:

Bash

git clone https://github.com/your-username/talentscout.git
cd talentscout
Create and activate a virtual environment:

Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
Install the required dependencies:

Bash

pip install streamlit requests pypdf2 python-docx python-dotenv
Configure your API Key:

Create a file named .env in the root directory of the project.

Add your OpenRouter API key to this file:

Code snippet

OPENROUTER_API_KEY="sk-or-your-openrouter-api-key-here"
3. Running the Application
Once your environment is set up, run the application using Streamlit:

Bash

streamlit run app.py
Open your web browser and navigate to the local URL provided (usually http://localhost:8501).

⚙️ How It Works (Application Flow)
The application functions as a state machine, guiding the user through a series of steps:

Welcome (Step 0): Greets the user and collects initial information (name, email, phone, language). It checks registered_candidates.csv to see if the user has applied before.

Tech Stack (Step 1): The user selects their primary technologies from a list.

Job Role (Step 2): The user selects the job role they are applying for.

Resume Upload (Step 4): The user uploads their resume, which is then parsed for text.

Technical Screening (Step 5):

The app sends all the collected data to the LLM to generate 5 unique questions.

The user is presented with each question one at a time and must submit an answer before proceeding.

Q&A and Feedback (Step 6): After the screening, the user enters an open-chat mode where they can ask the AI assistant questions or request feedback on their performance.

Session End (Step 7): The chat concludes, and the user is presented with a final message. The full session data is saved.

💾 Data Persistence
The application saves data locally to create a persistent record of all screenings:

registered_candidates.csv: A lightweight CSV file that acts as an index of all candidates who have started an application.

{candidate_name}_with_{application_id}.json: A comprehensive JSON file containing all information for a single candidate, including their info, the generated questions, and their answers.

submissions_anonymized.jsonl: If the candidate consents, anonymized metadata (job role, tech stack, experience) is saved to this JSON Lines file for analytics.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to:

Open an issue to discuss what you would like to change.

Fork the repository and create a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
