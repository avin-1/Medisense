# 🏥 Medisense: Advanced Multi-Agent Medical AI Assistant

Medisense is a state-of-the-art, multi-agent medical diagnostic system designed to provide high-precision health insights. Unlike traditional symptom checkers, Medisense employs a **collaborative AI architecture** where specialized agents work together to extract symptoms, retrieve medical knowledge, evaluate risks, and perform iterative clinical interviewing.

---

## 🌟 Core Features

### 🧠 Collaborative Multi-Agent Architecture
- **Clinical Intake Agent**: Extracts symptoms from natural language and conversational history.
- **Diagnostic RAG Agent**: High-speed retrieval from a vector database of 4,900+ disease mappings.
- **Triage & Risk Agent**: Real-time evaluation of symptom severity and emergency alerting.
- **Pure LLM Diagnostician**: Leverages `openai/gpt-oss-120b` for deep medical reasoning.
- **Consensus Synthesizer**: Resolves discrepancies between agents to provide a final, verified diagnosis.
- **Translation Agent**: Automatic multi-lingual support (Hindi, Spanish, etc.) for all user interactions.
- **Follow-up Agent**: Generates targeted clinical questions to refine diagnostic accuracy.

### 🔄 Iterative Clinical Interviewing
Medisense doesn't just guess; it interviews. If your symptoms are ambiguous, the system enters an investigative mode, asking 2-3 targeted follow-up questions to differentiate between potential conditions before reaching a high-confidence diagnosis.

### 👤 Personalized Onboarding
Every session begins with a structured profile setup (Age, Gender, Location, Medical History). This context is injected into every agent's reasoning process, ensuring that age-specific or location-specific health factors are always considered.

### 🌍 Global Accessibility
Full support for multi-lingual input and output. Speak in your native language, and Medisense will respond in kind, preserving the medical accuracy of the translation.

---

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI Models**: `openai/gpt-oss-120b` via Groq
- **Database**: ChromaDB (Vector Search)
- **Frontend**: React, Vite, TypeScript, TailwindCSS
- **Search**: Tavily API (Medical Updates)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm/yarn
- Groq API Key
- Tavily API Key (Optional, for web updates)

### 2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create a .env file with:
# GROQ_API_KEY=your_key
# TAVILY_API_KEY=your_key

# Start the API server
python api_server.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📄 Documentation
For a detailed breakdown of each agent's internal logic and the system architecture, please refer to [details.md](file:///c:/Users/Avinash/OneDrive/Desktop/Ideathon/details.md).

---

> [!CAUTION]
> **Medical Disclaimer**: Medisense is an AI-powered tool for informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. In case of emergency, call your local emergency services immediately.
