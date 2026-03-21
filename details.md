# Medisense Multi-Agent Architecture

This document outlines the specialized AI agents that power the Medisense intelligent diagnostic system. Each agent is responsible for a specific part of the clinical workflow.

## 1. Clinical Intake Agent (`agent_1_extraction.py`)
- **Main Work**: Natural Language Processing (NLP) and Symptom Extraction.
- **Details**: It takes the raw conversational input from the user (in any language) and maps it strictly to a predefined list of valid medical symptoms. It ensures the downstream agents receive clean, structured data.

## 2. Diagnostic RAG Agent (`agent_2_rag.py`)
- **Main Work**: Knowledge Retrieval (RAG).
- **Details**: Uses a Vector Database (ChromaDB) to perform Retrieval-Augmented Generation. It looks up symptoms in a local medical knowledge base to find diseases that match the patient's profile based on historical or research data.

## 3. Triage Agent (`agent_3_triage.py`)
- **Main Work**: Risk Assessment and Emergency Detection.
- **Details**: Calculates a "Risk Score" based on symptom severity. If it detects life-threatening symptoms or a high cumulative risk, it triggers an emergency alert to halt normal processing and advise immediate medical care.

## 4. Medical Updater Agent (`agent_4_updater.py`)
- **Main Work**: Knowledge Base Expansion.
- **Details**: A background agent that uses web search (Tavily) to research new diseases. It structures raw web data into symptoms and descriptions and injects them into the Vector DB and CSV files, keeping the system up-to-date.

## 5. Pure LLM Agent (`agent_5_pure_llm.py`)
- **Main Work**: Internal Intuition/Zero-Shot Prediction.
- **Details**: Relies entirely on the LLM's vast pre-trained medical knowledge (`openai/gpt-oss-120b`). It provides a "second opinion" to compare against the RAG results.

## 6. Consensus Synthesizer Agent (`agent_6_combiner.py`)
- **Main Work**: Data Fusion and Final Decision Making.
- **Details**: Acts as the "Chief Medical Officer." It analyzes the symptoms, the RAG predictions, and the Pure LLM predictions. It resolves discrepancies, ranks the final top 3 diseases, and provides clear medical reasoning for the diagnosis.

## 7. Translation Agent (`agent_7_translator.py`)
- **Main Work**: Multilingual Support.
- **Details**: Automatically detects the language of the user's input. It translates all final outputs (messages, descriptions, reasoning, and precautions) back into the user's native language (e.g., Hindi) to ensure accessibility.

---
**Core LLM**: All reasoning tasks currently utilize **openai/gpt-oss-120b** for high-precision clinical analysis.
