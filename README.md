# 🏡 HOA AI — Retrieval-Augmented Assistant for HOA Documents

A Retrieval-Augmented Generation (RAG) system that answers homeowner questions using HOA governing documents (CC&Rs, Bylaws, Rules) with **evidence-aware reasoning and calibrated confidence**.

---

## 🚀 Overview

HOA documents are:
- long
- legalistic
- difficult to search

This project builds an AI assistant that:
- retrieves relevant excerpts from HOA documents
- answers questions using those excerpts
- distinguishes between **explicit rules** and **inferred answers**
- avoids overconfident or hallucinated responses

---

## 🎯 Key Features

### 🔎 Retrieval-Augmented Answers
- Uses vector embeddings to search HOA documents
- Returns top relevant excerpts with scores

### 🧠 Evidence-Aware Reasoning
- Differentiates between:
  - **Explicit rules** (clearly stated)
  - **Inferred conclusions** (based on related text)
- Prevents misleading certainty

### ⚠️ Confidence Calibration
- Avoids hallucinations
- Clearly states when:
  - information is missing
  - rules are not explicitly defined

### 🧩 Fallback Knowledge Handling
- Injects known supporting evidence when retrieval misses key context
- Improves consistency across similar questions

### 📊 Debug Logging
- Logs:
  - queries
  - retrieved chunks
  - scores
  - outputs
- Enables systematic evaluation and tuning

---

## 🏗️ Architecture

PDF Documents → Chunking → Embeddings → Vector Search → Retrieval + Boosting + Trimming → LLM + Guardrails → Structured Answer

---

## 🧪 Example

### Question
Can owners rent out their homes?

### Answer (System Output)
Leasing appears to be permitted based on references to tenants/lessees,
but no explicit leasing rules were found in the governing text.

What the documents say:
- Owners may assign certain rights (such as voting rights) to a lessee,
  indicating leasing is contemplated.

Uncertainty:
- No dedicated leasing restrictions section was found.

---

## 🛠️ Installation

### 1. Clone repo
git clone <your-repo-url>
cd ai-hoa

### 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

---

## ▶️ Usage

Run the assistant:
python app.py arora

Example interaction:
You: do pet owners need to pick up dog waste?
You: can I lease my home?
You: what happens if I violate rules?

---

## 📂 Project Structure

ai-hoa/
├── app.py
├── data/
│   └── arora/
|   |__ milestone/
├── scripts/
├── README.md
├── requirements.txt

---

## ⚙️ Technologies Used

- Python
- sentence-transformers (MiniLM)
- FAISS
- Hugging Face Transformers
- PDF processing

---

## 🧠 Key Challenges & Solutions

Problem: Weak retrieval results  
Solution: scoring + boosting + trimming

Problem: Overconfident answers  
Solution: explicit vs inferred reasoning

Problem: Missing information  
Solution: fallback evidence + uncertainty messaging

Problem: Debugging AI behavior  
Solution: structured logging

---

## 📈 What I Learned

- Retrieval quality matters more than model choice
- Document structure is critical
- AI must distinguish known vs inferred vs unknown
- Logging is essential
- Handling missing data is critical

---

## 🚀 Future Improvements

- Web UI
- Multi-HOA support
- Better section extraction
- Confidence metrics

---

## 📌 Summary

This project demonstrates a production-style RAG system focused on:

accuracy, explainability, and reliability.
