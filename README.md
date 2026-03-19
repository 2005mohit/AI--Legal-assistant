# ⚖️ AI Legal Assistant
> Pretrained Models + Embeddings + RAG (Retrieval-Augmented Generation)

![Dashboard View](DashboardView.png)

A live AI system that automatically analyzes, summarizes, and extracts key information from legal documents. Upload any legal document (PDF/DOCX/XLSX) and ask questions in plain English.

🔗 **Live Demo:**https://ai--legal-assistant-vir32drdsmcrvmmnwuznwy.streamlit.app/

---

##  What This System Does

-  Summarizes legal documents (contracts, NDAs, agreements)
-  Extracts obligations of each party
-  Identifies key clauses — termination, payment, confidentiality, liability
-  Answers specific questions about uploaded documents
-  Answers general legal questions from model knowledge

---

##  Dataset

- **Type:** Real legal documents collected from SEC EDGAR public filings
- **Format:** PDF
- **Size:** 24 documents
- **Examples:**
  - Corporate agreements (10-K, 8-K, EX filings)
  - Affiliate program agreements
  - Software publisher contracts
  - Distribution agreements
- **No labeled data required** — this is not a training task

---

##  System Architecture

```
Legal Documents (PDF/DOCX)
        ↓
  Text Extraction          ← pdfplumber, python-docx
        ↓
  Text Cleaning            ← remove noise, keep structure
        ↓
  Chunking                 ← 500 char chunks with overlap
        ↓
  Embedding Generation     ← all-MiniLM-L6-v2 (Model 1)
        ↓
  Vector Database          ← FAISS index
        ↓
  Retrieval (RAG)          ← top-4 relevant chunks
        ↓
  LLM Summarization        ← llama-3.3-70b via Groq (Model 2)
        ↓
  Final Answer             ← structured summary / Q&A
        ↓
  Streamlit UI             ← deployed on Streamlit Cloud
```

---

##  Step-by-Step Breakdown

### Step 1: Dataset Collection
- Collected 24 real legal PDFs from SEC EDGAR public filings
- Document types: corporate agreements, affiliate contracts, distribution agreements
- No labeling needed — system uses pretrained models

### Step 2: Text Extraction
- **Tool:** `pdfplumber` for PDFs, `python-docx` for DOCX, `pandas` for XLSX
- Converts raw files into plain text
- Each document extracted page by page

### Step 3: Data Cleaning & Structuring
- Removed page numbers, headers, footers
- Preserved section titles and clause numbers
- Legal structure kept intact for accurate retrieval

### Step 4: Chunking
- Each document split into **500 character chunks**
- Chunking ensures text fits within model context limits
- Applied to both indexed documents and uploaded documents at runtime

### Step 5: Embedding Generation — Model 1
- **Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- Converts each text chunk into a 384-dimensional vector
- Captures semantic meaning of legal text
- Enables similarity-based search

### Step 6: Vector Database
- **Tool:** FAISS (Facebook AI Similarity Search)
- All chunk embeddings stored in `model/index.faiss`
- Chunk metadata (original text) stored in `model/metadata.pkl`
- Enables fast nearest-neighbor search

### Step 7: Retrieval Logic (RAG)
- User query converted to embedding using same `all-MiniLM-L6-v2` model
- FAISS searches for top-4 most semantically similar chunks
- Only relevant clauses passed to LLM — reduces hallucination

### Step 8: LLM Summarization — Model 2
- **Model:** `llama-3.3-70b-versatile` via **Groq API** (free)
- Retrieved chunks passed as context to LLM
- LLM generates structured summaries, extracts obligations, answers questions
- Temperature: 0.1 (precise, factual responses)

### Step 9: Final Output
The system delivers:
-  Executive summary of the document
-  Obligations of each party
-  Key clauses (payment, termination, confidentiality, liability)
-  Governing law and jurisdiction
-  Direct answers to specific legal questions

### Step 10: API Layer
- **Framework:** Streamlit
- Accepts document upload (PDF/DOCX/XLSX)
- Returns structured natural language responses
- Chat interface for multi-turn Q&A

### Step 11: Deployment
- **Platform:** Streamlit Cloud
- Deployed directly from GitHub repository
- FAISS index and metadata files stored in `model/` folder
- API key managed via Streamlit Secrets

---


##  Project Structure

```
AI-Based-Legal-Document-Summarizer/
├── model/
│   ├── index.faiss        ← FAISS vector index
│   └── metadata.pkl       ← chunk text metadata
├── app.py                 ← main Streamlit application
├── requirements.txt       ← Python dependencies
└── README.md
```

---


