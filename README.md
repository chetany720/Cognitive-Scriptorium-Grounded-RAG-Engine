# KORTEX: Cognitive Scriptorium & Grounded RAG Engine

> **Intelligent Document Retrieval, Vector Reasoning & Verifiable Citation Synthesis**

KORTEX is a high-craft Retrieval-Augmented Generation (RAG) conversational intelligence platform built with **Python**, **Flask**, and a bespoke **Research Scriptorium** user interface.

Unlike generic chatbot interfaces, KORTEX treats document interaction with academic rigor: every synthesized assertion is anchored with interactive citation pills `[1]`, `[2]` that link directly to verbatim evidence passages, similarity score gauges, and highlighted source terms.

---

## 🏛️ Architectural Overview

```
KORTEX Workstation
├── app.py                     # Flask server & RESTful RAG API endpoints
├── rag_engine.py              # Hybrid vector retrieval & grounded synthesis core
├── sample_data/               # Preloaded authoritative technical corpora
│   ├── retrieval_augmented_generation_principles.md
│   ├── transformer_attention_mechanisms.md
│   └── autonomous_agentic_architectures.md
├── templates/
│   └── index.html             # Semantic HTML5 Three-Pillar Workstation layout
├── static/
│   ├── css/styles.css         # Archival ink & parchment design system
│   └── js/app.js              # Client telemetry, citations, and upload controller
└── uploads/                   # Runtime user-ingested documents
```

---

## ✨ Key Features

- **Hybrid Vector Retrieval**: Combines sublinear TF-IDF dense vector space modeling (`ngram_range=(1, 2)`) with BM25-inspired term frequency boosting for exact technical acronyms and domain terms.
- **Strict Citation Grounding**: In-line citations `[1]`, `[2]` link directly to the **Evidence & Provenance Inspector**, opening the exact source chunk, match percentage, and highlighted terms.
- **Dual Synthesis Pipeline**:
  - **Local Deterministic Grounded Core**: Operates 100% out-of-the-box with zero API keys required.
  - **Frontier LLM Integration**: Pluggable support for Google Gemini, OpenAI, or Groq via the in-app Settings drawer.
- **Multi-Format Ingestion**: Drag and drop `.pdf`, `.md`, `.txt`, `.json`, or `.csv` files for instantaneous chunking and indexing.
- **Corpus Shelf Controls**: Include or exclude individual documents from search scope in real time.
- **Bespoke Editorial Aesthetics**: Designed without generic AI templates—featuring deep archival ink tones (`#0C0E12`), warm parchment typography (`Newsreader` serif + `Plus Jakarta Sans`), and precision monospaced telemetry.

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- Dependencies:
  ```bash
  pip install flask scikit-learn numpy scipy pypdf werkzeug
  ```
  *(Optional for frontier LLM models: `pip install google-generativeai openai groq`)*

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/chetany720/kortex-rag.git
   cd kortex-rag
   ```

2. Start the application:
   ```bash
   python app.py
   ```

3. Open your browser:
   Navigate to `http://127.0.0.1:5000`

---

## ⚙️ Configuration & Settings

Configure RAG hyperparameters directly within the UI Settings drawer:
- **Retrieval Top-K**: Number of candidate chunks fetched (1–10).
- **Similarity Threshold**: Cutoff score for relevant context passages (0.02–0.50).
- **Temperature**: Synthesis creativity vs strict factual determinism (0.0–1.0).
- **Synthesis Provider**: Switch between Built-in Grounded Core, Google Gemini, OpenAI, or Groq.

---

## 📄 License

MIT License. Designed and built with craft.
