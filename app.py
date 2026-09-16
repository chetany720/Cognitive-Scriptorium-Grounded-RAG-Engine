"""
KORTEX - Flask Server & RAG API
Authoritative document intelligence and grounded synthesis server.
"""

import os
import time
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from rag_engine import RAGEngine

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB limit

# Initialize RAG Engine
rag_engine = RAGEngine(chunk_size=450, chunk_overlap=75)


def preload_sample_documents():
    """Load default authoritative research corpus into the vector index."""
    if not os.path.exists(SAMPLE_DIR):
        return

    sample_files = [
        "retrieval_augmented_generation_principles.md",
        "transformer_attention_mechanisms.md",
        "autonomous_agentic_architectures.md"
    ]

    for filename in sample_files:
        filepath = os.path.join(SAMPLE_DIR, filename)
        if os.path.exists(filepath):
            try:
                content = rag_engine.extract_text_from_file(filepath, filename)
                rag_engine.add_document(filename, content, file_type="markdown")
                print(f"[KORTEX] Preloaded sample document: {filename}")
            except Exception as e:
                print(f"[KORTEX] Error loading sample document {filename}: {e}")


# Initialize corpus on startup
preload_sample_documents()


# -------------------------------------------------------------
# Web Page Route
# -------------------------------------------------------------

@app.route("/")
def index():
    """Renders the bespoke KORTEX Research Scriptorium interface."""
    return render_template("index.html")


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns vector store metrics, document registry, and hyperparameter state."""
    stats = rag_engine.get_stats()
    return jsonify({
        "success": True,
        "data": stats
    })


@app.route("/api/query", methods=["POST"])
def execute_query():
    """
    Executes end-to-end RAG pipeline:
    1. Hybrid retrieval (cosine + BM25 keyword boost)
    2. Grounded synthesis with citation anchors
    3. Latency & confidence telemetry
    """
    payload = request.get_json() or {}
    query_text = payload.get("query", "").strip()

    if not query_text:
        return jsonify({
            "success": False,
            "error": "Inquiry text cannot be empty."
        }), 400

    top_k = payload.get("top_k")
    threshold = payload.get("threshold")

    start_time = time.perf_counter()

    # Step 1: Retrieval
    retrieved_chunks = rag_engine.retrieve(query_text, top_k=top_k, threshold=threshold)
    retrieval_time = round((time.perf_counter() - start_time) * 1000, 2)

    # Step 2: Synthesis
    synth_start = time.perf_counter()
    synthesis_result = rag_engine.generate_grounded_answer(query_text, retrieved_chunks)
    synthesis_time = round((time.perf_counter() - synth_start) * 1000, 2)

    total_latency = round(retrieval_time + synthesis_time, 2)

    return jsonify({
        "success": True,
        "query": query_text,
        "answer": synthesis_result["answer"],
        "citations_used": synthesis_result.get("citations_used", []),
        "engine": synthesis_result.get("engine", "KORTEX Core"),
        "grounded": synthesis_result.get("grounded", False),
        "warning": synthesis_result.get("warning"),
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "retrieval_latency_ms": retrieval_time,
            "synthesis_latency_ms": synthesis_time,
            "total_latency_ms": total_latency,
            "chunks_matched": len(retrieved_chunks),
            "top_relevance_score": retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
        }
    })


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    """Handles file uploads (PDF, Markdown, Text, JSON, CSV)."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file payload found."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    raw_filename = file.filename
    clean_filename = secure_filename(raw_filename)
    if not clean_filename:
        clean_filename = f"upload_{int(time.time())}.txt"

    save_path = os.path.join(UPLOAD_DIR, clean_filename)
    file.save(save_path)

    try:
        content = rag_engine.extract_text_from_file(save_path, clean_filename)
        ext = os.path.splitext(clean_filename)[1].replace(".", "").lower() or "text"
        doc = rag_engine.add_document(clean_filename, content, file_type=ext)

        return jsonify({
            "success": True,
            "message": f"Successfully ingested and indexed '{clean_filename}'.",
            "document": doc.to_dict(),
            "stats": rag_engine.get_stats()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to ingest document: {str(e)}"
        }), 500


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Deletes an indexed document and recalibrates the vector store."""
    success = rag_engine.remove_document(doc_id)
    if success:
        return jsonify({
            "success": True,
            "message": f"Document {doc_id} purged from corpus.",
            "stats": rag_engine.get_stats()
        })
    return jsonify({"success": False, "error": "Document ID not found."}), 404


@app.route("/api/documents/<doc_id>/toggle", methods=["POST"])
def toggle_document(doc_id):
    """Toggles active participation of a document in retrieval queries."""
    payload = request.get_json() or {}
    active_state = payload.get("active")
    success = rag_engine.toggle_document(doc_id, active_state)
    if success:
        return jsonify({
            "success": True,
            "message": f"Document {doc_id} state toggled.",
            "stats": rag_engine.get_stats()
        })
    return jsonify({"success": False, "error": "Document ID not found."}), 404


@app.route("/api/documents/reset-sample", methods=["POST"])
def reset_sample_data():
    """Resets or loads curated sample papers into the corpus."""
    preload_sample_documents()
    return jsonify({
        "success": True,
        "message": "Curated research documents restored.",
        "stats": rag_engine.get_stats()
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Updates hyperparameter and frontier LLM API configurations."""
    payload = request.get_json() or {}

    provider = payload.get("provider", rag_engine.llm_settings["provider"])
    api_key = payload.get("api_key", rag_engine.llm_settings["api_key"])
    model = payload.get("model", rag_engine.llm_settings["model"])
    top_k = payload.get("top_k", rag_engine.llm_settings["top_k"])
    similarity_threshold = payload.get("similarity_threshold", rag_engine.llm_settings["similarity_threshold"])
    temperature = payload.get("temperature", rag_engine.llm_settings["temperature"])

    try:
        top_k = max(1, min(12, int(top_k)))
        similarity_threshold = max(0.01, min(0.9, float(similarity_threshold)))
        temperature = max(0.0, min(1.0, float(temperature)))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid hyperparameter values."}), 400

    rag_engine.llm_settings.update({
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "temperature": temperature
    })

    return jsonify({
        "success": True,
        "message": "Engine configuration updated.",
        "settings": {
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "temperature": temperature,
            "has_api_key": bool(api_key)
        }
    })


if __name__ == "__main__":
    print("[KORTEX] Starting Flask RAG Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
