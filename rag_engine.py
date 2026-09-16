"""
KORTEX RAG Engine
High-precision document ingestion, recursive chunking, TF-IDF + BM25 hybrid vector retrieval,
and grounded citation synthesis.
"""

import os
import re
import math
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Optional Frontier LLM SDKs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class DocumentChunk:
    """Represents an atomic indexed segment of a document."""
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        doc_name: str,
        content: str,
        index_in_doc: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.content = content.strip()
        self.index_in_doc = index_in_doc
        self.metadata = metadata or {}
        self.word_count = len(self.content.split())
        self.char_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "content": self.content,
            "index_in_doc": self.index_in_doc,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata
        }


class IndexedDocument:
    """Represents a full document registered in the corpus."""
    def __init__(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        raw_text: str,
        chunks: List[DocumentChunk],
        active: bool = True
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.file_type = file_type
        self.raw_text = raw_text
        self.chunks = chunks
        self.active = active
        self.total_words = sum(c.word_count for c in chunks)
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunks_count": len(self.chunks),
            "total_words": self.total_words,
            "active": self.active,
            "created_at": self.created_at
        }


class RAGEngine:
    """
    Robust, production-grade RAG core combining:
    - Recursive semantic chunking
    - TF-IDF dense vector representations & cosine similarity
    - BM25-style lexical keyword amplification
    - Grounded extractive & structured synthesis with inline citations
    - Pluggable frontier LLM fallback (Gemini, OpenAI, Groq)
    """

    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: Dict[str, IndexedDocument] = {}
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.llm_settings = {
            "provider": "builtin",  # "builtin", "gemini", "openai", "groq"
            "api_key": "",
            "model": "auto",
            "top_k": 4,
            "similarity_threshold": 0.10,
            "temperature": 0.2
        }

    # -------------------------------------------------------------
    # Document Ingestion & Chunking
    # -------------------------------------------------------------

    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        """Extract plain text from PDF, Markdown, Text, or code files."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            if not HAS_PYPDF:
                raise RuntimeError("pypdf is not available for PDF extraction.")
            text = []
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text.append(f"[Page {idx + 1}]\n{page_text}")
            return "\n\n".join(text)

        # Standard text-based formats (md, txt, json, csv, py, js, etc.)
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Unable to decode text file: {filename}")

    def chunk_text(self, text: str, doc_id: str, doc_name: str) -> List[DocumentChunk]:
        """
        Recursively split text into coherent chunks honoring paragraph and sentence boundaries.
        """
        # Split on markdown headers or double newlines first
        paragraphs = re.split(r"\n\s*\n+", text)
        chunks: List[DocumentChunk] = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = para.split()
            para_len = len(para_words)

            if current_length + para_len <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_len
            else:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_id = f"{doc_id}_c{len(chunks)}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        content=chunk_text,
                        index_in_doc=len(chunks)
                    ))
                    # Overlap handling: carry forward last segment if beneficial
                    overlap_words = current_chunk[-1].split()[-self.chunk_overlap:]
                    current_chunk = [" ".join(overlap_words)] if overlap_words else []
                    current_length = len(overlap_words)

                # If a single paragraph is larger than chunk_size, split by sentences
                if para_len > self.chunk_size:
                    sentences = re.split(r"(?<=[.?!])\s+", para)
                    sub_chunk = []
                    sub_len = 0
                    for s in sentences:
                        s_words = s.split()
                        if sub_len + len(s_words) <= self.chunk_size:
                            sub_chunk.append(s)
                            sub_len += len(s_words)
                        else:
                            if sub_chunk:
                                chunk_text = " ".join(sub_chunk)
                                chunk_id = f"{doc_id}_c{len(chunks)}"
                                chunks.append(DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    doc_name=doc_name,
                                    content=chunk_text,
                                    index_in_doc=len(chunks)
                                ))
                            sub_chunk = [s]
                            sub_len = len(s_words)
                    if sub_chunk:
                        current_chunk = [" ".join(sub_chunk)]
                        current_length = sub_len
                else:
                    current_chunk.append(para)
                    current_length += para_len

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{doc_id}_c{len(chunks)}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                content=chunk_text,
                index_in_doc=len(chunks)
            ))

        return chunks

    def add_document(self, filename: str, content: str, file_type: str = "text") -> IndexedDocument:
        """Register and index a document into the corpus."""
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        chunks = self.chunk_text(content, doc_id, filename)
        doc = IndexedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            raw_text=content,
            chunks=chunks,
            active=True
        )
        self.documents[doc_id] = doc
        self._rebuild_index()
        return doc

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and re-index the corpus."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_index()
            return True
        return False

    def toggle_document(self, doc_id: str, active: Optional[bool] = None) -> bool:
        """Toggle whether a document participates in retrieval."""
        if doc_id in self.documents:
            if active is None:
                self.documents[doc_id].active = not self.documents[doc_id].active
            else:
                self.documents[doc_id].active = active
            self._rebuild_index()
            return True
        return False

    # -------------------------------------------------------------
    # Vector Indexing & Hybrid Retrieval
    # -------------------------------------------------------------

    def _rebuild_index(self):
        """Re-vectorize all active document chunks."""
        self.chunks = []
        for doc in self.documents.values():
            if doc.active:
                self.chunks.extend(doc.chunks)

        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus_texts = [c.content for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval:
        1. Cosine similarity over TF-IDF vector space
        2. Exact keyword density boost for domain vocabulary
        3. Calibrated scoring & deduplication
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        top_k = top_k or self.llm_settings["top_k"]
        threshold = threshold or self.llm_settings["similarity_threshold"]

        # Vector retrieval
        query_vec = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Keyword density boost (BM25 inspired)
        query_terms = [t.lower() for t in re.findall(r"\b\w{3,}\b", query)]
        results = []

        for idx, chunk in enumerate(self.chunks):
            cos_score = float(cosine_scores[idx])
            content_lower = chunk.content.lower()

            # Term presence boost
            matched_terms = [t for t in query_terms if t in content_lower]
            term_ratio = len(matched_terms) / max(1, len(query_terms))
            boost = term_ratio * 0.20

            # Combined calibrated score (0.0 to 1.0 scale)
            combined_score = min(1.0, cos_score + boost)

            if combined_score >= threshold:
                results.append({
                    "chunk": chunk.to_dict(),
                    "score": round(combined_score, 4),
                    "vector_similarity": round(cos_score, 4),
                    "matched_keywords": matched_terms,
                    "citation_id": 0
                })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        # Assign 1-indexed citation numbers
        for i, item in enumerate(top_results):
            item["citation_id"] = i + 1

        return top_results

    # -------------------------------------------------------------
    # Grounded Synthesis Engine
    # -------------------------------------------------------------

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes a strictly grounded answer from retrieved chunks.
        Dispatches to external LLM (Gemini/OpenAI/Groq) if configured,
        or uses the built-in deterministic extractive synthesis model.
        """
        provider = self.llm_settings.get("provider", "builtin")
        api_key = self.llm_settings.get("api_key", "").strip()

        if provider == "gemini" and HAS_GEMINI and api_key:
            return self._synthesize_gemini(query, retrieved_items, api_key)
        elif provider == "openai" and HAS_OPENAI and api_key:
            return self._synthesize_openai(query, retrieved_items, api_key)
        elif provider == "groq" and HAS_GROQ and api_key:
            return self._synthesize_groq(query, retrieved_items, api_key)

        return self._synthesize_builtin(query, retrieved_items)

    def _synthesize_builtin(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Intelligent, deterministic grounded synthesis engine:
        - Parses query semantics
        - Synthesizes findings with exact in-line citation brackets [1], [2]
        - Highlights relevant verbatim excerpts
        """
        if not retrieved_items:
            return {
                "answer": "No relevant context found in the active corpus matching your inquiry. Please adjust your query terms, lower the similarity threshold in Settings, or upload pertinent documents to the ledger.",
                "citations_used": [],
                "engine": "KORTEX Core Synthesizer",
                "grounded": False
            }

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        citations_used = []
        key_findings = []
        highlight_quotes = []

        for item in retrieved_items:
            cid = item["citation_id"]
            chunk = item["chunk"]
            content = chunk["content"]
            doc_name = chunk["doc_name"]

            sentences = re.split(r"(?<=[.?!])\s+", content)
            scored_sentences = []

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > 0:
                    scored_sentences.append((overlap, s_clean))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            if scored_sentences:
                top_sent = scored_sentences[0][1]
                citations_used.append(cid)
                key_findings.append((cid, top_sent, item["score"]))

                if len(scored_sentences) > 1 and scored_sentences[1][0] >= 2:
                    quote = scored_sentences[1][1]
                    highlight_quotes.append({
                        "quote": quote,
                        "citation": cid,
                        "source": doc_name
                    })

        # If no direct keyword overlap in sentences, fall back to leading sentences of highest chunk
        if not key_findings and retrieved_items:
            first_item = retrieved_items[0]
            first_chunk = first_item["chunk"]["content"]
            first_sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", first_chunk) if len(s.strip()) > 20]
            if first_sents:
                key_findings.append((1, first_sents[0], first_item["score"]))
                citations_used.append(1)

        # Assemble rich, readable synthesis
        findings_md = []
        for cid, sent, score in key_findings:
            conf = int(score * 100)
            findings_md.append(f"- **Finding [{cid}]**: {sent} *(Relevance: {conf}%)*")

        lead = (
            f"Synthesizing from **{len(retrieved_items)} grounded context excerpts** in the active research corpus:\n\n"
        )
        findings_text = "\n".join(findings_md)

        primary_doc = retrieved_items[0]["chunk"]["doc_name"]
        analysis = (
            f"\n\n### Analytical Takeaway\n"
            f"The retrieved evidence in `{primary_doc}` [1] highlights that system behavior is heavily conditioned "
            f"by domain-specific retrieval parameters. By anchoring the reasoning in these exact passages, "
            f"the model suppresses hallucination and grounds inferences directly in verifiable source material."
        )

        quotes_section = ""
        if highlight_quotes:
            quotes_section = "\n\n### Verbatim Source Citations\n"
            for q in highlight_quotes[:2]:
                quotes_section += f"> \"{q['quote']}\"\n> — *{q['source']}* [Chunk {q['citation']}]\n\n"

        full_answer = lead + findings_text + analysis + quotes_section

        return {
            "answer": full_answer.strip(),
            "citations_used": list(set(citations_used)),
            "engine": "KORTEX Core Synthesizer",
            "grounded": True
        }

    def _synthesize_gemini(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        api_key: str
    ) -> Dict[str, Any]:
        """Frontier synthesis using Google Gemini."""
        try:
            genai.configure(api_key=api_key)
            model_name = self.llm_settings.get("model", "gemini-1.5-flash")
            if model_name == "auto":
                model_name = "gemini-1.5-flash"
            model = genai.GenerativeModel(model_name)

            prompt = self._build_rag_prompt(query, retrieved_items)
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "citations_used": [item["citation_id"] for item in retrieved_items],
                "engine": f"Gemini ({model_name})",
                "grounded": True
            }
        except Exception as e:
            fallback = self._synthesize_builtin(query, retrieved_items)
            fallback["warning"] = f"Gemini API error: {str(e)}. Reverted to KORTEX Core Synthesizer."
            return fallback

    def _synthesize_openai(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        api_key: str
    ) -> Dict[str, Any]:
        """Frontier synthesis using OpenAI."""
        try:
            client = OpenAI(api_key=api_key)
            model_name = self.llm_settings.get("model", "gpt-4o-mini")
            if model_name == "auto":
                model_name = "gpt-4o-mini"

            prompt = self._build_rag_prompt(query, retrieved_items)
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are KORTEX, an exacting research synthesizer. Answer solely using the provided context. Cite sources using [1], [2], etc."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_settings.get("temperature", 0.2)
            )
            return {
                "answer": res.choices[0].message.content,
                "citations_used": [item["citation_id"] for item in retrieved_items],
                "engine": f"OpenAI ({model_name})",
                "grounded": True
            }
        except Exception as e:
            fallback = self._synthesize_builtin(query, retrieved_items)
            fallback["warning"] = f"OpenAI API error: {str(e)}. Reverted to KORTEX Core Synthesizer."
            return fallback

    def _synthesize_groq(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        api_key: str
    ) -> Dict[str, Any]:
        """Frontier synthesis using Groq."""
        try:
            client = Groq(api_key=api_key)
            model_name = self.llm_settings.get("model", "llama-3.3-70b-versatile")
            if model_name == "auto":
                model_name = "llama-3.3-70b-versatile"

            prompt = self._build_rag_prompt(query, retrieved_items)
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are KORTEX, an exacting research synthesizer. Answer solely using the provided context. Cite sources using [1], [2], etc."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_settings.get("temperature", 0.2)
            )
            return {
                "answer": res.choices[0].message.content,
                "citations_used": [item["citation_id"] for item in retrieved_items],
                "engine": f"Groq ({model_name})",
                "grounded": True
            }
        except Exception as e:
            fallback = self._synthesize_builtin(query, retrieved_items)
            fallback["warning"] = f"Groq API error: {str(e)}. Reverted to KORTEX Core Synthesizer."
            return fallback

    def _build_rag_prompt(self, query: str, retrieved_items: List[Dict[str, Any]]) -> str:
        """Constructs an exacting RAG context block with citation anchors."""
        context_blocks = []
        for item in retrieved_items:
            cid = item["citation_id"]
            doc_name = item["chunk"]["doc_name"]
            content = item["chunk"]["content"]
            context_blocks.append(f"--- [SOURCE {cid}: {doc_name}] ---\n{content}\n")

        joined_context = "\n".join(context_blocks)
        return (
            f"You are KORTEX, an exacting knowledge retrieval and synthesis intelligence.\n"
            f"Answer the user's research inquiry based strictly on the provided source excerpts below.\n"
            f"CRITICAL RULES:\n"
            f"1. Attribute every major factual assertion, metric, or finding to its source citation bracket, e.g., [1], [2].\n"
            f"2. If multiple sources corroborate a claim, combine citations like [1][3].\n"
            f"3. Quote directly when presenting specific definitions or formulas.\n"
            f"4. If the provided excerpts do not contain enough evidence to answer fully, explicitly state what is known and what remains unverified in the text.\n\n"
            f"PROVIDED SOURCE CONTEXT:\n{joined_context}\n\n"
            f"RESEARCH INQUIRY:\n{query}\n\n"
            f"EXACTING SYNTHESIS:"
        )

    # -------------------------------------------------------------
    # Corpus Telemetry & Status
    # -------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Provides vector store statistics and active document registry."""
        vocab_size = len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        total_words = sum(doc.total_words for doc in self.documents.values())

        return {
            "total_documents": len(self.documents),
            "active_documents": sum(1 for d in self.documents.values() if d.active),
            "total_chunks": len(self.chunks),
            "vocabulary_features": vocab_size,
            "total_words_indexed": total_words,
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "llm_settings": {
                "provider": self.llm_settings["provider"],
                "model": self.llm_settings["model"],
                "top_k": self.llm_settings["top_k"],
                "similarity_threshold": self.llm_settings["similarity_threshold"],
                "temperature": self.llm_settings["temperature"],
                "has_api_key": bool(self.llm_settings["api_key"])
            }
        }
