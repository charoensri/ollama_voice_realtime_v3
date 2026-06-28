import os
import json
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
VECTOR_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 280
CHUNK_OVERLAP_WORDS = 60


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"\n[Page {page_num}]\n{page_text}")
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    return ""


def chunk_text(text: str, chunk_size_words=280, overlap_words=60):
    words = text.split()
    chunks = []
    if not words:
        return chunks
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def load_documents():
    all_chunks = []
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(exist_ok=True)
    supported_ext = {".pdf", ".txt", ".md", ".docx"}

    for file_path in docs_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_ext:
            continue
        print(f"Reading: {file_path}")
        try:
            text = extract_text(str(file_path))
        except Exception as e:
            print(f"Skipping {file_path}, error: {e}")
            continue
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": len(all_chunks),
                "source": str(file_path),
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


def build_faiss_index(chunks):
    if not chunks:
        raise ValueError("No document chunks found. Add files to the docs/ folder first.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    chunks = load_documents()
    print(f"Total chunks: {len(chunks)}")
    index = build_faiss_index(chunks)
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saving chunks metadata to {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("RAG ingest complete.")


if __name__ == "__main__":
    main()
