import os
import json
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalFAISSRAG:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = None

    def load(self):
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"Missing FAISS index: {FAISS_INDEX_PATH}. Run python rag_ingest.py first."
            )
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                f"Missing chunks metadata: {CHUNKS_PATH}. Run python rag_ingest.py first."
            )
        print("Loading RAG embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"RAG ready. Chunks loaded: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 4) -> str:
        if self.model is None or self.index is None or self.chunks is None:
            self.load()
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            score = float(scores[0][rank - 1])
            results.append(
                f"[Result {rank}]\n"
                f"Source: {chunk['source']}\n"
                f"Chunk: {chunk['chunk_index']}\n"
                f"Score: {score:.4f}\n"
                f"Text:\n{chunk['text']}\n"
            )
        if not results:
            return "No relevant document chunks found."
        return "\n---\n".join(results)


rag_engine = LocalFAISSRAG()


def rag_search(query: str) -> str:
    try:
        return rag_engine.search(query=query, top_k=4)
    except Exception as e:
        return f"RAG search error: {e}"
