"""
RAG retriever backed by ChromaDB.

Embeds the AI Act corpus (chunked by legal unit) and retrieves
the most relevant provisions for a given use case description.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_act_corpus")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)


class AIActRetriever:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def retrieve(self, query: str, n_results: int = 8) -> list[dict]:
        """
        Returns the top-n most relevant corpus chunks for the query.
        Each result dict has: id, text, metadata (article, section, language).
        """
        if not self.is_populated:
            raise RuntimeError(
                "ChromaDB collection is empty. Run: python scripts/ingest_corpus.py"
            )

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "article": meta.get("article", ""),
                    "section": meta.get("section", ""),
                    "language": meta.get("language", ""),
                    "relevance_score": round(1 - dist, 4),  # cosine → similarity
                }
            )

        return chunks

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """Formats retrieved chunks as a numbered list for inclusion in the LLM prompt."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            ref = chunk["article"] or chunk["section"] or "Unknown reference"
            lines.append(f"[{i}] {ref}\n{chunk['text']}\n")
        return "\n".join(lines)
