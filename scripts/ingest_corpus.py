"""
Ingests the chunked AI Act corpus into ChromaDB.

Run AFTER corpus/build_corpus.py:
  python scripts/ingest_corpus.py

This populates the vector store that the retriever uses at query time.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_act_corpus")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
CHUNKS_FILE = Path("corpus/chunks/ai_act_combined.json")
BATCH_SIZE = 50  # ChromaDB recommends batching large ingestions


def main():
    if not CHUNKS_FILE.exists():
        print(f"Chunks file not found: {CHUNKS_FILE}")
        print("Run: python corpus/build_corpus.py")
        sys.exit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection to allow re-ingestion
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"Deleted existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch ingestion
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "article": c.get("article", ""),
                    "section": c.get("section", ""),
                    "language": c.get("language", ""),
                }
                for c in batch
            ],
        )
        total += len(batch)
        print(f"  Ingested {total}/{len(chunks)} chunks...")

    print(f"\nDone. {total} chunks stored in ChromaDB at '{CHROMA_DB_PATH}'")
    print(f"Collection '{CHROMA_COLLECTION}' is ready for queries.")


if __name__ == "__main__":
    main()
