import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from service import search_service

DOCUMENTS = [
{
"text": "FastAPI is a modern and fast web framework for Python 3.8+. It automatically generates OpenAPI documentation.",
"metadata": {"topic": "backend", "source": "docs"},
},
{
"text": "ChromaDB is an open-source vector database. It stores embedments and searches using cosine similarity.",
"metadata": {"topic": "database", "source": "docs"},

},
{
"text": "Ollama is a tool used to run LLMs locally. llama3 supports models like mistral.",
"metadata": {"topic": "llm", "source": "docs"},
}, {
"text": "RAG (Retrieval-Augmented Generation) hallucinates by adding an external information source to the LLM reduces.",
"metadata": {"topic": "ai", "source": "docs"},
},
{
"text": "Semantic search searches based on semantic similarity rather than keywords. It uses embedding vectors.",
"metadata": {"topic": "ai", "source": "docs"},
}, {
"text": "nomic-embed-text is a high-quality embedding model that runs on Ollama. It produces 768-dimensional vectors.",
"metadata": {"topic": "embedding", "source": "docs"},
},
]


async def main():
    texts = [d["text"] for d in DOCUMENTS]
    metadatas = [d["metadata"] for d in DOCUMENTS]

    print(f"📥 {len(texts)} document is loading...")
    result = await search_service.ingest(texts=texts, metadatas=metadatas)
    print(f"✅ {result['indexed']} document is loaded.")
    print(f"📦 Total ib DB: {result['total_in_db']}")


if __name__ == "__main__":
    asyncio.run(main())
