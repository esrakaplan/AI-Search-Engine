"""
Örnek veri yükleme scripti.
Kendi verini buraya ekle veya texts listesini dışarıdan oku.

Çalıştır: python scripts/ingest.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from service import search_service

# ─── KENDİ METİNLERİNİ BURAYA EKLE ─────────────────────────────────── #
DOCUMENTS = [
    {
        "text": "FastAPI, Python 3.8+ için modern ve hızlı bir web framework'tür. Otomatik OpenAPI dokümantasyonu üretir.",
        "metadata": {"topic": "backend", "source": "docs"},
    },
    {
        "text": "ChromaDB açık kaynaklı bir vektör veritabanıdır. Embedding'leri depolar ve cosine similarity ile arama yapar.",
        "metadata": {"topic": "database", "source": "docs"},



    },
    {
        "text": "Ollama, LLM'leri lokal olarak çalıştırmak için kullanılan bir araçtır. llama3, mistral gibi modelleri destekler.",
        "metadata": {"topic": "llm", "source": "docs"},
    },
    {
        "text": "RAG (Retrieval-Augmented Generation), LLM'e dış bilgi kaynağı ekleyerek halüsinasyonu azaltır.",
        "metadata": {"topic": "ai", "source": "docs"},
    },
    {
        "text": "Semantic search, anahtar kelime yerine anlam benzerliğine göre arama yapar. Embedding vektörleri kullanır.",
        "metadata": {"topic": "ai", "source": "docs"},
    },
    {
        "text": "nomic-embed-text, Ollama üzerinde çalışan yüksek kaliteli bir embedding modelidir. 768 boyutlu vektör üretir.",
        "metadata": {"topic": "embedding", "source": "docs"},
    },
]
# ────────────────────────────────────────────────────────────────────── #


async def main():
    texts = [d["text"] for d in DOCUMENTS]
    metadatas = [d["metadata"] for d in DOCUMENTS]

    print(f"📥 {len(texts)} döküman yükleniyor...")
    result = await search_service.ingest(texts=texts, metadatas=metadatas)
    print(f"✅ {result['indexed']} döküman eklendi.")
    print(f"📦 Toplam DB'de: {result['total_in_db']}")


if __name__ == "__main__":
    asyncio.run(main())
