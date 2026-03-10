from app.ollama_client import ollama
from app.vector_store import vector_store
from app.config import settings


SYSTEM_PROMPT = """Sen yardımcı bir arama asistanısın.
Verilen bağlam bilgilerini kullanarak soruyu yanıtla.
Eğer bağlamda cevap yoksa dürüstçe "Bu konuda bilgim bulunmuyor." de.
Asla uydurma. Yanıtını Türkçe ver."""


class SearchService:

    # ------------------------------------------------------------------ #
    # INGEST
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> dict:
        """Metinleri embed edip vector DB'ye kaydet."""
        if not texts:
            return {"indexed": 0}

        embeddings = await ollama.embed_batch(texts)
        ids = vector_store.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return {
            "indexed": len(ids),
            "ids": ids,
            "total_in_db": vector_store.count(),
        }

    # ------------------------------------------------------------------ #
    # SEARCH (saf semantic search — LLM yok)
    # ------------------------------------------------------------------ #
    async def search(self, query: str, top_k: int | None = None) -> dict:
        """Sorguya en yakın dökümanları getir."""
        k = top_k or settings.top_k
        query_emb = await ollama.embed(query)
        hits = vector_store.search(query_emb, top_k=k)
        return {
            "query": query,
            "results": hits,
            "count": len(hits),
        }

    # ------------------------------------------------------------------ #
    # ASK (RAG — search + LLM)
    # ------------------------------------------------------------------ #
    async def ask(self, question: str, top_k: int | None = None) -> dict:
        """Soru sor → ilgili dökümanları bul → LLM ile cevapla."""
        k = top_k or settings.top_k

        # 1. Semantic search
        query_emb = await ollama.embed(question)
        hits = vector_store.search(query_emb, top_k=k)

        if not hits:
            return {
                "question": question,
                "answer": "Veritabanında henüz döküman bulunmuyor.",
                "sources": [],
            }

        # 2. Bağlamı oluştur
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(f"[{i}] {hit['text']}")
        context = "\n\n".join(context_parts)

        # 3. Prompt
        prompt = f"""Bağlam:
{context}

Soru: {question}

Yanıt:"""

        # 4. LLM çağrısı
        answer = await ollama.generate(prompt=prompt, system=SYSTEM_PROMPT)

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": hits,
        }


# Singleton
search_service = SearchService()
