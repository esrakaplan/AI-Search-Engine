"""
Tüm endpointleri test et.
Çalıştır: python scripts/test_api.py
(Sunucunun ayakta olması gerekir: uvicorn app.main:app --reload)
"""
import asyncio
import httpx

BASE = "http://localhost:8000"


async def main():
    async with httpx.AsyncClient(timeout=120) as c:

        # ── 1. HEALTH ──────────────────────────────────────────────── #
        print("\n── HEALTH ────────────────────────────────")
        r = await c.get(f"{BASE}/health")
        print(r.json())

        # ── 2. INGEST ──────────────────────────────────────────────── #
        print("\n── INGEST ────────────────────────────────")
        r = await c.post(f"{BASE}/ingest", json={
            "texts": [
                "Python, 1991 yılında Guido van Rossum tarafından geliştirildi.",
                "Makine öğrenmesi, verilerden otomatik olarak öğrenen algoritmaları kapsar.",
            ],
            "metadatas": [
                {"source": "test", "topic": "python"},
                {"source": "test", "topic": "ml"},
            ]
        })
        print(r.json())

        # ── 3. SEARCH ──────────────────────────────────────────────── #
        print("\n── SEARCH ────────────────────────────────")
        r = await c.post(f"{BASE}/search", json={
            "query": "Python kim tarafından geliştirildi?",
            "top_k": 3
        })
        data = r.json()
        for hit in data["results"]:
            print(f"  [{hit['score']:.3f}] {hit['text'][:70]}...")

        # ── 4. ASK (RAG) ───────────────────────────────────────────── #
        print("\n── ASK / RAG ─────────────────────────────")
        r = await c.post(f"{BASE}/ask", json={
            "question": "Makine öğrenmesi nedir?",
            "top_k": 3
        })
        data = r.json()
        print(f"Soru   : {data['question']}")
        print(f"Cevap  : {data['answer']}")
        print(f"Kaynak : {len(data['sources'])} döküman kullanıldı")


if __name__ == "__main__":
    asyncio.run(main())
