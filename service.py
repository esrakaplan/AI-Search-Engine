from ollama_client import ollama
from vector_store import vector_store
from config import settings


SYSTEM_PROMPT = """You are a helpful search assistant.
Answer the question using the given context information.
If there is no answer in the context, honestly say "I don't have information on this."
Never make things up."""

class SearchService:

    # ------------------------------------------------------------------ #
    # INGEST
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> dict:
        """Embed the text and save it to a vector DB."""
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
    # SEARCH (saf semantic search — no LLM)
    # ------------------------------------------------------------------ #
    async def search(self, query: str, top_k: int | None = None) -> dict:
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
        k = top_k or settings.top_k

        # 1. Semantic search
        query_emb = await ollama.embed(question)
        hits = vector_store.search(query_emb, top_k=k)

        if not hits:
            return {
                "question": question,
                "answer": "There are no documents in the database yet.",
                "sources": [],
            }

        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(f"[{i}] {hit['text']}")
        context = "\n\n".join(context_parts)

        # 3. Prompt
        prompt = f"""Context:
{context}

question: {question}

answer:"""

        answer = await ollama.generate(prompt=prompt, system=SYSTEM_PROMPT)

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": hits,
        }


search_service = SearchService()
