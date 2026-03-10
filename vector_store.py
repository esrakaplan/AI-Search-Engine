import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


class VectorStore:
    """ChromaDB ile tüm vektör işlemleri."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    # ------------------------------------------------------------------ #
    # EKLEME
    # ------------------------------------------------------------------ #
    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Dökümanları ve vektörlerini kaydet."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    # ------------------------------------------------------------------ #
    # ARAMA
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Verilen vektöre en yakın dökümanları getir."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        hits = []
        for i, doc in enumerate(results["documents"][0]):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "score": round(1 - results["distances"][0][i], 4),  # cosine → similarity
                }
            )
        return hits

    # ------------------------------------------------------------------ #
    # SİLME / BİLGİ
    # ------------------------------------------------------------------ #
    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Koleksiyonu tamamen temizle."""
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# Singleton
vector_store = VectorStore()
