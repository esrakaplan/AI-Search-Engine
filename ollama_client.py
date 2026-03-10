import httpx
from config import settings


class OllamaClient:

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.timeout = httpx.Timeout(120.0)

    # ------------------------------------------------------------------ #
    # EMBEDDING
    # ------------------------------------------------------------------ #
    async def embed(self, text: str) -> list[float]:
        """Convert text to vector."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple text files to vectors."""
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return embeddings

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    async def generate(self, prompt: str, system: str = "") -> str:
        """Receive response from LLM (streaming closed)."""
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            r.raise_for_status()
            return r.json()["response"]

    # ------------------------------------------------------------------ #
    # HEALTH
    # ------------------------------------------------------------------ #
    async def health(self) -> dict:
        """check Ollama health"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                return {"status": "ok", "models": models}
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# Singleton
ollama = OllamaClient()
