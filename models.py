from pydantic import BaseModel, Field


# ── REQUEST ────────────────────────────────────────────────────────────── #

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Eklenecek metinler")
    metadatas: list[dict] | None = Field(None, description="Her metin için metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "FastAPI, Python ile yüksek performanslı API geliştirme framework'üdür.",
                    "ChromaDB açık kaynaklı bir vektör veritabanıdır.",
                ],
                "metadatas": [
                    {"source": "docs", "topic": "backend"},
                    {"source": "docs", "topic": "database"},
                ],
            }
        }
    }


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Arama sorgusu")
    top_k: int = Field(5, ge=1, le=20, description="Kaç sonuç dönsün")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "vektör veritabanı nedir", "top_k": 5}
        }
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="LLM'e sorulacak soru")
    top_k: int = Field(5, ge=1, le=20, description="Kaç kaynak kullanılsın")

    model_config = {
        "json_schema_extra": {
            "example": {"question": "ChromaDB'nin özellikleri nelerdir?", "top_k": 3}
        }
    }


# ── RESPONSE ───────────────────────────────────────────────────────────── #

class SearchHit(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float


class IngestResponse(BaseModel):
    indexed: int
    ids: list[str]
    total_in_db: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    count: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    ollama: dict
    documents_in_db: int
