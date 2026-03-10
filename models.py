from pydantic import BaseModel, Field


# ── REQUEST ────────────────────────────────────────────────────────────── #

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Text to be added")
    metadatas: list[dict] | None = Field(None, description="Metadata for each text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "FastAPI is a high-performance API development framework using Python.",
                    "ChromaDB is an open-source vector database."
                ],
                "metadatas": [
                    {"source": "docs", "topic": "backend"},
                    {"source": "docs", "topic": "database"},
                ],
            }
        }
    }


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="How many results should we get?")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "What is a vector database?", "top_k": 5}
        }
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Questions to ask LLM")
    top_k: int = Field(5, ge=1, le=20, description="How many resources should be used?")

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What are the features of ChromaDB?", "top_k": 3}
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
