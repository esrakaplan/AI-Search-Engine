from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AskRequest, AskResponse,
    IngestRequest, IngestResponse,
    SearchRequest, SearchResponse,
    HealthResponse,
)
from service import search_service
from ollama_client import ollama
from vector_store import vector_store

app = FastAPI(
    title="AI Search Engine",
    description="Semantic search + RAG — FastAPI · ChromaDB · Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HEALTH ──────────────────────────────────────────────────────────────── #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    ollama_status = await ollama.health()
    return HealthResponse(
        status="ok" if ollama_status["status"] == "ok" else "degraded",
        ollama=ollama_status,
        documents_in_db=vector_store.count(),
    )


# ── INGEST ──────────────────────────────────────────────────────────────── #

@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest(body: IngestRequest):
    """Embed the text and save it to a vector DB."""
    if body.metadatas and len(body.metadatas) != len(body.texts):
        raise HTTPException(
            status_code=422,
            detail="The metadata length must be equal to the text length.",
        )
    try:
        result = await search_service.ingest(
            texts=body.texts,
            metadatas=body.metadatas,
        )
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ingest", tags=["Data"])
async def reset_db():
    vector_store.reset()
    return {"message": "The database has been reset.", "documents_in_db": 0}


# ── SEARCH ──────────────────────────────────────────────────────────────── #

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(body: SearchRequest):
    """Semantic search —  just retrieves similar documents."""
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="The database is empty. First, add data using /ingest.",
        )
    try:
        result = await search_service.search(
            query=body.query,
            top_k=body.top_k,
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ASK (RAG) ───────────────────────────────────────────────────────────── #

@app.post("/ask", response_model=AskResponse, tags=["Search"])
async def ask(body: AskRequest):
    """RAG — semantic search + answer with Ollama LLM"""
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="The database is empty. First, add data using /ingest.",
        )
    try:
        result = await search_service.ask(
            question=body.question,
            top_k=body.top_k,
        )
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
