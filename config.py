from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    chroma_path: str = "./chroma_db"
    collection_name: str = "ai_search"
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
