from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./docmind.db"
    upload_dir: str = "./uploads"

    # Milestone 2: local RAG
    embedding_model: str = "all-MiniLM-L6-v2"   # sentence-transformers model name
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"           # chat / answer generation
    retrieval_top_k: int = 5

    # Milestone 3: knowledge graph
    extraction_model: str = "qwen2.5:7b"        # entity/relationship extraction
    entity_merge_threshold: float = 0.87        # cosine sim to auto-merge two entity names

    # Figure understanding (opt-in per upload via ?describe_figures=true)
    vision_model: str = "granite3.2-vision"     # local VLM for describing charts/diagrams/images
    vision_dpi: int = 150                        # render resolution for pages sent to the VLM

    # Optional cloud providers. When a key is set the UI can switch chat /
    # extraction / vision from the local models to that provider.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"      # set ANTHROPIC_MODEL=claude-sonnet-5 for ~2.5x lower cost
    gemini_api_key: str = ""                     # free tier at aistudio.google.com
    gemini_model: str = "gemini-3.6-flash"
    llm_provider: str = "local"                 # startup default: "local" | "anthropic" | "gemini"

    # Seed a ready-made "DocMind demo.pdf" document + knowledge graph on startup
    # (only if it isn't already there) so the UI is testable without an upload+build.
    seed_demo: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
