from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings as cfg
from app.services.provider import available, get_provider, set_provider

router = APIRouter(prefix="/settings", tags=["settings"])


class ProviderUpdate(BaseModel):
    provider: str  # "local" | "anthropic" | "gemini"


@router.get("")
async def read_settings():
    return {
        "provider": get_provider(),
        "available": available(),
        "models": {
            "anthropic": cfg.anthropic_model,
            "gemini": cfg.gemini_model,
            "local": {
                "chat_model": cfg.ollama_model,
                "extraction_model": cfg.extraction_model,
                "vision_model": cfg.vision_model,
                "embedding_model": cfg.embedding_model,
            },
        },
    }


@router.put("")
async def update_settings(update: ProviderUpdate):
    try:
        set_provider(update.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"provider": get_provider()}
