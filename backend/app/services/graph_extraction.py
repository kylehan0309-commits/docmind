"""
Knowledge-graph extraction.

Pulls entities + relationships out of a chunk of text as strict JSON parsed into
Pydantic models. Local provider: Ollama structured outputs. Cloud provider:
Claude via `messages.parse(output_format=...)`.
"""
import json
import re
from typing import Literal

from ollama import Client
from pydantic import BaseModel

from app.config import settings
from app.services.provider import (
    anthropic_client,
    call_with_retry,
    gemini_client,
    get_provider,
)

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(host=settings.ollama_base_url)
    return _client


EntityType = Literal[
    "PERSON", "ORGANIZATION", "CONCEPT", "TECHNOLOGY", "LOCATION", "EVENT", "OTHER"
]


class ExtractedEntity(BaseModel):
    name: str
    type: EntityType


class ExtractedRelationship(BaseModel):
    source: str  # entity name; should match one listed in `entities`
    target: str  # entity name
    label: str   # short verb phrase, e.g. "acquired", "is a type of", "works at"


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


SYSTEM_PROMPT = (
    "You extract a knowledge graph from a passage of text. Identify the significant "
    "entities (people, organizations, concepts, technologies, locations, events) and "
    "the explicit relationships between them. Use ONLY information stated in the "
    "passage - do not invent entities or links. Every relationship 'source' and "
    "'target' must be a name you also listed in 'entities'. Keep relationship labels "
    "short (1-4 words). If the passage has no meaningful entities, return empty lists."
)


def normalize_name(name: str) -> str:
    """Dedup key for an entity: lowercase, punctuation -> space, collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def differ_only_by_number(a: str, b: str) -> bool:
    """True if two names differ only in numeric tokens (GPT-4 vs GPT-3, Q3 vs Q4).

    Used to veto fuzzy auto-merges: these embed very close but are distinct
    entities. Compared on normalized tokens.
    """
    ta, tb = set(normalize_name(a).split()), set(normalize_name(b).split())
    diff = ta ^ tb
    return bool(diff) and all(any(ch.isdigit() for ch in tok) for tok in diff)


def resolve_entity(
    name: str,
    vec,  # np.ndarray, normalized name embedding
    etype: str,
    known: list[tuple],  # (id, type, name, vec)
    threshold: float,
) -> str | None:
    """Return the id of an existing same-type entity whose name is a fuzzy match
    for `name` (cosine >= threshold), or None. Names differing only by a number
    are never matched.
    """
    best_id, best_sim = None, 0.0
    for ent_id, ent_type, ent_name, ent_vec in known:
        if ent_type != etype or differ_only_by_number(name, ent_name):
            continue
        sim = float(vec @ ent_vec)
        if sim > best_sim:
            best_id, best_sim = ent_id, sim
    return best_id if best_sim >= threshold else None


def extract_from_chunk(text: str) -> ChunkExtraction:
    """Run one chunk through the active extraction model. Blocking (sync HTTP call)."""
    provider = get_provider()

    if provider == "anthropic":
        resp = call_with_retry(
            anthropic_client().messages.parse,
            model=settings.anthropic_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=ChunkExtraction,
        )
        return resp.parsed_output

    if provider == "gemini":
        from google.genai import types

        resp = call_with_retry(
            gemini_client().models.generate_content,
            model=settings.gemini_model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ChunkExtraction,
                max_output_tokens=4096,
            ),
        )
        if isinstance(resp.parsed, ChunkExtraction):
            return resp.parsed
        return ChunkExtraction.model_validate_json(resp.text or "{}")

    resp = _get_client().chat(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        format=ChunkExtraction.model_json_schema(),
        options={"temperature": 0},
    )
    return ChunkExtraction.model_validate_json(resp["message"]["content"])


# --------------------------------------------------------------------------
# Entity consolidation: a second pass over the *whole* entity list to catch
# what the per-chunk embedding-similarity dedup can't - acronym <-> expansion
# ("RAG" / "retrieval-augmented generation") and other synonyms. One LLM call,
# not one per chunk, so it's cheap enough to run on demand.
# --------------------------------------------------------------------------
class ConsolidationGroup(BaseModel):
    keep_id: str            # the entity id to keep (most complete/formal name)
    merge_ids: list[str]    # entity ids that are the same thing, to fold into keep_id


class ConsolidationResult(BaseModel):
    groups: list[ConsolidationGroup]


CONSOLIDATION_SYSTEM_PROMPT = (
    "You are deduplicating a knowledge graph's entity list. You will be given a "
    "JSON list of entities, each with an id, name, and type. Group together "
    "entities that refer to the EXACT SAME real-world thing under a different "
    "name - most commonly an acronym and its expansion (e.g. \"RAG\" and "
    "\"retrieval-augmented generation\"), or a clear synonym or alternate "
    "spelling of the same name. Do NOT group entities that are merely related, "
    "similar, or in the same family but distinct - e.g. \"GPT-4\" and \"GPT-3\" "
    "are different versions, \"Canada\" and \"Toronto\" are different places. "
    "When in doubt, leave them separate. Only return groups with 2 or more "
    "entities; an entity with no duplicate should not appear in any group. For "
    "each group, set `keep_id` to whichever entity has the most complete or "
    "formal name, and `merge_ids` to the ids of the rest."
)


def consolidate_entities(entities: list[tuple[str, str, str]]) -> ConsolidationResult:
    """One LLM call over the full (id, name, type) entity list. Blocking."""
    payload = json.dumps([{"id": i, "name": n, "type": t} for i, n, t in entities])
    provider = get_provider()

    if provider == "anthropic":
        resp = call_with_retry(
            anthropic_client().messages.parse,
            model=settings.anthropic_model,
            max_tokens=4096,
            system=CONSOLIDATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
            output_format=ConsolidationResult,
        )
        return resp.parsed_output

    if provider == "gemini":
        from google.genai import types

        resp = call_with_retry(
            gemini_client().models.generate_content,
            model=settings.gemini_model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=CONSOLIDATION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ConsolidationResult,
                max_output_tokens=4096,
            ),
        )
        if isinstance(resp.parsed, ConsolidationResult):
            return resp.parsed
        return ConsolidationResult.model_validate_json(resp.text or "{}")

    resp = call_with_retry(
        _get_client().chat,
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        format=ConsolidationResult.model_json_schema(),
        options={"temperature": 0},
    )
    return ConsolidationResult.model_validate_json(resp["message"]["content"])
