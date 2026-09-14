"""
PDF parsing service.

MVP uses PyMuPDF (fitz) for fast text extraction per page.
Swap in Docling later if you need better table/layout handling —
same interface (return list of {page_number, text}).

Optionally, pages that contain figures (embedded images or lots of vector
drawing) are rendered and passed to a local vision model, whose description is
appended to that page's text. Opt-in per call (`describe_figures=True`) because
it costs one VLM call per figure page.
"""
import re

import fitz  # PyMuPDF

from app.config import settings
from app.services.vision import describe_page

# A text-only page has a handful of vector ops (rules, table borders); a chart or
# diagram has many. Above this count we treat the page as carrying a figure.
_DRAWING_FIGURE_THRESHOLD = 15


def _page_has_figure(page: "fitz.Page") -> bool:
    return bool(page.get_images()) or len(page.get_drawings()) >= _DRAWING_FIGURE_THRESHOLD


def extract_pages(pdf_path: str, describe_figures: bool = False) -> list[dict]:
    """Return a list of {page_number, text} dicts, 1-indexed pages.

    When `describe_figures` is true and a vision model is configured, figure-bearing
    pages get a "[Figure description: ...]" block appended to their text.
    """
    use_vision = describe_figures and bool(settings.vision_model)
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if use_vision and _page_has_figure(page):
            png = page.get_pixmap(dpi=settings.vision_dpi).tobytes("png")
            description = describe_page(png)
            if description:
                text = f"{text}\n\n[Figure description (page {i + 1}): {description}]".strip()

        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


# Paragraph = one or more blank lines. Sentence end = . ! ? followed by
# whitespace and something that looks like the start of the next sentence
# (capital, digit, or an opening quote/bracket). Python's re only allows a
# fixed-width look-behind, hence the single-char class.
_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text)) if s]


def _hard_wrap(piece: str, limit: int) -> list[str]:
    """Last-resort split of an over-long sentence: on whitespace, then raw chars."""
    out: list[str] = []
    cur = ""
    for word in piece.split():
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)

    wrapped: list[str] = []
    for part in out:
        while len(part) > limit:  # a single token longer than the limit (URL, hash)
            wrapped.append(part[:limit])
            part = part[limit:]
        if part:
            wrapped.append(part)
    return wrapped


def _atoms(text: str, limit: int) -> list[str]:
    """Break text into pieces no longer than `limit`, cutting on paragraph, then
    sentence, then (rarely) word boundaries — never mid-word."""
    atoms: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            atoms.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= limit:
                atoms.append(sent)
            else:
                atoms.extend(_hard_wrap(sent, limit))
    return atoms


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Trailing whole sentences of `chunk` totalling <= `overlap` chars, so the
    next chunk starts with the tail end of this one's context."""
    picked: list[str] = []
    total = 0
    for sent in reversed(_split_sentences(chunk)):
        if picked and total + 1 + len(sent) > overlap:
            break
        picked.insert(0, sent)
        total += len(sent) + (1 if len(picked) > 1 else 0)
    if picked:
        return " ".join(picked)
    return chunk[-overlap:].lstrip()  # one sentence already longer than `overlap`


def chunk_page_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split page text into overlapping chunks that respect natural boundaries.

    Greedily packs whole paragraphs — then whole sentences, for a paragraph
    longer than the budget — up to ``chunk_size`` characters, so a chunk ends
    mid-sentence only when a single sentence is itself over budget. Consecutive
    chunks share the last ~``overlap`` characters of whole sentences from the
    previous chunk, preserving context across the cut. A chunk can therefore run
    up to roughly ``chunk_size + overlap`` characters.

    Sized in characters (cheap, deterministic, no tokenizer dependency); the
    default 1000 keeps typical prose comfortably inside all-MiniLM-L6-v2's
    256-token input window.
    """
    text = (text or "").strip()
    if not text:
        return []
    overlap = max(0, min(overlap, chunk_size // 2))

    atoms = _atoms(text, chunk_size)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [], 0

    for atom in atoms:
        if cur and cur_len + 1 + len(atom) > chunk_size:
            flush()
            if overlap and chunks:
                tail = _overlap_tail(chunks[-1], overlap)
                if tail:
                    cur, cur_len = [tail], len(tail)
        cur.append(atom)
        cur_len += len(atom) + (1 if len(cur) > 1 else 0)
    flush()
    return chunks
