import re

from app.services.pdf_parser import chunk_page_text

# A few paragraphs of clean, sentence-delimited prose to chunk.
PROSE = "\n\n".join(
    " ".join(f"Sentence {i}-{j} has some filler words to give it length." for j in range(6))
    for i in range(8)
)


def test_empty_or_blank_text_gives_no_chunks():
    assert chunk_page_text("") == []
    assert chunk_page_text("   \n\n  \t ") == []


def test_short_text_is_one_stripped_chunk():
    assert chunk_page_text("  just a little text.  ", chunk_size=800) == ["just a little text."]


def test_chunks_stay_within_budget_plus_overlap():
    size, overlap = 300, 60
    chunks = chunk_page_text(PROSE, chunk_size=size, overlap=overlap)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= size + overlap + 5  # small slack for join spaces


def test_chunks_do_not_break_mid_sentence():
    # every chunk but the last should end on sentence punctuation
    chunks = chunk_page_text(PROSE, chunk_size=300, overlap=60)
    for c in chunks[:-1]:
        assert c.rstrip()[-1] in ".!?"


def test_every_source_sentence_survives_in_some_chunk():
    chunks = chunk_page_text(PROSE, chunk_size=300, overlap=60)
    joined = " ".join(chunks)
    for sentence in re.split(r"(?<=[.!?])\s+", PROSE.replace("\n", " ")):
        if sentence.strip():
            assert sentence.strip() in joined


def test_consecutive_chunks_share_a_whole_sentence():
    chunks = chunk_page_text(PROSE, chunk_size=300, overlap=80)
    for prev, cur in zip(chunks, chunks[1:]):
        prev_sents = set(re.split(r"(?<=[.!?])\s+", prev))
        cur_sents = re.split(r"(?<=[.!?])\s+", cur)
        assert any(s in prev_sents for s in cur_sents), (prev, cur)


def test_no_overlap_when_overlap_is_zero():
    chunks = chunk_page_text(PROSE, chunk_size=300, overlap=0)
    all_sents = []
    for c in chunks:
        all_sents += re.split(r"(?<=[.!?])\s+", c)
    assert len(all_sents) == len(set(all_sents))  # every sentence appears once


def test_a_single_over_budget_sentence_is_hard_split_on_words():
    giant = "word " * 400  # 2000 chars, no sentence punctuation
    chunks = chunk_page_text(giant, chunk_size=200, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200
        assert "wor" not in c.split()[-1] or c.split()[-1] == "word"  # no split words


def test_cut_falls_on_the_paragraph_boundary_when_the_paragraph_fills_the_budget():
    p1 = " ".join(f"Alpha sentence {i} here." for i in range(8))
    p2 = " ".join(f"Beta sentence {i} here." for i in range(8))
    chunks = chunk_page_text(f"{p1}\n\n{p2}", chunk_size=len(p1) + 10, overlap=0)
    assert chunks[0] == p1  # broke exactly at the blank line, no bleed into p2
    assert chunks[1] == p2


def test_a_tiny_paragraph_is_packed_with_the_next_not_left_alone():
    text = "Short intro.\n\n" + " ".join(f"Body sentence {i} here." for i in range(6))
    chunks = chunk_page_text(text, chunk_size=400, overlap=0)
    assert chunks[0].startswith("Short intro. Body sentence 0")
