from __future__ import annotations

import pytest

from docs_chat.chunking import chunk_page, chunk_pages
from docs_chat.loaders import Page


def _page(text: str, page_no: int = 1, doc: str = "doc.pdf") -> Page:
    return Page(doc_name=doc, page_no=page_no, text=text)


def test_short_page_is_one_chunk():
    chunks = chunk_page(_page("A short paragraph about leave policy."), chunk_words=50, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc.pdf:1:0"
    assert chunks[0].page_no == 1
    assert chunks[0].section_hint.startswith("A short paragraph")


def test_empty_page_produces_no_chunks():
    assert chunk_page(_page("   \n  ")) == []
    assert chunk_page(_page("ok")) == []  # below MIN_CHARS_PER_PAGE → treated as empty


def test_chunks_never_cross_page_boundaries():
    pages = [_page("one " * 300, page_no=1), _page("two " * 300, page_no=2)]
    chunks = chunk_pages(pages, chunk_words=100, overlap_words=20)
    for c in chunks:
        words = set(c.text.split())
        assert words == {"one"} or words == {"two"}, "a chunk mixed text from two pages"
    assert {c.page_no for c in chunks} == {1, 2}


def test_paragraphs_are_kept_together_when_they_fit():
    text = "First paragraph about salaries.\n\nSecond paragraph about expenses.\n\nThird paragraph about leave."
    chunks = chunk_page(_page(text), chunk_words=250, overlap_words=50)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0].text


def test_overlap_carries_tail_of_previous_chunk():
    paras = [f"Paragraph {i} " + " ".join(f"w{i}_{j}" for j in range(40)) for i in range(6)]
    chunks = chunk_page(_page("\n\n".join(paras)), chunk_words=100, overlap_words=10)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-10:]
    assert chunks[1].text.split()[:10] == first_tail


def test_chunk_size_is_respected_for_normal_text():
    paras = ["Sentence number %d is here. " % i * 5 for i in range(40)]
    chunks = chunk_page(_page("\n\n".join(paras)), chunk_words=120, overlap_words=20)
    # body ≤ chunk_words, plus at most overlap_words carried over
    assert all(c.word_count <= 120 + 20 for c in chunks)
    assert len(chunks) > 1


def test_oversized_paragraph_is_split_at_sentences():
    sentences = ["This is sentence %d about something." % i for i in range(60)]
    chunks = chunk_page(_page(" ".join(sentences)), chunk_words=60, overlap_words=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.endswith(".")  # sentence boundary, not mid-sentence


def test_single_huge_sentence_is_split_by_words():
    chunks = chunk_page(_page("word " * 500), chunk_words=100, overlap_words=0)
    assert len(chunks) == 5
    assert all(c.word_count == 100 for c in chunks)


def test_turkish_sentences_split():
    text = "Çalışanlar yılda yirmi dört gün izin alır. İzinler ayda iki gün birikir. Şirket her yıl maaşları gözden geçirir."
    chunks = chunk_page(_page(text), chunk_words=8, overlap_words=0)
    assert len(chunks) == 3


@pytest.mark.parametrize("chunk_words,overlap", [(0, 0), (50, 50), (50, -1)])
def test_invalid_parameters_raise(chunk_words, overlap):
    with pytest.raises(ValueError):
        chunk_page(_page("some text that is long enough"), chunk_words=chunk_words, overlap_words=overlap)
