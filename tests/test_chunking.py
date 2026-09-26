from hybridrag.chunking import chunk_corpus, chunk_text, dedupe_to_documents


def test_chunk_text_splits_on_chunk_size():
    text = " ".join(f"word{i}" for i in range(10))

    chunks = chunk_text(text, chunk_size=4)

    assert chunks == ["word0 word1 word2 word3", "word4 word5 word6 word7", "word8 word9"]


def test_chunk_text_shorter_than_chunk_size_stays_one_chunk():
    assert chunk_text("just a few words", chunk_size=100) == ["just a few words"]


def test_chunk_text_empty_string_yields_one_empty_chunk():
    assert chunk_text("", chunk_size=10) == [""]


def test_chunk_corpus_keys_and_titles():
    corpus = {"d1": {"title": "Title", "text": "one two three four five"}}

    chunked = chunk_corpus(corpus, chunk_size=2)

    assert set(chunked.keys()) == {"d1::chunk0", "d1::chunk1", "d1::chunk2"}
    assert all(doc["title"] == "Title" for doc in chunked.values())
    assert chunked["d1::chunk0"]["text"] == "one two"
    assert chunked["d1::chunk2"]["text"] == "five"


def test_dedupe_keeps_best_scoring_chunk_per_document():
    hits = [("d1::chunk0", 0.3), ("d1::chunk1", 0.9), ("d2::chunk0", 0.5)]

    result = dedupe_to_documents(hits, top_k=10)

    assert result == [("d1", 0.9), ("d2", 0.5)]


def test_dedupe_truncates_to_top_k():
    hits = [("d1::chunk0", 0.9), ("d2::chunk0", 0.5), ("d3::chunk0", 0.1)]

    result = dedupe_to_documents(hits, top_k=2)

    assert result == [("d1", 0.9), ("d2", 0.5)]
