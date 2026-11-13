import pytest

from evals.judge import parse_verdict
from evals.run_evals import load_golden, relevant_ids
from rag.retriever import tokenize


def test_every_golden_evidence_string_is_in_the_corpus(retriever):
    """Guards the eval set: reference evidence must be quoted verbatim from the documents."""
    for case in load_golden():
        if case["type"] != "answerable":
            continue
        for ev in case["evidence"]:
            assert relevant_ids({"evidence": [ev]}, retriever.chunks), f"{case['id']}: {ev[:50]}"


def test_golden_ids_unique():
    ids = [c["id"] for c in load_golden()]
    assert len(ids) == len(set(ids))


def test_tokenize_keeps_subcategory_ids():
    assert "1.1" in tokenize("What does GOVERN 1.1 say?")
    assert tokenize("Risks") == ["risk"]


def test_bm25_finds_core_functions(retriever):
    hits = retriever.search("four functions of the AI RMF Core GOVERN MAP MEASURE MANAGE", k=5, mode="bm25")
    assert any("GOVERN" in h.chunk.text and "MANAGE" in h.chunk.text for h in hits)


def test_hybrid_falls_back_to_bm25_without_embeddings(retriever):
    assert not retriever.dense_available
    assert len(retriever.search("confabulation", k=3, mode="hybrid")) == 3


def test_judge_parsing():
    v = parse_verdict('Sure: {"correctness": 4, "faithfulness": "Yes", "reason": "ok"}')
    assert v == {"correctness": 4, "faithfulness": "yes", "reason": "ok"}
    with pytest.raises(ValueError):
        parse_verdict('{"correctness": 9, "faithfulness": "yes"}')
