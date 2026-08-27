import copy

from neurograph.models.document import Chunk
from neurograph.retrieval.context import select_context


def make_chunk(chunk_id: str, text: str, chunk_index: int = 0, document_id: str = "doc-1") -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, chunk_index=chunk_index, text=text)


def test_normal_selection_returns_all_chunks_within_budget_and_threshold() -> None:
    results = [
        (make_chunk("a", "texto uno", 0), 0.1),
        (make_chunk("b", "texto dos", 1), 0.2),
        (make_chunk("c", "texto tres", 2), 0.3),
    ]

    selected = select_context(results, max_distance=0.5, max_chars=1000)

    assert [c.id for c in selected] == ["a", "b", "c"]


def test_discards_results_worse_than_max_distance() -> None:
    results = [
        (make_chunk("a", "cerca", 0), 0.1),
        (make_chunk("b", "lejos", 1), 0.9),
        (make_chunk("c", "medio", 2), 0.4),
    ]

    selected = select_context(results, max_distance=0.5, max_chars=1000)

    assert [c.id for c in selected] == ["a", "c"]


def test_no_result_within_threshold_returns_empty_list() -> None:
    results = [
        (make_chunk("a", "uno", 0), 0.8),
        (make_chunk("b", "dos", 1), 0.9),
    ]

    selected = select_context(results, max_distance=0.5, max_chars=1000)

    assert selected == []


def test_respects_max_chars_budget() -> None:
    results = [
        (make_chunk("a", "x" * 100, 0), 0.1),
        (make_chunk("b", "y" * 100, 1), 0.2),
        (make_chunk("c", "z" * 100, 2), 0.3),
    ]

    selected = select_context(results, max_distance=1.0, max_chars=150)

    assert [c.id for c in selected] == ["a"]
    assert sum(len(c.text) for c in selected) <= 150


def test_no_chunk_is_truncated() -> None:
    oversized_text = "w" * 500
    results = [(make_chunk("a", oversized_text, 0), 0.1)]

    selected = select_context(results, max_distance=1.0, max_chars=100)

    assert selected == []


def test_preserves_original_relevance_order() -> None:
    results = [
        (make_chunk("c", "texto c", 2), 0.3),
        (make_chunk("a", "texto a", 0), 0.1),
        (make_chunk("b", "texto b", 1), 0.2),
    ]

    selected = select_context(results, max_distance=1.0, max_chars=1000)

    assert [c.id for c in selected] == ["c", "a", "b"]


def test_empty_results_returns_empty_list() -> None:
    assert select_context([], max_distance=0.5, max_chars=1000) == []


def test_selection_exactly_at_max_chars_limit_is_included() -> None:
    results = [
        (make_chunk("a", "x" * 50, 0), 0.1),
        (make_chunk("b", "y" * 50, 1), 0.2),
    ]

    selected = select_context(results, max_distance=1.0, max_chars=100)

    assert [c.id for c in selected] == ["a", "b"]
    assert sum(len(c.text) for c in selected) == 100


def test_chunk_that_does_not_fit_is_skipped_and_next_chunk_is_still_considered() -> None:
    results = [
        (make_chunk("a", "x" * 90, 0), 0.1),
        (make_chunk("b", "y" * 20, 1), 0.2),
        (make_chunk("c", "z" * 5, 2), 0.3),
    ]

    selected = select_context(results, max_distance=1.0, max_chars=100)

    assert [c.id for c in selected] == ["a", "c"]


def test_does_not_mutate_original_results() -> None:
    results = [
        (make_chunk("a", "texto uno", 0), 0.1),
        (make_chunk("b", "texto dos", 1), 0.2),
    ]
    results_copy = copy.deepcopy(results)

    select_context(results, max_distance=0.5, max_chars=1000)

    assert results == results_copy


def test_determinism() -> None:
    results = [
        (make_chunk("a", "texto uno", 0), 0.1),
        (make_chunk("b", "texto dos", 1), 0.2),
        (make_chunk("c", "texto tres", 2), 0.9),
    ]

    first = select_context(results, max_distance=0.5, max_chars=15)
    second = select_context(results, max_distance=0.5, max_chars=15)

    assert first == second
