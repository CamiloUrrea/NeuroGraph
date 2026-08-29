import copy
from unittest.mock import patch

import pytest

from neurograph.inference import engine
from neurograph.models.document import Chunk
from neurograph.models.inference import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    Answer,
    GenerationResult,
    InferenceError,
    Source,
)


def make_chunk(index: int, document_id: str = "doc-1", source: str = "wiki") -> Chunk:
    return Chunk(
        id=f"{document_id}::{index}",
        document_id=document_id,
        chunk_index=index,
        text=f"Texto del chunk numero {index}.",
        source=source,
        uri=f"file:///{document_id}",
    )


def expected_source(chunk: Chunk) -> Source:
    return Source(
        document_id=chunk.document_id,
        uri=chunk.uri,
        source=chunk.source,
        chunk_index=chunk.chunk_index,
    )


def test_empty_context_returns_fallback_without_calling_client() -> None:
    with patch("neurograph.inference.engine.client.generate") as mock_generate:
        answer = engine.generate_answer("pregunta cualquiera", [])

    mock_generate.assert_not_called()
    assert answer == Answer(content=INSUFFICIENT_CONTEXT_MESSAGE, sources=[])


def test_valid_response_resolves_citations_to_correct_sources() -> None:
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    fake_result = GenerationResult(answer="respuesta generada", citations=[1, 3])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        answer = engine.generate_answer("pregunta", chunks)

    assert answer.content == "respuesta generada"
    assert answer.sources == [expected_source(chunks[0]), expected_source(chunks[2])]


def test_duplicate_citations_are_deduplicated_preserving_first_appearance_order() -> None:
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    fake_result = GenerationResult(answer="respuesta", citations=[1, 1, 2, 1])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        answer = engine.generate_answer("pregunta", chunks)

    assert answer.sources == [expected_source(chunks[0]), expected_source(chunks[1])]


def test_empty_citations_discards_answer_and_falls_back() -> None:
    chunks = [make_chunk(0)]
    fake_result = GenerationResult(answer="La respuesta parece correcta", citations=[])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        answer = engine.generate_answer("pregunta", chunks)

    assert answer == Answer(content=INSUFFICIENT_CONTEXT_MESSAGE, sources=[])


def test_citation_above_context_length_falls_back() -> None:
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    fake_result = GenerationResult(answer="respuesta", citations=[4])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        answer = engine.generate_answer("pregunta", chunks)

    assert answer == Answer(content=INSUFFICIENT_CONTEXT_MESSAGE, sources=[])


def test_citation_below_one_falls_back() -> None:
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    fake_result = GenerationResult(answer="respuesta", citations=[0])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        answer = engine.generate_answer("pregunta", chunks)

    assert answer == Answer(content=INSUFFICIENT_CONTEXT_MESSAGE, sources=[])


def test_inference_error_propagates_and_is_not_converted_to_fallback() -> None:
    chunks = [make_chunk(0)]

    with patch(
        "neurograph.inference.engine.client.generate",
        side_effect=InferenceError("429"),
    ):
        with pytest.raises(InferenceError):
            engine.generate_answer("pregunta", chunks)


def test_context_and_chunks_are_not_mutated() -> None:
    chunks = [make_chunk(0), make_chunk(1)]
    chunks_copy = copy.deepcopy(chunks)
    fake_result = GenerationResult(answer="respuesta", citations=[1])

    with patch("neurograph.inference.engine.client.generate", return_value=fake_result):
        engine.generate_answer("pregunta", chunks)

    assert chunks == chunks_copy


def test_prompt_contains_grounding_and_injection_protection_instructions() -> None:
    chunks = [make_chunk(0), make_chunk(1)]
    fake_result = GenerationResult(answer="respuesta", citations=[1])
    captured = {}

    def fake_generate(prompt: str) -> GenerationResult:
        captured["prompt"] = prompt
        return fake_result

    with patch("neurograph.inference.engine.client.generate", side_effect=fake_generate):
        engine.generate_answer("cual es la capital de Francia", chunks)

    prompt = captured["prompt"]
    assert "cual es la capital de Francia" in prompt
    assert "[Contexto 1]" in prompt
    assert chunks[0].text in prompt
    assert "[Contexto 2]" in prompt
    assert chunks[1].text in prompt
    assert "DATOS" in prompt
    assert "no instrucciones" in prompt
    assert INSUFFICIENT_CONTEXT_MESSAGE in prompt
