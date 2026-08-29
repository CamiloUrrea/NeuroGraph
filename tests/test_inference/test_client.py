import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from neurograph.inference import client
from neurograph.models.inference import GenerationResult, InferenceError


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def _response_body(answer: str = "respuesta generada", citations=(1, 2), status="completed") -> bytes:
    return json.dumps(
        {
            "id": "interaction-1",
            "status": status,
            "output_text": json.dumps({"answer": answer, "citations": list(citations)}),
        }
    ).encode("utf-8")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=client.ENDPOINT, code=code, msg="error", hdrs=None, fp=io.BytesIO(b"")
    )


def test_missing_api_key_raises_inference_error(monkeypatch) -> None:
    monkeypatch.delenv(client.API_KEY_ENV_VAR, raising=False)

    with pytest.raises(InferenceError):
        client.generate("prompt")


def test_blank_api_key_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "   ")

    with pytest.raises(InferenceError):
        client.generate("prompt")


def test_builds_correct_http_request(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret-key-123")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(_response_body())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.generate("mi prompt de prueba")

    request = captured["request"]
    assert request.full_url == client.ENDPOINT
    assert request.get_method() == "POST"
    assert captured["timeout"] == client.TIMEOUT_SECONDS

    payload = json.loads(request.data)
    assert payload["model"] == client.MODEL
    assert payload["input"] == "mi prompt de prueba"
    assert payload["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["answer", "citations"],
        },
    }


def test_api_key_header_is_sent_and_not_logged(monkeypatch, capsys) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "top-secret-value")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse(_response_body())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.generate("prompt")

    request = captured["request"]
    assert request.get_header("X-goog-api-key") == "top-secret-value"

    captured_output = capsys.readouterr()
    assert "top-secret-value" not in captured_output.out
    assert "top-secret-value" not in captured_output.err


def test_valid_response_returns_generation_result(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(_response_body(answer="la respuesta", citations=(1, 3))),
    ):
        result = client.generate("prompt")

    assert isinstance(result, GenerationResult)
    assert result.answer == "la respuesta"
    assert result.citations == [1, 3]


def test_invalid_top_level_json_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"not valid json{")):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_invalid_output_text_json_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")
    body = json.dumps({"status": "completed", "output_text": "not valid json{"}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_structured_output_missing_required_field_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")
    body = json.dumps(
        {"status": "completed", "output_text": json.dumps({"answer": "solo respuesta"})}
    ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_structured_output_wrong_type_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")
    body = json.dumps(
        {
            "status": "completed",
            "output_text": json.dumps({"answer": "respuesta", "citations": ["a", "b"]}),
        }
    ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_missing_output_text_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")
    body = json.dumps({"status": "completed"}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_http_429_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch("urllib.request.urlopen", side_effect=_http_error(429)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_http_500_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch("urllib.request.urlopen", side_effect=_http_error(500)):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_timeout_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(InferenceError):
            client.generate("prompt")


def test_connection_error_raises_inference_error(monkeypatch) -> None:
    monkeypatch.setenv(client.API_KEY_ENV_VAR, "secret")

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(InferenceError):
            client.generate("prompt")
