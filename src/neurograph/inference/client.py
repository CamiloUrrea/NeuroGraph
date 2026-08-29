import json
import os
import urllib.error
import urllib.request

from pydantic import ValidationError

from neurograph.models.inference import GenerationResult, InferenceError

API_KEY_ENV_VAR = "NEUROGRAPH_GEMINI_API_KEY"
MODEL = "gemini-3.7-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
TIMEOUT_SECONDS = 15

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["answer", "citations"],
}


def generate(prompt: str) -> GenerationResult:
    api_key = _read_api_key()
    payload = _build_payload(prompt)
    raw_body = _send_request(api_key, payload)
    return _parse_generation_result(raw_body)


def _read_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR, "")
    if not api_key.strip():
        raise InferenceError(
            f"La variable de entorno {API_KEY_ENV_VAR} no está definida o está vacía."
        )
    return api_key


def _build_payload(prompt: str) -> dict:
    return {
        "model": MODEL,
        "input": prompt,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _RESPONSE_SCHEMA,
        },
    }


def _send_request(api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise InferenceError(f"Gemini devolvió un error HTTP {exc.code}.") from exc
    except TimeoutError as exc:
        raise InferenceError("La solicitud a Gemini superó el tiempo de espera.") from exc
    except urllib.error.URLError as exc:
        raise InferenceError(f"No se pudo conectar con Gemini: {exc.reason}.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceError("Gemini devolvió una respuesta que no es JSON válido.") from exc


def _parse_generation_result(raw_body: dict) -> GenerationResult:
    if not isinstance(raw_body, dict):
        raise InferenceError("La respuesta de Gemini no tiene el formato esperado.")

    status = raw_body.get("status")
    if status is not None and status != "completed":
        raise InferenceError(
            f"La interacción con Gemini no se completó (status={status!r})."
        )

    output_text = raw_body.get("output_text")
    if not isinstance(output_text, str) or output_text == "":
        raise InferenceError("La respuesta de Gemini no contiene 'output_text'.")

    try:
        structured = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise InferenceError(
            "El contenido de 'output_text' no es un JSON válido de Structured Outputs."
        ) from exc

    if not isinstance(structured, dict):
        raise InferenceError("La salida estructurada de Gemini no es un objeto JSON.")

    try:
        return GenerationResult.model_validate(structured)
    except ValidationError as exc:
        raise InferenceError(
            "La salida estructurada de Gemini no cumple el contrato GenerationResult."
        ) from exc
