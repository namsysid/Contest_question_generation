from __future__ import annotations

import json
import os
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import load_dotenv

try:
    from ..ollama_client import embed_texts as ollama_embed_texts
    from ..ollama_client import generate_json as ollama_generate_json
except ImportError:  # Scripts executed directly from src/ import circuit_lab as top-level.
    from ollama_client import embed_texts as ollama_embed_texts
    from ollama_client import generate_json as ollama_generate_json

load_dotenv()


def _ssl_context() -> ssl.SSLContext:
    ca_bundle = os.getenv("SSL_CERT_FILE", "").strip()
    if not ca_bundle:
        try:
            import certifi
        except ImportError:
            certifi = None
        if certifi is not None:
            ca_bundle = certifi.where()
    return ssl.create_default_context(cafile=ca_bundle or None)


def _log_usage(endpoint: str, model: str, body: dict[str, Any]) -> None:
    path = os.getenv("OPENAI_USAGE_LOG", "").strip()
    if not path:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "status": body.get("status"),
        "usage": body.get("usage") or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _post(endpoint: str, payload: dict[str, Any], timeout: float = 75.0) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env or the environment")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = request.Request(f"{base}/{endpoint}", data=json.dumps(payload).encode(),
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout, context=_ssl_context()) as response:
            return json.loads(response.read().decode())
    except error.HTTPError as exc:
        raise RuntimeError(f"OpenAI request failed ({exc.code}): {exc.read().decode(errors='replace')}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("OpenAI request timed out") from exc


def _responses_output_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts).strip()


def _parse_json_text(raw_text: str, source: str) -> dict[str, Any]:
    """Accept strict JSON and compatibility servers that wrap it in a JSON code fence."""
    raw_text = str(raw_text or "").strip()
    try:
        return json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"{source} returned invalid JSON text: {raw_text[:500]}")


def generate_json(model: str, prompt: str, *, provider: str, system: str = "", temperature: float = 0.0,
                  max_output_tokens: int = 1200, seed: int | None = None,
                  reasoning_effort: str | None = None,
                  json_schema: dict[str, Any] | None = None,
                  ollama_think: bool | None = None) -> dict[str, Any]:
    if provider == "ollama":
        return ollama_generate_json(model, prompt, system=system, temperature=temperature,
                                    max_output_tokens=max_output_tokens, seed=seed,
                                    json_schema=json_schema, think=ollama_think)
    # GPT-5 and o-series models reject non-default temperature values. Some older
    # callers do not set reasoning_effort, so route them through Responses with a
    # conservative default instead of falling through to Chat Completions.
    if reasoning_effort is None and model.startswith(("gpt-5", "o1", "o3", "o4")):
        reasoning_effort = "low"
    if reasoning_effort:
        response_format: dict[str, Any] = {"type": "json_object"}
        if json_schema:
            response_format = {
                "type": "json_schema", "name": "contest_pipeline_response",
                "strict": True, "schema": json_schema,
            }
        body = _post("responses", {
            "model": model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
            "text": {"format": response_format, "verbosity": "low"},
        }, timeout=float(os.getenv("OPENAI_REASONING_TIMEOUT_SECONDS", "900")))
        _log_usage("responses", model, body)
        raw_text = _responses_output_text(body)
        if body.get("status") == "incomplete":
            details = body.get("incomplete_details") or {}
            usage = body.get("usage") or {}
            raise ValueError(
                "OpenAI Responses API output was incomplete "
                f"(reason={details.get('reason')}, output_tokens={usage.get('output_tokens')})"
            )
        if not raw_text:
            details = body.get("incomplete_details") or {}
            usage = body.get("usage") or {}
            raise ValueError(
                "OpenAI Responses API produced no JSON "
                f"(status={body.get('status')}, reason={details.get('reason')}, "
                f"output_tokens={usage.get('output_tokens')})"
            )
        return _parse_json_text(raw_text, "OpenAI Responses API")
    chat_format: dict[str, Any] = {"type": "json_object"}
    if json_schema:
        chat_format = {"type": "json_schema", "json_schema": {
            "name": "contest_pipeline_response", "strict": True, "schema": json_schema,
        }}
    body = _post("chat/completions", {"model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_completion_tokens": max_output_tokens, "temperature": temperature,
        "response_format": chat_format})
    _log_usage("chat/completions", model, body)
    try:
        return _parse_json_text(body["choices"][0]["message"]["content"], "OpenAI chat API")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenAI returned invalid JSON: {body}") from exc


def embed_texts(model: str, texts: list[str], *, provider: str) -> list[list[float]]:
    if provider == "ollama":
        return ollama_embed_texts(model, texts)
    body = _post("embeddings", {"model": model, "input": texts, "encoding_format": "float"})
    _log_usage("embeddings", model, body)
    try:
        return [row["embedding"] for row in sorted(body["data"], key=lambda row: row["index"])]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"OpenAI returned invalid embeddings: {body}") from exc
