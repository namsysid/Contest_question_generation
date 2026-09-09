from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import error, request


DEFAULT_BASE_URL = "http://192.168.50.130:11434"


def resolve_base_url(explicit: Optional[str] = None) -> str:
    base = (explicit or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).strip()
    return base[:-3].rstrip("/") if base.endswith("/v1") else base.rstrip("/")


def generate_json(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 1200,
    json_schema: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    seed: Optional[int] = None,
    think: Optional[bool] = None,
) -> Dict[str, Any]:
    if timeout is None:
        timeout = float(os.getenv("OLLAMA_GENERATION_TIMEOUT_SECONDS", "900"))
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_output_tokens,
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    }
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": json_schema or "json",
        "options": options,
    }
    if model.casefold().startswith("qwen3"):
        if think is None:
            think_setting = os.getenv("OLLAMA_THINK", "true").strip().casefold()
            think = think_setting not in {"0", "false", "no", "off"}
        payload["think"] = think
    req = request.Request(
        f"{resolve_base_url(base_url)}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout:.0f}s") from exc
    raw = str(result.get("response") or "").strip()
    if not raw:
        raise ValueError("Ollama returned an empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Ollama did not return JSON: {raw}")
        return json.loads(raw[start : end + 1])


def embed_texts(
    model: str,
    texts: List[str],
    *,
    base_url: Optional[str] = None,
    timeout: float = 300.0,
) -> List[List[float]]:
    req = request.Request(
        f"{resolve_base_url(base_url)}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embedding request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama embedding request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama embedding request timed out after {timeout:.0f}s") from exc
    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError(f"Unexpected Ollama embedding response: {result}")
    return embeddings
