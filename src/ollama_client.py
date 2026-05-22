from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import error, request


DEFAULT_BASE_URL = "http://192.168.50.186:11434"


def resolve_base_url(explicit: Optional[str] = None) -> str:
    base = (
        explicit
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc
    return json.loads(raw)


def parse_json_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty content")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i = raw.find("{")
        j = raw.rfind("}")
        if i < 0 or j < 0 or j <= i:
            raise ValueError(f"Model did not return JSON. Output was:\n{raw}")
        return json.loads(raw[i : j + 1])


def generate_text(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    seed: Optional[int] = None,
    format_json: bool = False,
    base_url: Optional[str] = None,
    timeout: float = 180.0,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if seed is not None:
        payload["options"]["seed"] = seed
    if format_json:
        payload["format"] = "json"

    data = _post_json(f"{resolve_base_url(base_url)}/api/generate", payload, timeout)
    text = data.get("response")
    if not isinstance(text, str):
        raise ValueError(f"Unexpected Ollama generate response: {data}")
    return text


def generate_json(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    seed: Optional[int] = None,
    base_url: Optional[str] = None,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    return parse_json_text(
        generate_text(
            model,
            prompt,
            system=system,
            temperature=temperature,
            seed=seed,
            format_json=True,
            base_url=base_url,
            timeout=timeout,
        )
    )


def embed_texts(
    model: str,
    texts: List[str],
    *,
    base_url: Optional[str] = None,
    timeout: float = 180.0,
) -> List[List[float]]:
    data = _post_json(
        f"{resolve_base_url(base_url)}/api/embed",
        {"model": model, "input": texts},
        timeout,
    )
    if isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    if isinstance(data.get("embedding"), list):
        return [data["embedding"]]
    raise ValueError(f"Unexpected Ollama embed response: {data}")
