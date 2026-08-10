from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    find_dotenv = None
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency
    certifi = None


DEFAULT_BASE_URL = "http://192.168.50.186:11434"


def _provider_for_model(model: str, kind: str) -> str:
    explicit = (
        os.getenv(f"{kind.upper()}_PROVIDER")
        or os.getenv("MODEL_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ""
    ).strip().lower()
    if explicit:
        return explicit

    name = (model or "").strip().lower()
    if kind == "embed" and name.startswith(("text-embedding-", "embedding-")):
        return "openai"
    if name.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt-")):
        return "openai"
    return "ollama"


def _load_openai_env() -> None:
    if not os.getenv("OPENAI_API_KEY") and load_dotenv is not None:
        env_path = find_dotenv(usecwd=True) if find_dotenv is not None else ""
        load_dotenv(env_path or None)
    if not os.getenv("OPENAI_API_KEY") and load_dotenv is not None:
        repo_env = Path(__file__).resolve().parent.parent / ".env"
        if repo_env.is_file():
            load_dotenv(repo_env)


def _openai_client(timeout: float) -> Any:
    if OpenAI is None:
        raise RuntimeError("OpenAI client not available. Install the `openai` package.")
    _load_openai_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env.")
    return OpenAI(timeout=timeout)


def _openai_api_base_url() -> str:
    base = (
        os.getenv("OPENAI_API_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip()
    if not base:
        base = "https://api.openai.com/v1"
    return base.rstrip("/")


def _openai_post_json(path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    _load_openai_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env.")

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{_openai_api_base_url()}/{path.lstrip('/')}",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        context = ssl.create_default_context(cafile=certifi.where()) if certifi is not None else None
        with request.urlopen(req, timeout=timeout, context=context) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    return json.loads(raw)


def _extract_openai_text(resp: Any) -> str:
    direct = getattr(resp, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    text_parts: List[str] = []
    for out in getattr(resp, "output", []) or []:
        if getattr(out, "type", None) != "message":
            continue
        for content in getattr(out, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text_parts.append(content.text)
    return "".join(text_parts).strip()


def _extract_openai_text_from_json(data: Dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    text_parts: List[str] = []
    for out in data.get("output") or []:
        if not isinstance(out, dict) or out.get("type") != "message":
            continue
        for content in out.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts).strip()


def _openai_generate_text(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> str:
    max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096"))
    input_messages = []
    if system:
        input_messages.append({"role": "system", "content": system})
    input_messages.append({"role": "user", "content": prompt})

    data = _openai_post_json(
        "responses",
        {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        timeout,
    )
    text = _extract_openai_text_from_json(data)
    if not text:
        raise ValueError(f"Unexpected OpenAI response: {data}")
    return text


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
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i = raw.find("{")
        if i < 0:
            raise ValueError(f"Model did not return JSON. Output was:\n{raw}")
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw[i:])
        if not isinstance(obj, dict):
            raise ValueError(f"Model did not return a JSON object. Output was:\n{raw}")
        return obj


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
    if _provider_for_model(model, "llm") == "openai":
        if format_json:
            prompt = prompt + "\n\nReturn strict JSON only."
        return _openai_generate_text(
            model,
            prompt,
            system=system,
            temperature=temperature,
            timeout=timeout,
        )

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
    if _provider_for_model(model, "embed") == "openai":
        data = _openai_post_json(
            "embeddings",
            {"model": model, "input": texts},
            timeout,
        )
        items = data.get("data")
        if not isinstance(items, list):
            raise ValueError(f"Unexpected OpenAI embeddings response: {data}")
        return [item["embedding"] for item in items]

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
