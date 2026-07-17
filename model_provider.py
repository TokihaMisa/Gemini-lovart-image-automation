"""Provider-neutral discovery of text-generation models.

This module deliberately keeps credentials out of its errors.  Callers can show
``ModelProviderError.user_message`` directly in the UI or logs.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import socket
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import urllib.request


@dataclass(frozen=True)
class DiscoveredModel:
    provider: str
    model_id: str
    display_name: str
    supports_generation: bool
    supports_thinking: bool | None
    image_input_status: str
    recommendation: str


class ModelProviderError(RuntimeError):
    def __init__(self, code: str, user_message: str, status_code: int | None = None):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.status_code = status_code


@dataclass(frozen=True)
class ModelTestResult:
    ok: bool
    message: str
    latency_ms: int


_GEMINI_EXCLUDED = ("embedding", "imagen", "veo", "live", "tts", "speech", "audio")
_NVIDIA_EXCLUDED = (
    "embed", "rerank", "retrieval", "tts", "speech", "audio", "flux",
    "stable-diffusion", "imagen", "veo",
)
_TEST_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Y9Z4WQAAAAASUVORK5CYII="
)


def validate_base_url(base_url: object) -> str:
    """Validate and normalize a provider base URL without making a request."""
    raw = str(base_url or "")
    if "\r" in raw or "\n" in raw:
        raise ModelProviderError(
            "invalid_base_url",
            "API 地址必须是包含主机名的 http 或 https 地址。",
        )
    cleaned = raw.strip()
    if not cleaned:
        raise ModelProviderError("missing_base_url", "请先填写 API 地址。")
    try:
        parts = urlsplit(cleaned)
        hostname = parts.hostname
    except ValueError:
        parts = None
        hostname = None
    if (
        parts is None
        or parts.scheme not in {"http", "https"}
        or not hostname
    ):
        raise ModelProviderError(
            "invalid_base_url",
            "API 地址必须是包含主机名的 http 或 https 地址。",
        )
    return cleaned.rstrip("/")


def validate_model_id(model_id: object) -> str:
    """Validate and normalize a model identifier without making a request."""
    raw = str(model_id or "")
    if "\r" in raw or "\n" in raw:
        raise ModelProviderError("invalid_model", "模型 ID 不能包含换行符。")
    cleaned = raw.strip()
    if not cleaned:
        raise ModelProviderError("missing_model", "请先选择模型。")
    return cleaned


def discover_models(
    provider: str, api_key: str, base_url: str, timeout: float = 20
) -> list[DiscoveredModel]:
    """Return prompt-generation models exposed by a supported provider."""
    normalized_provider = provider.strip().lower()
    if not api_key.strip():
        raise ModelProviderError("missing_key", "请先填写 API 密钥。")
    normalized_base_url = validate_base_url(base_url)
    if normalized_provider == "gemini":
        return _discover_gemini(api_key, normalized_base_url, timeout)
    if normalized_provider == "nvidia":
        return _discover_nvidia(api_key, normalized_base_url, timeout)
    raise ModelProviderError("unsupported_provider", "不支持的模型服务商。")


def test_selected_model(
    provider: str,
    api_key: str,
    base_url: str,
    model_id: str,
    timeout: float = 30,
) -> ModelTestResult:
    """Send one small image-and-text request to validate the selected model."""
    normalized_provider = provider.strip().lower()
    if not api_key.strip():
        raise ModelProviderError("missing_key", "\u8bf7\u5148\u586b\u5199 API \u5bc6\u94a5\u3002")
    normalized_base_url = validate_base_url(base_url)
    normalized_model_id = validate_model_id(model_id)

    if normalized_provider == "gemini":
        url = _append_query(
            _gemini_generate_url(normalized_base_url, normalized_model_id), {"key": api_key}
        )
        payload = _gemini_test_payload()
    elif normalized_provider == "nvidia":
        url = f"{normalized_base_url}/chat/completions"
        payload = _nvidia_test_payload(normalized_model_id)
    else:
        raise ModelProviderError("unsupported_provider", "\u4e0d\u652f\u6301\u7684\u6a21\u578b\u670d\u52a1\u5546\u3002")

    started = time.perf_counter()
    response = _request_json(
        url,
        api_key,
        normalized_provider,
        method="POST",
        payload=payload,
        timeout=timeout,
        operation="probe",
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    text = _extract_gemini_text(response) if normalized_provider == "gemini" else _extract_nvidia_text(response)
    if not text or not text.strip():
        raise ModelProviderError("empty_response", "\u6a21\u578b\u63a5\u53d7\u4e86\u8bf7\u6c42\uff0c\u4f46\u6ca1\u6709\u8fd4\u56de\u53ef\u7528\u6587\u5b57\u3002")
    return ModelTestResult(
        True,
        "API \u4e0e\u6240\u9009\u6a21\u578b\u53ef\u7528\uff0c\u652f\u6301\u56fe\u7247\u8f93\u5165\u5e76\u8fd4\u56de\u6587\u5b57",
        elapsed_ms,
    )


test_selected_model.__test__ = False


def _gemini_generate_url(base_url: str, model_id: str) -> str:
    return f"{base_url.strip().rstrip('/')}/models/{model_id}:generateContent"


def _gemini_test_payload() -> dict[str, Any]:
    return {
        "contents": [{"parts": [
            {"text": "Reply with OK."},
            {"inline_data": {"mime_type": "image/png", "data": _TEST_PNG_BASE64}},
        ]}],
        "generationConfig": {"maxOutputTokens": 32},
    }


def _nvidia_test_payload(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Reply with OK."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TEST_PNG_BASE64}"}},
        ]}],
        "max_tokens": 32,
    }


def _extract_gemini_text(payload: dict[str, Any]) -> str | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            continue
        for part in content["parts"]:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                return part["text"]
    return None


def _extract_nvidia_text(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"]
    return None


def _discover_gemini(api_key: str, base_url: str, timeout: float) -> list[DiscoveredModel]:
    models: list[DiscoveredModel] = []
    page_token: str | None = None
    while True:
        query = {"key": api_key}
        if page_token:
            query["pageToken"] = page_token
        try:
            request_url = _append_query(_models_url(base_url), query)
        except ValueError as exc:
            raise _map_network_error(exc) from None
        payload = _request_json(
            request_url, api_key, "gemini", timeout=timeout, operation="discovery"
        )
        for raw_model in payload.get("models", []):
            if not isinstance(raw_model, dict):
                continue
            model_id = _gemini_model_id(raw_model.get("name"))
            if not model_id or not _is_gemini_prompt_model(raw_model, model_id):
                continue
            models.append(DiscoveredModel(
                provider="gemini",
                model_id=model_id,
                display_name=str(raw_model.get("displayName") or model_id),
                supports_generation=True,
                supports_thinking=_gemini_thinking(raw_model),
                image_input_status=("reported" if _supports_image_input(raw_model) else "unknown"),
                recommendation=("recommended" if _is_recommended_gemini(model_id) else "available"),
            ))
        next_page_token = payload.get("nextPageToken")
        if not isinstance(next_page_token, str) or not next_page_token:
            return models
        page_token = next_page_token


def _discover_nvidia(api_key: str, base_url: str, timeout: float) -> list[DiscoveredModel]:
    payload = _request_json(
        _models_url(base_url), api_key, "nvidia", timeout=timeout, operation="discovery"
    )
    models: list[DiscoveredModel] = []
    for raw_model in payload.get("data", []):
        if not isinstance(raw_model, dict):
            continue
        model_id = raw_model.get("id")
        if not isinstance(model_id, str) or not model_id or _contains_any(model_id, _NVIDIA_EXCLUDED):
            continue
        models.append(DiscoveredModel(
            provider="nvidia",
            model_id=model_id,
            display_name=str(raw_model.get("display_name") or raw_model.get("name") or model_id),
            supports_generation=True,
            supports_thinking=None,
            image_input_status="unknown",
            recommendation=("recommended" if "kimi" in model_id.lower() else "available"),
        ))
    return models


def _request_json(
    url: str,
    api_key: str,
    provider: str,
    method: str = "GET",
    payload: Any = None,
    timeout: float = 20,
    operation: str = "discovery",
) -> dict[str, Any]:
    """Make a credentialed provider request and return a JSON object safely."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    try:
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if provider == "nvidia":
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise _map_http_error(provider, exc, operation) from None
    except (
        URLError,
        TimeoutError,
        socket.timeout,
        ssl.SSLError,
        OSError,
        http.client.HTTPException,
        ValueError,
    ) as exc:
        raise _map_network_error(exc) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelProviderError("invalid_response", "模型服务返回了无法识别的数据。") from None
    if not isinstance(decoded, dict):
        raise ModelProviderError("invalid_response", "模型服务返回了无法识别的数据。")
    return decoded


def _map_http_error(provider: str, exc: HTTPError, operation: str) -> ModelProviderError:
    if exc.code == 404:
        if operation == "probe":
            return ModelProviderError(
                "model_unavailable", "所选模型不存在或不可用，请检查模型 ID。", exc.code
            )
        if provider == "nvidia":
            return ModelProviderError(
                "models_endpoint_unsupported",
                "服务可访问，但不支持模型列表接口；请保留或手动填写模型 ID。",
                exc.code,
            )
        return ModelProviderError(
            "not_found", "API 地址或模型列表接口错误，请检查配置。", exc.code
        )
    messages = {
        401: ("authentication", "API 密钥无效或已过期，请检查后重试。"),
        403: ("authentication", "API 密钥无效或没有访问权限，请检查后重试。"),
        429: ("rate_limit", "请求过于频繁，请稍后再试。"),
    }
    code, message = messages.get(exc.code, ("http_error", "模型服务暂时无法访问，请稍后重试。"))
    return ModelProviderError(code, message, exc.code)


def _map_network_error(exc: BaseException) -> ModelProviderError:
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return ModelProviderError("timeout", "连接模型服务超时，请稍后重试。")
    return ModelProviderError("network", "无法连接模型服务，请检查网络和 API 地址。")


def model_choice_labels(models: list[DiscoveredModel]) -> list[tuple[str, str]]:
    """Return stable, Gradio-compatible labels for a discovered model list."""
    choices: list[tuple[str, str]] = []
    for model in models:
        details: list[str] = []
        if model.supports_thinking:
            details.append("Thinking")
        image_status_label = {
            "reported": "图片已报告支持",
            "verified": "图片已验证支持",
            "failed": "图片不支持",
            "unknown": "图片未验证",
        }.get(model.image_input_status, "图片未验证")
        details.append(image_status_label)
        choices.append((f"{model.display_name} · {' · '.join(details)}", model.model_id))
    return choices


def _models_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    return base if base.endswith("/models") else f"{base}/models"


def _append_query(url: str, values: dict[str, str]) -> str:
    parts = urlsplit(url)
    query = list(parse_qsl(parts.query, keep_blank_values=True))
    query.extend(values.items())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _gemini_model_id(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    return name.removeprefix("models/") or None


def _is_gemini_prompt_model(raw_model: dict[str, Any], model_id: str) -> bool:
    methods = raw_model.get("supportedGenerationMethods")
    return (
        isinstance(methods, list)
        and "generateContent" in methods
        and not _contains_any(model_id, _GEMINI_EXCLUDED)
    )


def _gemini_thinking(raw_model: dict[str, Any]) -> bool | None:
    thinking = raw_model.get("thinking")
    return thinking if isinstance(thinking, bool) else None


def _supports_image_input(raw_model: dict[str, Any]) -> bool:
    if raw_model.get("supportsImageInput") is True:
        return True
    modalities = raw_model.get("supportedInputModalities", raw_model.get("inputModalities"))
    return isinstance(modalities, list) and any(str(item).lower() == "image" for item in modalities)


def _is_recommended_gemini(model_id: str) -> bool:
    return "gemini" in model_id.lower() and ("flash" in model_id.lower() or "pro" in model_id.lower())


def _contains_any(model_id: str, terms: tuple[str, ...]) -> bool:
    lowered = model_id.lower()
    return any(term in lowered for term in terms)
