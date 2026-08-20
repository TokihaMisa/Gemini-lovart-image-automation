"""Validated configuration primitives for OpenAI Images-compatible APIs."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import http.client
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import ssl
import tempfile
import threading
import time
from typing import Any, Callable, Final, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
import urllib.request

from PIL import Image, ImageOps, UnidentifiedImageError

from network_retry import PERMANENT_TLS_GUIDANCE, RetryKind, classify_network_error


DEFAULT_OPENAI_IMAGE_BASE_URL: Final = ""
MAX_REFERENCE_IMAGES: Final = 14
MAX_REFERENCE_BYTES: Final = 10 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES: Final = 30 * 1024 * 1024
MAX_CREATE_BODY_BYTES: Final = 50 * 1024 * 1024
TASK_WAIT_LIMIT_SECONDS: Final = 600.0
MAX_TASK_ID_LENGTH: Final = 128
_VALID_RESOLUTIONS: Final = {"1K", "2K", "4K"}
_PIXEL_SIZES_BY_RESOLUTION: Final = {
    "1K": (
        (1 / 2, "960x1920"),
        (2 / 3, "1024x1536"),
        (3 / 4, "960x1280"),
        (4 / 5, "1024x1280"),
        (1.0, "1024x1024"),
        (5 / 4, "1280x1024"),
        (4 / 3, "1280x960"),
        (3 / 2, "1536x1024"),
        (2.0, "1920x960"),
        (9 / 16, "1088x1920"),
        (16 / 9, "1920x1088"),
    ),
    "2K": (
        (1 / 2, "1280x2560"),
        (2 / 3, "2048x3072"),
        (3 / 4, "1920x2560"),
        (4 / 5, "2048x2560"),
        (1.0, "2048x2048"),
        (5 / 4, "2560x2048"),
        (4 / 3, "2560x1920"),
        (3 / 2, "3072x2048"),
        (2.0, "2560x1280"),
        (9 / 16, "1440x2560"),
        (16 / 9, "2560x1440"),
    ),
    "4K": (
        (1 / 2, "1920x3840"),
        (2 / 3, "2304x3456"),
        (3 / 4, "2400x3200"),
        (4 / 5, "2560x3200"),
        (1.0, "2880x2880"),
        (5 / 4, "3200x2560"),
        (4 / 3, "3200x2400"),
        (3 / 2, "3456x2304"),
        (2.0, "3840x1920"),
        (9 / 16, "2160x3840"),
        (16 / 9, "3840x2160"),
    ),
}
_UNSAFE_IPV6_TRANSITION_NETWORKS: Final = (
    ipaddress.IPv6Network("::/96"),  # deprecated IPv4-compatible addresses
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped addresses
    ipaddress.IPv6Network("64:ff9b::/96"),  # well-known NAT64 translation
    ipaddress.IPv6Network("64:ff9b:1::/48"),  # local-use NAT64 translation
    ipaddress.IPv6Network("2001::/32"),  # Teredo
    ipaddress.IPv6Network("2002::/16"),  # 6to4
)
_TEST_IMAGE_BASE64: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8Dw"
    "HwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


class OpenAIImageAPIError(Exception):
    """A safe, user-facing error that deliberately carries no request secrets."""

    def __init__(
        self,
        code: str,
        user_message: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.user_message = user_message
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(user_message)


@dataclass(frozen=True)
class GeneratedImage:
    local_path: str
    model: str
    task: ImageTaskSnapshot


@dataclass(frozen=True, slots=True)
class ImageTaskSnapshot:
    task_id: str
    state: str
    is_final: bool
    task_created_at: float
    progress: str = ""
    status: str = ""
    status_group: str = ""
    result_url: str = ""
    result_type: str = ""
    error: str = ""
    cost: object = None


class ImageTaskStillRunning(OpenAIImageAPIError):
    def __init__(self, task: ImageTaskSnapshot) -> None:
        self.task = task
        super().__init__(
            "task_still_running",
            "GPT Image task is still running on the provider; its saved task ID can be resumed.",
        )


class _InvocationDeadlineExceeded(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _ResolvedAddress:
    family: int
    ip_literal: str
    socket_address: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedResultURL:
    scheme: str
    hostname: str
    port: int
    authority: str
    request_target: str
    addresses: tuple[_ResolvedAddress, ...]


class _OpenAIImageAPIKeyAccess:
    __slots__ = ("_api_key",)

    @property
    def api_key(self) -> str:
        """Return the resolved key for the request authorizer only."""
        return self._api_key


@dataclass(frozen=True, slots=True, init=False)
class OpenAIImageAPIConfig(_OpenAIImageAPIKeyAccess):
    base_url: str = DEFAULT_OPENAI_IMAGE_BASE_URL
    model: str = "gpt-image-2"
    resolution: str = "1K"
    timeout: float = 600.0
    max_attempts: int = 4
    retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0)
    merge_reference_images: bool = False

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OPENAI_IMAGE_BASE_URL,
        model: str = "gpt-image-2",
        resolution: str = "1K",
        timeout: float = 600.0,
        max_attempts: int = 4,
        retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0),
        merge_reference_images: bool = False,
    ) -> None:
        object.__setattr__(self, "_api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "retry_delays", retry_delays)
        object.__setattr__(
            self,
            "merge_reference_images",
            bool(merge_reference_images),
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, object], api_key: str
    ) -> "OpenAIImageAPIConfig":
        section_value = config.get("openai_image", {}) if isinstance(config, Mapping) else {}
        section = section_value if isinstance(section_value, Mapping) else {}
        resolved_key = str(api_key or "").strip()
        if not resolved_key:
            raise OpenAIImageAPIError("missing_key", "请先填写 GPT Image API 密钥。")

        resolution = str(section.get("resolution", "1K") or "1K").upper()
        if resolution not in _VALID_RESOLUTIONS:
            raise OpenAIImageAPIError(
                "invalid_resolution", "GPT Image 分辨率必须是 1K、2K 或 4K。"
            )

        model = str(section.get("model") or "gpt-image-2").strip() or "gpt-image-2"
        base_url = normalize_openai_image_base_url(section.get("base_url"))
        merge_reference_images = _config_boolean(
            section.get("merge_reference_images"),
            default=False,
        )
        return cls(
            api_key=resolved_key,
            base_url=base_url,
            model=model,
            resolution=resolution,
            merge_reference_images=merge_reference_images,
        )


def normalize_openai_image_base_url(value: object | None) -> str:
    """Validate a provider endpoint without inventing provider-specific path segments."""
    raw_value = "" if value is None else str(value)
    if "\r" in raw_value or "\n" in raw_value:
        _raise_invalid_base_url()

    cleaned = raw_value.strip()
    if not cleaned:
        _raise_invalid_base_url()
    try:
        parsed = urlsplit(cleaned)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        _raise_invalid_base_url()

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in parsed.netloc)
        or "\\" in parsed.netloc
        or parsed.netloc.endswith(":")
        or port is not None and not 0 < port <= 65535
    ):
        _raise_invalid_base_url()

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _provider_image_size(resolution: str, image_size: str) -> str:
    """Map the requested ratio to the closest documented pixel size."""
    ratio_text = str(image_size or "").strip().replace("：", ":").replace("×", "x")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:xX]\s*(\d+(?:\.\d+)?)", ratio_text)
    target_ratio = 1.0
    if match:
        width = float(match.group(1))
        height = float(match.group(2))
        if width > 0 and height > 0:
            target_ratio = width / height
    candidates = _PIXEL_SIZES_BY_RESOLUTION[resolution]
    _, candidate_size = min(
        candidates,
        key=lambda item: abs(item[0] - target_ratio),
    )
    return candidate_size


def _config_boolean(value: object, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return bool(default)


def _protocol_base_url(base_url: str) -> str:
    normalized = normalize_openai_image_base_url(base_url)
    return normalized[:-3] if normalized.lower().endswith("/v1") else normalized


def _media_endpoint(base_url: str, resource: str, *, task_id: str = "") -> str:
    base = _protocol_base_url(base_url)
    if resource == "generate":
        return f"{base}/v1/media/generate"
    if resource == "status" and task_id:
        return f"{base}/v1/media/status?{urlencode({'task_id': task_id})}"
    raise ValueError("invalid media endpoint")


def _decode_json_response(raw_response: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned an unreadable response."
        ) from None
    if not isinstance(decoded, dict):
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned an invalid response."
        )
    return decoded


def _notify_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _notify_task(
    callback: Callable[[ImageTaskSnapshot], None] | None,
    task: ImageTaskSnapshot,
) -> None:
    if callback is not None:
        callback(task)


def _normalize_task_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise OpenAIImageAPIError("invalid_response", "GPT Image API returned no valid task ID.")
    task_id = str(value).strip()
    if (
        not task_id
        or len(task_id) > MAX_TASK_ID_LENGTH
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", task_id) is None
    ):
        raise OpenAIImageAPIError("invalid_response", "GPT Image API returned no valid task ID.")
    return task_id


def _task_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "task_id" in payload:
        return payload
    nested = payload.get("data")
    return nested if isinstance(nested, Mapping) else payload


def _task_text(value: object, *, limit: int = 500) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]


def _validate_task_state(task: ImageTaskSnapshot) -> ImageTaskSnapshot:
    if task.state in {"pending", "running"} and task.is_final is False:
        return task
    if task.state == "success" and task.is_final is True and task.result_url:
        return task
    if task.state == "failed" and task.is_final is True:
        return task
    raise OpenAIImageAPIError(
        "invalid_response", "GPT Image API returned an invalid task state."
    )


def _redact_snapshot_value(value: object, api_key: str) -> object:
    if not api_key:
        return value
    if isinstance(value, str):
        return value.replace(api_key, "[redacted]")
    if isinstance(value, Mapping):
        return {
            _redact_snapshot_value(key, api_key): _redact_snapshot_value(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_snapshot_value(item, api_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_snapshot_value(item, api_key) for item in value)
    if isinstance(value, set):
        return {_redact_snapshot_value(item, api_key) for item in value}
    if api_key in repr(value):
        return repr(value).replace(api_key, "[redacted]")
    return value


def _sanitize_task_snapshot(
    task: ImageTaskSnapshot,
    api_key: str,
) -> ImageTaskSnapshot:
    if api_key and api_key in task.task_id:
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned an unsafe task ID."
        )
    return ImageTaskSnapshot(
        task_id=task.task_id,
        state=str(_redact_snapshot_value(task.state, api_key)),
        is_final=task.is_final,
        task_created_at=task.task_created_at,
        progress=str(_redact_snapshot_value(task.progress, api_key)),
        status=str(_redact_snapshot_value(task.status, api_key)),
        status_group=str(_redact_snapshot_value(task.status_group, api_key)),
        result_url=str(_redact_snapshot_value(task.result_url, api_key)),
        result_type=str(_redact_snapshot_value(task.result_type, api_key)),
        error=str(_redact_snapshot_value(task.error, api_key)),
        cost=_redact_snapshot_value(task.cost, api_key),
    )


def _validate_resume_task(
    task: ImageTaskSnapshot,
    api_key: str,
) -> ImageTaskSnapshot:
    if not isinstance(task, ImageTaskSnapshot):
        raise OpenAIImageAPIError("invalid_response", "Saved GPT Image task data is invalid.")
    task_id = _normalize_task_id(task.task_id)
    if task_id != task.task_id:
        raise OpenAIImageAPIError("invalid_response", "Saved GPT Image task data is invalid.")
    if not isinstance(task.is_final, bool) or not _is_valid_timestamp(task.task_created_at):
        raise OpenAIImageAPIError("invalid_response", "Saved GPT Image task data is invalid.")
    return _validate_task_state(_sanitize_task_snapshot(task, api_key))


def _is_valid_timestamp(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number >= 0.0 and math.isfinite(number)


def _task_failed_error(task: ImageTaskSnapshot, api_key: str) -> OpenAIImageAPIError:
    provider_error = _task_text(task.error)
    if api_key:
        provider_error = provider_error.replace(api_key, "[redacted]")
    message = "GPT Image task failed."
    if provider_error:
        message = f"{message} Provider message: {provider_error}"
    error = OpenAIImageAPIError("task_failed", message)
    error.task = task
    return error


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw_value = next(
        (value for name, value in headers.items() if name.lower() == "retry-after"),
        None,
    )
    if raw_value is None:
        return None
    try:
        return max(1.0, min(60.0, float(raw_value)))
    except (TypeError, ValueError):
        return None


def _deadline_remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _InvocationDeadlineExceeded
    return remaining


def _set_response_read_timeout(response: Any, timeout: float) -> None:
    candidates = [response]
    current = response
    for attribute in ("fp", "raw", "_sock"):
        current = getattr(current, attribute, None)
        if current is None:
            break
        candidates.append(current)
    for candidate in reversed(candidates):
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            setter(timeout)
            return


def _read_response_body(response: Any, deadline: float | None) -> bytes:
    if deadline is None:
        return response.read()
    read_chunk = getattr(response, "read1", None)
    if not callable(read_chunk):
        read_chunk = response.read
    chunks: list[bytes] = []
    while True:
        remaining = _deadline_remaining(deadline)
        assert remaining is not None
        _set_response_read_timeout(response, remaining)
        chunk = read_chunk(64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    _deadline_remaining(deadline)
    return b"".join(chunks)


class OpenAIImageAPI:
    """Transport for the documented GPT Image media task protocol."""

    def __init__(
        self,
        config: OpenAIImageAPIConfig,
        logger: Any | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self._sleep = sleep or time.sleep

    def generate_edit(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | Path,
        image_size: str = "",
        status_callback: Callable[[str], None] | None = None,
        task_callback: Callable[[ImageTaskSnapshot], None] | None = None,
        resume_task: ImageTaskSnapshot | None = None,
    ) -> GeneratedImage:
        images = _encode_reference_images(
            image_paths,
            merge=bool(self.config.merge_reference_images),
        )
        body = _build_create_body(
            self.config.model,
            append_aspect_instruction(prompt, image_size),
            _provider_image_size(self.config.resolution, image_size),
            images,
        )
        if resume_task is None:
            task = self._submit_task_once(body)
        else:
            task = _validate_resume_task(resume_task, self.config.api_key)
        _notify_task(task_callback, task)

        if task.state == "failed":
            raise _task_failed_error(task, self.config.api_key)
        if task.is_final and task.state == "success":
            return self._download_task_result(task, output_path, status_callback)

        task = self._poll_task(task, status_callback, task_callback)
        return self._download_task_result(task, output_path, status_callback)

    def test_edit(self, output_dir: str | Path) -> GeneratedImage:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        source_path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=".openai-image-test-", suffix=".png", dir=directory
            )
            source_path = Path(raw_path)
            with os.fdopen(descriptor, "wb") as source:
                source.write(base64.b64decode(_TEST_IMAGE_BASE64))
            return self.generate_edit(
                "Generate a small product-image API compatibility test.",
                [source_path],
                directory / "openai-image-test.png",
            )
        finally:
            if source_path is not None:
                try:
                    source_path.unlink()
                except FileNotFoundError:
                    pass

    def _request_json(
        self,
        endpoint: str,
        body: bytes,
        content_type: str,
        *,
        max_attempts: int | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(endpoint, data=body, method="POST")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        request.add_header("Content-Type", content_type)
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        raw_response = self._request_bytes(
            request,
            opener.open,
            "json",
            max_attempts=max_attempts,
            status_callback=status_callback,
        )
        return _decode_json_response(raw_response)

    def _request_json_url(
        self,
        url: str,
        status_callback: Callable[[str], None] | None = None,
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], float | None]:
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def request_status() -> None:
            response_headers: dict[str, str] = {}
            try:
                raw_response = self._request_bytes(
                    request,
                    opener.open,
                    "json",
                    status_callback=status_callback,
                    response_headers=response_headers,
                    deadline=deadline,
                )
                result = (
                    _decode_json_response(raw_response),
                    _retry_after_seconds(response_headers),
                )
            except Exception as exc:
                result_queue.put(("error", exc))
            else:
                result_queue.put(("result", result))

        # A status GET is idempotent. Running it in a daemon worker gives the
        # caller a hard invocation deadline even if urllib is stuck parsing a
        # peer that dribbles response headers without ever timing out a read.
        threading.Thread(
            target=request_status,
            name="gpt-image-status-get",
            daemon=True,
        ).start()
        remaining = _deadline_remaining(deadline)
        try:
            result_type, value = result_queue.get(timeout=remaining)
        except queue.Empty:
            raise _InvocationDeadlineExceeded from None
        _deadline_remaining(deadline)
        if result_type == "error":
            if isinstance(value, BaseException):
                raise value
            raise RuntimeError("image status worker returned an invalid error")
        return cast(tuple[dict[str, Any], float | None], value)

    def _submit_task_once(self, body: bytes) -> ImageTaskSnapshot:
        try:
            payload = self._request_json(
                _media_endpoint(self.config.base_url, "generate"),
                body,
                "application/json",
                max_attempts=1,
            )
        except OpenAIImageAPIError as exc:
            if exc.status_code is None and exc.code in {"timeout", "network"}:
                raise OpenAIImageAPIError(
                    "ambiguous_submission",
                    "GPT Image task submission may have reached the provider, but no task ID was returned; automatic resubmission is disabled.",
                ) from None
            raise
        task_id = _normalize_task_id(_task_payload(payload).get("task_id"))
        return _sanitize_task_snapshot(
            ImageTaskSnapshot(
                task_id=task_id,
                state="pending",
                is_final=False,
                task_created_at=time.time(),
            ),
            self.config.api_key,
        )

    def _parse_status_task(
        self,
        payload: Mapping[str, Any],
        previous: ImageTaskSnapshot,
    ) -> ImageTaskSnapshot:
        values = _task_payload(payload)
        task_id = _normalize_task_id(values.get("task_id"))
        if task_id != previous.task_id:
            raise OpenAIImageAPIError(
                "invalid_response", "GPT Image API returned a mismatched task ID."
            )
        state_value = values.get("state")
        if not isinstance(state_value, str) or not state_value.strip():
            raise OpenAIImageAPIError(
                "invalid_response", "GPT Image API returned no task state."
            )
        is_final = values.get("is_final")
        if not isinstance(is_final, bool):
            raise OpenAIImageAPIError(
                "invalid_response", "GPT Image API returned no final-state flag."
            )
        task = ImageTaskSnapshot(
            task_id=task_id,
            state=state_value.strip().lower(),
            is_final=is_final,
            task_created_at=previous.task_created_at,
            progress=_task_text(values.get("progress"), limit=80),
            status=_task_text(values.get("status"), limit=160),
            status_group=_task_text(values.get("status_group"), limit=160),
            result_url=_task_text(values.get("result_url"), limit=4096),
            result_type=_task_text(values.get("result_type"), limit=80),
            error=_task_text(values.get("error")),
            cost=values.get("cost"),
        )
        return _validate_task_state(
            _sanitize_task_snapshot(task, self.config.api_key)
        )

    def _poll_task(
        self,
        task: ImageTaskSnapshot,
        status_callback: Callable[[str], None] | None,
        task_callback: Callable[[ImageTaskSnapshot], None] | None,
    ) -> ImageTaskSnapshot:
        started_at = time.monotonic()
        deadline = started_at + TASK_WAIT_LIMIT_SECONDS
        status_url = _media_endpoint(self.config.base_url, "status", task_id=task.task_id)
        while True:
            if time.monotonic() >= deadline:
                raise ImageTaskStillRunning(task)
            try:
                payload, _retry_after = self._request_json_url(
                    status_url,
                    status_callback=status_callback,
                    deadline=deadline,
                )
            except _InvocationDeadlineExceeded:
                raise ImageTaskStillRunning(task) from None
            if time.monotonic() >= deadline:
                raise ImageTaskStillRunning(task)
            task = self._parse_status_task(payload, task)
            _notify_task(task_callback, task)
            elapsed = max(0.0, time.monotonic() - started_at)
            if task.state == "failed":
                raise _task_failed_error(task, self.config.api_key)
            if task.is_final:
                _notify_status(status_callback, "✅ GPT Image 生成完成，正在安全下载图片")
                return task

            display = task.status or task.status_group or task.state
            progress = f" · {task.progress}" if task.progress else ""
            suffix = task.task_id[-6:]
            _notify_status(
                status_callback,
                f"⏳ GPT Image {display}{progress} · 已等待 {int(elapsed)} 秒 · 任务 …{suffix}",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ImageTaskStillRunning(task)
            delay = 5.0 if elapsed <= 120.0 else 10.0
            self._sleep(min(delay, remaining))

    def _download_task_result(
        self,
        task: ImageTaskSnapshot,
        output_path: str | Path,
        status_callback: Callable[[str], None] | None = None,
    ) -> GeneratedImage:
        task = _validate_task_state(task)
        if task.state != "success":
            raise OpenAIImageAPIError(
                "invalid_response", "GPT Image task has no downloadable result."
            )
        resolved_url = validate_remote_image_url(task.result_url)
        image_bytes = _download_resolved_image(
            resolved_url,
            self.config.timeout,
            self.config.max_attempts,
            lambda attempt, attempts: self._notice_retry(
                attempt,
                attempts,
                status_callback=status_callback,
            ),
        )
        target = Path(output_path)
        atomic_save_validated_image(image_bytes, target)
        return GeneratedImage(local_path=str(target), model=self.config.model, task=task)

    def _request_bytes(
        self,
        request: urllib.request.Request,
        open_request: Callable[..., Any] | None = None,
        expected_content_type: str | None = None,
        *,
        max_attempts: int | None = None,
        status_callback: Callable[[str], None] | None = None,
        response_headers: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> bytes:
        attempts = max(
            1,
            int(self.config.max_attempts if max_attempts is None else max_attempts),
        )
        request_opener = open_request or urllib.request.urlopen
        for attempt in range(1, attempts + 1):
            try:
                remaining = _deadline_remaining(deadline)
                request_timeout = float(self.config.timeout)
                if remaining is not None:
                    request_timeout = min(request_timeout, remaining)
                with request_opener(request, timeout=request_timeout) as response:
                    response_body = _read_response_body(response, deadline)
                    if expected_content_type is not None:
                        _validate_response_contract(
                            response, response_body, expected_content_type
                        )
                    if response_headers is not None:
                        response_headers.clear()
                        response_headers.update(
                            (str(name), str(value))
                            for name, value in response.headers.items()
                        )
                    return response_body
            except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
                _deadline_remaining(deadline)
                error = _transport_error(exc)
                if not error.retryable or attempt >= attempts:
                    raise error from None
                self._notice_retry(
                    attempt,
                    attempts,
                    status_callback=status_callback,
                    deadline=deadline,
                )
        raise RuntimeError("image request retry loop exhausted")

    def _notice_retry(
        self,
        attempt: int,
        attempts: int,
        status_callback: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> None:
        delay = _retry_delay(self.config.retry_delays, attempt)
        if self.logger is not None:
            self.logger.warning("GPT Image request failed (%s/%s); retrying.", attempt, attempts)
        _notify_status(
            status_callback,
            f"🔄 GPT Image 请求失败，正在重试（{attempt}/{attempts}）",
        )
        remaining = _deadline_remaining(deadline)
        if remaining is not None:
            delay = min(delay, remaining)
        self._sleep(delay)
        _deadline_remaining(deadline)



def append_aspect_instruction(prompt: str, image_size: str) -> str:
    """Keep the spreadsheet's requested ratio in the provider prompt without resizing."""
    source = str(prompt or "").strip()
    aspect = str(image_size or "").strip()
    if not aspect:
        return source
    return f"{source}\n\nPreserve the requested image aspect ratio exactly: {aspect}."


def _read_validated_reference(path: str | Path) -> tuple[str, bytes]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        if not raw:
            raise ValueError
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image_format = str(image.format or "").upper()
    except (OSError, UnidentifiedImageError, SyntaxError, ValueError):
        raise OpenAIImageAPIError(
            "invalid_input_image", "Each source file must be a non-empty readable image."
        ) from None

    media_types = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
    try:
        return media_types[image_format], raw
    except KeyError:
        raise OpenAIImageAPIError(
            "invalid_input_image", "Each source image must be PNG, JPEG, or WebP."
        ) from None


def _encode_reference_images(paths: Sequence[str | Path], merge: bool) -> list[str]:
    source_paths = [Path(path) for path in paths]
    if not source_paths:
        raise OpenAIImageAPIError("missing_input_image", "At least one source image is required.")
    if len(source_paths) > MAX_REFERENCE_IMAGES:
        raise OpenAIImageAPIError(
            "reference_image_limit", f"At most {MAX_REFERENCE_IMAGES} reference images are allowed."
        )

    validated = [_read_validated_reference(path) for path in source_paths]
    _validate_reference_byte_limits(validated)

    if merge and len(source_paths) > 1:
        reference_sheet = _build_reference_sheet(source_paths)
        try:
            validated = [_read_validated_reference(reference_sheet)]
            _validate_reference_byte_limits(validated)
        finally:
            try:
                reference_sheet.unlink()
            except FileNotFoundError:
                pass
    return [f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}" for media_type, raw in validated]


def _validate_reference_byte_limits(validated: Sequence[tuple[str, bytes]]) -> None:
    if any(len(raw) > MAX_REFERENCE_BYTES for _, raw in validated):
        raise OpenAIImageAPIError("reference_image_too_large", "A reference image exceeds the size limit.")
    if sum(len(raw) for _, raw in validated) > MAX_REFERENCE_TOTAL_BYTES:
        raise OpenAIImageAPIError("reference_total_too_large", "Reference images exceed the total size limit.")


def _build_create_body(model: str, prompt: str, size: str, images: Sequence[str]) -> bytes:
    payload = {
        "model": str(model),
        "prompt": str(prompt),
        "params": {"images": list(images), "size": str(size), "quality": "auto", "n": 1},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_CREATE_BODY_BYTES:
        raise OpenAIImageAPIError("create_body_too_large", "Image request body exceeds the size limit.")
    return body


def validate_remote_image_url(url: str) -> _ResolvedResultURL:
    """Resolve one public address set and retain it for the eventual connection."""
    raw_url = str(url or "")
    if "\r" in raw_url or "\n" in raw_url:
        _raise_unsafe_result_url()
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        _raise_unsafe_result_url()
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in parsed.netloc)
        or "\\" in parsed.netloc
        or parsed.netloc.endswith(":")
        or hostname.rstrip(".").lower() == "localhost"
        or port is not None and not 0 < port <= 65535
    ):
        _raise_unsafe_result_url()

    scheme = parsed.scheme.lower()
    resolved_port = port or (443 if scheme == "https" else 80)
    ascii_hostname = _ascii_result_hostname(hostname)
    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    try:
        request_target.encode("ascii")
    except UnicodeEncodeError:
        _raise_unsafe_result_url()

    try:
        resolved = socket.getaddrinfo(
            ascii_hostname,
            resolved_port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        _raise_unsafe_result_url()
    if not resolved:
        _raise_unsafe_result_url()

    addresses: list[_ResolvedAddress] = []
    seen_addresses: set[tuple[int, str]] = set()
    for entry in resolved:
        try:
            family = entry[0]
            ip_literal = entry[4][0]
            address = ipaddress.ip_address(ip_literal)
        except (TypeError, ValueError, IndexError):
            _raise_unsafe_result_url()
        if (
            family not in {socket.AF_INET, socket.AF_INET6}
            or not _is_safe_public_result_address(address)
        ):
            _raise_unsafe_result_url()
        key = (family, str(address))
        if key in seen_addresses:
            continue
        seen_addresses.add(key)
        if family == socket.AF_INET:
            socket_address: tuple[Any, ...] = (str(address), resolved_port)
        else:
            socket_address = (str(address), resolved_port, 0, 0)
        addresses.append(_ResolvedAddress(family, str(address), socket_address))

    if not addresses:
        _raise_unsafe_result_url()
    host_for_authority = (
        f"[{ascii_hostname}]"
        if _is_ipv6_literal(ascii_hostname)
        else ascii_hostname
    )
    authority = (
        f"{host_for_authority}:{resolved_port}"
        if port is not None
        else host_for_authority
    )
    return _ResolvedResultURL(
        scheme=scheme,
        hostname=ascii_hostname,
        port=resolved_port,
        authority=authority,
        request_target=request_target,
        addresses=tuple(addresses),
    )


def _ascii_result_hostname(hostname: str) -> str:
    candidate = hostname.rstrip(".")
    if not candidate:
        _raise_unsafe_result_url()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        pass
    try:
        ascii_hostname = candidate.encode("idna").decode("ascii").lower()
    except UnicodeError:
        _raise_unsafe_result_url()
    if not ascii_hostname or len(ascii_hostname) > 253:
        _raise_unsafe_result_url()
    return ascii_hostname


def _is_safe_public_result_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    explicitly_unsafe = (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )
    if explicitly_unsafe:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.is_site_local:
            return False
        if (
            address.ipv4_mapped is not None
            or address.sixtofour is not None
            or address.teredo is not None
            or any(address in network for network in _UNSAFE_IPV6_TRANSITION_NETWORKS)
            or _is_isatap_address(address)
        ):
            return False
    return address.is_global


def _is_isatap_address(address: ipaddress.IPv6Address) -> bool:
    """Recognize both common ISATAP interface-identifier encodings."""
    interface_prefix = (int(address) >> 32) & 0xFFFFFFFF
    return interface_prefix in {0x00005EFE, 0x02005EFE}


def _is_ipv6_literal(hostname: str) -> bool:
    try:
        return ipaddress.ip_address(hostname).version == 6
    except ValueError:
        return False


def _connect_validated_address(
    address: _ResolvedAddress, timeout: float
) -> socket.socket:
    connected_socket = socket.socket(address.family, socket.SOCK_STREAM)
    try:
        connected_socket.settimeout(timeout)
        connected_socket.connect(address.socket_address)
        try:
            peer = ipaddress.ip_address(connected_socket.getpeername()[0])
            expected = ipaddress.ip_address(address.ip_literal)
        except (OSError, TypeError, ValueError, IndexError):
            _raise_unsafe_result_url()
        if peer != expected:
            _raise_unsafe_result_url()
        return connected_socket
    except BaseException:
        connected_socket.close()
        raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection whose socket target is an already-verified IP literal."""

    def __init__(
        self,
        hostname: str,
        port: int,
        address: _ResolvedAddress,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = _connect_validated_address(self._validated_address, self.timeout)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    """A pinned TCP connection that still authenticates the original TLS host."""

    def __init__(
        self,
        hostname: str,
        port: int,
        address: _ResolvedAddress,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(hostname, port, address, timeout)
        self._context = context

    def connect(self) -> None:
        connected_socket = _connect_validated_address(
            self._validated_address, self.timeout
        )
        try:
            self.sock = self._context.wrap_socket(
                connected_socket, server_hostname=self.host
            )
        except BaseException:
            connected_socket.close()
            raise


def _download_resolved_image(
    resolved_url: _ResolvedResultURL,
    timeout: float,
    max_attempts: int,
    on_retry: Callable[[int, int], None],
) -> bytes:
    """Retry only the GET while reusing the original validated address snapshot."""
    tls_context = ssl.create_default_context() if resolved_url.scheme == "https" else None
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        address = resolved_url.addresses[(attempt - 1) % len(resolved_url.addresses)]
        failure: OpenAIImageAPIError | None = None
        response: http.client.HTTPResponse | None = None
        if tls_context is None:
            connection: http.client.HTTPConnection = _PinnedHTTPConnection(
                resolved_url.hostname,
                resolved_url.port,
                address,
                timeout,
            )
        else:
            connection = _PinnedHTTPSConnection(
                resolved_url.hostname,
                resolved_url.port,
                address,
                timeout,
                tls_context,
            )
        try:
            connection.request(
                "GET",
                resolved_url.request_target,
                headers={
                    "Accept": "image/*",
                    "Host": resolved_url.authority,
                },
                encode_chunked=False,
            )
            response = connection.getresponse()
            response_body = response.read()
            _validate_result_download_contract(response, response_body)
            return response_body
        except OpenAIImageAPIError as exc:
            failure = exc
        except (TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
            failure = _transport_error(exc)
        finally:
            if response is not None:
                response.close()
            connection.close()
        if failure is None:
            raise RuntimeError("result image attempt ended without a result")
        if not failure.retryable or attempt >= attempts:
            raise failure from None
        on_retry(attempt, attempts)
    raise RuntimeError("result image retry loop exhausted")


def atomic_save_validated_image(image_bytes: bytes, target: Path) -> None:
    """Verify response bytes with Pillow before atomically publishing them."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise OpenAIImageAPIError("invalid_image", "GPT Image API returned an invalid image.") from None
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(image_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    except OSError:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise OpenAIImageAPIError("save_failed", "Could not save the generated image.") from None


def _validate_result_download_contract(response: Any, response_body: bytes) -> None:
    status = getattr(response, "status", None)
    if status == 429:
        raise OpenAIImageAPIError(
            "rate_limit",
            "GPT Image API is rate limiting requests.",
            status,
            True,
        )
    if isinstance(status, int) and 500 <= status < 600:
        raise OpenAIImageAPIError(
            "server_error",
            "GPT Image API is temporarily unavailable.",
            status,
            True,
        )
    _validate_response_contract(response, response_body, "image")


def _validate_response_contract(
    response: Any, response_body: bytes, expected_content_type: str
) -> None:
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    content_type = headers.get("Content-Type") if headers is not None else None
    if (
        not isinstance(status, int)
        or not 200 <= status < 300
        or not isinstance(content_type, str)
        or not _is_expected_content_type(content_type, expected_content_type)
        or not response_body
    ):
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned an invalid response."
        )


def _is_expected_content_type(content_type: str, expected: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if expected == "json":
        return media_type == "application/json" or (
            media_type.startswith("application/") and media_type.endswith("+json")
        )
    return media_type.startswith("image/") and len(media_type) > len("image/")


def _build_reference_sheet(image_paths: Sequence[Path]) -> Path:
    """Flatten multiple references for providers that accept one `image` upload."""
    descriptor, raw_path = tempfile.mkstemp(
        prefix="merged-reference-sheet-",
        suffix=".png",
    )
    os.close(descriptor)
    target = Path(raw_path)
    try:
        prepared: list[Image.Image] = []
        for source_path in image_paths:
            with Image.open(source_path) as source:
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                normalized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                prepared.append(normalized.copy())
        if not prepared:
            raise ValueError("no reference images")

        columns = 2 if len(prepared) > 1 else 1
        rows = (len(prepared) + columns - 1) // columns
        padding = 24
        cell_width = max(image.width for image in prepared)
        cell_height = max(image.height for image in prepared)
        canvas = Image.new(
            "RGB",
            (
                columns * cell_width + (columns + 1) * padding,
                rows * cell_height + (rows + 1) * padding,
            ),
            "white",
        )
        for index, image in enumerate(prepared):
            column = index % columns
            row = index // columns
            left = padding + column * (cell_width + padding)
            top = padding + row * (cell_height + padding)
            left += (cell_width - image.width) // 2
            top += (cell_height - image.height) // 2
            canvas.paste(image, (left, top))
        canvas.save(target, format="PNG", optimize=True)
        return target
    except (OSError, UnidentifiedImageError, ValueError):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise OpenAIImageAPIError(
            "invalid_input_image",
            "Could not combine source images for the image edit request.",
        ) from None


def _transport_error(exc: BaseException) -> OpenAIImageAPIError:
    if isinstance(exc, HTTPError):
        if exc.code in {401, 403}:
            return OpenAIImageAPIError(
                "authentication",
                f"GPT Image API authentication failed (HTTP {exc.code}).",
                exc.code,
            )
        if exc.code in {400, 404, 422}:
            return OpenAIImageAPIError(
                "invalid_request",
                f"GPT Image API request or endpoint is invalid (HTTP {exc.code}).",
                exc.code,
            )
        if exc.code == 429:
            return OpenAIImageAPIError(
                "rate_limit",
                "GPT Image API is rate limiting requests (HTTP 429).",
                exc.code,
                True,
            )
        if 500 <= exc.code < 600:
            return OpenAIImageAPIError(
                "server_error",
                f"GPT Image API is temporarily unavailable (HTTP {exc.code}).",
                exc.code,
                True,
            )
        return OpenAIImageAPIError(
            "http_error",
            f"GPT Image API request failed (HTTP {exc.code}).",
            exc.code,
        )
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return OpenAIImageAPIError(
            "timeout",
            "GPT Image API did not respond before the local wait timeout.",
            retryable=True,
        )
    kind = classify_network_error(exc)
    if kind is RetryKind.PERMANENT_TLS:
        return OpenAIImageAPIError("tls_certificate", PERMANENT_TLS_GUIDANCE)
    if kind is RetryKind.TRANSIENT:
        return OpenAIImageAPIError("network", "Could not reach GPT Image API.", retryable=True)
    return OpenAIImageAPIError("network", "Could not reach GPT Image API.")


def _retry_delay(delays: Sequence[float], failed_attempt: int) -> float:
    if not delays:
        return 0.0
    return max(0.0, float(delays[min(failed_attempt - 1, len(delays) - 1)]))




def _raise_unsafe_result_url() -> None:
    raise OpenAIImageAPIError("unsafe_result_url", "Generated image URL is not safe to download.")


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not let a validated public result URL redirect to an unvalidated host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> urllib.request.Request:
        _raise_unsafe_result_url()


def _raise_invalid_base_url() -> None:
    raise OpenAIImageAPIError(
        "invalid_base_url",
        "请填写有效的 GPT Image API 地址。请按服务商提供的地址填写，可带或不带 /v1。",
    )
