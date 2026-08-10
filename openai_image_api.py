"""Validated configuration primitives for OpenAI Images-compatible APIs."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
import http.client
import io
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import ssl
import tempfile
import time
from typing import Any, Callable, Final, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
import urllib.request

from PIL import Image, UnidentifiedImageError

from network_retry import PERMANENT_TLS_GUIDANCE, RetryKind, classify_network_error


DEFAULT_OPENAI_IMAGE_BASE_URL: Final = "https://api.openai.com/v1"
_VALID_RESOLUTIONS: Final = {"1K", "2K", "4K"}
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
    async_edits: bool = False

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OPENAI_IMAGE_BASE_URL,
        model: str = "gpt-image-2",
        resolution: str = "1K",
        timeout: float = 600.0,
        max_attempts: int = 4,
        retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0),
        async_edits: bool = False,
    ) -> None:
        object.__setattr__(self, "_api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "retry_delays", retry_delays)
        object.__setattr__(self, "async_edits", bool(async_edits))

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
        return cls(
            api_key=resolved_key,
            base_url=base_url,
            model=model,
            resolution=resolution,
            async_edits=_is_hapi_image_service(base_url),
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


def _is_hapi_image_service(base_url: str) -> bool:
    """Recognize HAPI's standard and dedicated image gateways."""
    try:
        parsed = urlsplit(base_url)
        return (parsed.hostname or "").rstrip(".").lower() in {
            "hapiopen.cc",
            "image.hapiopen.cc",
        } and parsed.path.rstrip("/") in {"", "/v1"}
    except ValueError:
        return False


def _hapi_images_endpoint(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url)
    prefix = "/v1/images" if (
        (parsed.hostname or "").rstrip(".").lower() == "hapiopen.cc"
        and not parsed.path.rstrip("/")
    ) else "/images"
    return f"{prefix}{suffix}"


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


def _validated_hapi_task_id(value: object) -> str:
    task_id = str(value or "").strip()
    if (
        not task_id
        or len(task_id) > 160
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in task_id
        )
    ):
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned an invalid async task ID."
        )
    return task_id


def _hapi_task_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise OpenAIImageAPIError(
            "invalid_response", "GPT Image API returned no async task result."
        )
    return dict(result)


def _notify_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


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


class OpenAIImageAPI:
    """Minimal transport for the standard OpenAI-compatible Images edits API."""

    def __init__(
        self,
        config: OpenAIImageAPIConfig,
        logger: Any | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self._sleep = sleep or time.sleep
        self._hapi_async_available: bool | None = None

    def generate_edit(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | Path,
        image_size: str = "",
        status_callback: Callable[[str], None] | None = None,
    ) -> GeneratedImage:
        files = _validated_image_paths(image_paths)
        use_hapi_async = bool(self.config.async_edits)
        image_field = "image" if use_hapi_async else "image[]"
        body, content_type = encode_multipart(
            fields={
                "model": self.config.model,
                "prompt": append_aspect_instruction(prompt, image_size),
                "size": self.config.resolution,
            },
            files=[(image_field, path) for path in files],
        )
        if use_hapi_async:
            payload = self._request_hapi_async_edit(
                body,
                content_type,
                status_callback=status_callback,
            )
        else:
            payload = self._request_json(
                "/images/edits",
                body,
                content_type,
                status_callback=status_callback,
            )
        image_bytes = self._extract_image_bytes(payload, status_callback=status_callback)
        target = Path(output_path)
        atomic_save_validated_image(image_bytes, target)
        return GeneratedImage(local_path=str(target), model=self.config.model)

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
        request = urllib.request.Request(
            f"{self.config.base_url}{endpoint}", data=body, method="POST"
        )
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
    ) -> tuple[dict[str, Any], float | None]:
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        response_headers: dict[str, str] = {}
        raw_response = self._request_bytes(
            request,
            opener.open,
            "json",
            status_callback=status_callback,
            response_headers=response_headers,
        )
        return _decode_json_response(raw_response), _retry_after_seconds(response_headers)

    def _request_hapi_async_edit(
        self,
        body: bytes,
        content_type: str,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if self._hapi_async_available is False:
            return self._request_hapi_sync_edit(
                body,
                content_type,
                status_callback=status_callback,
            )
        # An ambiguous retry of a multipart submit can create a second paid job.
        # Submit once, then make all subsequent retries against the safe task GET.
        try:
            payload = self._request_json(
                _hapi_images_endpoint(self.config.base_url, "/edits/async"),
                body,
                content_type,
                max_attempts=1,
                status_callback=status_callback,
            )
        except OpenAIImageAPIError as exc:
            if exc.status_code != 404:
                raise
            self._hapi_async_available = False
            if self.logger is not None:
                self.logger.warning(
                    "HAPI async image tasks are unavailable; falling back to sync edits."
                )
            _notify_status(
                status_callback,
                "↩️ HAPI 未启用异步任务，已切换为同步生成",
            )
            return self._request_hapi_sync_edit(
                body,
                content_type,
                status_callback=status_callback,
            )
        self._hapi_async_available = True
        task_id = _validated_hapi_task_id(payload.get("task_id"))
        _notify_status(status_callback, "📨 GPT Image 任务已提交，正在等待生成")
        if self.logger is not None:
            self.logger.info("GPT Image async task submitted: %s", task_id)

        status = str(payload.get("status") or "").strip().lower()
        if status == "completed":
            return _hapi_task_result(payload)
        if status == "failed":
            self._raise_hapi_task_failure(payload)
        if status not in {"processing", "pending", "queued"}:
            raise OpenAIImageAPIError(
                "invalid_response",
                "GPT Image API returned an invalid async task status.",
            )

        poll_url = (
            f"{self.config.base_url}"
            f"{_hapi_images_endpoint(self.config.base_url, f'/tasks/{task_id}')}"
        )
        poll_interval = 3.0
        max_polls = max(
            1,
            int(max(float(self.config.timeout), poll_interval) / poll_interval),
        )
        deadline = time.monotonic() + max(float(self.config.timeout), poll_interval)
        for poll_number in range(1, max_polls + 1):
            polled, retry_after = self._request_json_url(
                poll_url,
                status_callback=status_callback,
            )
            polled_task_id = polled.get("task_id")
            if polled_task_id not in {None, "", task_id}:
                raise OpenAIImageAPIError(
                    "invalid_response",
                    "GPT Image API returned a mismatched async task.",
                )
            status = str(polled.get("status") or "").strip().lower()
            if self.logger is not None:
                self.logger.info(
                    "GPT Image async task %s: %s (poll %s/%s)",
                    task_id,
                    status or "unknown",
                    poll_number,
                    max_polls,
                )
            if status == "completed":
                _notify_status(status_callback, "✅ GPT Image 生成完成，正在保存图片")
                return _hapi_task_result(polled)
            if status == "failed":
                self._raise_hapi_task_failure(polled)
            if status not in {"processing", "pending", "queued"}:
                raise OpenAIImageAPIError(
                    "invalid_response",
                    "GPT Image API returned an invalid async task status.",
                )
            _notify_status(
                status_callback,
                f"⏳ GPT Image 正在生成（状态检查 {poll_number}/{max_polls}）",
            )
            if poll_number < max_polls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._sleep(min(retry_after or poll_interval, remaining))

        raise OpenAIImageAPIError(
            "timeout",
            "GPT Image async task did not finish before the local wait timeout.",
            retryable=True,
        )

    def _request_hapi_sync_edit(
        self,
        body: bytes,
        content_type: str,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            _hapi_images_endpoint(self.config.base_url, "/edits"),
            body,
            content_type,
            status_callback=status_callback,
        )

    def _raise_hapi_task_failure(self, payload: Mapping[str, Any]) -> None:
        raw_status = payload.get("http_status")
        try:
            status_code = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError):
            status_code = None
        error_value = payload.get("error")
        if isinstance(error_value, Mapping):
            detail = str(
                error_value.get("message") or error_value.get("type") or ""
            ).strip()
        else:
            detail = str(error_value or "").strip()
        detail = " ".join(detail.replace(self.config.api_key, "[redacted]").split())[:300]
        if status_code == 429:
            code = "rate_limit"
            retryable = True
        elif status_code is not None and 500 <= status_code < 600:
            code = "server_error"
            retryable = True
        else:
            code = "task_failed"
            retryable = False
        message = "GPT Image async task failed."
        if detail:
            message = f"GPT Image async task failed: {detail}"
        raise OpenAIImageAPIError(code, message, status_code, retryable)

    def _request_bytes(
        self,
        request: urllib.request.Request,
        open_request: Callable[..., Any] | None = None,
        expected_content_type: str | None = None,
        *,
        max_attempts: int | None = None,
        status_callback: Callable[[str], None] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> bytes:
        attempts = max(
            1,
            int(self.config.max_attempts if max_attempts is None else max_attempts),
        )
        request_opener = open_request or urllib.request.urlopen
        for attempt in range(1, attempts + 1):
            try:
                with request_opener(request, timeout=self.config.timeout) as response:
                    response_body = response.read()
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
                error = _transport_error(exc)
                if not error.retryable or attempt >= attempts:
                    raise error from None
                self._notice_retry(attempt, attempts, status_callback=status_callback)
        raise RuntimeError("image request retry loop exhausted")

    def _notice_retry(
        self,
        attempt: int,
        attempts: int,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        delay = _retry_delay(self.config.retry_delays, attempt)
        if self.logger is not None:
            self.logger.warning("GPT Image request failed (%s/%s); retrying.", attempt, attempts)
        _notify_status(
            status_callback,
            f"🔄 GPT Image 请求失败，正在重试（{attempt}/{attempts}）",
        )
        self._sleep(delay)

    def _extract_image_bytes(
        self,
        payload: dict[str, Any],
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> bytes:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise OpenAIImageAPIError("invalid_response", "GPT Image API returned no image.")
        first = data[0]
        if not isinstance(first, dict):
            raise OpenAIImageAPIError("invalid_response", "GPT Image API returned an invalid image.")
        encoded = first.get("b64_json")
        if isinstance(encoded, str) and encoded:
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise OpenAIImageAPIError(
                    "invalid_response", "GPT Image API returned invalid image data."
                ) from None
        remote_url = first.get("url")
        if not isinstance(remote_url, str) or not remote_url:
            raise OpenAIImageAPIError("invalid_response", "GPT Image API returned no image data.")
        resolved_url = validate_remote_image_url(remote_url)
        return _download_resolved_image(
            resolved_url,
            self.config.timeout,
            self.config.max_attempts,
            lambda attempt, attempts: self._notice_retry(
                attempt,
                attempts,
                status_callback=status_callback,
            ),
        )


def append_aspect_instruction(prompt: str, image_size: str) -> str:
    """Keep the spreadsheet's requested ratio in the provider prompt without resizing."""
    source = str(prompt or "").strip()
    aspect = str(image_size or "").strip()
    if not aspect:
        return source
    return f"{source}\n\nPreserve the requested image aspect ratio exactly: {aspect}."


def encode_multipart(
    fields: Mapping[str, str], files: Sequence[tuple[str, Path]]
) -> tuple[bytes, str]:
    """Create a conventional multipart body using a fresh opaque boundary."""
    boundary = secrets.token_hex(24)
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{_safe_multipart_token(name)}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ))
    for field_name, path in files:
        filename = _safe_filename(path.name)
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        chunks.extend((
            f"--{boundary}\r\n".encode("ascii"),
            (
                "Content-Disposition: form-data; "
                f'name="{_safe_multipart_token(field_name)}"; filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {media_type}\r\n\r\n".encode("ascii"),
            path.read_bytes(),
            b"\r\n",
        ))
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


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


def _validated_image_paths(image_paths: Sequence[str | Path]) -> list[Path]:
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise OpenAIImageAPIError("missing_input_image", "At least one source image is required.")
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, SyntaxError, ValueError):
            raise OpenAIImageAPIError(
                "invalid_input_image", "Each source file must be a non-empty readable image."
            ) from None
    return paths


def _transport_error(exc: BaseException) -> OpenAIImageAPIError:
    if isinstance(exc, HTTPError):
        if exc.code in {401, 403}:
            return OpenAIImageAPIError("authentication", "GPT Image API authentication failed.", exc.code)
        if exc.code in {400, 404}:
            return OpenAIImageAPIError("invalid_request", "GPT Image API request or endpoint is invalid.", exc.code)
        if exc.code == 429:
            return OpenAIImageAPIError("rate_limit", "GPT Image API is rate limiting requests.", exc.code, True)
        if 500 <= exc.code < 600:
            return OpenAIImageAPIError("server_error", "GPT Image API is temporarily unavailable.", exc.code, True)
        return OpenAIImageAPIError("http_error", "GPT Image API request failed.", exc.code)
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


def _safe_multipart_token(value: str) -> str:
    return str(value).replace("\r", "").replace("\n", "").replace('"', "")


def _safe_filename(filename: str) -> str:
    safe = _safe_multipart_token(Path(filename).name)
    return safe or "image"


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
