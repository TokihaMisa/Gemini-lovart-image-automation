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


DEFAULT_OPENAI_IMAGE_BASE_URL: Final = "https://hapiopen.cc/v1"
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

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OPENAI_IMAGE_BASE_URL,
        model: str = "gpt-image-2",
        resolution: str = "1K",
        timeout: float = 600.0,
        max_attempts: int = 4,
        retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0),
    ) -> None:
        object.__setattr__(self, "_api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "retry_delays", retry_delays)

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
        return cls(
            api_key=resolved_key,
            base_url=normalize_openai_image_base_url(section.get("base_url")),
            model=model,
            resolution=resolution,
        )


def normalize_openai_image_base_url(value: object | None) -> str:
    """Validate an endpoint and return its sole terminal OpenAI API ``/v1`` path."""
    raw_value = DEFAULT_OPENAI_IMAGE_BASE_URL if value is None else str(value)
    if "\r" in raw_value or "\n" in raw_value:
        _raise_invalid_base_url()

    cleaned = raw_value.strip() or DEFAULT_OPENAI_IMAGE_BASE_URL
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

    path_segments = [segment for segment in parsed.path.split("/") if segment]
    if any(segment.lower() == "v1" for segment in path_segments[:-1]):
        _raise_invalid_base_url()
    if not path_segments or path_segments[-1].lower() != "v1":
        path_segments.append("v1")

    path = "/" + "/".join(path_segments)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


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

    def generate_edit(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | Path,
        image_size: str = "",
    ) -> GeneratedImage:
        files = _validated_image_paths(image_paths)
        body, content_type = encode_multipart(
            fields={
                "model": self.config.model,
                "prompt": append_aspect_instruction(prompt, image_size),
                "size": self.config.resolution,
            },
            files=[("image[]", path) for path in files],
        )
        payload = self._request_json("/images/edits", body, content_type)
        image_bytes = self._extract_image_bytes(payload)
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

    def _request_json(self, endpoint: str, body: bytes, content_type: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url}{endpoint}", data=body, method="POST"
        )
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        request.add_header("Content-Type", content_type)
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        raw_response = self._request_bytes(request, opener.open, "json")
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

    def _request_bytes(
        self,
        request: urllib.request.Request,
        open_request: Callable[..., Any] | None = None,
        expected_content_type: str | None = None,
    ) -> bytes:
        attempts = max(1, int(self.config.max_attempts))
        request_opener = open_request or urllib.request.urlopen
        for attempt in range(1, attempts + 1):
            try:
                with request_opener(request, timeout=self.config.timeout) as response:
                    response_body = response.read()
                    if expected_content_type is not None:
                        _validate_response_contract(
                            response, response_body, expected_content_type
                        )
                    return response_body
            except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
                error = _transport_error(exc)
                if not error.retryable or attempt >= attempts:
                    raise error from None
                self._notice_retry(attempt, attempts)
        raise RuntimeError("image request retry loop exhausted")

    def _notice_retry(self, attempt: int, attempts: int) -> None:
        delay = _retry_delay(self.config.retry_delays, attempt)
        if self.logger is not None:
            self.logger.warning("GPT Image request failed (%s/%s); retrying.", attempt, attempts)
        self._sleep(delay)

    def _extract_image_bytes(self, payload: dict[str, Any]) -> bytes:
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
        return _download_resolved_image(resolved_url, self.config.timeout)


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
    resolved_url: _ResolvedResultURL, timeout: float
) -> bytes:
    """Fetch once from the prevalidated address set without another resolution."""
    tls_context = ssl.create_default_context() if resolved_url.scheme == "https" else None
    last_error: OpenAIImageAPIError | None = None
    for address in resolved_url.addresses:
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
            _validate_response_contract(response, response_body, "image")
            return response_body
        except OpenAIImageAPIError:
            raise
        except (TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
            last_error = _transport_error(exc)
            if last_error.code == "tls_certificate":
                raise last_error from None
        finally:
            if response is not None:
                response.close()
            connection.close()
    if last_error is not None:
        raise last_error from None
    raise OpenAIImageAPIError("network", "Could not reach GPT Image API.")


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
    raise OpenAIImageAPIError("invalid_base_url", "GPT Image API 地址无效。")
