import base64
import io
import json
import socket
import ssl
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import pytest
from PIL import Image

from openai_image_api import (
    OpenAIImageAPI,
    OpenAIImageAPIConfig,
    OpenAIImageAPIError,
    normalize_openai_image_base_url,
    validate_remote_image_url,
)


VALID_ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8Dw"
    "HwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)
CORRUPT_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Y9Z4WQAAAAASUVORK5CYII="
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        status: int = 200,
        content_type: str = "application/json",
    ):
        self.body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class TrackingBytesIO(io.BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.was_closed = False

    def close(self) -> None:
        self.was_closed = True
        super().close()


class FakeConnectedSocket:
    def __init__(self, response: bytes, peer_address: str, connect_error=None):
        self.response_file = TrackingBytesIO(response)
        self.peer_address = peer_address
        self.connect_error = connect_error
        self.timeout = None
        self.connected_to = None
        self.sent = bytearray()
        self.was_closed = False

    def settimeout(self, timeout) -> None:
        self.timeout = timeout

    def connect(self, address) -> None:
        self.connected_to = address
        if self.connect_error is not None:
            raise self.connect_error

    def getpeername(self):
        if ":" in self.peer_address:
            return (self.peer_address, self.connected_to[1], 0, 0)
        return (self.peer_address, self.connected_to[1])

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def makefile(self, mode: str):
        assert mode == "rb"
        return self.response_file

    def close(self) -> None:
        self.was_closed = True


class FakeSocketFactory:
    def __init__(self, sockets):
        self.sockets = list(sockets)
        self.calls = []

    def __call__(self, family, sock_type):
        self.calls.append((family, sock_type))
        return self.sockets.pop(0)


class FakeDefaultSSLContext:
    def __init__(self, wrap_error=None):
        self.check_hostname = True
        self.verify_mode = ssl.CERT_REQUIRED
        self.wrap_error = wrap_error
        self.wrap_calls = []

    def wrap_socket(self, connected_socket, *, server_hostname):
        self.wrap_calls.append((connected_socket, server_hostname))
        if self.wrap_error is not None:
            raise self.wrap_error
        return connected_socket


def raw_http_response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "image/png",
    extra_headers: bytes = b"",
) -> bytes:
    reason = b"OK" if status == 200 else b"Found"
    return b"".join((
        b"HTTP/1.1 " + str(status).encode("ascii") + b" " + reason + b"\r\n",
        b"Content-Type: " + content_type.encode("ascii") + b"\r\n",
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n",
        extra_headers,
        b"Connection: close\r\n\r\n",
        body,
    ))


def make_png(path: Path) -> Path:
    path.write_bytes(base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64))
    return path


def fake_png_response() -> FakeResponse:
    return FakeResponse(json.dumps({
        "created": 1,
        "data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}],
    }).encode("utf-8"))


def make_client(*, sleep=None, **overrides) -> OpenAIImageAPI:
    settings = {
        "api_key": "test-key",
        "base_url": "https://hapiopen.cc/v1",
        "timeout": 12.5,
        "retry_delays": (0.0,),
        "async_edits": False,
    }
    settings.update(overrides)
    config = OpenAIImageAPIConfig(
        **settings,
    )
    return OpenAIImageAPI(config, sleep=sleep or (lambda _delay: None))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://hapiopen.cc", "https://hapiopen.cc"),
        ("https://image.hapiopen.cc/", "https://image.hapiopen.cc"),
        ("https://hapiopen.cc/v1/", "https://hapiopen.cc/v1"),
        ("https://gateway.test/root/", "https://gateway.test/root"),
    ],
)
def test_normalize_openai_image_base_url(raw, expected):
    """Fails if validation invents a path segment not supplied by the provider."""
    assert normalize_openai_image_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "https://bad\nhost",
        "https://user:pass@gateway.test",
        "https:///v1",
        "https://gateway.test/v1?debug=true",
        "https://gateway.test/v1#anchor",
        "https://gateway.test:",
    ],
)
def test_normalize_openai_image_base_url_rejects_unsafe_or_ambiguous_urls(raw):
    """Fails if unsafe URL components reach the HTTP client configuration."""
    with pytest.raises(OpenAIImageAPIError) as ctx:
        normalize_openai_image_base_url(raw)

    assert ctx.value.code == "invalid_base_url"


def test_config_requires_key_without_echoing_it():
    """Fails if key-validation errors disclose an authorization secret."""
    with pytest.raises(OpenAIImageAPIError) as ctx:
        OpenAIImageAPIConfig.from_config(
            {"openai_image": {"base_url": "https://hapiopen.cc"}}, api_key=""
        )

    assert ctx.value.code == "missing_key"
    assert "Authorization" not in str(ctx.value)


def test_invalid_url_never_echoes_key_in_error_or_config_repr():
    """Fails if an invalid endpoint leaks the resolved API key through public output."""
    secret = "sensitive-test-key"
    with pytest.raises(OpenAIImageAPIError) as ctx:
        OpenAIImageAPIConfig.from_config(
            {"openai_image": {"base_url": "https://bad host"}}, api_key=secret
        )

    assert secret not in str(ctx.value)
    assert secret not in repr(ctx.value)

    config = OpenAIImageAPIConfig.from_config(
        {"openai_image": {"base_url": "https://api.openai.com/v1"}},
        api_key=secret,
    )
    assert secret not in repr(config)


def test_config_serialization_omits_key_but_authorization_can_access_it():
    """Fails if standard dataclass serialization exposes the resolved API key."""
    secret = "serialization-secret"
    config = OpenAIImageAPIConfig.from_config(
        {"openai_image": {"base_url": "https://api.openai.com/v1"}},
        api_key=secret,
    )

    serialized = asdict(config)
    assert secret not in serialized.values()
    assert "api_key" not in serialized
    assert config.api_key == secret


def test_config_normalizes_defaults_and_provider_values():
    """Fails if provider settings do not become a validated immutable configuration."""
    config = OpenAIImageAPIConfig.from_config(
        {
            "openai_image": {
                "base_url": "https://gateway.test/root/",
                "model": " custom-image-model ",
                "resolution": "2k",
            }
        },
        api_key="  resolved-key  ",
    )

    assert config.base_url == "https://gateway.test/root"
    assert config.model == "custom-image-model"
    assert config.resolution == "2K"
    assert config.api_key == "resolved-key"
    with pytest.raises(AttributeError):
        config.model = "another-model"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://image.hapiopen.cc", True),
        ("https://image.hapiopen.cc/v1", True),
        ("https://hapiopen.cc/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://gateway.test/v1", False),
    ],
)
def test_config_enables_async_edits_only_for_hapi(base_url, expected):
    config = OpenAIImageAPIConfig.from_config(
        {"openai_image": {"base_url": base_url}},
        api_key="test-key",
    )

    assert config.async_edits is expected


def test_config_rejects_unsupported_resolution_without_disclosing_key():
    """Fails if unsupported resolution values are accepted or secrets appear in the error."""
    secret = "resolution-secret"
    with pytest.raises(OpenAIImageAPIError) as ctx:
        OpenAIImageAPIConfig.from_config(
            {"openai_image": {"resolution": "8K"}}, api_key=secret
        )

    assert ctx.value.code == "invalid_resolution"
    assert secret not in str(ctx.value)
    assert secret not in repr(ctx.value)


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_posts_multipart_and_saves_b64_png(build_opener, tmp_path):
    """Fails if a standard Images edit request is not multipart or its image is lost."""
    build_opener.return_value.open.return_value = fake_png_response()

    result = make_client().generate_edit(
        "keep the product exact",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        image_size="4:5",
    )

    request = build_opener.return_value.open.call_args.args[0]
    body = request.data
    assert request.full_url == "https://hapiopen.cc/v1/images/edits"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body and b"gpt-image-2" in body
    assert b'name="size"' in body and b"1K" in body
    assert b'name="image[]"; filename="source.png"' in body
    assert b"keep the product exact" in body and b"4:5" in body
    assert result.local_path == str(tmp_path / "out.png")
    with Image.open(tmp_path / "out.png") as output:
        output.verify()
    assert build_opener.return_value.open.call_args.kwargs == {"timeout": 12.5}


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_uses_provider_root_without_inserting_v1(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({
            "task_id": "imgtask_test123",
            "status": "processing",
            "poll_url": "/images/tasks/imgtask_test123",
        }).encode("utf-8"), status=202),
        FakeResponse(json.dumps({
            "task_id": "imgtask_test123",
            "status": "completed",
            "result": {
                "data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}],
            },
        }).encode("utf-8")),
    ]

    sleeps = []
    make_client(
        base_url="https://image.hapiopen.cc",
        sleep=sleeps.append,
        async_edits=True,
    ).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
    )

    submit = build_opener.return_value.open.call_args_list[0].args[0]
    poll = build_opener.return_value.open.call_args_list[1].args[0]
    assert submit.full_url == "https://image.hapiopen.cc/images/edits/async"
    assert submit.method == "POST"
    assert b'name="image"; filename="source.png"' in submit.data
    assert b'name="image[]"' not in submit.data
    assert poll.full_url == "https://image.hapiopen.cc/images/tasks/imgtask_test123"
    assert poll.method == "GET"
    assert poll.get_header("Authorization") == "Bearer test-key"
    assert sleeps == []


@patch("openai_image_api.urllib.request.build_opener")
def test_hapi_main_root_async_edit_inserts_documented_v1_image_path(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({
            "task_id": "imgtask_mainroot",
            "status": "processing",
        }).encode("utf-8"), status=202),
        FakeResponse(json.dumps({
            "task_id": "imgtask_mainroot",
            "status": "completed",
            "result": {"data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}]},
        }).encode("utf-8")),
    ]

    make_client(
        base_url="https://hapiopen.cc",
        async_edits=True,
    ).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert requests[0].full_url == "https://hapiopen.cc/v1/images/edits/async"
    assert requests[1].full_url == "https://hapiopen.cc/v1/images/tasks/imgtask_mainroot"


@patch("openai_image_api.urllib.request.build_opener")
def test_hapi_async_edit_polls_processing_task_without_resubmitting(build_opener, tmp_path):
    processing_response = FakeResponse(json.dumps({
        "task_id": "imgtask_slow456",
        "status": "processing",
    }).encode("utf-8"))
    processing_response.headers["Retry-After"] = "5"
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({
            "task_id": "imgtask_slow456",
            "status": "processing",
            "poll_url": "/images/tasks/imgtask_slow456",
        }).encode("utf-8"), status=202),
        processing_response,
        FakeResponse(json.dumps({
            "task_id": "imgtask_slow456",
            "status": "completed",
            "result": {
                "data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}],
            },
        }).encode("utf-8")),
    ]
    sleeps = []
    statuses = []

    result = make_client(
        base_url="https://image.hapiopen.cc",
        sleep=sleeps.append,
        async_edits=True,
    ).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        status_callback=statuses.append,
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert sum(request.full_url.endswith("/images/edits/async") for request in requests) == 1
    assert sleeps == [5.0]
    assert statuses == [
        "📨 GPT Image 任务已提交，正在等待生成",
        "⏳ GPT Image 正在生成（状态检查 1/4）",
        "✅ GPT Image 生成完成，正在保存图片",
    ]
    assert result.local_path == str(tmp_path / "out.png")


@patch("openai_image_api.urllib.request.build_opener")
def test_hapi_async_edit_surfaces_failed_task_reason(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({
            "task_id": "imgtask_failed789",
            "status": "processing",
            "poll_url": "/images/tasks/imgtask_failed789",
        }).encode("utf-8"), status=202),
        FakeResponse(json.dumps({
            "task_id": "imgtask_failed789",
            "status": "failed",
            "http_status": 503,
            "error": {"message": "upstream image model unavailable"},
        }).encode("utf-8")),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(
            base_url="https://image.hapiopen.cc",
            async_edits=True,
        ).generate_edit(
            "prompt",
            [make_png(tmp_path / "source.png")],
            tmp_path / "out.png",
        )

    assert ctx.value.code == "server_error"
    assert ctx.value.status_code == 503
    assert "upstream image model unavailable" in str(ctx.value)
    assert build_opener.return_value.open.call_count == 2


@patch("openai_image_api.urllib.request.build_opener")
def test_hapi_async_disabled_falls_back_to_sync_once_per_client(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        HTTPError(
            "https://image.hapiopen.cc/images/edits/async",
            404,
            "async image tasks are not enabled",
            {},
            None,
        ),
        fake_png_response(),
        fake_png_response(),
    ]
    statuses = []
    client = make_client(
        base_url="https://image.hapiopen.cc",
        async_edits=True,
    )
    source = make_png(tmp_path / "source.png")

    client.generate_edit(
        "first",
        [source],
        tmp_path / "first.png",
        status_callback=statuses.append,
    )
    client.generate_edit(
        "second",
        [source],
        tmp_path / "second.png",
        status_callback=statuses.append,
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.full_url for request in requests] == [
        "https://image.hapiopen.cc/images/edits/async",
        "https://image.hapiopen.cc/images/edits",
        "https://image.hapiopen.cc/images/edits",
    ]
    assert all(b'name="image";' in request.data for request in requests)
    assert statuses.count("↩️ HAPI 未启用异步任务，已切换为同步生成") == 1


@patch("openai_image_api.urllib.request.build_opener")
def test_hapi_multiple_reference_images_are_uploaded_as_one_contact_sheet(
    build_opener,
    tmp_path,
):
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "task_id": "imgtask_contactsheet",
        "status": "completed",
        "result": {"data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}]},
    }).encode("utf-8"), status=202)
    first = make_png(tmp_path / "first.png")
    second = make_png(tmp_path / "second.png")

    make_client(
        base_url="https://image.hapiopen.cc",
        async_edits=True,
    ).generate_edit(
        "prompt",
        [first, second],
        tmp_path / "out.png",
    )

    request = build_opener.return_value.open.call_args.args[0]
    assert request.data.count(b'name="image"; filename=') == 1
    assert b'hapi-reference-sheet-' in request.data
    assert b'name="image[]"' not in request.data


@patch("openai_image_api.urllib.request.build_opener")
def test_generic_openai_multiple_reference_images_remain_separate_files(
    build_opener,
    tmp_path,
):
    build_opener.return_value.open.return_value = fake_png_response()

    make_client(base_url="https://gateway.test/v1").generate_edit(
        "prompt",
        [make_png(tmp_path / "first.png"), make_png(tmp_path / "second.png")],
        tmp_path / "out.png",
    )

    request = build_opener.return_value.open.call_args.args[0]
    assert request.data.count(b'name="image[]"; filename=') == 2


@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (302, "application/json", fake_png_response().body),
        (200, "text/html", fake_png_response().body),
        (200, "text/problem+json", fake_png_response().body),
        (200, "application/problem+json", b""),
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_rejects_invalid_api_response_contract(
    build_opener, status, content_type, body, tmp_path
):
    """Fails if an authenticated edit response is parsed without HTTP validation."""
    build_opener.return_value.open.return_value = FakeResponse(body, status, content_type)

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_response"
    assert "test-key" not in str(ctx.value)


@patch("openai_image_api.urllib.request.build_opener")
@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_uses_redirect_rejecting_transport_for_authenticated_post(
    urlopen, build_opener, tmp_path
):
    """Fails if the credentialed edit POST can use urllib's redirect-following opener."""
    urlopen.side_effect = AssertionError("authenticated request must not use urlopen")
    build_opener.return_value.open.return_value = fake_png_response()

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert result.local_path == str(tmp_path / "out.png")
    request = build_opener.return_value.open.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer test-key"
    assert build_opener.return_value.open.call_args.kwargs == {"timeout": 12.5}


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_retries_429_but_not_401(build_opener, tmp_path):
    """Fails if permanent authentication failures can create extra paid image jobs."""
    source = make_png(tmp_path / "source.png")
    build_opener.return_value.open.side_effect = [
        HTTPError("https://hapiopen.cc/v1/images/edits", 429, "busy", {}, None),
        fake_png_response(),
    ]

    make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert build_opener.return_value.open.call_count == 2

    build_opener.return_value.open.reset_mock()
    build_opener.return_value.open.side_effect = HTTPError(
        "https://hapiopen.cc/v1/images/edits", 401, "unauthorized", {}, None
    )
    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out-401.png")

    assert ctx.value.code == "authentication"
    assert build_opener.return_value.open.call_count == 1


@pytest.mark.parametrize(
    "remote_url",
    [
        "http://127.0.0.1/result.png",
        "https://localhost/result.png",
        "https://10.0.0.1/result.png",
        "https://169.254.1.1/result.png",
        "https://240.0.0.1/result.png",
        "file:///tmp/result.png",
    ],
)
def test_result_url_rejects_private_network_before_download(remote_url):
    """Fails if a gateway result URL can make the client fetch an internal host."""
    with pytest.raises(OpenAIImageAPIError) as ctx:
        validate_remote_image_url(remote_url)

    assert ctx.value.code == "unsafe_result_url"


@patch("openai_image_api.socket.getaddrinfo")
def test_result_url_rejects_hostname_when_any_resolved_address_is_private(getaddrinfo):
    """Fails if dual-stack DNS lets one unsafe result address bypass the public check."""
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        validate_remote_image_url("https://images.example.test/generated.png")

    assert ctx.value.code == "unsafe_result_url"


@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, "224.0.0.1"),
        (socket.AF_INET6, "ff02::1"),
        (socket.AF_INET6, "fec0::1"),
        (socket.AF_INET6, "::127.0.0.1"),
        (socket.AF_INET6, "::7f00:1"),
        (socket.AF_INET6, "::ffff:127.0.0.1"),
        (socket.AF_INET6, "::ffff:10.0.0.1"),
        (socket.AF_INET6, "::ffff:93.184.216.34"),
        (socket.AF_INET6, "2002:7f00:1::"),
        (socket.AF_INET6, "2002:5db8:d822::"),
        (socket.AF_INET6, "2001:0000:4136:e378:8000:63bf:3fff:fdd2"),
        (socket.AF_INET6, "64:ff9b::5db8:d822"),
        (socket.AF_INET6, "64:ff9b:1::5db8:d822"),
        (socket.AF_INET6, "2606:2800::5efe:5db8:d822"),
    ],
)
@patch("openai_image_api.socket.getaddrinfo")
def test_result_url_rejects_explicitly_unsafe_and_transition_addresses(
    getaddrinfo, family, address
):
    """Address safety must not rely on a broad is_global classification alone."""
    socket_address = (
        (address, 443)
        if family == socket.AF_INET
        else (address, 443, 0, 0)
    )
    getaddrinfo.return_value = [
        (family, socket.SOCK_STREAM, 6, "", socket_address),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        validate_remote_image_url("https://images.example.test/generated.png")

    assert ctx.value.code == "unsafe_result_url"


@patch("openai_image_api.socket.getaddrinfo")
def test_result_url_accepts_normal_public_native_ipv4_and_ipv6(getaddrinfo):
    """Conservative transition filtering must retain ordinary public addresses."""
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            6,
            "",
            ("2606:2800:220:1:248:1893:25c8:1946", 443, 0, 0),
        ),
    ]

    resolved = validate_remote_image_url("https://images.example.test/generated.png")

    assert [address.ip_literal for address in resolved.addresses] == [
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
    ]


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_pins_public_result_connection_and_preserves_https_identity(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """The validated address must be the address used with verified TLS and no key."""
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    connected_socket = FakeConnectedSocket(
        raw_http_response(png), "93.184.216.34"
    )
    socket_factory = FakeSocketFactory([connected_socket])
    socket_constructor.side_effect = socket_factory
    tls_context = FakeDefaultSSLContext()
    create_default_context.return_value = tls_context
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 8443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "created": 1,
        "data": [{"url": "https://images.example.test:8443/generated.png?sig=opaque"}],
    }).encode("utf-8"))

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert result.local_path == str(tmp_path / "out.png")
    getaddrinfo.assert_called_once_with(
        "images.example.test", 8443, type=socket.SOCK_STREAM
    )
    assert socket_factory.calls == [(socket.AF_INET, socket.SOCK_STREAM)]
    assert connected_socket.connected_to == ("93.184.216.34", 8443)
    assert connected_socket.timeout == 12.5
    create_default_context.assert_called_once_with()
    assert tls_context.wrap_calls == [(connected_socket, "images.example.test")]
    assert tls_context.check_hostname is True
    assert tls_context.verify_mode == ssl.CERT_REQUIRED
    assert b"GET /generated.png?sig=opaque HTTP/1.1\r\n" in connected_socket.sent
    assert b"Host: images.example.test:8443\r\n" in connected_socket.sent
    assert b"Authorization:" not in connected_socket.sent
    assert connected_socket.was_closed
    assert connected_socket.response_file.was_closed
    with Image.open(tmp_path / "out.png") as output:
        output.verify()


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_does_not_follow_changed_dns_answer_to_private_address(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """A second DNS answer cannot replace the public address selected by validation."""
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    connected_socket = FakeConnectedSocket(
        raw_http_response(png), "93.184.216.34"
    )
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.side_effect = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert getaddrinfo.call_count == 1
    assert connected_socket.connected_to == ("93.184.216.34", 443)
    assert b"Host: images.example.test\r\n" in connected_socket.sent
    assert b"Host: images.example.test:443\r\n" not in connected_socket.sent


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_tries_only_prevalidated_addresses_and_closes_failures(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """Address fallback stays inside one DNS snapshot and cleans each connection."""
    sleep_calls = []
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    first_socket = FakeConnectedSocket(
        b"", "93.184.216.34", ConnectionRefusedError("first address unavailable")
    )
    second_socket = FakeConnectedSocket(
        raw_http_response(png), "93.184.216.35"
    )
    socket_constructor.side_effect = FakeSocketFactory([first_socket, second_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert getaddrinfo.call_count == 1
    assert build_opener.return_value.open.call_count == 1
    assert sleep_calls == [0.25]
    assert first_socket.connected_to == ("93.184.216.34", 443)
    assert second_socket.connected_to == ("93.184.216.35", 443)
    assert first_socket.was_closed
    assert second_socket.was_closed


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_retries_timeout_without_repeating_edit_or_dns(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """A transient pinned connect timeout retries only the result GET."""
    sleep_calls = []
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    timed_out_socket = FakeConnectedSocket(
        b"", "93.184.216.34", socket.timeout("injected timeout")
    )
    successful_socket = FakeConnectedSocket(
        raw_http_response(png), "93.184.216.34"
    )
    socket_constructor.side_effect = FakeSocketFactory(
        [timed_out_socket, successful_socket]
    )
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert build_opener.return_value.open.call_count == 1
    assert getaddrinfo.call_count == 1
    assert sleep_calls == [0.25]
    assert timed_out_socket.connected_to == ("93.184.216.34", 443)
    assert successful_socket.connected_to == ("93.184.216.34", 443)
    assert timed_out_socket.was_closed
    assert successful_socket.was_closed


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_retries_429_then_succeeds_without_repeating_edit(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """A result-host 429 is retried without repeating the paid edit POST."""
    sleep_calls = []
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    rate_limited_socket = FakeConnectedSocket(
        raw_http_response(b"busy", status=429, content_type="text/plain"),
        "93.184.216.34",
    )
    successful_socket = FakeConnectedSocket(
        raw_http_response(png), "93.184.216.34"
    )
    socket_constructor.side_effect = FakeSocketFactory(
        [rate_limited_socket, successful_socket]
    )
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert build_opener.return_value.open.call_count == 1
    assert getaddrinfo.call_count == 1
    assert sleep_calls == [0.25]
    assert rate_limited_socket.was_closed
    assert rate_limited_socket.response_file.was_closed
    assert successful_socket.was_closed


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_exhausts_recoverable_5xx_with_exact_policy(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """Recoverable result-host failures stop exactly at the configured bound."""
    sleep_calls = []
    response_sockets = [
        FakeConnectedSocket(
            raw_http_response(b"unavailable", status=status, content_type="text/plain"),
            "93.184.216.34",
        )
        for status in (500, 503, 502)
    ]
    socket_constructor.side_effect = FakeSocketFactory(response_sockets)
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(
            sleep=sleep_calls.append,
            max_attempts=3,
            retry_delays=(0.25, 0.5),
        ).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "server_error"
    assert ctx.value.status_code == 502
    assert build_opener.return_value.open.call_count == 1
    assert getaddrinfo.call_count == 1
    assert sleep_calls == [0.25, 0.5]
    assert all(sock.was_closed for sock in response_sockets)
    assert all(sock.response_file.was_closed for sock in response_sockets)


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404])
@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_does_not_retry_permanent_http_status(
    build_opener,
    getaddrinfo,
    socket_constructor,
    create_default_context,
    status,
    tmp_path,
):
    """Redirects and permanent result-host statuses receive one pinned attempt."""
    sleep_calls = []
    failed_socket = FakeConnectedSocket(
        raw_http_response(b"failure", status=status, content_type="text/plain"),
        "93.184.216.34",
    )
    socket_constructor.side_effect = FakeSocketFactory([failed_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(
            sleep=sleep_calls.append,
            max_attempts=3,
            retry_delays=(0.25, 0.5),
        ).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_response"
    assert build_opener.return_value.open.call_count == 1
    assert getaddrinfo.call_count == 1
    assert sleep_calls == []
    assert failed_socket.was_closed
    assert failed_socket.response_file.was_closed


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_rejects_peer_mismatch_and_closes_connection(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """Response bytes are never accepted from a peer other than the pinned address."""
    connected_socket = FakeConnectedSocket(
        raw_http_response(base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)),
        "127.0.0.1",
    )
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "unsafe_result_url"
    assert connected_socket.was_closed
    assert not connected_socket.sent
    assert not (tmp_path / "out.png").exists()


@pytest.mark.parametrize(
    ("status", "content_type", "body", "extra_headers"),
    [
        (
            302,
            "image/png",
            base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64),
            b"Location: http://127.0.0.1/private.png\r\n",
        ),
        (200, "text/plain", base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64), b""),
        (200, "image/png", b"", b""),
    ],
)
@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_rejects_invalid_result_download_contract(
    build_opener,
    getaddrinfo,
    socket_constructor,
    create_default_context,
    status,
    content_type,
    body,
    extra_headers,
    tmp_path,
):
    """Fails if provider result bytes skip status, media-type, or empty-body checks."""
    connected_socket = FakeConnectedSocket(
        raw_http_response(
            body,
            status=status,
            content_type=content_type,
            extra_headers=extra_headers,
        ),
        "93.184.216.34",
    )
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_response"
    assert "test-key" not in str(ctx.value)
    assert connected_socket.was_closed
    assert connected_socket.response_file.was_closed


@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_http_ipv6_result_uses_pinned_literal_and_original_idna_authority(
    build_opener, getaddrinfo, socket_constructor, tmp_path
):
    """IPv6 and IDNA URLs retain the public hostname while connecting by literal."""
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    connected_socket = FakeConnectedSocket(
        raw_http_response(png), "2606:2800:220:1:248:1893:25c8:1946"
    )
    socket_factory = FakeSocketFactory([connected_socket])
    socket_constructor.side_effect = socket_factory
    getaddrinfo.return_value = [(
        socket.AF_INET6,
        socket.SOCK_STREAM,
        6,
        "",
        ("2606:2800:220:1:248:1893:25c8:1946", 8080, 0, 0),
    )]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "http://b\u00fccher.example:8080/generated.png"}],
    }).encode("utf-8"))

    make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    getaddrinfo.assert_called_once_with("xn--bcher-kva.example", 8080, type=socket.SOCK_STREAM)
    assert socket_factory.calls == [(socket.AF_INET6, socket.SOCK_STREAM)]
    assert connected_socket.connected_to == (
        "2606:2800:220:1:248:1893:25c8:1946",
        8080,
        0,
        0,
    )
    assert b"Host: xn--bcher-kva.example:8080\r\n" in connected_socket.sent


@pytest.mark.parametrize(
    ("remote_url", "port", "expected_host"),
    [
        (
            "https://[2606:2800:220:1:248:1893:25c8:1946]/generated.png",
            443,
            b"Host: [2606:2800:220:1:248:1893:25c8:1946]\r\n",
        ),
        (
            "https://[2606:2800:220:1:248:1893:25c8:1946]:8443/generated.png",
            8443,
            b"Host: [2606:2800:220:1:248:1893:25c8:1946]:8443\r\n",
        ),
    ],
)
@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_https_ipv6_literal_preserves_bracketed_host_authority(
    build_opener,
    getaddrinfo,
    socket_constructor,
    create_default_context,
    remote_url,
    port,
    expected_host,
    tmp_path,
):
    """IPv6 literal Host syntax remains bracketed for default and explicit ports."""
    ipv6 = "2606:2800:220:1:248:1893:25c8:1946"
    png = base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    connected_socket = FakeConnectedSocket(raw_http_response(png), ipv6)
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    tls_context = FakeDefaultSSLContext()
    create_default_context.return_value = tls_context
    getaddrinfo.return_value = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ipv6, port, 0, 0)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": remote_url}],
    }).encode("utf-8"))

    make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert expected_host in connected_socket.sent
    assert tls_context.wrap_calls == [(connected_socket, ipv6)]


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_result_download_closes_raw_socket_when_tls_wrap_fails(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """A TLS handshake failure must not leak the already-connected raw socket."""
    connected_socket = FakeConnectedSocket(b"", "93.184.216.34")
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    create_default_context.return_value = FakeDefaultSSLContext(
        ssl.SSLError("injected TLS wrap failure")
    )
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "network"
    assert connected_socket.was_closed
    assert not connected_socket.sent


@patch("openai_image_api.ssl.create_default_context")
@patch("openai_image_api.socket.socket")
@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
def test_downloaded_invalid_image_reaches_decode_rejection_after_cleanup(
    build_opener, getaddrinfo, socket_constructor, create_default_context, tmp_path
):
    """A valid HTTP image contract still requires full decode before publication."""
    connected_socket = FakeConnectedSocket(
        raw_http_response(b"not-an-image"), "93.184.216.34"
    )
    socket_constructor.side_effect = FakeSocketFactory([connected_socket])
    create_default_context.return_value = FakeDefaultSSLContext()
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "data": [{"url": "https://images.example.test/generated.png"}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_image"
    assert connected_socket.was_closed
    assert connected_socket.response_file.was_closed
    assert not (tmp_path / "out.png").exists()


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_rejects_invalid_input_before_network_call(build_opener, tmp_path):
    """Fails if arbitrary files are uploaded as images to the paid endpoint."""
    source = tmp_path / "not-an-image.png"
    source.write_bytes(b"not an image")

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    build_opener.assert_not_called()


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_rejects_checksum_corrupt_input_before_network_call(build_opener, tmp_path):
    """Fails if a structurally PNG-like but corrupt source reaches the paid endpoint."""
    source = tmp_path / "corrupt.png"
    source.write_bytes(base64.b64decode(CORRUPT_PNG_BASE64))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    build_opener.assert_not_called()


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_does_not_replace_existing_output_with_invalid_image(build_opener, tmp_path):
    """Fails if an invalid gateway response overwrites a previously valid output."""
    destination = tmp_path / "out.png"
    original = b"existing-output-must-survive"
    destination.write_bytes(original)
    build_opener.return_value.open.return_value = FakeResponse(json.dumps({
        "created": 1,
        "data": [{"b64_json": base64.b64encode(b"not an image").decode("ascii")}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], destination
        )

    assert ctx.value.code == "invalid_image"
    assert destination.read_bytes() == original


@patch("openai_image_api.urllib.request.build_opener")
def test_test_edit_uses_a_real_png_fixture_and_returns_saved_image(build_opener, tmp_path):
    """Fails if the explicit compatibility probe does not use the normal edit pipeline."""
    build_opener.return_value.open.return_value = fake_png_response()

    result = make_client().test_edit(tmp_path)

    assert Path(result.local_path).parent == tmp_path
    assert Path(result.local_path).is_file()
    request = build_opener.return_value.open.call_args.args[0]
    assert b'name="image[]"; filename="' in request.data
