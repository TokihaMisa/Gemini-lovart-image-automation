import base64
import io
import ipaddress
import json
import socket
import ssl
import threading
import time
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import pytest
from PIL import Image

from openai_image_api import (
    GeneratedImage,
    ImageTaskSnapshot,
    ImageTaskStillRunning,
    MAX_CREATE_BODY_BYTES,
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_TOTAL_BYTES,
    OpenAIImageAPI,
    OpenAIImageAPIConfig,
    OpenAIImageAPIError,
    _build_create_body,
    _encode_reference_images,
    _media_endpoint,
    _protocol_base_url,
    _uses_active_tun_fake_ip_route,
    atomic_save_validated_image,
    append_aspect_instruction,
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
        self._offset = 0

    def __enter__(self):
        self._offset = 0
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _amount=None) -> bytes:
        if _amount is None:
            chunk = self.body[self._offset:]
            self._offset = len(self.body)
            return chunk
        chunk = self.body[self._offset:self._offset + _amount]
        self._offset += len(chunk)
        return chunk

    def read1(self, amount=None) -> bytes:
        return self.read(amount)


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
    return fake_task_response("https://cdn.example/result.png")


def fake_task_response(result_url: str) -> FakeResponse:
    return FakeResponse(json.dumps({
        "task_id": "task-test-1",
        "state": "success",
        "is_final": True,
        "result_url": result_url,
        "result_type": "image",
    }).encode("utf-8"))


@pytest.fixture(autouse=True)
def stub_default_result_download(monkeypatch):
    """Keep transport tests local while security tests use their explicit result host."""
    import openai_image_api as module

    real_validate = module.validate_remote_image_url
    real_download = module._download_resolved_image
    sentinel = object()
    corrupt_sentinel = object()

    def validate(url):
        if url == "https://cdn.example/result.png":
            return sentinel
        if url == "https://cdn.example/corrupt.png":
            return corrupt_sentinel
        return real_validate(url)

    def download(resolved, timeout, max_attempts, notice_retry):
        if resolved is sentinel:
            return base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
        if resolved is corrupt_sentinel:
            return b"not an image"
        return real_download(resolved, timeout, max_attempts, notice_retry)

    monkeypatch.setattr(module, "validate_remote_image_url", validate)
    monkeypatch.setattr(module, "_download_resolved_image", download)


def make_client(*, sleep=None, **overrides) -> OpenAIImageAPI:
    settings = {
        "api_key": "test-key",
        "base_url": "https://api.lk888.ai/v1",
        "timeout": 12.5,
        "retry_delays": (0.0,),
        "merge_reference_images": False,
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
    ("base_url", "create_url", "status_url"),
    [
        ("https://api.lk888.ai", "https://api.lk888.ai/v1/media/generate", "https://api.lk888.ai/v1/media/status?task_id=abc-123"),
        ("https://api.lk888.ai/v1", "https://api.lk888.ai/v1/media/generate", "https://api.lk888.ai/v1/media/status?task_id=abc-123"),
    ],
)
def test_media_endpoints_strip_exactly_one_trailing_v1(base_url, create_url, status_url):
    assert _media_endpoint(base_url, "generate") == create_url
    assert _media_endpoint(base_url, "status", task_id="abc-123") == status_url


def test_static_backstop_allows_only_the_legacy_config_migration_boundary():
    """Dead protocol helpers and user-facing fallback notices must not return."""
    source = Path("openai_image_api.py").read_text(encoding="utf-8")
    assert "def _request_" + "hapi" not in source.lower()
    assert "def _is_" + "hapi_image_service" not in source.lower()
    assert "/images/" + "edits" not in source.lower()
    assert "/images/" + "tasks/" not in source.lower()
    assert "sync fallback" not in source.lower()


def test_build_create_body_uses_documented_json_contract():
    encoded_images = ["data:image/png;base64,cG5n"]
    prompt = append_aspect_instruction("sell it", "2:3")

    payload = json.loads(
        _build_create_body("gpt-image-2", prompt, "1024x1536", encoded_images)
    )

    assert payload == {
        "model": "gpt-image-2",
        "prompt": prompt,
        "params": {
            "images": encoded_images,
            "size": "1024x1536",
            "quality": "auto",
            "n": 1,
        },
    }


def make_image(path: Path, image_format: str) -> Path:
    Image.new("RGB", (2, 2), "navy").save(path, format=image_format)
    return path


@pytest.mark.parametrize(
    ("image_format", "extension", "prefix"),
    [
        ("PNG", ".jpg", "data:image/png;base64,"),
        ("JPEG", ".png", "data:image/jpeg;base64,"),
        ("WEBP", ".jpeg", "data:image/webp;base64,"),
    ],
)
def test_encode_reference_images_uses_verified_content_format(
    tmp_path, image_format, extension, prefix
):
    source = make_image(tmp_path / f"reference{extension}", image_format)

    encoded = _encode_reference_images([source], merge=False)

    assert encoded[0].startswith(prefix)


@patch("openai_image_api.urllib.request.build_opener")
def test_reference_count_is_limited_before_network_access(build_opener, tmp_path):
    references = [make_png(tmp_path / f"reference-{index}.png") for index in range(MAX_REFERENCE_IMAGES)]
    assert len(_encode_reference_images(references, merge=False)) == MAX_REFERENCE_IMAGES

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt",
            references + [make_png(tmp_path / "too-many.png")],
            tmp_path / "out.png",
        )

    assert ctx.value.code == "reference_image_limit"
    build_opener.assert_not_called()


def test_reference_byte_limits_accept_boundaries_and_reject_excess(monkeypatch, tmp_path):
    source = make_png(tmp_path / "source.png")
    raw = source.read_bytes()
    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_BYTES", len(raw))
    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_TOTAL_BYTES", len(raw))
    assert _encode_reference_images([source], merge=False)

    oversized = tmp_path / "oversized.png"
    oversized.write_bytes(raw + b"x")
    with pytest.raises(OpenAIImageAPIError) as ctx:
        _encode_reference_images([oversized], merge=False)
    assert ctx.value.code == "reference_image_too_large"


def test_total_reference_byte_limit_rejects_excess(monkeypatch, tmp_path):
    first = make_png(tmp_path / "first.png")
    second = make_png(tmp_path / "second.png")
    raw = first.read_bytes()
    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_BYTES", len(raw) + 1)
    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_TOTAL_BYTES", len(raw) * 2)
    assert len(_encode_reference_images([first, second], merge=False)) == 2

    second.write_bytes(raw + b"x")
    with pytest.raises(OpenAIImageAPIError) as ctx:
        _encode_reference_images([first, second], merge=False)
    assert ctx.value.code == "reference_total_too_large"


def test_create_body_limit_rejects_utf8_byte_overage(monkeypatch):
    images = ["data:image/png;base64,cG5n"]
    accepted = _build_create_body("model", "你好", "1024x1024", images)
    monkeypatch.setattr("openai_image_api.MAX_CREATE_BODY_BYTES", len(accepted))
    assert _build_create_body("model", "你好", "1024x1024", images) == accepted

    monkeypatch.setattr("openai_image_api.MAX_CREATE_BODY_BYTES", len(accepted) - 1)
    with pytest.raises(OpenAIImageAPIError) as ctx:
        _build_create_body("model", "你好", "1024x1024", images)
    assert ctx.value.code == "create_body_too_large"


def test_reference_merge_is_explicit_and_uses_one_data_url(tmp_path):
    references = [
        make_png(tmp_path / "first.png"),
        make_png(tmp_path / "second.png"),
    ]

    assert len(_encode_reference_images(references, merge=False)) == 2
    merged = _encode_reference_images(references, merge=True)

    assert len(merged) == 1
    assert merged[0].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    ("limit_name", "error_code"),
    [
        ("MAX_REFERENCE_BYTES", "reference_image_too_large"),
        ("MAX_REFERENCE_TOTAL_BYTES", "reference_total_too_large"),
    ],
)
def test_merged_reference_sheet_rechecks_decoded_byte_limits(
    monkeypatch, tmp_path, limit_name, error_code
):
    references = [make_png(tmp_path / "first.png"), make_png(tmp_path / "second.png")]
    original = references[0].read_bytes()
    merged_size = len(original) * 3

    def write_sheet(size):
        sheet = tmp_path / f"sheet-{size}.png"
        sheet.write_bytes(original + b"x" * (size - len(original)))
        return sheet

    def encode_sheet(size):
        monkeypatch.setattr(
            "openai_image_api._build_reference_sheet",
            lambda _paths: write_sheet(size),
        )
        return _encode_reference_images(references, merge=True)

    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_BYTES", merged_size)
    monkeypatch.setattr("openai_image_api.MAX_REFERENCE_TOTAL_BYTES", merged_size)
    assert len(encode_sheet(merged_size)) == 1

    if limit_name == "MAX_REFERENCE_BYTES":
        monkeypatch.setattr("openai_image_api.MAX_REFERENCE_BYTES", merged_size)
        monkeypatch.setattr("openai_image_api.MAX_REFERENCE_TOTAL_BYTES", merged_size + 1)
    else:
        monkeypatch.setattr("openai_image_api.MAX_REFERENCE_BYTES", merged_size + 1)
        monkeypatch.setattr("openai_image_api.MAX_REFERENCE_TOTAL_BYTES", merged_size)
    with pytest.raises(OpenAIImageAPIError) as ctx:
        encode_sheet(merged_size + 1)
    assert ctx.value.code == error_code


@patch("openai_image_api.urllib.request.build_opener")
@pytest.mark.parametrize("raw", [b"not an image", base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)[:-8]])
def test_invalid_reference_images_fail_before_network_access(build_opener, raw, tmp_path):
    source = tmp_path / "invalid.png"
    source.write_bytes(raw)

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    build_opener.assert_not_called()


@patch("openai_image_api.urllib.request.build_opener")
@pytest.mark.parametrize("image_format", ["BMP", "GIF"])
def test_unsupported_pillow_valid_reference_image_fails_before_network_access(
    build_opener, image_format, tmp_path
):
    source = make_image(tmp_path / f"reference.{image_format.lower()}", image_format)

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    build_opener.assert_not_called()


@pytest.mark.parametrize(
    ("base_url", "configured", "expected"),
    [
        ("https://image.hapiopen.cc", None, False),
        ("https://hapiopen.cc/v1", None, False),
        ("https://gateway.test/v1", None, False),
        ("https://gateway.test/v1", True, True),
        ("https://image.hapiopen.cc", False, False),
    ],
)
def test_config_resolves_reference_merge_switch(base_url, configured, expected):
    section = {"base_url": base_url}
    if configured is not None:
        section["merge_reference_images"] = configured

    config = OpenAIImageAPIConfig.from_config(
        {"openai_image": section},
        api_key="test-key",
    )

    assert config.merge_reference_images is expected


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
def test_generate_edit_posts_json_create_body_and_saves_b64_png(build_opener, tmp_path):
    """The media task submit uses JSON Data URLs, never multipart edit fields."""
    build_opener.return_value.open.return_value = fake_png_response()

    result = make_client().generate_edit(
        "keep the product exact",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        image_size="4:5",
    )

    request = build_opener.return_value.open.call_args_list[0].args[0]
    body = request.data
    assert request.full_url == "https://api.lk888.ai/v1/media/generate"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"
    assert b"/images/edits" not in request.full_url.encode()
    assert b"multipart" not in body and b"boundary" not in body
    assert b'"image"' not in body and b'"image[]"' not in body
    assert json.loads(body) == {
        "model": "gpt-image-2",
        "prompt": append_aspect_instruction("keep the product exact", "4:5"),
        "params": {
            "images": _encode_reference_images([tmp_path / "source.png"], merge=False),
            "size": "1024x1280",
            "quality": "auto",
            "n": 1,
        },
    }
    assert result.local_path == str(tmp_path / "out.png")
    with Image.open(tmp_path / "out.png") as output:
        output.verify()
    assert build_opener.return_value.open.call_args.kwargs == {"timeout": 12.5}


@pytest.mark.parametrize("create_payload", [{"task_id": " t-1 "}, {"data": {"task_id": 42}}])
@patch("openai_image_api.urllib.request.build_opener")
def test_create_normalizes_root_or_nested_task_id_and_persists_before_poll(
    build_opener, create_payload, tmp_path
):
    callbacks = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps(create_payload).encode()),
        FakeResponse(json.dumps({
            "task_id": "t-1" if isinstance(create_payload["task_id"] if "task_id" in create_payload else create_payload["data"]["task_id"], str) else "42",
            "state": "success",
            "is_final": True,
            "result_url": "https://cdn.example/result.png",
            "result_type": "image",
        }).encode()),
    ]

    result = make_client(max_attempts=4).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        task_callback=callbacks.append,
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET"]
    assert callbacks[0].state == "pending"
    assert callbacks[0].task_id in {"t-1", "42"}
    assert "test-key" not in repr(callbacks[0])
    assert "test-key" not in json.dumps(asdict(callbacks[0]))
    assert result.task.state == "success"


@patch("openai_image_api.urllib.request.build_opener")
def test_create_prefers_root_task_id_when_data_is_unrelated_mapping(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"root-1","data":{"request":"accepted"}}'),
        FakeResponse(b'{"task_id":"root-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}'),
    ]

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert result.task.task_id == "root-1"


@pytest.mark.parametrize(
    ("side_effect", "expected_code"),
    [
        (socket.timeout("submitted but response lost"), "submission_unknown"),
        (HTTPError("https://api.lk888.ai/v1/media/generate", 400, "bad", {}, None), "submission_unknown"),
        (HTTPError("https://api.lk888.ai/v1/media/generate", 429, "busy", {}, None), "submission_unknown"),
        (HTTPError("https://api.lk888.ai/v1/media/generate", 503, "down", {}, None), "submission_unknown"),
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_create_failure_never_repeats_paid_post(build_opener, side_effect, expected_code, tmp_path):
    build_opener.return_value.open.side_effect = side_effect

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(max_attempts=4).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == expected_code
    assert ctx.value.retryable is False
    assert build_opener.return_value.open.call_count == 1


@pytest.mark.parametrize("payload", [b"not-json", b"{}", b'{"task_id":"bad/id"}', json.dumps({"task_id": "x" * 129}).encode()])
@patch("openai_image_api.urllib.request.build_opener")
def test_invalid_create_response_does_not_poll_or_resubmit(build_opener, payload, tmp_path):
    build_opener.return_value.open.return_value = FakeResponse(payload)

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "submission_unknown"
    assert build_opener.return_value.open.call_count == 1


@patch("openai_image_api.urllib.request.build_opener")
def test_submission_marker_is_persisted_before_paid_post_opens(build_opener, tmp_path):
    order = []

    def open_request(*_args, **_kwargs):
        order.append("post")
        return FakeResponse(b'{"task_id":"t-1"}')

    build_opener.return_value.open.side_effect = open_request
    with pytest.raises(OpenAIImageAPIError):
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
            submission_callback=lambda: order.append("marker"),
        )
    assert order[:2] == ["marker", "post"]


def test_protocol_identity_treats_optional_v1_as_same_gateway():
    assert _protocol_base_url("https://gateway.example") == "https://gateway.example"
    assert _protocol_base_url("https://gateway.example/v1") == "https://gateway.example"
    assert _protocol_base_url("https://other.example") != _protocol_base_url("https://gateway.example")


@patch("openai_image_api.urllib.request.build_opener")
def test_running_task_polls_same_id_then_downloads_and_reports_progress(build_opener, tmp_path):
    task_id = "task-secret-prefix-abc123"
    callbacks = []
    statuses = []
    sleeps = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"task_id": task_id}).encode()),
        FakeResponse(json.dumps({
            "task_id": task_id, "state": "running", "is_final": False,
            "progress": "45%", "status": "处理中",
        }).encode()),
        FakeResponse(json.dumps({
            "task_id": task_id, "state": "success", "is_final": True,
            "result_url": "https://cdn.example/result.png", "result_type": "image",
        }).encode()),
    ]

    result = make_client(sleep=sleeps.append).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        status_callback=statuses.append, task_callback=callbacks.append,
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert all(requests[index].full_url.endswith("task_id=task-secret-prefix-abc123") for index in (1, 2))
    assert sleeps == [5.0]
    assert [task.state for task in callbacks] == ["pending", "running", "success"]
    assert any("45%" in message and "处理中" in message and "已等待" in message and "abc123" in message for message in statuses)
    assert all("task-secret-prefix" not in message for message in statuses)
    assert isinstance(result, GeneratedImage)


@patch("openai_image_api.urllib.request.build_opener")
@pytest.mark.parametrize("status_code", [408, 503])
def test_status_service_error_never_falls_back_to_another_paid_submit(
    build_opener, tmp_path, status_code
):
    task_id = "task-123"
    status_url = "https://api.lk888.ai/v1/media/status?task_id=task-123"
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"task_id": task_id}).encode()),
        HTTPError(status_url, status_code, "unavailable", {}, None),
    ]

    with pytest.raises(ImageTaskStillRunning) as ctx:
        make_client(max_attempts=1).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert ctx.value.code == "task_still_running"
    assert ctx.value.task.task_id == task_id
    assert [(request.method, request.full_url) for request in requests] == [
        ("POST", "https://api.lk888.ai/v1/media/generate"),
        ("GET", status_url),
    ]


@patch("openai_image_api.urllib.request.build_opener")
def test_nested_provider_status_redacts_exact_task_id_from_all_display_callbacks(
    build_opener, tmp_path
):
    task_id = "provider-private-task-12345678"
    statuses = []
    tasks = []
    display_events = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"data": {"task_id": task_id}}).encode()),
        FakeResponse(json.dumps({
            "data": {
                "task_id": task_id,
                "state": "running",
                "is_final": False,
                "progress": f"42% for {task_id}",
                "status": f"rendering {task_id}",
            }
        }).encode()),
        FakeResponse(json.dumps({
            "data": {
                "task_id": task_id,
                "state": "success",
                "is_final": True,
                "result_url": "https://cdn.example/result.png",
                "result_type": "image",
            }
        }).encode()),
    ]

    make_client(sleep=lambda _seconds: None).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        status_callback=statuses.append,
        task_callback=tasks.append,
        display_callback=display_events.append,
    )

    assert all(task_id not in message for message in statuses)
    assert all(task_id not in task.status for task in tasks)
    assert all(task_id not in task.progress for task in tasks)
    assert display_events
    assert all(task_id not in json.dumps(asdict(event)) for event in display_events)
    assert any(event.task_suffix == "12345678" for event in display_events)


@pytest.mark.parametrize(
    ("task_id", "expected_display", "forbidden_display"),
    [
        ("tiny", "hash:8950abfd", "tiny"),
        ("task-secret-prefix-abc123", "x-abc123", "task-secret-prefix"),
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_progress_status_uses_safe_task_display_token(
    build_opener,
    task_id,
    expected_display,
    forbidden_display,
    tmp_path,
):
    statuses = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"task_id": task_id}).encode()),
        FakeResponse(
            json.dumps(
                {
                    "task_id": task_id,
                    "state": "running",
                    "is_final": False,
                    "progress": "45%",
                    "status": "处理中",
                }
            ).encode()
        ),
        FakeResponse(
            json.dumps(
                {
                    "task_id": task_id,
                    "state": "success",
                    "is_final": True,
                    "result_url": "https://cdn.example/result.png",
                    "result_type": "image",
                }
            ).encode()
        ),
    ]

    make_client(sleep=lambda _delay: None).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        status_callback=statuses.append,
    )

    progress_messages = [message for message in statuses if "已等待" in message]
    assert len(progress_messages) == 1
    assert expected_display in progress_messages[0]
    assert forbidden_display not in progress_messages[0]
    assert task_id not in progress_messages[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"task_id": "wrong", "state": "running", "is_final": False},
        {"task_id": "t-1", "state": "running"},
        {"task_id": "t-1", "is_final": False},
        {"task_id": "t-1", "state": "success", "is_final": False, "result_url": "https://cdn.example/result.png"},
        {"task_id": "t-1", "state": "success", "is_final": True},
        {"task_id": "t-1", "state": "failed", "is_final": False},
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_invalid_poll_state_is_terminal_without_resubmission(build_opener, payload, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'),
        FakeResponse(json.dumps(payload).encode()),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_response"
    assert [call.args[0].method for call in build_opener.return_value.open.call_args_list] == ["POST", "GET"]


@patch("openai_image_api.urllib.request.build_opener")
def test_final_failed_task_is_terminal_sanitized_and_not_resubmitted(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'),
        FakeResponse(json.dumps({
            "task_id": "t-1", "state": "failed", "is_final": True,
            "error": "provider rejected task\nBearer test-key", "cost": 0,
        }).encode()),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "task_failed"
    assert ctx.value.retryable is False
    assert "provider rejected task" in str(ctx.value)
    assert "test-key" not in str(ctx.value)
    assert "test-key" not in repr(ctx.value.task)
    assert ctx.value.task.cost == 0
    assert build_opener.return_value.open.call_count == 2


@patch("openai_image_api.urllib.request.build_opener")
def test_provider_fields_and_nested_cost_are_redacted_before_callbacks(build_opener, tmp_path):
    secret = "test-key"
    callbacks = []
    statuses = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'),
        FakeResponse(json.dumps({
            "task_id": "t-1",
            "state": "running",
            "is_final": False,
            "progress": f"45% {secret}",
            "status": f"processing {secret}",
            "status_group": f"active {secret}",
            "result_url": f"https://cdn.example/result.png?token={secret}",
            "result_type": f"image/{secret}",
            "error": f"warning {secret}",
            "cost": {f"key-{secret}": [secret, {"nested": secret}]},
        }).encode()),
        FakeResponse(b'{"task_id":"t-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}'),
    ]

    make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        task_callback=callbacks.append, status_callback=statuses.append,
    )

    assert secret not in repr(callbacks)
    assert secret not in json.dumps([asdict(task) for task in callbacks])
    assert all(secret not in message for message in statuses)


@patch("openai_image_api.urllib.request.build_opener")
def test_percent_encoded_and_case_transformed_secrets_are_recursively_redacted(build_opener, tmp_path):
    secret = "TeSt-Key"
    task_id = "Private-Task-12345678"
    callbacks = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"task_id": task_id}).encode()),
        FakeResponse(json.dumps({
            "task_id": task_id, "state": "running", "is_final": False,
            "status": "key=TEST-KEY",
            "progress": "task=Private%2DTask%2D12345678",
            "cost": {"nested": ["TeSt%2DKey", {"task": "PRIVATE-TASK-12345678"}]},
        }).encode()),
        FakeResponse(json.dumps({"task_id": task_id, "state": "success", "is_final": True,
            "result_url": "https://cdn.example/result.png", "result_type": "image"}).encode()),
    ]
    make_client(api_key=secret).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        task_callback=callbacks.append,
    )
    rendered = json.dumps([asdict(task) for task in callbacks])
    assert "TEST-KEY" not in rendered
    assert "TeSt%2DKey" not in rendered
    assert "Private%2DTask%2D12345678" not in rendered
    assert "PRIVATE-TASK-12345678" not in rendered


@patch("openai_image_api.urllib.request.build_opener")
def test_non_string_repr_secret_variants_are_recursively_redacted(build_opener, tmp_path):
    class ProviderCost:
        def __repr__(self):
            return "ProviderCost(key=TeSt%252DKey, task=PRIVATE-TASK-12345678)"

    callbacks = []
    task_id = "Private-Task-12345678"
    build_opener.return_value.open.side_effect = [
        FakeResponse(json.dumps({"task_id": task_id}).encode()),
        FakeResponse(json.dumps({"task_id": task_id, "state": "running", "is_final": False}).encode()),
        FakeResponse(json.dumps({"task_id": task_id, "state": "success", "is_final": True,
            "result_url": "https://cdn.example/result.png", "result_type": "image"}).encode()),
    ]
    client = make_client(api_key="TeSt-Key", sleep=lambda _delay: None)
    original_parse = client._parse_status_task

    def inject_cost(payload, previous):
        task, url = original_parse(payload, previous)
        if task.state == "running":
            task = ImageTaskSnapshot(
                task.task_id, task.state, task.is_final, task.task_created_at,
                cost=ProviderCost(),
            )
            task = __import__("openai_image_api")._sanitize_task_snapshot(task, "TeSt-Key")
        return task, url

    client._parse_status_task = inject_cost
    client.generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        task_callback=callbacks.append,
    )
    rendered = repr(callbacks)
    assert "TeSt%252DKey" not in rendered
    assert "PRIVATE-TASK-12345678" not in rendered


@patch("openai_image_api.urllib.request.build_opener")
def test_final_credentialed_result_url_downloads_raw_but_exposes_only_redacted_snapshot(
    build_opener, monkeypatch, tmp_path
):
    import openai_image_api as module

    secret = "test-key"
    raw_url = f"https://cdn.example/result.png?token={secret}"
    callbacks = []
    statuses = []
    validated_urls = []
    sentinel = object()

    def validate(url):
        validated_urls.append(url)
        return sentinel

    monkeypatch.setattr(module, "validate_remote_image_url", validate)
    monkeypatch.setattr(
        module,
        "_download_resolved_image",
        lambda resolved, *_args: (
            base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
            if resolved is sentinel
            else (_ for _ in ()).throw(AssertionError("unexpected resolved URL"))
        ),
    )
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'),
        FakeResponse(json.dumps({
            "task_id": "t-1", "state": "success", "is_final": True,
            "result_url": raw_url, "result_type": f"image/{secret}",
        }).encode()),
    ]

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        task_callback=callbacks.append, status_callback=statuses.append,
    )

    assert validated_urls == [raw_url]
    assert result.task == callbacks[-1]
    assert result.task.result_url == "https://cdn.example/result.png?token=[redacted]"
    assert secret not in repr(callbacks)
    assert secret not in repr(result)
    assert all(secret not in message for message in statuses)


@patch("openai_image_api.urllib.request.build_opener")
def test_stale_resume_fields_and_nested_cost_are_redacted_before_callbacks(build_opener, tmp_path):
    secret = "test-key"
    callbacks = []
    statuses = []
    resume = ImageTaskSnapshot(
        "t-1", "running", False, 1787200000.0,
        progress=f"45% {secret}", status=f"processing {secret}",
        status_group=f"active {secret}",
        result_url=f"https://cdn.example/result.png?token={secret}",
        result_type=f"image/{secret}", error=f"warning {secret}",
        cost={f"key-{secret}": [secret, {"nested": secret}]},
    )
    build_opener.return_value.open.return_value = FakeResponse(
        b'{"task_id":"t-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}'
    )

    make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
        resume_task=resume, task_callback=callbacks.append,
        status_callback=statuses.append,
    )

    assert secret not in repr(callbacks)
    assert secret not in json.dumps([asdict(task) for task in callbacks])
    assert all(secret not in message for message in statuses)


@pytest.mark.parametrize("resume", [False, True])
@patch("openai_image_api.urllib.request.build_opener")
def test_api_key_cannot_be_persisted_as_provider_or_stale_task_id(
    build_opener, resume, tmp_path
):
    callbacks = []
    source = make_png(tmp_path / "source.png")
    kwargs = {}
    if resume:
        kwargs["resume_task"] = ImageTaskSnapshot(
            "test-key", "running", False, 1787200000.0
        )
    else:
        build_opener.return_value.open.return_value = FakeResponse(
            b'{"task_id":"test-key"}'
        )

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [source], tmp_path / "out.png",
            task_callback=callbacks.append, **kwargs,
        )

    assert ctx.value.code == "invalid_response"
    assert "test-key" not in str(ctx.value)
    assert "test-key" not in repr(ctx.value)
    assert callbacks == []


@patch("openai_image_api.time.monotonic")
@patch("openai_image_api.urllib.request.build_opener")
def test_polling_uses_five_seconds_through_120_then_ten(build_opener, monotonic, tmp_path):
    now = [0.0]
    monotonic.side_effect = lambda: now[0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    running = FakeResponse(b'{"task_id":"t-1","state":"running","is_final":false}')
    success = FakeResponse(b'{"task_id":"t-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}')
    build_opener.return_value.open.side_effect = [FakeResponse(b'{"task_id":"t-1"}')] + [running] * 26 + [success]

    make_client(sleep=sleep).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert sleeps[:25] == [5.0] * 25
    assert sleeps[24] == 5.0
    assert sleeps[25] == 10.0


@patch("openai_image_api.time.monotonic")
@patch("openai_image_api.urllib.request.build_opener")
def test_600_second_deadline_raises_last_snapshot_without_new_post(build_opener, monotonic, tmp_path):
    now = [0.0]
    monotonic.side_effect = lambda: now[0]
    callbacks = []

    def sleep(delay):
        now[0] += delay

    build_opener.return_value.open.side_effect = [FakeResponse(b'{"task_id":"t-1"}')] + [
        FakeResponse(b'{"task_id":"t-1","state":"running","is_final":false,"progress":"45%"}')
    ] * 80

    with pytest.raises(ImageTaskStillRunning) as ctx:
        make_client(sleep=sleep).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
            task_callback=callbacks.append,
        )

    assert now[0] == 600.0
    assert ctx.value.task == callbacks[-1]
    assert ctx.value.task.progress == "45%"
    assert sum(call.args[0].method == "POST" for call in build_opener.return_value.open.call_args_list) == 1


@patch("openai_image_api.time.monotonic")
@patch("openai_image_api.urllib.request.build_opener")
def test_poll_deadline_bounds_slow_active_get_to_remaining_window(build_opener, monotonic, tmp_path):
    now = [0.0]
    monotonic.side_effect = lambda: now[0]
    requests = []
    get_timeouts = []

    class SlowResponse(FakeResponse):
        def __init__(self, timeout):
            super().__init__(b"")
            self.timeout = timeout

        def read(self, _amount=None):
            now[0] += self.timeout
            raise socket.timeout("slow status body")

        def read1(self, amount=None):
            return self.read(amount)

    def open_request(request, *, timeout):
        requests.append(request)
        if request.method == "POST":
            return FakeResponse(b'{"task_id":"t-1"}')
        get_timeouts.append(timeout)
        return SlowResponse(timeout)

    build_opener.return_value.open.side_effect = open_request

    with pytest.raises(ImageTaskStillRunning) as ctx:
        make_client(timeout=1000.0, max_attempts=4).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.task.task_id == "t-1"
    assert get_timeouts == [600.0]
    assert now[0] == 600.0
    assert [request.method for request in requests] == ["POST", "GET"]


@patch("openai_image_api.TASK_WAIT_LIMIT_SECONDS", 0.05)
@patch("openai_image_api.urllib.request.build_opener")
def test_poll_deadline_returns_while_get_is_still_dribbling(build_opener, tmp_path):
    release = threading.Event()
    worker_finished = threading.Event()

    class CancellableResponse(FakeResponse):
        def read1(self, _amount=None):
            release.wait(1.0)
            if release.is_set():
                raise socket.timeout("cancelled")
            return super().read1(_amount)

        def close(self):
            release.set()

    def open_request(request, *, timeout):
        if request.method == "POST":
            return FakeResponse(b'{"task_id":"t-1"}')
        try:
            return CancellableResponse(
                b'{"task_id":"t-1","state":"running","is_final":false}'
            )
        finally:
            worker_finished.set()

    build_opener.return_value.open.side_effect = open_request
    started = time.monotonic()
    try:
        with pytest.raises(ImageTaskStillRunning):
            make_client(timeout=1000.0, max_attempts=4).generate_edit(
                "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
            )
        elapsed = time.monotonic() - started
    finally:
        release.set()
        worker_finished.wait(1.0)

    assert elapsed < 0.5
    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET"]
    assert not any(thread.name == "gpt-image-status-get" for thread in threading.enumerate())


def test_python314_real_https_handler_propagates_stable_network_error(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise socket.timeout("injected connect timeout")

    monkeypatch.setattr(socket, "create_connection", fail_connect)
    client = make_client(timeout=0.2, max_attempts=1)
    with pytest.raises(OpenAIImageAPIError) as ctx:
        client._request_json_url(
            "https://api.lk888.ai/v1/media/status?task_id=t-1",
            deadline=time.monotonic() + 0.5,
        )
    assert ctx.value.code in {"timeout", "network"}


def test_pre_socket_deadline_is_bounded_and_second_call_starts_no_worker(monkeypatch, tmp_path):
    release = threading.Event()
    started = threading.Event()
    connect_calls = []

    def blocked_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        started.set()
        release.wait(1.0)
        raise socket.timeout("blocked before socket publication")

    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    client = make_client(timeout=10.0, max_attempts=1)
    resume = ImageTaskSnapshot("t-1", "running", False, 1787200000.0)
    source = make_png(tmp_path / "source.png")
    try:
        started_at = time.monotonic()
        with pytest.raises(ImageTaskStillRunning):
            with patch("openai_image_api.TASK_WAIT_LIMIT_SECONDS", 0.05):
                client.generate_edit("prompt", [source], tmp_path / "one.png", resume_task=resume)
        assert time.monotonic() - started_at < 0.3
        assert started.wait(0.2)

        with pytest.raises(ImageTaskStillRunning):
            with patch("openai_image_api.TASK_WAIT_LIMIT_SECONDS", 0.05):
                client.generate_edit("prompt", [source], tmp_path / "two.png", resume_task=resume)
        assert len(connect_calls) == 1
        assert sum(thread.name == "gpt-image-status-get" for thread in threading.enumerate()) <= 1
    finally:
        release.set()
        cleanup_deadline = time.monotonic() + 0.5
        while client._active_status_worker is not None and time.monotonic() < cleanup_deadline:
            time.sleep(0.01)
        assert client._active_status_worker is None


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_atomic_save_rejects_header_valid_truncated_image_and_preserves_target(tmp_path, fmt):
    source = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(source, format=fmt, quality=90)
    raw = source.getvalue()
    truncated = None
    for trim in range(1, min(256, len(raw) - 1)):
        candidate = raw[:-trim]
        try:
            with Image.open(io.BytesIO(candidate)) as image:
                image.verify()
            with Image.open(io.BytesIO(candidate)) as image:
                image.load()
        except OSError:
            try:
                with Image.open(io.BytesIO(candidate)) as image:
                    image.verify()
            except OSError:
                continue
            truncated = candidate
            break
    if truncated is None:
        truncated = raw[: max(20, len(raw) // 2)]
    target = tmp_path / "existing.img"
    target.write_bytes(b"old-target")
    with pytest.raises(OpenAIImageAPIError) as ctx:
        atomic_save_validated_image(truncated, target)
    assert ctx.value.code == "invalid_image"
    assert target.read_bytes() == b"old-target"


@patch("openai_image_api.time.monotonic")
@patch("openai_image_api.urllib.request.build_opener")
def test_poll_deadline_caps_retry_backoff_and_prevents_next_get(build_opener, monotonic, tmp_path):
    now = [0.0]
    monotonic.side_effect = lambda: now[0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def open_request(request, *, timeout):
        if request.method == "POST":
            return FakeResponse(b'{"task_id":"t-1"}')
        now[0] = 599.0
        raise HTTPError(request.full_url, 429, "busy", {}, None)

    build_opener.return_value.open.side_effect = open_request

    with pytest.raises(ImageTaskStillRunning):
        make_client(
            timeout=1000.0, max_attempts=4, retry_delays=(10.0,), sleep=sleep
        ).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET"]
    assert sleeps == [1.0]
    assert now[0] == 600.0


@pytest.mark.parametrize("failure", [socket.timeout("slow"), HTTPError("url", 429, "busy", {}, None), HTTPError("url", 503, "down", {}, None)])
@patch("openai_image_api.urllib.request.build_opener")
def test_poll_transient_errors_retry_get_only(build_opener, failure, tmp_path):
    sleeps = []
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'), failure,
        FakeResponse(b'{"task_id":"t-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}'),
    ]

    make_client(sleep=sleeps.append, max_attempts=2, retry_delays=(0.25,)).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    requests = [call.args[0] for call in build_opener.return_value.open.call_args_list]
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert requests[1].full_url == requests[2].full_url
    assert sleeps == [0.25]


@patch("openai_image_api.urllib.request.build_opener")
def test_poll_permanent_error_is_not_retried_or_resubmitted(build_opener, tmp_path):
    build_opener.return_value.open.side_effect = [
        FakeResponse(b'{"task_id":"t-1"}'),
        HTTPError("url", 401, "denied", {}, None),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(max_attempts=4).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "authentication"
    assert [call.args[0].method for call in build_opener.return_value.open.call_args_list] == ["POST", "GET"]


@patch("openai_image_api.urllib.request.build_opener")
def test_resume_running_skips_create_and_uses_fresh_local_deadline(build_opener, tmp_path):
    build_opener.return_value.open.return_value = FakeResponse(b'{"task_id":"t-1","state":"success","is_final":true,"result_url":"https://cdn.example/result.png","result_type":"image"}')
    resume = ImageTaskSnapshot("t-1", "running", False, 1787200000.0)

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png", resume_task=resume
    )

    request = build_opener.return_value.open.call_args_list[0].args[0]
    assert request.method == "GET"
    assert result.task.state == "success"


@patch("openai_image_api.urllib.request.build_opener")
def test_resume_success_redownloads_missing_or_corrupt_output_without_api_requests(build_opener, tmp_path):
    target = tmp_path / "out.png"
    target.write_bytes(b"corrupt")
    resume = ImageTaskSnapshot(
        "t-1", "success", True, 1787200000.0,
        result_url="https://cdn.example/result.png", result_type="image",
    )
    callbacks = []

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], target,
        resume_task=resume, task_callback=callbacks.append,
    )

    build_opener.assert_not_called()
    assert callbacks == [resume]
    assert result.task == resume
    with Image.open(target) as image:
        image.verify()


@pytest.mark.parametrize(
    "saved_url",
    [
        "https://cdn.example/result.png?token=[redacted]",
        "https://cdn.example/result.png?token=test-key",
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_resume_success_rejects_redacted_or_secret_result_url_without_network(
    build_opener, monkeypatch, saved_url, tmp_path
):
    import openai_image_api as module

    callbacks = []
    monkeypatch.setattr(
        module,
        "validate_remote_image_url",
        lambda _url: (_ for _ in ()).throw(AssertionError("must fail before URL validation")),
    )
    monkeypatch.setattr(
        module,
        "_download_resolved_image",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    resume = ImageTaskSnapshot(
        "t-1", "success", True, 1787200000.0,
        result_url=saved_url, result_type="image",
    )

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png",
            resume_task=resume, task_callback=callbacks.append,
        )

    assert ctx.value.code == "unsafe_result_url"
    assert "test-key" not in str(ctx.value)
    assert "test-key" not in repr(callbacks)
    assert callbacks[-1].result_url.endswith("token=[redacted]")
    build_opener.assert_not_called()


@patch("openai_image_api.urllib.request.build_opener")
def test_resume_failed_is_terminal_without_network(build_opener, tmp_path):
    resume = ImageTaskSnapshot("t-1", "failed", True, 1787200000.0, error="provider rejected task")

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png", resume_task=resume
        )

    assert ctx.value.code == "task_failed"
    build_opener.assert_not_called()


def test_generated_image_requires_task_snapshot():
    with pytest.raises(TypeError):
        GeneratedImage("out.png", "gpt-image-2")


@patch("openai_image_api.urllib.request.build_opener")
def test_multiple_reference_images_remain_separate_data_urls(
    build_opener,
    tmp_path,
):
    build_opener.return_value.open.return_value = fake_png_response()

    make_client(base_url="https://gateway.test/v1").generate_edit(
        "prompt",
        [make_png(tmp_path / "first.png"), make_png(tmp_path / "second.png")],
        tmp_path / "out.png",
    )

    request = build_opener.return_value.open.call_args_list[0].args[0]
    assert len(json.loads(request.data)["params"]["images"]) == 2


@pytest.mark.parametrize(
    ("resolution", "image_size", "expected_size"),
    [
        ("1K", "", b"1024x1024"),
        ("1K", "1:1", b"1024x1024"),
        ("2K", "4:5", b"2048x2560"),
        ("4K", "16:9", b"3840x2160"),
        ("1K", "11:15", b"960x1280"),
    ],
)
@patch("openai_image_api.urllib.request.build_opener")
def test_media_create_uses_documented_pixel_sizes(
    build_opener,
    resolution,
    image_size,
    expected_size,
    tmp_path,
):
    build_opener.return_value.open.return_value = fake_png_response()

    make_client(
        base_url="https://api.lk888.ai/v1",
        resolution=resolution,
    ).generate_edit(
        "prompt",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        image_size=image_size,
    )

    request = build_opener.return_value.open.call_args_list[0].args[0]
    assert request.full_url == "https://api.lk888.ai/v1/media/generate"
    assert expected_size in request.data
    assert request.get_header("Content-type") == "application/json"


@patch("openai_image_api.urllib.request.build_opener")
def test_merge_switch_encodes_one_contact_sheet_data_url(build_opener, tmp_path):
    build_opener.return_value.open.return_value = fake_png_response()

    make_client(
        base_url="https://gateway.test/v1",
        merge_reference_images=True,
    ).generate_edit(
        "prompt",
        [make_png(tmp_path / "first.png"), make_png(tmp_path / "second.png")],
        tmp_path / "out.png",
    )

    request = build_opener.return_value.open.call_args_list[0].args[0]
    images = json.loads(request.data)["params"]["images"]
    assert len(images) == 1
    assert images[0].startswith("data:image/png;base64,")


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

    assert ctx.value.code == "submission_unknown"
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
    request = build_opener.return_value.open.call_args_list[0].args[0]
    assert request.get_header("Authorization") == "Bearer test-key"
    assert build_opener.return_value.open.call_args.kwargs == {"timeout": 12.5}


@patch("openai_image_api.urllib.request.build_opener")
def test_generate_edit_does_not_retry_paid_post_for_429_or_401(build_opener, tmp_path):
    """Fails if an HTTP error can automatically resubmit a paid synchronous edit."""
    source = make_png(tmp_path / "source.png")
    for status, expected_code in ((429, "submission_unknown"), (401, "submission_unknown")):
        build_opener.return_value.open.reset_mock()
        build_opener.return_value.open.side_effect = HTTPError(
            "https://gateway.test/v1/images/edits",
            status,
            "injected error",
            {},
            None,
        )
        with pytest.raises(OpenAIImageAPIError) as ctx:
            make_client(base_url="https://gateway.test/v1").generate_edit(
                "prompt", [source], tmp_path / f"out-{status}.png"
            )

        assert ctx.value.code == expected_code
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


@patch("openai_image_api._uses_active_tun_fake_ip_route", return_value=True)
@patch("openai_image_api.socket.getaddrinfo")
def test_result_url_accepts_https_hostname_routed_through_active_tun_fake_ip(
    getaddrinfo, active_route
):
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.1.139", 443)),
    ]

    resolved = validate_remote_image_url(
        "https://cos.lingkeai.vip/generated.png?signature=opaque"
    )

    assert [address.ip_literal for address in resolved.addresses] == ["198.18.1.139"]
    assert resolved.hostname == "cos.lingkeai.vip"
    assert resolved.scheme == "https"
    active_route.assert_called_once()


@pytest.mark.parametrize(
    "remote_url",
    [
        "http://cos.lingkeai.vip/generated.png",
        "https://198.18.1.139/generated.png",
        "https://3323068811/generated.png",
        "https://0xc612018b/generated.png",
        "https://030604400613/generated.png",
        "https://198.18.395/generated.png",
    ],
)
@patch("openai_image_api._uses_active_tun_fake_ip_route", return_value=True)
@patch("openai_image_api.socket.getaddrinfo")
def test_result_url_rejects_tun_fake_ip_without_https_hostname(
    getaddrinfo, active_route, remote_url
):
    port = 443 if remote_url.startswith("https:") else 80
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.1.139", port)),
    ]

    with pytest.raises(OpenAIImageAPIError) as ctx:
        validate_remote_image_url(remote_url)

    assert ctx.value.code == "unsafe_result_url"


@patch("openai_image_api.socket.socket", side_effect=OSError("socket unavailable"))
def test_tun_fake_ip_probe_fails_closed_when_route_socket_cannot_open(
    socket_constructor,
):
    assert _uses_active_tun_fake_ip_route(
        ipaddress.IPv4Address("198.18.1.139")
    ) is False
    socket_constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)


@pytest.mark.parametrize(
    ("source_ip", "expected"),
    [
        ("198.18.0.1", True),
        ("192.168.1.10", False),
    ],
)
@patch("openai_image_api.socket.socket")
def test_tun_fake_ip_probe_requires_matching_local_route_and_closes_socket(
    socket_constructor, source_ip, expected
):
    probe = socket_constructor.return_value
    probe.getsockname.return_value = (source_ip, 50123)

    assert _uses_active_tun_fake_ip_route(
        ipaddress.IPv4Address("198.18.1.139")
    ) is expected

    probe.connect.assert_called_once_with(("198.18.1.139", 443))
    probe.getsockname.assert_called_once_with()
    probe.close.assert_called_once_with()


@pytest.mark.parametrize("failing_method", ["connect", "getsockname"])
@patch("openai_image_api.socket.socket")
def test_tun_fake_ip_probe_closes_socket_when_route_inspection_fails(
    socket_constructor, failing_method
):
    probe = socket_constructor.return_value
    getattr(probe, failing_method).side_effect = OSError("route unavailable")

    assert _uses_active_tun_fake_ip_route(
        ipaddress.IPv4Address("198.18.1.139")
    ) is False

    probe.close.assert_called_once_with()


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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test:8443/generated.png?sig=opaque"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert getaddrinfo.call_count == 1
    assert build_opener.return_value.open.call_count == 2
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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert build_opener.return_value.open.call_count == 2
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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

    make_client(
        sleep=sleep_calls.append,
        max_attempts=3,
        retry_delays=(0.25, 0.5),
    ).generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert build_opener.return_value.open.call_count == 2
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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    assert build_opener.return_value.open.call_count == 2
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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client(
            sleep=sleep_calls.append,
            max_attempts=3,
            retry_delays=(0.25, 0.5),
        ).generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
        )

    assert ctx.value.code == "invalid_response"
    assert build_opener.return_value.open.call_count == 2
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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "http://b\u00fccher.example:8080/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(remote_url)

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://images.example.test/generated.png"
    )

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
    build_opener.return_value.open.return_value = fake_task_response(
        "https://cdn.example/corrupt.png"
    )

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
    request = build_opener.return_value.open.call_args_list[0].args[0]
    assert request.full_url == "https://api.lk888.ai/v1/media/generate"
    assert len(json.loads(request.data)["params"]["images"]) == 1
