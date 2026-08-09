import base64
import io
import json
import socket
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
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


def make_png(path: Path) -> Path:
    path.write_bytes(base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64))
    return path


def fake_png_response() -> FakeResponse:
    return FakeResponse(json.dumps({
        "created": 1,
        "data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}],
    }).encode("utf-8"))


def make_client(**overrides) -> OpenAIImageAPI:
    config = OpenAIImageAPIConfig(
        api_key="test-key",
        base_url="https://hapiopen.cc/v1",
        timeout=12.5,
        retry_delays=(0.0,),
        **overrides,
    )
    return OpenAIImageAPI(config, sleep=lambda _delay: None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://hapiopen.cc", "https://hapiopen.cc/v1"),
        ("https://hapiopen.cc/v1/", "https://hapiopen.cc/v1"),
        ("https://gateway.test/root/v1", "https://gateway.test/root/v1"),
    ],
)
def test_normalize_openai_image_base_url(raw, expected):
    """Fails if a valid provider endpoint is not normalized to one terminal /v1."""
    assert normalize_openai_image_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://bad\nhost",
        "https://user:pass@gateway.test",
        "https:///v1",
        "https://gateway.test/v1?debug=true",
        "https://gateway.test/v1#anchor",
        "https://gateway.test/v1/v1",
        "https://gateway.test/v1/root",
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

    config = OpenAIImageAPIConfig.from_config({}, api_key=secret)
    assert secret not in repr(config)


def test_config_serialization_omits_key_but_authorization_can_access_it():
    """Fails if standard dataclass serialization exposes the resolved API key."""
    secret = "serialization-secret"
    config = OpenAIImageAPIConfig.from_config({}, api_key=secret)

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

    assert config.base_url == "https://gateway.test/root/v1"
    assert config.model == "custom-image-model"
    assert config.resolution == "2K"
    assert config.api_key == "resolved-key"
    with pytest.raises(AttributeError):
        config.model = "another-model"


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


@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_posts_multipart_and_saves_b64_png(urlopen, tmp_path):
    """Fails if a standard Images edit request is not multipart or its image is lost."""
    urlopen.return_value = fake_png_response()

    result = make_client().generate_edit(
        "keep the product exact",
        [make_png(tmp_path / "source.png")],
        tmp_path / "out.png",
        image_size="4:5",
    )

    request = urlopen.call_args.args[0]
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
    assert urlopen.call_args.kwargs == {"timeout": 12.5}


@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_retries_429_but_not_401(urlopen, tmp_path):
    """Fails if permanent authentication failures can create extra paid image jobs."""
    source = make_png(tmp_path / "source.png")
    urlopen.side_effect = [
        HTTPError("https://hapiopen.cc/v1/images/edits", 429, "busy", {}, None),
        fake_png_response(),
    ]

    make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert urlopen.call_count == 2

    urlopen.reset_mock()
    urlopen.side_effect = HTTPError(
        "https://hapiopen.cc/v1/images/edits", 401, "unauthorized", {}, None
    )
    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out-401.png")

    assert ctx.value.code == "authentication"
    assert urlopen.call_count == 1


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


@patch("openai_image_api.socket.getaddrinfo")
@patch("openai_image_api.urllib.request.build_opener")
@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_downloads_public_result_url_without_forwarding_api_key(
    urlopen, build_opener, getaddrinfo, tmp_path
):
    """Fails if a safe remote result cannot be saved or inherits API credentials."""
    getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    urlopen.side_effect = [
        FakeResponse(json.dumps({"created": 1, "data": [
            {"url": "https://images.example.test/generated.png"}
        ]}).encode("utf-8")),
    ]
    build_opener.return_value.open.return_value = FakeResponse(
        base64.b64decode(VALID_ONE_PIXEL_PNG_BASE64)
    )

    result = make_client().generate_edit(
        "prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png"
    )

    assert result.local_path == str(tmp_path / "out.png")
    image_request = build_opener.return_value.open.call_args.args[0]
    assert image_request.full_url == "https://images.example.test/generated.png"
    assert image_request.get_header("Authorization") is None
    assert build_opener.return_value.open.call_args.kwargs == {"timeout": 12.5}
    with Image.open(tmp_path / "out.png") as output:
        output.verify()


@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_rejects_invalid_input_before_network_call(urlopen, tmp_path):
    """Fails if arbitrary files are uploaded as images to the paid endpoint."""
    source = tmp_path / "not-an-image.png"
    source.write_bytes(b"not an image")

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    urlopen.assert_not_called()


@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_rejects_checksum_corrupt_input_before_network_call(urlopen, tmp_path):
    """Fails if a structurally PNG-like but corrupt source reaches the paid endpoint."""
    source = tmp_path / "corrupt.png"
    source.write_bytes(base64.b64decode(CORRUPT_PNG_BASE64))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit("prompt", [source], tmp_path / "out.png")

    assert ctx.value.code == "invalid_input_image"
    urlopen.assert_not_called()


@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_does_not_replace_existing_output_with_invalid_image(urlopen, tmp_path):
    """Fails if an invalid gateway response overwrites a previously valid output."""
    destination = tmp_path / "out.png"
    original = b"existing-output-must-survive"
    destination.write_bytes(original)
    urlopen.return_value = FakeResponse(json.dumps({
        "created": 1,
        "data": [{"b64_json": base64.b64encode(b"not an image").decode("ascii")}],
    }).encode("utf-8"))

    with pytest.raises(OpenAIImageAPIError) as ctx:
        make_client().generate_edit(
            "prompt", [make_png(tmp_path / "source.png")], destination
        )

    assert ctx.value.code == "invalid_image"
    assert destination.read_bytes() == original


@patch("openai_image_api.urllib.request.urlopen")
def test_test_edit_uses_a_real_png_fixture_and_returns_saved_image(urlopen, tmp_path):
    """Fails if the explicit compatibility probe does not use the normal edit pipeline."""
    urlopen.return_value = fake_png_response()

    result = make_client().test_edit(tmp_path)

    assert Path(result.local_path).parent == tmp_path
    assert Path(result.local_path).is_file()
    request = urlopen.call_args.args[0]
    assert b'name="image[]"; filename="' in request.data
