from dataclasses import asdict

import pytest

from openai_image_api import (
    OpenAIImageAPIConfig,
    OpenAIImageAPIError,
    normalize_openai_image_base_url,
)


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
