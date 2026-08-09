"""Validated configuration primitives for OpenAI Images-compatible APIs."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit


DEFAULT_OPENAI_IMAGE_BASE_URL: Final = "https://hapiopen.cc/v1"
_VALID_RESOLUTIONS: Final = {"1K", "2K", "4K"}


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


def _raise_invalid_base_url() -> None:
    raise OpenAIImageAPIError("invalid_base_url", "GPT Image API 地址无效。")
