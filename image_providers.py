"""Provider-neutral image generation adapters and durable detail checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from image_generation import (
    PROVIDER_LOVART,
    DetailScreen,
    normalize_image_provider,
)
from utils import read_status, update_status


_CHECKPOINT_FIELD = "detail_checkpoints"


@dataclass(frozen=True)
class SupportImageRequest:
    product_id: str
    product_dir: Path
    step_name: str
    prompt: str
    image_paths: tuple[str, ...]
    image_size: str = ""


@dataclass(frozen=True)
class DetailSetRequest:
    product_id: str
    product_dir: Path
    screens: tuple[DetailScreen, ...]
    image_paths: tuple[str, ...]
    image_size: str
    target_count: int


@dataclass(frozen=True)
class ImageProviderResult:
    succeeded: bool
    local_paths: tuple[str, ...] = ()
    used_model: str = ""
    completed_count: int = 0
    failed_indexes: tuple[int, ...] = ()
    partial_complete: bool = False
    error: str = ""
    raw_result: Mapping[str, object] | None = None


class LazyImageProviderRegistry:
    def __init__(self, lovart_factory: Callable[[], Any], openai_factory: Callable[[], Any]) -> None:
        self._lovart_factory = lovart_factory
        self._openai_factory = openai_factory
        self._instances: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        normalized = normalize_image_provider(name)
        if normalized not in self._instances:
            factory = self._lovart_factory if normalized == PROVIDER_LOVART else self._openai_factory
            self._instances[normalized] = factory()
        return self._instances[normalized]


def record_detail_checkpoint(
    product_dir: str | Path,
    index: int,
    state: str,
    local_path: str = "",
    error: str = "",
    attempts: int = 0,
) -> None:
    """Atomically persist the state of one GPT detail screen."""
    status = read_status(product_dir)
    checkpoints = status.get(_CHECKPOINT_FIELD, {})
    saved = dict(checkpoints) if isinstance(checkpoints, Mapping) else {}
    saved[str(index)] = {
        "state": str(state),
        "local_path": str(local_path),
        "error": str(error),
        "attempts": max(0, int(attempts)),
    }
    update_status(product_dir, "detail_checkpoint_updated", **{_CHECKPOINT_FIELD: saved})


def read_completed_detail_indexes(product_dir: str | Path, expected_count: int) -> set[int]:
    """Return only checkpointed screens whose saved image remains usable."""
    status = read_status(product_dir)
    checkpoints = status.get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return set()
    completed: set[int] = set()
    for index in range(1, max(0, int(expected_count)) + 1):
        checkpoint = _checkpoint_for_index(checkpoints, index)
        if (
            isinstance(checkpoint, Mapping)
            and checkpoint.get("state") == "done"
            and _is_valid_image(checkpoint.get("local_path"))
        ):
            completed.add(index)
    return completed


class OpenAIImageProvider:
    def __init__(self, api: Any, logger: Any | None = None) -> None:
        self.api = api
        self.logger = logger

    def generate_support_image(self, request: SupportImageRequest) -> ImageProviderResult:
        output_path = request.product_dir / "gpt_image" / "support" / f"{request.step_name}.png"
        try:
            generated = self.api.generate_edit(
                prompt=request.prompt,
                image_paths=list(request.image_paths),
                output_path=output_path,
                image_size=request.image_size,
            )
        except Exception as exc:
            return ImageProviderResult(succeeded=False, error=str(exc))
        return ImageProviderResult(
            succeeded=True,
            local_paths=(str(generated.local_path),),
            used_model=str(generated.model),
            completed_count=1,
        )

    def generate_detail_set(self, request: DetailSetRequest) -> ImageProviderResult:
        completed = read_completed_detail_indexes(request.product_dir, request.target_count)
        failed: list[int] = []
        errors: list[str] = []
        used_model = ""
        for screen in sorted(request.screens, key=lambda item: item.index):
            if screen.index in completed:
                continue
            attempts = _checkpoint_attempts(request.product_dir, screen.index) + 1
            record_detail_checkpoint(request.product_dir, screen.index, "running", attempts=attempts)
            output_path = request.product_dir / "gpt_image" / "detail" / f"{screen.index:02d}.png"
            try:
                generated = self.api.generate_edit(
                    prompt=screen.prompt,
                    image_paths=list(request.image_paths),
                    output_path=output_path,
                    image_size=request.image_size,
                )
            except Exception as exc:
                failed.append(screen.index)
                errors.append(f"screen {screen.index}: {exc}")
                record_detail_checkpoint(
                    request.product_dir, screen.index, "failed", error=str(exc), attempts=attempts
                )
                continue
            local_path = str(generated.local_path)
            record_detail_checkpoint(request.product_dir, screen.index, "done", local_path, attempts=attempts)
            completed.add(screen.index)
            used_model = str(generated.model) or used_model

        local_paths = _completed_paths(request.product_dir, request.target_count)
        completed_count = len(completed)
        return ImageProviderResult(
            succeeded=completed_count == request.target_count,
            local_paths=local_paths,
            used_model=used_model,
            completed_count=completed_count,
            failed_indexes=tuple(failed),
            partial_complete=0 < completed_count < request.target_count,
            error="; ".join(errors),
        )


class LovartImageProvider:
    """Thin compatibility adapter that leaves Lovart's confirmation flow untouched."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot

    def generate_support_image(self, request: SupportImageRequest) -> ImageProviderResult:
        raw = self.bot.create_support_image(
            product_id=request.product_id,
            step_name=request.step_name,
            prompt=request.prompt,
            image_paths=list(request.image_paths),
        )
        return _lovart_result(raw, request.product_dir, 1)

    def generate_detail_set(self, request: DetailSetRequest) -> ImageProviderResult:
        prompt = "\n\n".join(screen.prompt for screen in sorted(request.screens, key=lambda item: item.index))
        raw = self.bot.create_and_generate(
            product_id=request.product_id,
            prompt=prompt,
            image_paths=list(request.image_paths),
        )
        return _lovart_result(raw, request.product_dir, request.target_count)


def _checkpoint_for_index(checkpoints: Mapping[str, object], index: int) -> object:
    return checkpoints.get(str(index), checkpoints.get(f"{index:02d}"))


def _checkpoint_attempts(product_dir: str | Path, index: int) -> int:
    checkpoints = read_status(product_dir).get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return 0
    checkpoint = _checkpoint_for_index(checkpoints, index)
    if not isinstance(checkpoint, Mapping):
        return 0
    try:
        return max(0, int(checkpoint.get("attempts", 0)))
    except (TypeError, ValueError):
        return 0


def _completed_paths(product_dir: str | Path, expected_count: int) -> tuple[str, ...]:
    checkpoints = read_status(product_dir).get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return ()
    paths: list[str] = []
    for index in range(1, max(0, int(expected_count)) + 1):
        checkpoint = _checkpoint_for_index(checkpoints, index)
        if isinstance(checkpoint, Mapping) and _is_valid_image(checkpoint.get("local_path")):
            paths.append(str(checkpoint["local_path"]))
    return tuple(paths)


def _is_valid_image(value: object) -> bool:
    try:
        with Image.open(Path(str(value))) as image:
            image.verify()
        with Image.open(Path(str(value))) as image:
            image.load()
            return bool(image.format and image.width > 0 and image.height > 0)
    except (FileNotFoundError, IsADirectoryError, OSError, SyntaxError, UnidentifiedImageError, ValueError):
        return False


def _lovart_result(raw_result: object, product_dir: Path, completed_count: int) -> ImageProviderResult:
    raw = raw_result if isinstance(raw_result, Mapping) else {}
    succeeded = bool(raw.get("generation_succeeded"))
    local_paths = _lovart_local_paths(raw, product_dir)
    error = str(raw.get("warning") or raw.get("error") or "")
    return ImageProviderResult(
        succeeded=succeeded,
        local_paths=local_paths,
        used_model=str(raw.get("used_model") or ""),
        completed_count=completed_count if succeeded else 0,
        error=error,
        raw_result=raw_result if isinstance(raw_result, Mapping) else None,
    )


def _lovart_local_paths(raw: Mapping[str, object], product_dir: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for key in ("local_path", "local_paths"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, (list, tuple)):
            paths.extend(str(item) for item in value if isinstance(item, (str, Path)) and str(item))
    if not paths:
        legacy_paths = read_status(product_dir).get("lovart_final_images", [])
        if isinstance(legacy_paths, (list, tuple)):
            paths.extend(str(item) for item in legacy_paths if isinstance(item, (str, Path)) and str(item))
    return tuple(paths)
