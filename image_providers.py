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
    product_name_cn: str = ""
    language: str = ""
    selling_points: str = ""
    confirmation_advisor: Any | None = None


@dataclass(frozen=True)
class DetailSetRequest:
    product_id: str
    product_dir: Path
    screens: tuple[DetailScreen, ...]
    image_paths: tuple[str, ...]
    image_size: str
    target_count: int
    prompt: str = ""
    project_id: str = ""
    product_name_cn: str = ""
    language: str = ""
    selling_points: str = ""
    confirmation_advisor: Any | None = None
    progress_callback: Callable[[int, int, int, tuple[int, ...]], None] | None = None


@dataclass(frozen=True)
class ImageProviderResult:
    succeeded: bool
    local_paths: tuple[str, ...] = ()
    used_model: str = ""
    completed_count: int = 0
    artifact_count: int = 0
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
            and is_valid_image_file(checkpoint.get("local_path"))
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
                if request.progress_callback:
                    request.progress_callback(
                        screen.index,
                        request.target_count,
                        len(completed),
                        tuple(failed),
                    )
                continue
            local_path = str(generated.local_path)
            record_detail_checkpoint(request.product_dir, screen.index, "done", local_path, attempts=attempts)
            completed.add(screen.index)
            used_model = str(generated.model) or used_model
            if request.progress_callback:
                request.progress_callback(
                    screen.index,
                    request.target_count,
                    len(completed),
                    tuple(failed),
                )

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

    def __init__(self, bot: Any, logger: Any | None = None) -> None:
        self.bot = bot
        self.logger = logger
        self.confirmation_advisor: Any | None = None
        self._project_ids: dict[str, str] = {}

    def validate_completed_support(
        self,
        product: Any,
        product_dir: str | Path,
        existing_status: Mapping[str, object],
    ) -> bool:
        """Validate the Lovart project before a completed product is skipped."""
        previous_project_id = _existing_project_id(existing_status)
        if not previous_project_id or self._can_reuse_project(previous_project_id):
            return True
        self._invalidate_support_resume(product, product_dir, previous_project_id)
        return False

    def prepare_support_images(
        self,
        product: Any,
        product_dir: str | Path,
        existing_status: Mapping[str, object],
    ) -> bool:
        """Resolve a reusable Lovart project only after Lovart support is selected."""
        previous_project_id = _existing_project_id(existing_status)
        restart = bool(existing_status.get("lovart_support_resume_invalidated"))
        if previous_project_id and self._can_reuse_project(previous_project_id):
            project_id = previous_project_id
            update_status(
                product_dir,
                "lovart_project_reused",
                project_id=project_id,
                project_url=_lovart_project_url(project_id),
            )
        else:
            if previous_project_id:
                restart = True
                self._invalidate_support_resume(product, product_dir, previous_project_id)
            project_id = self.bot.create_project(product.id, product.name_cn)
            update_status(
                product_dir,
                "lovart_project_created",
                project_id=project_id,
                project_url=_lovart_project_url(project_id),
            )
        self._project_ids[str(product.id)] = str(project_id)
        return restart

    def _invalidate_support_resume(
        self,
        product: Any,
        product_dir: str | Path,
        previous_project_id: str,
    ) -> None:
        if self.logger:
            self.logger.warning(
                f"Lovart project {previous_project_id} for '{product.id}' is invalid; "
                "restarting product"
            )
        update_status(
            product_dir,
            "lovart_project_invalid",
            previous_project_id=previous_project_id,
            previous_project_url=_lovart_project_url(previous_project_id),
            white_bg_local_path="",
            scene_local_path="",
            white_bg_provider="",
            scene_provider="",
            lovart_white_bg_local_path="",
            lovart_scene_local_path="",
            lovart_final_images=[],
            lovart_support_resume_invalidated=True,
            lovart_done=False,
            lovart_support_images_ready=False,
            lovart_final_images_ready=False,
            support_images_ready=False,
            artifact_count=0,
        )

    def complete_support_images(self, product_dir: str | Path) -> None:
        update_status(
            product_dir,
            "lovart_support_images_ready",
            lovart_support_resume_invalidated=False,
        )

    def _can_reuse_project(self, project_id: str) -> bool:
        if not hasattr(self.bot, "validate_project"):
            return True
        try:
            return bool(self.bot.validate_project(project_id))
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Lovart project validation failed for {project_id}: {exc}")
            return False

    def generate_support_image(self, request: SupportImageRequest) -> ImageProviderResult:
        kwargs = dict(
            product_id=request.product_id,
            step_name=request.step_name,
            prompt=request.prompt,
            image_paths=list(request.image_paths),
        )
        project_id = self._project_ids.get(request.product_id, "")
        if project_id:
            kwargs.update(
                project_id=project_id,
                confirmation_advisor=request.confirmation_advisor,
                product_name_cn=request.product_name_cn,
                language=request.language,
                selling_points=request.selling_points,
            )
        raw = self.bot.create_support_image(**kwargs)
        result = _lovart_result(raw, request.product_dir, 1)
        if result.succeeded and result.local_paths:
            local_path = result.local_paths[0]
            update_status(
                request.product_dir,
                f"lovart_{request.step_name}_done",
                local_path=local_path,
                **{f"lovart_{request.step_name}_local_path": local_path},
            )
        return result

    def generate_detail_set(self, request: DetailSetRequest) -> ImageProviderResult:
        prompt = request.prompt or "\n\n".join(
            screen.prompt for screen in sorted(request.screens, key=lambda item: item.index)
        )
        project_id = request.project_id or self._project_ids.get(request.product_id, "")
        raw = self.bot.create_and_generate(
            product_id=request.product_id,
            prompt=prompt,
            image_paths=list(request.image_paths),
            project_id=project_id,
            confirmation_advisor=request.confirmation_advisor or self.confirmation_advisor,
            product_name_cn=request.product_name_cn,
            language=request.language,
            selling_points=request.selling_points,
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
        if isinstance(checkpoint, Mapping) and is_valid_image_file(checkpoint.get("local_path")):
            paths.append(str(checkpoint["local_path"]))
    return tuple(paths)


def is_valid_image_file(value: object) -> bool:
    """Return whether a path is a fully decodable, non-empty image file."""
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
    generation_succeeded = raw.get("generation_succeeded")
    local_paths = _lovart_local_paths(raw)
    try:
        artifact_count = max(0, int(raw.get("artifact_count", len(local_paths))))
    except (TypeError, ValueError):
        artifact_count = len(local_paths)
    succeeded = bool(
        generation_succeeded is True
        or generation_succeeded is None and local_paths
    )
    error = str(raw.get("warning") or raw.get("error") or "")
    return ImageProviderResult(
        succeeded=succeeded,
        local_paths=local_paths,
        used_model=str(raw.get("used_model") or ""),
        completed_count=completed_count if succeeded else 0,
        artifact_count=artifact_count,
        error=error,
        raw_result=raw_result if isinstance(raw_result, Mapping) else None,
    )


def _lovart_local_paths(
    raw: Mapping[str, object],
) -> tuple[str, ...]:
    paths: list[str] = []
    for key in ("local_path", "local_paths"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, (list, tuple)):
            paths.extend(str(item) for item in value if isinstance(item, (str, Path)) and str(item))
    downloaded = raw.get("downloaded", [])
    if isinstance(downloaded, (list, tuple)):
        paths.extend(
            str(item["local_path"])
            for item in downloaded
            if isinstance(item, Mapping)
            and item.get("type") in {"image", "unknown", None}
            and isinstance(item.get("local_path"), (str, Path))
            and str(item["local_path"])
        )
    return tuple(dict.fromkeys(paths))


def _lovart_project_url(project_id: str = "") -> str:
    return f"https://www.lovart.ai/canvas?projectId={project_id}" if project_id else ""


def _existing_project_id(status: Mapping[str, object]) -> str:
    project_id = str(status.get("project_id") or "").strip()
    if project_id:
        return project_id
    project_url = str(status.get("project_url") or "")
    marker = "projectId="
    if marker not in project_url:
        return ""
    return project_url.split(marker, 1)[1].split("&", 1)[0].strip()
