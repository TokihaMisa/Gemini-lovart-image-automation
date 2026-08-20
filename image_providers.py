"""Provider-neutral image generation adapters and durable detail checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from image_generation import (
    PROVIDER_LOVART,
    DetailScreen,
    normalize_image_provider,
)
from openai_image_api import (
    GeneratedImage,
    ImageTaskSnapshot,
    ImageTaskStillRunning,
    _provider_image_size,
    append_aspect_instruction,
    normalize_openai_image_base_url,
)
from utils import read_status, update_status


_CHECKPOINT_FIELD = "detail_checkpoints"
_SUPPORT_TASK_CHECKPOINT_FIELD = "support_task_checkpoints"
_SUPPORT_TASK_CHECKPOINT_KEYS = frozenset(
    {
        "state",
        "task_id",
        "task_created_at",
        "is_final",
        "progress",
        "status",
        "status_group",
        "result_url",
        "result_type",
        "error",
        "cost",
        "input_fingerprint",
        "prompt_hash",
        "model",
        "size",
        "base_url",
        "merge_reference_images",
        "local_path",
        "attempts",
    }
)
_REQUEST_SETTING_CHECKPOINT_KEYS = frozenset(
    {"model", "size", "base_url", "merge_reference_images"}
)


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
    input_fingerprint: str = ""
    resume: bool = True
    confirmation_advisor: Any | None = None
    status_callback: Callable[[str], None] | None = None


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
    input_fingerprint: str = ""
    resume: bool = True
    confirmation_advisor: Any | None = None
    progress_callback: Callable[[int, int, int, tuple[int, ...]], None] | None = None
    status_callback: Callable[[str], None] | None = None


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
    still_running: bool = False
    task_id_suffix: str = ""


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
    input_fingerprint: str = "",
    prompt_hash: str = "",
    task_id: str = "",
    task_created_at: float = 0.0,
    is_final: bool = False,
    progress: str = "",
    status: str = "",
    status_group: str = "",
    result_url: str = "",
    result_type: str = "",
    cost: object = None,
    request_settings: Mapping[str, object] | None = None,
) -> None:
    """Atomically persist the state of one GPT detail screen."""
    product_status = read_status(product_dir)
    checkpoints = product_status.get(_CHECKPOINT_FIELD, {})
    saved = dict(checkpoints) if isinstance(checkpoints, Mapping) else {}
    checkpoint = {
        "state": str(state),
        "local_path": str(local_path),
        "error": str(error),
        "attempts": max(0, int(attempts)),
        "input_fingerprint": str(input_fingerprint or ""),
        "prompt_hash": str(prompt_hash or ""),
    }
    if task_id:
        checkpoint.update(
            task_id=str(task_id),
            task_created_at=float(task_created_at),
            is_final=bool(is_final),
            progress=str(progress),
            status=str(status),
            status_group=str(status_group),
            result_url=str(result_url),
            result_type=str(result_type),
            cost=_json_safe_setting(cost),
        )
    if request_settings:
        checkpoint.update(
            {
                str(key): _json_safe_setting(value)
                for key, value in request_settings.items()
                if str(key) in _REQUEST_SETTING_CHECKPOINT_KEYS
            }
        )
    saved[str(index)] = checkpoint
    update_status(product_dir, "detail_checkpoint_updated", **{_CHECKPOINT_FIELD: saved})


def record_support_task_checkpoint(
    product_dir: str | Path,
    step_name: str,
    checkpoint: Mapping[str, object],
) -> None:
    """Atomically replace one provider-owned support task checkpoint."""
    status = read_status(product_dir)
    checkpoints = status.get(_SUPPORT_TASK_CHECKPOINT_FIELD, {})
    saved = dict(checkpoints) if isinstance(checkpoints, Mapping) else {}
    saved[str(step_name)] = {
        str(key): _json_safe_setting(value)
        for key, value in checkpoint.items()
        if str(key) in _SUPPORT_TASK_CHECKPOINT_KEYS
    }
    update_status(
        product_dir,
        "support_task_checkpoint_updated",
        **{_SUPPORT_TASK_CHECKPOINT_FIELD: saved},
    )


def read_support_task_checkpoint(
    product_dir: str | Path,
    step_name: str,
) -> dict[str, object]:
    checkpoints = read_status(product_dir).get(_SUPPORT_TASK_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return {}
    checkpoint = checkpoints.get(str(step_name), {})
    return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}


def read_completed_detail_indexes(
    product_dir: str | Path,
    expected_count: int,
    input_fingerprint: str = "",
    screen_prompt_hashes: Mapping[int, str] | None = None,
) -> set[int]:
    """Return only checkpointed screens whose saved image remains usable."""
    status = read_status(product_dir)
    checkpoints = status.get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return set()
    completed: set[int] = set()
    repaired_checkpoints = dict(checkpoints)
    repaired = False
    for index in range(1, max(0, int(expected_count)) + 1):
        checkpoint = _checkpoint_for_index(checkpoints, index)
        resolved_path = _resolved_detail_checkpoint_path(product_dir, index, checkpoint)
        if (
            isinstance(checkpoint, Mapping)
            and checkpoint.get("state") == "done"
            and _checkpoint_matches_fingerprint(checkpoint, input_fingerprint)
            and _checkpoint_matches_prompt(
                checkpoint,
                _expected_prompt_hash(screen_prompt_hashes, index),
            )
            and resolved_path is not None
        ):
            completed.add(index)
            if str(checkpoint.get("local_path") or "") != str(resolved_path):
                updated_checkpoint = dict(checkpoint)
                updated_checkpoint["local_path"] = str(resolved_path)
                repaired_checkpoints[str(index)] = updated_checkpoint
                repaired = True
    if repaired:
        update_status(
            product_dir,
            "detail_checkpoint_paths_rebased",
            **{_CHECKPOINT_FIELD: repaired_checkpoints},
        )
    return completed


class OpenAIImageProvider:
    def __init__(self, api: Any, logger: Any | None = None) -> None:
        self.api = api
        self.logger = logger

    def detail_execution_settings(self) -> dict[str, object]:
        """Return only non-secret values that alter an Images edit request."""
        config = getattr(self.api, "config", None)
        return {
            "base_url": normalize_openai_image_base_url(
                getattr(config, "base_url", None)
            ),
            "model": str(getattr(config, "model", "gpt-image-2") or "gpt-image-2").strip(),
            "resolution": str(getattr(config, "resolution", "1K") or "1K").upper(),
            "merge_reference_images": bool(
                getattr(config, "merge_reference_images", False)
            ),
        }

    def generate_support_image(self, request: SupportImageRequest) -> ImageProviderResult:
        output_path = request.product_dir / "gpt_image" / "support" / f"{request.step_name}.png"
        identity = _support_request_identity(self.api, request)
        checkpoint = read_support_task_checkpoint(request.product_dir, request.step_name)
        identity_matches = _checkpoint_matches_identity(checkpoint, identity)
        saved_task = (
            _task_snapshot_from_checkpoint(checkpoint)
            if request.resume and identity_matches
            else None
        )
        if (
            request.resume
            and identity_matches
            and str(checkpoint.get("state") or "") == "done"
            and is_valid_image_file(checkpoint.get("local_path"))
        ):
            local_path = str(checkpoint["local_path"])
            return ImageProviderResult(
                succeeded=True,
                local_paths=(local_path,),
                used_model=str(identity["model"]),
                completed_count=1,
            )
        if saved_task is not None and saved_task.state == "failed":
            saved_task = None
        if saved_task is None:
            _remove_file(output_path)
            attempts = _checkpoint_attempt_value(checkpoint) + 1 if identity_matches else 1
            record_support_task_checkpoint(
                request.product_dir,
                request.step_name,
                {
                    **identity,
                    "state": "running",
                    "local_path": "",
                    "error": "",
                    "attempts": attempts,
                },
            )
        else:
            attempts = max(1, _checkpoint_attempt_value(checkpoint))
        callback_local_path = (
            str(checkpoint.get("local_path") or "") if saved_task is not None else ""
        )

        last_task = saved_task

        def persist(task: ImageTaskSnapshot) -> None:
            nonlocal last_task
            last_task = task
            record_support_task_checkpoint(
                request.product_dir,
                request.step_name,
                {
                    **identity,
                    "local_path": callback_local_path,
                    "attempts": attempts,
                    **_checkpoint_fields_for_persistence(self.api, task),
                },
            )

        try:
            generated = self.api.generate_edit(
                prompt=request.prompt,
                image_paths=list(request.image_paths),
                output_path=output_path,
                image_size=request.image_size,
                status_callback=request.status_callback,
                task_callback=persist,
                resume_task=saved_task,
            )
        except ImageTaskStillRunning as exc:
            persist(exc.task)
            return ImageProviderResult(
                succeeded=False,
                still_running=True,
                task_id_suffix=_task_id_suffix(exc.task.task_id),
            )
        except Exception as exc:
            exception_task = getattr(exc, "task", None)
            task = (
                exception_task
                if isinstance(exception_task, ImageTaskSnapshot)
                else last_task
            )
            safe_error = _sanitized_provider_error(self.api, exc)
            if (
                isinstance(task, ImageTaskSnapshot)
                and task.is_final
                and task.state == "failed"
            ):
                failed_checkpoint = {
                    **identity,
                    "state": "failed",
                    "local_path": callback_local_path,
                    "error": safe_error,
                    "attempts": attempts,
                }
                failed_checkpoint.update(
                    _checkpoint_fields_for_persistence(self.api, task)
                )
                failed_checkpoint["state"] = "failed"
                failed_checkpoint["error"] = (
                    _sanitized_task_text(self.api, task.error) or safe_error
                )
                record_support_task_checkpoint(
                    request.product_dir,
                    request.step_name,
                    failed_checkpoint,
                )
            return ImageProviderResult(
                succeeded=False,
                error=safe_error,
                task_id_suffix=(
                    _task_id_suffix(task.task_id)
                    if isinstance(task, ImageTaskSnapshot)
                    else ""
                ),
            )
        task = _generated_task(generated)
        local_path = str(generated.local_path)
        record_support_task_checkpoint(
            request.product_dir,
            request.step_name,
            {
                **identity,
                "local_path": local_path,
                "attempts": attempts,
                **_checkpoint_fields_for_persistence(self.api, task),
                "state": "done",
            },
        )
        return ImageProviderResult(
            succeeded=True,
            local_paths=(local_path,),
            used_model=str(generated.model),
            completed_count=1,
            task_id_suffix=_task_id_suffix(task.task_id),
        )

    def generate_detail_set(self, request: DetailSetRequest) -> ImageProviderResult:
        prompt_hashes = {
            screen.index: detail_screen_prompt_hash(
                screen,
                request.target_count,
                request.image_size,
            )
            for screen in request.screens
        }
        request_settings = _openai_request_settings(self.api, request.image_size)
        if request.resume:
            completed = read_completed_detail_indexes(
                request.product_dir,
                request.target_count,
                request.input_fingerprint,
                prompt_hashes,
            )
        else:
            update_status(
                request.product_dir,
                "detail_checkpoints_restarted",
                **{_CHECKPOINT_FIELD: {}},
            )
            completed = set()
        failed: list[int] = []
        errors: list[str] = []
        still_running = False
        last_task_suffix = ""
        config = getattr(self.api, "config", None)
        used_model = str(
            getattr(config, "model", "gpt-image-2") or "gpt-image-2"
        ).strip()
        for screen in sorted(request.screens, key=lambda item: item.index):
            if screen.index in completed:
                continue
            output_path = request.product_dir / "gpt_image" / "detail" / f"{screen.index:02d}.png"
            checkpoint = _read_detail_checkpoint(request.product_dir, screen.index)
            identity = {
                "input_fingerprint": request.input_fingerprint,
                "prompt_hash": prompt_hashes[screen.index],
                **request_settings,
            }
            identity_matches = _checkpoint_matches_identity(checkpoint, identity)
            saved_task = (
                _task_snapshot_from_checkpoint(checkpoint)
                if request.resume and identity_matches
                else None
            )
            if saved_task is not None and saved_task.state == "failed":
                saved_task = None
            if saved_task is None:
                _remove_file(output_path)
                attempts = _checkpoint_attempt_value(checkpoint) + 1 if identity_matches else 1
                record_detail_checkpoint(
                    request.product_dir,
                    screen.index,
                    "running",
                    attempts=attempts,
                    input_fingerprint=request.input_fingerprint,
                    prompt_hash=prompt_hashes[screen.index],
                    request_settings=request_settings,
                )
            else:
                attempts = max(1, _checkpoint_attempt_value(checkpoint))
            callback_local_path = (
                str(checkpoint.get("local_path") or "")
                if saved_task is not None
                else ""
            )

            last_task = saved_task

            def persist(task: ImageTaskSnapshot) -> None:
                nonlocal last_task
                last_task = task
                fields = _checkpoint_fields_for_persistence(self.api, task)
                record_detail_checkpoint(
                    request.product_dir,
                    screen.index,
                    str(fields.pop("state")),
                    local_path=callback_local_path,
                    error=str(fields.pop("error")),
                    attempts=attempts,
                    input_fingerprint=request.input_fingerprint,
                    prompt_hash=prompt_hashes[screen.index],
                    request_settings=request_settings,
                    **fields,
                )

            try:
                generated = self.api.generate_edit(
                    prompt=screen.prompt,
                    image_paths=list(request.image_paths),
                    output_path=output_path,
                    image_size=request.image_size,
                    status_callback=request.status_callback,
                    task_callback=persist,
                    resume_task=saved_task,
                )
            except ImageTaskStillRunning as exc:
                persist(exc.task)
                still_running = True
                last_task_suffix = _task_id_suffix(exc.task.task_id)
                break
            except Exception as exc:
                failed.append(screen.index)
                safe_error = _sanitized_provider_error(self.api, exc)
                errors.append(f"screen {screen.index}: {safe_error}")
                exception_task = getattr(exc, "task", None)
                task = (
                    exception_task
                    if isinstance(exception_task, ImageTaskSnapshot)
                    else last_task
                )
                if (
                    isinstance(task, ImageTaskSnapshot)
                    and task.is_final
                    and task.state == "failed"
                ):
                    task_fields = _checkpoint_fields_for_persistence(self.api, task)
                    task_fields.pop("state", None)
                    task_fields.pop("error", None)
                    last_task_suffix = _task_id_suffix(task.task_id)
                    record_detail_checkpoint(
                        request.product_dir,
                        screen.index,
                        "failed",
                        error=_sanitized_task_text(self.api, task.error) or safe_error,
                        attempts=attempts,
                        input_fingerprint=request.input_fingerprint,
                        prompt_hash=prompt_hashes[screen.index],
                        request_settings=request_settings,
                        **task_fields,
                    )
                elif isinstance(task, ImageTaskSnapshot):
                    last_task_suffix = _task_id_suffix(task.task_id)
                if request.progress_callback:
                    request.progress_callback(
                        screen.index,
                        request.target_count,
                        len(completed),
                        tuple(failed),
                    )
                break
            task = _generated_task(generated)
            task_fields = _checkpoint_fields_for_persistence(self.api, task)
            task_fields.pop("state", None)
            task_fields.pop("error", None)
            local_path = str(generated.local_path)
            record_detail_checkpoint(
                request.product_dir,
                screen.index,
                "done",
                local_path,
                attempts=attempts,
                input_fingerprint=request.input_fingerprint,
                prompt_hash=prompt_hashes[screen.index],
                error=_sanitized_task_text(self.api, task.error),
                request_settings=request_settings,
                **task_fields,
            )
            completed.add(screen.index)
            last_task_suffix = _task_id_suffix(task.task_id)
            used_model = str(generated.model) or used_model
            if request.progress_callback:
                request.progress_callback(
                    screen.index,
                    request.target_count,
                    len(completed),
                    tuple(failed),
                )

        local_paths = _completed_paths(
            request.product_dir,
            request.target_count,
            request.input_fingerprint,
            prompt_hashes,
        )
        completed_count = len(completed)
        return ImageProviderResult(
            succeeded=not still_running and completed_count == request.target_count,
            local_paths=local_paths,
            used_model=used_model,
            completed_count=completed_count,
            failed_indexes=tuple(failed),
            partial_complete=0 < completed_count < request.target_count,
            error="; ".join(errors),
            still_running=still_running,
            task_id_suffix=last_task_suffix,
        )


class LovartImageProvider:
    """Thin compatibility adapter that leaves Lovart's confirmation flow untouched."""

    def __init__(self, bot: Any, logger: Any | None = None) -> None:
        self.bot = bot
        self.logger = logger
        self.confirmation_advisor: Any | None = None
        self._project_ids: dict[str, str] = {}

    def detail_execution_settings(self) -> dict[str, object]:
        """Return the selected non-secret Lovart tool and run-mode settings."""
        tool_config = getattr(self.bot, "tool_config", {})
        source = tool_config if isinstance(tool_config, Mapping) else {}
        settings = {
            key: _json_safe_setting(source.get(key))
            for key in (
                "image_model",
                "image_models",
                "model_selection",
                "prefer_models",
                "include_tools",
                "mode",
                "tool_names",
            )
            if key in source
        }
        fast_mode = getattr(self.bot, "_fast_mode", None)
        if isinstance(fast_mode, bool):
            settings["run_mode"] = "fast" if fast_mode else "unlimited"
        configured_models = getattr(self.bot, "_configured_unlimited_models", None)
        if isinstance(fast_mode, bool):
            configured_selected = bool(
                not fast_mode
                and isinstance(configured_models, (list, tuple))
                and configured_models
            )
            settings["configured_unlimited_models_selected"] = configured_selected
            if configured_selected:
                settings["configured_unlimited_models"] = [
                    str(model) for model in configured_models
                ]
        return settings

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


def _read_detail_checkpoint(product_dir: str | Path, index: int) -> dict[str, object]:
    checkpoints = read_status(product_dir).get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return {}
    checkpoint = _checkpoint_for_index(checkpoints, index)
    return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}


def _task_snapshot_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> ImageTaskSnapshot | None:
    task_id = str(checkpoint.get("task_id") or "").strip()
    if not task_id:
        return None
    try:
        task_created_at = float(checkpoint.get("task_created_at") or 0.0)
    except (TypeError, ValueError):
        return None
    state = str(checkpoint.get("state") or "running")
    if state == "done":
        state = "success"
    return ImageTaskSnapshot(
        task_id=task_id,
        state=state,
        is_final=bool(checkpoint.get("is_final", False)),
        task_created_at=task_created_at,
        progress=str(checkpoint.get("progress") or ""),
        status=str(checkpoint.get("status") or ""),
        status_group=str(checkpoint.get("status_group") or ""),
        result_url=str(checkpoint.get("result_url") or ""),
        result_type=str(checkpoint.get("result_type") or ""),
        error=str(checkpoint.get("error") or ""),
        cost=checkpoint.get("cost"),
    )


def _task_checkpoint_fields(task: ImageTaskSnapshot) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "task_created_at": task.task_created_at,
        "state": task.state,
        "is_final": task.is_final,
        "progress": task.progress,
        "status": task.status,
        "status_group": task.status_group,
        "result_url": task.result_url,
        "result_type": task.result_type,
        "error": task.error,
        "cost": _json_safe_setting(task.cost),
    }


def _checkpoint_fields_for_persistence(
    api: object,
    task: ImageTaskSnapshot,
) -> dict[str, object]:
    fields = _task_checkpoint_fields(task)
    api_key = _resolved_api_key(api)
    if not api_key:
        return fields
    return {
        key: value
        if key == "task_id"
        else _redact_checkpoint_value(value, api_key)
        for key, value in fields.items()
    }


def _redact_checkpoint_value(value: object, api_key: str) -> object:
    if isinstance(value, str):
        return value.replace(api_key, "[redacted]")
    if isinstance(value, Mapping):
        return {
            str(key): _redact_checkpoint_value(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_checkpoint_value(item, api_key) for item in value]
    return _json_safe_setting(value)


def _generated_task(generated: object) -> ImageTaskSnapshot:
    task = generated.task if isinstance(generated, GeneratedImage) else getattr(generated, "task", None)
    if not isinstance(task, ImageTaskSnapshot):
        raise TypeError("GPT Image transport returned no task snapshot")
    return task


def _openai_request_settings(api: object, image_size: str) -> dict[str, object]:
    config = getattr(api, "config", None)
    raw_base_url = getattr(config, "base_url", "") if config is not None else ""
    base_url = ""
    if isinstance(raw_base_url, str) and raw_base_url.strip():
        base_url = normalize_openai_image_base_url(raw_base_url)
    raw_model = getattr(config, "model", "gpt-image-2") if config is not None else "gpt-image-2"
    model = raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else "gpt-image-2"
    raw_resolution = getattr(config, "resolution", "1K") if config is not None else "1K"
    resolution = raw_resolution.upper() if isinstance(raw_resolution, str) else "1K"
    if resolution not in {"1K", "2K", "4K"}:
        resolution = "1K"
    raw_merge = getattr(config, "merge_reference_images", False) if config is not None else False
    merge = raw_merge if isinstance(raw_merge, bool) else False
    return {
        "model": model,
        "size": _provider_image_size(resolution, image_size),
        "base_url": base_url,
        "merge_reference_images": merge,
    }


def _support_request_identity(
    api: object,
    request: SupportImageRequest,
) -> dict[str, object]:
    final_prompt = append_aspect_instruction(request.prompt, request.image_size)
    prompt_hash = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
    return {
        "input_fingerprint": str(request.input_fingerprint or ""),
        "prompt_hash": f"sha256:{prompt_hash}",
        **_openai_request_settings(api, request.image_size),
    }


def _checkpoint_matches_identity(
    checkpoint: Mapping[str, object],
    identity: Mapping[str, object],
) -> bool:
    return bool(checkpoint) and all(
        checkpoint.get(key) == expected for key, expected in identity.items()
    )


def _checkpoint_attempt_value(checkpoint: Mapping[str, object]) -> int:
    try:
        return max(0, int(checkpoint.get("attempts", 0)))
    except (TypeError, ValueError):
        return 0


def _task_id_suffix(task_id: str) -> str:
    value = str(task_id or "")
    if not value:
        return ""
    if len(value) <= 8:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return f"hash:{digest}"
    return value[-8:]


def _remove_file(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _sanitized_provider_error(api: object, exc: BaseException) -> str:
    message = str(exc)
    api_key = _resolved_api_key(api)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message


def _sanitized_task_text(api: object, value: object) -> str:
    message = str(value or "")
    api_key = _resolved_api_key(api)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message


def _resolved_api_key(api: object) -> str:
    config = getattr(api, "config", None)
    api_key = getattr(config, "api_key", "") if config is not None else ""
    return api_key if isinstance(api_key, str) else ""


def _checkpoint_matches_fingerprint(
    checkpoint: Mapping[str, object],
    input_fingerprint: str,
) -> bool:
    expected = str(input_fingerprint or "")
    return not expected or str(checkpoint.get("input_fingerprint") or "") == expected


def detail_screen_prompt_hash(
    screen: DetailScreen,
    target_count: int,
    image_size: str = "",
) -> str:
    """Hash the exact prompt payload associated with one paid detail screen."""
    payload = {
        "schema": 1,
        "index": int(screen.index),
        "target_count": int(target_count),
        "prompt": append_aspect_instruction(screen.prompt, image_size),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _expected_prompt_hash(
    screen_prompt_hashes: Mapping[int, str] | None,
    index: int,
) -> str | None:
    if screen_prompt_hashes is None:
        return None
    return str(screen_prompt_hashes.get(index) or "")


def _checkpoint_matches_prompt(
    checkpoint: Mapping[str, object],
    expected_prompt_hash: str | None,
) -> bool:
    if expected_prompt_hash is None:
        return True
    return bool(expected_prompt_hash) and str(checkpoint.get("prompt_hash") or "") == expected_prompt_hash


def _completed_paths(
    product_dir: str | Path,
    expected_count: int,
    input_fingerprint: str = "",
    screen_prompt_hashes: Mapping[int, str] | None = None,
) -> tuple[str, ...]:
    checkpoints = read_status(product_dir).get(_CHECKPOINT_FIELD, {})
    if not isinstance(checkpoints, Mapping):
        return ()
    paths: list[str] = []
    for index in range(1, max(0, int(expected_count)) + 1):
        checkpoint = _checkpoint_for_index(checkpoints, index)
        resolved_path = _resolved_detail_checkpoint_path(product_dir, index, checkpoint)
        if (
            isinstance(checkpoint, Mapping)
            and checkpoint.get("state") == "done"
            and _checkpoint_matches_fingerprint(checkpoint, input_fingerprint)
            and _checkpoint_matches_prompt(
                checkpoint,
                _expected_prompt_hash(screen_prompt_hashes, index),
            )
            and resolved_path is not None
        ):
            paths.append(str(resolved_path))
    return tuple(paths)


def _resolved_detail_checkpoint_path(
    product_dir: str | Path,
    index: int,
    checkpoint: object,
) -> Path | None:
    if not isinstance(checkpoint, Mapping):
        return None
    saved_path = Path(str(checkpoint.get("local_path") or ""))
    if is_valid_image_file(saved_path):
        return saved_path
    canonical = Path(product_dir) / "gpt_image" / "detail" / f"{index:02d}.png"
    return canonical if is_valid_image_file(canonical) else None


def _json_safe_setting(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_setting(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_setting(item) for item in value]
    return str(value)


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
