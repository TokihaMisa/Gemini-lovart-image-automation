import base64
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from image_generation import GenerationRouting
from image_providers import ImageProviderResult, LovartImageProvider, OpenAIImageProvider
from openai_image_api import GeneratedImage, ImageTaskSnapshot, ImageTaskStillRunning
from utils import read_status


VALID_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def write_valid_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64))
    return str(path)


def write_truncated_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64)[:-12])
    return str(path)


class CheckpointingOpenAIAPI:
    """Transport double that distinguishes paid creates from task resumes."""

    def __init__(
        self,
        snapshots: tuple[ImageTaskSnapshot, ...],
        *,
        base_url: str = "https://api.lk888.ai",
        model: str = "gpt-image-2",
        resolution: str = "1K",
        merge_reference_images: bool = False,
        outcome: str = "success",
    ) -> None:
        self.snapshots = snapshots
        self.outcome = outcome
        self.create_posts = 0
        self.calls: list[dict[str, object]] = []
        self.output_existed_at_call: list[bool] = []
        self.config = SimpleNamespace(
            base_url=base_url,
            model=model,
            resolution=resolution,
            merge_reference_images=merge_reference_images,
        )

    def generate_edit(self, **kwargs):
        self.calls.append(dict(kwargs))
        self.output_existed_at_call.append(Path(kwargs["output_path"]).exists())
        resume_task = kwargs.get("resume_task")
        if resume_task is None:
            self.create_posts += 1
        emitted = list(self.snapshots)
        if not emitted:
            if not isinstance(resume_task, ImageTaskSnapshot):
                raise AssertionError("a snapshot or resume task is required")
            emitted = [resume_task]
        callback = kwargs.get("task_callback")
        for snapshot in emitted:
            if callback is not None:
                callback(snapshot)
        final = emitted[-1]
        if self.outcome == "still_running":
            raise ImageTaskStillRunning(final)
        if self.outcome == "crash":
            raise KeyboardInterrupt("poll crashed after task callback")
        if self.outcome == "error":
            raise RuntimeError("ordinary transport failure after task callback")
        if self.outcome == "failed":
            error = RuntimeError(final.error or "provider task failed")
            error.task = final
            raise error
        local_path = write_valid_png(Path(kwargs["output_path"]))
        return GeneratedImage(local_path, self.config.model, final)


class RecordingImageProvider:
    def __init__(
        self,
        name: str,
        fail_indexes=frozenset(),
        fail_support_steps=frozenset(),
    ) -> None:
        self.name = name
        self.fail_indexes = frozenset(fail_indexes)
        self.fail_support_steps = frozenset(fail_support_steps)
        self.generated_indexes: list[int] = []
        self.support_steps: list[str] = []

    def generate_support_image(self, request):
        self.support_steps.append(request.step_name)
        if request.step_name in self.fail_support_steps:
            return ImageProviderResult(
                succeeded=False,
                error=f"{request.step_name} support failed",
            )
        path = write_valid_png(
            request.product_dir / self.name / "support" / f"{request.step_name}.png"
        )
        return ImageProviderResult(
            succeeded=True,
            local_paths=(path,),
            used_model=self.name,
            completed_count=1,
        )

    def generate_detail_set(self, request):
        paths = []
        failed = []
        for screen in request.screens:
            if screen.index in self.fail_indexes:
                failed.append(screen.index)
                continue
            self.generated_indexes.append(screen.index)
            paths.append(
                write_valid_png(
                    request.product_dir / self.name / "detail" / f"{screen.index:02d}.png"
                )
            )
        return ImageProviderResult(
            succeeded=not failed,
            local_paths=tuple(paths),
            used_model=self.name,
            completed_count=len(paths),
            failed_indexes=tuple(failed),
            partial_complete=bool(paths and failed),
            error="detail generation failed" if failed else "",
        )


class RecordingRegistry:
    def __init__(self, lovart_provider, openai_provider) -> None:
        self.providers = {
            "lovart": lovart_provider,
            "openai_image": openai_provider,
        }
        self.get_calls: list[str] = []

    def get(self, name: str):
        self.get_calls.append(name)
        return self.providers[name]


class RecordingOpenAIAPI:
    def __init__(
        self,
        fail_indexes=frozenset(),
        *,
        base_url="https://images.example/v1",
        model="gpt-image-2",
        resolution="1K",
    ) -> None:
        self.fail_indexes = frozenset(fail_indexes)
        self.generated_indexes: list[int] = []
        self.config = SimpleNamespace(
            base_url=base_url,
            model=model,
            resolution=resolution,
        )

    def generate_edit(self, *, output_path, **_kwargs):
        output_path = Path(output_path)
        index = int(output_path.stem)
        if index in self.fail_indexes:
            raise RuntimeError(f"screen {index} failed")
        self.generated_indexes.append(index)
        write_valid_png(output_path)
        return GeneratedImage(
            local_path=str(output_path),
            model=self.config.model,
            task=ImageTaskSnapshot(
                task_id=f"recording-task-{index}",
                state="success",
                is_final=True,
                task_created_at=0.0,
                result_url=f"https://cdn.example/{index}.png",
                result_type="image",
            ),
        )


class PipelineOpenAIProvider:
    name = "openai_image"

    def __init__(self, api, fail_support_steps=frozenset()) -> None:
        self.support = RecordingImageProvider(
            self.name,
            fail_support_steps=fail_support_steps,
        )
        self.detail = OpenAIImageProvider(api)

    def generate_support_image(self, request):
        return self.support.generate_support_image(request)

    def generate_detail_set(self, request):
        return self.detail.generate_detail_set(request)

    def detail_execution_settings(self):
        return self.detail.detail_execution_settings()


@dataclass(frozen=True)
class PipelineRunResult:
    success: int
    fail: int
    skipped: int
    still_running: int
    product_dir: Path
    registry: RecordingRegistry
    generated_indexes: tuple[int, ...]
    append_result: Mock
    gemini: Mock


def run_product_pipeline(
    tmp_path,
    support_provider,
    detail_provider,
    detail_count=2,
    prompt_screen_count=None,
    fail_indexes=frozenset(),
    fail_support_steps=frozenset(),
    lovart=None,
    openai_api=None,
    resume=True,
    openai_base_url="https://images.example/v1",
    openai_model="gpt-image-2",
    openai_resolution="1K",
    screen_label="Screen",
    prompt_responses=None,
):
    product_dir = Path(tmp_path) / "products" / "SKU-ROUTING"
    product_image = write_valid_png(product_dir / "product.png")
    product = SimpleNamespace(
        id="SKU-ROUTING",
        name_cn="Routing product",
        language="English",
        selling_points="Durable and compact",
        image_size="1:1",
        image_paths=[product_image],
        reference_images_are_product=False,
    )
    screen_count = detail_count if prompt_screen_count is None else prompt_screen_count
    marked_prompt = "\n\n".join(
        f"[[SCREEN {index:02d}]]\n{screen_label} {index}\n[[/SCREEN {index:02d}]]"
        for index in range(1, screen_count + 1)
    )
    gemini = Mock()
    if prompt_responses is None:
        gemini.generate_prompt.return_value = marked_prompt
    else:
        gemini.generate_prompt.side_effect = list(prompt_responses)

    supplied_lovart = lovart is not None
    if not supplied_lovart:
        lovart = Mock()
        lovart.create_project.return_value = "project-routing"
        lovart.validate_project.return_value = True
        lovart.create_support_image.side_effect = lambda **kwargs: {
            "generation_succeeded": True,
            "project_id": kwargs.get("project_id", "project-routing"),
            "local_path": write_valid_png(
                product_dir / "lovart_steps" / kwargs["step_name"] / f"{kwargs['step_name']}.png"
            ),
            "used_model": "lovart",
        }
        lovart.create_and_generate.return_value = {
            "generation_succeeded": True,
            "project_id": "project-routing",
            "used_model": "lovart",
        }
    if openai_api is None:
        openai_api = RecordingOpenAIAPI(
            fail_indexes=fail_indexes,
            base_url=openai_base_url,
            model=openai_model,
            resolution=openai_resolution,
        )
    lovart_provider = LovartImageProvider(lovart)
    openai_provider = PipelineOpenAIProvider(
        openai_api,
        fail_support_steps=fail_support_steps,
    )
    registry = RecordingRegistry(lovart_provider, openai_provider)
    routing = GenerationRouting(support_provider, detail_provider, detail_count)
    run_dir = Path(tmp_path) / "run"
    logger = Mock()
    append_result = Mock()

    with (
        patch("main.product_output_dir", return_value=product_dir),
        patch("main._backfill_result_project_urls", return_value=0),
        patch("main.append_result", append_result),
        patch("utils.organize_output_folders"),
    ):
        counters = main._process_products_once(
            [product],
            gemini,
            lovart,
            logger,
            run_dir,
            resume=resume,
            image_registry=registry,
            routing=routing,
        )

    recorded_indexes = getattr(openai_api, "generated_indexes", ())
    generated_indexes = tuple(
        recorded_indexes if isinstance(recorded_indexes, (list, tuple)) else ()
    )
    return PipelineRunResult(
        *counters, product_dir, registry, generated_indexes, append_result, gemini
    )
