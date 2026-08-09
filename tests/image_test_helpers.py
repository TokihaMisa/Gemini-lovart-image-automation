import base64
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from image_generation import GenerationRouting
from image_providers import ImageProviderResult


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


class RecordingImageProvider:
    def __init__(self, name: str, fail_indexes=frozenset()) -> None:
        self.name = name
        self.fail_indexes = frozenset(fail_indexes)
        self.generated_indexes: list[int] = []
        self.support_steps: list[str] = []

    def generate_support_image(self, request):
        self.support_steps.append(request.step_name)
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


@dataclass(frozen=True)
class PipelineRunResult:
    success: int
    fail: int
    skipped: int
    still_running: int
    product_dir: Path
    registry: RecordingRegistry
    generated_indexes: tuple[int, ...]


def run_product_pipeline(
    tmp_path,
    support_provider,
    detail_provider,
    detail_count=2,
    prompt_screen_count=None,
    fail_indexes=frozenset(),
    lovart=None,
    openai_api=None,
):
    del openai_api
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
        f"[[SCREEN {index:02d}]]\nScreen {index}\n[[/SCREEN {index:02d}]]"
        for index in range(1, screen_count + 1)
    )
    gemini = Mock()
    gemini.generate_prompt.return_value = marked_prompt

    if lovart is None:
        lovart = Mock()
    lovart.create_project.return_value = "project-routing"
    lovart.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "project-routing",
        "used_model": "lovart",
    }
    lovart_provider = RecordingImageProvider("lovart", fail_indexes=fail_indexes)
    openai_provider = RecordingImageProvider("openai_image", fail_indexes=fail_indexes)
    registry = RecordingRegistry(lovart_provider, openai_provider)
    routing = GenerationRouting(support_provider, detail_provider, detail_count)
    run_dir = Path(tmp_path) / "run"
    logger = Mock()

    with (
        patch("main.product_output_dir", return_value=product_dir),
        patch("main._backfill_result_project_urls", return_value=0),
        patch("main.append_result"),
        patch("utils.organize_output_folders"),
    ):
        counters = main._process_products_once(
            [product],
            gemini,
            lovart,
            logger,
            run_dir,
            image_registry=registry,
            routing=routing,
        )

    generated_indexes = tuple(openai_provider.generated_indexes)
    return PipelineRunResult(*counters, product_dir, registry, generated_indexes)
