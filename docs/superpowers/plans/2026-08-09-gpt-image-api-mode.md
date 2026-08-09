# GPT Image API Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently selectable Lovart or OpenAI-compatible GPT Image generation for support images and final detail sets while preserving the existing prompt-source choices, dynamic detail-page count, safe credential storage, and resumability.

**Architecture:** Introduce provider-neutral routing, screen parsing, checkpoint types, and lazy provider adapters. Keep `LovartBot` as the existing Lovart implementation, add a stdlib-based `OpenAIImageAPI` for `/images/edits`, and migrate `main.py` in two stages so legacy tests and existing Lovart resumes remain valid.

**Tech Stack:** Python 3.12+, urllib, Pillow, PyYAML, Gradio, pytest/unittest, existing retry and status helpers; no new runtime dependency.

## Global Constraints

- Prompt sources remain Gemini Browser, Gemini API, and NVIDIA; do not change their selection semantics.
- `prompt_settings.detail_page_count` remains an integer from 1 through 50; snapshot it per started product and never hard-code 12.
- OpenAI-compatible Base URLs are normalized to exactly one trailing `/v1`; default to `https://hapiopen.cc/v1` and model `gpt-image-2`.
- Store the image API credential only as `OPENAI_IMAGE_API_KEY` in local `.env`; never write or log it in config, status, summaries, tests, or exception text.
- All image-to-image steps call `/images/edits`; never silently downgrade to text-only generation or switch paid providers.
- Automated tests use mocked HTTP only. A real paid image call occurs only from the explicit WebUI test button.
- Keep existing Lovart output directories and `lovart_*` state readable; add provider-neutral state without breaking v1.3.x resumes.
- Generate final screens serially, checkpoint after each validated image, and resume only missing or failed indexes.
- Do not disable TLS verification. Reject non-HTTP(S), loopback, link-local, and private-network result URLs.

---

## File Map

- Create `image_generation.py`: provider names, routing normalization, screen parsing, target-count snapshot, detail checkpoints, and provider-neutral prompt composition.
- Create `openai_image_api.py`: OpenAI-compatible image configuration, multipart transport, retries, response parsing, secure URL download, image validation, and redaction-safe errors.
- Create `image_providers.py`: provider request/result types, Lovart/OpenAI adapters, and lazy provider registry.
- Modify `utils.py`: provider-neutral final prompt wording and stable screen delimiters.
- Modify `prompt_settings.py`: remove the obsolete “final images only in Lovart” locked rule.
- Modify `main.py`: CLI options, lazy routing, support-stage branching, detail-stage branching, partial-result reporting, and backward-compatible processor entry points.
- Modify `webui.py`: OpenAI Image settings, credential transaction, paid test action, run selectors, subprocess arguments, and progress display.
- Modify `config.example.yaml`, `.env.example`, `README.md`, and `PROJECT_OVERVIEW.md`: defaults and usage documentation.
- Create `tests/test_image_generation.py`, `tests/test_openai_image_api.py`, `tests/test_image_providers.py`, and `tests/test_image_provider_routing.py`; extend existing WebUI and medium-priority suites.
- Create `tests/image_test_helpers.py`: valid PNG creation plus explicit fake products, prompt sources, providers, registries, and pipeline runners shared by provider-routing tests.

---

### Task 1: Provider-neutral routing, dynamic screen parsing, and task snapshots

**Files:**
- Create: `image_generation.py`
- Modify: `utils.py:389-510`
- Modify: `prompt_settings.py:20-35`
- Test: `tests/test_image_generation.py`
- Test: `tests/test_high_priority.py:149-175`

**Interfaces:**
- Produces: `PROVIDER_LOVART`, `PROVIDER_OPENAI_IMAGE`, `GenerationRouting`, `DetailScreen`, `DetailScreenParseError`, `routing_from_config(config, prompt_settings)`, `split_detail_screens(text, expected_count)`, `ensure_detail_page_count_snapshot(product_dir, configured_count)`, and `compose_detail_image_prompt(screen, product_name_cn, language, selling_points, image_note, image_size, prompt_settings) -> str`.
- Consumes: `prompt_settings.normalize_prompt_settings`, `utils.read_status`, and `utils.update_status`.

- [ ] **Step 1: Write failing routing, delimiter, parser, and snapshot tests**

```python
from pathlib import Path

import pytest

from image_generation import (
    DetailScreenParseError,
    PROVIDER_OPENAI_IMAGE,
    ensure_detail_page_count_snapshot,
    routing_from_config,
    split_detail_screens,
)
from utils import build_design_prompt, read_status


def test_routing_uses_dynamic_detail_count():
    routing = routing_from_config(
        {"image_generation": {"support_provider": "openai_image", "detail_provider": "lovart"}},
        {"detail_page_count": 8},
    )
    assert routing.support_provider == PROVIDER_OPENAI_IMAGE
    assert routing.detail_page_count == 8


def test_design_prompt_requires_stable_screen_markers():
    prompt = build_design_prompt("杯子", "英语", "保温", prompt_settings={"detail_page_count": 3})
    assert "[[SCREEN 01]]" in prompt
    assert "[[/SCREEN 01]]" in prompt
    assert "[[SCREEN 03]]" in prompt


def test_parser_accepts_markers_and_rejects_wrong_count():
    text = "[[SCREEN 01]]\nHero\n[[/SCREEN 01]]\n[[SCREEN 02]]\nFeature\n[[/SCREEN 02]]"
    assert [item.index for item in split_detail_screens(text, 2)] == [1, 2]
    with pytest.raises(DetailScreenParseError):
        split_detail_screens(text, 3)


def test_snapshot_survives_later_global_setting_change(tmp_path: Path):
    assert ensure_detail_page_count_snapshot(tmp_path, 6) == 6
    assert ensure_detail_page_count_snapshot(tmp_path, 9) == 6
    assert read_status(tmp_path)["detail_page_count_snapshot"] == 6
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_image_generation.py tests/test_high_priority.py -q`

Expected: collection fails because `image_generation` does not exist, then the marker assertion fails after the module skeleton is introduced.

- [ ] **Step 3: Add the neutral types, strict parser, snapshot, and prompt markers**

```python
@dataclass(frozen=True)
class GenerationRouting:
    support_provider: str
    detail_provider: str
    detail_page_count: int


@dataclass(frozen=True)
class DetailScreen:
    index: int
    prompt: str


def ensure_detail_page_count_snapshot(product_dir: str | Path, configured_count: int) -> int:
    status = read_status(product_dir)
    existing = status.get("detail_page_count_snapshot")
    if existing is not None:
        return normalize_prompt_settings({"detail_page_count": existing})["detail_page_count"]
    count = normalize_prompt_settings({"detail_page_count": configured_count})["detail_page_count"]
    update_status(product_dir, "detail_target_snapshotted", detail_page_count_snapshot=count)
    return count
```

Implement marker parsing first, then a legacy fallback that recognizes sequential Chinese/English `屏 NN` headings. Require indexes `1..expected_count` exactly once. Update `build_design_prompt()` to demand literal markers for every configured screen and update `LOCKED_PROMPT_RULES` to say final images are generated only by the user-selected image provider.

- [ ] **Step 4: Run focused and prompt-regression tests and verify GREEN**

Run: `uv run pytest tests/test_image_generation.py tests/test_high_priority.py tests/test_webui_model_settings.py -q`

Expected: all selected tests pass; the locked-rules preview mentions the selected image provider rather than Lovart-only generation.

- [ ] **Step 5: Commit the neutral planning primitives**

```powershell
git add image_generation.py utils.py prompt_settings.py tests/test_image_generation.py tests/test_high_priority.py tests/test_webui_model_settings.py
git commit -m "feat: add provider-neutral image generation planning"
```

---

### Task 2: OpenAI Image configuration and redaction-safe errors

**Files:**
- Create: `openai_image_api.py`
- Test: `tests/test_openai_image_api.py`

**Interfaces:**
- Produces: `OpenAIImageAPIConfig.from_config(config, api_key)`, `normalize_openai_image_base_url(value)`, `OpenAIImageAPIError(code, user_message, status_code=None, retryable=False)`, and `GeneratedImage(local_path, model)`.
- Consumes: `utils.env_or_config` only at application construction time; the config object itself receives the resolved key and never serializes it.

- [ ] **Step 1: Write failing URL, validation, and secret-redaction tests**

```python
import pytest

from openai_image_api import OpenAIImageAPIConfig, OpenAIImageAPIError, normalize_openai_image_base_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://hapiopen.cc", "https://hapiopen.cc/v1"),
        ("https://hapiopen.cc/v1/", "https://hapiopen.cc/v1"),
        ("https://gateway.test/root/v1", "https://gateway.test/root/v1"),
    ],
)
def test_normalize_openai_image_base_url(raw, expected):
    assert normalize_openai_image_base_url(raw) == expected


def test_config_requires_key_without_echoing_it():
    with pytest.raises(OpenAIImageAPIError) as ctx:
        OpenAIImageAPIConfig.from_config({"openai_image": {"base_url": "https://hapiopen.cc"}}, api_key="")
    assert ctx.value.code == "missing_key"
    assert "Authorization" not in str(ctx.value)


def test_invalid_url_never_echoes_key():
    secret = "sensitive-test-key"
    with pytest.raises(OpenAIImageAPIError) as ctx:
        OpenAIImageAPIConfig.from_config({"openai_image": {"base_url": "https://bad host"}}, api_key=secret)
    assert secret not in str(ctx.value)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_openai_image_api.py -q`

Expected: import fails because `openai_image_api.py` does not exist.

- [ ] **Step 3: Implement immutable validated configuration**

```python
@dataclass(frozen=True)
class OpenAIImageAPIConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://hapiopen.cc/v1"
    model: str = "gpt-image-2"
    resolution: str = "1K"
    timeout: float = 600.0
    max_attempts: int = 4
    retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0)

    @classmethod
    def from_config(cls, config: Mapping[str, object], api_key: str) -> "OpenAIImageAPIConfig":
        section = config.get("openai_image", {}) if isinstance(config, Mapping) else {}
        if not api_key.strip():
            raise OpenAIImageAPIError("missing_key", "请先填写 GPT Image API 密钥。")
        resolution = str(section.get("resolution", "1K")).upper()
        if resolution not in {"1K", "2K", "4K"}:
            raise OpenAIImageAPIError("invalid_resolution", "GPT Image 分辨率必须是 1K、2K 或 4K。")
        return cls(api_key=api_key.strip(), base_url=normalize_openai_image_base_url(section.get("base_url")), model=str(section.get("model") or "gpt-image-2").strip(), resolution=resolution)
```

Reject CR/LF, credentials embedded in URLs, missing hostnames, query strings, fragments, and paths that already contain a duplicated terminal `/v1/v1`. Ensure `__repr__`, `str(error)`, and all public errors never contain the key.

- [ ] **Step 4: Run the configuration tests and verify GREEN**

Run: `uv run pytest tests/test_openai_image_api.py -q`

Expected: all configuration and redaction tests pass.

- [ ] **Step 5: Commit the image API configuration layer**

```powershell
git add openai_image_api.py tests/test_openai_image_api.py
git commit -m "feat: validate OpenAI image API settings"
```

---

### Task 3: Multipart image edits, retries, response parsing, and safe downloads

**Files:**
- Modify: `openai_image_api.py`
- Modify: `tests/test_openai_image_api.py`

**Interfaces:**
- Produces: `OpenAIImageAPI.generate_edit(prompt, image_paths, output_path, image_size="") -> GeneratedImage` and `OpenAIImageAPI.test_edit(output_dir) -> GeneratedImage`.
- Consumes: `OpenAIImageAPIConfig`; Pillow for final decode verification; `urllib.request.urlopen` with keyword `timeout=`.

- [ ] **Step 1: Add failing multipart, Base64, retry, URL-security, and image-validation tests**

```python
@patch("openai_image_api.urllib.request.urlopen")
def test_generate_edit_posts_multipart_and_saves_b64_png(urlopen, tmp_path):
    urlopen.return_value.__enter__.return_value.read.return_value = json.dumps({
        "data": [{"b64_json": VALID_ONE_PIXEL_PNG_BASE64}]
    }).encode("utf-8")
    client = make_client()
    result = client.generate_edit("keep the product exact", [make_png(tmp_path / "source.png")], tmp_path / "out.png")
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://hapiopen.cc/v1/images/edits"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Content-type"].startswith("multipart/form-data; boundary=")
    assert result.local_path == str(tmp_path / "out.png")


@patch("openai_image_api.urllib.request.urlopen")
def test_429_retries_but_401_does_not(urlopen, tmp_path):
    urlopen.side_effect = [HTTPError("https://hapiopen.cc/v1/images/edits", 429, "busy", {}, None), fake_png_response()]
    make_client(retry_delays=(0.0,)).generate_edit("prompt", [make_png(tmp_path / "source.png")], tmp_path / "out.png")
    assert urlopen.call_count == 2


def test_result_url_rejects_private_network():
    with pytest.raises(OpenAIImageAPIError) as ctx:
        validate_remote_image_url("http://127.0.0.1/result.png")
    assert ctx.value.code == "unsafe_result_url"
```

- [ ] **Step 2: Run the transport tests and verify RED**

Run: `uv run pytest tests/test_openai_image_api.py -q`

Expected: failures report missing `OpenAIImageAPI.generate_edit`, response helpers, and URL validation.

- [ ] **Step 3: Implement the multipart and response pipeline**

```python
def generate_edit(self, prompt, image_paths, output_path, image_size="") -> GeneratedImage:
    body, content_type = encode_multipart(
        fields={"model": self.config.model, "prompt": append_aspect_instruction(prompt, image_size), "size": self.config.resolution},
        files=[("image[]", Path(path)) for path in image_paths],
    )
    payload = self._request_json("/images/edits", body, content_type)
    image_bytes = self._extract_image_bytes(payload)
    target = Path(output_path)
    atomic_save_validated_image(image_bytes, target)
    return GeneratedImage(local_path=str(target), model=self.config.model)
```

Use an unpredictable multipart boundary; sanitize filenames to basenames; validate all input files exist, are non-empty, and decode as images before network access. Map 401/403 and 400/404 to permanent user errors, retry 429 and recoverable 5xx/network failures up to `max_attempts`, and call `urlopen(request, timeout=self.config.timeout)` with a keyword argument. For URL results, resolve every address and reject loopback/private/link-local targets before downloading. Save via a sibling temporary file and `os.replace()` only after Pillow successfully decodes the bytes.

Send the selected `1K` / `2K` / `4K` value as the multipart `size` field used by the target gateway. Preserve the Excel ratio by appending its exact value to the prompt through `append_aspect_instruction()`; do not crop, stretch, or silently rewrite the returned image. The explicit paid test is the compatibility gate when a different OpenAI-compatible gateway uses another size vocabulary.

- [ ] **Step 4: Run transport, retry, and network regression tests and verify GREEN**

Run: `uv run pytest tests/test_openai_image_api.py tests/test_model_provider.py tests/test_network_retry.py -q`

Expected: all selected tests pass; no test performs external I/O.

- [ ] **Step 5: Commit the OpenAI-compatible image transport**

```powershell
git add openai_image_api.py tests/test_openai_image_api.py
git commit -m "feat: call OpenAI-compatible image edits API"
```

---

### Task 4: Provider adapters, lazy construction, and screen checkpoints

**Files:**
- Create: `image_providers.py`
- Create: `tests/test_image_providers.py`
- Create: `tests/image_test_helpers.py`
- Modify: `image_generation.py`
- Modify: `tests/test_image_generation.py`

**Interfaces:**
- Consumes: `OpenAIImageAPI`, `LovartBot`, `DetailScreen`, `utils.read_status`, and `utils.update_status`.
- Produces: `SupportImageRequest`, `DetailSetRequest`, `ImageProviderResult`, `LovartImageProvider`, `OpenAIImageProvider`, `LazyImageProviderRegistry.get(name)`, `read_completed_detail_indexes(product_dir, expected_count)`, and `record_detail_checkpoint(product_dir, index, state, local_path="", error="", attempts=0)`.

- [ ] **Step 1: Write failing adapter, lazy-registry, and resume tests**

```python
import base64
from pathlib import Path


VALID_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk/x8AAusB9Y9Z4WQAAAAASUVORK5CYII="


def write_valid_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64))
    return str(path)


def test_registry_does_not_build_lovart_for_all_openai_run():
    lovart_factory = Mock(side_effect=AssertionError("Lovart must stay lazy"))
    openai_factory = Mock(return_value=Mock())
    registry = LazyImageProviderRegistry(lovart_factory, openai_factory)
    assert registry.get("openai_image") is openai_factory.return_value
    lovart_factory.assert_not_called()


def test_openai_detail_set_skips_valid_completed_indexes(tmp_path):
    first_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    record_detail_checkpoint(tmp_path, 1, "done", first_path)
    api = Mock()
    second_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "02.png")
    api.generate_edit.return_value = GeneratedImage(second_path, "gpt-image-2")
    provider = OpenAIImageProvider(api, logger=Mock())
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"), DetailScreen(2, "feature")),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=2,
    )
    result = provider.generate_detail_set(request)
    assert result.completed_count == 2
    assert api.generate_edit.call_count == 1
```

- [ ] **Step 2: Run the provider tests and verify RED**

Run: `uv run pytest tests/test_image_providers.py tests/test_image_generation.py -q`

Expected: import failures for `image_providers` and checkpoint helpers.

- [ ] **Step 3: Implement adapters and atomic per-screen state**

```python
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
    def get(self, name: str):
        normalized = normalize_image_provider(name)
        if normalized not in self._instances:
            factory = self._lovart_factory if normalized == PROVIDER_LOVART else self._openai_factory
            self._instances[normalized] = factory()
        return self._instances[normalized]
```

`OpenAIImageProvider.generate_detail_set()` processes screens in ascending order, validates any existing checkpoint file before skipping it, records `running/done/failed` per index, and returns `partial_complete=True` when at least one but fewer than the target count succeeds. `LovartImageProvider` delegates support generation and full-set generation to the current bot and maps its raw result without changing Lovart confirmation semantics.

- [ ] **Step 4: Run provider and existing Lovart tests and verify GREEN**

Run: `uv run pytest tests/test_image_providers.py tests/test_image_generation.py tests/test_lovart_api_upstream_sync.py tests/test_medium_priority.py -q`

Expected: all selected tests pass; existing Lovart tests observe unchanged bot calls.

- [ ] **Step 5: Commit provider adapters and checkpoints**

```powershell
git add image_generation.py image_providers.py tests/test_image_generation.py tests/test_image_providers.py
git commit -m "feat: add lazy image provider adapters"
```

---

### Task 5: Route support-image stages without eagerly creating Lovart

**Files:**
- Modify: `main.py:81-139, 341-711, 1160-1240`
- Create: `tests/test_image_provider_routing.py`
- Modify: `tests/image_test_helpers.py`
- Modify: `tests/test_medium_priority.py:1895-2145`

**Interfaces:**
- Consumes: `routing_from_config`, `LazyImageProviderRegistry`, `SupportImageRequest`, and legacy `lovart` arguments.
- Produces: `_build_image_provider_registry(config, logger, lovart=None)`, `_default_lovart_routing(prompt_settings)`, `_generate_support_images(product, product_dir, provider, prompt_settings, existing_status) -> tuple[str, str]`, backward-compatible `_process_products_once(products, gemini, lovart, logger, run_dir, resume=True, prompt_settings=None, image_registry=None, routing=None)` / `_process_products(products, gemini, lovart, logger, run_dir, resume=True, prompt_settings=None, image_registry=None, routing=None)`, and test types `PipelineRunResult` plus `run_product_pipeline(tmp_path, support_provider, detail_provider, detail_count=2, prompt_screen_count=None, fail_indexes=frozenset(), lovart=None, openai_api=None)`.

- [ ] **Step 1: Write failing four-combination and lazy-Lovart routing tests**

```python
@pytest.mark.parametrize(
    ("support", "detail", "expected_support", "expected_detail"),
    [
        ("lovart", "lovart", "lovart", "lovart"),
        ("openai_image", "lovart", "openai_image", "lovart"),
        ("lovart", "openai_image", "lovart", "openai_image"),
        ("openai_image", "openai_image", "openai_image", "openai_image"),
    ],
)
def test_pipeline_routes_each_stage_independently(tmp_path, support, detail, expected_support, expected_detail):
    run = run_product_pipeline(tmp_path, support, detail)
    assert run.registry.get_calls == [expected_support, expected_detail]


def test_all_openai_pipeline_never_validates_or_creates_lovart_project(tmp_path):
    lovart = Mock()
    lovart.create_project.side_effect = AssertionError("unexpected Lovart project")
    lovart.validate_project.side_effect = AssertionError("unexpected Lovart validation")
    run_product_pipeline(tmp_path, "openai_image", "openai_image", lovart=lovart)
    lovart.create_project.assert_not_called()
```

Implement `run_product_pipeline()` with a one-product fake, a prompt source returning exactly `detail_count` marked screens, providers that write valid PNGs through `write_valid_png()`, a registry that records every `get()` name, and patches for `main.product_output_dir` / result CSV writes so every artifact stays under `tmp_path`. Pass an explicit `GenerationRouting(support_provider, detail_provider, detail_count)` into `_process_products_once()`.

```python
@dataclass(frozen=True)
class PipelineRunResult:
    success: int
    fail: int
    skipped: int
    still_running: int
    product_dir: Path
    registry: RecordingRegistry
    generated_indexes: tuple[int, ...]
```

Have the helper unpack `_process_products_once()` into these four counters and derive `generated_indexes` from the recording OpenAI provider, so all later tests use one concrete return type.

- [ ] **Step 2: Run the routing tests and verify RED**

Run: `uv run pytest tests/test_image_provider_routing.py tests/test_medium_priority.py -q`

Expected: new routing tests fail because `main.py` still creates the Lovart project before support generation.

- [ ] **Step 3: Extract support routing and preserve legacy processor calls**

```python
def _process_products_once(products, gemini, lovart, logger, run_dir, resume=True, prompt_settings=None, image_registry=None, routing=None):
    registry = image_registry or _legacy_lovart_registry(lovart)
    effective_routing = routing or _default_lovart_routing(prompt_settings or {})
    support_provider = registry.get(effective_routing.support_provider)
```

Move Lovart project validation/creation into `LovartImageProvider` so GPT support generation never touches it. Keep the positional `lovart` parameter and default registry path until all existing tests are migrated. Store provider-neutral `white_bg_local_path` and `scene_local_path`, while the Lovart adapter continues writing old fields.

- [ ] **Step 4: Run routing, resume, timeout, and confirmation tests and verify GREEN**

Run: `uv run pytest tests/test_image_provider_routing.py tests/test_medium_priority.py tests/test_failed_retry_queue.py -q`

Expected: all selected tests pass, including old Lovart support-image resume behavior.

- [ ] **Step 5: Commit support-stage routing**

```powershell
git add main.py tests/test_image_provider_routing.py tests/test_medium_priority.py
git commit -m "feat: route support images by selected provider"
```

---

### Task 6: Route final detail sets, enforce dynamic completion, and report partial results

**Files:**
- Modify: `main.py:711-900, 939-1030`
- Modify: `utils.py:459-510`
- Modify: `tests/test_image_provider_routing.py`
- Modify: `tests/image_test_helpers.py`
- Modify: `tests/test_medium_priority.py:1398-2040`
- Modify: `tests/test_failed_retry_queue.py`

**Interfaces:**
- Consumes: `ensure_detail_page_count_snapshot`, `split_detail_screens`, `compose_detail_image_prompt`, `DetailSetRequest`, and `ImageProviderResult`.
- Produces: provider-neutral `detail_images`, `detail_completed_count`, `detail_generation_complete`, `partial_complete`, summary `artifact_count`, and UI `used_model` values.

- [ ] **Step 1: Write failing dynamic-count, partial-failure, and resume tests**

```python
def test_openai_detail_count_uses_snapshot_not_default_12(tmp_path):
    result = run_product_pipeline(tmp_path, "openai_image", "openai_image", detail_count=3)
    assert result.success == 1
    status = read_status(result.product_dir)
    assert status["detail_page_count_snapshot"] == 3
    assert status["detail_completed_count"] == 3


def test_partial_detail_failure_keeps_completed_images_and_resumes_only_missing(tmp_path):
    first = run_product_pipeline(tmp_path, "openai_image", "openai_image", detail_count=3, fail_indexes={2})
    assert read_status(first.product_dir)["partial_complete"] is True
    second = run_product_pipeline(tmp_path, "openai_image", "openai_image", detail_count=9, fail_indexes=set())
    assert second.generated_indexes == [2]
    assert read_status(second.product_dir)["detail_generation_complete"] is True


def test_screen_count_mismatch_makes_no_paid_image_calls(tmp_path):
    api = Mock()
    run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=4,
        prompt_screen_count=3,
        openai_api=api,
    )
    api.generate_edit.assert_not_called()
```

Extend the concrete helper signature to `run_product_pipeline(tmp_path, support_provider, detail_provider, detail_count=2, prompt_screen_count=None, fail_indexes=frozenset(), lovart=None, openai_api=None)`. Its fake OpenAI provider raises only for indexes in `fail_indexes`, records generated indexes, and leaves valid PNGs/checkpoints for all successful indexes so a second call against the same `tmp_path` exercises real resume logic.

- [ ] **Step 2: Run final-stage tests and verify RED**

Run: `uv run pytest tests/test_image_provider_routing.py tests/test_medium_priority.py tests/test_failed_retry_queue.py -q`

Expected: failures show the final stage still always calls `lovart.create_and_generate` and success still depends on Lovart state.

- [ ] **Step 3: Route the final stage and update status/results**

```python
target_count = ensure_detail_page_count_snapshot(product_dir, routing.detail_page_count)
screens = split_detail_screens(prompt, target_count)
detail_provider = registry.get(routing.detail_provider)
detail_result = detail_provider.generate_detail_set(DetailSetRequest(
    product_id=product.id,
    product_dir=product_dir,
    screens=tuple(screens),
    image_paths=tuple(final_reference_images),
    image_size=getattr(product, "image_size", ""),
    target_count=target_count,
))
```

For GPT Image, success requires exactly `target_count` validated files. For partial failure, write a failed/partial summary row with completed count and failed indexes but preserve all paths. For Lovart, preserve existing pending-confirmation, timeout, project URL, and artifact behavior. Rename the provider-neutral composed prompt file to `detail_prompt.txt`, while continuing to write `lovart_prompt.txt` only for Lovart runs.

- [ ] **Step 4: Run final routing and complete processor regressions and verify GREEN**

Run: `uv run pytest tests/test_image_provider_routing.py tests/test_medium_priority.py tests/test_failed_retry_queue.py -q`

Expected: all selected tests pass; non-default counts and resume snapshots behave deterministically.

- [ ] **Step 5: Commit final-set routing and partial recovery**

```powershell
git add main.py utils.py tests/test_image_provider_routing.py tests/test_medium_priority.py tests/test_failed_retry_queue.py
git commit -m "feat: generate resumable detail sets with GPT Image"
```

---

### Task 7: Persist GPT Image settings and credentials transactionally

**Files:**
- Modify: `webui.py:354-520, 750-806, 971-1055`
- Modify: `config.example.yaml`
- Modify: `.env.example`
- Modify: `tests/test_webui_model_settings.py:78-92, 443-690, 811-980`

**Interfaces:**
- Consumes: `normalize_openai_image_base_url` and existing config/.env snapshot helpers.
- Produces: `persist_openai_image_settings(config, base_url, model, resolution, support_provider, detail_provider)`, extended `save_env(gemini_key, nvidia_key, lovart_access, lovart_secret, openai_image_key=None, clear_openai_image_key=False, env_path=".env")`, and extended `save_api_settings(gemini_key, nvidia_key, lovart_access, lovart_secret, openai_image_key, gemini_base_url, gemini_model, nvidia_base_url, nvidia_model, openai_image_base_url, openai_image_model, openai_image_resolution, support_provider, detail_provider, config_path="config.yaml", env_path=".env")`.

- [ ] **Step 1: Write failing defaults, preserve-key, clear-key, and rollback tests**

```python
def test_example_defaults_include_image_routing_and_openai_image():
    config = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))
    assert config["image_generation"] == {"support_provider": "lovart", "detail_provider": "lovart"}
    assert config["openai_image"]["base_url"] == "https://hapiopen.cc/v1"
    assert config["openai_image"]["model"] == "gpt-image-2"


def test_blank_openai_image_key_preserves_existing_value(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_IMAGE_API_KEY=existing\n", encoding="utf-8")
    save_env("", "", "", "", openai_image_key="", env_path=env_path)
    assert "OPENAI_IMAGE_API_KEY=existing" in env_path.read_text(encoding="utf-8")


def test_transaction_rolls_back_config_and_openai_image_key_on_second_replace_failure(tmp_path, monkeypatch):
    config_path, env_path = tmp_path / "config.yaml", tmp_path / ".env"
    config_path.write_text("gemini_api:\n  base_url: https://gemini.test/v1beta\n  model: gemini-test\nnvidia_api:\n  base_url: https://nvidia.test/v1\n  model: nvidia-test\n", encoding="utf-8")
    env_path.write_text("OPENAI_IMAGE_API_KEY=old-key\n", encoding="utf-8")
    original_config, original_env = config_path.read_bytes(), env_path.read_bytes()
    real_replace, call_count = os.replace, 0

    def fail_second_replace(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected env replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(webui.os, "replace", fail_second_replace)
    status = save_api_settings(
        "", "", "", "", "new-key",
        "https://gemini.test/v1beta", "gemini-test",
        "https://nvidia.test/v1", "nvidia-test",
        "https://hapiopen.cc/v1", "gpt-image-2", "1K",
        "openai_image", "openai_image",
        config_path=config_path, env_path=env_path,
    )
    assert "失败" in status
    assert config_path.read_bytes() == original_config
    assert env_path.read_bytes() == original_env
```

- [ ] **Step 2: Run settings tests and verify RED**

Run: `uv run pytest tests/test_webui_model_settings.py -q`

Expected: assertions fail because new defaults, signature parameters, and persistence helpers do not exist.

- [ ] **Step 3: Extend defaults and the two-file transaction**

```python
def persist_openai_image_settings(config, base_url, model, resolution, support_provider, detail_provider):
    updated = deepcopy(config)
    updated["openai_image"] = {
        **updated.get("openai_image", {}),
        "base_url": normalize_openai_image_base_url(base_url),
        "model": str(model or "gpt-image-2").strip(),
        "resolution": normalize_resolution(resolution),
    }
    updated["image_generation"] = {
        "support_provider": normalize_image_provider(support_provider),
        "detail_provider": normalize_image_provider(detail_provider),
    }
    return updated
```

Extend `save_env()` without changing unknown lines. Empty `openai_image_key` preserves the current value; `clear_openai_image_key=True` removes it. Pass the new value through `_save_config_and_env_transaction`, `save_api_settings`, and `run_process` snapshots so any replace failure restores both files.

- [ ] **Step 4: Run all WebUI persistence tests and verify GREEN**

Run: `uv run pytest tests/test_webui_model_settings.py -q`

Expected: all tests pass, including prior Gemini/NVIDIA/Lovart atomic-save cases.

- [ ] **Step 5: Commit settings and credential persistence**

```powershell
git add webui.py config.example.yaml .env.example tests/test_webui_model_settings.py
git commit -m "feat: persist GPT Image settings securely"
```

---

### Task 8: Add WebUI controls, paid compatibility test, and subprocess arguments

**Files:**
- Modify: `webui.py:610-760, 1011-1130, 1717-2199, 2351-2370`
- Modify: `main.py:81-132, 1160-1240`
- Modify: `tests/test_webui_model_settings.py:348-566, 700-810`
- Modify: `tests/test_image_provider_routing.py`

**Interfaces:**
- Consumes: `OpenAIImageAPIConfig`, `OpenAIImageAPI.test_edit`, routing settings, and existing Gradio dependency registration.
- Produces: `test_openai_image_edit(api_key, base_url, model, resolution) -> str`, provider selectors, explicit “清除已保存 GPT Image 密钥” control, per-screen progress rendering, saved defaults, and CLI flags `--support-provider` / `--detail-provider`.

- [ ] **Step 1: Write failing WebUI dependency, explicit-charge, and CLI tests**

```python
@patch("webui.OpenAIImageAPI")
def test_paid_image_test_runs_only_when_handler_is_explicitly_called(api_cls):
    api_cls.return_value.test_edit.return_value.local_path = "test-output.png"
    message = test_openai_image_edit("test-key", "https://hapiopen.cc", "gpt-image-2", "1K")
    assert "测试成功" in message
    api_cls.return_value.test_edit.assert_called_once()


@patch("webui.load_config", return_value={})
def test_ui_exposes_two_independent_provider_selectors(_load_config):
    demo = build_ui()
    labels = {item["id"]: item.get("props", {}).get("label") for item in demo.config["components"]}
    dependencies = {item["api_name"]: item for item in demo.config["dependencies"]}
    run_labels = {labels[item] for item in dependencies["run_process"]["inputs"]}
    assert "白底图和场景图来源" in run_labels
    assert "最终套图来源" in run_labels
    assert "清除已保存 GPT Image 密钥" in run_labels
    paid_test = dependencies["test_openai_image_edit"]
    assert {labels[item] for item in paid_test["inputs"]} >= {"GPT Image API 地址", "GPT Image 模型", "GPT Image 分辨率"}
    button_values = [str(item.get("props", {}).get("value", "")) for item in demo.config["components"] if item["type"] == "button"]
    assert any("可能产生一次图片费用" in value for value in button_values)


def test_cli_provider_overrides_are_forwarded():
    args = parse_args(["--support-provider", "openai_image", "--detail-provider", "lovart"])
    assert args.support_provider == "openai_image"
    assert args.detail_provider == "lovart"
```

- [ ] **Step 2: Run UI and CLI tests and verify RED**

Run: `uv run pytest tests/test_webui_model_settings.py tests/test_image_provider_routing.py -q`

Expected: missing handler, controls, dependencies, and CLI options fail.

- [ ] **Step 3: Add the settings section, run selectors, and paid-test handler**

```python
def test_openai_image_edit(api_key, base_url, model, resolution):
    config = {"openai_image": {"base_url": base_url, "model": model, "resolution": resolution}}
    client = OpenAIImageAPI(OpenAIImageAPIConfig.from_config(config, api_key=api_key), logger=LOGGER)
    result = client.test_edit(Path("output") / ".api-tests")
    return f"✅ GPT Image 图生图测试成功：{Path(result.local_path).name}。本次测试可能已产生一次图片费用。"
```

Use password input for the key, an “已保存/未保存” indicator instead of echoing the stored value, an unchecked-by-default clear-key checkbox, Base URL/model/resolution controls, and a clearly labeled paid-test button. Add the two provider selectors to the run tab and persist them before subprocess launch. Append `--support-provider` and `--detail-provider` to `main.py` command arguments. Validate only credentials required by selected routes. Have `main.py` emit `[UI_DETAIL_PROGRESS] {"current": n, "target": total, "completed": done, "failed": [indexes]}` after every checkpoint, and update the WebUI log/status card without including prompts or request bodies.

- [ ] **Step 4: Run UI, CLI, and process-launch tests and verify GREEN**

Run: `uv run pytest tests/test_webui_model_settings.py tests/test_image_provider_routing.py tests/test_gemini_browser_session.py -q`

Expected: all selected tests pass; creating the UI does not call the image API.

- [ ] **Step 5: Commit the WebUI and CLI surface**

```powershell
git add webui.py main.py tests/test_webui_model_settings.py tests/test_image_provider_routing.py
git commit -m "feat: expose GPT Image mode in WebUI and CLI"
```

---

### Task 9: Documentation, migration regression, full verification, and packaging smoke

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_OVERVIEW.md`
- Modify: `config.example.yaml`
- Modify: `.env.example`
- Modify: `tests/test_high_priority.py`
- Modify: `tests/test_medium_priority.py`
- Modify: `Lovart_Auto.spec` and/or `Lovart自动化助手.spec` only if the import smoke proves hidden imports are required.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: user-facing setup instructions, migration notes, final regression evidence, and a buildable application.

- [ ] **Step 1: Add final migration and no-secret regression tests**

```python
def test_legacy_lovart_status_reuses_support_images_in_provider_neutral_pipeline(tmp_path):
    white = write_valid_png(tmp_path / "lovart_steps" / "white_bg" / "old-white.png")
    scene = write_valid_png(tmp_path / "lovart_steps" / "scene" / "old-scene.png")
    legacy = {
        "lovart_white_bg_local_path": white,
        "lovart_scene_local_path": scene,
        "lovart_final_images": [white, scene],
    }
    assert _find_support_image(tmp_path, legacy, "white_bg", 0) == white
    assert _find_support_image(tmp_path, legacy, "scene", 1) == scene


def test_repository_examples_contain_no_real_api_keys():
    for path in [Path("config.example.yaml"), Path(".env.example"), Path("README.md"), Path("PROJECT_OVERVIEW.md")]:
        text = path.read_text(encoding="utf-8")
        assert "sk-" not in text
        assert "OPENAI_IMAGE_API_KEY=your_openai_image_api_key" in Path(".env.example").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run migration and documentation tests and verify RED before docs updates**

Run: `uv run pytest tests/test_high_priority.py tests/test_medium_priority.py -q`

Expected: the new example/documentation assertion fails until examples and migration helpers are complete.

- [ ] **Step 3: Update documentation and packaging inputs**

Document:

- HAPI/OpenAI-compatible Base URL must end in `/v1`;
- `OPENAI_IMAGE_API_KEY` belongs in local `.env`;
- default model `gpt-image-2` and 1K/2K/4K selection;
- independent support/detail provider choices;
- detail count follows settings and is snapshotted per task;
- paid test behavior and standard `/images/edits` requirement;
- partial completion and resume semantics;
- no automatic provider fallback.

Run the frozen-import smoke before touching spec files. Only add `image_generation`, `image_providers`, or `openai_image_api` to hidden imports if PyInstaller fails to discover them.

- [ ] **Step 4: Run full verification**

Run:

```powershell
uv run pytest -q
uv run python -m compileall -q main.py webui.py image_generation.py image_providers.py openai_image_api.py
git diff --check
uv run python main.py --dry-run --limit 1 --support-provider openai_image --detail-provider openai_image
```

Expected:

- pytest reports zero failures;
- compileall exits 0;
- diff check prints nothing and exits 0;
- dry-run completes without reading GPT Image or Lovart credentials and without network calls.

- [ ] **Step 5: Commit documentation and final regression coverage**

```powershell
git add README.md PROJECT_OVERVIEW.md config.example.yaml .env.example tests/test_high_priority.py tests/test_medium_priority.py Lovart_Auto.spec Lovart自动化助手.spec
git diff --cached --name-only
git commit -m "docs: document GPT Image generation mode"
```

Before committing, unstage either spec file if it did not need a real change. The staged-name list must contain only files intentionally changed in this task.

---

## Final Review Checklist

- [ ] Each provider combination has a passing route test.
- [ ] Non-default detail counts (including 1 and a value other than 12) have passing generation and resume tests.
- [ ] Existing Lovart-only tests and old status recovery pass unchanged.
- [ ] No automatic provider fallback exists in code or UI copy.
- [ ] No real key, Authorization value, or raw multipart body appears in tracked files or logs.
- [ ] Real paid image generation is unreachable during UI construction, config save, dry-run, or automated tests.
- [ ] Final `git status --short` is reviewed and unrelated user files remain untouched.
