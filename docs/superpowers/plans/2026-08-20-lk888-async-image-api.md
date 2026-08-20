# LK888 Async Image API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy OpenAI Images/HAPI transport with the configurable LK888 asynchronous media-task protocol, persist paid task IDs before polling, resume support/detail tasks without duplicate POSTs, and publish the verified change as v1.3.21.

**Architecture:** Keep the provider-neutral pipeline and `openai_image` routing name, but make `openai_image_api.py` speak one JSON task protocol only. The transport owns request encoding, task submission, polling, result downloading, and protocol validation; `image_providers.py` owns durable support/detail task checkpoints; `main.py` owns product-level stage/result semantics; `webui.py` owns user-visible settings and live task progress. Base URL remains configurable, while endpoint paths and response semantics are fixed to the approved protocol.

**Tech Stack:** Python 3.11+, stdlib `urllib`/`http.client`/`ssl`/`socket`, Pillow, Gradio, pytest/unittest, PyInstaller, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-08-20-lk888-async-image-api-design.md`

## Global Constraints

- Use test-driven development for every production change: write the named failing test, run it and capture RED, implement the smallest behavior, then run GREEN.
- A paid `POST /v1/media/generate` is never automatically retried or repeated after an ambiguous network outcome.
- Once a valid `task_id` exists, all recovery uses `GET /v1/media/status?task_id=<saved-task-id>`; resuming a running task must make zero create POSTs.
- Base URL may end with `/v1` or omit it. Strip exactly one trailing `/v1` before appending protocol endpoints.
- API keys stay in `OPENAI_IMAGE_API_KEY`; never place them in `config.yaml`, status files, logs, fingerprints, exceptions, reports, or test snapshots.
- Preserve the existing SSRF-safe result downloader and its DNS pinning/TLS/peer/MIME/decode guarantees.
- Do not change Gemini prompt generation, Lovart behavior, dynamic detail-page count, or the one-prompt-per-detail-screen contract.
- Do not stage unrelated build outputs, `.planning/`, browser artifacts, prior release directories, or user files.
- Use the repository's working pytest invocation on Windows, for example `$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py -q`.

## File and Interface Map

- Modify `openai_image_api.py`: protocol endpoint normalization, Data URL encoding and limits, task submission, task polling, task snapshots, safe final download.
- Modify `image_providers.py`: durable support/detail task checkpoint schema, resume identity checks, task-progress persistence, still-running results.
- Modify `main.py`: pass resume policy into support generation, expose running task stages, avoid failed-queue resubmission for a live task, preserve dynamic detail counts.
- Modify `failed_retry.py`: classify ambiguous create submission and locally timed-out live tasks as non-automatic-retry outcomes.
- Modify `webui.py`: remove HAPI defaults/text, retain editable Base URL/model/resolution/merge controls, render async submit/poll progress in paid test and product cards.
- Modify `config.example.yaml`, `.env.example`, `README.md`: document only the async media-task protocol and migration behavior.
- Modify `version.py`, `version.json`: publish version v1.3.21 after verification.
- Extend `tests/test_openai_image_api.py`: transport, limits, state machine, endpoint, and SSRF preservation.
- Extend `tests/test_image_providers.py`: support/detail checkpoint persistence and resume.
- Extend `tests/test_image_provider_routing.py`: pipeline stage/running/retry behavior.
- Extend `tests/test_failed_retry_queue.py`: no automatic resubmit for ambiguous/live paid tasks.
- Extend `tests/test_webui_model_settings.py` and create `tests/test_webui_async_tasks.py`: settings, wiring, and progress UI.
- Extend `tests/test_high_priority.py`: documentation/version/release invariants.

The transport interfaces produced by this plan are:

```python
@dataclass(frozen=True, slots=True)
class ImageTaskSnapshot:
    task_id: str
    state: str
    is_final: bool
    task_created_at: float
    progress: str = ""
    status: str = ""
    status_group: str = ""
    result_url: str = ""
    result_type: str = ""
    error: str = ""
    cost: object = None


@dataclass(frozen=True)
class GeneratedImage:
    local_path: str
    model: str
    task: ImageTaskSnapshot


class ImageTaskStillRunning(OpenAIImageAPIError):
    task: ImageTaskSnapshot


```

`OpenAIImageAPI.generate_edit` receives `prompt`, `image_paths`, `output_path`, `image_size`, `status_callback`, `task_callback`, and `resume_task`, and returns `GeneratedImage`.

`task_callback` must be called synchronously immediately after a successful create response is parsed and after every valid status response, before sleeping or downloading.

---

### Task 1: Normalize the single protocol and encode validated references

**Files:**
- Modify: `openai_image_api.py`
- Modify: `tests/test_openai_image_api.py`

**Interfaces:**
- Produce `_media_endpoint(base_url, resource) -> str`.
- Produce `_encode_reference_images(paths, merge) -> list[str]`.
- Produce `_build_create_body(model, prompt, size, images) -> bytes`.
- Remove `async_edits` from `OpenAIImageAPIConfig` and stop using `_is_hapi_image_service`, `_hapi_images_endpoint`, and multipart request generation.

- [ ] **Step 1: Add endpoint and JSON-contract tests**

Add parameterized tests proving both user-entered Base URL forms resolve identically:

```python
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
```

Add a create-body test that decodes JSON and asserts this exact shape:

```python
assert payload == {
    "model": "gpt-image-2",
    "prompt": append_aspect_instruction("sell it", "2:3"),
    "params": {
        "images": encoded_images,
        "size": "1024x1536",
        "quality": "auto",
        "n": 1,
    },
}
```

Also assert `Content-Type: application/json`, Bearer authorization, and absence of `/images/edits`, multipart boundaries, `image`, and `image[]`.

- [ ] **Step 2: Add reference-format and preflight-limit tests**

Create real PNG/JPEG/WebP fixtures and assert their prefixes are respectively `data:image/png;base64,`, `data:image/jpeg;base64,`, and `data:image/webp;base64,` regardless of filename extension. Add boundary tests for:

- 14 images accepted; 15 rejected with zero network calls.
- one decoded image at the configured 10MB limit accepted; one byte over rejected.
- total decoded bytes at 30MB accepted; one byte over rejected.
- UTF-8 request body at 50MB accepted; one byte over rejected.
- merge disabled never silently combines images.
- merge enabled produces one Data URL using the existing local reference-sheet builder.
- Pillow-invalid, truncated, or unsupported image formats fail before network access.

Use monkeypatched limit constants for compact fixtures rather than allocating 50MB in every test.

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py -q
```

Expected: new endpoint/JSON/Data URL/limit tests fail because the implementation still sends multipart `/images/edits` and contains host-specific HAPI behavior.

- [ ] **Step 4: Implement endpoint, Data URL, limits, and size mapping**

Add constants and helpers:

```python
MAX_REFERENCE_IMAGES = 14
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES = 30 * 1024 * 1024
MAX_CREATE_BODY_BYTES = 50 * 1024 * 1024

def _protocol_base_url(base_url: str) -> str:
    normalized = normalize_openai_image_base_url(base_url)
    return normalized[:-3] if normalized.lower().endswith("/v1") else normalized

def _media_endpoint(base_url: str, resource: str, *, task_id: str = "") -> str:
    base = _protocol_base_url(base_url)
    if resource == "generate":
        return f"{base}/v1/media/generate"
    if resource == "status" and task_id:
        return f"{base}/v1/media/status?{urlencode({'task_id': task_id})}"
    raise ValueError("invalid media endpoint")
```

Move the existing pixel-size mapping out of hostname detection: this transport always sends a documented exact pixel size selected from `resolution` plus the spreadsheet ratio. Decode each source with Pillow, call `image.verify()`, reopen and `image.load()`, map actual `image.format` through `PNG/JPEG/WEBP`, then Base64-encode the original validated bytes. Serialize with compact separators and reject all size/count limits before constructing a request.

- [ ] **Step 5: Remove config host defaults and run GREEN**

Delete the `async_edits` field/constructor argument, `_is_hapi_image_service`, `_is_lk888_image_service`, `_hapi_images_endpoint`, and the HAPI-driven default merge rule. Change the config-level Base URL default from the live OpenAI endpoint to an empty string, so a selected GPT Image route requires an explicit user value. `merge_reference_images` defaults to `False` for every Base URL.

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py tests/test_webui_model_settings.py tests/test_image_providers.py -q
```

Expected: all selected tests pass after updating old helper construction to the new config shape.

- [ ] **Step 6: Commit**

```powershell
git add openai_image_api.py tests/test_openai_image_api.py tests/test_webui_model_settings.py tests/test_image_providers.py
git commit -m "refactor: encode GPT Image async task requests"
```

---

### Task 2: Implement submit-once async polling and safe result retrieval

**Files:**
- Modify: `openai_image_api.py`
- Modify: `tests/test_openai_image_api.py`

**Interfaces:**
- Produce `ImageTaskSnapshot`, `ImageTaskStillRunning`, and the extended `GeneratedImage`.
- Extend `OpenAIImageAPI.generate_edit` with keyword parameters `task_callback: Callable[[ImageTaskSnapshot], None] | None = None` and `resume_task: ImageTaskSnapshot | None = None`.
- Preserve `validate_remote_image_url`, `_download_resolved_image`, and `atomic_save_validated_image` behavior.

- [ ] **Step 1: Add task submission and response parsing tests**

Cover create responses with `task_id` at the root and inside `data`, accepting string or integer IDs and normalizing to a bounded safe string. Assert:

- create POST uses `max_attempts=1` and is observed exactly once.
- an HTTP timeout after submission raises `OpenAIImageAPIError(code="ambiguous_submission", retryable=False)` and does not poll or POST again.
- HTTP 4xx, invalid JSON, missing/unsafe task ID do not trigger a second POST.
- `task_callback` receives a `pending` snapshot before the first status GET.
- errors and reprs redact the API key and never serialize it into the snapshot.

- [ ] **Step 2: Add polling state-machine tests**

Use a fake monotonic clock and injected sleep function. Test:

```python
running = {"task_id": "t-1", "state": "running", "is_final": False, "progress": "45%", "status": "处理中"}
success = {"task_id": "t-1", "state": "success", "is_final": True, "result_url": "https://cdn.example/result.png", "result_type": "image"}
failed = {"task_id": "t-1", "state": "failed", "is_final": True, "error": "provider rejected task", "cost": 0}
```

Assertions:

- `pending/running + is_final=false` continues.
- `success + is_final=true + result_url` downloads and atomically saves.
- `failed + is_final=true` raises a terminal, non-automatic-retry task error containing sanitized provider text.
- invalid combinations, mismatched task IDs, missing `state`/`is_final`, or success without `result_url` raise `invalid_response` without POST repetition.
- polling sleeps at 5 seconds through elapsed 120 seconds and 10 seconds after it.
- a 600-second local deadline raises `ImageTaskStillRunning` carrying the last snapshot and invokes no new POST.
- GET timeouts, 429, and 5xx use existing retry delays against the same URL/task ID; permanent GET errors do not retry.
- status callback includes progress, display status, elapsed time, and only a short task-ID suffix.

- [ ] **Step 3: Add resume and redownload tests**

Pass `resume_task=ImageTaskSnapshot(task_id="t-1", state="running", is_final=False, task_created_at=1787200000.0)` and assert:

- a running snapshot starts directly with GET and create POST count stays zero.
- a final success snapshot with `result_url` downloads directly with both POST and status GET counts zero.
- a successfully downloaded corrupt/missing local file can be restored from the saved URL.
- a final failed snapshot is terminal and never resubmitted inside `generate_edit`.

- [ ] **Step 4: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py -q
```

Expected: task snapshot/resume tests fail because the current code contains HAPI async plus synchronous fallback and no durable resume input.

- [ ] **Step 5: Implement the state machine**

Implement these internal phases without a fallback branch:

```python
if resume_task is None:
    task = self._submit_task_once(create_body)
    _notify_task(task_callback, task)
else:
    task = _validate_resume_task(resume_task)

if task.is_final and task.state == "success":
    return self._download_task_result(task, output_path)

task = self._poll_task(task, status_callback, task_callback)
return self._download_task_result(task, output_path)
```

Use `time.monotonic()` for this invocation's 600-second wait; preserve `task_created_at` only for display/persistence. Do not let a stale creation timestamp eliminate the next resume's local wait window. Call `task_callback` before each sleep and before final download. Keep URL download retries separate from create/poll request logic.

Delete `_request_hapi_async_edit`, `_request_hapi_sync_edit`, `_request_sync_edit`, the daemon sync worker, ambiguous sync-specific text, and HAPI fallback state. Update `test_edit()` to use the same async path.

- [ ] **Step 6: Run transport and security GREEN**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py tests/test_network_retry.py -q
```

Expected: all task tests and all existing SSRF/TLS/download tests pass.

- [ ] **Step 7: Commit**

```powershell
git add openai_image_api.py tests/test_openai_image_api.py
git commit -m "feat: poll resumable GPT Image media tasks"
```

---

### Task 3: Persist support and detail task identities before polling

**Files:**
- Modify: `image_providers.py`
- Modify: `tests/test_image_providers.py`
- Modify: `tests/image_test_helpers.py`

**Interfaces:**
- Add `input_fingerprint: str = ""` and `resume: bool = True` to `SupportImageRequest`.
- Add `still_running: bool = False` and `task_id_suffix: str = ""` to `ImageProviderResult`.
- Extend `record_detail_checkpoint` with task snapshot fields.
- Produce `record_support_task_checkpoint(product_dir, step_name, checkpoint)` and `read_support_task_checkpoint(product_dir, step_name)` using `support_task_checkpoints`.

- [ ] **Step 1: Add support checkpoint tests**

Test that the callback from the transport immediately writes this provider-owned schema:

```json
{
  "support_task_checkpoints": {
    "white_bg": {
      "state": "running",
      "task_id": "task-123",
      "task_created_at": 1787200000.0,
      "progress": "45%",
      "status": "处理中",
      "status_group": "进行中",
      "result_url": "",
      "error": "",
      "input_fingerprint": "sha256:8f4c1d4a",
      "prompt_hash": "sha256:19f02e61",
      "model": "gpt-image-2",
      "size": "1024x1536",
      "base_url": "https://api.lk888.ai",
      "merge_reference_images": false
    }
  }
}
```

Assert a matching running checkpoint is reconstructed as `resume_task` and calls `api.generate_edit` once with that resume object; the fake transport confirms it performed zero create POSTs. Assert URL/model/size/merge/prompt/upstream fingerprint changes discard the old task and remove a stale canonical support file before a new request.

- [ ] **Step 2: Add detail checkpoint tests**

Extend current per-screen tests to assert:

- the running checkpoint contains the valid task ID before any poll callback can fail.
- restarting the process resumes the exact missing screen/task and does not submit a new one.
- completed earlier screens remain untouched.
- a success snapshot persists `result_url`; deleting/corrupting `NN.png` redownloads it with zero create POSTs.
- final failed tasks remain failed; the next explicit product retry may clear only that screen's task identity and create one replacement task.
- legacy `running` checkpoints without `task_id` are invalidated once.
- `resume=False` clears task identity and canonical output before submitting once.

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_image_providers.py -q
```

Expected: failures show task snapshots are not represented in support/detail checkpoints and the API cannot receive resume state.

- [ ] **Step 4: Implement checkpoint serialization and identity matching**

Keep transport types in `openai_image_api.py`; add explicit conversion helpers in `image_providers.py`:

```python
def _task_snapshot_from_checkpoint(checkpoint: Mapping[str, object]) -> ImageTaskSnapshot | None:
    task_id = str(checkpoint.get("task_id") or "").strip()
    if not task_id:
        return None
    return ImageTaskSnapshot(
        task_id=task_id,
        state=str(checkpoint.get("state") or "running"),
        is_final=bool(checkpoint.get("is_final", False)),
        task_created_at=float(checkpoint.get("task_created_at") or 0.0),
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
        "cost": task.cost,
    }
```

The task callback must merge fields atomically with existing `input_fingerprint`, `prompt_hash`, request settings, attempts, and local path. Do not persist auth headers or API keys. A local wait timeout must leave state `running`, not overwrite it with `failed`.

- [ ] **Step 5: Implement support/detail adapter behavior**

For each support step and detail screen:

1. Compute/receive the request identity.
2. Load an identity-matching checkpoint when resume is enabled.
3. Remove stale canonical output before a new paid task.
4. Mark the logical stage running.
5. Call `api.generate_edit(prompt=request.prompt, image_paths=request.image_paths, output_path=output_path, image_size=request.image_size, status_callback=request.status_callback, task_callback=persist, resume_task=saved_task)`.
6. On `ImageTaskStillRunning`, return `ImageProviderResult(still_running=True, succeeded=False)` without changing the checkpoint to failed.
7. On final success, persist `done`, result URL, and validated local path.
8. On final provider failure, persist `failed` with sanitized error/cost.

- [ ] **Step 6: Run GREEN and compatibility tests**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_image_providers.py tests/test_image_generation.py tests/test_lovart_unlimited.py -q
```

Expected: task checkpoint tests pass and Lovart adapter behavior remains unchanged.

- [ ] **Step 7: Commit**

```powershell
git add image_providers.py tests/test_image_providers.py tests/image_test_helpers.py
git commit -m "feat: persist GPT Image task checkpoints"
```

---

### Task 4: Integrate live tasks with pipeline status and failed-queue semantics

**Files:**
- Modify: `main.py`
- Modify: `failed_retry.py`
- Modify: `tests/test_image_provider_routing.py`
- Modify: `tests/test_failed_retry_queue.py`

**Interfaces:**
- Pass support `resume` and a deterministic support input fingerprint into `SupportImageRequest`.
- Preserve `detail_page_count_snapshot` and existing prompt hashes while adding task-aware running state.
- Add product status fields `openai_image_still_running`, `openai_image_active_stage`, and sanitized `openai_image_task_suffix`.

- [ ] **Step 1: Add pipeline resume tests**

Use recording providers to cover both support steps and details:

- white-background task times out locally: product shows still running at white background; scene/prompt/details are not called.
- next run resumes the white task with the same ID, then proceeds to scene.
- scene task times out locally after white success: white is reused, only scene resumes.
- detail screen 4 times out after screens 1–3 succeed: the next run reuses support and screens 1–3, resumes screen 4, then continues dynamically through the configured target count.
- target count comes from `detail_page_count_snapshot`, never a fixed 12.
- task progress events update the active product card while the process is running.
- a live task is not reported as ordinary failed and is not moved to the failed retry queue.

- [ ] **Step 2: Add failure/retry tests**

Assert these classifications:

```python
assert classify_failure(OpenAIImageAPIError("ambiguous_submission", "GPT Image task submission result is unknown.")) == permanent_stop
assert classify_failure(ImageTaskStillRunning(last_task)) == still_running
assert classify_failure(final_provider_failure) == retryable_product_failure
```

An ambiguous POST outcome must not participate in finite or infinite automated retries. A valid live task may be resumed only by the normal resume path. A final failed task may enter the existing product retry policy, which creates at most one replacement task per retry round.

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_image_provider_routing.py tests/test_failed_retry_queue.py -q
```

Expected: new still-running stage tests fail because `_SupportImageGenerationError` and detail failures currently collapse task timeouts into ordinary failures.

- [ ] **Step 4: Implement support fingerprint and resume propagation**

Compute a canonical SHA-256 identity from provider execution settings, step name, exact final prompt, image size, merge flag, and content hashes of sorted reference roles. Pass `resume=not args.no_resume` through `_generate_support_images` to `SupportImageRequest`. Do not include path spelling or API key in the hash.

- [ ] **Step 5: Implement product state transitions**

On a still-running result, update status atomically:

```python
update_status(
    product_dir,
    "openai_image_task_still_running",
    openai_image_still_running=True,
    openai_image_active_stage=stage,
    openai_image_task_suffix=suffix,
    failed=False,
    needs_manual_action=False,
    reason="",
)
```

Stop processing the current product without marking completion and continue to the next queued product. On resume/success/final failure, clear stale still-running fields. Keep the existing provider-neutral completion, result CSV, and immediate move-to-completed-folder behavior.

- [ ] **Step 6: Implement retry guards and run GREEN**

Teach `failed_retry.py` to recognize the stable error codes rather than matching translated display text. Preserve existing network retry behavior for polling GET and result download; the guard applies to paid-create ambiguity and live task timeout only.

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_image_provider_routing.py tests/test_failed_retry_queue.py tests/test_medium_priority.py tests/test_results.py -q
```

Expected: all pipeline and retry tests pass.

- [ ] **Step 7: Commit**

```powershell
git add main.py failed_retry.py tests/test_image_provider_routing.py tests/test_failed_retry_queue.py
git commit -m "feat: resume live GPT Image tasks in pipeline"
```

---

### Task 5: Update WebUI settings, paid test, and live progress rendering

**Files:**
- Modify: `webui.py`
- Modify: `tests/test_webui_model_settings.py`
- Create: `tests/test_webui_async_tasks.py`

**Interfaces:**
- Keep existing save/run callback parameter order unless adding a value is unavoidable.
- Make merge default always `False`; keep the explicit user checkbox.
- Render task progress from status events without exposing full task IDs.

- [ ] **Step 1: Add settings migration tests**

Assert:

- blank Base URL is rejected only when an OpenAI-image route is selected.
- `https://api.lk888.ai` and `https://api.lk888.ai/v1` both save unchanged as valid user input and produce the same runtime endpoints.
- unknown future-gateway hostnames remain accepted.
- merge defaults to `False` for LK888, HAPI, and every other host; an explicitly saved `True` remains `True`.
- legacy `async_edits` and YAML `api_key` fields are removed on save.
- an empty submitted key preserves `.env`; explicit clear deletes it; UI output never echoes a key.

- [ ] **Step 2: Add paid-test progress tests**

Mock the transport callbacks and assert the paid test successively renders:

- request accepted/uploading;
- task submitted with masked suffix;
- server `progress` and display `status` plus elapsed wait;
- final download/success or sanitized terminal error;
- button disabled during the run and re-enabled on all exits.

The test must verify `client.test_edit()` uses async create/poll, not a sync endpoint.

- [ ] **Step 3: Add product-card status tests**

Feed UI events for white background, scene, and detail screen tasks. Assert the current product changes immediately from pending to the real active stage and includes progress/elapsed/task suffix. A 600-second local timeout must render “任务仍在平台运行，下次将继续查询” rather than “失败” or an indefinitely spinning old stage.

- [ ] **Step 4: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_webui_model_settings.py tests/test_webui_async_tasks.py -q
```

Expected: failures identify HAPI merge defaults/wording and missing async task fields in the UI.

- [ ] **Step 5: Implement settings and UI text**

Change the section heading/description to “GPT Image 异步媒体任务 API”. Explain that compatible gateways must implement create-task, task-status, and result URL semantics. Keep the Base URL as a gray hint only; do not silently fill a provider address into an empty input. Update merge help to “最多可直接上传 14 张参考图；仅在网关限制或体积超限时手动开启合并”.

Remove `default_merge_reference_images()` host inspection and return `False` when the stored value is absent/invalid. Scrub `async_edits` while preserving forward-compatible unknown settings.

- [ ] **Step 6: Implement progress rendering and run GREEN**

Continue using `_emit_ui_status` JSON/log events, but include only non-secret task display data. Make UI polling/readback refresh running product cards without requiring process completion.

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_webui_model_settings.py tests/test_webui_async_tasks.py tests/test_image_provider_routing.py -q
```

Expected: all UI/settings/routing tests pass.

- [ ] **Step 7: Commit**

```powershell
git add webui.py tests/test_webui_model_settings.py tests/test_webui_async_tasks.py
git commit -m "feat: show async GPT Image task progress"
```

---

### Task 6: Remove legacy protocol behavior and document migration

**Files:**
- Modify: `openai_image_api.py`
- Modify: `webui.py`
- Modify: `config.example.yaml`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_openai_image_api.py`
- Modify: `tests/test_high_priority.py`

- [ ] **Step 1: Add legacy-absence regression tests**

Add source/behavior checks proving production code no longer contains or calls:

- `/images/edits`
- `/images/edits/async`
- `/images/tasks/`
- `_request_hapi_*`
- `_is_hapi_image_service`
- `async_edits`
- sync fallback notices

The tests should exercise behavior first; use a small source scan only to prevent an accidental dead fallback from surviving.

- [ ] **Step 2: Add documentation/config tests**

Assert the examples describe:

- configurable Base URL with and without `/v1`.
- fixed `/v1/media/generate` and `/v1/media/status` protocol.
- up to 14 direct Data URL references and the optional merge toggle.
- submit-once billing safety and task-ID resume.
- 5-second then 10-second polling and 600-second local wait window.
- API key stored only in `.env`.
- old sync checkpoints without task IDs are not recoverable and are migrated once.

Assert README/config contain no HAPI-specific recommendation or claim of general OpenAI Images compatibility.

- [ ] **Step 3: Run RED**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py tests/test_high_priority.py -q
```

Expected: legacy-absence and documentation assertions fail until cleanup is complete.

- [ ] **Step 4: Complete cleanup and migration documentation**

Delete unused multipart/HAPI/sync imports, constants, helpers, tests, and text. Keep `encode_multipart` only if another module imports it; verify with `rg` before deletion. Update config defaults to:

```yaml
openai_image:
  base_url: ""
  model: "gpt-image-2"
  resolution: "1K"
  merge_reference_images: false
```

The empty example value is intentional: UI hint/help may mention `https://api.lk888.ai`, but no live endpoint is silently selected.

- [ ] **Step 5: Run focused GREEN and static scans**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py tests/test_high_priority.py tests/test_webui_model_settings.py -q
rg -n "HAPI|hapiopen|/images/edits|async_edits|_request_hapi|_is_hapi" openai_image_api.py webui.py config.example.yaml README.md
```

Expected: tests pass; `rg` returns no matches in the named production/documentation files.

- [ ] **Step 6: Commit**

```powershell
git add openai_image_api.py webui.py config.example.yaml .env.example README.md tests/test_openai_image_api.py tests/test_high_priority.py
git commit -m "docs: migrate GPT Image mode to async media tasks"
```

---

### Task 7: Verify, package, and publish v1.3.21

**Files:**
- Modify: `version.py`
- Modify: `version.json`
- Create: release notes or release manifest files required by the existing packaging workflow
- Verify: `Lovart_Auto.spec`, generated EXE, `update.zip`, Git tag, GitHub Release

- [ ] **Step 1: Bump version metadata with a failing invariant test**

Update/add a test asserting `version.py`, `version.json`, release filenames, and UI version all agree on `1.3.21`. Run it before changing metadata and capture RED.

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_high_priority.py -q
```

- [ ] **Step 2: Set v1.3.21 metadata and run focused regressions**

Update version metadata without modifying existing release directories. Then run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_openai_image_api.py tests/test_image_providers.py tests/test_image_provider_routing.py tests/test_failed_retry_queue.py tests/test_webui_model_settings.py tests/test_webui_async_tasks.py tests/test_high_priority.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run complete verification**

Run:

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest -q
uv run python -m compileall -q main.py webui.py image_generation.py image_providers.py openai_image_api.py failed_retry.py
git diff --check
uv run python main.py --dry-run --limit 1 --support-provider openai_image --detail-provider openai_image
```

Expected: full suite passes, compilation succeeds, no whitespace errors, and dry-run constructs no provider/browser clients and performs no network calls.

- [ ] **Step 4: Request code review and fix only verified findings**

Use `superpowers:requesting-code-review`. Review paid POST uniqueness, task persistence timing, stale checkpoint invalidation, API-key redaction, SSRF downloader preservation, product move-to-completed behavior, dynamic detail count, and HAPI/sync removal. For each accepted finding, add a failing regression first and repeat focused/full verification.

- [ ] **Step 5: Build in a fresh v1.3.21 directory**

Use the same frozen dependencies/entry point as `build_exe.bat`, but direct all output to new versioned paths and do not overwrite v1.3.20 artifacts:

```powershell
uv run --no-sync pyinstaller --noconfirm --onedir --windowed `
  --name "Lovart_Auto" `
  --distpath "dist-v1.3.21" `
  --workpath "build-v1.3.21" `
  --specpath "build-v1.3.21" `
  --add-data "preamble.txt;." `
  --add-data "config.example.yaml;." `
  --add-data ".env.example;." `
  --collect-all gradio --collect-all gradio_client `
  --collect-data playwright --collect-data safehttpx --collect-data groovy `
  --hidden-import PIL --hidden-import PIL.Image `
  --hidden-import uvicorn.loops.auto `
  --hidden-import uvicorn.protocols.http.auto `
  --hidden-import uvicorn.protocols.websockets.auto `
  --hidden-import uvicorn.lifespan.on `
  --collect-data uvicorn app.py
```

Verify the packaged executable:

```powershell
$exe = ".\dist-v1.3.21\Lovart_Auto\Lovart_Auto.exe"
$smoke = Start-Process -FilePath $exe -ArgumentList "--run-main","--help" -Wait -PassThru -WindowStyle Hidden
if ($smoke.ExitCode -ne 0) { throw "Packaged CLI self-test failed" }
```

Create and inspect the OTA archive:

```powershell
Compress-Archive -Path ".\dist-v1.3.21\Lovart_Auto\*" -DestinationPath ".\update-v1.3.21.zip" -CompressionLevel Optimal -Force
$hash = (Get-FileHash ".\update-v1.3.21.zip" -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item ".\update-v1.3.21.zip").Length
```

Require the executable and bundled config assets expected by the updater, copy the verified archive to the release asset name `update.zip`, and write the exact `$hash` and `$size` plus the v1.3.21 release URL to `version.json`. Run `updater.validate_and_extract_update` in a new temporary child directory and assert the extracted `Lovart_Auto.exe` exists.

- [ ] **Step 6: Re-run release metadata tests and commit**

```powershell
$env:PYTHONPATH='.'; uv run --with pytest python -m pytest tests/test_high_priority.py -q
git add version.py version.json
git commit -m "release: prepare v1.3.21"
```

- [ ] **Step 7: Push, tag, and publish only after all evidence is green**

Confirm a clean scoped worktree and that `master`, the tag target, `version.json`, and packaged artifact hashes all refer to the same release commit. Then push the branch/approved merge, create annotated tag `v1.3.21`, push the tag, and publish the same asset shape as v1.3.20:

```powershell
git push origin master
git tag -a v1.3.21 -m "v1.3.21"
git push origin v1.3.21
gh release create v1.3.21 ".\update.zip#update.zip" --verify-tag --title "v1.3.21" --notes-file ".\release-v1.3.21-notes.md"
```

Verify the public release URLs and downloadable asset checksums before reporting completion.

---

## Final Acceptance Checklist

- [ ] Every support/detail create request is JSON `POST /v1/media/generate`, exactly once per new task.
- [ ] Every continuation is `GET /v1/media/status?task_id=<saved-task-id>` using the persisted task ID.
- [ ] Task ID is durably written before first poll/sleep/download.
- [ ] A live task survives restart and resumes with zero new POSTs.
- [ ] Success with missing local output redownloads without a new POST.
- [ ] 14-image and byte/body limits fail locally before billing.
- [ ] Ratio/resolution maps to documented exact pixel size; `n=1`, `quality=auto`.
- [ ] UI shows real active stage, progress, elapsed time, and masked task suffix.
- [ ] 600-second local timeout is “still running”, not ordinary failure or infinite retry.
- [ ] Ambiguous create outcome never enters automated retry.
- [ ] Dynamic detail count remains driven by the saved snapshot, never fixed at 12.
- [ ] Completed products are moved immediately using the existing completion flow.
- [ ] HAPI and synchronous OpenAI Images behavior/text are absent.
- [ ] API key remains `.env`-only and absent from logs/status/fingerprints.
- [ ] SSRF/TLS/result-image security regressions all pass.
- [ ] Full tests, compile, dry-run, packaged EXE smoke, ZIP integrity, tag, release, and version manifest are verified.
