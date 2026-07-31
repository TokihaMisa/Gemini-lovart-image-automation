# Gemini Full Health Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click isolated Gemini browser health check that actively verifies every production-critical control without sending a message or creating normal chat history.

**Architecture:** A new `gemini_health_check.py` module owns check results, the browser lifecycle, temporary image creation, and ordered execution. `app.py` exposes it through a dedicated subprocess command that writes newline-delimited JSON to a status file, while `webui.py` starts the process and streams the checklist into a read-only panel. The runner reuses `GeminiBot` and `gemini_browser_session` behavior so diagnostics exercise the same selectors as production.

**Tech Stack:** Python 3.12-3.14, Playwright sync API, Gradio, Pillow, `unittest`, PyInstaller.

## Global Constraints

- Never insert or send prompt text.
- Upload only a program-generated blank PNG.
- Upload only after temporary chat entry is confirmed.
- Inspect the send button without clicking it.
- Respect the existing Gemini profile owner lock.
- Close the browser context and delete the temporary image on every exit path.
- Do not expose account names, cookies, query strings, raw DOM, screenshots, raw Playwright errors, or local user paths.
- Ignore the regular-chat fallback setting during health checks.

---

## File Structure

- Create `gemini_health_check.py`: result model, JSONL serialization, ordered runner, browser/profile lifecycle, and blank-image management.
- Modify `gemini_bot.py`: add narrowly scoped health-check probes that reuse existing production selectors and never send.
- Modify `app.py`: route `--gemini-health-check --config <path> --status-file <path>` before normal WebUI startup.
- Modify `webui.py`: build the subprocess command, tail the JSONL status file, render the checklist, and wire the button.
- Create `tests/test_gemini_health_check.py`: runner, cleanup, skip behavior, safety, command protocol, and rendering tests.
- Modify `tests/test_medium_priority.py`: regression tests for new `GeminiBot` probe methods.
- Modify `tests/test_webui_model_settings.py`: WebUI component and click-event wiring tests.

---

### Task 1: Health Check Result Model And Ordered Runner

**Files:**
- Create: `gemini_health_check.py`
- Create: `tests/test_gemini_health_check.py`

**Interfaces:**
- Produces: `CheckState(str, Enum)` with `PASS`, `FAIL`, and `SKIP`.
- Produces: `HealthCheckResult(name: str, state: CheckState, message: str)`.
- Produces: `HealthCheckReporter.emit(result: HealthCheckResult) -> None`.
- Produces: `render_health_check(results: Sequence[HealthCheckResult]) -> str`.
- Produces: `run_gemini_health_check(config_path: str | Path, reporter: HealthCheckReporter) -> int`.

- [ ] **Step 1: Write failing result and rendering tests**

```python
class FakeHealthCheckHarness:
    def __init__(self, *, temporary_chat=True, upload_error=None):
        self.temporary_chat = temporary_chat
        self.upload_error = upload_error
        self.upload_calls = 0
        self.send_clicks = 0
        self.context_closed = False
        self.blank_image_exists = True
        self.results = []

    def check_browser(self): return True
    def check_profile_lock(self): return True
    def check_page_ready(self): return True
    def check_editor(self): return True
    def enter_temporary_chat(self): return self.temporary_chat
    def open_mode_menu(self): return True
    def select_flash(self): return True
    def select_extended_thinking(self): return True
    def check_upload_control(self): return True
    def upload_blank_image(self):
        self.upload_calls += 1
        if self.upload_error:
            raise self.upload_error
        return True
    def check_send_button(self): return True
    def close(self):
        self.context_closed = True
        self.blank_image_exists = False


def result_for(results, name):
    return next(item for item in results if item.name == name)


def test_health_check_result_json_is_bounded_and_rendered_in_order():
    results = [
        HealthCheckResult("页面加载", CheckState.PASS, "Gemini 页面已就绪"),
        HealthCheckResult("临时对话", CheckState.FAIL, "未找到临时对话按钮"),
        HealthCheckResult("图片上传", CheckState.SKIP, "前置检查失败"),
    ]
    payload = results[0].to_json()
    assert payload == {
        "name": "页面加载",
        "state": "pass",
        "message": "Gemini 页面已就绪",
    }
    rendered = render_health_check(results)
    assert rendered.index("页面加载") < rendered.index("临时对话") < rendered.index("图片上传")
    assert "✅" in rendered
    assert "❌" in rendered
    assert "⚠️" in rendered
    assert "正常 1 · 异常 1 · 跳过 1" in rendered
```

- [ ] **Step 2: Run the result test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check.GeminiHealthCheckTests.test_health_check_result_json_is_bounded_and_rendered_in_order
```

Expected: import failure because `gemini_health_check.py` does not exist.

- [ ] **Step 3: Implement the result model and renderer**

```python
class CheckState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    state: CheckState
    message: str

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name[:40],
            "state": self.state.value,
            "message": self.message[:240],
        }


def render_health_check(results: Sequence[HealthCheckResult]) -> str:
    icons = {
        CheckState.PASS: "✅",
        CheckState.FAIL: "❌",
        CheckState.SKIP: "⚠️",
    }
    lines = [
        f"{icons[item.state]} **{item.name}**：{item.message}"
        for item in results
    ]
    counts = {state: sum(item.state is state for item in results) for state in CheckState}
    lines.append(
        f"\n正常 {counts[CheckState.PASS]} · "
        f"异常 {counts[CheckState.FAIL]} · 跳过 {counts[CheckState.SKIP]}"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Write failing prerequisite-skip and cleanup tests**

```python
def test_runner_skips_upload_when_temporary_chat_fails():
    harness = FakeHealthCheckHarness(temporary_chat=False)
    results = list(run_checks(harness))
    assert result_for(results, "临时对话").state is CheckState.FAIL
    assert result_for(results, "图片上传").state is CheckState.SKIP
    assert harness.upload_calls == 0
    assert harness.send_clicks == 0


def test_runner_always_closes_context_and_removes_blank_image():
    harness = FakeHealthCheckHarness(upload_error=RuntimeError("private raw error"))
    list(run_checks(harness))
    assert harness.context_closed is True
    assert harness.blank_image_exists is False
    assert "private raw error" not in render_health_check(harness.results)
```

- [ ] **Step 5: Run the runner tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check -k runner
```

Expected: failure because `run_checks` and the runner harness contract are not implemented.

- [ ] **Step 6: Implement ordered checks with explicit dependencies**

Use one internal protocol so tests can inject a harness without launching Chrome:

```python
class HealthCheckHarness(Protocol):
    def check_browser(self) -> bool: ...
    def check_profile_lock(self) -> bool: ...
    def check_page_ready(self) -> bool: ...
    def check_editor(self) -> bool: ...
    def enter_temporary_chat(self) -> bool: ...
    def open_mode_menu(self) -> bool: ...
    def select_flash(self) -> bool: ...
    def select_extended_thinking(self) -> bool: ...
    def check_upload_control(self) -> bool: ...
    def upload_blank_image(self) -> bool: ...
    def check_send_button(self) -> bool: ...
    def close(self) -> None: ...


def run_checks(harness: HealthCheckHarness) -> Iterator[HealthCheckResult]:
    # Emit in the 12-step order from the approved design.
    # Stop mutation after temporary chat fails.
    # Mark dependent checks SKIP rather than attempting them.
    # Call harness.close() in finally.
```

Messages must come from a fixed public-message map; caught exceptions map to
`浏览器启动失败`, `页面加载失败`, `控件不可用`, or `上传失败` without including
`str(exc)`.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check
```

Expected: all Task 1 tests pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add gemini_health_check.py tests/test_gemini_health_check.py
git commit -m "feat: add Gemini health check runner"
```

---

### Task 2: Production Gemini Control Probes

**Files:**
- Modify: `gemini_bot.py`
- Modify: `tests/test_medium_priority.py`
- Modify: `gemini_health_check.py`
- Modify: `tests/test_gemini_health_check.py`

**Interfaces:**
- Consumes: `HealthCheckHarness` and `run_checks` from Task 1.
- Produces: `GeminiBot.health_check_editor() -> bool`.
- Produces: `GeminiBot.health_check_upload_control() -> bool`.
- Produces: `GeminiBot.health_check_send_button() -> bool`.
- Produces: `PlaywrightGeminiHealthCheckHarness`.

- [ ] **Step 1: Write failing non-mutating probe tests**

```python
class ProbeLocator:
    def __init__(self, visible=False, enabled=True):
        self.visible = visible
        self.enabled = enabled
        self.clicks = 0

    def count(self): return 1
    def nth(self, _index): return self
    def is_visible(self, **_kwargs): return self.visible
    def is_enabled(self): return self.enabled
    def click(self, **_kwargs): self.clicks += 1


class ProbePage:
    def __init__(self, selectors):
        self.selectors = selectors

    @property
    def total_clicks(self):
        return sum(locator.clicks for locator in self.selectors.values())

    def locator(self, selector):
        return self.selectors.get(selector, ProbeLocator())


def test_send_button_probe_never_clicks():
    page = ProbePage(
        selectors={
            'button[aria-label*="发送"]': ProbeLocator(visible=True, enabled=True)
        }
    )
    bot = GeminiBot(page, {"gemini": {}}, FakeFormalLogger())
    assert bot.health_check_send_button() is True
    assert page.total_clicks == 0


def test_upload_control_probe_accepts_current_chinese_label_without_clicking():
    page = ProbePage(
        selectors={
            'button[aria-label*="上传和工具"]': ProbeLocator(visible=True)
        }
    )
    bot = GeminiBot(page, {"gemini": {}}, FakeFormalLogger())
    assert bot.health_check_upload_control() is True
    assert page.total_clicks == 0
```

- [ ] **Step 2: Run probe tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_medium_priority.MediumPriorityBehaviorTests.test_send_button_probe_never_clicks tests.test_medium_priority.MediumPriorityBehaviorTests.test_upload_control_probe_accepts_current_chinese_label_without_clicking
```

Expected: `AttributeError` for the new methods.

- [ ] **Step 3: Implement minimal read-only probes**

Use existing selector vocabulary and check only `is_visible()` and
`is_enabled()`:

```python
def health_check_editor(self) -> bool:
    return self._any_visible(
        '[contenteditable="true"][role="textbox"], '
        'rich-textarea [contenteditable="true"], textarea, [role="textbox"]'
    )


def health_check_upload_control(self) -> bool:
    return self._any_visible(
        'button[aria-label*="上传和工具"], '
        'button[aria-label*="上传"], '
        'button[aria-label*="Upload"], '
        'button[aria-label*="Adjuntar"]'
    )


def health_check_send_button(self) -> bool:
    locator = self.page.locator(
        'button[aria-label*="发送"], button[aria-label*="Send"], '
        'button[aria-label*="Enviar"]'
    )
    return any(
        locator.nth(index).is_visible(timeout=500)
        and locator.nth(index).is_enabled()
        for index in range(min(locator.count(), 12))
    )
```

Do not call `click`, `press`, `fill`, `type`, or `_send_message` from any probe.

- [ ] **Step 4: Write failing Playwright harness safety test**

```python
class FakeGeminiBot:
    def __init__(self):
        self.calls = []

    def health_check_editor(self): self.calls.append("editor"); return True
    def _start_temporary_chat(self): self.calls.append("temporary_chat"); return True
    def _open_mode_menu(self): self.calls.append("open_mode"); return True
    def _click_flash_model(self): self.calls.append("flash"); return True
    def _select_extended_thinking_option(self): self.calls.append("extended"); return True
    def health_check_upload_control(self): self.calls.append("upload_control"); return True
    def _upload_images(self, _paths): self.calls.append("upload_blank"); return True
    def health_check_send_button(self): self.calls.append("send_probe"); return True


def test_playwright_harness_uploads_only_after_temporary_chat():
    bot = FakeGeminiBot()
    harness = PlaywrightGeminiHealthCheckHarness.from_test_bot(bot)
    results = list(run_checks(harness))
    assert bot.calls == [
        "editor",
        "temporary_chat",
        "open_mode",
        "flash",
        "extended",
        "upload_control",
        "upload_blank",
        "send_probe",
    ]
    assert "send_message" not in bot.calls
```

- [ ] **Step 5: Run harness test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check.GeminiHealthCheckTests.test_playwright_harness_uploads_only_after_temporary_chat
```

Expected: failure because `PlaywrightGeminiHealthCheckHarness` is missing.

- [ ] **Step 6: Implement the Playwright harness**

`PlaywrightGeminiHealthCheckHarness` must:

1. Load YAML through the existing config loader.
2. Resolve launch options with `build_browser_launch_options`.
3. Acquire `login_runtime_paths(...).owner_lock_path` through
   `acquire_login_helper_owner`.
4. Launch a persistent context with `sync_playwright`.
5. Navigate through `navigate_gemini_with_retry`.
6. Create `GeminiBot(page, config, logger)`.
7. Generate `blank-health-check.png` with Pillow inside
   `tempfile.TemporaryDirectory`.
8. Call existing production methods:
   `_start_temporary_chat`, `_open_mode_menu`, `_click_flash_model`,
   `_select_extended_thinking_option`, and `_upload_images`.
9. Call the new read-only probes for editor, upload control, and send button.
10. Close context, stop Playwright, release the owner lock, and clean the
    temporary directory in reverse ownership order.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_medium_priority tests.test_gemini_health_check
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add gemini_bot.py gemini_health_check.py tests/test_medium_priority.py tests/test_gemini_health_check.py
git commit -m "feat: exercise Gemini controls in health check"
```

---

### Task 3: Isolated Process Protocol

**Files:**
- Modify: `gemini_health_check.py`
- Modify: `app.py`
- Modify: `webui.py`
- Modify: `tests/test_gemini_health_check.py`

**Interfaces:**
- Consumes: `run_gemini_health_check` from Task 1.
- Produces: `write_jsonl_result(path: Path, result: HealthCheckResult) -> None`.
- Produces: `run_health_check_cli(config_path: str, status_file: str) -> int`.
- Produces: `build_gemini_health_check_command(config_path: str | Path, status_file: str | Path, *, executable: str | None = None, frozen: bool | None = None) -> list[str]`.

- [ ] **Step 1: Write failing source and frozen command tests**

```python
def test_health_check_command_uses_source_entrypoint():
    command = build_gemini_health_check_command(
        "config.yaml",
        "runs/check.jsonl",
        executable="python.exe",
        frozen=False,
    )
    assert command == [
        "python.exe",
        str(Path(webui.__file__).with_name("app.py").resolve()),
        "--gemini-health-check",
        "--config",
        str(Path("config.yaml").resolve()),
        "--status-file",
        str(Path("runs/check.jsonl").resolve()),
    ]


def test_health_check_command_uses_packaged_executable():
    command = build_gemini_health_check_command(
        "config.yaml",
        "runs/check.jsonl",
        executable="Lovart_Auto.exe",
        frozen=True,
    )
    assert command[0] == "Lovart_Auto.exe"
    assert command[1] == "--gemini-health-check"
    assert "app.py" not in command
```

- [ ] **Step 2: Run command tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check -k command
```

Expected: import failure for `build_gemini_health_check_command`.

- [ ] **Step 3: Implement command construction**

Mirror `build_login_helper_command`, but always include an absolute status path:

```python
def build_gemini_health_check_command(
    config_path,
    status_file,
    *,
    executable=None,
    frozen=None,
) -> list[str]:
    program = executable or sys.executable
    args = [
        "--gemini-health-check",
        "--config",
        str(Path(config_path).resolve()),
        "--status-file",
        str(Path(status_file).resolve()),
    ]
    if getattr(sys, "frozen", False) if frozen is None else frozen:
        return [program, *args]
    return [program, str(Path(__file__).with_name("app.py").resolve()), *args]
```

- [ ] **Step 4: Write failing JSONL and CLI routing tests**

```python
def test_cli_writes_one_json_object_per_result():
    with tempfile.TemporaryDirectory() as tmp:
        status_file = Path(tmp) / "check.jsonl"
        with patch("gemini_health_check.run_gemini_health_check", return_value=0):
            assert run_health_check_cli("config.yaml", str(status_file)) == 0
        lines = status_file.read_text(encoding="utf-8").splitlines()
        assert all(isinstance(json.loads(line), dict) for line in lines)


def test_app_routes_health_check_before_webui_start():
    with tempfile.TemporaryDirectory() as tmp:
        status_file = Path(tmp) / "status.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--gemini-health-check",
                "--config",
                "missing-config.yaml",
                "--status-file",
                str(status_file),
            ],
            cwd=Path(__file__).resolve().parents[1],
            timeout=15,
        )
        assert result.returncode != 0
        assert status_file.exists()
```

- [ ] **Step 5: Run protocol tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check -k cli
```

Expected: failures because the CLI and `app.py` route do not exist.

- [ ] **Step 6: Implement atomic JSONL status output and app routing**

The status file starts with a `running` record and ends with `complete`:

```json
{"event":"running","results":[]}
{"event":"result","result":{"name":"页面加载","state":"pass","message":"Gemini 页面已就绪"}}
{"event":"complete","exit_code":0}
```

Append each line using one UTF-8 file handle, flush, and `os.fsync` so WebUI
polling never depends on process exit.

In `app.py`, route before login helper and normal startup:

```python
if "--gemini-health-check" in sys.argv:
    from gemini_health_check import run_health_check_cli
    raise SystemExit(run_health_check_cli_from_argv(sys.argv[1:]))
```

- [ ] **Step 7: Run Task 3 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check
```

Expected: all protocol tests pass.

- [ ] **Step 8: Commit Task 3**

```powershell
git add app.py webui.py gemini_health_check.py tests/test_gemini_health_check.py
git commit -m "feat: isolate Gemini health check process"
```

---

### Task 4: WebUI Button And Streaming Checklist

**Files:**
- Modify: `webui.py`
- Modify: `tests/test_webui_model_settings.py`
- Modify: `tests/test_gemini_health_check.py`

**Interfaces:**
- Consumes: `build_gemini_health_check_command` and JSONL records from Task 3.
- Produces: `ui_run_gemini_health_check(config_path: str | Path = "config.yaml") -> Iterator[str]`.

- [ ] **Step 1: Write failing streaming renderer tests**

```python
class FakeHealthCheckProcess:
    def __init__(self, records, returncode):
        self.records = records
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_ui_health_check_streams_partial_and_final_results():
    with tempfile.TemporaryDirectory() as tmp:
        records = [
            {"event": "running", "results": []},
            {
                "event": "result",
                "result": {
                    "name": "页面加载",
                    "state": "pass",
                    "message": "Gemini 页面已就绪",
                },
            },
            {"event": "complete", "exit_code": 0},
        ]
        process = FakeHealthCheckProcess(records, returncode=0)
        with patch("webui._start_gemini_health_check_process", return_value=process):
            values = list(
                ui_run_gemini_health_check(Path(tmp) / "config.yaml")
            )
        assert "正在启动" in values[0]
        assert "页面加载" in values[-1]
        assert "正常 1" in values[-1]
```

- [ ] **Step 2: Run streaming test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check.GeminiHealthCheckTests.test_ui_health_check_streams_partial_and_final_results
```

Expected: import failure for `ui_run_gemini_health_check`.

- [ ] **Step 3: Implement bounded process polling**

`ui_run_gemini_health_check` must:

1. Create `runs/gemini_health_check/<time-ns>/status.jsonl`.
2. Start the subprocess with hidden-window flags on Windows.
3. Yield `正在启动 Gemini 完整体检…`.
4. Poll new complete JSONL lines every 250 ms.
5. Render accumulated `result` records after every new item.
6. Stop when `complete` arrives or the process exits.
7. Enforce a 10-minute deadline, terminate only the child process it started,
   and return `体检超时，已停止本次独立体检进程。`.
8. Delete the JSONL file and empty run directory after the final render.

- [ ] **Step 4: Write failing WebUI component and event-wiring tests**

```python
def test_webui_contains_full_gemini_health_check_controls():
    source = Path(webui.__file__).read_text(encoding="utf-8")
    assert 'gr.Button("Gemini 一键完整体检"' in source
    assert 'label="Gemini 体检结果"' in source
    assert "ui_run_gemini_health_check" in source
    assert 'api_name="run_gemini_health_check"' in source
```

- [ ] **Step 5: Run WebUI test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_webui_model_settings -k health_check
```

Expected: assertions fail because the controls are absent.

- [ ] **Step 6: Add the button, result panel, and event**

Place the button beside the existing login buttons:

```python
with gr.Row():
    open_gemini_login_btn = gr.Button("打开 Gemini 登录浏览器")
    check_gemini_login_btn = gr.Button("检查登录并关闭浏览器")
    gemini_health_check_btn = gr.Button(
        "Gemini 一键完整体检",
        variant="secondary",
    )

gemini_health_check_result = gr.Markdown(
    "尚未运行 Gemini 完整体检。",
    elem_id="gemini-health-check-result",
)

gemini_health_check_btn.click(
    fn=ui_run_gemini_health_check,
    outputs=gemini_health_check_result,
    api_name="run_gemini_health_check",
)
```

Disable the button on click and re-enable it on both success and failure using
Gradio event chaining. Do not add instructional marketing copy or extra cards.

- [ ] **Step 7: Run Task 4 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check tests.test_webui_model_settings
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```powershell
git add webui.py tests/test_webui_model_settings.py tests/test_gemini_health_check.py
git commit -m "feat: add Gemini health check button"
```

---

### Task 5: End-To-End Verification

**Files:**
- Modify only if verification exposes a defect in the files above.

**Interfaces:**
- Consumes all interfaces from Tasks 1-4.
- Produces a verified feature ready for packaging.

- [ ] **Step 1: Run focused health-check tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemini_health_check tests.test_medium_priority tests.test_webui_model_settings
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete application test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 3: Run source-mode CLI smoke test**

```powershell
$status = Join-Path $env:TEMP "gemini-health-check-smoke.jsonl"
.\.venv\Scripts\python.exe app.py --gemini-health-check --config config.yaml --status-file $status
Get-Content -LiteralPath $status -Encoding UTF8
```

Expected: valid JSONL beginning with `running`, containing ordered `result`
records, and ending with `complete`. The browser closes without a sent message.

- [ ] **Step 4: Verify the WebUI event is buildable**

```powershell
.\.venv\Scripts\python.exe -c "from webui import build_ui; build_ui(); print('webui_ok')"
```

Expected: `webui_ok`.

- [ ] **Step 5: Inspect final scope**

```powershell
git diff --check
git status --short
git diff --stat HEAD~4..HEAD
```

Expected: only the approved health-check files and documentation changed; the
pre-existing untracked note files remain unmodified and uncommitted.

---

### Task 6: Package And Publish v1.3.5

**Files:**
- Modify: `version.py`
- Modify: `version.json`
- Generate, ignored: `dist/v1.3.5/Lovart_Auto/`
- Generate, ignored: `update.zip`

**Interfaces:**
- Consumes the verified application from Tasks 1-5.
- Produces Git commit `Release v1.3.5`, tag `v1.3.5`, and a non-draft GitHub Release with `update.zip`.

- [ ] **Step 1: Update the application version and release metadata**

Set:

```python
VERSION = "1.3.5"
```

Keep the existing valid `version.json` integrity fields until the new ZIP exists.
Set its version, URL, changelog, SHA-256, and size together in Step 5 so the
repository never contains placeholder integrity metadata.

The final manifest values are:

- `version`: `1.3.5`
- `url`: `https://github.com/TokihaMisa/Gemini-lovart-image-automation/releases/download/v1.3.5/update.zip`
- `changelog`: `新增 Gemini 一键完整体检：独立验证登录、临时对话、Flash、扩展思考、图片上传和发送按钮，不发送提示词。`
- `sha256`: the lowercase `$hash` computed from the completed archive in Step 5
- `size`: the integer `$size` computed from the same archive in Step 5

- [ ] **Step 2: Re-run the complete suite with final version metadata**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 3: Build the isolated Windows package**

Use the same PyInstaller flags as v1.3.4, with absolute data paths and:

```powershell
--distpath dist\v1.3.5
--workpath build\v1.3.5
--specpath build\v1.3.5
```

Expected output:

```text
dist\v1.3.5\Lovart_Auto\Lovart_Auto.exe
dist\v1.3.5\Lovart_Auto\_internal\python314.dll
dist\v1.3.5\Lovart_Auto\_internal\VCRUNTIME140.dll
dist\v1.3.5\Lovart_Auto\_internal\VCRUNTIME140_1.dll
```

- [ ] **Step 4: Run packaged self-tests**

```powershell
$exe = "dist\v1.3.5\Lovart_Auto\Lovart_Auto.exe"
$help = Start-Process -FilePath $exe -ArgumentList "--run-main","--help" -Wait -PassThru -WindowStyle Hidden
if ($help.ExitCode -ne 0) { throw "Packaged CLI self-test failed" }
```

Run the packaged health check once from its UI button or CLI and confirm that
the JSONL ends with `complete`, the blank image upload passes, and no message is
sent.

- [ ] **Step 5: Create and validate the OTA archive**

```powershell
Compress-Archive `
  -Path "dist\v1.3.5\Lovart_Auto\*" `
  -DestinationPath "update.zip" `
  -CompressionLevel Optimal `
  -Force
```

Compute:

```powershell
$hash = (Get-FileHash update.zip -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item update.zip).Length
```

Write `$hash` and `$size` into `version.json`, then call
`updater.validate_and_extract_update` against a nonexistent child directory and
assert the extracted `Lovart_Auto.exe` exists.

- [ ] **Step 6: Commit, push, tag, and publish**

```powershell
git add version.py version.json
git commit -m "Release v1.3.5"
git push origin master
git tag -a v1.3.5 -m "v1.3.5"
git push origin v1.3.5
gh release create v1.3.5 "update.zip#update.zip" `
  --verify-tag `
  --title "v1.3.5 - Gemini 一键完整体检" `
  --notes "新增独立 Gemini 完整体检，验证所有生产关键控件且不发送消息。"
```

- [ ] **Step 7: Verify the remote release**

```powershell
gh release view v1.3.5 --json tagName,name,url,isDraft,isPrerelease,assets
gh api "repos/TokihaMisa/Gemini-lovart-image-automation/contents/version.json?ref=master" --jq ".content | @base64d"
git ls-remote origin refs/heads/master "refs/tags/v1.3.5^{}"
```

Expected: Release is not draft or prerelease; asset size and digest exactly
match `version.json`; remote `master` and dereferenced tag point to the release
commit.
