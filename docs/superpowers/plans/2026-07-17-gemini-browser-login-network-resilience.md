# Gemini Browser Login and Network Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable Gemini browser login assistant, block tasks until the persistent account session is ready, support Chinese/English/Spanish Gemini UI controls, and recover safely from transient network and SSL failures.

**Architecture:** Add `network_retry.py` as the provider-neutral retry classifier/executor and `gemini_browser_session.py` as the single owner of browser profile resolution, Gemini readiness detection, login-helper state, and navigation recovery. `app.py` exposes an internal helper-process entry point; `webui.py`, `main.py`, and `gemini_bot.py` consume the shared session APIs instead of using fixed sleeps or console input. Existing API and Lovart transports keep their request shapes but adopt the same retry classification and progress semantics.

**Tech Stack:** Python 3.12+, Playwright sync API, Gradio 6, PyYAML, standard-library JSON/urllib/ssl/subprocess, unittest/pytest, PyInstaller.

## Global Constraints

- Use the approved balanced policy: 5 network attempts, 90-second Gemini page-ready timeout, 2 complete Gemini-browser product attempts, and delays `[3, 6, 12, 20]` seconds.
- Keep TLS certificate verification enabled by default. Never add Chromium ignore-certificate flags or automatically set `LOVART_INSECURE_SSL=1`.
- Retry only transient timeout, connection, DNS, network-change, selected SSL protocol, HTTP 408/429, and 5xx failures.
- Do not retry 401/403, model/endpoint 404, login/verification challenges, or permanent certificate authority/name/date failures.
- Google cookies and account state stay only in the local `browser_profile`; do not log email, cookies, tokens, auth headers, or profile contents.
- The login helper and formal task must use the same persistent profile, browser executable resolution, launch arguments, readiness detector, and config defaults.
- Only one process may own the configured profile at a time. Never terminate unrelated user Chrome/Edge processes.
- Prefer structural selectors. Text fallbacks must cover Chinese, English, and Spanish with Unicode/accent normalization.
- Do not send the product prompt before image upload completion is verified.
- Gemini product retries happen before the final Lovart detail-page request and must not duplicate final Lovart drawing.
- Gemini API and NVIDIA sources must remain usable without browser login state.
- All tests are offline unless a final manual login smoke is explicitly performed; automated tests must not use a real Google account or API quota.
- After implementation verification, bump `version.py` and `version.json` from `1.2.0` to `1.3.0`, build and verify `update.zip`; create GitHub Release `v1.3.0` and push `master` only after the cumulative final code review passes.

---

## File Structure

- Create `network_retry.py`: retry policy, error classification, delay selection, and provider-neutral retry executor.
- Create `gemini_browser_session.py`: profile/runtime paths, browser launch options, page readiness, login status protocol, navigation recovery, and login helper loop.
- Create `tests/test_network_retry.py`: transient/permanent classification and exact balanced-policy behavior.
- Create `tests/test_gemini_browser_session.py`: readiness states, status persistence, profile locking, helper command/lifecycle, and navigation retries.
- Create `tests/test_webui_gemini_login.py`: login buttons, callbacks, task guard, and Gradio event graph.
- Modify `app.py`: internal `--gemini-login-helper` entry point.
- Modify `webui.py`: account controls, helper subprocess callbacks, and browser task guard.
- Modify `main.py`: reuse shared browser startup/readiness and remove packaged-console login input.
- Modify `gemini_bot.py`: structural/multilingual selectors, Thinking recovery, diagnostics, and product attempts.
- Modify `gemini_api.py`, `nvidia_api.py`, `model_provider.py`, and `lovart_api.py`: shared retry classification and consistent safe progress messages.
- Modify `config.example.yaml` and `webui.py` embedded defaults: balanced browser settings.
- Modify `README.md`: login workflow, Spanish behavior, network/SSL behavior, and troubleshooting.
- Modify `version.py`, `version.json`, and build/release artifacts only after the complete verification gate.

---

### Task 1: Shared retry policy and error classification

**Files:**
- Create: `network_retry.py`
- Create: `tests/test_network_retry.py`

**Interfaces:**
- Produces: `RetryKind(Enum)` values `TRANSIENT`, `PERMANENT_TLS`, `AUTH`, `NOT_FOUND`, `OTHER`.
- Produces: `RetryPolicy(network_attempts=5, page_ready_timeout=90.0, product_attempts=2, retry_delays=(3.0, 6.0, 12.0, 20.0))`.
- Produces: `retry_policy_from_config(config: Mapping[str, object]) -> RetryPolicy`.
- Produces: `classify_network_error(exc: BaseException) -> RetryKind`.
- Produces: `run_with_retry(operation, policy, *, on_retry=None, sleep=None)`; `None` resolves to `time.sleep` at call time so tests can patch it safely.
- Produces: `safe_retry_message(kind, attempt, attempts, delay) -> str` without raw exception text.

- [ ] **Step 1: Write failing balanced-policy and classification tests**

Create `tests/test_network_retry.py`:

```python
import socket
import ssl
import unittest
from urllib.error import HTTPError, URLError

from network_retry import (
    RetryKind,
    RetryPolicy,
    classify_network_error,
    retry_policy_from_config,
    run_with_retry,
    safe_retry_message,
)


class NetworkRetryTests(unittest.TestCase):
    def test_missing_config_uses_approved_balanced_policy(self):
        policy = retry_policy_from_config({})
        self.assertEqual(policy.network_attempts, 5)
        self.assertEqual(policy.page_ready_timeout, 90)
        self.assertEqual(policy.product_attempts, 2)
        self.assertEqual(policy.retry_delays, (3, 6, 12, 20))

    def test_classifies_transient_browser_and_transport_failures(self):
        cases = [
            TimeoutError(),
            socket.timeout(),
            ConnectionResetError(),
            URLError("temporary DNS failure"),
            RuntimeError("net::ERR_NETWORK_CHANGED"),
            RuntimeError("net::ERR_SSL_PROTOCOL_ERROR"),
            HTTPError("https://example.test", 429, "rate", {}, None),
            HTTPError("https://example.test", 503, "down", {}, None),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(classify_network_error(exc), RetryKind.TRANSIENT)

    def test_classifies_permanent_tls_and_auth_without_retry(self):
        for exc in (
            ssl.SSLCertVerificationError("certificate verify failed"),
            RuntimeError("net::ERR_CERT_AUTHORITY_INVALID"),
            RuntimeError("net::ERR_CERT_COMMON_NAME_INVALID"),
            RuntimeError("net::ERR_CERT_DATE_INVALID"),
        ):
            self.assertEqual(classify_network_error(exc), RetryKind.PERMANENT_TLS)
        self.assertEqual(
            classify_network_error(HTTPError("https://x", 403, "denied", {}, None)),
            RetryKind.AUTH,
        )
        self.assertEqual(
            classify_network_error(HTTPError("https://x", 404, "missing", {}, None)),
            RetryKind.NOT_FOUND,
        )

    def test_retry_executor_uses_exact_delays_and_returns_success(self):
        calls, delays, notices = [], [], []

        def operation():
            calls.append(len(calls) + 1)
            if len(calls) < 5:
                raise ConnectionResetError("secret must not be logged")
            return "ok"

        result = run_with_retry(
            operation,
            RetryPolicy(),
            on_retry=lambda notice: notices.append(notice),
            sleep=lambda delay: delays.append(delay),
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, [1, 2, 3, 4, 5])
        self.assertEqual(delays, [3, 6, 12, 20])
        self.assertTrue(all("secret" not in notice for notice in notices))

    def test_permanent_tls_is_not_retried(self):
        calls = []
        with self.assertRaises(ssl.SSLCertVerificationError):
            run_with_retry(
                lambda: calls.append(1) or (_ for _ in ()).throw(
                    ssl.SSLCertVerificationError("certificate verify failed")
                ),
                RetryPolicy(),
                sleep=lambda _delay: self.fail("must not sleep"),
            )
        self.assertEqual(calls, [1])

    def test_retry_notice_contains_progress_but_not_raw_exception(self):
        message = safe_retry_message(RetryKind.TRANSIENT, 2, 5, 6)
        self.assertIn("第 2/5 次", message)
        self.assertIn("6 秒后重试", message)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
uv run --with pytest python -m pytest tests/test_network_retry.py -v
```

Expected: import failure because `network_retry.py` does not exist.

- [ ] **Step 3: Implement the shared retry module**

Create `network_retry.py` with this public structure:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from urllib.error import HTTPError, URLError


class RetryKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT_TLS = "permanent_tls"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    OTHER = "other"


@dataclass(frozen=True)
class RetryPolicy:
    network_attempts: int = 5
    page_ready_timeout: float = 90.0
    product_attempts: int = 2
    retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0, 20.0)

    def delay_after(self, failed_attempt: int) -> float:
        index = min(max(failed_attempt - 1, 0), len(self.retry_delays) - 1)
        return self.retry_delays[index]


def retry_policy_from_config(config: Mapping[str, object]) -> RetryPolicy:
    browser = config.get("browser", {}) if isinstance(config, Mapping) else {}
    browser = browser if isinstance(browser, Mapping) else {}
    delays = tuple(float(value) for value in browser.get("retry_delays", (3, 6, 12, 20)))
    return RetryPolicy(
        network_attempts=max(1, int(browser.get("network_attempts", 5))),
        page_ready_timeout=max(1.0, float(browser.get("page_ready_timeout", 90))),
        product_attempts=max(1, int(browser.get("product_attempts", 2))),
        retry_delays=delays or (3.0, 6.0, 12.0, 20.0),
    )
```

Implement `classify_network_error` in this order: permanent TLS message markers; `HTTPError` status; timeout/connection/`URLError`; transient Playwright `net::ERR_*` markers; other. Implement `run_with_retry` so only `TRANSIENT` is retried, `on_retry` receives only `safe_retry_message`, and the final original exception is re-raised.

Use a runtime-resolved sleeper:

```python
def run_with_retry(operation, policy, *, on_retry=None, sleep=None):
    sleeper = sleep or time.sleep
    for attempt in range(1, policy.network_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            kind = classify_network_error(exc)
            if kind is not RetryKind.TRANSIENT or attempt >= policy.network_attempts:
                raise
            delay = policy.delay_after(attempt)
            if on_retry:
                on_retry(safe_retry_message(kind, attempt, policy.network_attempts, delay))
            sleeper(delay)
    raise RuntimeError("retry loop exhausted")
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
uv run --with pytest python -m pytest tests/test_network_retry.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the retry domain**

```powershell
git add network_retry.py tests/test_network_retry.py
git commit -m "feat: add shared network retry policy"
```

---

### Task 2: Gemini browser session, readiness, and login state protocol

**Files:**
- Create: `gemini_browser_session.py`
- Create: `tests/test_gemini_browser_session.py`
- Modify: `main.py` only to re-export the existing browser executable resolver during migration.

**Interfaces:**
- Consumes: `RetryPolicy`, `retry_policy_from_config`, `run_with_retry`.
- Produces: `GeminiPageState(Enum)` values `STARTING`, `PAGE_LOADING`, `WAITING_LOGIN`, `READY`, `CLOSING`, `CLOSED`, `ERROR`.
- Produces: `LoginStatus` dataclass with `pid`, `state`, `ready`, `url`, `language`, `message`, `updated_at`, plus `LoginStatus.create(state, ready, url, language, message, pid=None)`.
- Produces: `LoginRuntimePaths(status_path, close_request_path)`.
- Produces: `login_runtime_paths(config_path: str | Path) -> LoginRuntimePaths`.
- Produces: `inspect_gemini_page(page) -> LoginStatus`.
- Produces: `wait_for_gemini_ready(page, policy, logger=None) -> LoginStatus`.
- Produces: `navigate_gemini_with_retry(page, url, policy, logger=None) -> LoginStatus`.
- Produces: `read_login_status`, `write_login_status`, `request_login_helper_close`, `login_helper_is_active`, and `clear_stale_login_runtime`.
- Produces: `resolve_browser_executable`, `resolve_user_data_dir`, and `build_browser_launch_options` reused by `main.py` and helper runtime.

- [ ] **Step 1: Write failing readiness and state-protocol tests**

Create `tests/test_gemini_browser_session.py` with fake pages whose `evaluate` returns complete page-state payloads:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemini_browser_session import (
    GeminiPageState,
    LoginStatus,
    clear_stale_login_runtime,
    inspect_gemini_page,
    login_helper_is_active,
    login_runtime_paths,
    navigate_gemini_with_retry,
    read_login_status,
    request_login_helper_close,
    write_login_status,
)
from network_retry import RetryPolicy


class FakePage:
    def __init__(self, url, payload):
        self.url = url
        self.payload = payload
        self.goto_calls = 0

    def evaluate(self, _script):
        return self.payload

    def goto(self, _url, **_kwargs):
        self.goto_calls += 1


class GeminiBrowserSessionTests(unittest.TestCase):
    def test_accounts_url_is_waiting_login_even_if_editor_like_node_exists(self):
        page = FakePage("https://accounts.google.com/signin", {
            "language": "es", "has_editor": True, "has_login_prompt": True,
            "has_loading": False, "controls": [],
        })
        status = inspect_gemini_page(page)
        self.assertEqual(status.state, GeminiPageState.WAITING_LOGIN)
        self.assertFalse(status.ready)

    def test_structural_editor_marks_spanish_page_ready(self):
        page = FakePage("https://gemini.google.com/app", {
            "language": "es-ES", "has_editor": True, "has_login_prompt": False,
            "has_loading": False, "controls": ["Rápido", "Adjuntar archivos"],
        })
        status = inspect_gemini_page(page)
        self.assertEqual(status.state, GeminiPageState.READY)
        self.assertTrue(status.ready)
        self.assertEqual(status.language, "es-ES")

    def test_atomic_status_round_trip_and_close_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            status = LoginStatus.create(GeminiPageState.READY, True, "https://gemini.google.com/app", "es", "ready", pid=42)
            write_login_status(paths.status_path, status)
            self.assertEqual(read_login_status(paths.status_path), status)
            request_login_helper_close(paths.close_request_path)
            self.assertTrue(paths.close_request_path.exists())
            self.assertFalse(paths.status_path.with_suffix(".tmp").exists())

    @patch("gemini_browser_session.process_is_alive", return_value=False)
    def test_stale_status_is_cleared_without_killing_any_browser(self, _alive):
        with tempfile.TemporaryDirectory() as tmp:
            paths = login_runtime_paths(Path(tmp) / "config.yaml")
            write_login_status(paths.status_path, LoginStatus.create(GeminiPageState.READY, True, "", "", "", pid=99))
            self.assertTrue(clear_stale_login_runtime(paths))
            self.assertFalse(paths.status_path.exists())

    def test_navigation_retries_transient_failure_then_requires_ready_page(self):
        page = FakePage("https://gemini.google.com/app", {
            "language": "en", "has_editor": True, "has_login_prompt": False,
            "has_loading": False, "controls": [],
        })
        failures = [RuntimeError("net::ERR_CONNECTION_RESET")]
        original_goto = page.goto
        def flaky_goto(url, **kwargs):
            if failures:
                raise failures.pop()
            return original_goto(url, **kwargs)
        page.goto = flaky_goto
        with patch("network_retry.time.sleep"):
            status = navigate_gemini_with_retry(page, "https://gemini.google.com", RetryPolicy())
        self.assertTrue(status.ready)
        self.assertEqual(page.goto_calls, 1)
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_gemini_browser_session.py -v
```

Expected: import failure because the browser session module does not exist.

- [ ] **Step 3: Implement page inspection and atomic runtime state**

Implement `inspect_gemini_page` with one JavaScript evaluation returning exactly:

```python
{
    "language": str,
    "has_editor": bool,
    "has_login_prompt": bool,
    "has_loading": bool,
    "controls": list[str],
}
```

The script must detect visible editable `textarea`, `[contenteditable=true]`, and `[role=textbox]`; visible login/account/verification signals in Chinese, English, and Spanish; and visible loading/offline signals. Strip query and fragment from the URL before writing it to status.

Implement atomic state writes as:

```python
def write_login_status(path: str | Path, status: LoginStatus) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(status.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
```

Use `runs/gemini_login/status.json` and `runs/gemini_login/close.request` relative to the config directory. `clear_stale_login_runtime` may unlink only these runtime files after confirming the recorded PID is not alive; it must not terminate browser processes.

- [ ] **Step 4: Move shared browser launch configuration**

Move the behavior of `main.resolve_browser_executable`, user-data-dir resolution, and launch option construction into `gemini_browser_session.py`. Preserve a wrapper import in `main.py` so existing tests importing `resolve_browser_executable` continue to pass:

```python
from gemini_browser_session import resolve_browser_executable
```

`build_browser_launch_options` must return the current non-headless, no-viewport, automation-control arguments and optional installed Chrome/Edge executable, without any ignore-certificate option.

- [ ] **Step 5: Run session and existing browser tests**

```powershell
uv run --with pytest python -m pytest tests/test_gemini_browser_session.py tests/test_medium_priority.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit session primitives**

```powershell
git add gemini_browser_session.py main.py tests/test_gemini_browser_session.py tests/test_medium_priority.py
git commit -m "feat: add Gemini browser session readiness"
```

---

### Task 3: Login helper process and WebUI account controls

**Files:**
- Modify: `gemini_browser_session.py`
- Modify: `app.py`
- Modify: `webui.py`
- Create: `tests/test_webui_gemini_login.py`
- Modify: `tests/test_gemini_browser_session.py`

**Interfaces:**
- Produces: `run_login_helper(config_path: str | Path) -> int`.
- Produces: `build_login_helper_command(config_path, *, executable=None, frozen=None) -> list[str]`.
- Produces: `open_gemini_login_browser(config_path="config.yaml") -> str`.
- Produces: `check_gemini_login_and_close(config_path="config.yaml") -> str`.
- Produces: `guard_gemini_browser_task(prompt_source, config_path="config.yaml") -> str | None`.

- [ ] **Step 1: Write failing helper command/lifecycle and callback tests**

Append to `tests/test_gemini_browser_session.py`:

```python
def test_helper_command_matches_source_and_frozen_entry_points(self):
    source = build_login_helper_command("config.yaml", executable="python.exe", frozen=False)
    frozen = build_login_helper_command("config.yaml", executable="Lovart_Auto.exe", frozen=True)
    self.assertEqual(source[:2], ["python.exe", str(Path("app.py").resolve())])
    self.assertEqual(source[2:], ["--gemini-login-helper", "--config", str(Path("config.yaml").resolve())])
    self.assertEqual(frozen, ["Lovart_Auto.exe", "--gemini-login-helper", "--config", str(Path("config.yaml").resolve())])
```

Create `tests/test_webui_gemini_login.py`:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemini_browser_session import GeminiPageState, LoginStatus, login_runtime_paths, write_login_status
from webui import (
    build_ui,
    check_gemini_login_and_close,
    guard_gemini_browser_task,
    open_gemini_login_browser,
)


class WebUIGeminiLoginTests(unittest.TestCase):
    @patch("webui.subprocess.Popen")
    @patch("webui.login_helper_is_active", return_value=False)
    def test_open_button_starts_one_helper_and_returns_waiting_status(self, _active, popen):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            message = open_gemini_login_browser(config)
        self.assertEqual(popen.call_count, 1)
        self.assertIn("登录浏览器已打开", message)

    @patch("webui.login_helper_is_active", return_value=True)
    def test_open_button_does_not_start_duplicate_helper(self, _active):
        with patch("webui.subprocess.Popen") as popen:
            message = open_gemini_login_browser("config.yaml")
        popen.assert_not_called()
        self.assertIn("已经打开", message)

    def test_check_does_not_close_when_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            write_login_status(paths.status_path, LoginStatus.create(
                GeminiPageState.WAITING_LOGIN, False, "https://accounts.google.com", "es", "等待登录", pid=42
            ))
            with patch("webui.login_helper_is_active", return_value=True):
                message = check_gemini_login_and_close(config)
            self.assertFalse(paths.close_request_path.exists())
        self.assertIn("尚未完成登录", message)

    def test_ready_check_requests_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            write_login_status(paths.status_path, LoginStatus.create(
                GeminiPageState.READY, True, "https://gemini.google.com/app", "es", "ready", pid=42
            ))
            with patch("webui.login_helper_is_active", return_value=True):
                message = check_gemini_login_and_close(config)
            self.assertTrue(paths.close_request_path.exists())
        self.assertIn("登录已确认", message)

    def test_browser_task_guard_blocks_active_helper_only_for_browser_source(self):
        with patch("webui.login_helper_is_active", return_value=True):
            self.assertIn("登录浏览器", guard_gemini_browser_task("gemini_browser"))
            self.assertIsNone(guard_gemini_browser_task("gemini_api"))
            self.assertIsNone(guard_gemini_browser_task("nvidia"))
```

- [ ] **Step 2: Run helper/WebUI tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_gemini_browser_session.py tests/test_webui_gemini_login.py -v
```

Expected: missing helper and callback imports.

- [ ] **Step 3: Implement the helper loop and app entry point**

`run_login_helper` must:

1. Write `starting` with its PID.
2. Refuse to proceed if another live helper owns the runtime.
3. Launch the persistent browser using shared options.
4. Navigate with balanced retry.
5. Poll `inspect_gemini_page` once per second and atomically write changes.
6. On `close.request` while ready, write `closing`, close context, remove request, and write `closed`.
7. If the browser is manually closed, write `closed` or a safe `error` without raw cookies/profile details.

Add this branch before normal app startup in `app.py`:

```python
if "--gemini-login-helper" in sys.argv:
    from gemini_browser_session import run_login_helper
    config_index = sys.argv.index("--config") if "--config" in sys.argv else -1
    config_path = sys.argv[config_index + 1] if config_index >= 0 else "config.yaml"
    raise SystemExit(run_login_helper(config_path))
```

- [ ] **Step 4: Implement pure WebUI callbacks and controls**

Add a `Gemini 浏览器账号` section in the `API 与模型` tab with the approved two buttons and a read-only status Markdown. Give Gradio events stable API names `open_gemini_login_browser` and `check_gemini_login_and_close`.

`open_gemini_login_browser` must build the helper command, set UTF-8 environment flags, launch hidden only for the helper EXE process, and return immediately. `check_gemini_login_and_close` requests close only for a ready live helper. `guard_gemini_browser_task` blocks only `gemini_browser`.

At the beginning of `run_process`, before saving Excel or spawning `main.py`:

```python
guard_message = guard_gemini_browser_task(prompt_source, config_path=config_path)
if guard_message:
    yield f"❌ {guard_message}"
    return
```

- [ ] **Step 5: Add an event-graph assertion**

Build the real Gradio UI under patched config and assert both API names exist, each has one status output, and the two button labels are visible. Do not mock `build_ui` or the Gradio config.

- [ ] **Step 6: Run helper/UI and full WebUI tests**

```powershell
uv run --with pytest python -m pytest tests/test_webui_gemini_login.py tests/test_webui_model_settings.py tests/test_gemini_browser_session.py -v
```

Expected: all tests pass without launching a real browser.

- [ ] **Step 7: Commit login helper UI**

```powershell
git add app.py webui.py gemini_browser_session.py tests/test_gemini_browser_session.py tests/test_webui_gemini_login.py
git commit -m "feat: add Gemini browser login assistant"
```

---

### Task 4: Formal browser flow readiness gate

**Files:**
- Modify: `main.py`
- Modify: `gemini_browser_session.py`
- Modify: `tests/test_medium_priority.py`
- Modify: `tests/test_gemini_browser_session.py`

**Interfaces:**
- Consumes: shared profile launch options, `navigate_gemini_with_retry`, `wait_for_gemini_ready`, `GeminiPageState`.
- Produces: `GeminiLoginRequiredError`, `GeminiPageNotReadyError`, and `GeminiPermanentTlsError` with user-safe Chinese messages.
- Modifies: `_run_browser_flow` to fail before `_process_products` when the page is not ready.

- [ ] **Step 1: Write failing formal-flow gate tests**

Add a complete fake Playwright manager and call the real `_run_browser_flow` control flow:

```python
class FakeLogger:
    def info(self, _message):
        pass
    def warning(self, _message):
        pass


class FakeContext:
    def __init__(self):
        self.pages = [object()]
        self.closed = False
    def new_page(self):
        return self.pages[0]
    def close(self):
        self.closed = True


class FakePlaywrightManager:
    def __init__(self, context):
        self.playwright = type("PW", (), {
            "chromium": type("Chromium", (), {
                "launch_persistent_context": lambda _self, **_kwargs: context,
            })(),
        })()
    def __enter__(self):
        return self.playwright
    def __exit__(self, exc_type, exc, tb):
        return False


def run_formal_flow_for_test():
    config = {
        "browser": {"user_data_dir": "browser_profile"},
        "gemini": {"base_url": "https://gemini.google.com", "thinking_mode": True},
    }
    context = FakeContext()
    with patch("main.sync_playwright", return_value=FakePlaywrightManager(context)), patch(
        "main.build_browser_launch_options", return_value={}
    ):
        return main._run_browser_flow(
            config,
            products=[object()],
            lovart=object(),
            logger=FakeLogger(),
            run_dir=Path("runs/test"),
            resume=False,
            wait_for_ready=False,
            prompt_settings={},
        )


@patch("main._process_products")
@patch("main.navigate_gemini_with_retry")
def test_formal_browser_flow_does_not_process_products_when_login_required(navigate, process):
    navigate.side_effect = GeminiLoginRequiredError("Gemini 未登录，请先使用登录按钮")
    with self.assertRaises(GeminiLoginRequiredError):
        run_formal_flow_for_test()
    process.assert_not_called()

@patch("main._process_products", return_value=(1, 0, 0, 0))
@patch("main.navigate_gemini_with_retry")
def test_formal_browser_flow_processes_only_after_ready(navigate, process):
    navigate.return_value = LoginStatus.create(GeminiPageState.READY, True, "https://gemini.google.com/app", "es", "ready", pid=1)
    self.assertEqual(run_formal_flow_for_test(), (1, 0, 0, 0))
    process.assert_called_once()
```

- [ ] **Step 2: Run formal-flow tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_medium_priority.py -k "formal_browser_flow" -v
```

Expected: failures because `_run_browser_flow` still uses URL checks, fixed sleeps, and `input()`.

- [ ] **Step 3: Replace fixed login/input flow with shared readiness**

Refactor `_run_browser_flow` so it:

```python
policy = retry_policy_from_config(config)
launch_options = build_browser_launch_options(config, config_path=Path("config.yaml"))
context = pw.chromium.launch_persistent_context(**launch_options)
page = context.pages[0] if context.pages else context.new_page()
status = navigate_gemini_with_retry(page, config["gemini"]["base_url"], policy, logger)
if status.state is GeminiPageState.WAITING_LOGIN:
    raise GeminiLoginRequiredError("Gemini 未登录，请先在软件内打开登录浏览器")
if not status.ready:
    raise GeminiPageNotReadyError("Gemini 页面未在 90 秒内准备完成")
```

Do not call `input()` when `UI_MODE=1`. Preserve optional console confirmation only for explicit interactive CLI `ask` mode after the page is ready.

- [ ] **Step 4: Add Chinese error mapping to WebUI logs/cards**

Ensure these exceptions are not collapsed to `Gemini Thinking mode could not be selected`. The product error text must retain the Chinese root category and diagnostic path.

- [ ] **Step 5: Run browser-flow and full main tests**

```powershell
uv run --with pytest python -m pytest tests/test_medium_priority.py tests/test_high_priority.py tests/test_nvidia_api.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the readiness gate**

```powershell
git add main.py gemini_browser_session.py tests/test_medium_priority.py tests/test_gemini_browser_session.py
git commit -m "fix: gate Gemini tasks on browser readiness"
```

---

### Task 5: Multilingual controls, Thinking recovery, upload ordering, and product retries

**Files:**
- Modify: `gemini_bot.py`
- Modify: `gemini_browser_session.py`
- Modify: `tests/test_medium_priority.py`

**Interfaces:**
- Consumes: `RetryPolicy`, readiness inspection, safe diagnostic helpers.
- Produces: `normalize_ui_text(text: str) -> str`, `matches_ui_term(text, terms) -> bool`, and immutable `MODE_TERMS`, `EXTENDED_THINKING_TERMS`, `UPLOAD_TERMS`, `TEMPORARY_CHAT_TERMS`.
- Produces: `GeminiBot._generate_prompt_once(...)` containing the current one-attempt flow.
- Modifies: `GeminiBot.generate_prompt(...)` to run at most `policy.product_attempts` attempts for transient/page-structure failures.
- Produces: `save_gemini_diagnostics(page, run_dir, product_id, reason, attempts, error_kind) -> Path`.

- [ ] **Step 1: Write failing Spanish selector and ordering tests**

Add tests with fake DOM-scan text and ordered method overrides:

```python
class NullLogger:
    def info(self, _message):
        pass
    def warning(self, _message):
        pass


class NullPage:
    def goto(self, _url, **_kwargs):
        pass
    def wait_for_timeout(self, _milliseconds):
        pass


class OrderedGeminiBot(GeminiBot):
    def __init__(self, upload_result):
        super().__init__(
            NullPage(),
            {"gemini": {"thinking_mode": True}, "browser": {"product_attempts": 2}},
            NullLogger(),
        )
        self.upload_result = upload_result
        self.events = []
    def _start_temporary_chat(self):
        self.events.append("temporary_chat")
        return True
    def _select_thinking_mode(self):
        self.events.append("thinking")
        return True
    def _response_count(self):
        return 0
    def _send_message(self, text):
        self.events.append("product_prompt" if "产品信息/卖点" in text else "preamble")
    def _wait_for_reply(self, **_kwargs):
        pass
    def _upload_images(self, _paths):
        self.events.append("upload")
        return self.upload_result
    def _get_last_response(self):
        return "generated prompt " * 20


class RetryGeminiBot(GeminiBot):
    def __init__(self, first_error):
        super().__init__(
            NullPage(),
            {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}},
            NullLogger(),
        )
        self.first_error = first_error
        self.attempts = 0
        self.events = []
    def _generate_prompt_once(self, *_args, **_kwargs):
        self.attempts += 1
        self.events.append("temporary_chat")
        if self.attempts == 1:
            raise self.first_error
        return "generated prompt"


def test_spanish_text_normalization_removes_accents_and_case(self):
    self.assertEqual(normalize_ui_text("  PENSAMIENTO RÁPIDO  "), "pensamiento rapido")

def test_spanish_mode_and_upload_terms_are_recognized(self):
    self.assertTrue(matches_ui_term("Rápido", MODE_TERMS))
    self.assertTrue(matches_ui_term("Pensamiento ampliado", EXTENDED_THINKING_TERMS))
    self.assertTrue(matches_ui_term("Adjuntar archivos", UPLOAD_TERMS))
    self.assertTrue(matches_ui_term("Chat temporal", TEMPORARY_CHAT_TERMS))

def test_product_prompt_is_not_sent_when_upload_never_completes(self):
    bot = OrderedGeminiBot(upload_result=False)
    with self.assertRaisesRegex(RuntimeError, "图片上传未完成"):
        bot.generate_prompt("产品", "西班牙语", "卖点", ["image.jpg"])
    self.assertNotIn("product_prompt", bot.events)

def test_transient_first_product_attempt_restarts_temporary_chat_then_succeeds(self):
    bot = RetryGeminiBot(first_error=RuntimeError("net::ERR_CONNECTION_RESET"))
    with patch("network_retry.time.sleep"):
        result = bot.generate_prompt("产品", "西班牙语", "卖点", ["image.jpg"])
    self.assertEqual(result, "generated prompt")
    self.assertEqual(bot.attempts, 2)
    self.assertEqual(bot.events.count("temporary_chat"), 2)
```

- [ ] **Step 2: Run targeted Gemini tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_medium_priority.py -k "spanish or product_prompt or product_attempt" -v
```

Expected: missing normalization/term interfaces and current English error behavior.

- [ ] **Step 3: Implement structural-first multilingual matching**

Add immutable term groups for the approved languages and normalize via `unicodedata.normalize("NFKD", text)`, removal of combining marks, `casefold()`, and whitespace collapse. Keep structural selectors first. DOM scans must inspect text, `aria-label`, `title`, and `data-tooltip` but store only bounded, de-duplicated summaries.

Extend temporary, mode, extended-thinking, thinking-level, and upload paths with Spanish fallbacks. Do not remove existing Chinese/English selectors.

- [ ] **Step 4: Add Thinking recovery before failure**

When `_select_thinking_mode` initially fails:

1. Inspect readiness.
2. If login is required, raise `GeminiLoginRequiredError` immediately.
3. If transient/page-loading, navigate/refresh with shared retry and wait for ready.
4. Retry Thinking selection.
5. If still absent on a ready page, raise a distinct page-structure error and save diagnostics.

- [ ] **Step 5: Split one attempt from product retry wrapper**

Move the current body to `_generate_prompt_once`. `generate_prompt` reads `RetryPolicy`, runs at most two attempts, and retries only transient/page-loading/page-structure categories. Permanent TLS, login, auth, and verified upload failures keep their specific error. Each new attempt starts a fresh temporary chat. The returned text is still written once after success.

- [ ] **Step 6: Extend diagnostics**

Alongside existing screenshot/HTML, write a UTF-8 JSON metadata file containing sanitized URL origin/path, language, bounded visible-control summary, attempt count, and error kind. Exclude page storage, cookies, email-like text, request headers, and query strings.

- [ ] **Step 7: Run all Gemini browser tests**

```powershell
uv run --with pytest python -m pytest tests/test_medium_priority.py tests/test_gemini_browser_session.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit multilingual resilience**

```powershell
git add gemini_bot.py gemini_browser_session.py tests/test_medium_priority.py
git commit -m "fix: harden multilingual Gemini browser automation"
```

---

### Task 6: Align Gemini API, NVIDIA, model-provider, and Lovart retries

**Files:**
- Modify: `gemini_api.py`
- Modify: `nvidia_api.py`
- Modify: `model_provider.py`
- Modify: `lovart_api.py`
- Modify: `tests/test_model_provider.py`
- Modify: `tests/test_nvidia_api.py`
- Modify: `tests/test_low_priority.py`

**Interfaces:**
- Consumes: `RetryPolicy`, `classify_network_error`, `safe_retry_message`.
- Preserves all public request method signatures and payload formats.
- Adds optional injected `retry_policy` only where constructors already accept configuration; defaults remain balanced.

- [ ] **Step 1: Write failing transport-boundary tests**

Add these behaviors without real network calls:

For the Lovart test, add `import json`, `import ssl`, and `from lovart_api import AgentSkill` to `tests/test_low_priority.py`. `tests/test_model_provider.py` already provides `FakeResponse`, `HTTPError`, `ssl`, `discover_models`, and `ModelProviderError`.

```python
@patch("network_retry.time.sleep")
@patch("urllib.request.urlopen")
def test_model_discovery_retries_503_then_returns_models(self, urlopen, _sleep):
    urlopen.side_effect = [
        HTTPError("https://x", 503, "down", {}, None),
        FakeResponse({"data": [{"id": "moonshotai/kimi-k2.5"}]}),
    ]
    models = discover_models("nvidia", "key", "https://x/v1")
    self.assertEqual(models[0].model_id, "moonshotai/kimi-k2.5")
    self.assertEqual(urlopen.call_count, 2)

@patch("urllib.request.urlopen")
def test_model_discovery_does_not_retry_401_or_permanent_tls(self, urlopen):
    for error in (HTTPError("https://x", 401, "bad", {}, None), ssl.SSLCertVerificationError("verify")):
        urlopen.side_effect = error
        with self.assertRaises(ModelProviderError):
            discover_models("nvidia", "key", "https://x/v1")
        self.assertEqual(urlopen.call_count, 1)
        urlopen.reset_mock()

def test_lovart_request_retries_transient_ssl_with_resigning(self):
    class LovartResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return json.dumps({"code": 0, "data": {"ok": True}}).encode("utf-8")

    client = AgentSkill("https://lovart.test", "access-key", "secret-key")
    with patch.object(client, "_sign", wraps=client._sign) as sign, patch(
        "lovart_api.urllib.request.urlopen",
        side_effect=[ssl.SSLError("protocol interrupted"), LovartResponse()],
    ) as urlopen, patch("network_retry.time.sleep"):
        result = client._request("GET", "/status")
    self.assertTrue(result["ok"])
    self.assertEqual(urlopen.call_count, 2)
    self.assertEqual(sign.call_count, 2)
```

Also assert retry log messages contain attempt/delay but never the test API key.

- [ ] **Step 2: Run transport tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_model_provider.py tests/test_nvidia_api.py tests/test_low_priority.py -k "retry or tls or 503 or 401" -v
```

Expected: model discovery remains single-attempt and retry messages differ.

- [ ] **Step 3: Apply shared classification to formal Gemini/NVIDIA calls**

Replace unconditional retry decorators with `run_with_retry` around the actual request operation. Preserve request construction and response parsing. 400/401/403/404 must escape after one request; 408/429/5xx and transient transport errors use the balanced policy. Log only `safe_retry_message`.

- [ ] **Step 4: Apply shared classification to model discovery/probe**

Keep validation before retries. Wrap `_request_json` network execution so pagination retries only the failed page and does not restart already parsed pages. Preserve operation-aware 404 messages.

- [ ] **Step 5: Align Lovart without breaking request signatures**

Keep per-attempt signing inside the loop so timestamps remain fresh. Replace fixed `2 * (attempt + 1)` delays with the shared policy. Preserve existing optional insecure SSL environment behavior but never enable it automatically. Map permanent TLS failures to a Chinese certificate/system-time/proxy message after one attempt.

- [ ] **Step 6: Run all provider/Lovart tests**

```powershell
uv run --with pytest python -m pytest tests/test_model_provider.py tests/test_nvidia_api.py tests/test_low_priority.py -v
```

Expected: all tests pass with no secrets in output.

- [ ] **Step 7: Commit transport alignment**

```powershell
git add network_retry.py gemini_api.py nvidia_api.py model_provider.py lovart_api.py tests/test_model_provider.py tests/test_nvidia_api.py tests/test_low_priority.py
git commit -m "fix: align provider network retry behavior"
```

---

### Task 7: Defaults, documentation, and complete regression

**Files:**
- Modify: `config.example.yaml`
- Modify: `webui.py` embedded default config
- Modify: `README.md`
- Modify: `tests/test_setup_wizard.py`
- Modify: `tests/test_webui_gemini_login.py`

**Interfaces:**
- Publishes the exact balanced defaults in both config templates.
- Documents the two-button workflow and troubleshooting categories.

- [ ] **Step 1: Write failing default-consistency tests**

Assert parsed `config.example.yaml` and a newly created setup config both contain:

```python
{
    "network_attempts": 5,
    "page_ready_timeout": 90,
    "product_attempts": 2,
    "retry_delays": [3, 6, 12, 20],
}
```

Also inspect `build_ui()` and assert the login section, both button labels, and the read-only status exist.

- [ ] **Step 2: Run consistency tests and verify RED**

```powershell
uv run --with pytest python -m pytest tests/test_setup_wizard.py tests/test_webui_gemini_login.py -v
```

Expected: defaults are absent from current templates.

- [ ] **Step 3: Update both config templates**

Add under `browser` in `config.example.yaml` and `webui.DEFAULT_CONFIG`:

```yaml
  network_attempts: 5
  page_ready_timeout: 90
  product_attempts: 2
  retry_delays: [3, 6, 12, 20]
```

- [ ] **Step 4: Update README workflow and troubleshooting**

Document this exact user order:

1. Open `API 与模型`.
2. Click `打开 Gemini 登录浏览器`.
3. Complete Google login/account verification.
4. Click `检查登录并关闭浏览器` and wait for success.
5. Start a `gemini_browser` task.

Explain that Spanish UI is supported, Excel output language is independent, weak-network retries are bounded, and permanent certificate errors require checking system time, proxy/VPN, antivirus TLS interception, or corporate certificates. State that disabling TLS verification is not the default solution.

- [ ] **Step 5: Run focused and complete automated verification**

```powershell
uv run --with pytest python -m pytest tests/test_network_retry.py tests/test_gemini_browser_session.py tests/test_webui_gemini_login.py tests/test_medium_priority.py tests/test_model_provider.py tests/test_nvidia_api.py tests/test_low_priority.py -v
uv run --with pytest python -m pytest -q
```

Expected: all tests pass with no Gradio migration warning and no real browser/API calls.

- [ ] **Step 6: Run static checks and no-network UI smoke**

```powershell
uv run python -m compileall -q network_retry.py gemini_browser_session.py app.py webui.py main.py gemini_bot.py gemini_api.py nvidia_api.py model_provider.py lovart_api.py
git diff --check
uv run python -c "import webui; demo=webui.build_ui(); print(type(demo).__name__, len(demo.fns))"
git status --short
```

Expected: exit 0; only intended files are modified.

- [ ] **Step 7: Commit defaults and documentation**

```powershell
git add config.example.yaml webui.py README.md tests/test_setup_wizard.py tests/test_webui_gemini_login.py
git commit -m "docs: explain Gemini login and network recovery"
```

---

### Task 8: Manual helper smoke, version 1.3.0, and OTA artifact

**Files:**
- Modify: `version.py`
- Modify: `version.json`
- Produce ignored artifact: `dist/v1.3.0/Lovart_Auto/`
- Produce ignored artifact: `update.zip`

**Interfaces:**
- Publishes runtime and OTA metadata version `1.3.0`.
- Publishes GitHub asset URL `https://github.com/TokihaMisa/Gemini-lovart-image-automation/releases/download/v1.3.0/update.zip`.

- [ ] **Step 1: Perform a local login-helper smoke without credentials**

Use a temporary config whose profile and runtime paths are under `build/login-smoke/`. Launch the helper, verify `status.json` reaches `waiting_login` or `ready`, then close the browser manually or by close request when ready. Confirm no normal browser profile is touched. Do not enter or record account credentials during automated verification.

- [ ] **Step 2: Run the final verification gate before version changes**

```powershell
uv run --with pytest python -m pytest -q
uv run python -m compileall -q network_retry.py gemini_browser_session.py app.py webui.py main.py gemini_bot.py gemini_api.py nvidia_api.py model_provider.py lovart_api.py
git diff --check
git status --short
```

Expected: all tests pass; compile and diff checks exit 0.

- [ ] **Step 3: Update version metadata to 1.3.0**

Set:

```python
VERSION = "1.3.0"
```

and:

```json
{
  "version": "1.3.0",
  "url": "https://github.com/TokihaMisa/Gemini-lovart-image-automation/releases/download/v1.3.0/update.zip",
  "changelog": "新增 Gemini 浏览器账号登录助手、正式任务登录就绪检查、中英西界面兼容，以及弱网络和 SSL/TLS 瞬时错误的均衡重试与中文诊断。"
}
```

- [ ] **Step 4: Build into versioned directories without deleting unrelated artifacts**

Use fixed workspace paths verified before the command:

```powershell
uv run --no-sync pyinstaller --noconfirm --onedir --windowed --name "Lovart_Auto" --distpath "dist\v1.3.0" --workpath "build\v1.3.0" --specpath "build\v1.3.0" --add-data "D:\image-automation\preamble.txt;." --add-data "D:\image-automation\config.example.yaml;." --add-data "D:\image-automation\.env.example;." --collect-all gradio --collect-all gradio_client --collect-data playwright --collect-data safehttpx --collect-data groovy --hidden-import PIL --hidden-import PIL.Image --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on --collect-data uvicorn app.py
```

- [ ] **Step 5: Create and verify `update.zip`**

Create the archive from the contents of `dist/v1.3.0/Lovart_Auto` so the ZIP root directly contains `Lovart_Auto.exe` and `_internal`. Extract into a new ignored `build/ota_verify_1.3.0` directory. Verify the EXE, `preamble.txt`, `config.example.yaml`, `.env.example`, and packaged `--run-main --help` smoke. Record size and SHA-256.

- [ ] **Step 6: Commit release metadata after artifact verification**

```powershell
git add version.py version.json
git commit -m "release: prepare v1.3.0"
```

- [ ] **Step 7: Record the verified local release candidate**

Write the artifact byte size, SHA-256, extracted EXE smoke result, and local commit SHA to the task report. Do not create a tag, GitHub Release, or push `master` in this task; publication is gated on the cumulative final review below.

---

## Final Review Gate

Before claiming completion:

1. Request a cumulative code review over the implementation range.
2. Fix all Critical and Important findings with new failing tests.
3. Re-run:

```powershell
uv run --with pytest python -m pytest -q
uv run python -m compileall -q network_retry.py gemini_browser_session.py app.py webui.py main.py gemini_bot.py gemini_api.py nvidia_api.py model_provider.py lovart_api.py
git diff --check
git status --short
```

4. If and only if the reviewer says the branch is ready and the verification commands pass, check `gh auth status` and confirm `v1.3.0` does not exist, then publish in this order:

```powershell
git tag -a v1.3.0 -m "Lovart Auto v1.3.0"
git push origin v1.3.0
gh release create v1.3.0 "D:\image-automation\update.zip#update.zip" --verify-tag --title "v1.3.0" --notes "新增 Gemini 浏览器账号登录助手、正式任务登录就绪检查、中英西界面兼容，以及弱网络和 SSL/TLS 瞬时错误的均衡重试与中文诊断。"
```

Verify the live asset name, byte size, and URL exactly match `version.json`. Only then merge the feature branch to `master`, re-run the full suite on the merged result, and push `master` so OTA metadata becomes visible after the asset exists.

5. Read remote `master/version.json`, release asset metadata, and remote `master` SHA. Report exact test/subtest counts, login-helper smoke state, artifact size/SHA-256, Release URL, remote OTA version, and any remaining external account action.
