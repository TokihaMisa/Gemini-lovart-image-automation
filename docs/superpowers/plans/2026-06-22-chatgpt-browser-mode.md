# ChatGPT Browser Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gpt_browser` prompt source that uses a persistent Playwright browser profile to create a new temporary ChatGPT conversation per product, upload images, send prompts, and return the response to the existing Lovart pipeline.

**Architecture:** Add a focused `ChatGPTBot` adapter implementing the same `generate_prompt` and `advise_lovart_confirmation` interface used by the current pipeline. Generalize the existing browser launcher only enough to select Gemini or ChatGPT URL, login wording, and adapter; preserve existing output/status field names for compatibility.

**Tech Stack:** Python 3.11+, Playwright sync API, argparse, Gradio, unittest/pytest-compatible tests, YAML.

---

## File map

- Create `chatgpt_bot.py`: ChatGPT-only DOM/session/upload/reply adapter.
- Create `tests/test_chatgpt_browser.py`: focused behavioral tests for ChatGPT automation and source routing.
- Modify `main.py`: expose `gpt_browser`, select the adapter, and make browser launch provider-aware.
- Modify `webui.py`: expose `gpt_browser` in the prompt-source dropdown and default config text.
- Modify `config.example.yaml`: document ChatGPT browser timeouts and URL.
- Modify `README.md`: document the new source and persistent-login behavior.

### Task 1: ChatGPT prompt-generation contract

**Files:**
- Create: `chatgpt_bot.py`
- Create: `tests/test_chatgpt_browser.py`

- [ ] **Step 1: Write the failing ordered-flow test**

Create `tests/test_chatgpt_browser.py` with an ordered test that subclasses the wished-for adapter and records calls:

```python
import os
import tempfile
import unittest
from pathlib import Path

from chatgpt_bot import ChatGPTBot


class ChatGPTBrowserTests(unittest.TestCase):
    def test_generate_prompt_uses_fresh_temporary_chat_and_waits_in_order(self):
        events = []

        class FakePage:
            def goto(self, url, wait_until="domcontentloaded", timeout=30000):
                events.append(("goto", url))

            def wait_for_timeout(self, milliseconds):
                events.append(("wait", milliseconds))

        class FakeLogger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

        class OrderedBot(ChatGPTBot):
            def _start_new_chat(self):
                events.append("new_chat")
                return True

            def _enable_temporary_chat(self):
                events.append("temporary_chat")
                return True

            def _response_count(self):
                return len([event for event in events if event == "wait_reply"])

            def _send_message(self, text):
                events.append("send_preamble" if text == "preamble" else "send_product")

            def _wait_for_reply(self, previous_response_count=None):
                events.append("wait_reply")

            def _upload_images(self, image_paths):
                events.append(("upload", tuple(image_paths)))
                return True

            def _get_last_response(self):
                return "generated prompt " * 30

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            Path("preamble.txt").write_text("preamble", encoding="utf-8")
            try:
                bot = OrderedBot(FakePage(), {"chatgpt": {"base_url": "https://chatgpt.com"}}, FakeLogger())
                result = bot.generate_prompt("产品", "Portuguese", "卖点", ["a.png"], product_id="SKU-1")
            finally:
                os.chdir(cwd)

        self.assertTrue(result.startswith("generated prompt"))
        self.assertEqual(
            [event for event in events if event == "new_chat" or event == "temporary_chat" or isinstance(event, str) and event.startswith("send_") or event == "wait_reply" or isinstance(event, tuple) and event[0] == "upload"],
            ["new_chat", "temporary_chat", "send_preamble", "wait_reply", ("upload", ("a.png",)), "send_product", "wait_reply"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the ordered-flow test and verify RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py::ChatGPTBrowserTests::test_generate_prompt_uses_fresh_temporary_chat_and_waits_in_order -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'chatgpt_bot'`.

- [ ] **Step 3: Implement the minimal public adapter and orchestration**

Create `chatgpt_bot.py` with `ChatGPTBot.__init__`, `generate_prompt`, and stubbed private methods that raise `NotImplementedError`. The orchestration must:

```python
def generate_prompt(self, product_name_cn, language, selling_points, image_paths, product_id=None, image_size=""):
    product_id = product_id or product_name_cn
    self.page.goto(self.cfg.get("base_url", "https://chatgpt.com"), wait_until="domcontentloaded", timeout=30000)
    self.page.wait_for_timeout(3000)
    if not self._start_new_chat():
        raise RuntimeError("ChatGPT new chat could not be started")
    if not self._enable_temporary_chat():
        raise RuntimeError("ChatGPT temporary chat could not be enabled")
    preamble = get_resource_path("preamble.txt").read_text(encoding="utf-8")
    previous = self._response_count()
    self._send_message(preamble)
    self._wait_for_reply(previous_response_count=previous)
    if image_paths and not self._upload_images(image_paths):
        raise RuntimeError("ChatGPT image upload did not complete")
    prompt = build_design_prompt(product_name_cn, language, selling_points, image_size=image_size)
    previous = self._response_count()
    self._send_message(prompt)
    self._wait_for_reply(previous_response_count=previous)
    result = self._get_last_response().strip()
    if len(result) < 200:
        raise RuntimeError("ChatGPT response was too short")
    out_dir = product_output_dir(product_id)
    (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
    update_status(out_dir, "gemini_done", gemini_chars=len(result))
    return result
```

Wrap the body in `try/except` so exceptions call `_save_debug_snapshot(product_id, "exception")` before re-raising. Import the same prompt/status utilities used by `GeminiBot`.

- [ ] **Step 4: Run the ordered-flow test and verify GREEN**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py::ChatGPTBrowserTests::test_generate_prompt_uses_fresh_temporary_chat_and_waits_in_order -v`

Expected: PASS.

- [ ] **Step 5: Add and test Lovart confirmation advice**

Add a test that overrides `_response_count`, `_send_message`, `_wait_for_reply`, and `_get_last_response`, returns a valid existing confirmation JSON response, then asserts `decision["decision"]` matches the parser result. Implement `advise_lovart_confirmation(...)` using `build_lovart_confirmation_prompt`, `parse_lovart_confirmation_decision`, and the same status/artifact convention as `GeminiBot`, with filenames retaining `lovart_confirmation_gemini_<round>.txt` for compatibility.

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the public adapter contract**

```powershell
git add chatgpt_bot.py tests/test_chatgpt_browser.py
git commit -m "feat: add ChatGPT browser adapter contract"
```

### Task 2: Fail-closed new and temporary chat controls

**Files:**
- Modify: `chatgpt_bot.py`
- Modify: `tests/test_chatgpt_browser.py`

- [ ] **Step 1: Write failing tests for semantic selectors**

Add fake locator/page tests asserting `_start_new_chat()` tries selectors containing `data-testid`, `New chat`, and `新聊天`, and `_enable_temporary_chat()` tries `Temporary`, `临时`, or `临時`. Add a test where every locator is invisible and assert both methods return `False`.

- [ ] **Step 2: Verify RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k "new_chat or temporary_chat" -v`

Expected: FAIL because the private methods still raise `NotImplementedError`.

- [ ] **Step 3: Implement layered control discovery**

Implement `_click_visible_control(selectors, patterns, label)` to:

1. Try each Playwright selector with `.first.is_visible(timeout=1200)` and `.click(timeout=3000)`.
2. Fall back to `page.evaluate` scanning visible `button`, `[role=button]`, and `a` elements against bilingual regular expressions.
3. Log the chosen method and return a boolean.

Implement `_start_new_chat()` with selectors such as:

```python
selectors = [
    '[data-testid="create-new-chat-button"]',
    'a[aria-label*="New chat"]',
    'button[aria-label*="New chat"]',
    'a[aria-label*="新聊天"]',
    'button[aria-label*="新聊天"]',
]
```

Implement `_enable_temporary_chat()` with semantic selectors/text patterns and then call `_temporary_chat_is_active()`. The active-state check must inspect visible controls and page text for positive signals such as `Temporary Chat`, `临时聊天`, or `臨時聊天`; a click without a positive signal returns `False`.

- [ ] **Step 4: Verify GREEN and fail-closed behavior**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k "new_chat or temporary_chat" -v`

Expected: PASS, including the all-invisible case.

- [ ] **Step 5: Commit session controls**

```powershell
git add chatgpt_bot.py tests/test_chatgpt_browser.py
git commit -m "feat: enforce fresh temporary ChatGPT sessions"
```

### Task 3: Upload, reply waiting, extraction, and debug artifacts

**Files:**
- Modify: `chatgpt_bot.py`
- Modify: `tests/test_chatgpt_browser.py`

- [ ] **Step 1: Write failing upload test**

Add a fake `input[type=file]` locator and assert `_upload_images(["a.png", "b.png"])` calls `set_input_files` once, then `_wait_for_uploads_complete(2)`. Add a retry test where the first attempt fails and the second succeeds.

- [ ] **Step 2: Verify upload tests RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k upload -v`

Expected: FAIL because upload helpers are not implemented.

- [ ] **Step 3: Implement upload and completion detection**

Implement direct file-input upload first and attachment-button/file-chooser fallback second. `_wait_for_uploads_complete(expected_count)` must poll until two consecutive stable observations have attachment/file count at least `expected_count`, no `uploading/processing/正在上传/处理中` text, and an enabled send button. Read retry count and timeouts from `self.cfg`.

- [ ] **Step 4: Write failing reply/extraction tests**

Add tests that:

- `_wait_for_reply(previous_response_count=1)` waits until the assistant-message count becomes `2` and no visible Stop button or busy state remains for three observations.
- `_get_last_response()` returns only the final `[data-message-author-role="assistant"]` text when sidebar/body text is also present.
- `_save_debug_snapshot("SKU/1", "upload failed")` creates sanitized `.png` and `.html` files.

- [ ] **Step 5: Verify reply/extraction tests RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k "reply or response or snapshot" -v`

Expected: FAIL because reply/extraction/snapshot helpers are not implemented.

- [ ] **Step 6: Implement message sending, reply state, extraction, and snapshots**

Use these primary ChatGPT selectors:

```python
EDITOR_SELECTORS = ['#prompt-textarea', 'div[contenteditable="true"]', 'textarea', '[role="textbox"]']
SEND_SELECTORS = ['button[data-testid="send-button"]', 'button[aria-label*="Send"]', 'button[aria-label*="发送"]']
ASSISTANT_SELECTOR = '[data-message-author-role="assistant"]'
```

`_send_message` inserts text into the visible editor and clicks a visible send button, falling back to Enter. `_read_generation_state` reports visible Stop controls, busy/progress elements, and assistant response count. `_get_last_response` uses only the last assistant container and raises if none exists. `_save_debug_snapshot` follows the existing `runs/<run>/browser-debug/<sanitized-id>/` convention.

- [ ] **Step 7: Run all ChatGPT adapter tests GREEN**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -v`

Expected: PASS with no warnings or errors.

- [ ] **Step 8: Commit robust page automation**

```powershell
git add chatgpt_bot.py tests/test_chatgpt_browser.py
git commit -m "feat: automate ChatGPT uploads and replies"
```

### Task 4: Provider-aware browser launcher and CLI routing

**Files:**
- Modify: `main.py`
- Modify: `tests/test_chatgpt_browser.py`

- [ ] **Step 1: Write failing CLI and factory tests**

Add tests asserting:

```python
self.assertEqual(parse_args(["--prompt-source", "gpt_browser"]).prompt_source, "gpt_browser")
```

and that a new pure helper `_browser_provider_spec(config, "gpt_browser")` returns:

```python
{
    "name": "ChatGPT",
    "base_url": "https://chatgpt.com",
    "login_keywords": ["auth", "login"],
    "bot_class": ChatGPTBot,
}
```

Also assert `gemini_browser` still returns `GeminiBot` and the configured Gemini URL.

- [ ] **Step 2: Verify routing tests RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k "prompt_source or provider_spec" -v`

Expected: FAIL because `gpt_browser` is rejected and `_browser_provider_spec` does not exist.

- [ ] **Step 3: Implement minimal CLI and provider factory changes**

In `main.py`:

- Import `ChatGPTBot`.
- Add `gpt_browser` to `--prompt-source` choices.
- Add an interactive `ChatGPT Browser` option in `_choose_prompt_source` and adjust numeric bounds.
- Add `_browser_provider_spec(config, prompt_source)` returning provider label, base URL, login keywords, and bot class.
- Change `_run_browser_flow(..., prompt_source="gemini_browser")` to use the spec for navigation, logging, login messages, and adapter construction.
- Pass `prompt_source` from `main()` into `_run_browser_flow`.

Do not alter API/NVIDIA branches.

- [ ] **Step 4: Run routing tests GREEN**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k "prompt_source or provider_spec" -v`

Expected: PASS.

- [ ] **Step 5: Run existing browser regressions**

Run: `uv run python -m pytest tests/test_medium_priority.py -v`

Expected: PASS; existing `_run_browser_flow` callers continue to default to Gemini.

- [ ] **Step 6: Commit launcher routing**

```powershell
git add main.py tests/test_chatgpt_browser.py
git commit -m "feat: route ChatGPT browser prompt source"
```

### Task 5: WebUI, config, and documentation

**Files:**
- Modify: `webui.py`
- Modify: `config.example.yaml`
- Modify: `README.md`
- Modify: `tests/test_chatgpt_browser.py`

- [ ] **Step 1: Write failing source-surface contract test**

Add a file-content test asserting `gpt_browser` appears in `webui.py`, `config.example.yaml` contains a top-level `chatgpt:` section with `https://chatgpt.com`, and `README.md` describes `gpt_browser` plus temporary chats.

- [ ] **Step 2: Verify surface test RED**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k source_surfaces -v`

Expected: FAIL because the new source is absent from WebUI/config/docs.

- [ ] **Step 3: Update WebUI and default configuration**

Add `gpt_browser` to the Gradio prompt-source choices. Add this block to both `config.example.yaml` and the embedded `DEFAULT_CONFIG` text in `webui.py`:

```yaml
chatgpt:
  base_url: "https://chatgpt.com"
  reply_timeout: 300
  upload_timeout: 120
  upload_attempts: 3
```

- [ ] **Step 4: Update README**

Document:

- `gpt_browser` in supported prompt sources.
- It reuses `browser_profile`.
- The user must log in manually when prompted.
- Each product uses a new temporary chat and the account's current default model.

- [ ] **Step 5: Run surface test GREEN**

Run: `uv run python -m pytest tests/test_chatgpt_browser.py -k source_surfaces -v`

Expected: PASS.

- [ ] **Step 6: Commit user-facing surfaces**

```powershell
git add webui.py config.example.yaml README.md tests/test_chatgpt_browser.py
git commit -m "docs: expose ChatGPT browser mode"
```

### Task 6: Full regression and manual browser smoke test

**Files:**
- Modify only if a failing test exposes an in-scope defect.

- [ ] **Step 1: Run focused and full automated tests**

Run:

```powershell
uv run python -m pytest tests/test_chatgpt_browser.py -v
uv run python -m pytest -v
```

Expected: all tests PASS with no new warnings.

- [ ] **Step 2: Run static/import checks**

Run:

```powershell
uv run python -m py_compile chatgpt_bot.py main.py webui.py
git diff --check
```

Expected: both commands exit 0 with no output indicating syntax or whitespace errors.

- [ ] **Step 3: Perform one-product ChatGPT browser smoke test**

With an already logged-in persistent profile, run:

```powershell
uv run python main.py --prompt-source gpt_browser --lovart unlimited --limit 1 --no-resume
```

Verify visibly that ChatGPT opens a new chat, enables temporary chat, receives the preamble, receives the uploaded files and product prompt, and produces a response before Lovart processing begins. If login is required, complete login manually when prompted.

- [ ] **Step 4: Inspect artifacts and status**

Verify the processed product contains `gemini_prompt.txt`, its `status.json` reached `gemini_done` before Lovart states, and the current run summary reports a non-empty `gemini_chars`. On failure, verify `runs/<run>/browser-debug/<SKU>/` contains both PNG and HTML diagnostics.

- [ ] **Step 5: Commit only smoke-test fixes, if any**

```powershell
git add chatgpt_bot.py main.py webui.py config.example.yaml README.md tests/test_chatgpt_browser.py
git commit -m "fix: stabilize ChatGPT browser smoke flow"
```

Run this commit step only when the smoke test required an in-scope source fix; otherwise leave the verified tree unchanged.

- [ ] **Step 6: Review final scope**

Run: `git status --short` and `git log --oneline -8`.

Expected: no unintended files, credentials, browser profiles, generated outputs, or cache artifacts are staged or committed.
