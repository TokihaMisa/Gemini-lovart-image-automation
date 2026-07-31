# Gemini Responsive Temporary Chat Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support the compact responsive-header temporary-chat icon and provide an opt-in regular-chat fallback when the control cannot be safely identified.

**Architecture:** Extend `GeminiBot._start_temporary_chat` with two bounded discovery strategies after the existing selectors: nested metadata matching and hover-tooltip matching. Gate regular-chat continuation behind the persisted, default-false `gemini.allow_regular_chat_fallback` setting.

**Tech Stack:** Python 3.14, Playwright sync API, unittest/pytest.

## Global Constraints

- Never click an unlabeled top-bar control by position alone.
- Preserve existing Chinese, English, and Spanish selectors.
- Do not change thinking-mode, upload, or response parsing behavior.

---

### Task 1: Add compact-header DOM regression coverage

**Files:**
- Modify: `tests/test_medium_priority.py`

**Interfaces:**
- Consumes: `GeminiBot._start_temporary_chat() -> bool`
- Produces: Regression tests for hover-only tooltips

- [x] **Step 1: Write failing tests using real Playwright pages**
- [x] **Step 2: Run the two tests and confirm they fail because no control is discovered**
- [x] **Step 3: Keep fixtures limited to the observable DOM variants**

### Task 2: Add an opt-in regular-chat fallback

**Files:**
- Modify: `tests/test_medium_priority.py`
- Modify: `gemini_bot.py`

**Interfaces:**
- Consumes: `_start_temporary_chat() -> bool`
- Produces: `_generate_prompt_once()` continues only when the setting is enabled

- [x] **Step 1: Cover strict default behavior and explicitly enabled fallback**
- [x] **Step 2: Run the tests and confirm the unconditional fallback fails strict mode**
- [x] **Step 3: Gate the warning and continuation behind the setting**
- [x] **Step 4: Run the regression tests and confirm both paths pass**

### Task 3: Implement safe compact-header discovery

**Files:**
- Modify: `gemini_bot.py`
- Test: `tests/test_medium_priority.py`

**Interfaces:**
- Produces: `_click_temporary_chat_via_hover_tooltip() -> bool`

- [x] **Step 1: Aggregate known metadata from clickable descendants**
- [x] **Step 2: Hover a bounded set of visible top-area controls and inspect visible tooltips**
- [x] **Step 3: Click only candidates whose tooltip matches `TEMPORARY_CHAT_TERMS`**
- [x] **Step 4: Run the focused Gemini tests**
- [x] **Step 5: Run the complete test suite and compile check**

### Task 4: Expose and persist the browser setting

**Files:**
- Modify: `webui.py`
- Modify: `config.example.yaml`
- Test: `tests/test_webui_model_settings.py`

**Interfaces:**
- Produces: `save_gemini_browser_settings(...) -> str`
- Persists: `gemini.allow_regular_chat_fallback: false`

- [x] **Step 1: Add failing tests for the default, UI control, and persistence**
- [x] **Step 2: Add a checkbox to the Gemini browser account settings**
- [x] **Step 3: Save the value without replacing unrelated configuration**
- [x] **Step 4: Run focused UI and persistence tests**
