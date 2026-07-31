# Gemini Full Health Check Design

## Goal

Add a `Gemini 一键完整体检` button to the Gemini browser account section of
the WebUI. The check must exercise every control required by the production
Gemini browser workflow without sending a prompt or creating a saved chat.

## User Experience

The button appears beside the existing Gemini login controls under
`API 与模型 -> Gemini 浏览器账户`.

Clicking it starts an isolated browser process and streams a checklist into a
read-only result panel:

- `✅ 正常`: the control was found and its expected state transition completed.
- `❌ 异常`: the control or transition failed, with a short actionable reason.
- `⚠️ 跳过`: a prerequisite failed, so the dependent check was not attempted.

The button is disabled while a check is running. The final summary reports the
number of passed, failed, and skipped checks.

## Checks

The process performs these checks in order:

1. Browser executable and configured profile are available.
2. The profile lock can be acquired.
3. Gemini navigation succeeds and the page reaches `READY`.
4. The account is logged in.
5. The prompt editor is visible.
6. The temporary chat control is found and entering temporary chat succeeds.
7. The mode menu opens.
8. A non-Lite Flash model is found and selected.
9. Extended thinking is found and selected.
10. The upload/tools control is visible.
11. A generated blank PNG uploads and produces a stable attachment preview.
12. The send button becomes available, but is not clicked.

If temporary chat cannot be entered, the health check stops before uploading so
it cannot create a normal chat record. The regular-chat fallback setting is
ignored by diagnostics.

## Architecture

### WebUI

`webui.py` adds the button, a read-only multiline result component, and a
generator callback that streams status updates.

The callback starts the packaged executable with a dedicated
`--gemini-health-check` command. Source mode launches `app.py` with the same
argument. Output is emitted as newline-delimited JSON so the WebUI can update
the checklist without parsing human console text.

### Health Check Runner

A focused module, `gemini_health_check.py`, owns:

- the ordered check definitions;
- result records and summary rendering;
- browser startup and profile-lock lifecycle;
- creation and deletion of the blank PNG;
- guaranteed browser cleanup.

It reuses production helpers from `gemini_browser_session.py` and
`gemini_bot.py` rather than duplicating selectors. Small read-only probe methods
may be added to `GeminiBot` where production methods currently combine lookup
and mutation.

### Process Isolation

The health check acquires the same profile owner lock used by formal browser
tasks and the login helper. If the profile is busy, it reports one failure and
does not terminate or interfere with the owning process.

The browser uses the configured executable and `browser_profile`. A fresh page
is opened for the check and the entire browser context is closed in `finally`.

## Safety

- The check never inserts or sends prompt text.
- It only uploads a program-generated blank PNG.
- Upload is attempted only after temporary chat is confirmed.
- The send button is inspected for visibility/enabled state but never clicked.
- No screenshots, DOM dumps, account names, cookies, URLs with query strings,
  or local user paths are written to the result.
- Temporary files are created under a temporary directory and deleted on exit.
- Existing profile locks are respected.

## Error Handling

Each check produces a bounded public message. Raw Playwright errors remain in
local debug logging and are not returned to the WebUI.

A failed prerequisite marks dependent checks as skipped. Independent cleanup
always runs. Browser launch, TLS, login, page readiness, control mismatch,
upload failure, and profile-busy conditions receive distinct result messages.

## Testing

Automated tests cover:

- ordered pass/fail/skip aggregation;
- profile-busy behavior;
- no upload after temporary-chat failure;
- blank image cleanup;
- send-button inspection without clicking;
- subprocess command construction in source and packaged modes;
- WebUI event wiring and streamed rendering;
- existing Gemini selector and upload tests as regression coverage.

A real-browser smoke check validates that the runner enters temporary chat,
selects Flash extended mode, uploads the blank PNG, observes the attachment,
and exits without sending.

## Acceptance Criteria

- One click runs the complete isolated check and streams progress.
- Every production-critical Gemini control has an explicit result.
- No normal chat history or prompt message is created.
- The browser and temporary image are cleaned up on success and failure.
- A busy profile is reported without disrupting another task.
- Existing application tests remain green.
