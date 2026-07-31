# Gemini Responsive Temporary Chat Compatibility

## Goal

Keep Gemini browser automation working across wide and compact responsive
headers where the temporary-chat control collapses into an icon beside the
account avatar.

## Design

Temporary chat remains preferred. The bot first uses the
existing text and accessibility selectors, then inspects metadata on nested
elements, and finally checks tooltips revealed by hovering visible header
controls. A candidate is clicked only when its own metadata, descendant
metadata, or visible tooltip matches a known temporary-chat label.

If no safe match is found, the default behavior is to stop the product so no
persistent chat history is created. A Gemini browser setting can explicitly
allow regular-chat fallback; when enabled, the bot logs a warning and
continues. The setting is persisted as
`gemini.allow_regular_chat_fallback` and defaults to `false`, including for
older configurations where the field is absent. Thinking-mode selection,
uploads, prompt submission, and response parsing remain unchanged.

## Safety

The fallback never clicks an unlabeled control based only on screen
coordinates or proximity to the account avatar. This avoids accidentally
opening the Google account or apps menu.

## Verification

Regression tests cover nested temporary-chat metadata, hover-only tooltips,
the classic selectors, strict default behavior, enabled regular-chat
fallback, and settings persistence. The complete Python test suite must pass.
