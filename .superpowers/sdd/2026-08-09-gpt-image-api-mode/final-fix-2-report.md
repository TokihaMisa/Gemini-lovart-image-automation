# Final Fix 2 Report

Base commit: `dcfa472b1c489165b202314c76b4d332f6705b0a`

## Outcome

Implemented only the final-review fixes for dry-run startup ordering, GPT Image credential ownership and migration, mixed-route Lovart project retention, GPT model reporting/results upserts, and provider-neutral prompt copy. No SSRF or download transport changes were made.

## RED evidence

After adding the regressions and before changing production code:

```text
13 failed, 2 passed, 150 deselected in 7.37s
```

The failures covered:

- all four support/detail provider combinations reading credential environment state before `--dry-run` returned;
- Lovart support project state being cleared after successful GPT detail generation;
- the legacy YAML GPT key being accepted at runtime and retained by WebUI persistence;
- immediate GPT failure reporting an empty/stale model;
- ordinary blank-model results upserts retaining an earlier provider model;
- provider-neutral prompt and settings preview still containing Lovart-specific downstream wording and no explicit no-fallback rule.

The credential sentinel used by the regressions is intentionally not reproduced in test output or this report.

## GREEN evidence

Focused startup, WebUI, routing, results, prompt, provider, and GPT API tests:

```text
243 passed, 8 subtests passed in 31.66s
```

Full suite on the final tree:

```text
530 passed, 118 subtests passed in 129.15s (0:02:09)
```

Compile and diff checks:

```text
uv run --no-sync python -m py_compile main.py utils.py image_providers.py webui.py prompt_settings.py tests/image_test_helpers.py tests/test_image_provider_routing.py tests/test_high_priority.py tests/test_webui_model_settings.py tests/test_prompt_settings.py
PASS

git diff --check
PASS

git diff --exit-code -- openai_image_api.py
PASS (no SSRF/download transport diff)
```

The first full-suite invocation used a 120-second harness limit and timed out without reporting a test failure. It was rerun with a sufficient bound and produced the passing full-suite result above.

## Files changed

Production:

- `main.py`
- `utils.py`
- `image_providers.py`
- `webui.py`
- `prompt_settings.py`

Tests:

- `tests/image_test_helpers.py`
- `tests/test_image_provider_routing.py`
- `tests/test_high_priority.py`
- `tests/test_webui_model_settings.py`
- `tests/test_prompt_settings.py`

Report:

- `.superpowers/sdd/2026-08-09-gpt-image-api-mode/final-fix-2-report.md`

## Self-review

- Dry-run parses config/routing and products without loading `.env`, performing Lovart credential validation, choosing a prompt provider, constructing image/prompt clients, launching a browser, or creating/validating a project. Tests clear the environment and make every external/provider constructor fail fast if reached.
- Production GPT Image client construction reads only `OPENAI_IMAGE_API_KEY`; the legacy YAML member is ignored. Dedicated GPT settings saves and route-only launch persistence both remove legacy `openai_image.api_key` while preserving unrelated settings.
- GPT detail completion preserves Lovart `project_id`, `project_url`, and legacy support fields only when the selected support provider is Lovart. Non-Lovart support still clears stale project identity.
- GPT detail model identity is persisted before the first screen call from the provider's configured model and survives all-success, partial-failure, and immediate HTTP 401 failure paths. Completed same-provider skips explicitly opt into CSV model preservation; ordinary upserts do not.
- Rendered design prompts and locked-rule previews use the user-selected image provider wording, require final generation only through that provider, and explicitly prohibit switching or fallback.
- The final diff is limited to the listed startup/state/results/prompt files and tests; `openai_image_api.py` and its transport logic are unchanged.

## Concerns

No known functional concerns. Verification on this Windows worktree requires `PYTHONPATH=.` for direct pytest invocation, and the complete suite takes slightly over two minutes.
