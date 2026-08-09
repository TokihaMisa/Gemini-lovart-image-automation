# Final Fix 1 Report: detail lifecycle, invalidation, and resume

## Status

Implemented Final Fix 1 only, from base commit `e488046f746426ce4d922b43bc4ea87ed2a963b5`.

The completed behavior now:

- snapshots the product detail target at product start, before support-provider work;
- persists a deterministic SHA-256 detail-input fingerprint over both active image providers, target count, normalized prompt settings, relevant product fields (including image size and reference-image semantics), and role-preserving content hashes for product/support/accessory/dimension/reference inputs;
- invalidates stale provider-neutral/Lovart prompts, GPT checkpoints and canonical detail outputs, completion/partial/failure/model/artifact state when those inputs change, while preserving identical-input resume;
- preserves the one-time legacy completed all-Lovart migration path;
- stops a GPT detail set immediately after the first screen exhausts API retries and resumes sequentially from failed/missing work;
- carries `resume` into `DetailSetRequest`; `resume=False` clears prior GPT checkpoints and regenerates the configured set;
- reconciles only a `running` checkpoint with the current fingerprint/index and a fully valid canonical `gpt_image/detail/NN.png`, avoiding a duplicate paid call after an atomic-save/pre-done crash.

## Strict TDD evidence

### RED

Behavior tests and changed expectations were written before production code.

Command:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_image_providers.py tests/test_image_provider_routing.py -q
```

Observed result before implementation:

```text
10 failed, 45 passed in 11.53s
```

The failures demonstrated the intended missing behavior:

- `DetailSetRequest` and checkpoints had no fingerprint/resume contract;
- GPT generation continued to screen 3 after screen 2 failed;
- identical inputs had no persisted fingerprint;
- changed support content/provider did not invalidate prompt/checkpoint state;
- the target snapshot was absent after a support failure;
- `resume=False` reused paid GPT checkpoints;
- a saved canonical output could not be reconciled from a matching `running` checkpoint.

The first attempted RED command used the system Python and returned `No module named pytest`; it did not exercise code and is not counted as behavioral RED evidence. The repository `.venv` was then used for all test evidence.

### GREEN

Focused provider/routing lifecycle suite after implementation:

```text
55 passed in 13.97s
```

Final focused rerun after self-review:

```text
55 passed in 13.95s
```

Focused retry/API/snapshot suite:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_openai_image_api.py tests/test_failed_retry_queue.py tests/test_image_generation.py -q
54 passed in 1.39s
```

Final full suite:

```text
.\.venv\Scripts\python.exe -m pytest -q
505 passed, 118 subtests passed in 128.99s (0:02:08)
```

An earlier full-suite invocation was terminated by the command wrapper at 120 seconds without a pytest failure result. A longer-timeout rerun completed with `505 passed, 118 subtests passed in 135.50s`; the final full run above verifies the exact self-reviewed state.

## Files changed

- `image_generation.py`
  - deterministic detail-input fingerprint construction;
  - optional snapshot replacement for explicit no-resume reprocessing.
- `image_providers.py`
  - `DetailSetRequest.resume` and `input_fingerprint`;
  - fingerprint-bound checkpoints and completed-path reads;
  - running/canonical crash reconciliation;
  - stop-after-first-exhaustion and no-resume checkpoint restart.
- `main.py`
  - snapshot moved before support-provider calls;
  - detail lifecycle preparation/invalidation before prompt/detail generation;
  - fingerprint passed into the provider request;
  - legacy all-Lovart completion preserved.
- `tests/image_test_helpers.py`
  - external-call recording, support-failure injection, and resume control.
- `tests/test_image_providers.py`
  - stop/resume, no-resume, and crash-reconciliation behavior tests with valid images and paid-call counts.
- `tests/test_image_provider_routing.py`
  - support-content change, provider switch, identical-input reuse, early snapshot, and pipeline no-resume tests; updated first-failure expectations.

## Self-review

- Fingerprint canonicalization uses sorted JSON keys and streamed file SHA-256 hashes; paths themselves do not make identical contents appear different, while image roles/order remain significant.
- The snapshotted target replaces the mutable global count inside the fingerprint and prompt settings, so a support failure followed by a setting change retains the original target.
- Invalidation removes only detail prompt files and files inside the canonical GPT detail output directory; support artifacts remain available for reuse.
- Invalidation clears checkpoints, detail outputs/status counts, failed indexes, completion/partial flags, terminal failure flags/reason, model/artifact fields, and stale prompt/result/done stage flags.
- A missing fingerprint on a legacy completed task is accepted without regeneration only when both active providers are Lovart; provider changes cannot take that compatibility path.
- Reconciliation requires all of: current resume policy, checkpoint state `running`, matching input fingerprint/index, and a fully decodable canonical image. A valid file without that matching checkpoint still triggers the API call.
- The GPT loop breaks on the first exhausted screen; later screens receive no call in that run. Earlier `done` checkpoints remain valid and the next run processes failed/missing indexes in order.
- No Final Fix 2/3 topics (dry-run, YAML scrubbing, project state, prompt copy, DNS transport, etc.) were changed.

## Verification and concerns

- `py_compile` passed for all three modified production modules and three modified Python test files.
- `git diff --check` passed.
- No known functional concerns remain within Final Fix 1 scope.

## Review round 1: provider execution settings and paid-prompt binding

### Status

Implemented review round 1 as a separate strict-TDD change on top of commit `81e8e522581d55ec931c0c814c0435a79ab104fc`.

- The upstream detail-input fingerprint schema is now version 2 and includes an explicit, non-secret detail-provider execution-settings object.
- OpenAI execution settings include only normalized `base_url`, `model`, and `resolution`, matching the non-secret configuration that changes the actual Images edit request. The API key property is never read, enumerated, copied, or serialized.
- Lovart execution settings include the selected image model(s), model-selection policy, preferred/included tools, reasoning mode, tool names, and fast/unlimited run mode actually used by detail generation.
- Every GPT detail checkpoint state (`running`, `done`, and `failed`) now stores a deterministic per-screen paid-prompt hash over the exact prompt after the same aspect-ratio composition used by the API, plus screen index and target count.
- Done reuse and post-save/pre-done reconciliation require both the current upstream fingerprint and the exact current screen prompt hash. Legacy checkpoints without a prompt hash regenerate once.
- Completed GPT products are not skipped until the saved detail prompt is present, parseable, composed into the final paid prompts, and matched against all checkpoint prompt hashes.

### Strict TDD evidence

RED command:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_image_providers.py tests/test_image_provider_routing.py -q
```

Observed before production changes:

```text
12 failed, 53 passed in 15.08s
```

The failures demonstrated that provider settings did not affect the fingerprint, valid hashless checkpoints were reused, missing/edited prompt files bypassed regeneration through the completed-product skip, no exact paid-prompt hash existed, and wrong-fingerprint running state was not covered by a production-shaped checkpoint test.

Focused GREEN after implementation:

```text
65 passed in 17.30s
```

Expanded settings/retry/provider GREEN:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_image_generation.py tests/test_openai_image_api.py tests/test_lovart_unlimited_guard.py tests/test_low_priority.py tests/test_failed_retry_queue.py -q
84 passed, 3 subtests passed in 1.84s
```

Full suite GREEN:

```text
.\.venv\Scripts\python.exe -m pytest -q
515 passed, 118 subtests passed in 129.16s (0:02:09)
```

### Review-round regressions

- OpenAI model change invalidates prompt, checkpoints, canonical outputs, and completion state before regenerating the paid set.
- OpenAI resolution change performs the same invalidation; identical execution settings remain stable.
- Lovart selected image-model/tool/mode change invalidates the prompt and prior completion before invoking Lovart again.
- Provider settings serialization is allowlisted and a real `OpenAIImageAPIConfig` test proves the API key is absent.
- A deleted detail prompt with an identically regenerated nondeterministic plan reuses exact matching prompt-bound checkpoints without paid calls.
- A deleted detail prompt with changed regenerated content makes paid calls for the changed set.
- A valid edited saved prompt regenerates only screens whose final composed prompt hash changed.
- A valid legacy `done` checkpoint without a prompt hash regenerates once.
- A `running` checkpoint with a valid canonical image but the wrong upstream fingerprint cannot reconcile and makes the paid call.

### Self-review and verification

- Prompt hashing calls the same `append_aspect_instruction` function used immediately before the network request, avoiding a hash of an earlier plan fragment.
- Screen index and snapshotted target count are included in the prompt-hash payload; model/resolution/base URL remain bound by the upstream fingerprint.
- OpenAI execution settings access only three named non-secret attributes and never introspect the configuration object.
- Lovart settings use an explicit allowlist and JSON-safe canonical values; arbitrary bot/config fields are not serialized.
- Missing or unparsable GPT prompt files deliberately prevent completed-product skip but do not immediately discard checkpoints; exact hashes after regeneration determine safe reuse.
- `py_compile` passed for all modified production/test Python files, and `git diff --check` passed.
- No Final Fix 2/3 concerns were changed. No known functional concerns remain in this review scope.
