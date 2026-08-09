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

## Review round 2: stale canonical crash boundary and Lovart unlimited order

### Status

Implemented review round 2 as a separate strict-TDD change on top of commit `598e34a2d4e41122ff01f0e0b32bff1288c1c580`.

- Before publishing a new current-identity `running` checkpoint, GPT detail generation now removes the canonical `gpt_image/detail/NN.png` when the prior checkpoint's upstream fingerprint or paid-prompt hash differs.
- `resume=False` always removes an existing canonical output before writing the new `running` checkpoint.
- The enforced order is: inspect prior checkpoint identity, remove stale canonical output, write current `running`, then call the paid API. A crash after `running` but before a new API output therefore cannot reconcile the old image.
- Lovart's explicit non-secret execution settings now include the ordered `_configured_unlimited_models` values and a boolean indicating whether that configured selection is active. The existing `run_mode` field remains the directly coupled fast/unlimited selector.

### Strict TDD evidence

RED command:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_image_providers.py tests/test_image_provider_routing.py -q
```

Observed before production changes:

```text
4 failed, 64 passed in 18.72s
```

The failures proved both wrong-upstream-fingerprint and wrong-prompt-hash crash windows retained a stale canonical image, while Lovart settings omitted the ordered configured unlimited model list and reordering that list did not invalidate completion.

After the production change, three older provider fixtures failed because they returned a path without performing the real API contract's canonical write after the new pre-call unlink. Those fixtures were corrected to write valid images during the mocked successful call; this was a test realism correction, not a production behavior relaxation.

Final focused GREEN:

```text
68 passed in 19.81s
```

Expanded Lovart/OpenAI/retry GREEN:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_lovart_unlimited_guard.py tests/test_openai_image_api.py tests/test_failed_retry_queue.py -q
54 passed in 1.29s
```

Full suite GREEN:

```text
.\.venv\Scripts\python.exe -m pytest -q
518 passed, 118 subtests passed in 143.72s (0:02:23)
```

### Review-round regressions and self-review

- Wrong upstream fingerprint: a valid stale canonical and mismatched `running` checkpoint are replaced by a current `running`; an injected pre-output crash leaves no canonical, and the next resume calls the API.
- Wrong paid-prompt hash: the same crash-boundary proof applies when only the prompt hash differs.
- The post-save/pre-done success path remains intact: a matching current `running` checkpoint with a newly saved valid canonical image still reconciles without a duplicate paid call.
- No-resume uses the same pre-checkpoint canonical removal unconditionally.
- Reordering otherwise identical Lovart unlimited models changes the upstream detail fingerprint and forces prompt/detail regeneration.
- Lovart serialization remains allowlisted: only existing selected tool/model/mode fields, ordered configured unlimited models, the configured-selection boolean, and run mode are included. No bot/config dump or secret access was added.
- `py_compile` passed for the modified provider and two modified test modules; `git diff --check` passed.
- No other behavior or Final Fix 2/3 concern was changed. No known functional concerns remain in this review scope.

## Review round 3: mode-accurate Lovart unlimited settings

### Status

Implemented the isolated review round 3 correction on top of commit `4635e7b586e996a9b07dbabd3c80a33d8c9015bf`.

- Fast mode now emits `configured_unlimited_models_selected=false` and omits the configured unlimited model list entirely because that list does not participate in fast-mode execution.
- Unlimited mode emits the ordered configured list only when it is nonempty and actually selected; in that case `configured_unlimited_models_selected=true`.
- Unlimited mode with no configured list emits `selected=false` and omits the list, leaving the existing actual fallback/tool settings to describe execution.

### Strict TDD evidence

RED command:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_image_providers.py tests/test_image_provider_routing.py -q
```

Observed before production changes:

```text
2 failed, 67 passed in 19.27s
```

Both failures proved that fast-mode fingerprints incorrectly included and reacted to reordered unlimited-only configuration.

Focused GREEN:

```text
69 passed in 17.89s
```

The existing unlimited-mode regression remains in the focused suite and proves that a nonempty ordered configured list still invalidates reuse when reordered. The new fast-mode regression proves the same reorder leaves the fingerprint stable, skips the completed product, and makes neither prompt nor Lovart generation calls.

### Verification and scope

- `py_compile` passed for the provider and two focused test modules.
- `git diff --check` passed.
- Per the review instruction, the full suite was not rerun because round 2 completed `518 passed, 118 subtests` and this change is isolated to the Lovart settings projection. No known concerns remain.
