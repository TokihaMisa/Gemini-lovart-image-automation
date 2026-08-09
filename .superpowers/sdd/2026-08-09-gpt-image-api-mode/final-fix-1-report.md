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
