# Final async-image release safety fix report

Base: `bc540a2`

## Review findings and TDD evidence

1. **Unknown paid POST result blocks resubmission**
   - RED: create 400/429/503, timeout, malformed JSON, invalid/missing task ID retained ordinary retryable/error classifications; pipeline restart created again.
   - GREEN: every result after the paid POST begins but before a validated task ID is available raises stable `submission_unknown`. Support/detail checkpoints and failed-retry classification permanently block normal resume, compensation queues, and infinite retry; only explicit `resume=False` restarts.
   - Pipeline regressions cover 429, 503, malformed JSON, missing ID, and generic 4xx with zero POSTs on restart.

2. **Two-stage durable submission marker**
   - RED: no callback or durable checkpoint existed before opening the POST; process crash or task-ID callback persistence failure left a resubmittable `running` checkpoint.
   - GREEN: `submission_callback` is invoked synchronously after local preflight/body construction and before opening the paid POST. Providers atomically persist `submission_unknown`; the first validated task snapshot atomically replaces it. Support/detail crash and simulated durable task-ID write failure tests prove restart remains blocked without pretending to recover a missing provider task ID.

3. **Protocol-normalized gateway identity**
   - RED: provider identities used the generic normalized URL while endpoint composition removed optional trailing `/v1`, so equivalent spellings invalidated live tasks.
   - GREEN: provider execution identity uses `_protocol_base_url`, identical to endpoint composition. Support and detail live-task tests switch host ↔ host `/v1`, resume the saved task ID, and make zero create POSTs; different gateways remain distinct.

4. **Hard GET deadline without residual workers/sockets**
   - RED: a daemon `gpt-image-status-get` worker survived the invocation deadline.
   - GREEN: each API instance serializes status workers with one lock, uses a non-daemon worker, tracks both the active response and underlying `HTTPSConnection`, closes response/connection/opener at the deadline, and joins the worker before returning. The deadline regression asserts no named worker remains. Result-download SSRF/DNS pinning/TLS/peer validation code was not weakened.

5. **Central recursive sanitizer**
   - RED: case-changed and percent-encoded API keys/task IDs survived transport callbacks and nested `cost` structures.
   - GREEN: one bounded recursive sanitizer covers mappings, sequences, object representations, transport snapshots, provider checkpoints/errors, and WebUI text. It detects case-insensitive and up-to-three-level percent-encoded variants while preserving benign encoded text. Exact task IDs remain only in the resumable checkpoint field; display uses the existing safe suffix/hash token. Operational result URLs remain internally recoverable while API-key variants are redacted.

6. **Fully decoded atomic image save**
   - RED: a real JPEG that passed Pillow `verify()` but failed `load()` replaced the target; real truncated WebP coverage was also added.
   - GREEN: validation now reopens and fully loads the image before creating/replacing the temporary target. Invalid/truncated bytes leave the previous target unchanged.

7. **Fail-closed post-package verification**
   - RED: no tracked command existed and absent archives silently skipped release integrity validation.
   - GREEN: `scripts/verify_release_artifacts.py --version 1.3.21` requires `update.zip`, `update-v1.3.21.zip`, and `version.json`; checks byte identity, SHA-256, size, version, exact root EXE/config assets, and real updater extraction. Unit tests prove success and required-artifact failure without making clean-checkout tests depend on release files.

## Verification

- RED matrix before production changes: `23 failed, 270 passed, 1 skipped` (all failures matched the seven findings).
- Final focused matrix: `308 passed in 46.35s`.
- Final full suite: `740 passed, 128 subtests passed in 128.90s`.
- `python -m compileall` for changed runtime/release script: exit 0.
- `git diff --check`: exit 0 (only Windows LF→CRLF informational warnings).
- Dry-run with the repository example config: parsed one product and completed with no provider/browser/network construction.

## Release state / concerns

- Production code changed after local release commit `bc540a2`; existing `dist-v1.3.21`, `update-v1.3.21.zip`, `update.zip`, and current `version.json` hash/size are therefore **stale** and must be rebuilt/recomputed by the controller before push/tag/release.
- No version, release hash, ZIP, tag, remote branch, or GitHub release was modified in this fix wave.
- Code concerns remaining: none identified after focused and full verification.

Commit: `78b3ce3` (`fix: close async image release safety gaps`).
