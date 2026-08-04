from __future__ import annotations

import hashlib
import io
import json
import os
import ssl
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import updater
from network_retry import RetryPolicy


VALID_SHA = "a" * 64
REQUIRED_FILES = {
    "Lovart_Auto.exe": b"new executable",
    "_internal/python314.dll": b"python",
    "_internal/VCRUNTIME140.dll": b"runtime",
    "_internal/VCRUNTIME140_1.dll": b"runtime one",
}


class FakeResponse(io.BytesIO):
    def __init__(self, payload, final_url="https://downloads.example.test/resource"):
        super().__init__(payload)
        self.final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def info(self):
        return {"Content-Length": str(len(self.getvalue()))}

    def geturl(self):
        return self.final_url


def make_zip(entries=None, *, corrupt_crc=False):
    stream = io.BytesIO()
    compression = zipfile.ZIP_STORED if corrupt_crc else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(stream, "w", compression) as archive:
        for name, value in (entries or REQUIRED_FILES).items():
            if isinstance(value, zipfile.ZipInfo):
                archive.writestr(value, b"target")
            else:
                archive.writestr(name, value)
    payload = bytearray(stream.getvalue())
    if corrupt_crc:
        marker = b"new executable"
        position = payload.find(marker)
        if position >= 0:
            payload[position] ^= 0x01
    return bytes(payload)


class UpdateCheckTests(unittest.TestCase):
    def manifest(self, **overrides):
        value = {
            "version": "1.3.1",
            "url": "https://downloads.example.test/update.zip",
            "changelog": "safe changes",
            "sha256": VALID_SHA,
            "size": 42,
        }
        value.update(overrides)
        return json.dumps(value).encode()

    def test_uses_master_manifest_with_cache_busting_headers(self):
        requests = []

        def opener(request, data=None, timeout=None):
            requests.append((request, timeout))
            return FakeResponse(self.manifest(version=updater.VERSION))

        result = updater.check_update_details(opener=opener, sleep=lambda _delay: None)

        self.assertEqual(result.status, updater.UpdateStatus.UP_TO_DATE)
        self.assertIn("/master/version.json?", requests[0][0].full_url)
        self.assertEqual(requests[0][0].get_header("Cache-control"), "no-cache")
        self.assertEqual(requests[0][0].get_header("Pragma"), "no-cache")

    def test_https_context_merges_bundled_public_ca_certificates(self):
        empty_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.assertEqual(empty_context.cert_store_stats()["x509_ca"], 0)

        with mock.patch(
            "updater.ssl.create_default_context", return_value=empty_context
        ):
            context = updater._build_https_context()

        self.assertIs(context, empty_context)
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)

    def test_default_update_check_uses_hybrid_https_opener(self):
        response = FakeResponse(self.manifest(version=updater.VERSION))
        open_https = mock.Mock(return_value=response)

        with mock.patch("updater._open_https", open_https):
            result = updater.check_update_details(sleep=lambda _delay: None)

        self.assertEqual(result.status, updater.UpdateStatus.UP_TO_DATE)
        open_https.assert_called_once()

    def test_stdlib_signature_receives_manifest_timeout_as_keyword(self):
        calls = []

        def opener(url, data=None, timeout=None):
            self.assertIsNone(data)
            self.assertEqual(timeout, 10)
            calls.append(url)
            return FakeResponse(self.manifest(version=updater.VERSION))

        result = updater.check_update_details(opener=opener, sleep=lambda _delay: None)

        self.assertEqual(result.status, updater.UpdateStatus.UP_TO_DATE)
        self.assertEqual(len(calls), 1)

    def test_404_is_error_not_latest_and_is_not_retried(self):
        calls = []
        error = HTTPError(
            "https://example.test/private", 404, "private response", {}, None
        )

        def opener(request, data=None, timeout=None):
            calls.append(request.full_url)
            raise error

        try:
            result = updater.check_update_details(
                opener=opener, sleep=lambda _delay: None
            )
        finally:
            error.close()

        self.assertEqual(result.status, updater.UpdateStatus.ERROR)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("private response", result.message)
        self.assertNotIn("example.test", result.message)

    def test_transient_check_retries_with_balanced_policy(self):
        calls, delays = [], []

        def opener(_request, data=None, timeout=None):
            calls.append(1)
            if len(calls) < 5:
                raise ConnectionResetError("api-key=private")
            return FakeResponse(self.manifest(version=updater.VERSION))

        result = updater.check_update_details(opener=opener, sleep=delays.append)

        self.assertEqual(result.status, updater.UpdateStatus.UP_TO_DATE)
        self.assertEqual(len(calls), 5)
        self.assertEqual(delays, [3, 6, 12, 20])

    def test_manifest_rejects_https_to_http_redirect(self):
        result = updater.check_update_details(
            opener=lambda _request, data=None, timeout=None: FakeResponse(
                self.manifest(), final_url="http://mirror.example.test/version.json"
            ),
            sleep=lambda _delay: None,
        )
        self.assertEqual(result.status, updater.UpdateStatus.ERROR)
        self.assertIn("HTTPS", result.message)
        self.assertNotIn("mirror.example.test", result.message)

    def test_semantic_numeric_order_never_offers_downgrade(self):
        for latest in ("1.3.0", "1.2.99", "1.3.1"):
            with self.subTest(latest=latest):
                result = updater.check_update_details(
                    opener=lambda _request, data=None, timeout=None, latest=latest: FakeResponse(
                        self.manifest(version=latest)
                    ),
                    current_version="1.3.1",
                    sleep=lambda _delay: None,
                )
                self.assertEqual(result.status, updater.UpdateStatus.UP_TO_DATE)

        result = updater.check_update_details(
            opener=lambda _request, data=None, timeout=None: FakeResponse(
                self.manifest(version="1.10.0")
            ),
            current_version="1.9.9",
            sleep=lambda _delay: None,
        )
        self.assertEqual(result.status, updater.UpdateStatus.UPDATE_AVAILABLE)

    def test_rejects_malformed_or_incomplete_metadata(self):
        invalid = (
            {"version": "one.two.three"},
            {"url": "http://downloads.example.test/update.zip"},
            {"url": ""},
            {"sha256": "abcd"},
            {"sha256": "g" * 64},
            {"size": 0},
            {"size": -1},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                result = updater.check_update_details(
                    opener=lambda _request, data=None, timeout=None, overrides=overrides: FakeResponse(
                        self.manifest(**overrides)
                    ),
                    sleep=lambda _delay: None,
                )
                self.assertEqual(result.status, updater.UpdateStatus.ERROR)
                self.assertNotIn("downloads.example.test", result.message)

    def test_legacy_wrapper_preserves_four_tuple(self):
        with mock.patch.object(
            updater,
            "check_update_details",
            return_value=updater.UpdateCheckResult(
                updater.UpdateStatus.UPDATE_AVAILABLE,
                version="1.3.2",
                url="https://example.test/update.zip",
                changelog="changes",
                sha256=VALID_SHA,
                size=1,
            ),
        ):
            self.assertEqual(
                updater.check_for_updates(),
                (True, "1.3.2", "https://example.test/update.zip", "changes"),
            )


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_zip(self, payload):
        path = self.root / "update.zip"
        path.write_bytes(payload)
        return path

    def test_valid_archive_extracts_only_after_validation(self):
        destination = self.root / "payload"
        entries = dict(REQUIRED_FILES)
        entries["_internal/"] = b""
        updater.validate_and_extract_update(self.write_zip(make_zip(entries)), destination)
        self.assertEqual((destination / "Lovart_Auto.exe").read_bytes(), b"new executable")

    def test_rejects_corrupt_crc_without_partial_extraction(self):
        destination = self.root / "payload"
        with self.assertRaises(updater.UpdateError):
            updater.validate_and_extract_update(
                self.write_zip(make_zip(corrupt_crc=True)), destination
            )
        self.assertFalse(destination.exists())

    def test_rejects_zip_slip_absolute_drive_and_duplicate_entries(self):
        cases = (
            {"../outside.txt": b"x", **REQUIRED_FILES},
            {"/absolute.txt": b"x", **REQUIRED_FILES},
            {"C:/drive.txt": b"x", **REQUIRED_FILES},
            {"_internal/C:/drive.txt": b"x", **REQUIRED_FILES},
            {"_internal/CON.txt": b"x", **REQUIRED_FILES},
        )
        for index, entries in enumerate(cases):
            with self.subTest(index=index):
                destination = self.root / f"payload-{index}"
                with self.assertRaises(updater.UpdateError):
                    updater.validate_and_extract_update(
                        self.write_zip(make_zip(entries)), destination
                    )
                self.assertFalse(destination.exists())

        duplicate = io.BytesIO()
        with zipfile.ZipFile(duplicate, "w") as archive:
            for name, body in REQUIRED_FILES.items():
                archive.writestr(name, body)
            with self.assertWarns(UserWarning):
                archive.writestr("Lovart_Auto.exe", b"duplicate")
        with self.assertRaises(updater.UpdateError):
            updater.validate_and_extract_update(
                self.write_zip(duplicate.getvalue()), self.root / "duplicate"
            )

    def test_rejects_symlink_reparse_and_unexpected_layout(self):
        symlink = zipfile.ZipInfo("_internal/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        reparse = zipfile.ZipInfo("_internal/reparse")
        reparse.external_attr = 0x400
        cases = (
            {**REQUIRED_FILES, "_internal/link": symlink},
            {**REQUIRED_FILES, "_internal/reparse": reparse},
            {**REQUIRED_FILES, "config.yaml": b"must never ship"},
            {**REQUIRED_FILES, "runs/history.txt": b"user data"},
        )
        for index, entries in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(updater.UpdateError):
                    updater.validate_and_extract_update(
                        self.write_zip(make_zip(entries)), self.root / f"unsafe-{index}"
                    )

    def test_rejects_missing_required_runtime_files(self):
        for missing in REQUIRED_FILES:
            with self.subTest(missing=missing):
                entries = dict(REQUIRED_FILES)
                entries.pop(missing)
                with self.assertRaises(updater.UpdateError):
                    updater.validate_and_extract_update(
                        self.write_zip(make_zip(entries)), self.root / missing.replace("/", "-")
                    )


class DownloadAndInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app_dir = Path(self.temp.name) / "installed"
        self.app_dir.mkdir()
        self.archive = make_zip()

    def tearDown(self):
        self.temp.cleanup()

    def test_download_uses_fresh_app_local_part_and_verifies_before_rename(self):
        seen = []

        def opener(request, data=None, timeout=None):
            seen.append((request.full_url, timeout))
            return FakeResponse(self.archive)

        prepared = updater.prepare_update(
            "https://downloads.example.test/update.zip?signature=private",
            hashlib.sha256(self.archive).hexdigest(),
            len(self.archive),
            app_dir=self.app_dir,
            opener=opener,
            sleep=lambda _delay: None,
        )

        self.assertEqual(prepared.staging_dir.parent, self.app_dir / ".lovart-update")
        self.assertTrue(prepared.archive_path.name.endswith(".zip"))
        self.assertFalse(any(prepared.staging_dir.glob("*.part")))
        self.assertTrue((prepared.payload_dir / "Lovart_Auto.exe").is_file())

    def test_stdlib_signature_receives_download_timeout_as_keyword(self):
        calls = []

        def opener(url, data=None, timeout=None):
            self.assertIsNone(data)
            self.assertEqual(timeout, 30)
            calls.append(url)
            return FakeResponse(self.archive)

        prepared = updater.prepare_update(
            "https://downloads.example.test/update.zip",
            hashlib.sha256(self.archive).hexdigest(),
            len(self.archive),
            app_dir=self.app_dir,
            opener=opener,
            sleep=lambda _delay: None,
        )

        self.assertTrue(prepared.archive_path.exists())
        self.assertEqual(len(calls), 1)

    def test_download_tls_failure_names_github_release_proxy_rule(self):
        detail = "certificate detail must stay private"

        with self.assertRaises(updater.UpdateError) as raised:
            updater.prepare_update(
                "https://github.com/example/project/releases/download/v1/update.zip",
                hashlib.sha256(self.archive).hexdigest(),
                len(self.archive),
                app_dir=self.app_dir,
                opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    ssl.SSLCertVerificationError(detail)
                ),
                sleep=lambda _delay: None,
            )

        message = str(raised.exception)
        self.assertIn("release-assets.githubusercontent.com", message)
        self.assertIn("Clash", message)
        self.assertNotIn(detail, message)

    def test_size_and_hash_mismatch_leave_no_verified_archive(self):
        cases = (
            (len(self.archive) + 1, hashlib.sha256(self.archive).hexdigest()),
            (len(self.archive), "0" * 64),
        )
        for index, (size, digest) in enumerate(cases):
            app_dir = self.app_dir / str(index)
            app_dir.mkdir()
            with self.subTest(index=index), self.assertRaises(updater.UpdateError):
                updater.prepare_update(
                    "https://downloads.example.test/update.zip?secret=value",
                    digest,
                    size,
                    app_dir=app_dir,
                    opener=lambda _request, data=None, timeout=None: FakeResponse(self.archive),
                    sleep=lambda _delay: None,
                )
            self.assertFalse(list((app_dir / ".lovart-update").rglob("*.zip")))

    def test_download_retries_transient_partial_reads_from_scratch(self):
        calls, delays = [], []

        class PartialResponse(FakeResponse):
            def read(self, size=-1):
                if self.tell() > 0:
                    raise ConnectionResetError("C:\\Users\\Private\\secret")
                return super().read(min(size, 8))

        def opener(_request, data=None, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                return PartialResponse(self.archive)
            return FakeResponse(self.archive)

        prepared = updater.prepare_update(
            "https://downloads.example.test/update.zip",
            hashlib.sha256(self.archive).hexdigest(),
            len(self.archive),
            app_dir=self.app_dir,
            opener=opener,
            policy=RetryPolicy(network_attempts=3, retry_delays=(0, 0)),
            sleep=delays.append,
        )
        self.assertTrue(prepared.archive_path.exists())
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [0, 0])

    def test_clean_short_read_retries_from_empty_part(self):
        calls, delays = [], []

        def opener(_request, data=None, timeout=None):
            calls.append(1)
            payload = self.archive[:-7] if len(calls) == 1 else self.archive
            return FakeResponse(payload)

        prepared = updater.prepare_update(
            "https://downloads.example.test/update.zip",
            hashlib.sha256(self.archive).hexdigest(),
            len(self.archive),
            app_dir=self.app_dir,
            opener=opener,
            policy=RetryPolicy(network_attempts=2, retry_delays=(0,)),
            sleep=delays.append,
        )

        self.assertTrue(prepared.archive_path.exists())
        self.assertEqual(calls, [1, 1])
        self.assertEqual(delays, [0])

    def test_explicit_oversize_and_hash_mismatch_do_not_retry(self):
        cases = (
            (self.archive + b"x", hashlib.sha256(self.archive).hexdigest(), len(self.archive)),
            (self.archive, "0" * 64, len(self.archive)),
        )
        for index, (payload, digest, size) in enumerate(cases):
            calls = []
            app_dir = self.app_dir / f"permanent-{index}"
            app_dir.mkdir()

            def opener(_request, data=None, timeout=None, payload=payload):
                calls.append(1)
                return FakeResponse(payload)

            with self.subTest(index=index), self.assertRaises(updater.UpdateError):
                updater.prepare_update(
                    "https://downloads.example.test/update.zip",
                    digest,
                    size,
                    app_dir=app_dir,
                    opener=opener,
                    policy=RetryPolicy(network_attempts=5, retry_delays=(0, 0, 0, 0)),
                    sleep=lambda _delay: None,
                )
            self.assertEqual(calls, [1])

    def test_payload_rejects_https_to_http_redirect(self):
        with self.assertRaises(updater.UpdateError) as raised:
            updater.prepare_update(
                "https://downloads.example.test/update.zip",
                hashlib.sha256(self.archive).hexdigest(),
                len(self.archive),
                app_dir=self.app_dir,
                opener=lambda _request, data=None, timeout=None: FakeResponse(
                    self.archive,
                    final_url="http://mirror.example.test/update.zip?token=private",
                ),
                sleep=lambda _delay: None,
            )
        self.assertIn("HTTPS", str(raised.exception))
        self.assertNotIn("mirror.example.test", str(raised.exception))

    def test_exhausted_download_does_not_leak_query_or_local_path(self):
        private_url = "https://downloads.example.test/update.zip?token=private"

        with self.assertRaises(updater.UpdateError) as raised:
            updater.prepare_update(
                private_url,
                hashlib.sha256(self.archive).hexdigest(),
                len(self.archive),
                app_dir=self.app_dir,
                opener=lambda _request, data=None, timeout=None: (_ for _ in ()).throw(
                    ConnectionResetError(str(self.app_dir / "secret"))
                ),
                policy=RetryPolicy(network_attempts=1),
                sleep=lambda _delay: None,
            )

        message = str(raised.exception)
        self.assertNotIn("token", message)
        self.assertNotIn("private", message)
        self.assertNotIn(str(self.app_dir), message)

    def test_non_frozen_mode_refuses_install_after_verification(self):
        messages = []
        with mock.patch.object(updater.sys, "frozen", False, create=True), mock.patch.object(
            updater, "prepare_update"
        ) as prepare:
            prepare.return_value = updater.PreparedUpdate(
                self.app_dir / "stage",
                self.app_dir / "stage/update.zip",
                self.app_dir / "stage/payload",
            )
            result = updater.download_and_install_update(
                "https://downloads.example.test/update.zip",
                output_queue=None,
                expected_sha256=VALID_SHA,
                expected_size=1,
                log=messages.append,
            )
        self.assertFalse(result)
        self.assertTrue(prepare.called)
        self.assertTrue(any("源码" in message for message in messages))

    def test_frozen_application_directory_comes_from_executable_not_cwd(self):
        with mock.patch.object(updater.sys, "frozen", True, create=True), mock.patch.object(
            updater.sys, "executable", str(self.app_dir / "Lovart_Auto.exe")
        ), mock.patch("updater.Path.cwd", return_value=self.app_dir / "wrong"):
            self.assertEqual(updater.application_directory(), self.app_dir.resolve())

    def test_busy_installer_returns_without_exiting_or_releasing_active_owner(self):
        update_root = self.app_dir / ".lovart-update"
        active_lock = updater._reserve_installer_lock(update_root, "active-owner")
        contender = update_root / "contender"
        (contender / "payload").mkdir(parents=True)
        prepared = updater.PreparedUpdate(
            contender,
            contender / "update.zip",
            contender / "payload",
        )
        messages = []
        try:
            with mock.patch.object(
                updater.sys, "frozen", True, create=True
            ), mock.patch.object(updater.os, "name", "nt"), mock.patch.object(
                updater, "application_directory", return_value=self.app_dir
            ), mock.patch.object(
                updater, "prepare_update", return_value=prepared
            ), mock.patch.object(
                updater,
                "_write_and_start_installer",
                side_effect=updater.UpdateError("另一个更新安装正在进行，请稍后重试。"),
            ), mock.patch.object(
                updater.os, "_exit"
            ) as exit_process:
                result = updater.download_and_install_update(
                    "https://downloads.example.test/update.zip",
                    expected_sha256=VALID_SHA,
                    expected_size=1,
                    log=messages.append,
                )
            self.assertFalse(result)
            exit_process.assert_not_called()
            self.assertTrue(active_lock.exists())
            self.assertEqual(
                (active_lock / "token").read_text(encoding="ascii"),
                "active-owner",
            )
            self.assertTrue(any("进行" in message for message in messages))
        finally:
            updater._release_installer_lock(update_root, "active-owner")

    def test_missing_prepare_claim_is_rejected_before_installer_process_starts(self):
        update_root = self.app_dir / ".lovart-update"
        staging = update_root / "abcdef0123456789"
        payload = staging / "payload"
        payload.mkdir(parents=True)
        prepared = updater.PreparedUpdate(
            staging,
            staging / "update.zip",
            payload,
        )
        with mock.patch("updater.subprocess.Popen") as popen:
            with self.assertRaises(updater.UpdateError):
                updater._write_and_start_installer(
                    prepared,
                    self.app_dir,
                    current_pid=1234,
                )
        popen.assert_not_called()
        self.assertFalse((update_root / "install.lock").exists())

    def test_installer_script_is_transactional_owner_safe_and_preserves_user_files(self):
        payload = self.app_dir / ".lovart-update" / "owner-123" / "payload"
        payload.mkdir(parents=True)
        script = updater.build_windows_installer_script(
            app_dir=self.app_dir,
            payload_dir=payload,
            staging_dir=payload.parent,
            current_pid=4242,
            owner_token="owner-123",
        )

        self.assertIn("Get-Process -Id 4242", script)
        self.assertIn("Lovart_Auto.exe.bak-owner-123", script)
        self.assertIn("_internal.bak-owner-123", script)
        self.assertIn("--run-main --help", script)
        self.assertIn(":rollback", script)
        self.assertIn("goto rollback", script)
        self.assertNotIn("xcopy", script.lower())
        self.assertNotIn("/C", script)
        for protected in ("config.yaml", "browser_profile", "runs", "user data"):
            self.assertNotIn(protected, script)
        smoke = script.index("--run-main --help")
        commit = script.index('move /Y "!PENDING_MARKER!" "!COMMIT_MARKER!"')
        cleanup = script.index('call :cleanup_committed "!TOKEN!"', commit)
        launch = script.index('start "" /b "!TARGET!\\Lovart_Auto.exe"')
        lock_validation = script.index("call :validate_lock")
        recovery = script.index("call :recover_stale")
        self.assertLess(smoke, commit)
        self.assertLess(commit, launch)
        self.assertLess(launch, cleanup)
        self.assertLess(lock_validation, recovery)
        self.assertIn("installer.identity", script)
        self.assertIn(":release_lock", script)
        self.assertIn(
            'copy /Y "!PAYLOAD!\\config.example.yaml" "!TARGET!\\config.example.yaml"',
            script,
        )
        self.assertIn(
            'copy /Y "!PAYLOAD!\\.env.example" "!TARGET!\\.env.example"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
