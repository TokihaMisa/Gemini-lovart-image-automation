from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from ctypes import wintypes
from pathlib import Path
from unittest import mock

import updater


WINDOWS = os.name == "nt"
REQUIRED_RUNTIME = (
    "_internal/python314.dll",
    "_internal/VCRUNTIME140.dll",
    "_internal/VCRUNTIME140_1.dll",
)
APPROVED_ASSETS = ("config.example.yaml", ".env.example")


class BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def geturl(self):
        return "https://downloads.example.test/update.zip"


class SyntheticArchive:
    def __init__(self, infos):
        self._infos = infos
        self.testzip_called = False

    def infolist(self):
        return self._infos

    def testzip(self):
        self.testzip_called = True
        return None


def synthetic_info(name, file_size, compress_size):
    info = zipfile.ZipInfo(name)
    info.file_size = file_size
    info.compress_size = compress_size
    return info


def required_synthetic_infos():
    return [
        synthetic_info("Lovart_Auto.exe", 1024, 768),
        *(synthetic_info(name, 1024, 768) for name in REQUIRED_RUNTIME),
    ]


class ZipBombBoundTests(unittest.TestCase):
    def test_rejects_per_entry_limit_at_equality_before_crc(self):
        archive = SyntheticArchive(
            required_synthetic_infos()
            + [synthetic_info("_internal/huge.bin", 1024 * 1024 * 1024, 512 * 1024 * 1024)]
        )
        with self.assertRaises(updater.UpdateError):
            updater._validate_zip_members(archive)
        self.assertFalse(archive.testzip_called)

    def test_rejects_4095_megabyte_entries_with_tiny_overall_archive_before_crc(self):
        infos = required_synthetic_infos()
        infos.extend(
            synthetic_info(f"_internal/chunk-{index}.bin", 1024 * 1024, 1050)
            for index in range(4095)
        )
        archive = SyntheticArchive(infos)
        with self.assertRaises(updater.UpdateError):
            updater._validate_zip_members(archive)
        self.assertFalse(archive.testzip_called)

    def test_accepts_realistic_pyinstaller_scale(self):
        infos = required_synthetic_infos()
        infos.extend(
            synthetic_info(f"_internal/package-{index}.bin", 128 * 1024, 64 * 1024)
            for index in range(4000)
        )
        archive = SyntheticArchive(infos)
        validated = updater._validate_zip_members(archive)
        self.assertEqual(len(validated), len(infos))
        self.assertTrue(archive.testzip_called)


class AppSmokeExitTests(unittest.TestCase):
    def test_run_main_import_failure_exits_nonzero_in_real_subprocess(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="ota-app-red-") as temp:
            temp_path = Path(temp)
            copied_app = temp_path / "app.py"
            shutil.copy2(Path("app.py"), copied_app)
            (temp_path / "main.py").write_text(
                "def main():\n    raise RuntimeError('offline smoke import failure')\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, str(copied_app), "--run-main"],
                cwd=temp_path,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Fatal Error", result.stdout + result.stderr)

    def test_real_run_main_help_remains_successful(self):
        result = subprocess.run(
            [sys.executable, "app.py", "--run-main", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class InstallerLockBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="ota-lock-")
        self.update_root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_recent_lock_without_pid_is_busy_then_explicitly_stale_lock_is_replaced(self):
        first = updater._reserve_installer_lock(self.update_root, "first-owner")
        with self.assertRaises(updater.UpdateError):
            updater._reserve_installer_lock(self.update_root, "second-owner")
        with mock.patch(
            "updater.time.time",
            return_value=time.time() + updater._INSTALL_LOCK_GRACE_SECONDS + 1,
        ):
            second = updater._reserve_installer_lock(
                self.update_root, "second-owner"
            )
        self.assertEqual(first, second)
        self.assertFalse(list(self.update_root.glob("install.lock.stale-*")))
        self.assertEqual(
            (second / "token").read_text(encoding="ascii"), "second-owner"
        )
        self.assertTrue(
            updater._release_installer_lock(self.update_root, "second-owner")
        )

    def test_live_pid_identity_is_busy_and_mismatch_is_treated_as_reuse(self):
        first = updater._reserve_installer_lock(self.update_root, "first-owner")
        (first / "installer.pid").write_text(str(os.getpid()), encoding="ascii")
        identity = updater._process_identity(os.getpid())
        self.assertIsNotNone(identity)
        (first / "installer.identity").write_text(identity, encoding="ascii")
        with self.assertRaises(updater.UpdateError):
            updater._reserve_installer_lock(self.update_root, "second-owner")
        (first / "installer.identity").write_text("1", encoding="ascii")
        second = updater._reserve_installer_lock(self.update_root, "second-owner")
        self.assertEqual(
            (second / "token").read_text(encoding="ascii"), "second-owner"
        )
        self.assertFalse(
            updater._release_installer_lock(self.update_root, "first-owner")
        )
        self.assertTrue(second.exists())
        self.assertTrue(
            updater._release_installer_lock(self.update_root, "second-owner")
        )


@unittest.skipUnless(WINDOWS, "real cmd.exe OTA transaction tests require Windows")
class WindowsInstallerTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler_temp = tempfile.TemporaryDirectory(
            dir=Path.cwd(), prefix="ota-compiler-"
        )
        compiler_root = Path(cls.compiler_temp.name)
        source = compiler_root / "stub.cs"
        source.write_text(
            """
using System;
using System.IO;
class Program {
    static int Main(string[] args) {
        bool smoke = Array.IndexOf(args, "--run-main") >= 0;
        if (smoke && Environment.GetEnvironmentVariable("LOVART_OTA_SMOKE_FAIL") == "1") {
            return 17;
        }
        if (!smoke) {
            string marker = Environment.GetEnvironmentVariable("LOVART_OTA_LAUNCH_MARKER");
            if (!String.IsNullOrEmpty(marker)) {
                File.AppendAllText(marker, "launched\\n");
            }
        }
        return 0;
    }
}
""".strip(),
            encoding="utf-8",
        )
        compiler = next(
            path
            for path in (
                Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe",
                Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework/v4.0.30319/csc.exe",
            )
            if path.exists()
        )
        cls.stub_exe = compiler_root / "stub.exe"
        subprocess.run(
            [
                str(compiler),
                "/nologo",
                "/target:exe",
                f"/out:{cls.stub_exe}",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.compiler_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="ota-cmd-e2e-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_exe(self, path, version):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.stub_exe, path)
        with path.open("ab") as output:
            output.write(version.encode("ascii"))

    def _create_target(self, root=None):
        target = (root or self.root) / "app"
        target.mkdir(parents=True)
        self._write_exe(target / "Lovart_Auto.exe", "OLD")
        for name in REQUIRED_RUNTIME:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"OLD")
        for name in APPROVED_ASSETS:
            (target / name).write_bytes(b"OLD")
        return target

    def _create_prepared(self, target, token, version="NEW"):
        update_root = target / ".lovart-update"
        update_root.mkdir(parents=True, exist_ok=True)
        updater._create_prepare_claim(update_root, token)
        staging = update_root / token
        payload = staging / "payload"
        payload.mkdir(parents=True)
        self._write_exe(payload / "Lovart_Auto.exe", version)
        for name in REQUIRED_RUNTIME:
            path = payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(version.encode("ascii"))
        for name in APPROVED_ASSETS:
            (payload / name).write_bytes(version.encode("ascii"))
        (staging / "update.zip").write_bytes(b"verified archive")
        return updater.PreparedUpdate(staging, staging / "update.zip", payload)

    def _archive_bytes(self, version):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(
            archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                "Lovart_Auto.exe",
                self.stub_exe.read_bytes() + version.encode("ascii"),
            )
            for name in REQUIRED_RUNTIME:
                archive.writestr(name, version.encode("ascii"))
            for name in APPROVED_ASSETS:
                archive.writestr(name, version.encode("ascii"))
        return archive_buffer.getvalue()

    def _prepare_real(self, target, version):
        archive = self._archive_bytes(version)
        return updater.prepare_update(
            "https://downloads.example.test/update.zip",
            hashlib.sha256(archive).hexdigest(),
            len(archive),
            app_dir=target,
            opener=lambda _request, _timeout: BytesResponse(archive),
            sleep=lambda _delay: None,
        )

    def _write_prepare_claim(
        self,
        update_root,
        token,
        *,
        created,
        pid=None,
        identity=None,
    ):
        claim = update_root / f"prepare-{token}.claim"
        claim.mkdir()
        (claim / "token").write_text(token, encoding="ascii")
        (claim / "created").write_text(str(created), encoding="ascii")
        if pid is not None:
            (claim / "pid").write_text(str(pid), encoding="ascii")
        if identity is not None:
            (claim / "identity").write_text(identity, encoding="ascii")
        staging = update_root / token
        staging.mkdir()
        (staging / "partial.zip").write_bytes(b"active preparation")
        return claim, staging

    def _exclusive_file_handle(self, path):
        kernel32 = __import__("ctypes").windll.kernel32
        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel32.CreateFileW.restype = wintypes.HANDLE
        handle = kernel32.CreateFileW(
            str(path),
            0x80000000,
            0,
            None,
            3,
            0x80,
            None,
        )
        self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
        return handle

    def _write_script(self, target, prepared, token, failure_step=None):
        update_root = target / ".lovart-update"
        update_root.mkdir(exist_ok=True)
        lock_dir = updater._reserve_installer_lock(update_root, token)
        (lock_dir / "installer.pid").write_text(str(os.getpid()), encoding="ascii")
        (lock_dir / "installer.identity").write_text(
            updater._process_identity(os.getpid()), encoding="ascii"
        )
        self.assertTrue(updater._release_prepare_claim(update_root, token))
        updater._mark_installer_handoff_ready(lock_dir, token)
        (update_root / f"install-{token}.owner").write_text(token, encoding="ascii")
        arguments = dict(
            app_dir=target,
            payload_dir=prepared.payload_dir,
            staging_dir=prepared.staging_dir,
            current_pid=999999,
            owner_token=token,
        )
        if failure_step is not None:
            arguments["failure_step"] = failure_step
        content = updater.build_windows_installer_script(**arguments)
        script = update_root / f"install-{token}.bat"
        script.write_text(content, encoding="utf-8", newline="\r\n")
        return script

    def _run_script(self, script, *, smoke_fail=False, marker=None):
        env = os.environ.copy()
        env["LOVART_OTA_SMOKE_FAIL"] = "1" if smoke_fail else "0"
        if marker is not None:
            env["LOVART_OTA_LAUNCH_MARKER"] = str(marker)
        return subprocess.run(
            updater._windows_installer_command(script),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _assert_version(self, target, expected):
        self.assertTrue((target / "Lovart_Auto.exe").read_bytes().endswith(expected))
        for name in REQUIRED_RUNTIME:
            self.assertEqual((target / name).read_bytes(), expected)
        for name in APPROVED_ASSETS:
            self.assertEqual((target / name).read_bytes(), expected)

    def _assert_bounded_cleanup(self, target):
        update_root = target / ".lovart-update"
        leftovers = sorted(path.name for path in update_root.iterdir())
        self.assertEqual(leftovers, ["last-install.log"])
        self.assertFalse(list(target.glob("*.bak-*")))
        self.assertFalse(list(target.glob("*.absent-*")))

    def test_real_smoke_failure_rolls_back_core_and_assets(self):
        target = self._create_target()
        prepared = self._create_prepared(target, "smoke-red")
        script = self._write_script(target, prepared, "smoke-red")

        result = self._run_script(script, smoke_fail=True)

        self.assertNotEqual(result.returncode, 0)
        self._assert_version(target, b"OLD")
        self._assert_bounded_cleanup(target)

    def test_every_forward_failure_rolls_back_and_next_run_succeeds(self):
        failure_steps = (
            "backup_exe",
            "backup_internal",
            "backup_config",
            "backup_env",
            "install_exe",
            "install_internal",
            "copy_config",
            "copy_env",
            "smoke",
            "commit",
        )
        for index, failure_step in enumerate(failure_steps):
            with self.subTest(failure_step=failure_step):
                case_root = self.root / str(index)
                target = self._create_target(case_root)
                failed = self._create_prepared(target, f"failed-{index}")
                failed_script = self._write_script(
                    target, failed, f"failed-{index}", failure_step=failure_step
                )
                result = self._run_script(failed_script)
                self.assertNotEqual(result.returncode, 0)
                self._assert_version(target, b"OLD")
                self._assert_bounded_cleanup(target)

                retry = self._create_prepared(target, f"retry-{index}")
                retry_script = self._write_script(target, retry, f"retry-{index}")
                marker = case_root / "launch.txt"
                retry_result = self._run_script(retry_script, marker=marker)
                self.assertEqual(
                    retry_result.returncode,
                    0,
                    retry_result.stdout + retry_result.stderr,
                )
                self._assert_version(target, b"NEW")
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                time.sleep(0.1)
                self._assert_bounded_cleanup(target)

    def test_second_run_self_heals_stale_partial_transaction(self):
        target = self._create_target()
        update_root = target / ".lovart-update"
        update_root.mkdir()
        stale = "stale-owner"
        (target / "Lovart_Auto.exe").replace(
            target / f"Lovart_Auto.exe.bak-{stale}"
        )
        (target / "_internal").replace(target / f"_internal.bak-{stale}")
        for name in APPROVED_ASSETS:
            (target / name).replace(target / f"{name}.bak-{stale}")
            (target / name).write_bytes(b"MIXED")
        self._write_exe(target / "Lovart_Auto.exe", "MIXED")
        prepared = self._create_prepared(target, "recovery-run")
        script = self._write_script(target, prepared, "recovery-run")
        marker = self.root / "recovery-launch.txt"
        result = self._run_script(script, marker=marker)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self._assert_version(target, b"NEW")
        self._assert_bounded_cleanup(target)

    def test_real_cmd_handoff_accepts_space_and_ampersand_and_confirms_owner(self):
        special_root = self.root / "space & amp"
        special_root.mkdir()
        target = self._create_target(special_root)
        prepared = self._create_prepared(target, "quoted-owner")
        sleeper = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Start-Sleep -Seconds 30",
            ]
        )
        marker = special_root / "launch.txt"
        old_marker = os.environ.get("LOVART_OTA_LAUNCH_MARKER")
        os.environ["LOVART_OTA_LAUNCH_MARKER"] = str(marker)
        try:
            process = updater._write_and_start_installer(
                prepared,
                target,
                current_pid=sleeper.pid,
                start_timeout=5,
            )
            started = target / ".lovart-update" / "install-quoted-owner.started"
            self.assertEqual(
                started.read_text(encoding="ascii").strip(), "quoted-owner"
            )
            sleeper.terminate()
            sleeper.wait(timeout=5)
            self.assertEqual(process.wait(timeout=20), 0)
        finally:
            if sleeper.poll() is None:
                sleeper.terminate()
                sleeper.wait(timeout=5)
            if old_marker is None:
                os.environ.pop("LOVART_OTA_LAUNCH_MARKER", None)
            else:
                os.environ["LOVART_OTA_LAUNCH_MARKER"] = old_marker
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(marker.exists())
        time.sleep(0.1)
        self._assert_version(target, b"NEW")
        self._assert_bounded_cleanup(target)

    def test_real_dual_installer_barrier_allows_only_one_complete_owner(self):
        for first_version, second_version in (("A", "B"), ("B", "A")):
            with self.subTest(first=first_version):
                case_root = self.root / f"{first_version}-first"
                target = self._create_target(case_root)
                first = self._create_prepared(
                    target, f"owner-{first_version}", first_version
                )
                second = self._create_prepared(
                    target, f"owner-{second_version}", second_version
                )
                sleeper = subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-Command",
                        "Start-Sleep -Seconds 30",
                    ]
                )
                first_process = None
                second_process = None
                busy = False
                try:
                    first_process = updater._write_and_start_installer(
                        first,
                        target,
                        current_pid=sleeper.pid,
                        start_timeout=5,
                    )
                    try:
                        second_process = updater._write_and_start_installer(
                            second,
                            target,
                            current_pid=sleeper.pid,
                            start_timeout=2,
                        )
                    except updater.UpdateError as exc:
                        busy = "进行" in str(exc) or "busy" in str(exc).lower()
                finally:
                    sleeper.terminate()
                    sleeper.wait(timeout=5)
                    if first_process is not None:
                        first_process.wait(timeout=20)
                    if second_process is not None:
                        second_process.wait(timeout=20)
                self.assertTrue(busy, "the second live installer was not rejected")
                self._assert_version(target, first_version.encode("ascii"))

                shutil.rmtree(second.staging_dir, ignore_errors=True)
                updater._release_prepare_claim(
                    target / ".lovart-update", second.staging_dir.name
                )
                third = self._create_prepared(target, "owner-C", "C")
                third_process = updater._write_and_start_installer(
                    third,
                    target,
                    current_pid=999999,
                    start_timeout=5,
                )
                self.assertEqual(third_process.wait(timeout=20), 0)
                self._assert_version(target, b"C")
                self._assert_bounded_cleanup(target)

    def test_committed_install_survives_locked_staging_and_next_run_cleans_orphan(self):
        target = self._create_target()
        first = self._create_prepared(target, "locked-owner", "A")
        marker = self.root / "locked-launch.txt"
        handle = self._exclusive_file_handle(first.archive_path)
        try:
            first_result = self._run_script(
                self._write_script(target, first, "locked-owner"),
                marker=marker,
            )
            self.assertEqual(
                first_result.returncode,
                0,
                first_result.stdout + first_result.stderr,
            )
            self._assert_version(target, b"A")
            self.assertTrue(marker.exists())
            update_root = target / ".lovart-update"
            pending = update_root / "cleanup-pending-locked-owner"
            self.assertTrue(pending.exists())
            self.assertFalse((update_root / "install.lock").exists())
            self.assertIn(
                "cleanup_pending",
                (update_root / "last-install.log").read_text(encoding="utf-8"),
            )
        finally:
            __import__("ctypes").windll.kernel32.CloseHandle(handle)

        raw_orphan = target / ".lovart-update" / "crashed-owner"
        raw_orphan.mkdir()
        (raw_orphan / "partial.zip").write_bytes(b"partial")
        second = self._create_prepared(target, "cleanup-owner", "B")
        second_result = self._run_script(
            self._write_script(target, second, "cleanup-owner")
        )
        self.assertEqual(
            second_result.returncode,
            0,
            second_result.stdout + second_result.stderr,
        )
        self._assert_version(target, b"B")
        self.assertFalse(first.staging_dir.exists())
        self.assertFalse(raw_orphan.exists())
        self._assert_bounded_cleanup(target)

    def test_live_prepare_during_validation_survives_real_installer_cleanup(self):
        target = self._create_target()
        validation_entered = threading.Event()
        validation_release = threading.Event()
        worker_result = {}
        real_validate = updater.validate_and_extract_update

        def paused_validate(archive_path, destination):
            validation_entered.set()
            if not validation_release.wait(timeout=20):
                raise AssertionError("validation barrier timed out")
            return real_validate(archive_path, destination)

        def prepare_worker():
            try:
                worker_result["prepared"] = self._prepare_real(target, "B")
            except BaseException as exc:
                worker_result["error"] = exc

        with mock.patch(
            "updater.validate_and_extract_update", side_effect=paused_validate
        ):
            worker = threading.Thread(target=prepare_worker)
            worker.start()
            self.assertTrue(validation_entered.wait(timeout=10))
            update_root = target / ".lovart-update"
            live_staging = next(
                path
                for path in update_root.iterdir()
                if path.is_dir()
                and re.fullmatch(r"[0-9a-f]{16}", path.name)
                and (path / "update.zip").exists()
            )
            installer = self._create_prepared(target, "owner-A", "A")
            try:
                install_result = self._run_script(
                    self._write_script(target, installer, "owner-A")
                )
                survived = live_staging.exists() and (
                    live_staging / "update.zip"
                ).exists()
            finally:
                validation_release.set()
                worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(
            install_result.returncode,
            0,
            install_result.stdout + install_result.stderr,
        )
        self.assertTrue(survived, "installer deleted active validation staging")
        self.assertNotIn("error", worker_result)
        prepared = worker_result["prepared"]
        self.assertTrue((prepared.payload_dir / "Lovart_Auto.exe").exists())
        shutil.rmtree(prepared.staging_dir, ignore_errors=True)

    def test_prepared_claim_survives_install_then_converts_to_next_install_owner(self):
        target = self._create_target()
        prepared_b = self._prepare_real(target, "B")
        update_root = target / ".lovart-update"
        claim_b = update_root / f"prepare-{prepared_b.staging_dir.name}.claim"

        prepared_a = self._create_prepared(target, "owner-A", "A")
        result_a = self._run_script(
            self._write_script(target, prepared_a, "owner-A")
        )
        self.assertEqual(result_a.returncode, 0, result_a.stdout + result_a.stderr)
        self.assertTrue(prepared_b.staging_dir.exists())
        self.assertTrue(claim_b.exists())

        process_b = updater._write_and_start_installer(
            prepared_b,
            target,
            current_pid=999999,
            start_timeout=5,
        )
        self.assertFalse(claim_b.exists())
        self.assertEqual(process_b.wait(timeout=20), 0)
        self._assert_version(target, b"B")
        self._assert_bounded_cleanup(target)

    def test_prepare_claim_liveness_controls_bounded_orphan_reclamation(self):
        target = self._create_target()
        update_root = target / ".lovart-update"
        update_root.mkdir()
        now = int(time.time())
        identity = updater._process_identity(os.getpid())
        self.assertIsNotNone(identity)
        recent_claim, recent_staging = self._write_prepare_claim(
            update_root,
            "1111111111111111",
            created=now,
        )
        live_claim, live_staging = self._write_prepare_claim(
            update_root,
            "2222222222222222",
            created=now - 60,
            pid=os.getpid(),
            identity=identity,
        )
        dead_claim, dead_staging = self._write_prepare_claim(
            update_root,
            "3333333333333333",
            created=now,
            pid=999999,
            identity="1",
        )
        reused_claim, reused_staging = self._write_prepare_claim(
            update_root,
            "4444444444444444",
            created=now,
            pid=os.getpid(),
            identity="1",
        )
        expired_claim, expired_staging = self._write_prepare_claim(
            update_root,
            "5555555555555555",
            created=now - 60,
        )

        installer = self._create_prepared(target, "claim-cleaner", "A")
        result = self._run_script(
            self._write_script(target, installer, "claim-cleaner")
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for claim, staging in (
            (recent_claim, recent_staging),
            (live_claim, live_staging),
        ):
            self.assertTrue(claim.exists())
            self.assertTrue(staging.exists())
        for claim, staging in (
            (dead_claim, dead_staging),
            (reused_claim, reused_staging),
            (expired_claim, expired_staging),
        ):
            self.assertFalse(claim.exists())
            self.assertFalse(staging.exists())

    def test_prepare_failure_removes_its_claim_and_staging(self):
        target = self._create_target()
        archive = self._archive_bytes("B")
        claims_seen = []

        def fail_validation(_archive_path, _destination):
            claims_seen.extend(
                (target / ".lovart-update").glob("prepare-*.claim")
            )
            raise updater.UpdateError("injected validation failure")

        with mock.patch(
            "updater.validate_and_extract_update", side_effect=fail_validation
        ), self.assertRaises(updater.UpdateError):
            updater.prepare_update(
                "https://downloads.example.test/update.zip",
                hashlib.sha256(archive).hexdigest(),
                len(archive),
                app_dir=target,
                opener=lambda _request, _timeout: BytesResponse(archive),
                sleep=lambda _delay: None,
            )
        update_root = target / ".lovart-update"
        self.assertTrue(claims_seen)
        self.assertFalse(list(update_root.glob("prepare-*.claim")))
        self.assertFalse(
            [
                path
                for path in update_root.iterdir()
                if path.is_dir() and re.fullmatch(r"[0-9a-f]{16}", path.name)
            ]
        )

    def test_download_failure_removes_its_live_claim_and_staging(self):
        target = self._create_target()
        claims_seen = []

        def failing_opener(_request, _timeout):
            claims_seen.extend(
                (target / ".lovart-update").glob("prepare-*.claim")
            )
            raise ConnectionResetError("injected download failure")

        with self.assertRaises(updater.UpdateError):
            updater.prepare_update(
                "https://downloads.example.test/update.zip",
                "a" * 64,
                123,
                app_dir=target,
                opener=failing_opener,
                policy=updater.RetryPolicy(network_attempts=1),
                sleep=lambda _delay: None,
            )
        update_root = target / ".lovart-update"
        self.assertTrue(claims_seen)
        self.assertFalse(list(update_root.glob("prepare-*.claim")))
        self.assertFalse(
            [
                path
                for path in update_root.iterdir()
                if path.is_dir() and re.fullmatch(r"[0-9a-f]{16}", path.name)
            ]
        )

    def test_batch_path_rejects_unsafe_expansion_characters_before_handoff(self):
        for name in ('percent%name', 'bang!name', 'caret^name', 'quote"name', "line\nname"):
            with self.subTest(name=name), self.assertRaises(updater.UpdateError):
                updater._batch_path(self.root / name)


if __name__ == "__main__":
    unittest.main()
