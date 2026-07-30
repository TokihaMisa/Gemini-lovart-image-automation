from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import updater


WINDOWS = os.name == "nt"
REQUIRED_RUNTIME = (
    "_internal/python314.dll",
    "_internal/VCRUNTIME140.dll",
    "_internal/VCRUNTIME140_1.dll",
)
APPROVED_ASSETS = ("config.example.yaml", ".env.example")


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

    def _create_prepared(self, target, token):
        staging = target / ".lovart-update" / token
        payload = staging / "payload"
        payload.mkdir(parents=True)
        self._write_exe(payload / "Lovart_Auto.exe", "NEW")
        for name in REQUIRED_RUNTIME:
            path = payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"NEW")
        for name in APPROVED_ASSETS:
            (payload / name).write_bytes(b"NEW")
        (staging / "update.zip").write_bytes(b"verified archive")
        return updater.PreparedUpdate(staging, staging / "update.zip", payload)

    def _write_script(self, target, prepared, token, failure_step=None):
        update_root = target / ".lovart-update"
        update_root.mkdir(exist_ok=True)
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

    def test_batch_path_rejects_unsafe_expansion_characters_before_handoff(self):
        for name in ('percent%name', 'bang!name', 'caret^name', 'quote"name', "line\nname"):
            with self.subTest(name=name), self.assertRaises(updater.UpdateError):
                updater._batch_path(self.root / name)


if __name__ == "__main__":
    unittest.main()
