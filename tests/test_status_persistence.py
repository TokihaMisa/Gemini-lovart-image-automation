import errno
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import utils


def test_update_status_retries_transient_windows_access_denied(tmp_path: Path):
    real_replace = os.replace
    calls = []
    denied = PermissionError(errno.EACCES, "temporarily locked")
    denied.winerror = 5

    def flaky_replace(source, target):
        calls.append((Path(source), Path(target)))
        if len(calls) == 1:
            raise denied
        return real_replace(source, target)

    with patch("utils.os.replace", side_effect=flaky_replace), patch("utils.time.sleep") as sleep:
        saved = utils.update_status(tmp_path, "started", product_id="SKU-1")

    assert saved["started"] is True
    assert utils.read_status(tmp_path)["product_id"] == "SKU-1"
    assert len(calls) == 2
    sleep.assert_called_once()


def test_update_status_uses_unique_same_directory_temp_file(tmp_path: Path):
    real_replace = os.replace
    replace_calls = []

    def recording_replace(source, target):
        replace_calls.append((Path(source), Path(target)))
        return real_replace(source, target)

    with patch("utils.os.replace", side_effect=recording_replace):
        utils.update_status(tmp_path, "started")

    source, target = replace_calls[-1]
    assert source.parent == tmp_path
    assert source.name != "status.json.tmp"
    assert source.name.startswith(".status.")
    assert target == tmp_path / "status.json"
    assert list(tmp_path.glob(".status.*.json.tmp")) == []


def test_update_status_removes_unique_temp_after_non_transient_failure(tmp_path: Path):
    with patch("utils.os.replace", side_effect=OSError(errno.ENOSPC, "disk full")):
        with pytest.raises(OSError, match="disk full"):
            utils.update_status(tmp_path, "started")

    assert not (tmp_path / "status.json").exists()
    assert list(tmp_path.glob(".status.*.json.tmp")) == []


def test_output_run_owner_rejects_second_live_process(tmp_path: Path):
    owner = utils.acquire_output_run_owner(tmp_path)
    try:
        with pytest.raises(utils.OutputDirectoryInUseError) as raised:
            utils.acquire_output_run_owner(tmp_path)
    finally:
        utils.release_output_run_owner(tmp_path, owner)

    assert str(os.getpid()) in str(raised.value)
    assert not (tmp_path / ".automation-run.lock").exists()


def test_output_run_owner_reclaims_confirmed_stale_lock(tmp_path: Path):
    lock_path = tmp_path / ".automation-run.lock"
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 999999, "token": "stale", "created_at": 1}),
        encoding="utf-8",
    )

    with patch("utils._output_process_matches", return_value=False):
        owner = utils.acquire_output_run_owner(tmp_path)

    assert owner.token != "stale"
    utils.release_output_run_owner(tmp_path, owner)
    assert not lock_path.exists()


def test_output_run_owner_release_requires_matching_token(tmp_path: Path):
    owner = utils.acquire_output_run_owner(tmp_path)
    impostor = utils.OutputRunOwner(
        pid=owner.pid,
        token="not-the-owner",
        created_at=owner.created_at,
    )

    utils.release_output_run_owner(tmp_path, impostor)
    assert (tmp_path / ".automation-run.lock").exists()

    utils.release_output_run_owner(tmp_path, owner)
    assert not (tmp_path / ".automation-run.lock").exists()


def test_output_run_owner_blocks_a_separate_process(tmp_path: Path):
    owner = utils.acquire_output_run_owner(tmp_path)
    script = (
        "import sys, utils; "
        "\ntry: utils.acquire_output_run_owner(sys.argv[1])"
        "\nexcept utils.OutputDirectoryInUseError: raise SystemExit(7)"
        "\nraise SystemExit(0)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
    finally:
        utils.release_output_run_owner(tmp_path, owner)

    assert result.returncode == 7


def test_output_run_lock_releases_after_exception(tmp_path: Path):
    with pytest.raises(RuntimeError, match="boom"):
        with utils.output_run_lock(tmp_path):
            raise RuntimeError("boom")

    assert not (tmp_path / ".automation-run.lock").exists()


def test_main_holds_output_lock_while_running_pipeline(tmp_path: Path):
    import main

    events = []

    @contextmanager
    def fake_lock(output_dir):
        events.append(("enter", output_dir))
        try:
            yield
        finally:
            events.append(("exit", output_dir))

    def run_pipeline(_args):
        assert events == [("enter", str(tmp_path))]
        events.append(("run", str(tmp_path)))
        return "done"

    args = SimpleNamespace(generate_template=False)
    with (
        patch("main.parse_args", return_value=args),
        patch("main.get_output_dir", return_value=str(tmp_path)),
        patch("main.output_run_lock", side_effect=fake_lock),
        patch("main._run_main", side_effect=run_pipeline),
    ):
        result = main.main([])

    assert result == "done"
    assert events == [
        ("enter", str(tmp_path)),
        ("run", str(tmp_path)),
        ("exit", str(tmp_path)),
    ]
