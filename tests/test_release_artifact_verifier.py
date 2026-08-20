import json
from pathlib import Path
import subprocess
import sys
import zipfile


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_release_artifacts.py"


def _write_release_fixture(root: Path) -> None:
    payload = root / "payload"
    payload.mkdir()
    for name, data in {
        "Lovart_Auto.exe": b"exe",
        "config.example.yaml": b"config",
        ".env.example": b"env",
        "_internal/python314.dll": b"python",
        "_internal/VCRUNTIME140.dll": b"runtime",
        "_internal/VCRUNTIME140_1.dll": b"runtime1",
    }.items():
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive = root / "update-v1.3.21.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for item in payload.rglob("*"):
            if item.is_file():
                bundle.write(item, item.relative_to(payload).as_posix())
    (root / "update.zip").write_bytes(archive.read_bytes())
    import hashlib
    raw = archive.read_bytes()
    (root / "version.json").write_text(json.dumps({
        "version": "1.3.21", "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }), encoding="utf-8")


def test_release_verifier_passes_exact_matching_artifacts(tmp_path):
    _write_release_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--version", "1.3.21"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_release_verifier_fails_when_required_archive_is_missing(tmp_path):
    _write_release_fixture(tmp_path)
    (tmp_path / "update.zip").unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--version", "1.3.21"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "update.zip" in result.stderr
