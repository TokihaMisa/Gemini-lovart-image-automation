"""Fail-closed verification for the two OTA archives and release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from updater import validate_and_extract_update  # noqa: E402


def verify(root: Path, version: str) -> None:
    release = root / "update.zip"
    versioned = root / f"update-v{version}.zip"
    manifest_path = root / "version.json"
    for required in (release, versioned, manifest_path):
        if not required.is_file():
            raise ValueError(f"required release artifact is missing: {required.name}")
    release_bytes = release.read_bytes()
    if release_bytes != versioned.read_bytes():
        raise ValueError("update.zip and versioned update archive are not byte-identical")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(release_bytes).hexdigest()
    if manifest.get("version") != version:
        raise ValueError("version.json version does not match requested release")
    if manifest.get("sha256") != digest:
        raise ValueError("version.json sha256 does not match update.zip")
    if manifest.get("size") != len(release_bytes):
        raise ValueError("version.json size does not match update.zip")
    required_members = {"Lovart_Auto.exe", "config.example.yaml", ".env.example"}
    with zipfile.ZipFile(release, "r") as archive:
        names = {name.replace("\\", "/").rstrip("/") for name in archive.namelist()}
    missing = sorted(required_members - names)
    if missing:
        raise ValueError(f"release archive is missing required assets: {', '.join(missing)}")
    temporary_root = Path(tempfile.mkdtemp(prefix="lovart-release-verify-"))
    try:
        extracted = temporary_root / "payload"
        validate_and_extract_update(release, extracted)
        if not any(path.name == "Lovart_Auto.exe" for path in extracted.rglob("*")):
            raise ValueError("updater extraction produced no Lovart_Auto.exe")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        verify(args.root.resolve(), args.version)
    except Exception as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    print("release artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
