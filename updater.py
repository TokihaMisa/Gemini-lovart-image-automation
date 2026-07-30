from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import zipfile

from network_retry import RetryKind, RetryPolicy, classify_network_error, run_with_retry
from version import UPDATE_INFO_URL, VERSION


_USER_AGENT = f"Lovart-Auto/{VERSION}"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_OWNER_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]+$")
_WINDOWS_RESERVED_NAMES = frozenset(
    ("con", "prn", "aux", "nul", "clock$")
    + tuple(f"com{number}" for number in range(1, 10))
    + tuple(f"lpt{number}" for number in range(1, 10))
)
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ZIP_ENTRIES = 20_000
_MAX_ZIP_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_SINGLE_ENTRY_BYTES = 2 * 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1_000
_ALLOWED_ROOT_FILES = frozenset(("Lovart_Auto.exe", "config.example.yaml", ".env.example"))
_REQUIRED_ARCHIVE_FILES = frozenset(
    (
        "Lovart_Auto.exe",
        "_internal/python314.dll",
        "_internal/VCRUNTIME140.dll",
        "_internal/VCRUNTIME140_1.dll",
    )
)


class UpdateStatus(str, Enum):
    UPDATE_AVAILABLE = "update_available"
    UP_TO_DATE = "up_to_date"
    ERROR = "error"


@dataclass(frozen=True)
class UpdateCheckResult:
    status: UpdateStatus
    version: str | None = None
    url: str | None = None
    changelog: str = ""
    sha256: str | None = None
    size: int | None = None
    message: str = ""


@dataclass(frozen=True)
class PreparedUpdate:
    staging_dir: Path
    archive_path: Path
    payload_dir: Path


class UpdateError(Exception):
    """An expected update failure whose message is safe to display."""


def _numeric_version(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise UpdateError("更新信息中的版本号无效，请稍后重试。")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def _validated_manifest(data: object) -> tuple[str, str, str, str, int]:
    if not isinstance(data, dict):
        raise UpdateError("更新信息格式无效，请稍后重试。")
    version = data.get("version")
    _numeric_version(version)
    url = data.get("url")
    if not isinstance(url, str):
        raise UpdateError("更新信息缺少安全下载地址，请稍后重试。")
    parsed_url = urlsplit(url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        raise UpdateError("更新信息缺少安全下载地址，请稍后重试。")
    digest = data.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise UpdateError("更新信息缺少有效的完整性校验值，请稍后重试。")
    size = data.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise UpdateError("更新信息缺少有效的文件大小，请稍后重试。")
    changelog = data.get("changelog", "")
    if not isinstance(changelog, str):
        raise UpdateError("更新信息格式无效，请稍后重试。")
    return version, url, changelog, digest.lower(), size


def _cache_busted_manifest_url() -> str:
    separator = "&" if "?" in UPDATE_INFO_URL else "?"
    return f"{UPDATE_INFO_URL}{separator}_={time.time_ns()}"


def _safe_network_guidance(exc: BaseException, action: str) -> str:
    kind = classify_network_error(exc)
    if kind is RetryKind.PERMANENT_TLS:
        return f"{action}失败：TLS 证书验证未通过，请检查系统时间、代理、VPN 或安全软件后重试。"
    if kind is RetryKind.AUTH:
        return f"{action}失败：更新服务拒绝了请求，请稍后重试或联系维护人员。"
    if kind is RetryKind.NOT_FOUND:
        return f"{action}失败：更新文件暂不可用（404），请稍后重试。"
    if kind is RetryKind.TRANSIENT:
        return f"{action}失败：网络暂时不可用，请检查连接后重试。"
    return f"{action}失败，请稍后重试。"


def check_update_details(
    *,
    opener=None,
    policy: RetryPolicy | None = None,
    sleep=None,
    current_version: str = VERSION,
) -> UpdateCheckResult:
    """Return a structured, display-safe result for an update check."""

    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(
        _cache_busted_manifest_url(),
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )

    def fetch_manifest():
        with opener(request, 10) as response:
            raw = response.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise UpdateError("更新信息过大，已停止检查。")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise UpdateError("更新信息格式无效，请稍后重试。") from None

    try:
        data = run_with_retry(
            fetch_manifest,
            policy or RetryPolicy(),
            sleep=sleep,
        )
        latest, url, changelog, digest, size = _validated_manifest(data)
        current = _numeric_version(current_version)
        status = (
            UpdateStatus.UPDATE_AVAILABLE
            if _numeric_version(latest) > current
            else UpdateStatus.UP_TO_DATE
        )
        return UpdateCheckResult(
            status=status,
            version=latest,
            url=url,
            changelog=changelog,
            sha256=digest,
            size=size,
        )
    except UpdateError as exc:
        return UpdateCheckResult(UpdateStatus.ERROR, message=str(exc))
    except Exception as exc:
        return UpdateCheckResult(
            UpdateStatus.ERROR,
            message=_safe_network_guidance(exc, "检查更新"),
        )


def check_for_updates():
    """Compatibility wrapper returning the original four-tuple."""

    result = check_update_details()
    if result.status is UpdateStatus.UPDATE_AVAILABLE:
        return True, result.version, result.url, result.changelog
    return False, None, None, None


def application_directory() -> Path:
    """Resolve the installed application root without trusting the CWD."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _validated_integrity(expected_sha256: object, expected_size: object) -> tuple[str, int]:
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise UpdateError("缺少有效的更新完整性校验值，已停止安装。")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise UpdateError("缺少有效的更新文件大小，已停止安装。")
    return expected_sha256.lower(), expected_size


def _download_verified_archive(
    url: str,
    archive_part: Path,
    archive_path: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    opener,
    policy: RetryPolicy,
    sleep,
    log,
) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/zip, application/octet-stream",
            "Cache-Control": "no-store",
        },
    )

    def download_once():
        if archive_part.exists():
            archive_part.unlink()
        digest = hashlib.sha256()
        downloaded = 0
        with opener(request, 30) as response, archive_part.open("xb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > expected_size:
                    raise UpdateError("更新包大小与发布信息不一致，已停止安装。")
                digest.update(block)
                output.write(block)
                if log:
                    percent = min(100, int(downloaded * 100 / expected_size))
                    log(f"正在下载更新包… {percent}%")
        if downloaded != expected_size:
            raise UpdateError("更新包大小与发布信息不一致，已停止安装。")
        if digest.hexdigest().lower() != expected_sha256:
            raise UpdateError("更新包完整性校验失败，已停止安装。")
        archive_part.replace(archive_path)

    run_with_retry(download_once, policy, sleep=sleep, on_retry=log)


def _normalized_zip_name(info: zipfile.ZipInfo) -> str:
    raw = info.filename
    if not raw or "\x00" in raw:
        raise UpdateError("更新包包含无效文件名，已停止安装。")
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE_PATH_RE.match(normalized):
        raise UpdateError("更新包包含不安全路径，已停止安装。")
    path = PurePosixPath(normalized)
    if any(part in ("", ".", "..") or ":" in part for part in path.parts):
        raise UpdateError("更新包包含不安全路径，已停止安装。")
    if any(
        part.rstrip(" .").split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        for part in path.parts
    ):
        raise UpdateError("更新包包含不安全路径，已停止安装。")
    return path.as_posix()


def _is_link_or_reparse(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    dos_attributes = info.external_attr & 0xFFFF
    return stat.S_ISLNK(unix_mode) or bool(dos_attributes & 0x400)


def _validate_zip_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_ZIP_ENTRIES:
        raise UpdateError("更新包文件数量异常，已停止安装。")
    total_size = 0
    seen: set[str] = set()
    present: set[str] = set()
    validated: list[tuple[zipfile.ZipInfo, str]] = []
    for info in infos:
        normalized = _normalized_zip_name(info)
        duplicate_key = normalized.rstrip("/").casefold()
        if duplicate_key in seen:
            raise UpdateError("更新包包含重复文件，已停止安装。")
        seen.add(duplicate_key)
        if info.flag_bits & 0x1:
            raise UpdateError("更新包包含加密文件，已停止安装。")
        if _is_link_or_reparse(info):
            raise UpdateError("更新包包含链接或重解析项，已停止安装。")
        root = normalized.split("/", 1)[0]
        if root == "_internal":
            pass
        elif normalized.rstrip("/") not in _ALLOWED_ROOT_FILES:
            raise UpdateError("更新包包含未批准的顶层内容，已停止安装。")
        if info.file_size < 0 or info.file_size > _MAX_SINGLE_ENTRY_BYTES:
            raise UpdateError("更新包中的文件大小异常，已停止安装。")
        total_size += info.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            raise UpdateError("更新包解压后体积异常，已停止安装。")
        if (
            info.file_size > 1024 * 1024
            and info.compress_size > 0
            and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO
        ):
            raise UpdateError("更新包压缩比例异常，已停止安装。")
        if not info.is_dir():
            present.add(normalized)
        validated.append((info, normalized))
    if not _REQUIRED_ARCHIVE_FILES.issubset(present):
        raise UpdateError("更新包缺少必要的程序运行文件，已停止安装。")
    try:
        bad_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile):
        raise UpdateError("更新包已损坏，已停止安装。") from None
    if bad_member is not None:
        raise UpdateError("更新包已损坏，已停止安装。")
    return validated


def validate_and_extract_update(archive_path: Path, destination: Path) -> None:
    """Validate the complete ZIP before extracting any member."""

    if destination.exists():
        raise UpdateError("更新暂存目录冲突，已停止安装。")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _validate_zip_members(archive)
            destination.mkdir(parents=False)
            for info, normalized in members:
                target = destination.joinpath(*PurePosixPath(normalized).parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except UpdateError:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile):
        if destination.exists():
            shutil.rmtree(destination)
        raise UpdateError("更新包无法安全解压，已停止安装。") from None


def prepare_update(
    url: str,
    expected_sha256: str,
    expected_size: int,
    *,
    app_dir: Path | None = None,
    opener=None,
    policy: RetryPolicy | None = None,
    sleep=None,
    log=None,
) -> PreparedUpdate:
    """Download, verify, validate and extract into a fresh app-local staging area."""

    digest, size = _validated_integrity(expected_sha256, expected_size)
    opener = opener or urllib.request.urlopen
    parsed_url = urlsplit(url) if isinstance(url, str) else None
    if parsed_url is None or parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        raise UpdateError("更新下载地址不安全，已停止安装。")
    target_root = (app_dir or application_directory()).resolve()
    staging_parent = target_root / ".lovart-update"
    staging_parent.mkdir(parents=True, exist_ok=True)
    owner_token = secrets.token_hex(8)
    staging_dir = staging_parent / owner_token
    staging_dir.mkdir()
    archive_part = staging_dir / "update.zip.part"
    archive_path = staging_dir / "update.zip"
    payload_dir = staging_dir / "payload"
    try:
        _download_verified_archive(
            url,
            archive_part,
            archive_path,
            digest,
            size,
            opener=opener,
            policy=policy or RetryPolicy(),
            sleep=sleep,
            log=log,
        )
        if log:
            log("下载完成，大小和 SHA-256 完整性校验成功。")
        validate_and_extract_update(archive_path, payload_dir)
        if log:
            log("更新包结构与运行文件校验成功，准备安全安装。")
        return PreparedUpdate(staging_dir, archive_path, payload_dir)
    except UpdateError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise UpdateError(_safe_network_guidance(exc, "下载更新")) from None


def _batch_path(path: Path) -> str:
    value = str(path.resolve())
    if any(character in value for character in ('"', "\r", "\n")):
        raise UpdateError("更新安装路径无效，已停止安装。")
    return value


def build_windows_installer_script(
    *,
    app_dir: Path,
    payload_dir: Path,
    staging_dir: Path,
    current_pid: int,
    owner_token: str,
) -> str:
    """Build a transactional installer bound to this staging area's owner token."""

    if current_pid <= 0 or not _OWNER_TOKEN_RE.fullmatch(owner_token):
        raise UpdateError("更新安装参数无效，已停止安装。")
    target = _batch_path(app_dir)
    payload = _batch_path(payload_dir)
    staging = _batch_path(staging_dir)
    return f"""@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "TARGET={target}"
set "PAYLOAD={payload}"
set "STAGING={staging}"
set "BACKUP_EXE=%TARGET%\\Lovart_Auto.exe.bak-{owner_token}"
set "BACKUP_INTERNAL=%TARGET%\\_internal.bak-{owner_token}"
set "EXE_BACKED_UP=0"
set "INTERNAL_BACKED_UP=0"
set "NEW_EXE_INSTALLED=0"
set "NEW_INTERNAL_INSTALLED=0"

powershell.exe -NoProfile -NonInteractive -Command "while (Get-Process -Id {current_pid} -ErrorAction SilentlyContinue) {{ Start-Sleep -Milliseconds 500 }}"
if errorlevel 1 goto rollback

if exist "%BACKUP_EXE%" goto rollback
if exist "%BACKUP_INTERNAL%" goto rollback
if not exist "%PAYLOAD%\\Lovart_Auto.exe" goto rollback
if not exist "%PAYLOAD%\\_internal\\python314.dll" goto rollback
if not exist "%PAYLOAD%\\_internal\\VCRUNTIME140.dll" goto rollback
if not exist "%PAYLOAD%\\_internal\\VCRUNTIME140_1.dll" goto rollback
if not exist "%TARGET%\\Lovart_Auto.exe" goto rollback
if not exist "%TARGET%\\_internal" goto rollback

ren "%TARGET%\\Lovart_Auto.exe" "Lovart_Auto.exe.bak-{owner_token}"
if errorlevel 1 goto rollback
set "EXE_BACKED_UP=1"
ren "%TARGET%\\_internal" "_internal.bak-{owner_token}"
if errorlevel 1 goto rollback
set "INTERNAL_BACKED_UP=1"

move /Y "%PAYLOAD%\\Lovart_Auto.exe" "%TARGET%\\Lovart_Auto.exe" >nul
if errorlevel 1 goto rollback
set "NEW_EXE_INSTALLED=1"
move /Y "%PAYLOAD%\\_internal" "%TARGET%\\_internal" >nul
if errorlevel 1 goto rollback
set "NEW_INTERNAL_INSTALLED=1"

if not exist "%TARGET%\\config.example.yaml" if exist "%PAYLOAD%\\config.example.yaml" (
  copy /Y "%PAYLOAD%\\config.example.yaml" "%TARGET%\\config.example.yaml" >nul
  if errorlevel 1 goto rollback
)
if not exist "%TARGET%\\.env.example" if exist "%PAYLOAD%\\.env.example" (
  copy /Y "%PAYLOAD%\\.env.example" "%TARGET%\\.env.example" >nul
  if errorlevel 1 goto rollback
)

"%TARGET%\\Lovart_Auto.exe" --run-main --help >nul 2>&1
if errorlevel 1 goto rollback

rmdir /s /q "%BACKUP_INTERNAL%"
if errorlevel 1 goto commit_failed
del /f /q "%BACKUP_EXE%"
if errorlevel 1 goto commit_failed
start "" "%TARGET%\\Lovart_Auto.exe"
if errorlevel 1 goto commit_failed
exit /b 0

:rollback
if "%NEW_INTERNAL_INSTALLED%"=="1" (
  rmdir /s /q "%TARGET%\\_internal"
  if errorlevel 1 goto rollback_failed
)
if "%NEW_EXE_INSTALLED%"=="1" (
  del /f /q "%TARGET%\\Lovart_Auto.exe"
  if errorlevel 1 goto rollback_failed
)
if "%INTERNAL_BACKED_UP%"=="1" (
  ren "%BACKUP_INTERNAL%" "_internal"
  if errorlevel 1 goto rollback_failed
)
if "%EXE_BACKED_UP%"=="1" (
  ren "%BACKUP_EXE%" "Lovart_Auto.exe"
  if errorlevel 1 goto rollback_failed
)
exit /b 1

:rollback_failed
exit /b 2

:commit_failed
exit /b 3
"""


def _write_and_start_installer(prepared: PreparedUpdate, app_dir: Path) -> None:
    owner_token = prepared.staging_dir.name
    content = build_windows_installer_script(
        app_dir=app_dir,
        payload_dir=prepared.payload_dir,
        staging_dir=prepared.staging_dir,
        current_pid=os.getpid(),
        owner_token=owner_token,
    )
    script_path = prepared.staging_dir / "install_update.bat"
    script_path.write_text(content, encoding="utf-8", newline="\r\n")
    creation_flags = 0
    if os.name == "nt":
        creation_flags = 0x00000008 | 0x00000200
    subprocess.Popen(
        ["cmd.exe", "/d", "/s", "/c", str(script_path)],
        cwd=str(app_dir),
        close_fds=True,
        creationflags=creation_flags,
    )


def download_and_install_update(
    url: str,
    output_queue=None,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    log=None,
) -> bool:
    """Compatibility entrypoint that installs only a fully verified frozen update."""

    def emit(message: str) -> None:
        if log is not None:
            log(message)
        elif output_queue is not None:
            output_queue.put(message)
        else:
            print(message)

    try:
        emit("开始下载更新包，请保持网络连接…")
        prepared = prepare_update(
            url,
            expected_sha256,
            expected_size,
            app_dir=application_directory(),
            log=emit,
        )
        if not getattr(sys, "frozen", False):
            emit("源码运行模式只完成下载与校验，不会覆盖源码目录。")
            shutil.rmtree(prepared.staging_dir, ignore_errors=True)
            return False
        if os.name != "nt":
            raise UpdateError("当前平台不支持自动安装，更新包已安全校验但未应用。")
        _write_and_start_installer(prepared, application_directory())
        emit("安全安装程序已启动；应用即将退出，校验成功后才会切换版本。")
        os._exit(0)
    except UpdateError as exc:
        emit(str(exc))
        return False
    except Exception:
        emit("启动安全安装程序失败，当前版本未被修改，请稍后重试。")
        return False
    finally:
        if output_queue is not None:
            output_queue.put(None)
