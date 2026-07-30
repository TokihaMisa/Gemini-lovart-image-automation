from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import http.client
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
_OWNER_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    ("con", "prn", "aux", "nul", "clock$")
    + tuple(f"com{number}" for number in range(1, 10))
    + tuple(f"lpt{number}" for number in range(1, 10))
)
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ZIP_ENTRIES = 8_192
_MAX_ZIP_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SINGLE_ENTRY_BYTES = 1024 * 1024 * 1024
_MAX_ENTRY_COMPRESSION_RATIO = 500
_MAX_OVERALL_COMPRESSION_RATIO = 100
_OVERALL_RATIO_MIN_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_INSTALL_LOCK_GRACE_SECONDS = 15.0
_PREPARE_CLAIM_GRACE_SECONDS = 15.0
_INSTALL_LOCK_NAME = "install.lock"
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


def _require_https_response(response, action: str) -> None:
    geturl = getattr(response, "geturl", None)
    final_url = geturl() if callable(geturl) else None
    if (
        not isinstance(final_url, str)
        or urlsplit(final_url).scheme.lower() != "https"
        or not urlsplit(final_url).netloc
    ):
        raise UpdateError(f"{action}发生不安全的 HTTPS 降级，已停止操作。")


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
            _require_https_response(response, "更新信息请求")
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
            _require_https_response(response, "更新包下载")
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
            raise http.client.IncompleteRead(b"", expected_size - downloaded)
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
    if not infos or len(infos) >= _MAX_ZIP_ENTRIES:
        raise UpdateError("更新包文件数量异常，已停止安装。")
    total_size = 0
    total_compressed_size = 0
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
        if info.file_size < 0 or info.file_size >= _MAX_SINGLE_ENTRY_BYTES:
            raise UpdateError("更新包中的文件大小异常，已停止安装。")
        total_size += info.file_size
        total_compressed_size += max(0, info.compress_size)
        if total_size >= _MAX_ZIP_UNCOMPRESSED_BYTES:
            raise UpdateError("更新包解压后体积异常，已停止安装。")
        if (
            info.file_size > 1024 * 1024
            and (
                info.compress_size <= 0
                or info.file_size / info.compress_size > _MAX_ENTRY_COMPRESSION_RATIO
            )
        ):
            raise UpdateError("更新包压缩比例异常，已停止安装。")
        if (
            total_size >= _OVERALL_RATIO_MIN_UNCOMPRESSED_BYTES
            and (
                total_compressed_size <= 0
                or total_size / total_compressed_size
                > _MAX_OVERALL_COMPRESSION_RATIO
            )
        ):
            raise UpdateError("更新包总体压缩比例异常，已停止安装。")
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
    archive_part = staging_dir / "update.zip.part"
    archive_path = staging_dir / "update.zip"
    payload_dir = staging_dir / "payload"
    claim_dir: Path | None = None
    staging_created = False
    try:
        claim_dir = _create_prepare_claim(staging_parent, owner_token)
        staging_dir.mkdir()
        staging_created = True
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
        if staging_created:
            shutil.rmtree(staging_dir, ignore_errors=True)
        if claim_dir is not None:
            _release_prepare_claim(staging_parent, owner_token)
        raise
    except Exception as exc:
        if staging_created:
            shutil.rmtree(staging_dir, ignore_errors=True)
        if claim_dir is not None:
            _release_prepare_claim(staging_parent, owner_token)
        raise UpdateError(_safe_network_guidance(exc, "下载更新")) from None


def _batch_path(path: Path) -> str:
    value = str(path.resolve())
    if any(character in value for character in ('"', "%", "!", "^", "\r", "\n")):
        raise UpdateError("更新安装路径无效，已停止安装。")
    return value


def _prepare_claim_path(update_root: Path, owner_token: str) -> Path:
    return update_root / f"prepare-{owner_token}.claim"


def _prepare_claim_token(claim_dir: Path) -> str | None:
    try:
        token = (claim_dir / "token").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return token if _OWNER_TOKEN_RE.fullmatch(token) else None


def _create_prepare_claim(update_root: Path, owner_token: str) -> Path:
    """Publish preparation ownership without taking the global install lock."""

    if not _OWNER_TOKEN_RE.fullmatch(owner_token):
        raise UpdateError("更新准备参数无效，已停止操作。")
    identity = _process_identity(os.getpid())
    if identity is None:
        raise UpdateError("无法确认更新准备进程，已停止操作。")
    claim_dir = _prepare_claim_path(update_root, owner_token)
    claim_dir.mkdir()
    try:
        (claim_dir / "created").write_text(
            str(int(time.time())), encoding="ascii"
        )
        (claim_dir / "token").write_text(owner_token, encoding="ascii")
        (claim_dir / "pid").write_text(str(os.getpid()), encoding="ascii")
        (claim_dir / "identity").write_text(identity, encoding="ascii")
        return claim_dir
    except Exception:
        shutil.rmtree(claim_dir, ignore_errors=True)
        raise


def _prepare_claim_owned_by_current_process(
    update_root: Path, owner_token: str
) -> bool:
    claim_dir = _prepare_claim_path(update_root, owner_token)
    if _prepare_claim_token(claim_dir) != owner_token:
        return False
    try:
        pid = int((claim_dir / "pid").read_text(encoding="ascii").strip())
        identity = (claim_dir / "identity").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError, ValueError):
        return False
    return pid == os.getpid() and _process_identity(pid) == identity


def _release_prepare_claim(update_root: Path, owner_token: str) -> bool:
    """Remove only the prepare claim that still belongs to this token."""

    claim_dir = _prepare_claim_path(update_root, owner_token)
    if not claim_dir.exists():
        return True
    if _prepare_claim_token(claim_dir) != owner_token:
        return False
    released = update_root / (
        f"prepare-{owner_token}.claim.release-{secrets.token_hex(8)}"
    )
    try:
        claim_dir.replace(released)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if _prepare_claim_token(released) != owner_token:
        try:
            if not claim_dir.exists():
                released.replace(claim_dir)
        except OSError:
            pass
        return False
    shutil.rmtree(released, ignore_errors=True)
    return not released.exists()


def _process_identity(pid: int) -> str | None:
    """Return a live process creation identity, not just its reusable PID."""

    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = (
                ("low", wintypes.DWORD),
                ("high", wintypes.DWORD),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            if (
                not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                or exit_code.value != 259
            ):
                return None
            created = FILETIME()
            exited = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            return str((created.high << 32) | created.low)
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    try:
        return str((Path("/proc") / str(pid)).stat().st_ctime_ns)
    except OSError:
        return f"pid-{pid}"


def _installer_lock_token(lock_dir: Path) -> str | None:
    try:
        token = (lock_dir / "token").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return token if _OWNER_TOKEN_RE.fullmatch(token) else None


def _installer_lock_is_busy(lock_dir: Path) -> bool:
    try:
        pid_text = (lock_dir / "installer.pid").read_text(encoding="ascii").strip()
        pid = int(pid_text)
    except (OSError, ValueError):
        pid = 0
    if pid > 0:
        live_identity = _process_identity(pid)
        if live_identity is None:
            return False
        try:
            recorded_identity = (
                lock_dir / "installer.identity"
            ).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return True
        return live_identity == recorded_identity
    try:
        age = max(0.0, time.time() - lock_dir.stat().st_mtime)
    except OSError:
        return True
    return age < _INSTALL_LOCK_GRACE_SECONDS


def _release_installer_lock(update_root: Path, owner_token: str) -> bool:
    """Release only a lock whose token still belongs to this owner."""

    lock_dir = update_root / _INSTALL_LOCK_NAME
    if not lock_dir.exists():
        return True
    if _installer_lock_token(lock_dir) != owner_token:
        return False
    released = update_root / f"{_INSTALL_LOCK_NAME}.release-{secrets.token_hex(8)}"
    try:
        lock_dir.replace(released)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if _installer_lock_token(released) != owner_token:
        try:
            if not lock_dir.exists():
                released.replace(lock_dir)
        except OSError:
            pass
        return False
    shutil.rmtree(released, ignore_errors=True)
    return not released.exists()


def _reserve_installer_lock(update_root: Path, owner_token: str) -> Path:
    """Atomically reserve the application-wide installer lock."""

    if not _OWNER_TOKEN_RE.fullmatch(owner_token):
        raise UpdateError("更新安装参数无效，已停止安装。")
    update_root.mkdir(parents=True, exist_ok=True)
    lock_dir = update_root / _INSTALL_LOCK_NAME
    for _attempt in range(3):
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if _installer_lock_is_busy(lock_dir):
                raise UpdateError("另一个更新安装正在进行，请稍后重试。")
            stale = update_root / f"{_INSTALL_LOCK_NAME}.stale-{secrets.token_hex(8)}"
            try:
                lock_dir.replace(stale)
            except FileNotFoundError:
                continue
            except OSError:
                raise UpdateError("另一个更新安装正在进行，请稍后重试。") from None
            shutil.rmtree(stale, ignore_errors=True)
            continue
        try:
            (lock_dir / "token").write_text(owner_token, encoding="ascii")
            (lock_dir / "created").write_text(
                str(time.time_ns()), encoding="ascii"
            )
            return lock_dir
        except Exception:
            _release_installer_lock(update_root, owner_token)
            raise
    raise UpdateError("另一个更新安装正在进行，请稍后重试。")


def _record_installer_process(
    lock_dir: Path, owner_token: str, process: subprocess.Popen
) -> None:
    if _installer_lock_token(lock_dir) != owner_token:
        raise UpdateError("更新安装锁所有者不匹配，已停止安装。")
    identity = _process_identity(process.pid)
    if identity is None:
        raise UpdateError("无法确认安全安装进程，当前应用将继续运行。")
    for name, value in (
        ("installer.pid", str(process.pid)),
        ("installer.identity", identity),
    ):
        temporary = lock_dir / f"{name}.tmp-{owner_token}"
        temporary.write_text(value, encoding="ascii")
        temporary.replace(lock_dir / name)


def _mark_installer_handoff_ready(lock_dir: Path, owner_token: str) -> None:
    if _installer_lock_token(lock_dir) != owner_token:
        raise UpdateError("更新安装锁所有者不匹配，已停止安装。")
    temporary = lock_dir / f"handoff.ready.tmp-{owner_token}"
    temporary.write_text(owner_token, encoding="ascii")
    temporary.replace(lock_dir / "handoff.ready")


_INSTALLER_FAILURE_STEPS = frozenset(
    (
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
)


def build_windows_installer_script(
    *,
    app_dir: Path,
    payload_dir: Path,
    staging_dir: Path,
    current_pid: int,
    owner_token: str,
    failure_step: str | None = None,
) -> str:
    """Build a transactional installer bound to this staging area's owner token."""

    if (
        current_pid <= 0
        or not _OWNER_TOKEN_RE.fullmatch(owner_token)
        or (failure_step is not None and failure_step not in _INSTALLER_FAILURE_STEPS)
    ):
        raise UpdateError("更新安装参数无效，已停止安装。")
    target = _batch_path(app_dir)
    payload = _batch_path(payload_dir)
    staging = _batch_path(staging_dir)
    update_root = _batch_path(app_dir / ".lovart-update")
    injected_failure = failure_step or ""
    return f"""@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
set "TARGET={target}"
set "PAYLOAD={payload}"
set "STAGING={staging}"
set "UPDATE_ROOT={update_root}"
set "TOKEN={owner_token}"
set "FAILURE_STEP={injected_failure}"
set "OWNER_MARKER=!UPDATE_ROOT!\\install-{owner_token}.owner"
set "STARTED_MARKER=!UPDATE_ROOT!\\install-{owner_token}.started"
set "PENDING_MARKER=!UPDATE_ROOT!\\transaction-{owner_token}.pending"
set "COMMIT_MARKER=!UPDATE_ROOT!\\transaction-{owner_token}.commit"
set "CLEANUP_MARKER=!UPDATE_ROOT!\\cleanup-pending-{owner_token}"
set "LOG_FILE=!UPDATE_ROOT!\\last-install.log"
set "SCRIPT_PATH=%~f0"
set "LOCK_DIR=!UPDATE_ROOT!\\install.lock"
set "LOCK_TOKEN_FILE=!LOCK_DIR!\\token"
set "LOCK_PID_FILE=!LOCK_DIR!\\installer.pid"
set "LOCK_IDENTITY_FILE=!LOCK_DIR!\\installer.identity"
set "LOCK_READY_FILE=!LOCK_DIR!\\handoff.ready"
set "RUN_NOTE="

call :validate_lock
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto lock_rejected
if not exist "!OWNER_MARKER!" goto lock_rejected
set "OWNER_VALUE="
set /p OWNER_VALUE=<"!OWNER_MARKER!"
if /I not "!OWNER_VALUE!"=="!TOKEN!" goto lock_rejected
>"!STARTED_MARKER!" echo(!TOKEN!

powershell.exe -NoProfile -NonInteractive -Command "while (Get-Process -Id {current_pid} -ErrorAction SilentlyContinue) {{ Start-Sleep -Milliseconds 500 }}"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto abort

call :recover_stale
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto recovery_failed
call :cleanup_orphans
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto recovery_failed

if not exist "!PAYLOAD!\\Lovart_Auto.exe" goto abort
if not exist "!PAYLOAD!\\_internal\\python314.dll" goto abort
if not exist "!PAYLOAD!\\_internal\\VCRUNTIME140.dll" goto abort
if not exist "!PAYLOAD!\\_internal\\VCRUNTIME140_1.dll" goto abort
if not exist "!TARGET!\\Lovart_Auto.exe" goto abort
if not exist "!TARGET!\\_internal" goto abort

>"!PENDING_MARKER!" echo(!TOKEN!
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto abort

call :inject backup_exe
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
ren "!TARGET!\\Lovart_Auto.exe" "Lovart_Auto.exe.bak-{owner_token}"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

call :inject backup_internal
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
ren "!TARGET!\\_internal" "_internal.bak-{owner_token}"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

if exist "!PAYLOAD!\\config.example.yaml" (
  call :inject backup_config
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
  if exist "!TARGET!\\config.example.yaml" (
    ren "!TARGET!\\config.example.yaml" "config.example.yaml.bak-{owner_token}"
    set "STEP_RC=!ERRORLEVEL!"
    if not "!STEP_RC!"=="0" goto rollback
  ) else (
    >"!UPDATE_ROOT!\\config.example.yaml.absent-{owner_token}" echo absent
    set "STEP_RC=!ERRORLEVEL!"
    if not "!STEP_RC!"=="0" goto rollback
  )
)
if exist "!PAYLOAD!\\.env.example" (
  call :inject backup_env
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
  if exist "!TARGET!\\.env.example" (
    ren "!TARGET!\\.env.example" ".env.example.bak-{owner_token}"
    set "STEP_RC=!ERRORLEVEL!"
    if not "!STEP_RC!"=="0" goto rollback
  ) else (
    >"!UPDATE_ROOT!\\.env.example.absent-{owner_token}" echo absent
    set "STEP_RC=!ERRORLEVEL!"
    if not "!STEP_RC!"=="0" goto rollback
  )
)

call :inject install_exe
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
move /Y "!PAYLOAD!\\Lovart_Auto.exe" "!TARGET!\\Lovart_Auto.exe" >nul
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

call :inject install_internal
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
move /Y "!PAYLOAD!\\_internal" "!TARGET!\\_internal" >nul
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

if exist "!PAYLOAD!\\config.example.yaml" (
  call :inject copy_config
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
  copy /Y "!PAYLOAD!\\config.example.yaml" "!TARGET!\\config.example.yaml" >nul
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
)
if exist "!PAYLOAD!\\.env.example" (
  call :inject copy_env
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
  copy /Y "!PAYLOAD!\\.env.example" "!TARGET!\\.env.example" >nul
  set "STEP_RC=!ERRORLEVEL!"
  if not "!STEP_RC!"=="0" goto rollback
)

call :inject smoke
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
"!TARGET!\\Lovart_Auto.exe" --run-main --help >nul 2>&1
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

call :inject commit
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback
move /Y "!PENDING_MARKER!" "!COMMIT_MARKER!" >nul
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto rollback

start "" /b "!TARGET!\\Lovart_Auto.exe"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" goto launch_failed

set "CLEANUP_PENDING=0"
call :cleanup_committed "!TOKEN!"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" set "CLEANUP_PENDING=1"
call :cleanup_current
set "CLEANUP_RC=!ERRORLEVEL!"
if not "!CLEANUP_RC!"=="0" set "CLEANUP_PENDING=1"
if "!CLEANUP_PENDING!"=="1" (
  >"!CLEANUP_MARKER!" echo(!TOKEN!
  >"!LOG_FILE!" echo success cleanup_pending=!TOKEN! !RUN_NOTE!
  goto finish_success
)
if exist "!CLEANUP_MARKER!" del /f /q "!CLEANUP_MARKER!" >nul 2>&1
>"!LOG_FILE!" echo success !RUN_NOTE!
goto finish_success

:rollback
call :rollback_token "!TOKEN!"
set "ROLLBACK_RC=!ERRORLEVEL!"
if not "!ROLLBACK_RC!"=="0" goto rollback_failed
call :cleanup_current
set "CLEANUP_RC=!ERRORLEVEL!"
if not "!CLEANUP_RC!"=="0" goto rollback_cleanup_failed
if exist "!PENDING_MARKER!" (
  del /f /q "!PENDING_MARKER!" >nul 2>&1
  if exist "!PENDING_MARKER!" goto rollback_cleanup_failed
)
>"!LOG_FILE!" echo rolled_back
goto finish_failure

:abort
>"!LOG_FILE!" echo rejected
call :cleanup_current
goto finish_failure

:recovery_failed
>"!LOG_FILE!" echo recovery_failed
call :cleanup_current
goto finish_failure

:rollback_failed
>"!LOG_FILE!" echo rollback_failed
call :cleanup_current
goto finish_rollback_failed

:rollback_cleanup_failed
>"!LOG_FILE!" echo rollback_cleanup_failed
goto finish_rollback_failed

:launch_failed
>"!LOG_FILE!" echo launch_failed
call :cleanup_current
goto finish_rollback_failed

:lock_rejected
call :release_lock
exit /b 1

:validate_lock
if not exist "!LOCK_TOKEN_FILE!" exit /b 70
set "LOCK_TOKEN_VALUE="
set /p LOCK_TOKEN_VALUE=<"!LOCK_TOKEN_FILE!"
if /I not "!LOCK_TOKEN_VALUE!"=="!TOKEN!" exit /b 71
set /a LOCK_WAIT_COUNT=0
:wait_lock_metadata
if exist "!LOCK_PID_FILE!" if exist "!LOCK_IDENTITY_FILE!" if exist "!LOCK_READY_FILE!" goto verify_lock_metadata
set /a LOCK_WAIT_COUNT+=1
if !LOCK_WAIT_COUNT! GEQ 250 exit /b 72
powershell.exe -NoProfile -NonInteractive -Command "Start-Sleep -Milliseconds 20"
goto wait_lock_metadata
:verify_lock_metadata
set "LOCK_PID="
set "LOCK_IDENTITY="
set "LOCK_READY="
set /p LOCK_PID=<"!LOCK_PID_FILE!"
set /p LOCK_IDENTITY=<"!LOCK_IDENTITY_FILE!"
set /p LOCK_READY=<"!LOCK_READY_FILE!"
if /I not "!LOCK_READY!"=="!TOKEN!" exit /b 73
echo(!LOCK_PID!| findstr.exe /r /x "[1-9][0-9]*" >nul
if errorlevel 1 exit /b 73
echo(!LOCK_IDENTITY!| findstr.exe /r /x "[1-9][0-9]*" >nul
if errorlevel 1 exit /b 73
powershell.exe -NoProfile -NonInteractive -Command "$p = Get-Process -Id !LOCK_PID! -ErrorAction Stop; if (([string]$p.StartTime.ToFileTimeUtc()) -ne '!LOCK_IDENTITY!') {{ exit 1 }}"
if errorlevel 1 exit /b 74
exit /b 0

:recover_stale
set "STALE_TOKEN="
set "STALE_MODE="
for %%F in ("!UPDATE_ROOT!\\transaction-*.pending") do if exist "%%~fF" (
  set "CANDIDATE=%%~nF"
  set "CANDIDATE=!CANDIDATE:transaction-=!"
  if defined STALE_TOKEN exit /b 21
  set "STALE_TOKEN=!CANDIDATE!"
  set "STALE_MODE=pending"
)
for %%F in ("!UPDATE_ROOT!\\transaction-*.commit") do if exist "%%~fF" (
  set "CANDIDATE=%%~nF"
  set "CANDIDATE=!CANDIDATE:transaction-=!"
  if defined STALE_TOKEN exit /b 21
  set "STALE_TOKEN=!CANDIDATE!"
  set "STALE_MODE=commit"
)
if defined STALE_TOKEN goto recover_found
set "LEGACY_EXE_TOKEN="
set "LEGACY_INTERNAL_TOKEN="
set /a LEGACY_EXE_COUNT=0
set /a LEGACY_INTERNAL_COUNT=0
for %%F in ("!TARGET!\\Lovart_Auto.exe.bak-*") do if exist "%%~fF" (
  set /a LEGACY_EXE_COUNT+=1
  set "CANDIDATE=%%~nxF"
  set "CANDIDATE=!CANDIDATE:Lovart_Auto.exe.bak-=!"
  set "LEGACY_EXE_TOKEN=!CANDIDATE!"
)
for /d %%F in ("!TARGET!\\_internal.bak-*") do if exist "%%~fF" (
  set /a LEGACY_INTERNAL_COUNT+=1
  set "CANDIDATE=%%~nxF"
  set "CANDIDATE=!CANDIDATE:_internal.bak-=!"
  set "LEGACY_INTERNAL_TOKEN=!CANDIDATE!"
)
if !LEGACY_EXE_COUNT! GTR 1 exit /b 22
if !LEGACY_INTERNAL_COUNT! GTR 1 exit /b 22
if defined LEGACY_EXE_TOKEN if not defined LEGACY_INTERNAL_TOKEN exit /b 23
if defined LEGACY_INTERNAL_TOKEN if not defined LEGACY_EXE_TOKEN exit /b 23
if defined LEGACY_EXE_TOKEN if /I not "!LEGACY_EXE_TOKEN!"=="!LEGACY_INTERNAL_TOKEN!" exit /b 23
if not defined LEGACY_EXE_TOKEN exit /b 0
set "STALE_TOKEN=!LEGACY_EXE_TOKEN!"
set "STALE_MODE=legacy"
>"!UPDATE_ROOT!\\transaction-!STALE_TOKEN!.pending" echo(!STALE_TOKEN!
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" exit /b !STEP_RC!

:recover_found
if /I "!STALE_MODE!"=="commit" (
  call :cleanup_committed "!STALE_TOKEN!"
) else (
  call :rollback_token "!STALE_TOKEN!"
)
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" exit /b !STEP_RC!
call :cleanup_stale "!STALE_TOKEN!"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" exit /b !STEP_RC!
if /I not "!STALE_MODE!"=="commit" (
  set "STALE_PENDING=!UPDATE_ROOT!\\transaction-!STALE_TOKEN!.pending"
  if exist "!STALE_PENDING!" (
    del /f /q "!STALE_PENDING!" >nul 2>&1
    if exist "!STALE_PENDING!" exit /b 24
  )
)
set "RUN_NOTE=recovered=!STALE_TOKEN!"
exit /b 0

:cleanup_orphans
set /a ORPHAN_COUNT=0
for %%F in ("!UPDATE_ROOT!\\cleanup-pending-*") do if exist "%%~fF" (
  set /a ORPHAN_COUNT+=1
  if !ORPHAN_COUNT! LEQ 16 call :cleanup_orphan_marker "%%~fF"
)
for %%F in ("!UPDATE_ROOT!\\install-*.owner") do if exist "%%~fF" (
  set /a ORPHAN_COUNT+=1
  if !ORPHAN_COUNT! LEQ 16 call :cleanup_orphan_owner "%%~fF"
)
for /d %%C in ("!UPDATE_ROOT!\\prepare-*.claim") do if exist "%%~fC\\" (
  set /a ORPHAN_COUNT+=1
  if !ORPHAN_COUNT! LEQ 16 call :cleanup_orphan_claim "%%~fC"
)
for /d %%D in ("!UPDATE_ROOT!\\*") do if exist "%%~fD\\" (
  set /a ORPHAN_COUNT+=1
  if !ORPHAN_COUNT! LEQ 16 call :cleanup_orphan_staging "%%~fD"
)
exit /b 0

:cleanup_orphan_marker
set "O_PATH=%~1"
set "O_NAME=%~n1"
set "O_TOKEN=!O_NAME:cleanup-pending-=!"
if /I "!O_TOKEN!"=="!TOKEN!" exit /b 0
if not "!O_TOKEN:~64,1!"=="" exit /b 0
set "O_VALUE="
set /p O_VALUE=<"!O_PATH!"
if /I not "!O_VALUE!"=="!O_TOKEN!" exit /b 0
echo(!O_TOKEN!| findstr.exe /r /x "[A-Za-z0-9-][A-Za-z0-9-]*" >nul
if errorlevel 1 exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.pending" exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.commit" exit /b 0
call :cleanup_stale "!O_TOKEN!"
if errorlevel 1 exit /b 0
del /f /q "!O_PATH!" >nul 2>&1
if exist "!O_PATH!" exit /b 0
set "RUN_NOTE=orphan_cleaned=!O_TOKEN!"
exit /b 0

:cleanup_orphan_owner
set "O_PATH=%~1"
set "O_NAME=%~n1"
set "O_TOKEN=!O_NAME:install-=!"
if /I "!O_TOKEN!"=="!TOKEN!" exit /b 0
if not "!O_TOKEN:~64,1!"=="" exit /b 0
set "O_VALUE="
set /p O_VALUE=<"!O_PATH!"
if /I not "!O_VALUE!"=="!O_TOKEN!" exit /b 0
echo(!O_TOKEN!| findstr.exe /r /x "[A-Za-z0-9-][A-Za-z0-9-]*" >nul
if errorlevel 1 exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.pending" exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.commit" exit /b 0
call :cleanup_stale "!O_TOKEN!"
if errorlevel 1 exit /b 0
set "RUN_NOTE=orphan_cleaned=!O_TOKEN!"
exit /b 0

:cleanup_orphan_claim
set "O_NAME=%~nx1"
set "O_TOKEN=!O_NAME:prepare-=!"
set "O_TOKEN=!O_TOKEN:.claim=!"
call :cleanup_prepare_orphan "!O_TOKEN!"
exit /b 0

:cleanup_orphan_staging
set "O_PATH=%~1"
set "O_TOKEN=%~nx1"
call :cleanup_prepare_orphan "!O_TOKEN!"
exit /b 0

:cleanup_prepare_orphan
set "O_TOKEN=%~1"
if /I "!O_TOKEN!"=="!TOKEN!" exit /b 0
if not "!O_TOKEN:~64,1!"=="" exit /b 0
echo(!O_TOKEN!| findstr.exe /r /x "[A-Za-z0-9-][A-Za-z0-9-]*" >nul
if errorlevel 1 exit /b 0
set "CLAIM_DIR=!UPDATE_ROOT!\\prepare-!O_TOKEN!.claim"
set "CLAIM_EXPECTED=!O_TOKEN!"
call :prepare_claim_active
if not errorlevel 1 exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.pending" exit /b 0
if exist "!UPDATE_ROOT!\\transaction-!O_TOKEN!.commit" exit /b 0
call :cleanup_stale "!O_TOKEN!"
if errorlevel 1 exit /b 0
if exist "!CLAIM_DIR!\\" (
  rmdir /s /q "!CLAIM_DIR!"
  if exist "!CLAIM_DIR!\\" exit /b 0
)
set "RUN_NOTE=orphan_cleaned=!O_TOKEN!"
exit /b 0

:prepare_claim_active
powershell.exe -NoProfile -NonInteractive -Command "$d = $env:CLAIM_DIR; $expected = $env:CLAIM_EXPECTED; $grace = {_PREPARE_CLAIM_GRACE_SECONDS}; if (-not (Test-Path -LiteralPath $d -PathType Container)) {{ exit 91 }}; $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); try {{ $created = [long](Get-Content -LiteralPath (Join-Path $d 'created') -Raw -ErrorAction Stop).Trim() }} catch {{ $created = [DateTimeOffset](Get-Item -LiteralPath $d).LastWriteTimeUtc; $created = $created.ToUnixTimeSeconds() }}; $age = $now - $created; $recent = ($age -ge 0) -and ($age -lt $grace); try {{ $claimToken = (Get-Content -LiteralPath (Join-Path $d 'token') -Raw -ErrorAction Stop).Trim(); $pidValue = [int](Get-Content -LiteralPath (Join-Path $d 'pid') -Raw -ErrorAction Stop).Trim(); $identity = (Get-Content -LiteralPath (Join-Path $d 'identity') -Raw -ErrorAction Stop).Trim() }} catch {{ if ($recent) {{ exit 0 }} else {{ exit 91 }} }}; if ($claimToken -cne $expected) {{ if ($recent) {{ exit 0 }} else {{ exit 91 }} }}; try {{ $p = Get-Process -Id $pidValue -ErrorAction Stop }} catch {{ exit 91 }}; if (([string]$p.StartTime.ToFileTimeUtc()) -eq $identity) {{ exit 0 }}; exit 91"
if "!ERRORLEVEL!"=="91" exit /b 1
exit /b 0

:rollback_token
set "R_TOKEN=%~1"
set "R_BACKUP_EXE=!TARGET!\\Lovart_Auto.exe.bak-!R_TOKEN!"
set "R_BACKUP_INTERNAL=!TARGET!\\_internal.bak-!R_TOKEN!"
set "R_BACKUP_CONFIG=!TARGET!\\config.example.yaml.bak-!R_TOKEN!"
set "R_BACKUP_ENV=!TARGET!\\.env.example.bak-!R_TOKEN!"
set "R_ABSENT_CONFIG=!UPDATE_ROOT!\\config.example.yaml.absent-!R_TOKEN!"
set "R_ABSENT_ENV=!UPDATE_ROOT!\\.env.example.absent-!R_TOKEN!"
if exist "!R_BACKUP_INTERNAL!\\" if not exist "!R_BACKUP_EXE!" exit /b 41
if exist "!R_BACKUP_INTERNAL!\\" (
  if exist "!TARGET!\\_internal" (
    rmdir /s /q "!TARGET!\\_internal"
    if exist "!TARGET!\\_internal" exit /b 42
  )
  ren "!R_BACKUP_INTERNAL!" "_internal"
  if exist "!R_BACKUP_INTERNAL!\\" exit /b 43
  if not exist "!TARGET!\\_internal\\" exit /b 43
)
if exist "!R_BACKUP_EXE!" (
  if exist "!TARGET!\\Lovart_Auto.exe" (
    del /f /q "!TARGET!\\Lovart_Auto.exe" >nul 2>&1
    if exist "!TARGET!\\Lovart_Auto.exe" exit /b 44
  )
  ren "!R_BACKUP_EXE!" "Lovart_Auto.exe"
  if exist "!R_BACKUP_EXE!" exit /b 45
  if not exist "!TARGET!\\Lovart_Auto.exe" exit /b 45
)
call :restore_asset "!R_BACKUP_CONFIG!" "config.example.yaml" "!R_ABSENT_CONFIG!"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" exit /b !STEP_RC!
call :restore_asset "!R_BACKUP_ENV!" ".env.example" "!R_ABSENT_ENV!"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" exit /b !STEP_RC!
exit /b 0

:restore_asset
set "A_BACKUP=%~1"
set "A_NAME=%~2"
set "A_ABSENT=%~3"
if exist "!A_BACKUP!" (
  if exist "!TARGET!\\!A_NAME!" (
    del /f /q "!TARGET!\\!A_NAME!" >nul 2>&1
    if exist "!TARGET!\\!A_NAME!" exit /b 46
  )
  ren "!A_BACKUP!" "!A_NAME!"
  if exist "!A_BACKUP!" exit /b 47
  if not exist "!TARGET!\\!A_NAME!" exit /b 47
) else if exist "!A_ABSENT!" (
  if exist "!TARGET!\\!A_NAME!" (
    del /f /q "!TARGET!\\!A_NAME!" >nul 2>&1
    if exist "!TARGET!\\!A_NAME!" exit /b 48
  )
  del /f /q "!A_ABSENT!" >nul 2>&1
  if exist "!A_ABSENT!" exit /b 49
)
exit /b 0

:cleanup_committed
set "C_TOKEN=%~1"
if not exist "!TARGET!\\Lovart_Auto.exe" exit /b 51
if not exist "!TARGET!\\_internal\\python314.dll" exit /b 51
if not exist "!TARGET!\\_internal\\VCRUNTIME140.dll" exit /b 51
if not exist "!TARGET!\\_internal\\VCRUNTIME140_1.dll" exit /b 51
set "C_BACKUP_INTERNAL=!TARGET!\\_internal.bak-!C_TOKEN!"
set "C_BACKUP_EXE=!TARGET!\\Lovart_Auto.exe.bak-!C_TOKEN!"
set "C_BACKUP_CONFIG=!TARGET!\\config.example.yaml.bak-!C_TOKEN!"
set "C_BACKUP_ENV=!TARGET!\\.env.example.bak-!C_TOKEN!"
set "C_ABSENT_CONFIG=!UPDATE_ROOT!\\config.example.yaml.absent-!C_TOKEN!"
set "C_ABSENT_ENV=!UPDATE_ROOT!\\.env.example.absent-!C_TOKEN!"
set "C_COMMIT=!UPDATE_ROOT!\\transaction-!C_TOKEN!.commit"
if exist "!C_BACKUP_INTERNAL!\\" (
  rmdir /s /q "!C_BACKUP_INTERNAL!"
  if exist "!C_BACKUP_INTERNAL!\\" exit /b 52
)
for %%F in ("!C_BACKUP_EXE!" "!C_BACKUP_CONFIG!" "!C_BACKUP_ENV!" "!C_ABSENT_CONFIG!" "!C_ABSENT_ENV!") do if exist "%%~fF" (
  del /f /q "%%~fF" >nul 2>&1
  if exist "%%~fF" exit /b 53
)
if exist "!C_COMMIT!" (
  del /f /q "!C_COMMIT!" >nul 2>&1
  if exist "!C_COMMIT!" exit /b 54
)
exit /b 0

:cleanup_stale
set "S_TOKEN=%~1"
set "S_STAGING=!UPDATE_ROOT!\\!S_TOKEN!"
if exist "!S_STAGING!\\" (
  rmdir /s /q "!S_STAGING!"
  if exist "!S_STAGING!\\" exit /b 61
)
for %%F in ("!UPDATE_ROOT!\\install-!S_TOKEN!.bat" "!UPDATE_ROOT!\\install-!S_TOKEN!.owner" "!UPDATE_ROOT!\\install-!S_TOKEN!.started") do if exist "%%~fF" (
  del /f /q "%%~fF" >nul 2>&1
  if exist "%%~fF" exit /b 62
)
exit /b 0

:cleanup_current
if exist "!STAGING!\\" (
  rmdir /s /q "!STAGING!"
  if exist "!STAGING!\\" exit /b 63
)
for %%F in ("!OWNER_MARKER!" "!STARTED_MARKER!") do if exist "%%~fF" (
  del /f /q "%%~fF" >nul 2>&1
  if exist "%%~fF" exit /b 64
)
exit /b 0

:inject
if /I "%~1"=="!FAILURE_STEP!" exit /b 97
exit /b 0

:release_lock
if not exist "!LOCK_DIR!\\" exit /b 0
if not exist "!LOCK_TOKEN_FILE!" exit /b 81
set "RELEASE_TOKEN="
set /p RELEASE_TOKEN=<"!LOCK_TOKEN_FILE!"
if /I not "!RELEASE_TOKEN!"=="!TOKEN!" exit /b 82
rmdir /s /q "!LOCK_DIR!"
if exist "!LOCK_DIR!\\" exit /b 83
exit /b 0

:finish_success
call :release_lock
if errorlevel 1 >>"!LOG_FILE!" echo lock_release_pending=!TOKEN!
exit /b 0

:finish_failure
call :release_lock
if errorlevel 1 >>"!LOG_FILE!" echo lock_release_pending=!TOKEN!
exit /b 1

:finish_rollback_failed
call :release_lock
if errorlevel 1 >>"!LOG_FILE!" echo lock_release_pending=!TOKEN!
exit /b 2
"""


def _windows_installer_command(script_path: Path) -> str:
    script = _batch_path(script_path)
    command_processor = os.environ.get("COMSPEC", "cmd.exe")
    if any(character in command_processor for character in ('"', "\r", "\n")):
        raise UpdateError("系统命令解释器路径无效，已停止安装。")
    return (
        f'"{command_processor}" /v:on /d /s /c '
        f'""{script}" & set "ota_rc=!errorlevel!" '
        f'& del /f /q "{script}" & exit /b !ota_rc!"'
    )


def _write_and_start_installer(
    prepared: PreparedUpdate,
    app_dir: Path,
    *,
    current_pid: int | None = None,
    start_timeout: float = 5.0,
):
    owner_token = prepared.staging_dir.name
    update_root = app_dir / ".lovart-update"
    update_root.mkdir(parents=True, exist_ok=True)
    script_path = update_root / f"install-{owner_token}.bat"
    owner_marker = update_root / f"install-{owner_token}.owner"
    started_marker = update_root / f"install-{owner_token}.started"
    content = build_windows_installer_script(
        app_dir=app_dir,
        payload_dir=prepared.payload_dir,
        staging_dir=prepared.staging_dir,
        current_pid=current_pid or os.getpid(),
        owner_token=owner_token,
    )
    lock_dir = _reserve_installer_lock(update_root, owner_token)
    process: subprocess.Popen | None = None
    claim_converted = False
    try:
        if not _prepare_claim_owned_by_current_process(update_root, owner_token):
            raise UpdateError("更新准备所有者不匹配或已失效，已停止安装。")
        started_marker.unlink(missing_ok=True)
        owner_marker.write_text(owner_token, encoding="ascii")
        script_path.write_text(content, encoding="utf-8", newline="\r\n")
        creation_flags = 0
        if os.name == "nt":
            creation_flags = 0x08000000 | 0x00000200
        process = subprocess.Popen(
            _windows_installer_command(script_path),
            cwd=str(app_dir),
            close_fds=True,
            creationflags=creation_flags,
        )
        _record_installer_process(lock_dir, owner_token, process)
        if not _release_prepare_claim(update_root, owner_token):
            raise UpdateError("无法安全转换更新准备所有权，当前应用将继续运行。")
        claim_converted = True
        _mark_installer_handoff_ready(lock_dir, owner_token)
        deadline = time.monotonic() + max(0.1, start_timeout)
        while time.monotonic() < deadline:
            if started_marker.exists():
                try:
                    if (
                        started_marker.read_text(encoding="utf-8").strip()
                        == owner_token
                    ):
                        return process
                except OSError:
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.02)
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for path in (started_marker, owner_marker, script_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if claim_converted:
            shutil.rmtree(prepared.staging_dir, ignore_errors=True)
        _release_installer_lock(update_root, owner_token)
        raise
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    for path in (started_marker, owner_marker, script_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if claim_converted:
        shutil.rmtree(prepared.staging_dir, ignore_errors=True)
    _release_installer_lock(update_root, owner_token)
    raise UpdateError("安全安装程序未确认接管更新，当前应用将继续运行。")


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

    prepared: PreparedUpdate | None = None
    handoff_started = False
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
        handoff_started = True
        emit("安全安装程序已启动；应用即将退出，校验成功后才会切换版本。")
        os._exit(0)
    except UpdateError as exc:
        emit(str(exc))
        return False
    except Exception:
        emit("启动安全安装程序失败，当前版本未被修改，请稍后重试。")
        return False
    finally:
        if prepared is not None and not handoff_started:
            shutil.rmtree(prepared.staging_dir, ignore_errors=True)
            update_root = application_directory() / ".lovart-update"
            token = prepared.staging_dir.name
            _release_prepare_claim(update_root, token)
            for suffix in ("bat", "owner", "started"):
                try:
                    (update_root / f"install-{token}.{suffix}").unlink(missing_ok=True)
                except OSError:
                    pass
        if output_queue is not None:
            output_queue.put(None)
