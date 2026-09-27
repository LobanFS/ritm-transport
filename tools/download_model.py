"""Скачать проверенную модель из GitHub Release. Нужен только Python 3.10+."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = ROOT / "config/model-release.json"
MODEL_DIR = ROOT / "artifacts/model"
REQUIRED_MODEL_FILES = {"encoder.pt", "manifest.json", "model.joblib", "probability.json"}
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_release(path: Path) -> dict:
    release = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(release, dict) or not isinstance(release.get("files"), dict):
        raise ValueError("Неверный формат config/model-release.json")
    if not REQUIRED_MODEL_FILES.issubset(release["files"]):
        raise ValueError("В описании релиза отсутствуют обязательные файлы модели")
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) for name in release["files"]):
        raise ValueError("Имена файлов релиза не должны содержать пути")
    hashes = [release.get("sha256"), *release["files"].values()]
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("Неверный SHA-256 в описании релиза")
    url = release.get("url")
    if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
        raise ValueError("URL модели должен использовать HTTPS")
    return release


def bundle_valid(out: Path, release: dict) -> bool:
    if not out.is_dir() or out.is_symlink():
        return False
    for name, expected in release["files"].items():
        path = out / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            return False
    return True


def download_archive(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "ritm-transport-model-download/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as target:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("Сервер перенаправил скачивание на адрес без HTTPS")
        size = 0
        while chunk := response.read(CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Архив модели превышает допустимый размер")
            target.write(chunk)


def verify_and_unpack(archive: Path, destination: Path, release: dict) -> None:
    """Записать файлы в staging; рабочая модель меняется только после проверки."""
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Архив модели превышает допустимый размер")
    if sha256(archive) != release["sha256"]:
        raise ValueError("SHA-256 архива не совпадает с опубликованным релизом")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(release["files"]) or set(names) != set(release["files"]):
            raise ValueError("Файлы архива должны точно совпадать с описанием релиза, без каталогов и повторов")
        if sum(entry.file_size for entry in entries) > MAX_UNPACKED_BYTES:
            raise ValueError("Распакованная модель превышает допустимый размер")
        for entry in entries:
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError(f"Недопустимый тип файла в архиве: {entry.filename}")
        destination.mkdir()
        for entry in entries:
            path = destination / entry.filename
            digest = hashlib.sha256()
            size = 0
            with bundle.open(entry) as source, path.open("wb") as target:
                while chunk := source.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > entry.file_size:
                        raise ValueError(f"Размер файла не совпадает: {entry.filename}")
                    digest.update(chunk)
                    target.write(chunk)
            if size != entry.file_size or digest.hexdigest() != release["files"][entry.filename]:
                raise ValueError(f"SHA-256 файла не совпадает: {entry.filename}")


def install_model(*, out: Path = MODEL_DIR, archive: Path | None = None,
                  release_path: Path = RELEASE_PATH) -> dict:
    """Подготовить целый проверенный bundle; повторный запуск не скачивает его."""
    release = read_release(Path(release_path))
    # absolute(), в отличие от resolve(), не скрывает симлинк в последнем компоненте.
    out = Path(out).absolute()
    if out.is_symlink():
        raise ValueError("Каталог назначения модели не должен быть симлинком")
    if bundle_valid(out, release):
        return {"path": str(out), "already_present": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(f".{out.name}.install.lock")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Установка уже идёт; проверьте каталог блокировки: {lock}") from error
    try:
        if bundle_valid(out, release):
            return {"path": str(out), "already_present": True}
        if out.exists():
            if not out.is_dir() or out.is_symlink():
                raise ValueError("Путь назначения должен быть обычным каталогом")
            if any(path.name not in release["files"] or not path.is_file() or path.is_symlink()
                   for path in out.iterdir()):
                raise ValueError("В каталоге назначения есть посторонние файлы; укажите отдельный --out")
        work = Path(tempfile.mkdtemp(prefix=f".{out.name}.download-", dir=out.parent))
        preserve_backup = False
        try:
            source = Path(archive).resolve() if archive is not None else work / "model.zip"
            if archive is None:
                download_archive(release["url"], source)
            staged = work / "verified"
            verify_and_unpack(source, staged, release)
            backup = work / "previous"
            had_previous = out.exists()
            if had_previous:
                os.replace(out, backup)
            try:
                os.replace(staged, out)
            except OSError:
                if had_previous:
                    try:
                        os.replace(backup, out)
                    except OSError as error:
                        preserve_backup = True
                        raise OSError(f"Не удалось восстановить каталог модели. Резервная копия: {backup}") from error
                raise
        finally:
            if not preserve_backup:
                shutil.rmtree(work)
        return {"path": str(out), "already_present": False}
    finally:
        lock.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Использовать скачанный ZIP релиза без обращения к сети")
    parser.add_argument("--out", type=Path, default=MODEL_DIR, help="Каталог модели (по умолчанию artifacts/model)")
    args = parser.parse_args()
    try:
        result = install_model(out=args.out, archive=args.archive)
    except (OSError, ValueError, URLError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, f"Не удалось подготовить модель: {error}\n")
    action = "Модель уже готова, SHA-256 проверены" if result["already_present"] else "Модель установлена, SHA-256 проверены"
    print(f"{action}: {result['path']}")


if __name__ == "__main__":
    main()
