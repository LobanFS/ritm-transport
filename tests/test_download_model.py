"""Проверки установки модели: источник недоверенный, рабочие файлы сохраняются."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from tools import download_model


@pytest.fixture
def release(tmp_path):
    files = {name: f"verified {name}".encode() for name in download_model.REQUIRED_MODEL_FILES}
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    metadata = {
        "url": "https://example.test/model.zip",
        "sha256": download_model.sha256(archive),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(metadata))
    return archive, path, metadata, files


def test_offline_install_then_repeat_skips_download(tmp_path, monkeypatch, release):
    archive, config, _, files = release
    monkeypatch.setattr(download_model, "download_archive", lambda *_: pytest.fail("network access"))
    out = tmp_path / "artifacts/model"
    result = download_model.install_model(out=out, archive=archive, release_path=config)
    assert not result["already_present"]
    assert {path.name: path.read_bytes() for path in out.iterdir()} == files
    assert download_model.install_model(out=out, release_path=config)["already_present"]
    assert list(out.parent.iterdir()) == [out]


def test_download_uses_release_url_and_verifies_bytes(tmp_path, monkeypatch, release):
    archive, config, metadata, files = release
    calls = []
    def download(url, target):
        calls.append(url)
        target.write_bytes(archive.read_bytes())
    monkeypatch.setattr(download_model, "download_archive", download)
    out = tmp_path / "model"
    download_model.install_model(out=out, release_path=config)
    assert calls == [metadata["url"]]
    assert {path.name: path.read_bytes() for path in out.iterdir()} == files


def test_additional_release_file_is_installed_and_verified(tmp_path, release):
    archive, config, metadata, files = release
    content = b'{"reference": [1, 2, 3]}'
    files["explanation_reference.json"] = content
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("explanation_reference.json", content)
    metadata["sha256"] = download_model.sha256(archive)
    metadata["files"]["explanation_reference.json"] = hashlib.sha256(content).hexdigest()
    config.write_text(json.dumps(metadata))
    out = tmp_path / "model"
    download_model.install_model(out=out, archive=archive, release_path=config)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == files
    assert download_model.bundle_valid(out, metadata)


@pytest.mark.parametrize("corruption", ["archive", "file"])
def test_hash_failure_preserves_previous_model(tmp_path, release, corruption):
    archive, config, metadata, _ = release
    metadata["sha256" if corruption == "archive" else "files"] = (
        "0" * 64 if corruption == "archive" else {**metadata["files"], "encoder.pt": "0" * 64})
    config.write_text(json.dumps(metadata))
    out = tmp_path / "model"
    out.mkdir()
    (out / "encoder.pt").write_bytes(b"previous")
    with pytest.raises(ValueError, match="SHA-256"):
        download_model.install_model(out=out, archive=archive, release_path=config)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == {"encoder.pt": b"previous"}
    assert not list(tmp_path.glob(".model.*"))


@pytest.mark.parametrize("extra", ["../outside", "/absolute", "nested/model.joblib", "unused.txt", "model.joblib"])
def test_archive_member_allowlist_rejects_extras_paths_and_duplicates(tmp_path, release, extra):
    archive, config, metadata, _ = release
    with zipfile.ZipFile(archive, "a") as bundle:
        with pytest.warns(UserWarning) if extra == "model.joblib" else nullcontext():
            bundle.writestr(extra, b"unexpected")
    metadata["sha256"] = download_model.sha256(archive)
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="совпадать с описанием"):
        download_model.install_model(out=tmp_path / "model", archive=archive, release_path=config)
    assert not (tmp_path / "model").exists()
    assert not (tmp_path / "outside").exists()


def test_archive_symlink_rejected_before_model_install(tmp_path, release):
    archive, config, metadata, files = release
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in files.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, content)
    metadata["sha256"] = download_model.sha256(archive)
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="тип файла"):
        download_model.install_model(out=tmp_path / "model", archive=archive, release_path=config)
    assert not (tmp_path / "model").exists()


def test_failed_final_rename_rolls_back_complete_previous_bundle(tmp_path, monkeypatch, release):
    archive, config, _, _ = release
    out = tmp_path / "model"
    out.mkdir()
    for name in download_model.REQUIRED_MODEL_FILES:
        (out / name).write_text("previous")
    original_replace = download_model.os.replace
    def replace(source, target):
        if Path(source).name == "verified":
            raise OSError("simulated rename failure")
        return original_replace(source, target)
    monkeypatch.setattr(download_model.os, "replace", replace)
    with pytest.raises(OSError, match="rename failure"):
        download_model.install_model(out=out, archive=archive, release_path=config)
    assert {path.name: path.read_text() for path in out.iterdir()} == {
        name: "previous" for name in download_model.REQUIRED_MODEL_FILES}


def test_repair_replaces_incomplete_bundle_after_validation(tmp_path, release):
    archive, config, _, files = release
    out = tmp_path / "model"
    out.mkdir()
    (out / "encoder.pt").write_bytes(b"broken")
    download_model.install_model(out=out, archive=archive, release_path=config)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == files


def test_backup_is_retained_even_if_rollback_fails(tmp_path, monkeypatch, release):
    archive, config, _, _ = release
    out = tmp_path / "model"
    out.mkdir()
    (out / "encoder.pt").write_text("previous")
    original_replace = download_model.os.replace
    def replace(source, target):
        if Path(source).name in {"verified", "previous"}:
            raise OSError("simulated filesystem failure")
        return original_replace(source, target)
    monkeypatch.setattr(download_model.os, "replace", replace)
    with pytest.raises(OSError, match="Резервная копия"):
        download_model.install_model(out=out, archive=archive, release_path=config)
    backup, = tmp_path.glob(".model.download-*/previous/encoder.pt")
    assert backup.read_text() == "previous"


def test_replacement_does_not_delete_unrelated_files(tmp_path, release):
    archive, config, _, _ = release
    out = tmp_path / "model"
    out.mkdir()
    (out / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="посторонние файлы"):
        download_model.install_model(out=out, archive=archive, release_path=config)
    assert (out / "important.txt").read_text() == "keep"


def test_does_not_follow_target_symlink(tmp_path, release):
    archive, config, _, _ = release
    target = tmp_path / "elsewhere"
    target.mkdir()
    out = tmp_path / "model"
    out.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="симлинком"):
        download_model.install_model(out=out, archive=archive, release_path=config)
    assert not list(target.iterdir())


def test_active_install_lock_is_not_removed(tmp_path, release):
    archive, config, _, _ = release
    lock = tmp_path / ".model.install.lock"
    lock.mkdir()
    with pytest.raises(ValueError, match="Установка уже идёт"):
        download_model.install_model(out=tmp_path / "model", archive=archive, release_path=config)
    assert lock.is_dir()
