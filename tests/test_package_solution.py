"""Отчёт сдачи не остаётся успешным после рассинхронизации или неудачи проверки."""
import hashlib
import json
from pathlib import Path
import shlex
import zipfile

import pytest

from tools.package_solution import DATASET_FILES, REPORTS, ROOT_FILES, SOURCE_DIRS, existing_reports, verify_archive


def test_release_contains_replay_inputs_with_original_checksums():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'dataset/manifest.json').read_text())
    assert set(DATASET_FILES) == {'dataset/manifest.json'} | {'dataset/'+name for name in manifest['files']}
    for name, metadata in manifest['files'].items():
        path = root/'dataset'/name
        assert path.stat().st_size == metadata['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']


def test_docker_build_inputs_are_present_in_the_release():
    root = Path(__file__).resolve().parents[1]
    for dockerfile in root.glob('*Dockerfile*'):
        assert dockerfile.name in ROOT_FILES, f'Образ отсутствует в ZIP: {dockerfile.name}'
        for line in dockerfile.read_text().splitlines():
            parts = shlex.split(line)
            if not parts or parts[0] != 'COPY' or any(p.startswith('--from=') for p in parts):
                continue
            for name in parts[1:-1]:
                if name.startswith('--'):
                    continue
                path = Path(name)
                assert path.parts[0] in SOURCE_DIRS or name in ROOT_FILES, f'{dockerfile.name}: COPY {name} отсутствует в ZIP'


def make_archive(tmp_path, *, content=b'current', digest_content=None, preflight=None):
    root = tmp_path/'source'
    root.mkdir(exist_ok=True)
    (root/'README.md').write_bytes(content)
    files = {'README.md': content}
    if preflight is not None:
        (root/'tools').mkdir(exist_ok=True)
        (root/'tools/start_solution.py').write_text(preflight)
        files['tools/start_solution.py'] = preflight.encode()
    manifest = {'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    if digest_content is not None:
        manifest['files']['README.md'] = hashlib.sha256(digest_content).hexdigest()
    archive = tmp_path/'ritm-solution.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, data in files.items():
            bundle.writestr('ritm-transport/'+name, data)
        bundle.writestr('ritm-transport/MANIFEST.json', json.dumps(manifest))
    return archive, root


def test_verification_belongs_to_current_archive_and_detects_working_drift(tmp_path):
    archive, root = make_archive(tmp_path)
    report = verify_archive(archive, root=root, preflight=False)
    assert report['passed'] and report['files'] == 1
    assert report['archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    (root/'README.md').write_text('changed after packaging')
    with pytest.raises(ValueError, match='Рабочий файл отличается'):
        verify_archive(archive, root=root, preflight=False)
    saved = json.loads((tmp_path/'verification.json').read_text())
    assert saved['passed'] is False
    assert saved['archive_sha256'] == report['archive_sha256']


def test_valid_zip_crc_is_not_enough_when_manifest_hash_is_wrong(tmp_path):
    archive, root = make_archive(tmp_path, digest_content=b'different')
    with pytest.raises(ValueError, match='Хеш файла ZIP'):
        verify_archive(archive, root=root, preflight=False)
    assert json.loads((tmp_path/'verification.json').read_text())['passed'] is False


@pytest.mark.parametrize('status, passed', [('inputs_valid', True), ('invalid', False)])
def test_preflight_is_run_from_extracted_archive_and_controls_pass(tmp_path, status, passed):
    script = "import json\nfrom pathlib import Path\nassert Path('README.md').read_text() == 'current'\nprint(json.dumps({'status': '"+status+"'}))\n"
    archive, root = make_archive(tmp_path, preflight=script)
    if passed:
        report = verify_archive(archive, root=root)
        assert report['passed'] and report['extracted_preflight']['status'] == 'inputs_valid'
    else:
        with pytest.raises(ValueError, match='preflight'):
            verify_archive(archive, root=root)
        assert json.loads((tmp_path/'verification.json').read_text())['passed'] is False


def test_missing_archive_replaces_stale_success(tmp_path):
    (tmp_path/'verification.json').write_text('{"passed": true}')
    with pytest.raises(FileNotFoundError):
        verify_archive(tmp_path/'missing.zip', root=tmp_path)
    assert json.loads((tmp_path/'verification.json').read_text())['passed'] is False


def test_release_reports_are_optional_and_only_existing_files_are_selected(tmp_path):
    first = tmp_path/REPORTS[0]
    last = tmp_path/REPORTS[-1]
    first.parent.mkdir(parents=True)
    last.parent.mkdir(parents=True, exist_ok=True)
    first.write_text('{}')
    last.write_text('{}')
    assert existing_reports(tmp_path) == (REPORTS[0], REPORTS[-1])
