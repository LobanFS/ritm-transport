"""Собирает автономный ZIP решения с моделью и проверяет его после распаковки."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('backend','common','ml_service','generator','dashboard','config','tests','tools','training','docs')
ROOT_FILES = ('README.md','.gitignore','.gitattributes','.dockerignore','Dockerfile','dashboard.Dockerfile','Dockerfile.train',
              'pytest.ini','requirements.txt','requirements-dev.txt','compose.yaml','compose.model.yaml',
              'compose.replay.yaml','compose.emulator.yaml','compose.learning.yaml','compose.retrain.yaml','compose.scale.yaml','docs/.nojekyll')
REPORTS = ('artifacts/release-check/summary.json','artifacts/release-check/performance.json')


def existing_reports(root: Path = ROOT) -> tuple[str, ...]:
    """Вернуть только уже созданные отчёты, не требуя их для поставки."""
    return tuple(name for name in REPORTS if (root/name).is_file())


def verify_archive(archive: Path, *, root: Path, preflight=True):
    """Проверка именно собранного ZIP; не запускает Docker или сетевые вызовы.

    verification.json пишется и при исключении, всегда с passed=false.
    Старый успешный отчёт не может остаться после новой неудачной проверки.
    """
    report = dict(passed=False, created_at=datetime.now(timezone.utc).isoformat(),
                  scope='ZIP CRC, все manifest-хеши и соответствие рабочим файлам; отдельная распаковка и preflight без Docker')
    try:
        report.update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), bytes=archive.stat().st_size)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or bundle.testzip() is not None:
                raise ValueError('Повтор ZIP entry или ошибка CRC')
            manifest = json.loads(bundle.read('ritm-transport/MANIFEST.json'))
            expected = {'ritm-transport/'+name for name in manifest['files']} | {'ritm-transport/MANIFEST.json'}
            if set(names) != expected:
                raise ValueError('Состав ZIP не совпадает с MANIFEST.json')
            for name, digest in manifest['files'].items():
                path = Path(name)
                if path.is_absolute() or '..' in path.parts or not path.parts:
                    raise ValueError('Недопустимый путь в manifest')
                data = bundle.read('ritm-transport/'+name)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('Хеш файла ZIP не совпадает: '+name)
                if name != 'REVIEW.html' and hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Рабочий файл отличается от ZIP: '+name)
            report.update(files=len(manifest['files']), crc_passed=True,
                          manifest_hashes_passed=True, working_files_match=True)
            if preflight:
                with tempfile.TemporaryDirectory(prefix='ritm-release-verify-') as directory:
                    bundle.extractall(directory)
                    extracted = Path(directory)/'ritm-transport'
                    result = subprocess.run([sys.executable, '-I', str(extracted/'tools/start_solution.py'), '--check-only'],
                                            cwd=extracted, text=True, capture_output=True, timeout=30, check=True)
                    report['extracted_preflight'] = json.loads(result.stdout)
                    if report['extracted_preflight'].get('status') != 'inputs_valid':
                        raise ValueError('Распакованный preflight не подтвердил входы')
            else:
                report['extracted_preflight'] = None
                report['scope'] = 'ZIP CRC, все manifest-хеши и соответствие рабочим файлам; preflight отключён'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != report['archive_sha256']:
            raise ValueError('ZIP изменился во время проверки')
        report['passed'] = True
        return report
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (archive.parent/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/release')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'verification.json').write_text(json.dumps({'passed':False,'stage':'building'})+'\n')
    files = [ROOT/name for name in ROOT_FILES]
    for directory in SOURCE_DIRS:
        files += [p for p in (ROOT/directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and not p.name.startswith('.')
                  and p.suffix not in ('.pyc','.pyo')]
    files += [ROOT/'artifacts/model'/name for name in
              ('model.joblib','encoder.pt','manifest.json','probability.json','explanation_reference.json')]
    files += [ROOT/name for name in existing_reports()]
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise SystemExit('Отсутствуют файлы: '+', '.join(missing))
    if not (ROOT/'docs/generated/index.html').is_file():
        raise SystemExit('Сначала сгенерируйте документацию: python tools/export_docs.py')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(set(files))},
                    excluded=['dataset','submission','.git','.venv','.env','GPS logs'])
    archive = out/'ritm-solution.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(set(files)):
            bundle.write(p,'ritm-transport/'+str(p.relative_to(ROOT)))
        bundle.writestr('ritm-transport/MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    verify_archive(archive,root=ROOT)
    print(archive)


if __name__ == '__main__':
    main()
