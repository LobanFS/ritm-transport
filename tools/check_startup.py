"""Повторяемый запуск контейнеров из остановленного состояния с готовыми образами.

Меняет текущую демонстрацию: останавливает и пересоздаёт только сервисы выбранного
Compose-проекта. Не включает сборку образов/старт VM, не очищает образы или данные.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', help='Именованный Docker context; текущий не меняется')
    parser.add_argument('--compose', action='append', help='Compose-файлы в порядке наложения')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'startup')
    parser.add_argument('--ml-url', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    files = args.compose or ['compose.yaml','compose.model.yaml']
    command = ['docker'] + (['--context',args.context] if args.context else []) + ['compose']
    for path in files:
        command += ['-f',path]
    args.out.mkdir(parents=True,exist_ok=True)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),passed=False,
        scope='Контейнеры из остановленного состояния; образы и VM уже готовы, кэш ОС не очищен',
        command=command,compose_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in files})
    try:
        with (args.out/'stop.log').open('w') as log:
            subprocess.run(command+['stop'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        start = time.monotonic()
        with (args.out/'up.log').open('w') as log:
            subprocess.run(command+['up','-d','--force-recreate','--wait','--wait-timeout','90'],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        report['compose_ready_s'] = time.monotonic()-start
        with urlopen(args.ml_url+'/health',timeout=10) as response:
            report['ml_health'] = json.load(response)
        with urlopen(args.ml_url+'/model',timeout=10) as response:
            report['model_card'] = json.load(response)
        if report['ml_health'].get('trained') is not True:
            raise AssertionError('Контейнер доступен, но обученная модель не загружена')
        report['trained_model_confirmed_s'] = time.monotonic()-start
        result = subprocess.run(command+['ps','--format','json'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=10)
        report['containers'] = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'startup_s':report['trained_model_confirmed_s'],
                      'report':str(args.out/'report.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
