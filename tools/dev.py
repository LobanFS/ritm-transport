"""Поднимает ML, live backend и UI; генератор доступен отдельно для API-тестов."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ui-mode', choices=('dispatcher', 'full'),
                        default=os.getenv('DASHBOARD_UI_MODE', 'full'))
    args = parser.parse_args()
    if args.ui_mode not in ('dispatcher', 'full'):
        parser.error('DASHBOARD_UI_MODE должен быть dispatcher или full')
    root=Path(__file__).resolve().parents[1]
    env = dict(os.environ, DASHBOARD_UI_MODE=args.ui_mode, INITIAL_MODE='live')
    commands=[
        [sys.executable,'-m','uvicorn','generator.app:app','--host','127.0.0.1','--port','8002'],
        [sys.executable,'-m','uvicorn','ml_service.app:app','--host','127.0.0.1','--port','8001'],
        [sys.executable,'-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000'],
        [sys.executable,'tools/serve_dashboard.py','--port','8080','--ui-mode',args.ui_mode],
    ]
    children=[]
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for command in commands: children.append(subprocess.Popen(command,cwd=root,env=env))
        print(f'Dashboard http://127.0.0.1:8080 ({args.ui_mode}) · live, ожидание данных · API http://127.0.0.1:8000/docs · ML http://127.0.0.1:8001/docs',flush=True)
        while all(p.poll() is None for p in children): time.sleep(.5)
        raise RuntimeError('Один из сервисов завершился; проверьте логи и свободные порты')
    except KeyboardInterrupt: pass
    finally:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: p.kill(); p.wait()

if __name__=='__main__': main()
