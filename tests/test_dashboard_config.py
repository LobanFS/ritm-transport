"""UI-конфиг выбирает оформление; загрузка конфига не управляет источником."""
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import dev, serve_dashboard

ROOT = Path(__file__).resolve().parents[1]


def read_config(source):
    prefix='window.RITM_CONFIG = '
    assert source.strip().startswith(prefix) and source.strip().endswith(';')
    return json.loads(source.strip()[len(prefix):-1])


def response_from(handler_class, path='/config.js'):
    handler=object.__new__(handler_class)
    handler.path=path
    handler.wfile=BytesIO()
    headers={}
    status=[]
    handler.send_response=status.append
    handler.send_header=lambda key,value: headers.update({key:value})
    handler.end_headers=lambda: None
    handler.do_GET()
    return status,headers,handler.wfile.getvalue()


def test_static_config_defaults_to_full():
    assert read_config((ROOT/'dashboard/config.js').read_text()) == {'uiMode':'full'}


@pytest.mark.parametrize('ui_mode',['dispatcher','full'])
def test_dynamic_config_is_uncached_and_does_not_change_static_file_or_call_backend(ui_mode, monkeypatch):
    static=ROOT/'dashboard/config.js'
    before=static.read_bytes()
    monkeypatch.setattr(serve_dashboard,'urlopen',lambda *a,**k: pytest.fail('config GET must not access backend'))
    handler=serve_dashboard.make_handler('http://unused',ROOT/'dashboard',ui_mode)
    status,headers,body=response_from(handler,'/config.js?v=1')
    assert status == [200]
    assert headers['Cache-Control'] == 'no-store'
    assert headers['Content-Type'].startswith('application/javascript')
    assert int(headers['Content-Length']) == len(body)
    assert read_config(body.decode()) == {'uiMode':ui_mode}
    assert static.read_bytes() == before


@pytest.mark.parametrize('env_mode,cli_mode,expected',[
    (None,None,'full'),('dispatcher',None,'dispatcher'),('full','dispatcher','dispatcher'),
])
def test_local_server_config_accepts_env_and_cli_without_writing_source(monkeypatch,env_mode,cli_mode,expected):
    captured={}
    class Server:
        def __init__(self,address,handler): captured['handler']=handler
        def serve_forever(self): pass
        def server_close(self): pass
    monkeypatch.setattr(serve_dashboard,'ThreadingHTTPServer',Server)
    if env_mode is None: monkeypatch.delenv('DASHBOARD_UI_MODE',raising=False)
    else: monkeypatch.setenv('DASHBOARD_UI_MODE',env_mode)
    monkeypatch.setattr(sys,'argv',['serve_dashboard.py']+(['--ui-mode',cli_mode] if cli_mode else []))
    serve_dashboard.main()
    _,_,body=response_from(captured['handler'])
    assert read_config(body.decode()) == {'uiMode':expected}


@pytest.mark.parametrize('mode',[None,'dispatcher','full'])
def test_container_entrypoint_generates_the_same_runtime_contract(tmp_path,mode):
    env=dict(os.environ)
    env.pop('DASHBOARD_UI_MODE',None)
    if mode is not None: env['DASHBOARD_UI_MODE']=mode
    output=tmp_path/'config.js'
    subprocess.run(['sh',str(ROOT/'config/40-dashboard-config.sh'),str(output)],env=env,check=True)
    assert read_config(output.read_text()) == {'uiMode':mode or 'full'}


def test_invalid_modes_are_rejected_without_overwriting_config(tmp_path):
    output=tmp_path/'config.js'
    output.write_text('existing')
    result=subprocess.run(['sh',str(ROOT/'config/40-dashboard-config.sh'),str(output)],
                          env=dict(os.environ,DASHBOARD_UI_MODE='invalid'),capture_output=True)
    assert result.returncode != 0 and output.read_text() == 'existing'
    with pytest.raises(ValueError): serve_dashboard.dashboard_config('invalid')


@pytest.mark.parametrize('env_mode,cli_mode,expected',[
    (None,None,'full'),('dispatcher',None,'dispatcher'),('full','dispatcher','dispatcher'),
])
def test_dev_passes_ui_config_to_children_and_starts_in_live(monkeypatch,env_mode,cli_mode,expected):
    calls=[]
    class Process:
        def poll(self): return 1
        def wait(self,timeout=None): return 1
    def spawn(command,**kwargs):
        calls.append((command,kwargs))
        return Process()
    monkeypatch.setattr(dev.subprocess,'Popen',spawn)
    monkeypatch.setattr(dev.signal,'signal',lambda *args:None)
    if env_mode is None: monkeypatch.delenv('DASHBOARD_UI_MODE',raising=False)
    else: monkeypatch.setenv('DASHBOARD_UI_MODE',env_mode)
    monkeypatch.setenv('INITIAL_MODE','demo')
    monkeypatch.setattr(sys,'argv',['dev.py']+(['--ui-mode',cli_mode] if cli_mode else []))
    # Stopped fake children exercise coordinator cleanup without any real process.
    with pytest.raises(RuntimeError,match='Один из сервисов'):
        dev.main()
    assert len(calls) == 4
    assert all(kwargs['env']['DASHBOARD_UI_MODE'] == expected for _,kwargs in calls)
    assert all(kwargs['env']['INITIAL_MODE'] == 'live' for _,kwargs in calls)
    dashboard=next(command for command,_ in calls if 'tools/serve_dashboard.py' in command)
    assert dashboard[-2:] == ['--ui-mode',expected]
