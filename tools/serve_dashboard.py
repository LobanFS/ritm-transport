"""Локальная статика и same-origin API proxy без Node; Docker использует nginx."""
import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def import_limits():
    limits = (int(os.getenv('RITM_IMPORT_MAX_BYTES', '268435456')),
              int(os.getenv('RITM_IMPORT_TIMEOUT_SECONDS', '120')))
    if any(value <= 0 for value in limits):
        raise ValueError('Лимиты импорта должны быть положительными целыми числами')
    return limits


def dashboard_config(ui_mode):
    if ui_mode not in ('dispatcher', 'full'):
        raise ValueError('DASHBOARD_UI_MODE должен быть dispatcher или full')
    max_bytes, timeout = import_limits()
    return ('window.RITM_CONFIG = '+json.dumps({'uiMode':ui_mode,
        'importMaxBytes':max_bytes, 'importTimeoutSeconds':timeout})+';\n').encode('utf-8')


def make_handler(backend, directory, ui_mode):
    config = dashboard_config(ui_mode)
    max_bytes, import_timeout = import_limits()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw): super().__init__(*a,directory=str(directory),**kw)
        def do_GET(self):
            if self.path.split('?',1)[0] == '/config.js':
                self.send_response(200)
                self.send_header('Content-Type','application/javascript; charset=utf-8')
                self.send_header('Content-Length',str(len(config)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(config)
                return
            if self.path.startswith(('/api/','/health','/docs','/openapi.json','/redoc')): self.proxy()
            else: super().do_GET()
        def do_POST(self): self.proxy()
        def proxy(self):
            length=int(self.headers.get('Content-Length','0'))
            if length<0 or length>max_bytes: self.send_error(413); return
            body=self.rfile.read(length) if length else None
            req=Request(backend+self.path,data=body,method=self.command,headers={'Content-Type':self.headers.get('Content-Type','application/json')})
            try:
                response=urlopen(req,timeout=import_timeout if self.path.split('?',1)[0] in
                    ('/api/v1/replay/import', '/api/v1/replay/load', '/api/v1/live/context') else 5)
            except HTTPError as e: response=e
            except (URLError,TimeoutError): self.send_error(502,'Backend unavailable'); return
            with response:
                content=response.read()
                self.send_response(response.status)
                self.send_header('Content-Type',response.headers.get('Content-Type','application/json'))
                self.send_header('Content-Length',str(len(content)))
                if response.headers.get('Content-Disposition'):
                    self.send_header('Content-Disposition',response.headers['Content-Disposition'])
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(content)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--backend',default='http://127.0.0.1:8000')
    parser.add_argument('--ui-mode',choices=('dispatcher','full'),
                        default=os.getenv('DASHBOARD_UI_MODE','full'))
    args=parser.parse_args()
    if args.ui_mode not in ('dispatcher','full'):
        parser.error('DASHBOARD_UI_MODE должен быть dispatcher или full')
    directory=Path(__file__).resolve().parents[1]/'dashboard'
    Handler=make_handler(args.backend,directory,args.ui_mode)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Dashboard: http://127.0.0.1:{args.port} ({args.ui_mode})',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
