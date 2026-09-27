"""Проверяет живую цепочку TCP NDTP → backend → ML → HTTP дашборда.

Меняет контекст локального backend и в finally возвращает демонстрацию.
Не использует датасет, не измеряет качество прогноза.
Без --require-learned ожидается baseline; для сервиса с MODEL_PATH передайте
--require-learned. Флаг проверяет модель, но не переключает конфигурацию сервиса.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.ndtp import crc16_modbus
LEARNED_VERSION = 'swiss-transformer-hgbr-prior-v1'


def call(url, body=None):
    data=None if body is None else json.dumps(body).encode()
    with urlopen(Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=5) as response:
        return json.load(response)


def frame(unit,service,kind,body,seq):
    nph=struct.pack('<HHHI',service,kind,1,seq)+body
    crc=crc16_modbus(nph)
    crc=((crc&255)<<8)|(crc>>8)
    return struct.pack('<HHHHBIH',0x7e7e,len(nph),0,crc,2,unit,0)+nph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',default='http://127.0.0.1:8000')
    p.add_argument('--dashboard',default='http://127.0.0.1:8080')
    p.add_argument('--ndtp-host',default='127.0.0.1')
    p.add_argument('--ndtp-port',type=int,default=9201)
    p.add_argument('--require-learned',action='store_true',help='Требовать frozen learned модель вместо baseline')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts')
    a=p.parse_args()
    now=datetime.now(timezone.utc)
    context={'vehicles':[{'tr_id':101,'unit_id':1166336,'label':'Smoke · синтетический','route_id':'smoke'}],
        'arrival_mode':'external','plan_complete':True,'plan_timezone':'UTC',
        'plan_version':f'synthetic-smoke-{int(now.timestamp())}',
        'schedule':[{'tr_id':101,'target':{'id':f'smoke-target-{i}','name':f'Демо-цель {i+1}',
                    'scheduled_at':(now+timedelta(minutes=12+3*i)).isoformat(),
                    'lat':55.76+(i%3)*.001,'lon':37.61+(i%3)*.001,'manual_fill':False}} for i in range(20)],
        'hints':[{'tr_id':101,'observed_at':now.isoformat(),'delay_s':180,
                  'source':'synthetic_hint','sample_id':'synthetic-smoke-initial-hint'}]}
    started=time.perf_counter()
    try:
        assert call(a.backend+'/health')['status']=='ok'
        call(a.backend+'/api/v1/live/context',context)
        assert call(a.backend+'/api/v1/state')['summary']['unknown']==1
        handshake=frame(1166336,0,100,struct.pack('<HHHIII',6,2,0,1166336,65535,0),1)
        nav=struct.pack('<IIIBBHHHHHBB',int(now.timestamp()),376000000,557500000,0xe0,100,22,25,45,0,150,10,1)
        realtime=frame(1166336,1,101,bytes([0,0])+nav,2)
        with socket.create_connection((a.ndtp_host,a.ndtp_port),timeout=5) as stream:
            payload=handshake+realtime
            stream.sendall(payload[:8])
            stream.sendall(payload[8:])
            for _ in range(30):
                state=call(a.dashboard+'/api/v1/state')
                vehicle=state['vehicles'][0]
                if vehicle['prediction'] and state['health']['ml']=='ok' and vehicle['status']=='fresh': break
                time.sleep(.2)
            else: raise AssertionError('Поток не дошёл до прогноза за 6 секунд')
            pred=vehicle['prediction']
            assert vehicle['lat']==55.75 and vehicle['lon']==37.6
            expected='learned' if a.require_learned else 'persistence'
            assert pred['method']==expected, f'Ожидался {expected}; для стека с MODEL_PATH передайте --require-learned'
            assert pred['model_version']==(LEARNED_VERSION if a.require_learned else 'persistence-v1')
            assert pred['predicted_delay_s'] is not None and math.isfinite(pred['predicted_delay_s'])
            assert pred['probability_late'] is None  # synthetic_hint не область калибровки
            assert vehicle['cur_dev_s']==180 and vehicle['current_deviation']['source']=='synthetic_hint'
            if not a.require_learned:
                assert pred['predicted_delay_s']==180 and pred['risk']=='red'
            assert 600 < (datetime.fromisoformat(pred['target']['scheduled_at'].replace('Z','+00:00'))-datetime.fromisoformat(pred['issued_at'].replace('Z','+00:00'))).total_seconds()<=900
            if pred['risk']=='red':
                assert any(incident['tr_id']==101 for incident in state['incidents'])
            trace=call(a.backend+'/api/v1/vehicles/101/forecast-trace')
            assert trace['execution']=='ml_http'
            assert trace['request']['current_delay_s']==180
            assert trace['request']['current_delay_source']=='synthetic_hint'
            assert trace['response']['method']==expected
            if a.require_learned:
                assert trace['request']['plan_context']['complete']
                assert len(trace['request']['plan_context']['stops'])==20
        report={'result':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
                'flow':'TCP handshake + Nav00 -> backend -> separate ML HTTP -> dashboard proxy',
                'scenario':'synthetic plan + synthetic_hint, not an arrival fact or ML evaluation',
                'method':expected,'model_version':pred['model_version'],'prediction':pred,
                'elapsed_s':round(time.perf_counter()-started,3),
                'inference_p95_ms':state['metrics']['inference_p95_ms']}
        out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
        (out/'smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (out/'smoke-trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally: call(a.backend+'/api/v1/mode',{'mode':'demo'})

if __name__=='__main__': main()
