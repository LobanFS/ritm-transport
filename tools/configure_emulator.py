"""Связывает официальный эмулятор с ЯВНО синтетическим полным live-планом.

Начальное отклонение задаётся synthetic_hint для демонстрации интеграции;
это не событие прибытия и не значение, полученное из NDTP.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from urllib.request import Request, urlopen


def post(url, body):
    with urlopen(Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=5) as response:
        return response.read().decode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',default='http://127.0.0.1:8000')
    p.add_argument('--emulator',default='http://127.0.0.1:18080')
    p.add_argument('--target-host',default='backend',help='backend для Compose; host.docker.internal для локального backend')
    p.add_argument('--target-port',type=int,default=9201)
    args=p.parse_args()
    now=datetime.now(timezone.utc)
    coords=[(37.60,55.76),(37.61,55.765),(37.62,55.77)]
    context={'vehicles':[{'tr_id':101,'unit_id':1166336,'label':'ДЕМО NDTP · 101','route_id':'demo-live'}],
        'arrival_mode':'external','plan_version':f'synthetic-ndtp-{int(now.timestamp())}',
        'plan_timezone':'UTC','plan_complete':True,
        'routes':[{'route_id':'demo-live','name':'Синтетический NDTP-сценарий','color':'#4169e1','path':coords,'stops':[]}],
        'schedule':[{'tr_id':101,'target':{'id':f'demo-live-{i}','name':f'Демо-остановка {i+1}',
            'scheduled_at':(now+timedelta(minutes=12+3*i)).isoformat(),'lon':coords[i%3][0],
            'lat':coords[i%3][1],'manual_fill':False}} for i in range(20)],
        'hints':[{'tr_id':101,'observed_at':now.isoformat(),'delay_s':180,
                  'source':'synthetic_hint','sample_id':'synthetic-ndtp-initial-hint'}]}
    print(post(args.backend+'/api/v1/live/context',context))
    config={'targetHost':args.target_host,'targetPort':args.target_port,'units':[{'unitId':1166336,'intervalMs':1000,'autoGenerate':True,'cells':[]}]}
    print(post(args.emulator+'/api/config',config))
    print('Полный искусственный план: 20 посещений, manual_fill=False, timezone=UTC. '
          'Начальное +180 с — synthetic_hint, не факт NDTP и не измерение MAE. '
          'Вероятность для такой подсказки недоступна. Подсказка истечёт через 5 минут; '
          'повторный запуск заменяет live-контекст.')

if __name__=='__main__': main()
