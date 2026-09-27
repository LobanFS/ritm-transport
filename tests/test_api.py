"""API: валидация входов, пустой live и отсутствие смешения источников."""
from fastapi.testclient import TestClient
from backend.app import create_app


def test_control_and_live_empty_state():
    with TestClient(create_app(start_background=False,enable_ndtp=False)) as client:
        assert client.get('/health').json()['status']=='ok'
        initial=client.get('/api/v1/state').json()
        assert initial['mode']=='live' and initial['vehicles']==[]
        assert client.post('/api/v1/mode',json={'mode':'demo'}).status_code==200
        assert client.get('/api/v1/state').json()['summary']['vehicles']==4
        assert client.post('/api/v1/demo/control',json={'action':'speed','speed':30}).status_code==200
        assert client.post('/api/v1/demo/control',json={'action':'speed','speed':0}).status_code==422
        assert client.post('/api/v1/mode',json={'mode':'live'}).status_code==200
        assert client.get('/api/v1/state').json()['vehicles']==[]
        assert client.post('/api/v1/demo/control',json={'action':'resume'}).status_code==409
        assert client.post('/api/v1/mode',json={'mode':'demo'}).status_code==200


def test_no_future_facts_field_in_live_contract():
    with TestClient(create_app(start_background=False,enable_ndtp=False)) as client:
        data={'vehicles':[{'tr_id':1,'unit_id':2,'label':'A','route_id':'r'}], 'time_fact_begin':'2026-01-06T08:00:00Z'}
        assert client.post('/api/v1/live/context',json=data).status_code==422
        data.pop('time_fact_begin')
        assert client.post('/api/v1/live/context',json=data).status_code==200
        state=client.get('/api/v1/state').json()
        assert state['summary']['unknown']==1
        assert state['vehicles'][0]['prediction'] is None
        assert client.post('/api/v1/telemetry',json=[]).status_code==422
        schema=client.get('/openapi.json').json()
        assert '/api/v1/live/context' in schema['paths']


def test_http_unit_mapping_and_receive_timestamp():
    from datetime import datetime, timezone
    with TestClient(create_app(start_background=False,enable_ndtp=False)) as client:
        client.post('/api/v1/live/context',json={'vehicles':[{'tr_id':1,'unit_id':2,'label':'A','route_id':'r'}]})
        timestamp=datetime.now(timezone.utc).isoformat()
        point={'tr_id':1,'unit_id':3,'event_time':timestamp,'received_at':'2000-01-01T00:00:00Z','lat':55.75,'lon':37.61,'speed_kmh':10,'heading':273}
        assert client.post('/api/v1/telemetry',json=[point]).status_code==422
        point['unit_id']=2
        assert client.post('/api/v1/telemetry',json=[point]).json()['accepted']==1
        saved=client.app.state.engine.history[1][0]
        assert saved.received_at.year==datetime.now(timezone.utc).year
        assert saved.source=='http'
        assert client.get('/api/v1/state').json()['vehicles'][0]['heading']==273
        # A newer rejected GPS must not rotate the marker of the retained position.
        invalid={**point,'event_time':datetime.now(timezone.utc).isoformat(),
                 'heading':15,'location_valid':False}
        assert client.post('/api/v1/telemetry',json=[invalid]).status_code==200
        assert client.get('/api/v1/state').json()['vehicles'][0]['heading']==273
