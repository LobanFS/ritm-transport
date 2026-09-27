"""HTTP API диспетчера и жизненный цикл TCP-приёмника; запуск строго в один worker."""
import asyncio
import contextlib
import csv
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from backend.engine import Engine, LiveContext, utcnow
from backend.arrivals import ArrivalConflict, ArrivalInput
from backend.generator_bridge import GeneratorControl, GeneratorStart, ProducerState
from backend.ndtp import NDTPServer
from backend.replay import (CUSTOM_IMPORT_SECONDS, MAX_CUSTOM_IMPORT_BYTES, ReplayConfig,
                            ReplayControl, ReplayImport, load_replay, prepare_custom_replay)
from common.contracts import Contract, Telemetry
from common.state import DashboardState, ForecastTrace, IncidentExport, Metrics

log = logging.getLogger(__name__)


class DemoControl(Contract):
    action: Literal["pause", "resume", "reset", "speed", "source_off", "source_on"]
    speed: float | None = Field(default=None, ge=1, le=30)


class ModeControl(Contract):
    mode: Literal["demo", "live"]


def create_app(*, start_background: bool = True, enable_ndtp: bool = True, ml_url: str | None = None, generator_url: str | None = None) -> FastAPI:
    """Фабрика для изолированных тестов и одного экземпляра in-memory состояния."""
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=.5), trust_env=False) as client:
            learning_store = None
            if learning_path := os.getenv("LEARNING_STORE_PATH"):
                from backend.learning_store import LearningStore
                learning_store = LearningStore(Path(learning_path))
            engine = Engine(ml_url or os.getenv("ML_URL", "http://127.0.0.1:8001"), client,
                            learning_store=learning_store)
            initial_mode = os.getenv('INITIAL_MODE', 'live')
            if initial_mode not in ('live', 'demo'):
                raise ValueError('INITIAL_MODE должен быть live или demo')
            if initial_mode == 'live':
                engine.set_live()
            engine.generator_url = generator_url or os.getenv('GENERATOR_URL', 'http://127.0.0.1:8002')
            app.state.engine = engine
            app.state.generator_control_lock = asyncio.Lock()
            app.state.replay_load_lock = asyncio.Lock()
            app.state.custom_dataset = None
            custom_storage = tempfile.TemporaryDirectory(prefix='ritm-custom-')
            app.state.custom_storage = Path(custom_storage.name)
            listener = NDTPServer(engine.on_nav, engine.ndtp_error, host=os.getenv("NDTP_HOST", "127.0.0.1"), port=int(os.getenv("NDTP_PORT", "9201")))
            if enable_ndtp:
                await listener.start()
            async def loop():
                last = time.monotonic()
                while True:
                    now = time.monotonic()
                    try:
                        engine.ndtp_connections = listener.connections
                        await engine.tick(now-last)
                    except Exception:
                        log.exception("Неожиданная ошибка цикла обработки")
                        engine.last_error = "Ошибка цикла обработки; подробности в логах"
                    last = now
                    await asyncio.sleep(1)
            task = asyncio.create_task(loop()) if start_background else None
            try:
                yield
            finally:
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if enable_ndtp:
                    await listener.close()
                custom_storage.cleanup()

    app = FastAPI(title="Транспорт · Backend", version="0.1.0", lifespan=lifespan,
                  description="Диспетчерская система: demo, отдельный synthetic generator, live NDTP и CSV replay. Отдельный ML-сервис с frozen-моделью; при недоступности — явный persistence fallback. Времена RFC3339 с offset.")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail":[{"loc":list(e["loc"]),"msg":e["msg"],"type":e["type"]} for e in exc.errors()]})

    @app.get("/health", tags=["Состояние"])
    async def health(request: Request):
        """Живость backend; недоступность ML отдельно отмечена и не роняет процесс."""
        e = request.app.state.engine
        return {"status":"ok", "ml":e.ml_status,"mode":e.mode,"version":"0.1.0"}

    @app.get("/api/v1/state", response_model=DashboardState, tags=["Диспетчер"])
    async def state(request: Request):
        """Атомарное состояние карты, ТС и предупреждений. Обновление UI раз в секунду."""
        return request.app.state.engine.state()

    @app.get("/api/v1/metrics", response_model=Metrics, tags=["Состояние"])
    async def metrics(request: Request):
        """Счётчики процесса; p95 HTTP ML и полного цикла последних 200 вызовов, без ожидания цикла/UI."""
        return request.app.state.engine.state()["metrics"]

    @app.get("/api/v1/incidents/export", response_model=IncidentExport, tags=["Диспетчер"])
    async def export_incidents(request: Request, response: Response):
        """Последние 200 алертов текущей сессии с объяснением на момент выдачи."""
        e = request.app.state.engine
        response.headers['Content-Disposition'] = 'attachment; filename="incidents.json"'
        response.headers['Cache-Control'] = 'no-store'
        return IncidentExport(exported_at=utcnow(),mode=e.mode,
            note='Горизонт (10,15] минут рассчитан до планового прибытия согласно README раздачи. estimated_arrival_at — отдельная оценка времени прибытия, не фактическое время. Причина помечена как наблюдение/гипотеза. Demo — синтетика; replay — исторические CSV, часы сценария. Журнал очищается при сбросе/смене контекста.',
            incidents=list(e.incidents))

    @app.post("/api/v1/demo/control", tags=["Демонстрация"])
    async def control(command: DemoControl, request: Request):
        e = request.app.state.engine
        if e.mode != "demo":
            raise HTTPException(409, "Демонстрационные команды доступны только в demo")
        if command.action == "reset": e.reset_demo()
        elif command.action == "pause": e.running = False
        elif command.action == "resume": e.running = True
        elif command.action == "source_off": e.source_enabled = False
        elif command.action == "source_on": e.source_enabled = True
        elif command.action == "speed":
            if command.speed is None: raise HTTPException(422, "Для speed укажите скорость")
            e.speed = command.speed
        return {"ok":True,"demo":e.state()["demo"]}

    @app.post("/api/v1/mode", tags=["Демонстрация"])
    async def mode(command: ModeControl, request: Request):
        """Переключение очищает старые координаты, расписание, прогнозы и алерты."""
        e = request.app.state.engine
        e.reset_demo() if command.mode=="demo" else e.set_live()
        return {"ok":True,"mode":e.mode}

    @app.post('/api/v1/replay/load', tags=['Исторический replay'])
    async def replay_load(config: ReplayConfig, request: Request):
        """Загрузить validate/train из REPLAY_DATA_DIR или ранее импортированный custom.

        Из train/schedule.csv используются только плановые поля. labels и
        фактические прибытия не становятся входами детектора или модели.

        tr_ids=null выбирает все ТС с планом и сообщениями в срезе с прогревом.
        Явный список — ручной выбор до 128 ТС. Счётчики и будущие времена
        в replay.streams служат только диагностикой очереди проигрывателя.

        duration_minutes=null (по умолчанию) читает до последней доставки
        выбранного архива: max(event_time, receive_time) для GPS, T для CSV.
        Явное число ограничивает окно сверху. Пустой хвост не проигрывается.
        По умолчанию начинает с паузы. Загрузка очищает текущий контекст только
        после успешной проверки CSV. Пути файлов через API не задаются.
        """
        lock = request.app.state.replay_load_lock
        if lock.locked():
            raise HTTPException(409, 'Другая загрузка CSV ещё выполняется')
        async with lock:
            if config.dataset_split == 'custom':
                custom = request.app.state.custom_dataset
                if custom is None:
                    raise HTTPException(409, 'Сначала импортируйте свои CSV; после перезапуска backend импорт нужно повторить')
                root = custom['root']
                if 'timezone' not in config.model_fields_set:
                    config = config.model_copy(update={'timezone': custom['timezone']})
                if config.deviation_source == 'csv_snapshot' and not custom['has_points']:
                    raise HTTPException(422, 'Для csv_snapshot загрузите points.csv')
            else:
                root = os.getenv('REPLAY_DATA_DIR')
                if not root:
                    raise HTTPException(409, 'Задайте REPLAY_DATA_DIR и перезапустите backend; см. README')
            e = request.app.state.engine
            version = e.version
            try:
                replay = await asyncio.to_thread(load_replay, Path(root), config)
            except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as exc:
                log.warning('Replay load failed: %s', type(exc).__name__)
                raise HTTPException(422, 'Не удалось загрузить срез: проверьте CSV, ТС, окно и логи backend') from exc
            if e.version != version:
                raise HTTPException(409, 'Контекст изменился во время чтения CSV; повторите загрузку')
            e.set_replay(replay)
            return {'ok': True, 'replay': replay.state(e)}

    def custom_metadata(request):
        custom = request.app.state.custom_dataset
        return {'available': custom is not None, 'has_points': bool(custom and custom['has_points']),
                'timezone': custom['timezone'] if custom else None,
                'start': custom['start'] if custom else None, 'end': custom['end'] if custom else None,
                'expires_on_restart': True}

    @app.get('/api/v1/replay/custom', tags=['Исторический replay'])
    async def custom_status(request: Request):
        """Доступность пользовательского архива; серверные пути не раскрываются."""
        return custom_metadata(request)

    @app.post('/api/v1/replay/import', tags=['Исторический replay'], openapi_extra={
        'requestBody': {'required': True, 'content': {'application/json': {'schema': ReplayImport.model_json_schema()}}}})
    async def replay_import(request: Request):
        """Проверить CSV и активировать custom на паузе, с GPS и началом по данным.

        Весь JSON ограничен 80 MiB; приём и обработка — по 30 секунд.
        Максимум 500000 строк traffic+points, 20000 строк плана, 128 ТС.
        Имена файлов фиксированы. Bundled train/validate не изменяются.
        Ошибка оставляет прежний контекст и прежний custom; импорт удаляется
        после перезапуска backend. points необязателен и включается только
        отдельным /replay/load с deviation_source=csv_snapshot.
        """
        lock = request.app.state.replay_load_lock
        if lock.locked():
            raise HTTPException(409, 'Другая загрузка CSV ещё выполняется')
        async with lock:
            e = request.app.state.engine
            version = e.version
            content_length = request.headers.get('content-length')
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise HTTPException(400, 'Некорректный Content-Length') from exc
                if declared < 0:
                    raise HTTPException(400, 'Некорректный Content-Length')
                if declared > MAX_CUSTOM_IMPORT_BYTES:
                    raise HTTPException(413, 'JSON с CSV превышает 80 MiB')
            async def read_limited():
                body = bytearray()
                async for chunk in request.stream():
                    if len(body)+len(chunk) > MAX_CUSTOM_IMPORT_BYTES:
                        raise HTTPException(413, 'JSON с CSV превышает 80 MiB')
                    body.extend(chunk)
                return body
            try:
                body = await asyncio.wait_for(read_limited(), timeout=CUSTOM_IMPORT_SECONDS)
            except TimeoutError as exc:
                raise HTTPException(408, 'Время приёма CSV истекло') from exc
            staged = None
            try:
                payload = ReplayImport.model_validate_json(body)
                del body
                staged = Path(tempfile.mkdtemp(prefix='upload-', dir=request.app.state.custom_storage))
                replay = await asyncio.to_thread(prepare_custom_replay, staged, payload)
                if e.version != version:
                    raise HTTPException(409, 'Контекст изменился во время импорта CSV; повторите загрузку')
                previous = request.app.state.custom_dataset
                # Проверка завершена; commit без await не смешивается с другими командами.
                e.set_replay(replay)
                request.app.state.custom_dataset = {'root': staged, 'timezone': payload.timezone,
                    'has_points': payload.points_csv is not None,
                    'start': replay.config.start.isoformat(), 'end': replay.end.isoformat()}
                staged = None
                if previous:
                    shutil.rmtree(previous['root'], ignore_errors=True)
                return {'ok': True, 'replay': replay.state(e), 'custom': custom_metadata(request)}
            except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as exc:
                log.warning('Custom CSV import failed: %s', type(exc).__name__)
                raise HTTPException(422, 'Не удалось импортировать CSV: проверьте заголовки, даты, координаты и ограничения размера') from exc
            finally:
                if staged is not None:
                    shutil.rmtree(staged, ignore_errors=True)

    @app.post('/api/v1/generator/start', tags=['Сценарный генератор'])
    async def generator_start(config: GeneratorStart, request: Request):
        """Новая сессия отдельного HTTP producer; план синтетический, будущих фактов нет.

        Параметры воспроизводимы через seed. Сначала проверяется ответ producer,
        затем заменяется контекст. URL задаётся только конфигурацией сервера.
        """
        e = request.app.state.engine
        async with request.app.state.generator_control_lock:
            version = e.version
            try:
                response = await e.client.post(e.generator_url+'/reset', json=config.model_dump())
                response.raise_for_status()
                payload = response.json()
                status = ProducerState.model_validate(payload)
                context = LiveContext.model_validate(payload['context'])
                if context.hints:
                    raise ValueError('Generator reset must not include deviation hints')
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                log.warning('Generator start failed: %s', exc)
                raise HTTPException(502, 'Генератор недоступен или вернул неверный контракт; проверьте его сервис') from exc
            if e.version != version:
                raise HTTPException(409, 'Режим изменился во время запуска генератора; повторите команду')
            e.set_generator(context, status)
            await e.poll_generator()
            if e.mode != 'generator' or e.generator.view.session_id != status.session_id:
                raise HTTPException(409, 'Режим изменился во время получения первого кадра')
            return {'ok': True, 'generator': e.generator.view.model_dump(mode='json')}

    @app.post('/api/v1/generator/control', tags=['Сценарный генератор'])
    async def generator_control(command: GeneratorControl, request: Request):
        """Пауза/скорость меняют часы producer. Backend забирает все доступные кадры по cursor."""
        e = request.app.state.engine
        async with request.app.state.generator_control_lock:
            if e.mode != 'generator':
                raise HTTPException(409, 'Сначала запустите наш генератор')
            if command.action == 'speed' and command.speed is None:
                raise HTTPException(422, 'Для speed укажите скорость')
            session, version = e.generator, e.version
            try:
                response = await e.client.post(e.generator_url+'/control', json=command.model_dump(exclude_none=True))
                response.raise_for_status()
            except httpx.HTTPError as exc:
                if e.version == version:
                    session.fail('Не удалось передать команду генератору')
                raise HTTPException(502, 'Не удалось передать команду генератору; проверьте сервис') from exc
            if e.version != version:
                raise HTTPException(409, 'Режим изменился во время команды')
            await e.poll_generator()
            if e.version != version:
                raise HTTPException(409, 'Режим изменился во время получения кадров')
            return {'ok': True, 'generator': session.view.model_dump(mode='json')}

    @app.get('/api/v1/generator/logs', tags=['Сценарный генератор'])
    async def generator_logs(request: Request, response: Response):
        """Последние 100 записей producer и счётчики приёма backend; это не объяснение ML."""
        e = request.app.state.engine
        if e.mode != 'generator':
            raise HTTPException(409, 'Сначала запустите наш генератор')
        response.headers['Cache-Control'] = 'no-store'
        return e.generator.view

    @app.post('/api/v1/replay/control', tags=['Исторический replay'])
    async def replay_control(command: ReplayControl, request: Request):
        e = request.app.state.engine
        if e.mode != 'replay':
            raise HTTPException(409, 'Сначала загрузите replay')
        if command.action == 'reset':
            e.set_replay(e.replay)
        elif command.action == 'pause':
            e.running = False
        elif command.action == 'resume':
            if e.clock >= e.replay.end:
                raise HTTPException(409, 'Архив завершён; нажмите Сброс')
            e.running = True
        elif command.action == 'speed':
            if command.speed is None:
                raise HTTPException(422, 'Для speed укажите скорость')
            e.speed = command.speed
        return {'ok': True, 'replay': e.replay.state(e)}

    @app.post("/api/v1/live/context", tags=["Входные данные"])
    async def context(context: LiveContext, request: Request):
        """Заменяет live-контекст: соответствие unit/ТС, план и известные подсказки. Фактов будущих прибытий нет в контракте."""
        request.app.state.engine.set_live(context)
        return {"ok":True,"vehicles":len(context.vehicles),"stops":len(context.schedule)}

    @app.get('/api/v1/context', response_model=LiveContext, tags=['Входные данные'])
    async def context_export(request: Request):
        """Текущие соответствия ТС/терминалов и план. Никаких фактических прибытий или hints.

        Возвращаемые даты не сдвигаются к текущему дню: исторический план нельзя
        автоматически применять к сегодняшнему NDTP. POST live/context заменяет всё состояние.
        """
        from backend.engine import ScheduledStop
        e = request.app.state.engine
        if not e.vehicles:
            raise HTTPException(409, 'Контекст пока не загружен')
        return LiveContext(vehicles=list(e.vehicles.values()), routes=e.routes,
            schedule=[ScheduledStop(tr_id=tr, target=stop) for tr, items in e.schedule.items() for stop in items],
            hints=[], arrival_mode='gps' if e.arrival_mode == 'gps' else 'external',
            plan_version=e.plan_version, plan_timezone=e.plan_timezone, plan_complete=e.plan_complete)

    @app.post("/api/v1/telemetry", tags=["Входные данные"])
    async def telemetry(points: Annotated[list[Telemetry], Body(min_length=1,max_length=500)], request: Request):
        """HTTP-адаптер decoded CSV/других источников; при приёме received_at ставит сервер."""
        e = request.app.state.engine
        if e.mode != "live": raise HTTPException(409,"Переключитесь в live и задайте контекст")
        unknown = sorted({p.tr_id for p in points if p.tr_id not in e.vehicles})
        if unknown: raise HTTPException(422, {"unknown_tr_ids":unknown})
        if any(p.unit_id is not None and p.unit_id != e.vehicles[p.tr_id].unit_id for p in points):
            raise HTTPException(422, "unit_id не соответствует настроенному ТС")
        received = utcnow()
        accepted = sum(e.ingest(p.model_copy(update={"received_at":received,"source":"http"})) for p in points)
        return {"accepted":accepted,"duplicates":len(points)-accepted}

    @app.post('/api/v1/arrivals', tags=['Входные данные'])
    async def arrival(event: ArrivalInput, request: Request):
        """Факт остановки от отдельного источника. NDTP Nav00 такого события не даёт.

        Backend вычисляет cur_dev_s = arrived_at − scheduled_at. Повтор того же
        факта идемпотентен, конфликт времени — 409; будущий факт/неизвестная ссылка — 422.
        Операционные факты поступают сюда отдельно. Экспериментальный внутренний
        GPS-детектор включается через arrival_mode=gps; подтверждённый операционный
        факт может уточнить его оценку, сохраняя время доступности обеих версий.
        """
        e = request.app.state.engine
        if e.mode != 'live':
            raise HTTPException(409, 'События прибытия принимаются только в live')
        received = utcnow()
        try:
            accepted = e.ingest_arrival(event, received_at=received)
        except ArrivalConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return dict(accepted=accepted, received_at=received, current_deviation=e.deviation_at(event.tr_id, received))

    @app.get('/api/v1/vehicles/{tr_id}/forecast-trace', response_model=ForecastTrace, tags=['Диспетчер'])
    async def forecast_trace(tr_id: int, request: Request, response: Response):
        """Последний завершённый запрос/ответ инференса с происхождением current_delay.

        Результат может быть историческим: issued_at и published_at нужно проверять.
        При новом контексте очищается. Ответ не содержит будущих фактов прибытия.
        """
        e = request.app.state.engine
        if tr_id not in e.vehicles:
            raise HTTPException(404, 'Неизвестное ТС')
        trace = e.forecast_traces.get(tr_id)
        if trace is None:
            raise HTTPException(404, 'Для этого ТС ещё нет завершённого инференса в текущем контексте')
        response.headers['Cache-Control'] = 'no-store'
        return trace

    return app


app = create_app()
