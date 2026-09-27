"""Независимый HTTP producer искусственной телеметрии, порт 8002.

Запуск: ``uvicorn generator.app:app --host 0.0.0.0 --port 8002``.
Backend забирает причинные кадры по cursor seq. Это decoded HTTP stream,
а не NDTP: совместимость с NDTP проверяет отдельный официальный эмулятор.
Один worker, одна ограниченная сессия в памяти. Сервис не обращается к ML.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from generator.scenarios import ControlRequest, GeneratorSession, ResetRequest


def create_app(*, start_background: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.session = GeneratorSession(ResetRequest())
        app.state.last_tick = time.monotonic()

        async def run():
            while True:
                await asyncio.sleep(1)
                pump(app)

        task = asyncio.create_task(run()) if start_background else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    application = FastAPI(title='Синтетический генератор телеметрии', version='1.0', lifespan=lifespan,
        description='Условная маршрутная сеть и GPS. Истинные прибытия отделены в /truth для evaluator; /stream их не передаёт. Сценарные причины — истина синтетики, не вывод ML.')

    @application.exception_handler(RequestValidationError)
    async def invalid_payload(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={'detail':[
            {key: item[key] for key in ('loc', 'msg', 'type')} for item in exc.errors()]})

    @application.get('/health')
    async def health(request: Request):
        return {'status': 'ok', 'synthetic': True, 'transport': 'decoded_http',
                **request.app.state.session.status()}

    @application.post('/reset')
    async def reset(config: ResetRequest, request: Request):
        request.app.state.session = GeneratorSession(config)
        request.app.state.last_tick = time.monotonic()
        return request.app.state.session.status(include_context=True)

    @application.get('/stream')
    async def stream(request: Request, session_id: str = Query(min_length=1, max_length=80),
                     after: int = Query(default=0, ge=0)):
        session = request.app.state.session
        if session_id != session.session_id:
            raise HTTPException(409, 'Сессия генератора сменилась; требуется повторный reset и загрузка плана')
        try:
            return session.stream(after)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.post('/control')
    async def control(body: ControlRequest, request: Request):
        pump(request.app)
        try:
            return request.app.state.session.control(body)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get('/truth')
    async def truth(request: Request, session_id: str = Query(min_length=1, max_length=80)):
        """Проверочный эталон уже наступивших событий; backend его не потребляет."""
        session = request.app.state.session
        if session_id != session.session_id:
            raise HTTPException(409, 'Сессия генератора сменилась')
        return session.truth()

    return application


def pump(app: FastAPI):
    """Снять реальные elapsed до изменения скорости/паузы, без скачка часов."""
    now = time.monotonic()
    elapsed = max(0, now-app.state.last_tick)
    app.state.last_tick = now
    app.state.session.advance(elapsed)


app = create_app()
