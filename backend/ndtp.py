"""Ограниченный TCP-приёмник формата NDTP 6.2 из спецификации эмулятора.

Принимаются handshake и realtime с Nav00 и опциональными дверями IRMA04.
Другие ячейки пропускаются
только при известной длине payload. Неизвестная ячейка отклоняет весь кадр.
Формат ответа ACK в спецификации не задан, а эмулятор ответы не разбирает:
этот приёмник не отправляет ACK. Сопоставление unit_id с рейсом выполняется
снаружи, как и назначение received_at при вызове on_nav.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import struct
from typing import Literal


MAX_FRAME_SIZE = 65535
NPL_SIZE = 15
NPH_SIZE = 10
_NPL = struct.Struct("<HHHHBIH")
_NPH = struct.Struct("<HHHI")
_HANDSHAKE = struct.Struct("<HHHIII")
_NAV = struct.Struct("<IIIBBHHHHHBB")
# Размеры именно payload: двухбайтовый заголовок ячейки сюда не входит.
_CELL_PAYLOAD_SIZES = {0: 26, 2: 26, 4: 15, 8: 6, 10: 37, 15: 50, 16: 8}
# IRMA04: u32 odometer, u16 zone, 8*u8 counters, u8 flags.
# flags: present1..4 в битах 0..3, closed1..4 в битах 4..7.
# Layout подтверждён G6CellIrma04.class официального образа и реальными
# Nav00+IRMA04+Usi08 кадрами tools/check_irma.py. Счётчики не интерпретируем.
_IRMA = struct.Struct("<IH8BB")
_LOGGER = logging.getLogger(__name__)


class NDTPError(ValueError):
    """Ошибка кадра со стабильным кодом reason для счётчиков приёмника."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True, slots=True)
class ParsedFrame:
    """Проверенный кадр: идентификатор устройства, вид и навигация."""

    unit_id: int
    kind: Literal["handshake", "realtime"]
    nav: dict | None


def crc16_modbus(data: bytes) -> int:
    """Вычисляет CRC-16/Modbus до перестановки байтов для заголовка NPL."""
    value = 0xFFFF
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
    return value


def _frame_size(header: bytes) -> int:
    """Проверяет NPL до чтения тела и возвращает полный размер кадра."""
    if len(header) != NPL_SIZE:
        raise NDTPError("truncated_npl")
    signature, data_size, flags, _, packet_type, unit_id, _ = _NPL.unpack(header)
    if signature != 0x7E7E:
        raise NDTPError("invalid_signature")
    size = NPL_SIZE + data_size
    if data_size < NPH_SIZE or size > MAX_FRAME_SIZE:
        raise NDTPError("invalid_length", f"frame_size={size}")
    if flags != 0:
        raise NDTPError("unsupported_npl_flags", f"flags={flags}")
    if packet_type != 0x02:
        raise NDTPError("unsupported_npl_type", f"type={packet_type}")
    if unit_id > 2147483647:
        raise NDTPError("unit_id_out_of_range")
    return size


def parse_frame(frame: bytes) -> ParsedFrame:
    """Разбирает один полный кадр; при любой ошибке поднимает NDTPError.

    Проверяет обе длины, signature, CRC со свапом байтов, поддерживаемые
    заголовки и все ячейки до возврата навигации. Состояние TCP-handshake
    проверяет NDTPServer, поскольку один кадр его не содержит.
    Недостоверные либо географически невозможные координаты сохраняются
    без исправления, но получают location_valid=False.
    """
    expected_size = _frame_size(frame[:NPL_SIZE])
    if len(frame) != expected_size:
        raise NDTPError("frame_length_mismatch", f"expected={expected_size}, actual={len(frame)}")
    _, _, _, stored_crc, _, unit_id, _ = _NPL.unpack_from(frame)
    data = frame[NPL_SIZE:]
    crc = crc16_modbus(data)
    swapped_crc = ((crc & 0xFF) << 8) | (crc >> 8)
    if stored_crc != swapped_crc:
        raise NDTPError("crc_mismatch")
    service_id, message_type, flags, _ = _NPH.unpack_from(data)
    if flags != 1:
        raise NDTPError("unsupported_nph_flags", f"flags={flags}")
    body = data[NPH_SIZE:]

    if (service_id, message_type) == (0, 100):
        if len(body) != _HANDSHAKE.size:
            raise NDTPError("invalid_handshake_length")
        major, minor, handshake_flags, peer, _, _ = _HANDSHAKE.unpack(body)
        if (major, minor) != (6, 2):
            raise NDTPError("unsupported_protocol_version", f"version={major}.{minor}")
        if handshake_flags != 0:
            raise NDTPError("unsupported_handshake_flags")
        if peer != unit_id:
            raise NDTPError("handshake_unit_mismatch")
        # maxPacketSize и reserved не описывают входные ячейки. ACK нет,
        # поэтому объявленный размер приёмного буфера клиента не используется.
        return ParsedFrame(unit_id, "handshake", None)

    if (service_id, message_type) != (1, 101):
        raise NDTPError("unsupported_nph_message", f"service={service_id}, type={message_type}")

    nav = None
    door_states: list[bool] = []
    door_sensors: list[tuple[int,int]] = []
    irma_numbers: set[int] = set()
    offset = 0
    while offset < len(body):
        if len(body) - offset < 2:
            raise NDTPError("truncated_cell_header")
        cell_type, number = body[offset : offset + 2]
        payload_size = _CELL_PAYLOAD_SIZES.get(cell_type)
        if payload_size is None:
            raise NDTPError("unknown_cell_type", f"type={cell_type}")
        if offset == 0 and cell_type != 0:
            raise NDTPError("nav_not_first")
        start = offset + 2
        end = start + payload_size
        if end > len(body):
            raise NDTPError("truncated_cell", f"type={cell_type}")
        if cell_type == 0:
            if nav is not None:
                raise NDTPError("duplicate_nav")
            if number != 0:
                raise NDTPError("invalid_nav_number")
            (
                timestamp, longitude, latitude, extra, _, speed_avg, _,
                course, _, _, _, _,
            ) = _NAV.unpack_from(body, start)
            if course > 360:
                raise NDTPError("invalid_heading", f"heading={course}")
            lat = latitude / 10_000_000 * (1 if extra & 0x20 else -1)
            lon = longitude / 10_000_000 * (1 if extra & 0x40 else -1)
            nav = {
                "event_time": datetime.fromtimestamp(timestamp, timezone.utc),
                "lat": lat,
                "lon": lon,
                "speed_kmh": speed_avg,
                "heading": course,
                "location_valid": bool(extra & 0x80) and abs(lat) <= 90 and abs(lon) <= 180,
                # Повтор одного и того же кадра получает тот же идентификатор;
                # изменение полезной нагрузки не теряется при сбросе requestId.
                "event_id": f"ndtp:{hashlib.sha256(frame).hexdigest()}",
            }
        elif cell_type == 4:
            if number in irma_numbers:
                raise NDTPError("duplicate_irma_number", f"number={number}")
            irma_numbers.add(number)
            flags = _IRMA.unpack_from(body, start)[-1]
            if flags & 0x0F:
                door_sensors.append((number, flags & 0x0F))
            # Нулевой closed без present не означает открытую дверь:
            # у отсутствующего/непредставленного датчика состояние неизвестно.
            for door in range(4):
                if flags & (1 << door):
                    door_states.append(not bool(flags & (1 << (door + 4))))
        offset = end
    if nav is None:
        raise NDTPError("missing_nav")
    # Агрегат только по заявленным датчикам, без переноса состояния из
    # прошлого кадра. Отсутствие IRMA/всех present-флагов остаётся None.
    nav["doors_open"] = any(door_states) if door_states else None
    # Closed-биты не входят в ключ: он меняется только вместе с набором
    # датчиков, чтобы их появление не выглядело открытием прежней двери.
    nav["door_sensor_key"] = ("irma04:"+hashlib.sha256(bytes(
        value for pair in sorted(door_sensors) for value in pair)).hexdigest()) if door_sensors else None
    return ParsedFrame(unit_id, "realtime", nav)


class NDTPServer:
    """Принимает NDTP с ограничениями памяти, соединений и времени ожидания.

    on_nav вызывается последовательно в каждом соединении и ожидается до
    чтения следующего кадра: неограниченной очереди событий нет. on_error —
    синхронный короткий callback; error_counts содержит счётчики по reason.
    Повреждённый полный кадр пропускается; ошибка framing закрывает соединение.
    После каждого reconnect требуется новый корректный handshake.
    """

    def __init__(
        self,
        on_nav: Callable[[int, dict], Awaitable[None]],
        on_error: Callable[[str], None],
        host: str = "0.0.0.0",
        port: int = 9201,
        *,
        max_connections: int = 0,
        idle_timeout: float = 30.0,
        callback_timeout: float = 10.0,
    ) -> None:
        if max_connections < 0 or idle_timeout <= 0 or callback_timeout <= 0:
            raise ValueError("max_connections должен быть ≥0 (0 — без потолка); таймауты должны быть положительными")
        self.on_nav = on_nav
        self.on_error = on_error
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self.callback_timeout = callback_timeout
        self.error_counts: Counter[str] = Counter()
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False

    @property
    def connections(self) -> int:
        """Возвращает число принятых и ещё не закрытых соединений."""
        return len(self._writers)

    async def start(self) -> None:
        """Запускает TCP listener; при port=0 сохраняет выбранный порт в port."""
        if self._server is not None:
            raise RuntimeError("NDTPServer уже запущен")
        self._closing = False
        self._server = await asyncio.start_server(
            self._accept, self.host, self.port, limit=MAX_FRAME_SIZE,
        )
        if self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        """Останавливает приём, закрывает сокеты и дожидается обработчиков."""
        self._closing = True
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for writer in tuple(self._writers):
            writer.close()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Задача могла быть отменена ещё до входа в _handle/finally.
        self._writers.clear()
        if server is not None:
            # В Python 3.12 wait_closed ждёт и принятые соединения;
            # вызывать его до закрытия их writer означало бы зависнуть.
            await server.wait_closed()

    def _error(self, reason: str, detail: str = "") -> None:
        self.error_counts[reason] += 1
        try:
            self.on_error(f"{reason}: {detail}" if detail else reason)
        except Exception:
            # Сбой наблюдателя ошибок не должен терять cleanup сокета.
            _LOGGER.exception("Ошибка callback on_error при обработке NDTP")

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing or (self.max_connections and len(self._writers) >= self.max_connections):
            if not self._closing:
                self._error("max_connections")
            writer.close()
            return
        self._writers.add(writer)
        task = asyncio.create_task(self._handle(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _read(self, reader: asyncio.StreamReader, size: int) -> bytes:
        # Таймаут относится к чтению всей части кадра: медленная отправка по
        # одному байту не может бесконечно продлевать жизнь соединения.
        return await asyncio.wait_for(reader.readexactly(size), self.idle_timeout)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection_unit: int | None = None
        try:
            while not self._closing:
                try:
                    header = await self._read(reader, NPL_SIZE)
                except asyncio.IncompleteReadError as error:
                    if error.partial:
                        self._error("truncated_npl")
                    return
                # Невалидный NPL не позволяет доверять границам следующего
                # кадра. Не ищем signature внутри произвольной нагрузки.
                size = _frame_size(header)
                try:
                    body = await self._read(reader, size - NPL_SIZE)
                except asyncio.IncompleteReadError:
                    self._error("truncated_frame")
                    return
                try:
                    parsed = parse_frame(header + body)
                except NDTPError as error:
                    self._error(error.reason, error.detail)
                    continue
                if connection_unit is not None and parsed.unit_id != connection_unit:
                    self._error("connection_unit_mismatch")
                    return
                if parsed.kind == "handshake":
                    connection_unit = parsed.unit_id
                    continue
                if connection_unit is None:
                    self._error("realtime_before_handshake")
                    return
                try:
                    await asyncio.wait_for(
                        self.on_nav(parsed.unit_id, parsed.nav), self.callback_timeout,
                    )
                except asyncio.TimeoutError:
                    self._error("nav_callback_timeout")
                except Exception as error:
                    self._error("nav_callback_error", type(error).__name__)
        except asyncio.TimeoutError:
            self._error("idle_timeout")
        except NDTPError as error:
            self._error(error.reason, error.detail)
        except (ConnectionError, OSError) as error:
            self._error("connection_error", type(error).__name__)
        finally:
            self._writers.discard(writer)
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 1.0)
            except (ConnectionError, OSError, asyncio.TimeoutError):
                pass
