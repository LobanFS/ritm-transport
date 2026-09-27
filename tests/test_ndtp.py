"""Проверки бинарного контракта NDTP и реальных asyncio TCP-соединений."""

import asyncio
from datetime import datetime, timezone
import struct

import pytest

from backend.ndtp import NDTPError, NDTPServer, crc16_modbus, parse_frame


def reference_crc(data):
    """Независимый побайтовый CRC для сборки входных кадров в тестах."""
    crc = 0xFFFF
    for byte in data:
        for bit in range(8):
            carry = (crc ^ (byte >> bit)) & 1
            crc >>= 1
            if carry:
                crc ^= 0xA001
    return crc


def frame(body, *, unit=1166336, service=1, kind=101, request=2, flags=1):
    """Собирает fixture по little-endian layout и big-endian байтам CRC."""
    payload = struct.pack("<HHHI", service, kind, flags, request) + body
    header = (
        struct.pack("<HHH", 0x7E7E, len(payload), 0)
        + struct.pack(">H", reference_crc(payload))
        + struct.pack("<BIH", 2, unit, 0)
    )
    return header + payload


def handshake(unit=1166336, *, peer=None, major=6, minor=2):
    body = struct.pack("<HHHIII", major, minor, 0, unit if peer is None else peer, 65535, 0)
    return frame(body, unit=unit, service=0, kind=100, request=1)


def nav_cell(*, extra=0xE0, latitude=557551234, longitude=376173210, heading=123, number=0, speed=42):
    return bytes([0, number]) + struct.pack(
        "<IIIBBHHHHHBB", 1758790800, longitude, latitude, extra,
        180, speed, 57, heading, 1234, 150, 12, 3,
    )


def error_reason(packet):
    with pytest.raises(NDTPError) as error:
        parse_frame(packet)
    return error.value.reason


def test_crc_standard_vector_and_wire_byte_order():
    assert crc16_modbus(b"123456789") == 0x4B37
    assert reference_crc(b"123456789") == 0x4B37
    packet = frame(nav_cell())
    assert packet[6:8] == struct.pack(">H", reference_crc(packet[15:]))
    assert parse_frame(packet).kind == "realtime"
    wrong_endian = packet[:6] + packet[6:8][::-1] + packet[8:]
    assert error_reason(wrong_endian) == "crc_mismatch"


def test_handshake_checks_version_and_matching_unit():
    parsed = parse_frame(handshake())
    assert (parsed.unit_id, parsed.kind, parsed.nav) == (1166336, "handshake", None)
    assert error_reason(handshake(peer=42)) == "handshake_unit_mismatch"
    assert error_reason(handshake(major=7)) == "unsupported_protocol_version"


@pytest.mark.parametrize("extra,lat_sign,lon_sign", [(0xE0, 1, 1), (0x80, -1, -1), (0xA0, 1, -1), (0xC0, -1, 1)])
def test_navigation_values_hemispheres_and_stable_event_id(extra, lat_sign, lon_sign):
    packet = frame(nav_cell(extra=extra))
    nav = parse_frame(packet).nav
    assert nav["event_time"] == datetime.fromtimestamp(1758790800, timezone.utc)
    assert nav["lat"] == pytest.approx(lat_sign * 55.7551234)
    assert nav["lon"] == pytest.approx(lon_sign * 37.617321)
    assert nav["speed_kmh"] == 42
    assert nav["heading"] == 123
    assert nav["location_valid"] is True
    assert "received_at" not in nav
    assert nav["event_id"] == parse_frame(packet).nav["event_id"]
    assert nav["event_id"] != parse_frame(frame(nav_cell(), request=3)).nav["event_id"]


@pytest.mark.parametrize("kwargs", [dict(extra=0x60), dict(latitude=910000000), dict(longitude=1810000000)])
def test_unreliable_and_out_of_range_locations_are_flagged(kwargs):
    nav = parse_frame(frame(nav_cell(**kwargs))).nav
    assert nav["location_valid"] is False


def test_documented_zero_location_is_not_silently_rewritten():
    nav = parse_frame(frame(nav_cell(latitude=0, longitude=0))).nav
    assert (nav["lat"], nav["lon"], nav["location_valid"]) == (0, 0, True)


@pytest.mark.parametrize("unit", [0, 2147483647])
def test_unit_boundaries_and_full_wire_speed_range(unit):
    assert parse_frame(handshake(unit)).unit_id == unit
    assert parse_frame(frame(nav_cell(speed=65535), unit=unit)).nav["speed_kmh"] == 65535


def test_all_documented_sensor_payload_lengths_and_repeated_sensors():
    sensors = b"".join(bytes([kind, 0]) + bytes(size) for kind, size in [(2, 26), (8, 6), (10, 37), (15, 50), (16, 8)])
    sensors += bytes([8, 1]) + bytes(6)
    assert parse_frame(frame(nav_cell() + sensors)).nav["speed_kmh"] == 42


@pytest.mark.parametrize("kind", [3, 7, 100, 255])
def test_cell_without_documented_length_rejects_entire_packet(kind):
    assert error_reason(frame(nav_cell() + bytes([kind, 0]) + bytes(50))) == "unknown_cell_type"


def irma_cell(flags, number=0):
    # Длина/порядок независимо подтверждены официальным захватом ниже.
    return bytes([4, number]) + bytes.fromhex("78563412bc9a0102030405060708") + bytes([flags])


def test_official_emulator_captured_nav_irma_usi_packet():
    # Снято 2026-09-26 с ndtp-telemetry-emulator:1.0, фиксированные
    # искусственные координаты и поля; персональных/исторических данных нет.
    # Воспроизведение и provenance: tools/check_irma.py.
    packet = bytes.fromhex(
        "7e7e3f00000089950200cc11000000010065000100020000000000"
        "aeb0b76a9af26b16828e3b21e000000000000000000000000000"
        "040078563412bc9a0102030405060708a50800000000000000"
    )
    nav = parse_frame(packet).nav
    assert nav["lat"] == pytest.approx(55.7551234)
    assert nav["lon"] == pytest.approx(37.617321)
    assert nav["location_valid"] is True
    # present=0101, closed=1010: двери 1 и 3 представлены и открыты.
    assert nav["doors_open"] is True


@pytest.mark.parametrize("door", range(4))
def test_irma_each_door_open_closed_and_unavailable(door):
    present = 1 << door
    closed = 1 << (door + 4)
    assert parse_frame(frame(nav_cell() + irma_cell(present))).nav["doors_open"] is True
    assert parse_frame(frame(nav_cell() + irma_cell(present | closed))).nav["doors_open"] is False
    assert parse_frame(frame(nav_cell() + irma_cell(closed))).nav["doors_open"] is None


def test_irma_absence_is_unknown_and_multiple_sensors_are_aggregated():
    assert parse_frame(frame(nav_cell())).nav["doors_open"] is None
    assert parse_frame(frame(nav_cell() + irma_cell(0))).nav["doors_open"] is None
    assert parse_frame(frame(nav_cell() + irma_cell(0xFF))).nav["doors_open"] is False
    mixed = nav_cell() + irma_cell(0xFF) + irma_cell(0x01, number=1)
    assert parse_frame(frame(mixed)).nav["doors_open"] is True
    assert error_reason(frame(nav_cell() + irma_cell(1) + irma_cell(0x11))) == "duplicate_irma_number"


def test_irma_sensor_identity_tracks_presence_not_opening_or_cell_order():
    def key(*cells):
        return parse_frame(frame(nav_cell() + b"".join(cells))).nav["door_sensor_key"]

    # Переход состояния одной и той же двери сохраняет набор датчиков.
    closed, opened = irma_cell(0x11), irma_cell(0x01)
    assert key(closed) is not None and key(closed) == key(opened)
    # Появление другой двери или другого IRMA-модуля — новый набор.
    assert key(closed) != key(irma_cell(0x02))
    assert key(closed) != key(irma_cell(0x01, number=1))
    # Перестановка ячеек провода не меняет физическую идентичность.
    second = irma_cell(0x22, number=1)
    assert key(closed, second) == key(second, opened)
    assert key() is None and key(irma_cell(0xF0)) is None


def test_irma_malformed_trailing_cell_rejects_whole_frame():
    assert error_reason(frame(nav_cell() + irma_cell(1)[:-1])) == "truncated_cell"
    assert error_reason(frame(nav_cell() + irma_cell(1) + bytes([255, 0]))) == "unknown_cell_type"


@pytest.mark.parametrize("body,reason", [
    (b"", "missing_nav"),
    (nav_cell()[:-1], "truncated_cell"),
    (nav_cell() + b"\x08", "truncated_cell_header"),
    (nav_cell() + b"\x08\x00" + bytes(5), "truncated_cell"),
    (nav_cell() + nav_cell(), "duplicate_nav"),
    (b"\x08\x00" + bytes(6) + nav_cell(), "nav_not_first"),
    (nav_cell(number=1), "invalid_nav_number"),
    (nav_cell(heading=361), "invalid_heading"),
])
def test_malformed_cells(body, reason):
    assert error_reason(frame(body)) == reason


def test_frame_integrity_lengths_and_signature():
    packet = frame(nav_cell())
    assert error_reason(packet[:10]) == "truncated_npl"
    assert error_reason(packet[:-1]) == "frame_length_mismatch"
    assert error_reason(packet + b"x") == "frame_length_mismatch"
    assert error_reason(b"xx" + packet[2:]) == "invalid_signature"
    assert error_reason(packet[:-1] + bytes([packet[-1] ^ 1])) == "crc_mismatch"
    assert error_reason(packet[:2] + struct.pack("<H", 9) + packet[4:]) == "invalid_length"
    assert error_reason(packet[:2] + struct.pack("<H", 65535) + packet[4:]) == "invalid_length"


def test_unsupported_envelope_and_message_types():
    packet = frame(nav_cell())
    assert error_reason(packet[:4] + b"\x01\x00" + packet[6:]) == "unsupported_npl_flags"
    assert error_reason(packet[:8] + b"\x03" + packet[9:]) == "unsupported_npl_type"
    assert error_reason(frame(nav_cell(), unit=2147483648)) == "unit_id_out_of_range"
    assert error_reason(frame(nav_cell(), flags=0)) == "unsupported_nph_flags"
    assert error_reason(frame(nav_cell(), kind=102)) == "unsupported_nph_message"


async def wait_until(predicate):
    """Ожидает условие, сохраняя жёсткий предел длительности теста."""
    async def check():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(check(), 1)


async def make_server(**kwargs):
    events = asyncio.Queue()
    errors = asyncio.Queue()

    async def on_nav(unit, nav):
        events.put_nowait((unit, nav))

    server = NDTPServer(on_nav, errors.put_nowait, host="127.0.0.1", port=0, **kwargs)
    await server.start()
    return server, events, errors


async def close_client(writer):
    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionError:
        pass


def test_tcp_fragmentation_multiple_frames_and_no_ack():
    async def scenario():
        server, events, errors = await make_server()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            packet = handshake() + frame(nav_cell()) + frame(nav_cell(), request=3)
            # Разрывы внутри signature, NPL, NPH и payload; затем два кадра
            # оказываются в одном оставшемся chunk.
            offsets = [0, 1, 7, 16, 28, 47, len(packet)]
            for start, end in zip(offsets, offsets[1:]):
                writer.write(packet[start:end])
                await writer.drain()
                await asyncio.sleep(0)
            first = await asyncio.wait_for(events.get(), 1)
            second = await asyncio.wait_for(events.get(), 1)
            assert first[0] == second[0] == 1166336
            assert first[1]["event_id"] != second[1]["event_id"]
            assert server.connections == 1
            assert errors.empty()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(reader.read(1), 0.02)
        finally:
            await close_client(writer)
            await server.close()
        assert server.connections == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("packets,reason", [
    (frame(nav_cell()), "realtime_before_handshake"),
    (handshake() + frame(nav_cell(), unit=42), "connection_unit_mismatch"),
    (handshake() + handshake(unit=42), "connection_unit_mismatch"),
])
def test_connection_requires_handshake_and_same_unit(packets, reason):
    async def scenario():
        server, events, errors = await make_server()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            writer.write(packets)
            await writer.drain()
            assert await asyncio.wait_for(errors.get(), 1) == reason
            assert await asyncio.wait_for(reader.read(), 1) == b""
            assert events.empty()
            assert server.error_counts[reason] == 1
        finally:
            await close_client(writer)
            await server.close()
    asyncio.run(scenario())


def test_bad_complete_frames_are_counted_and_next_frame_still_works():
    async def scenario():
        server, events, errors = await make_server()
        _, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            corrupt = bytearray(frame(nav_cell()))
            corrupt[-1] ^= 1
            unknown = frame(nav_cell() + b"\x03\x00" + bytes(32))
            writer.write(handshake() + corrupt + unknown + frame(nav_cell(), request=4))
            await writer.drain()
            assert (await asyncio.wait_for(events.get(), 1))[0] == 1166336
            assert events.empty()
            assert server.error_counts["crc_mismatch"] == 1
            assert server.error_counts["unknown_cell_type"] == 1
            assert errors.qsize() == 2
        finally:
            await close_client(writer)
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("packet,reason", [(handshake()[:7], "truncated_npl"), (handshake()[:-2], "truncated_frame")])
def test_tcp_truncated_disconnect(packet, reason):
    async def scenario():
        server, events, errors = await make_server()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            writer.write(packet)
            writer.write_eof()
            await writer.drain()
            assert await asyncio.wait_for(errors.get(), 1) == reason
            assert await asyncio.wait_for(reader.read(), 1) == b""
            assert events.empty()
            await wait_until(lambda: server.connections == 0)
        finally:
            await close_client(writer)
            await server.close()
    asyncio.run(scenario())


def test_reconnect_requires_new_handshake_and_then_accepts_same_unit():
    async def scenario():
        server, events, errors = await make_server()
        try:
            for _ in range(2):
                _, writer = await asyncio.open_connection("127.0.0.1", server.port)
                writer.write(handshake() + frame(nav_cell()))
                await writer.drain()
                assert (await asyncio.wait_for(events.get(), 1))[0] == 1166336
                await close_client(writer)
                await wait_until(lambda: server.connections == 0)
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(frame(nav_cell()))
            await writer.drain()
            assert await asyncio.wait_for(errors.get(), 1) == "realtime_before_handshake"
            assert await asyncio.wait_for(reader.read(), 1) == b""
            await close_client(writer)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_idle_timeout_and_max_connections():
    async def scenario():
        server, events, errors = await make_server(max_connections=1, idle_timeout=0.1)
        first_reader, first_writer = await asyncio.open_connection("127.0.0.1", server.port)
        second_writer = None
        try:
            await wait_until(lambda: server.connections == 1)
            second_reader, second_writer = await asyncio.open_connection("127.0.0.1", server.port)
            assert await asyncio.wait_for(errors.get(), 1) == "max_connections"
            assert await asyncio.wait_for(second_reader.read(), 1) == b""
            assert await asyncio.wait_for(errors.get(), 1) == "idle_timeout"
            assert await asyncio.wait_for(first_reader.read(), 1) == b""
            assert events.empty()
            assert server.error_counts == {"max_connections": 1, "idle_timeout": 1}
        finally:
            await close_client(first_writer)
            if second_writer:
                await close_client(second_writer)
            await server.close()
    asyncio.run(scenario())


def test_oversized_frame_is_rejected_from_header_without_waiting_for_body():
    async def scenario():
        server, events, errors = await make_server()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            header = struct.pack("<HHHHBIH", 0x7E7E, 65535, 0, 0, 2, 1166336, 0)
            writer.write(header)
            await writer.drain()
            assert (await asyncio.wait_for(errors.get(), 1)).startswith("invalid_length:")
            assert await asyncio.wait_for(reader.read(), 1) == b""
            assert events.empty()
            assert server.error_counts["invalid_length"] == 1
        finally:
            await close_client(writer)
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure,reason", [("timeout", "nav_callback_timeout"), ("exception", "nav_callback_error")])
def test_callback_failure_is_bounded_and_does_not_drop_next_frame(failure, reason):
    async def scenario():
        calls = []
        finished = asyncio.Event()
        cancelled = asyncio.Event()
        errors = []

        async def on_nav(unit, nav):
            calls.append(nav["event_id"])
            if len(calls) == 1:
                if failure == "exception":
                    raise RuntimeError("Ошибка downstream")
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            else:
                finished.set()

        server = NDTPServer(on_nav, errors.append, host="127.0.0.1", port=0, callback_timeout=0.02)
        await server.start()
        _, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            writer.write(handshake() + frame(nav_cell()) + frame(nav_cell(), request=3))
            await writer.drain()
            await asyncio.wait_for(finished.wait(), 1)
            assert len(calls) == 2
            assert server.error_counts[reason] == 1
            assert errors[0].startswith(reason)
            if failure == "timeout":
                assert cancelled.is_set()
        finally:
            await close_client(writer)
            await server.close()
    asyncio.run(scenario())


def test_shutdown_cancels_callback_and_partial_reads_without_leaking_connections():
    async def scenario():
        entered = asyncio.Event()
        exited = asyncio.Event()
        errors = []

        async def blocked_callback(unit, nav):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()

        server = NDTPServer(blocked_callback, errors.append, host="127.0.0.1", port=0)
        await server.start()
        clients = []
        try:
            for packet in [handshake() + frame(nav_cell()), b"\x7e"]:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                clients.append((reader, writer))
                writer.write(packet)
                await writer.drain()
            await asyncio.wait_for(entered.wait(), 1)
            await wait_until(lambda: server.connections == 2)
            await asyncio.wait_for(server.close(), 1)
            assert server.connections == 0
            assert exited.is_set()
            assert not errors
            for reader, _ in clients:
                try:
                    assert await asyncio.wait_for(reader.read(), 1) == b""
                except ConnectionResetError:
                    # Закрытие TCP с непрочитанным неполным кадром может
                    # завершаться RST вместо FIN в зависимости от ОС.
                    pass
            await server.close()
        finally:
            for _, writer in clients:
                await close_client(writer)
            await server.close()
    asyncio.run(scenario())
