"use strict";

(() => {
  const requestedUI = new URLSearchParams(window.location.search).get("ui");
  const fullUI = (requestedUI || window.RITM_CONFIG?.uiMode || "full") !== "dispatcher";
  document.documentElement.dataset.uiMode = fullUI ? "full" : "dispatcher";
  const RISK = {
    red: { label: "Опоздание больше 2,5 мин", short: "Опоздание", color: "#d74e4b", rank: 0 },
    amber: { label: "Опоздание от 1 до 2,5 мин", short: "Наблюдение", color: "#e4aa24", rank: 1 },
    blue: { label: "Опережение больше 30 с", short: "Опережение", color: "#397cc6", rank: 2 },
    green: { label: "По графику · опоздание меньше минуты", short: "По графику", color: "#328a6d", rank: 3 },
    unknown: { label: "Нет оценки", short: "Нет оценки", color: "#328a6d", rank: 4 },
  };
  // Presentation bands are independent of P(delay > 120s) and the model contract.
  function delayBand(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "unknown";
    if (seconds > 150) return "red";
    if (seconds >= 60) return "amber";
    if (seconds < -30) return "blue";
    return "green";
  }
  function delayColor(seconds, band = delayBand(seconds)) {
    if (band !== "red") return RISK[band]?.color || RISK.unknown.color;
    const mix = Math.min(1, Math.max(0, (seconds - 150) / 450));
    const start = [215, 78, 75], end = [103, 17, 48];
    return "#" + start.map((v, i) => Math.round(v + (end[i] - v) * mix).toString(16).padStart(2, "0")).join("");
  }
  const $ = (id) => document.getElementById(id);
  const navItems = [...document.querySelectorAll(".section-nav .nav-item")];
  function syncNavigation() {
    const current = navItems.some(item => !item.hidden && (fullUI || !item.classList.contains("technical")) && item.hash === window.location.hash)
      ? window.location.hash : "#overview";
    navItems.forEach(item => {
      const active = item.hash === current;
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "location");
      else item.removeAttribute("aria-current");
    });
  }
  const refs = new Map();
  let snapshot = null;
  let selectedId = null;
  let connected = false;
  let lastSuccess = 0;
  let polling = false;
  let refreshRequested = false;
  let pollTimer = null;
  let busy = false;
  let incidentSignature = "";
  let attentionSignature = "";
  let routeSignature = "";
  let map = null;
  let tileLayer = null;
  let targetMarker = null;
  let mapContext = null;
  let directionContext = null;
  let directionClock = null;
  let followSelected = false;
  let replayConfigSignature = "";
  let toastTimer = null;
  const markers = new Map();
  const tableRows = new Map();

  const text = (id, value) => {
    const element = refs.get(id) || $(id);
    if (!element) return;
    refs.set(id, element);
    const next = value === null || value === undefined ? "—" : String(value);
    if (element.textContent !== next) element.textContent = next;
  };
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const list = (value) => Array.isArray(value) ? value : [];
  const idOf = (value) => String(value ?? "");
  const integer = (value) => finite(value) ? Math.round(value).toLocaleString("ru-RU") : "—";
  const timestamp = (value) => {
    if (typeof value !== "string" || !value) return null;
    const ms = Date.parse(value);
    return Number.isFinite(ms) ? ms : null;
  };
  const time = (value, seconds = false) => {
    const ms = timestamp(value);
    if (ms === null) return "—";
    return new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", hour: "2-digit", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}) }).format(ms);
  };
  const duration = (value, compact = false) => {
    if (!finite(value) || value < 0) return "Нет данных";
    const total = Math.round(value);
    if (total < 60) return `${total} с`;
    const minutes = Math.floor(total / 60);
    if (minutes < 60) return compact || total % 60 === 0 ? `${minutes} мин` : `${minutes} мин ${total % 60} с`;
    return `${Math.floor(minutes / 60)} ч ${minutes % 60} мин`;
  };
  const delay = (value, compact = false) => {
    if (!finite(value)) return "Не оценена";
    const total = Math.round(Math.abs(value));
    const sign = value > 0 ? "+" : value < 0 ? "−" : "";
    if (compact) return `${sign}${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
    return sign + duration(total);
  };
  const probabilityLabel = (value, status) => finite(value) && value >= 0 && value <= 1 && status !== "unavailable"
    ? status === "transferred" ? `≈${Math.round(value * 100)}% · приближённо` : `${Math.round(value * 100)}%` : "не оценена";
  const expectedArrival = (plannedAt, delaySeconds) => {
    const planned = timestamp(plannedAt);
    if (planned === null || !finite(delaySeconds)) return null;
    const expected = new Date(planned + delaySeconds * 1000);
    return Number.isFinite(expected.getTime()) ? expected.toISOString() : null;
  };
  const safeColor = (value, fallback = "#819bb0") => typeof value === "string" && /^#[\da-f]{3,8}$/i.test(value) ? value : fallback;
  const apiStale = () => !lastSuccess || Date.now() - lastSuccess > 3500;
  const effectiveRisk = (vehicle) => {
    if (apiStale() || vehicle.status !== "fresh" || vehicle.prediction?.risk === "unknown" ||
        (vehicle.prediction_availability && vehicle.prediction_availability.code !== "ready")) return "unknown";
    return delayBand(vehicle.prediction?.predicted_delay_s);
  };
  const vehicleColor = vehicle => delayColor(vehicle.prediction?.predicted_delay_s, effectiveRisk(vehicle));
  const displayedDelay = vehicle => fullUI || effectiveRisk(vehicle) !== "unknown" ? vehicle.prediction?.predicted_delay_s : null;
  const availabilityLabel = (vehicle) => {
    if (apiStale()) return "Нет связи с API";
    const labels = { source_unavailable: "Нет связи с источником", no_target: "Нет цели в 10–15 мин",
      telemetry_missing: "Нет GPS", telemetry_stale: "GPS устарел",
      deviation_expired: "Отклонение устарело", deviation_missing: "Нет текущего отклонения",
      target_reached: "Цель уже посещена", prediction_pending: "Ожидается расчёт",
      model_input_unavailable: "Недостаточно входов" };
    return labels[vehicle.prediction_availability?.code] || "Нет оценки";
  };
  const routeOf = (vehicle) => list(snapshot?.routes).find((route) => idOf(route.route_id) === idOf(vehicle.route_id));
  const vehicleName = (vehicle) => vehicle?.label || `Автобус ${vehicle?.tr_id ?? "—"}`;
  const routeName = (vehicle) => routeOf(vehicle)?.name || vehicle.route_id || "Без маршрута";
  const routeCode = (vehicle) => routeOf(vehicle)?.name || vehicle.route_id || "—";
  const node = (tag, className, value) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined) element.textContent = String(value);
    return element;
  };
  const setStatus = (id, status) => { $(id).className = `status-dot ${status ? `is-${status}` : ""}`; };
  const setButtonBusy = () => {
    document.querySelectorAll("[data-mode]").forEach((button) => { button.disabled = busy || !snapshot; });
    ["play-button", "demo-speed", "source-button", "reset-button"].forEach((id) => {
      $(id).disabled = busy || !connected || !["demo", "replay", "generator"].includes(snapshot?.mode);
    });
    if (playback().finished) $("play-button").disabled = true;
    $("source-button").disabled = busy || !connected || snapshot?.mode !== "demo";
    ["replay-load", "replay-split", "replay-source", "replay-start", "replay-warmup", "replay-selection", "replay-vehicles", "replay-timezone"].forEach(id => { $(id).disabled = busy || !connected; });
    $("replay-vehicles").disabled = busy || !connected || $("replay-selection").value !== "manual";
    $("replay-vehicles").required = $("replay-selection").value === "manual";
  };

  function filteredVehicles() {
    const route = $("route-filter").value;
    const risk = $("risk-filter").value;
    const query = $("vehicle-search").value.trim().toLocaleLowerCase("ru-RU");
    return list(snapshot?.vehicles).filter((vehicle) => {
      if (route !== "all" && idOf(vehicle.route_id) !== route) return false;
      if (risk === "stale" && vehicle.status === "fresh" && !apiStale()) return false;
      if (risk !== "all" && risk !== "stale" && effectiveRisk(vehicle) !== risk) return false;
      const searchable = [vehicle.label, vehicle.tr_id, vehicle.unit_id, vehicle.route_id, routeName(vehicle)].join(" ").toLocaleLowerCase("ru-RU");
      return !query || searchable.includes(query);
    }).sort((a, b) => RISK[effectiveRisk(a)].rank - RISK[effectiveRisk(b)].rank || (b.prediction?.predicted_delay_s || 0) - (a.prediction?.predicted_delay_s || 0) || idOf(a.tr_id).localeCompare(idOf(b.tr_id)));
  }

  function renderRouteOptions() {
    const routes = list(snapshot?.routes);
    const signature = JSON.stringify([snapshot?.mode, routes.map((route) => [route.route_id, route.name])]);
    if (signature === routeSignature) return;
    routeSignature = signature;
    const previous = $("route-filter").value;
    const replay = snapshot?.mode === "replay";
    text("route-filter-label", replay ? (fullUI ? "План автобуса" : "Автобус") : "Маршрут");
    $("route-filter").setAttribute("aria-label", replay ? "Фильтр по плану автобуса" : "Фильтр по маршруту");
    $("route-filter").replaceChildren(new Option(replay ? (fullUI ? "Все планы ТС" : "Все автобусы") : "Все маршруты", "all"));
    routes.forEach((route) => $("route-filter").append(new Option(replay && !fullUI ? String(route.name || route.route_id).replace(/^План ТС /, "Автобус ") : String(route.name || route.route_id), idOf(route.route_id))));
    $("route-filter").value = Array.from($("route-filter").options).some((option) => option.value === previous) ? previous : "all";
  }

  const latLng = (lon, lat) => finite(lon) && finite(lat) && Math.abs(lon) <= 180 && Math.abs(lat) <= 85.0511 ? [lat, lon] : null;
  const chosenVehicle = () => list(snapshot?.vehicles).find((vehicle) => idOf(vehicle.tr_id) === selectedId);
  const playback = () => snapshot?.mode === "generator" ? snapshot.generator || {} : snapshot?.mode === "replay" ? snapshot.replay || {} : snapshot?.demo || {};

  function setFollow(enabled) {
    followSelected = enabled;
    $("map-follow").setAttribute("aria-pressed", String(enabled));
    $("map-follow").classList.toggle("is-active", enabled);
    text("map-follow", `Следовать: ${enabled ? "вкл." : "выкл."}`);
  }

  function focusVehicle() {
    const vehicle = chosenVehicle();
    const position = vehicle && latLng(vehicle.lon, vehicle.lat);
    if (!map || !position) { toast("У выбранного автобуса пока нет координат."); return; }
    map.setView(position, Math.max(map.getZoom(), 15), { animate: false });
  }

  function fitMap(onlySelectedRoute = false) {
    if (!map) return;
    const vehicle = chosenVehicle();
    const routeId = onlySelectedRoute ? idOf(vehicle?.route_id) : $("route-filter").value;
    const points = [];
    if (onlySelectedRoute) list(snapshot?.routes).filter(route => idOf(route.route_id) === routeId).forEach(route => {
      list(route.path).forEach(point => { const p = Array.isArray(point) && latLng(point[0], point[1]); if (p) points.push(p); });
      list(route.stops).forEach(stop => { const p = latLng(stop.lon, stop.lat); if (p) points.push(p); });
    });
    filteredVehicles().filter(v => !onlySelectedRoute || idOf(v.route_id) === routeId).forEach(v => { const p = latLng(v.lon, v.lat); if (p) points.push(p); });
    if (points.length) { setFollow(false); map.fitBounds(points, { padding: [48, 48], maxZoom: 18, animate: false }); }
    else toast("Нет координат для выбранного маршрута.");
  }

  function initMap() {
    if (map || !window.L) return;
    map = L.map("route-map", { zoomControl: false, scrollWheelZoom: true, attributionControl: true, maxZoom: 19, minZoom: 3 });
    map.setView([55.751244, 37.618423], 11);
    L.control.zoom({ position: "topright", zoomInTitle: "Увеличить масштаб", zoomOutTitle: "Уменьшить масштаб" }).addTo(map);
    map.attributionControl.setPrefix(false);
    tileLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
    });
    let tileErrors = 0;
    let tileSuccesses = 0;
    let tileWait = null;
    tileLayer.on("loading", () => {
      tileErrors = 0; tileSuccesses = 0;
      text("tile-status", "Подложка OpenStreetMap загружается…");
      clearTimeout(tileWait);
      tileWait = setTimeout(() => {
        if (map.hasLayer(tileLayer) && !tileSuccesses) {
          text("tile-status", "Подложка не отвечает. GPS и остановки доступны; можно скрыть подложку.");
          $("tile-status").classList.add("is-warning");
        }
      }, 8000);
    });
    tileLayer.on("tileload", () => { tileSuccesses += 1; });
    tileLayer.on("tileerror", () => {
      tileErrors += 1;
      if (!map.hasLayer(tileLayer)) return;
      text("tile-status", "Подложка недоступна полностью или частично. GPS и остановки продолжают работать.");
      $("tile-status").classList.add("is-warning");
    });
    tileLayer.on("load", () => {
      clearTimeout(tileWait);
      if (!map.hasLayer(tileLayer)) return;
      text("tile-status", tileErrors ? "Часть подложки не загрузилась. GPS и остановки продолжают работать." : tileSuccesses ? "Подложка OpenStreetMap · требуется интернет" : "Подложка OpenStreetMap");
      $("tile-status").classList.toggle("is-warning", tileErrors > 0);
    });
    tileLayer.addTo(map);
    $("tile-toggle").setAttribute("aria-pressed", "true");
    map.on("dragstart", () => setFollow(false));
    // Programmatic setView never disables following; a deliberate wheel/pinch does.
    $("route-map").addEventListener("wheel", () => setFollow(false), { passive: true });
    $("route-map").addEventListener("touchstart", event => { if (event.touches.length > 1) setFollow(false); }, { passive: true });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => map.invalidateSize({ pan: false })).observe($("map-canvas"));
  }

  // A direction belongs to a particular accepted GPS event, never to a marker's
  // screen position. Repainting, filtering and replay resets are not movement.
  function markerDirection(vehicle, position, previous, clock, stale) {
    const eventTime = timestamp(vehicle.event_time);
    if (stale || vehicle.status !== "fresh" || eventTime === null || clock === null ||
        eventTime > clock || clock - eventTime > 60000) return null;
    const current = { eventTime, position, direction: null, source: null };
    if (finite(vehicle.heading) && vehicle.heading >= 0 && vehicle.heading < 360) {
      current.direction = vehicle.heading;
      current.source = "курс GPS";
      return current;
    }
    if (!previous) return current;
    const samePosition = position[0] === previous.position[0] && position[1] === previous.position[1];
    if (eventTime === previous.eventTime && samePosition) {
      // Multiple UI refreshes of one event retain only its derived bearing.
      // A removed/invalid heading must not silently reuse the preceding course.
      if (previous.source === "движение между GPS-точками") {
        current.direction = previous.direction;
        current.source = previous.source;
      }
      return current;
    }
    if (eventTime <= previous.eventTime || eventTime - previous.eventTime > 60000) return current;
    const radians = Math.PI / 180;
    const lat1 = previous.position[0] * radians, lat2 = position[0] * radians;
    const deltaLon = (position[1] - previous.position[1]) * radians;
    const dy = lat2 - lat1, dx = deltaLon * Math.cos((lat1 + lat2) / 2);
    const distance = Math.hypot(dx, dy) * 6371000;
    // Suppress tiny GPS jitter and implausible jumps; this is display logic,
    // not evidence of an arrival or a feature passed to the model.
    if (distance < 2 || distance / ((eventTime - previous.eventTime) / 1000) > 70) return current;
    current.direction = (Math.atan2(Math.sin(deltaLon) * Math.cos(lat2),
      Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(deltaLon)) / radians + 360) % 360;
    current.source = "движение между GPS-точками";
    return current;
  }

  function renderMap(vehicles) {
    initMap();
    const clock = timestamp(snapshot?.clock_time);
    const contextKey = JSON.stringify([snapshot?.mode, snapshot?.context?.version,
      snapshot?.context?.loaded_at, snapshot?.generator?.session_id,
      snapshot?.replay?.dataset_split, snapshot?.replay?.start, snapshot?.replay?.tr_ids]);
    if (contextKey !== directionContext || (clock !== null && directionClock !== null && clock < directionClock)) {
      markers.forEach(marker => { marker.directionState = null; marker.directionAngle = null; });
      directionContext = contextKey;
    }
    directionClock = clock;
    const routeFilter = $("route-filter").value;
    const visibleIds = new Set();
    let onMap = 0;
    vehicles.forEach(vehicle => {
      const key = idOf(vehicle.tr_id);
      const position = latLng(vehicle.lon, vehicle.lat);
      if (!position || !map) return;
      visibleIds.add(key);
      onMap += 1;
      let marker = markers.get(key);
      if (!marker) {
        const body = node("div", "bus-pin");
        const code = node("span", "bus-pin-code");
        const label = node("span", "bus-pin-label");
        body.append(code, label);
        const layer = L.marker(position, { icon: L.divIcon({ className: "bus-map-marker", html: body, iconSize: [100, 44], iconAnchor: [12, 22] }), keyboard: true, riseOnHover: true });
        layer.on("click", () => selectVehicle(key));
        layer.addTo(map);
        marker = { layer, body, code, label };
        markers.set(key, marker);
      }
      if (!map.hasLayer(marker.layer)) marker.layer.addTo(map);
      marker.directionState = markerDirection(vehicle, position, marker.directionState, clock, apiStale());
      const direction = marker.directionState?.direction;
      const hasDirection = finite(direction);
      marker.body.classList.toggle("has-direction", hasDirection);
      if (hasDirection) {
        const previous = finite(marker.directionAngle) ? marker.directionAngle : direction;
        marker.directionAngle = previous + ((direction - previous + 540) % 360 + 360) % 360 - 180;
        marker.body.style.setProperty("--pin-angle", `${marker.directionAngle - 90}deg`);
      } else {
        marker.directionAngle = null;
        marker.body.style.removeProperty("--pin-angle");
      }
      marker.layer.setLatLng(position);
      const risk = effectiveRisk(vehicle);
      const directionLabel = hasDirection ? `направление ${Math.round(direction)}°, ${marker.directionState.source}` : "направление неизвестно";
      const availability = risk === "unknown" ? `, ${availabilityLabel(vehicle)}` : "";
      const accessibleLabel = `${vehicleName(vehicle)}, ${RISK[risk].label}${availability}, прогноз ${delay(displayedDelay(vehicle))}, ${directionLabel}`;
      const element = marker.layer.getElement();
      element.classList.toggle("is-selected", key === selectedId);
      element.classList.toggle("is-stale", vehicle.status !== "fresh" || apiStale());
      element.setAttribute("aria-pressed", String(key === selectedId));
      element.setAttribute("aria-label", accessibleLabel);
      element.title = accessibleLabel;
      marker.body.style.setProperty("--pin-color", vehicleColor(vehicle));
      marker.body.style.setProperty("--pin-text", risk === "amber" ? "#4b3309" : "#fff");
      element.classList.toggle("is-unassessed", risk === "unknown");
      marker.code.textContent = String(vehicle.tr_id);
      const label = vehicleName(vehicle).replace(/^Автобус\s*/i, "").slice(0, 18);
      marker.label.textContent = label === String(vehicle.tr_id) ? "" : label;
      marker.layer.setZIndexOffset(key === selectedId ? 1000 : 0);
    });
    markers.forEach((marker, key) => { if (!visibleIds.has(key) && map?.hasLayer(marker.layer)) map.removeLayer(marker.layer); });
    const existingIds = new Set(list(snapshot?.vehicles).map(vehicle => idOf(vehicle.tr_id)));
    markers.forEach((marker, key) => { if (!existingIds.has(key)) { marker.layer.remove(); markers.delete(key); } });
    const selected = chosenVehicle();
    const target = selected?.prediction?.target || selected?.target;
    const targetPosition = target && latLng(target.lon, target.lat);
    if (map && targetPosition) {
      if (!targetMarker) targetMarker = L.circleMarker(targetPosition, { radius: 10, color: "#203349", weight: 3, fillColor: "#fff", fillOpacity: .6 }).addTo(map);
      targetMarker.setLatLng(targetPosition).bindTooltip(node("span", "", `Цель: ${target.name} · план ${time(target.scheduled_at)} МСК`));
      if (!map.hasLayer(targetMarker)) targetMarker.addTo(map);
    } else if (targetMarker) { targetMarker.remove(); targetMarker = null; }
    const context = snapshot?.mode === "generator" ? `generator:${snapshot.generator?.session_id}` : snapshot?.mode === "replay" ? `replay:${snapshot.replay?.dataset_split || "validate"}:${snapshot.replay?.start}:${list(snapshot.replay?.tr_ids).join(",")}` : snapshot?.mode;
    if (map && context && context !== mapContext && onMap > 0) { mapContext = context; fitMap(); }
    if (map && followSelected && selected) { const p = latLng(selected.lon, selected.lat); if (p) map.panTo(p, { animate: false }); }
    $("map-empty").hidden = onMap > 0;
    if (!snapshot) {
      text("map-empty-title", "Ожидаем данные");
      text("map-empty-text", "Автобусы появятся после получения координат.");
    } else if (!list(snapshot.vehicles).length && snapshot.mode === "live") {
      text("map-empty-title", "Ожидаем живой поток");
      text("map-empty-text", "Подключите источник NDTP. Здесь появятся полученные автобусы.");
    } else if (!vehicles.length && list(snapshot.vehicles).length) {
      text("map-empty-title", "По фильтрам ничего не найдено");
      text("map-empty-text", "Выберите другой маршрут, состояние или измените поиск.");
    } else if (vehicles.length) {
      text("map-empty-title", "Пока нет координат");
      text("map-empty-text", "Автобусы доступны в списке. Ждём валидную телеметрию.");
    } else {
      text("map-empty-title", "На линии пока нет автобусов");
      text("map-empty-text", "Ожидаем телеметрию от источника.");
    }
    text("map-count", `${onMap} из ${vehicles.length} на карте`);
    ["map-focus", "map-follow"].forEach(id => { $(id).disabled = !selected; });
  }

  function renderSummary() {
    const vehicles = list(snapshot?.vehicles);
    text("kpi-total", snapshot ? integer(vehicles.length) : "—");
    text("kpi-red", snapshot ? vehicles.filter((v) => effectiveRisk(v) === "red").length : "—");
    text("kpi-amber", snapshot ? vehicles.filter((v) => effectiveRisk(v) === "amber").length : "—");
    const stale = vehicles.filter((v) => v.status !== "fresh" || apiStale()).length;
    const unknown = vehicles.filter((v) => effectiveRisk(v) === "unknown").length;
    text("kpi-stale", snapshot ? unknown : "—");
    text("kpi-total-note", snapshot ? `${list(snapshot.routes).length} ${snapshot.mode === "replay" ? "планов ТС" : "маршрутов"} · ${vehicles.filter((v) => v.status === "fresh" && !apiStale()).length} со свежими данными` : "Ожидаем телеметрию");
    text("kpi-stale-note", snapshot ? `${stale} ТС без свежего GPS` : "Ожидаем данные");
    text("nav-event-count", list(snapshot?.incidents).length);
  }

  function renderAttention() {
    // This queue is built only from current vehicle predictions, never incident history.
    const vehicles = list(snapshot?.vehicles);
    const unknown = vehicles.filter((vehicle) => effectiveRisk(vehicle) === "unknown").length;
    const problems = vehicles.filter((vehicle) => ["red", "amber"].includes(effectiveRisk(vehicle)))
      .sort((a, b) => RISK[effectiveRisk(a)].rank - RISK[effectiveRisk(b)].rank
        || (b.prediction?.predicted_delay_s || 0) - (a.prediction?.predicted_delay_s || 0)
        || idOf(a.tr_id).localeCompare(idOf(b.tr_id)));
    text("attention-count", problems.length);
    text("attention-note", apiStale() ? "Нет актуального состояния: ожидаем связь с API."
      : !vehicles.length ? "Ожидаем автобусы от выбранного источника."
      : `${unknown ? `${unknown} ТС без актуальной оценки риска. ` : ""}${problems.length ? "Сначала красные сигналы; нажмите строку, чтобы открыть автобус." : "Повышенных прогнозов среди ТС с актуальной оценкой сейчас нет."}`);
    const rows = problems.map((vehicle) => ({
      id: idOf(vehicle.tr_id), label: vehicleName(vehicle), route: routeCode(vehicle), risk: effectiveRisk(vehicle),
      section: vehicle.section || "Плановый участок не указан", target: vehicle.prediction?.target || vehicle.target, predicted: vehicle.prediction?.predicted_delay_s,
      probability: vehicle.prediction?.probability_late, probabilityStatus: vehicle.prediction?.probability_status, probabilityNote: vehicle.prediction?.probability_note,
      selected: idOf(vehicle.tr_id) === selectedId,
    }));
    const signature = JSON.stringify(rows);
    if (signature === attentionSignature) return;
    attentionSignature = signature;
    const focusedId = $("attention-list").contains(document.activeElement) ? document.activeElement?.dataset.vehicleId : null;
    const buttons = [];
    $("attention-list").replaceChildren(...rows.map((row) => {
      const item = node("li");
      const button = node("button", `attention-item risk-${row.risk}${row.selected ? " is-selected" : ""}`);
      button.type = "button";
      button.dataset.vehicleId = row.id;
      button.setAttribute("aria-pressed", String(row.selected));
      const identity = node("span", "attention-identity");
      identity.append(node("span", `risk-badge risk-${row.risk}`, RISK[row.risk].short), node("strong", "", !fullUI && snapshot?.mode === "replay" ? row.label : `${row.route} · ${row.label}`));
      const section = node("span", "attention-section");
      section.append(node("strong", "", row.target?.name || "Цель не определена"), node("span", "", row.section));
      if (row.target) section.append(node("span", "attention-arrival", `План ${time(row.target.scheduled_at)} → ожидаем ${time(expectedArrival(row.target.scheduled_at, row.predicted))} МСК`));
      const forecast = node("span", "attention-forecast");
      forecast.append(node("strong", "", delay(row.predicted)), node("span", "", `Вероятность >2 мин: ${probabilityLabel(row.probability, row.probabilityStatus)}`));
      forecast.title = row.probabilityNote || "Вероятность задержки более двух минут; цвет определяется прогнозом в секундах.";
      button.append(identity, section, forecast);
      buttons.push(button);
      item.append(button);
      return item;
    }));
    if (focusedId) buttons.find((button) => button.dataset.vehicleId === focusedId)?.focus({ preventScroll: true });
  }

  function renderDetail() {
    const vehicle = list(snapshot?.vehicles).find((v) => idOf(v.tr_id) === selectedId);
    $("detail-empty").hidden = !!vehicle;
    $("detail-content").hidden = !vehicle;
    $("forecast-insights").hidden = !vehicle;
    text("cause-vehicle", vehicle ? vehicleName(vehicle) : "Выберите автобус");
    if (!vehicle) {
      text("detail-cause", "Выберите автобус на карте или в списке");
      text("detail-cause-status", ""); text("detail-observed-at", "");
      $("detail-observations").replaceChildren(); delete $("detail-observations").dataset.content;
      text("detail-recommendation", "—");
      return;
    }
    const prediction = vehicle.prediction;
    const target = prediction?.target || vehicle.target;
    const risk = effectiveRisk(vehicle);
    text("detail-label", vehicleName(vehicle));
    text("detail-route", routeCode(vehicle));
    $("detail-route").title = String(routeName(vehicle));
    text("detail-id", `ID ${vehicle.tr_id}`);
    $("detail-id").hidden = !fullUI;
    $("detail-route").hidden = !fullUI && snapshot.mode === "replay";
    $("detail-risk-dot").style.background = vehicleColor(vehicle);
    $("forecast-box").className = `forecast-box risk-${risk}`;
    text("detail-delay", delay(displayedDelay(vehicle)));
    $("detail-delay").style.color = vehicleColor(vehicle);
    const inputMode = snapshot.mode === "replay" ? snapshot.replay?.deviation_source || "csv_snapshot" : snapshot.context?.arrival_mode;
    const missingInput = inputMode === "csv_snapshot" ? "На текущий момент нет действующего отклонения из points.csv"
      : inputMode === "gps" ? "Ожидаем подтверждение посещения остановки для расчёта отклонения"
      : "Ожидаем подтверждённое прибытие или внешнее текущее отклонение";
    const forecastStatus = apiStale() ? "Сохранённые данные · ожидаем восстановление связи с API"
      : vehicle.prediction_availability && vehicle.prediction_availability.code !== "ready" ? vehicle.prediction_availability.message
      : vehicle.status !== "fresh" ? "Нет свежего GPS · актуальный риск не показывается"
      : !target ? "В плановом окне 10–15 минут сейчас нет остановки"
      : !finite(vehicle.cur_dev_s) ? missingInput
      : !finite(prediction?.predicted_delay_s) ? "Входные данные есть · ожидаем завершения расчёта"
      : prediction.method === "fallback" ? "Резервная оценка · подробности в запросе и ответе"
      : "Отклонение от расписания";
    text("detail-forecast-status", fullUI ? forecastStatus : risk === "unknown" ? availabilityLabel(vehicle)
      : prediction?.method === "fallback" ? "Резервная оценка" : "Отклонение от расписания");
    text("detail-target", target?.name || "Цель не определена");
    text("detail-planned-arrival", target ? `${time(target.scheduled_at, true)} МСК` : "—");
    const expected = expectedArrival(target?.scheduled_at, displayedDelay(vehicle));
    text("detail-expected-arrival", expected ? `${time(expected, true)} МСК` : "Не оценено");
    text("detail-risk", RISK[risk].short);
    text("detail-current", delay(vehicle.cur_dev_s));
    const deviation = vehicle.current_deviation;
    const fromArrival = deviation && ["arrival", "demo_arrival", "generator_arrival", "gps_estimate", "door_estimate"].includes(deviation.source);
    text("detail-deviation-source", !deviation ? "Текущее отклонение пока неизвестно" : fromArrival
      ? `${deviation.source === "gps_estimate" ? "GPS + расписание" : "Подтверждение прибытия"} · ${deviation.stop_name || deviation.planned_stop_id || "остановка"}`
      : deviation.source === "csv_snapshot" ? "Текущее отклонение из points.csv" : "Текущее отклонение из внешнего источника");
    const deviationAge = deviation ? (timestamp(snapshot.clock_time) - timestamp(deviation.observed_at)) / 1000 : null;
    const gpsStatus = vehicle.gps_detector;
    const gpsReasons = {awaiting_gps:"ожидаем GPS", terminal_occupancy_without_reentry:"ждём выезда и возвращения на конечную", no_schedule:"нет плана", moving:"автобус движется", confirming_stop:"ждём следующую точку для подтверждения", candidate_after_gap:"подтверждаем остановку после пропуска GPS", ambiguous_stop:"несколько подходящих посещений; пока неизвестно", outside_stop_window:"не найдено подходящее плановое посещение", invalid_gps:"недостоверный GPS", late_or_duplicate:"повтор или запоздалый пакет", event_after_receipt:"несогласованные часы сообщения", receipt_out_of_order:"нарушен порядок получения", plan_finished:"план завершён", at_confirmed_stop:"остановка уже распознана", confirmed:"прибытие распознано", confirmed_by_doors:"прибытие оценено по телеметрии", confirmed_after_departure:"прибытие подтверждено после отправления", confirmed_after_skipped_visits:"прибытие распознано после пропуска посещений", outside_absolute_schedule_window:"GPS не согласуется с планом в допустимом окне", skip_before_expected_time:"ожидаем однозначное сопоставление с планом"};
    text("detail-deviation-time", ["gps_estimate", "door_estimate"].includes(deviation?.source) ? `Подтверждено в ${time(deviation.received_at, true)} МСК; оценка по дискретной телеметрии, не точный операционный факт.${finite(deviation.uncertainty_s) ? ` Интервал наблюдений ${deviation.uncertainty_s} с.` : ""}` : deviation ? `${fromArrival ? "Прибытие" : "Подсказка"} ${duration(deviationAge)} назад${deviation.valid_until ? ` · действует до ${time(deviation.valid_until, true)} МСК` : ""}` : gpsStatus ? `GPS-детектор: ${gpsReasons[gpsStatus.reason] || "ожидание подтверждения остановки"}` : "Нет входного события прибытия");
    text("detail-prediction-input", prediction ? `Вход расчёта: ${delay(vehicle.prediction_current_delay_s)} · ${time(prediction.issued_at, true)} МСК` : "Завершённого прогноза для текущей цели нет");
    $("detail-trace").href = `/api/v1/vehicles/${encodeURIComponent(vehicle.tr_id)}/forecast-trace`;
    $("detail-trace").hidden = !prediction;
    text("detail-speed", finite(vehicle.speed_kmh) ? `${Math.round(vehicle.speed_kmh)} км/ч` : "Нет данных");
    const probability = risk === "unknown" ? null : prediction?.probability_late;
    text("detail-probability", probabilityLabel(probability, prediction?.probability_status));
    $("detail-probability").title = prediction?.probability_note || "Для текущих входов оценка вероятности недоступна";
    text("detail-probability-note", apiStale() || vehicle.status !== "fresh" ? "Актуальность данных не подтверждена; вероятность сейчас не показана." : prediction?.probability_note || "Для этого прогноза вероятность не оценена");
    text("detail-freshness", finite(vehicle.age_s) ? `${duration(vehicle.age_s)} назад${apiStale() ? " · данные не обновляются" : ""}` : "Нет данных");
    $("detail-freshness").classList.toggle("is-stale", vehicle.status !== "fresh" || apiStale());
    const explanation = vehicle.explanation;
    const modelExplanation = prediction?.forecast_explanation;
    const transformerAnalysis = modelExplanation?.transformer_analysis;
    text("detail-cause", apiStale() ? "Нет актуальных данных" : explanation?.cause_status === "hypothesis"
      ? explanation.possible_cause : explanation?.summary || "Причина задержки не установлена");
    text("detail-cause-status", apiStale() ? "Данные не обновляются; актуальность не подтверждена" : explanation?.cause_status === "hypothesis" ? "Гипотеза по телеметрии, требует проверки" : explanation?.observation_status === "observed" ? "Наблюдение · физическая причина не установлена" : "Причина не установлена");
    text("detail-observed-at", explanation?.evaluated_at ? `Наблюдения на ${time(explanation.evaluated_at, true)} МСК` : "");
    const observations = list(explanation?.observations)
      .filter(item => item.kind !== "data_quality" && item.title !== explanation?.summary)
      .slice(0, 3).map(item => item.evidence);
    const observationsKey = JSON.stringify(observations);
    if ($("detail-observations").dataset.content !== observationsKey) {
      $("detail-observations").replaceChildren(...observations.map(value => node("li", "", value)));
      $("detail-observations").dataset.content = observationsKey;
    }
    const modelFactors = list(modelExplanation?.factors)
      .filter(item => item && finite(item.effect_s))
      .sort((left, right) => Math.abs(right.effect_s) - Math.abs(left.effect_s))
      .slice(0, 3)
      .map(item => `${item.title}: ${item.effect_s >= 0 ? "повышает" : "снижает"} прогноз примерно на ${Math.abs(item.effect_s).toFixed(0)} с`);
    const factorsKey = JSON.stringify(modelFactors);
    if ($("detail-model-factors").dataset.content !== factorsKey) {
      $("detail-model-factors").replaceChildren(...modelFactors.map(value => node("li", "", value)));
      $("detail-model-factors").dataset.content = factorsKey;
    }
    text("detail-model-explanation-note", modelExplanation
      ? `${transformerAnalysis?.summary || ""} ${modelExplanation.history_points} наблюдений за ${duration(modelExplanation.history_span_s)}. Вклад в расчёт, не физическая причина.`
      : "Аддитивное объяснение обученной модели для этого прогноза недоступно.");
    text("detail-recommendation", vehicle.recommendation || "Оцените ситуацию на маршруте. Эффект изменения выпуска и времени стоянки модель не рассчитывает.");

  }

  function createTableRow(key) {
    const row = node("tr");
    row.dataset.vehicleId = key;
    const first = node("td");
    const name = node("span", "bus-name");
    const code = node("span", "route-chip");
    const label = node("span");
    name.append(code, label);
    const id = node("span", "bus-id");
    first.append(name, id);
    const predicted = node("td", "delay-cell");
    const stateCell = node("td");
    const badge = node("span", "risk-badge");
    stateCell.append(badge);
    const age = node("td", "age-cell");
    const action = node("td");
    const button = node("button", "row-open", "›");
    button.type = "button";
    button.dataset.vehicleId = key;
    action.append(button);
    row.append(first, predicted, stateCell, age, action);
    return { row, code, label, id, predicted, badge, age, button };
  }

  function renderTable(vehicles) {
    const body = $("vehicle-table-body");
    const keep = new Set();
    vehicles.forEach((vehicle, index) => {
      const key = idOf(vehicle.tr_id);
      keep.add(key);
      let record = tableRows.get(key);
      if (!record) { record = createTableRow(key); tableRows.set(key, record); }
      const risk = effectiveRisk(vehicle);
      record.row.classList.toggle("is-selected", key === selectedId);
      record.code.textContent = String(routeCode(vehicle));
      record.code.title = String(routeName(vehicle));
      record.label.textContent = vehicleName(vehicle);
      record.id.textContent = fullUI ? `ID ${vehicle.tr_id} · ${time(vehicle.event_time, true)} МСК` : "";
      record.code.hidden = !fullUI && snapshot.mode === "replay";
      record.predicted.textContent = delay(displayedDelay(vehicle), true);
      record.predicted.title = delay(displayedDelay(vehicle));
      record.predicted.style.color = vehicleColor(vehicle);
      record.badge.textContent = risk === "unknown" ? availabilityLabel(vehicle) : RISK[risk].short;
      record.badge.title = apiStale() ? "Связь с API потеряна; актуальность данных не подтверждена."
        : vehicle.prediction_availability?.message || RISK[risk].label;
      record.badge.className = `risk-badge risk-${risk}`;
      record.age.textContent = duration(vehicle.age_s, true);
      record.age.classList.toggle("is-stale", vehicle.status !== "fresh" || apiStale());
      record.button.setAttribute("aria-label", `Открыть: ${vehicleName(vehicle)}`);
      record.button.setAttribute("aria-pressed", String(key === selectedId));
      if (body.children[index] !== record.row) body.insertBefore(record.row, body.children[index] || null);
    });
    tableRows.forEach((record, key) => { if (!keep.has(key)) { record.row.remove(); tableRows.delete(key); } });
    $("table-empty").hidden = vehicles.length > 0;
    text("table-empty", list(snapshot?.vehicles).length ? "По выбранным фильтрам автобусы не найдены." : snapshot?.mode === "live" ? "Ждём входящие сообщения. Демонстрационные данные выключены." : "Транспорт пока не поступил в систему.");
    text("table-count", vehicles.length);
  }

  function renderIncidents() {
    const route = $("route-filter").value;
    const incidents = list(snapshot?.incidents).filter((incident) => route === "all" || idOf(incident.route_id) === route).slice().sort((a, b) => (timestamp(b.created_at) || 0) - (timestamp(a.created_at) || 0));
    text("timeline-count", incidents.length);
    $("timeline-empty").hidden = incidents.length > 0;
    const clock = timestamp(snapshot?.clock_time || snapshot?.server_time);
    const historical = (incident) => clock !== null && timestamp(incident.target_time) !== null && timestamp(incident.target_time) <= clock;
    const signature = JSON.stringify([incidents, incidents.map(historical)]);
    if (signature === incidentSignature) return;
    incidentSignature = signature;
    $("incident-timeline").replaceChildren(...incidents.slice(0, 50).map((incident) => {
      const risk = delayBand(incident.predicted_delay_s);
      const item = node("li", `timeline-item risk-${risk}${historical(incident) ? " is-historical" : ""}`);
      const heading = node("div", "timeline-title");
      const vehicle = list(snapshot?.vehicles).find((v) => idOf(v.tr_id) === idOf(incident.tr_id));
      const incidentRoute = list(snapshot?.routes).find((item) => idOf(item.route_id) === idOf(incident.route_id));
      const button = node("button", "", `${incidentRoute?.name || incident.route_id || "—"} · ${vehicleName(vehicle || { tr_id: incident.tr_id })}`);
      button.type = "button";
      button.dataset.vehicleId = idOf(incident.tr_id);
      button.title = "Открыть текущее состояние этого автобуса; сохранённый прогноз остаётся в журнале.";
      const at = node("time", "", time(incident.created_at, true));
      if (timestamp(incident.created_at) !== null) at.dateTime = incident.created_at;
      heading.append(button, at);
      const state = node("span", "incident-state", historical(incident) ? "История · плановое время цели прошло" : "Сохранённое предупреждение");
      const target = node("p", "incident-target", incident.target_name || "Название целевой остановки не передано");
      const section = node("p", "", `Участок: ${incident.section || "не указан"}`);
      const tags = node("div", "timeline-tags");
      const expected = timestamp(incident.estimated_arrival_at) !== null ? incident.estimated_arrival_at : expectedArrival(incident.target_time, incident.predicted_delay_s);
      tags.append(node("strong", "", `Прогноз ${delay(incident.predicted_delay_s)}`), node("span", "", `План ${time(incident.target_time, true)} МСК`), node("span", "", `Ожидалось ${expected ? `${time(expected, true)} МСК` : "не оценено"}`));
      if (fullUI && finite(incident.lead_time_s)) tags.append(node("span", "", `При выдаче до плана: ${duration(incident.lead_time_s)}`));
      if (fullUI && finite(incident.estimated_lead_time_s)) tags.append(node("span", "", `До ожидаемого прибытия: ${duration(incident.estimated_lead_time_s)} · оценка`));
      const probability = node("p", "incident-probability", `Вероятность >2 мин при выдаче: ${probabilityLabel(incident.probability_late, incident.probability_status)}`);
      if (incident.probability_note) probability.title = incident.probability_note;
      const explanation = incident.explanation;
      const cause = node("p", "incident-cause", explanation?.cause_status === "hypothesis" ? `Предполагаемая причина: ${explanation.possible_cause}` : explanation?.summary ? `Наблюдение: ${explanation.summary}` : "Причина не установлена");
      const description = node("p", "", incident.reason || "Основание сигнала не передано");
      item.append(heading, state, target, tags, probability, cause);
      if (fullUI) item.append(section, description);
      return item;
    }));
  }

  function renderHealth() {
    const stale = apiStale();
    const health = snapshot?.health || {};
    const metrics = snapshot?.metrics || {};
    const apiStatus = connected ? "ok" : snapshot ? "error" : "warning";
    ["api-dot", "side-api-dot"].forEach((id) => setStatus(id, apiStatus));
    text("api-status", connected ? "На связи" : snapshot ? "Нет связи" : "Подключение");
    text("side-api-label", connected ? "Система на связи" : snapshot ? "Связь с API потеряна" : "Подключение к системе");
    const mlStatus = stale ? null : health.ml;
    setStatus("ml-dot", mlStatus === "ok" ? "ok" : mlStatus === "unavailable" ? "error" : "warning");
    text("ml-status", stale ? "Нет актуального статуса" : mlStatus === "ok" ? "Доступен" : mlStatus === "unavailable" ? "Недоступен" : mlStatus === "starting" ? "Запускается" : "—");
    const sourceOff = snapshot?.mode === "demo" && !snapshot.demo?.source_enabled;
    setStatus("source-dot", stale ? "warning" : sourceOff ? "warning" : snapshot?.mode === "live" && !health.ndtp_connections ? "warning" : "ok");
    const sourceNames = { receiving: "Приём данных", awaiting: "Ожидание данных", idle: "Ожидание данных", disconnected: "Нет соединения", disabled: "Выключен", paused: "Пауза", stale: "Данные устарели" };
    text("source-status", stale ? "Нет актуального статуса" : sourceOff ? "Выключен" : sourceNames[health.source_status] || "Ожидание данных");
    text("inference-latency", !stale && finite(metrics.inference_p95_ms) ? `${metrics.inference_p95_ms.toFixed(1)} мс` : "—");
    text("pipeline-latency", !stale && finite(metrics.pipeline_p95_ms) ? `${metrics.pipeline_p95_ms.toFixed(1)} мс` : "—");
    text("connection-count", `${integer(health.ndtp_connections ?? 0)} NDTP-подключений`);
    text("pipeline-counts", snapshot ? `Принято ${integer(metrics.received ?? 0)} · дубли ${integer(metrics.duplicates ?? 0)} · невалидные ${integer(metrics.invalid ?? 0)} · ошибки NDTP ${integer(metrics.ndtp_errors ?? 0)}` : "Телеметрия: ожидаем сообщения");
    text("last-update", lastSuccess ? `API обновлён ${duration((Date.now() - lastSuccess) / 1000)} назад` : "Состояние ещё не получено");
    $("clock-dot").style.background = connected && playback().running !== false ? "var(--teal)" : "var(--slate)";
    text("clock", time(snapshot?.clock_time || snapshot?.server_time, true));
    $("clock").dateTime = snapshot?.clock_time || snapshot?.server_time || "";
    text("clock-label", snapshot?.mode === "generator" ? "МСК · генератор" : snapshot?.mode === "demo" ? "МСК · демо" : snapshot?.mode === "replay" ? "МСК · история" : "МСК");
    text("map-state-text", !snapshot ? "Ожидание" : stale ? "Данные не обновляются" : playback().finished ? "Данные закончились" : snapshot.mode !== "live" && !playback().running ? "Пауза" : sourceOff ? "Источник выключен" : "Обновление 1 с");
    $("map-state").querySelector(".status-dot").style.background = stale || sourceOff ? "var(--slate)" : playback().running === false ? "var(--amber)" : "var(--teal)";
    if (snapshot) {
      const isDemo = snapshot.mode === "demo";
      const isReplay = snapshot.mode === "replay";
      const isGenerator = snapshot.mode === "generator";
      document.querySelectorAll("[data-mode]").forEach((button) => {
        const active = button.dataset.mode === snapshot.mode;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });
      $("demo-controls").hidden = !(isDemo || isReplay || isGenerator);
      $("live-controls").hidden = isDemo || isReplay || isGenerator;
      $("source-button").hidden = isReplay || isGenerator;
      $("demo-controls").querySelector(".controls-label strong").textContent = isGenerator ? "Наш генератор" : isReplay ? "История CSV" : "Сценарий демо";
      text("play-label", playback().finished ? "Завершено" : playback().running ? "Пауза" : "Продолжить");
      text("play-symbol", playback().running ? "Ⅱ" : "▷");
      text("source-button", snapshot.demo?.source_enabled ? "Выключить источник" : "Включить источник");
      $("demo-speed").value = String(playback().speed || 1);
      const playbackStatus = (snapshot.replay?.finished || snapshot.generator?.finished) ? "Данные закончились" : playback().running ? `Время идёт · ${playback().speed || 1}×` : "На паузе";
      text("demo-status", isGenerator && snapshot.generator?.error ? `Ошибка источника: ${snapshot.generator.error}` : isReplay ? `${snapshot.replay.start.slice(0, 10)} · ${snapshot.replay.dataset_split} · ${playbackStatus}` : playbackStatus);
    }
    text("feed-label", !snapshot ? "Подключение" : snapshot.mode === "live" ? "Живой поток"
      : snapshot.mode === "replay" ? `Архивный поток · ${playback().finished ? "завершён" : playback().running ? "воспроизведение" : "пауза"}` : "Демонстрационный поток");
    setButtonBusy();
  }

  function render() {
    const ctx = snapshot?.context;
    text("context-summary", ctx ? `План: ${{synthetic_generator:"наш генератор",builtin_demo:"быстрое демо",live_context:"загруженный контекст"}[ctx.source] || ctx.source} · ${ctx.vehicles} ТС · ${ctx.planned_visits} посещений · источник отклонения: ${ctx.arrival_mode === "gps" ? "GPS" : ctx.arrival_mode === "csv_snapshot" ? "points.csv" : "внешние события"}` : "План пока не загружен");
    renderRouteOptions();
    const vehicles = filteredVehicles();
    if (!vehicles.some((vehicle) => idOf(vehicle.tr_id) === selectedId)) selectedId = vehicles.length ? idOf(vehicles[0].tr_id) : null;
    renderSummary();
    renderAttention();
    renderMap(vehicles);
    renderDetail();
    renderTable(vehicles);
    renderIncidents();
    renderHealth();
    renderReplay();
  }

  function renderReplay() {
    const active = snapshot?.mode === "replay";
    $("replay-settings").hidden = !active;
    const ctx = snapshot?.context;
    const source = active ? snapshot.replay?.deviation_source || "csv_snapshot" : ctx?.arrival_mode;
    text("input-source-badge", `Отклонение: ${source === "gps" ? "GPS + план" : source === "csv_snapshot" ? "готовое points.csv" : snapshot?.mode === "demo" ? "демо-прибытия" : source ? "внешние события" : "ожидание"}`);
    if (!active) return;
    const replay = snapshot.replay;
    const split = replay.dataset_split || "validate";
    text("replay-active-source", `${split} · ${source === "gps" ? "считаем сами по GPS" : "готовое отклонение CSV"}`);
    const signature = JSON.stringify([split, replay.start, replay.end, replay.timezone, replay.selection_mode, replay.tr_ids, source, replay.warmup_minutes]);
    if (signature !== replayConfigSignature) {
      replayConfigSignature = signature;
      $("replay-split").value = split;
      $("replay-source").value = source;
      $("replay-start").value = new Date(replay.start).toISOString().slice(0,16);
      $("replay-warmup").value = String(replay.warmup_minutes ?? 5);
      $("replay-selection").value = replay.selection_mode || "manual";
      $("replay-vehicles").value = replay.selection_mode === "all_available" ? "" : list(replay.tr_ids).join(", ");
      $("replay-timezone").value = replay.timezone;
      renderReplayHelp();
    }
  }

  function renderReplayHelp() {
    const train = $("replay-split").value === "train";
    $("replay-source").querySelector('option[value="csv_snapshot"]').disabled = train;
    if (train) $("replay-source").value = "gps";
    text("replay-source-help", $("replay-source").value === "gps"
      ? "Отклонение определяется по GPS и расписанию. История до начала нужна для распознавания предыдущих остановок."
      : "Отклонение берётся из points.csv; координаты — из телеметрии.");
    setButtonBusy();
  }

  function startReplay(event) {
    event.preventDefault();
    const tr_ids = $("replay-selection").value === "manual" ? $("replay-vehicles").value.trim().split(/[,;\s]+/).map(Number) : null;
    if (tr_ids && (!tr_ids.length || tr_ids.length > 128 || tr_ids.some(id => !Number.isSafeInteger(id) || id <= 0) || new Set(tr_ids).size !== tr_ids.length)) {
      toast("Укажите от 1 до 128 разных положительных ID автобусов через запятую."); return;
    }
    const start = new Date($("replay-start").value + "Z");
    if (!Number.isFinite(start.getTime())) { toast("Укажите корректное начало среза в UTC."); return; }
    command("/api/v1/replay/load", { start: start.toISOString(), duration_minutes: null,
      warmup_minutes: Number($("replay-warmup").value), timezone: $("replay-timezone").value, tr_ids,
      dataset_split: $("replay-split").value, deviation_source: $("replay-source").value, speed: Number($("demo-speed").value || 10), paused: true },
      "Архив загружен до конца данных. Нажмите Продолжить.");
  }

  function selectVehicle(id) {
    const vehicle = list(snapshot?.vehicles).find((item) => idOf(item.tr_id) === idOf(id));
    if (!vehicle) { toast("Автобус уже отсутствует в текущем состоянии."); return; }
    // Selecting an incident should also work when a table filter hides the vehicle.
    if (!filteredVehicles().some((item) => idOf(item.tr_id) === idOf(id))) {
      $("route-filter").value = "all";
      $("risk-filter").value = "all";
      $("vehicle-search").value = "";
    }
    selectedId = idOf(id);
    render();
  }

  function toast(message) {
    text("action-status", message);
    $("action-status").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("action-status").hidden = true; }, 4200);
  }

  async function request(url, options = {}, timeoutMs = 5000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { cache: "no-store", ...options, signal: controller.signal });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(typeof body?.detail === "string" ? body.detail : `HTTP ${response.status}`);
      }
      if (response.status === 204) return null;
      return await response.json();
    } finally { clearTimeout(timer); }
  }

  async function poll() {
    if (polling) { refreshRequested = true; return; }
    clearTimeout(pollTimer);
    polling = true;
    try {
      const next = await request("/api/v1/state");
      if (!next || !Array.isArray(next.vehicles) || !Array.isArray(next.routes) || !["demo", "live", "replay", "generator"].includes(next.mode)) throw new Error("Неверный формат состояния");
      snapshot = next;
      connected = true;
      lastSuccess = Date.now();
      $("error-banner").hidden = true;
    } catch (error) {
      connected = false;
      $("error-banner").hidden = false;
      text("error-text", snapshot ? "Нет связи с API. Показаны последние сохранённые данные; актуальность не подтверждена." : "Не удалось подключиться к API. Проверьте, что сервис запущен.");
    } finally {
      polling = false;
      render();
      pollTimer = setTimeout(poll, refreshRequested ? 0 : 1000);
      refreshRequested = false;
    }
  }

  async function command(url, payload, successText) {
    if (busy) return;
    busy = true;
    setButtonBusy();
    try {
      await request(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }, url === "/api/v1/replay/load" ? 30000 : 5000);
      if (payload.action === "reset" || payload.mode) {
        selectedId = null;
        incidentSignature = "";
      }
      await poll();
      toast(successText);
    } catch (error) {
      toast(`Команда не выполнена. ${error.name === "AbortError" ? "Сервис не ответил вовремя." : error.message}`);
    } finally { busy = false; setButtonBusy(); }
  }

  ["route-filter", "risk-filter"].forEach((id) => $(id).addEventListener("change", render));
  $("vehicle-search").addEventListener("input", render);
  ["vehicle-table-body", "incident-timeline", "attention-list"].forEach((id) => $(id).addEventListener("click", (event) => {
    const target = event.target.closest("[data-vehicle-id]");
    if (target) selectVehicle(target.dataset.vehicleId);
  }));
  document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.mode === "replay") {
      if (snapshot?.mode !== "replay") command("/api/v1/replay/load", {}, "Архив загружен. Нажмите Продолжить.");
    } else if (button.dataset.mode !== snapshot?.mode) command("/api/v1/mode", { mode: button.dataset.mode }, button.dataset.mode === "live" ? "Включён приём живых данных" : "Включена демонстрация");
  }));
  const playbackControl = () => snapshot?.mode === "generator" ? "/api/v1/generator/control" : snapshot?.mode === "replay" ? "/api/v1/replay/control" : "/api/v1/demo/control";
  $("play-button").addEventListener("click", () => command(playbackControl(), { action: playback().running ? "pause" : "resume" }, playback().running ? "Воспроизведение на паузе" : "Воспроизведение продолжено"));
  $("source-button").addEventListener("click", () => command("/api/v1/demo/control", { action: snapshot?.demo?.source_enabled ? "source_off" : "source_on" }, snapshot?.demo?.source_enabled ? "Источник выключен. Возраст данных будет расти." : "Источник включён"));
  $("reset-button").addEventListener("click", () => command(playbackControl(), { action: "reset" }, "Воспроизведение сброшено"));
  $("replay-form").addEventListener("submit", startReplay);
  $("replay-split").addEventListener("change", () => { if ($("replay-split").value === "train") $("replay-warmup").value = "30"; renderReplayHelp(); });
  $("replay-source").addEventListener("change", () => { $("replay-warmup").value = $("replay-source").value === "gps" ? "30" : "5"; renderReplayHelp(); });
  $("replay-selection").addEventListener("change", setButtonBusy);
  $("demo-speed").addEventListener("change", () => command(playbackControl(), { action: "speed", speed: Number($("demo-speed").value) }, "Скорость воспроизведения изменена"));
  $("retry-button").addEventListener("click", poll);
  $("map-fit").addEventListener("click", () => fitMap());
  $("map-focus").addEventListener("click", focusVehicle);
  $("map-follow").addEventListener("click", () => { setFollow(!followSelected); if (followSelected) focusVehicle(); });
  $("tile-toggle").addEventListener("click", () => {
    if (!map || !tileLayer) return;
    const enabled = !map.hasLayer(tileLayer);
    if (enabled) tileLayer.addTo(map); else tileLayer.remove();
    $("tile-toggle").setAttribute("aria-pressed", String(enabled));
    text("tile-toggle", enabled ? "Скрыть подложку" : "Показать подложку");
    if (!enabled) { text("tile-status", "Подложка выключена · GPS, остановки и управление картой работают без интернета"); $("tile-status").classList.remove("is-warning"); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  window.addEventListener("focus", poll);
  window.addEventListener("pageshow", poll);
  window.addEventListener("hashchange", syncNavigation);
  window.addEventListener("pagehide", () => clearTimeout(pollTimer));
  syncNavigation();
  render();
  poll();
})();
