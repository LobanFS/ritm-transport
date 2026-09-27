# Входные данные и эксплуатация

## Входные данные

Для прогноза нужны три источника: план движения, соответствие автобуса терминалу и телеметрия. NDTP не содержит готового расписания. Координаты остановок берутся из поля `geom` (`POINT (lon lat)`) планового CSV. Геометрия между остановками не считается подтверждённой дорожной трассой.

| Источник | Приём | Содержание |
|---|---|---|
| План и справочники | `POST /api/v1/live/context` | `vehicles`: `tr_id`, `unit_id`, подпись и маршрут; `schedule`: идентификатор посещения, остановка, координаты, плановое время |
| NDTP | TCP `9201` | Handshake NDTP 6.2, Nav00: время, координаты, скорость; поддерживается IRMA04 |
| Декодированная телеметрия | `POST /api/v1/telemetry` | Массив до 500 сообщений; фактическое время получения ставит сервер |
| Подтверждённое прибытие | `POST /api/v1/arrivals` | Автобус, ссылка на плановое посещение и время прибытия |
| Исторический архив | `POST /api/v1/replay/load` | Выбор train, validate или загруженного custom, начала и источника отклонения |
| Собственный архив | `POST /api/v1/replay/import` | CSV-текст телеметрии и плана; файл текущих отклонений необязателен |

Точные схемы и примеры полей доступны в [Backend Swagger](http://127.0.0.1:8000/docs). Новый live-контекст заменяет расписание, историю и предупреждения текущей сессии. Даты плана должны соответствовать датам потока; январское расписание не сдвигается автоматически на сегодняшний день.

`arrival_mode: external` ожидает подтверждённые прибытия через API. `arrival_mode: gps` включает детектор посещений по последовательности GPS и положению остановок. Он отбрасывает скачки и неоднозначные совпадения, поэтому не каждое посещение получает оценку. Доступные сигналы дверей могут дополнять GPS; официальный эмулятор не моделирует их автоматическое открытие и закрытие.

Текущее отклонение — время обнаруженного/подтверждённого прибытия минус его плановое время. Оно не растёт каждую секунду. GPS-оценки и CSV-подсказки действуют не более пяти минут; подтверждённый факт — до следующего прибытия. Будущие факты из `schedule.csv` исключены из входов.

При каждом цикле backend выбирает первую остановку в `(T+10, T+15]` минут по плану. В ML передаются актуальное отклонение, доступная история и план. ML возвращает прогноз секунд, вероятность в допустимой области и факторы расчёта. Последний полный обмен доступен в `GET /api/v1/vehicles/{tr_id}/forecast-trace`.

Веса Transformer и HGBR загружаются в отдельном сервисе. Если ML недоступен, резервная оценка явно обозначается; отсутствие входных данных не превращается в нулевую задержку. После обрыва NDTP отправитель должен повторить handshake. Телеметрия повторного пакета не дублирует историю.

## Собственный исторический архив

`POST /api/v1/replay/import` принимает JSON: обязательные строки `traffic_csv` и `schedule_csv` содержат целиком CSV-текст, необязательный `points_csv` — строку или `null`, `timezone` — часовой пояс IANA, по умолчанию `UTC`. Общий размер JSON — не более 80 МиБ; пределы: 500 000 строк телеметрии и подсказок суммарно, 20 000 плановых посещений, 128 ТС. Это загрузка файлов через API; `--data-dir` монтирует уже существующий каталог на сервере.

CSV — UTF-8, разделитель запятая; BOM допускается. Обязательные столбцы:

| Файл | Столбцы |
|---|---|
| Телеметрия | `tr_id,unit_id,event_time,receive_time,packet_id,lat,lon,speed,heading,location_valid` |
| План | `tr_id,tt_action_item_id,time_begin,geom` |
| Подсказки | `sample_id,tr_id,T,target_stop_id,target_time_begin,cur_dev_s` |

`location_valid` — `true`/`false`; координаты, скорость и курс могут быть пустыми. `geom` — `POINT (lon lat)`; `building_address` и `manual_fill` в плане необязательны. Фактическое время `time_fact_begin` не используется как признак. Схема запроса и ошибки — в [Swagger](http://127.0.0.1:8000/docs#/).

После успешной проверки создаётся набор `custom`: GPS-режим, пауза, скорость 10×, начало — минимальное время доступности сообщения `max(event_time, receive_time)`. `points.csv` используется только при последующем выборе `csv_snapshot`. Вероятность, проверенная для validate, автоматически на собственные данные не переносится.

Файлы хранятся во временном каталоге сервера, не перезаписывают встроенные наборы и недоступны после перезапуска backend. `GET /api/v1/replay/custom` возвращает `available`, `has_points`, `timezone` и `expires_on_restart`.

## Подготовить собственный live-план

JSON можно отправить напрямую по схеме Swagger. Для подготовки из CSV используйте `tools/load_context.py` внутри уже запущенного контейнера. Пример `vehicles.json`:

```json
{"vehicles": [{"tr_id": 101, "unit_id": 1166336, "label": "Автобус 101", "route_id": "route-1"}]}
```

```bash
docker compose cp /path/schedule_plan.csv backend:/tmp/schedule_plan.csv
docker compose cp /path/vehicles.json backend:/tmp/vehicles.json
docker compose exec backend python tools/load_context.py --schedule-plan /tmp/schedule_plan.csv \
  --vehicles /tmp/vehicles.json --timezone UTC --complete-plan --out /tmp/context.json
docker compose cp backend:/tmp/context.json ./context.json
# Проверьте context.json и при необходимости измените arrival_mode, затем примените:
curl -H 'Content-Type: application/json' --data-binary @context.json \
  http://127.0.0.1:8000/api/v1/live/context
```

Обязательные столбцы CSV: `tr_id,tt_action_item_id,time_begin,geom,building_address`. Допускаются `order_date,manual_fill` и полностью пустой `time_fact_begin`. Фактические времена в файле отклоняются. Флаг `--complete-plan` означает, что передан полный план выбранных автобусов. Для GPS-детектора измените `arrival_mode` в полученном JSON на `gps` перед применением.

## Накопление и обновление модели

Эта возможность выключена по умолчанию. Для SQLite-журнала в отдельном Docker volume:

```bash
docker compose -f compose.yaml -f compose.model.yaml -f compose.learning.yaml up --build -d
```

Сохраняются только live-прогнозы обученной модели и позднее подтверждённые прибытия `source=arrival`. GPS-оценки, replay и синтетика не становятся обучающими метками. На одно посещение сохраняется один запрос. Журнал переживает смену диспетчерского контекста; удаление volume удалит накопленные данные.

Для периодического обучения дополнительно нужны исходный датасет и подготовленный командой `sequences.npz`; файл последовательностей в публичную поставку не входит. Каталог результатов должен быть доступен для записи:

```bash
mkdir -p artifacts/retraining
chmod g+rwx artifacts/retraining
export RETRAIN_GID="$(id -g)"
export RETRAIN_DATA_DIR=/path/to/dataset
export RETRAIN_SEQUENCES_DIR=/path/to/sequences-directory
docker compose -f compose.yaml -f compose.model.yaml -f compose.learning.yaml \
  -f compose.retrain.yaml up --build -d
```

По умолчанию проверка запускается сразу, затем раз в неделю. При менее чем 200 новых подтверждённых посещениях обучение пропускается. HGBR обучается заново на исходных и новых данных; Transformer остаётся неизменным. Проверка использует более поздние запросы, а обучение — только метки, известные до начала проверки. Кандидат сохраняется при прохождении порога MAE; запущенная модель автоматически не заменяется.

Результаты: `artifacts/retraining/runs/`, `candidates/` и `scheduler_state.json`. Логи: `docker compose -f compose.yaml -f compose.model.yaml -f compose.learning.yaml -f compose.retrain.yaml logs -f model-trainer`.

После проверки кандидата оператор может явно выбрать его:

```bash
MODEL_DIR=./artifacts/retraining/candidates/VERIFIED_VERSION \
MODEL_MANIFEST_PATH=/models/manifest.json PROBABILITY_PATH= \
docker compose -f compose.yaml -f compose.model.yaml up --build -d
```

Старый калибратор вероятности к новой регрессионной модели не применяется. Его нужно подготовить и проверить отдельно.

## Несколько ML-реплик

```bash
docker compose -f compose.yaml -f compose.model.yaml -f compose.scale.yaml \
  up --build -d --scale ml=3
```

Gateway распределяет запросы между репликами с общими read-only весами. Backend остаётся одним экземпляром: оперативная история хранится в памяти. Число реплик выбирают по доступной памяти и измеренной нагрузке. После изменений состава сервисов используйте тот же набор `-f` при обслуживании стека.

Все порты по умолчанию доступны только локально. При размещении для внешних пользователей нужны настройка доступа и TLS. Подложка использует OpenStreetMap; внешние пробки и погода к инференсу не подключены. Синтетический генератор для интеграционных сценариев включается отдельно: `docker compose --profile generator up --build -d generator`; его API доступно на порту 8002.
