"""HTTP-инференс frozen Transformer + HGBR с явным persistence fallback.

Запуск из каталога ``filipp``: ``uvicorn ml_service.app:app --port 8001``.
MODEL_PATH активирует проверенный team artifact; без него доступен baseline.
"""

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from common.contracts import Prediction, PredictionRequest, baseline
from ml_service.learned import fallback, runtime


MODEL_VERSION = "persistence-v1"
MAX_BATCH_SIZE = 128


@asynccontextmanager
async def lifespan(application: FastAPI):
    runtime()  # загрузка один раз при старте, не на первом диспетчерском запросе
    yield


app = FastAPI(
    title="Предиктор задержки: ML-модуль",
    version="2.0.0",
    lifespan=lifespan,
    description=(
        "Frozen Swiss Transformer prior + Moscow HistGradientBoosting. "
        "При отсутствии модели/плана доступен явно обозначенный persistence fallback. "
        "Контракт, происхождение модели и ограничения: /model."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Вернуть 422 даже при NaN/Infinity во входе, не допуская ошибки сериализации."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {key: error[key] for key in ("loc", "msg", "type")}
                for error in exc.errors()
            ]
        },
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Указать фактически загруженную модель; ошибка загрузки не скрывается."""
    model, error = runtime()
    if model is not None:
        return {"status": "ok", "model_version": model.model_version, "trained": True}
    if error:
        return {"status": "degraded", "model_version": MODEL_VERSION, "trained": False,
                "error": error}
    return {"status": "ok", "model_version": MODEL_VERSION, "trained": False}


@app.get("/model")
def model_card() -> dict[str, Any]:
    """Описать контракт, ограничения и правила интерпретации результата baseline."""
    card = {
        "model_version": MODEL_VERSION,
        "trained": False,
        "method": "persistence",
        "target": (
            "Фактическое прибытие минус плановое на выданном плановом прибытии "
            "target; положительное значение — опоздание, отрицательное — опережение."
        ),
        "units": "seconds",
        "prediction_horizon_seconds": {"lower_exclusive": 600, "upper_inclusive": 900},
        "input": {
            "schema": "PredictionRequest",
            "current_delay_s": "cur_dev_s, известное на issued_at; null при отсутствии",
            "target": "Конкретное плановое прибытие, выбранное вызывающей стороной",
            "timestamps": "Обязателен явный UTC offset",
            "extra_fields": "forbidden",
        },
        "output": {
            "schema": "Prediction",
            "predicted_delay_s": "Равно current_delay_s; null, если подсказка неизвестна",
            "probability_late": None,
            "risk": (
                "Эвристика величины задержки: green ≤60 с, amber (60,120] с, red >120 с; "
                "unknown при неизвестной подсказке, отсутствии или возрасте GPS >60 с."
            ),
        },
        "limits": {"max_batch_size": MAX_BATCH_SIZE, "stale_after_s": 60},
        "limitations": [
            "Модель не обучалась; сервис не заявляет измеренное качество ML.",
            "Численный прогноз использует только current_delay_s; остальные признаки его не меняют.",
            "Валидность и свежесть GPS передаются вызывающей стороной; сервис не читает телеметрию.",
            "При устаревшем GPS подсказка сохраняется в прогнозе, но уровень риска становится unknown.",
            "Неизвестная подсказка не заменяется нулём: численный прогноз недоступен.",
            "Цвет риска не является вероятностью и не описывает риск раннего прибытия.",
            "Горизонт относится к плановому прибытию; раннее обнаружение начала сбоя не доказано.",
            "Причины в ответе объясняют правило baseline, а не устанавливают причину задержки.",
        ],
    }
    model, error = runtime()
    if error:
        card["load_error"] = error
    if model is not None:
        from ml_service.frozen_plan.plan import PLAN_FEATURES
        training = model.manifest["training"] if model.manifest else {"official_rows": 4434, "new_targets": 0}
        card.update({
            "model_version": model.model_version,
            "trained": True,
            "method": "learned",
            "estimator": "Swiss pretrained causal Transformer prior + Moscow HistGradientBoostingRegressor",
            "features": list(PLAN_FEATURES) + ["neural_prior"],
            "provenance": {
                "source": ("versioned full-HGBR-refit bundle" if model.manifest
                           else "alexchist/artifacts/experiments/swiss_sequence_hybrid"),
                "sha256": model.sha256,
                "encoder_sha256": model.encoder_sha256,
                "training_rows": training["official_rows"] + training.get("new_targets", 0),
                "update_strategy": training.get("update_strategy", "initial_full_fit"),
                "runtime_versions": model.versions,
                "feature_builder": "Точная копия alexchist/task1_ml/plan.py, hashes в frozen_plan/provenance.json",
            },
        })
        card["input"]["plan_context"] = (
            "Полный план данного tr_id: version, IANA timezone, complete=true, "
            "stops с известным boolean manual_fill. Нужен весь план, не только горизонт."
        )
        card["input"]["delay_history"] = (
            "До 11 строго прошлых cur_dev_s того же ТС за 90 минут; current_delay_s становится 12-й точкой. "
            "Пустая история разрешена как cold start."
        )
        card["output"]["predicted_delay_s"] = "Регрессия знакового отклонения; null при неизвестном current_delay_s"
        card['probability'] = model.probability.metadata() if model.probability else None
        card['probability_error'] = model.probability_error
        card['output']['probability_late'] = ('P(target_delay_s >120); current_delay_source=csv_snapshot с выданной подсказкой, '
            'либо gps_estimate с telemetry_domain=historical_real и точной версией/SHA детектора из принятого GPS-профиля; '
            'для historical_mixed при том же детекторе — приближённый перенос со статусом transferred, '
            'его качество на train не проверено. Обязательны свежие входы. '
            'Без принятого GPS-профиля доступен только csv_snapshot') if model.probability else None
        card['output']['probability_status'] = {
            'validated': 'Оценка в области вторичной проверки из probability metadata; не независимый новый день.',
            'transferred': 'Приближённый перенос того же mapping на смешанный train; качество здесь не проверено.',
            'unavailable': 'Совместимого калибратора, поддержанного источника или свежих входов нет.',
        }
        card["limitations"] = [
            "Калибратор вероятности привязан к SHA HGBR и Transformer encoder; старые коэффициенты не переносятся. Цвет по порогам секунд — отдельное правило.",
            "Vehicle-OOF калибровка исключает оцениваемые автобусы из fit; история использует только прошлые входы. Проверка на выданном test вторичная, не новый день и не временной rolling holdout.",
            "Для калибратора операционное arrival, external_hint, двери, live, синтетика и неизвестный источник требуют отдельной проверки.",
            "Смешанный архив train допускает только transferred с проверенным GPS-детектором; этот статус не подтверждает качество переноса на синтетику.",
            "Transformer обучен на Swiss и остаётся замороженным; обновляется только HGBR полным refit на разрешённых московских данных.",
            "Выданный cur_dev_s не всегда равен отклонению последнего фактически пройденного посещения; этот аудит не разрешает использовать будущие факты в live.",
            "При неполном плане или неизвестном manual_fill: явный persistence fallback.",
            "cur_dev_s должен соответствовать определению датасета; неточное GPS-отклонение переносит ошибку во вход модели.",
            "Новый город/день и искусственный генератор не являются проверенным распределением модели.",
            "Shapley-вклады и history counterfactual объясняют поведение модели, но не доказывают физическую причину задержки; traffic.csv не используется.",
            "Горизонт (10,15] относится к плановому прибытию, не гарантирует упреждение начала инцидента.",
            "Будущие фактические прибытия, labels и target_class не допускаются в запрос.",
        ]
    return card


@app.post("/predict", response_model=Prediction)
def predict(request: PredictionRequest) -> Prediction:
    """Исполнить frozen модель либо явно обозначить причину резервного прогноза."""
    model, error = runtime()
    if model is not None:
        return model.predict(request)
    if error:
        return fallback(request, "Обученная модель не загрузилась")
    return baseline(request)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(
    requests: Annotated[list[PredictionRequest], Body(max_length=MAX_BATCH_SIZE)],
) -> list[Prediction]:
    """Обработать не более 128 запросов, сохранив их порядок; ошибка отклоняет весь пакет."""
    model, error = runtime()
    if model is not None:
        return model.predict_batch(requests)
    if error:
        return [fallback(request, "Обученная модель не загрузилась") for request in requests]
    return [baseline(request) for request in requests]
