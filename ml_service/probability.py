"""Оценка P(delay >120с), отдельно от прогноза секунд и только с происхождением.

Runtime читает небольшой JSON коэффициентов, не pickle; sklearn ему не нужен.
Калибратор обучен на vehicle-OOF предиктах train и проверен на прежнем test.
Основной профиль — выданная CSV-подсказка cur_dev_s. Дополнительный GPS-допуск
загружается только из принятого source-profile с точной версией/SHA детектора
и ограничен реальной историей. На смешанном train тот же mapping допускается
только с явным статусом transferred: его качество там не подтверждено.
Двери, генератор и новый live-поток требуют отдельной проверки.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class ProbabilityArtifact(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    schema_version: Literal['late-probability-logistic-v1', 'late-probability-logistic-v2']
    regression_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    encoder_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    training_protocol: Literal['vehicle_oof'] = 'vehicle_oof'
    history_protocol: Literal['provided_current_delay_strict_past_90m_12steps'] | None = None
    contract_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    sources_sha256: dict[str,str]
    event: Literal['target_delay_s > 120']
    feature_offset_s: Literal[120.0]
    feature_scale_s: Literal[120.0]
    coefficient: FiniteFloat
    intercept: FiniteFloat
    train_rows: Literal[4434]
    test_rows: Literal[353]
    train_constant: float = Field(gt=0,lt=1)
    test_brier: float = Field(ge=0,le=1)
    test_constant_brier: float = Field(ge=0,le=1)
    validation_status: Literal['accepted_secondary_test']
    scope: Literal['real_data_with_provided_current_deviation']

    @model_validator(mode='after')
    def valid_evidence(self):
        if self.schema_version == 'late-probability-logistic-v2' and (not self.encoder_sha256 or not self.history_protocol):
            raise ValueError('Гибридный калибратор требует SHA encoder и контракт причинной истории')
        if self.test_brier >= self.test_constant_brier:
            raise ValueError('Калибровка не прошла зафиксированный критерий Brier < train constant')
        if not self.sources_sha256 or any(len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
                                          for digest in self.sources_sha256.values()):
            raise ValueError('Неверные SHA-256 источников')
        return self


class GPSProbabilityScope(BaseModel):
    """Допуск неизменённого mapping по измеренной GPS-цепочке, не новые веса."""
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    schema_version: Literal['gps-probability-scope-v1']
    validation_status: Literal['accepted_secondary_diagnostic']
    scope: Literal['historical_real_gps_estimate']
    regression_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    probability_artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    detector_version: str = Field(min_length=1)
    detector_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contract_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    report_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    report_path: str
    evaluated_rows: int = Field(ge=50)
    total_rows: int = Field(ge=50)
    positives: int = Field(ge=10)
    vehicles: int = Field(ge=5)
    brier: float = Field(ge=0, le=1)
    constant_brier: float = Field(ge=0, le=1)
    log_loss: float = Field(ge=0)
    constant_log_loss: float = Field(ge=0)
    cluster_improvement_p05: float = Field(gt=0)
    expected_calibration_error: float = Field(ge=0, le=.15)

    @model_validator(mode='after')
    def evidence_passes(self):
        if self.total_rows < self.evaluated_rows or self.evaluated_rows-self.positives < 10:
            raise ValueError('Недостаточно оценимых GPS-меток обоих классов')
        if self.brier >= self.constant_brier or self.log_loss >= self.constant_log_loss:
            raise ValueError('GPS-профиль не прошёл Brier/logloss gate')
        return self


def finite(value) -> float | None:
    if value is None or isinstance(value,bool):
        return None
    try:
        result=float(value)
    except (TypeError,ValueError,OverflowError):
        return None
    return result if math.isfinite(result) else None


class ProbabilityModel:
    def __init__(self,artifact: ProbabilityArtifact, *, artifact_sha256: str):
        self.artifact=artifact
        self.artifact_sha256=artifact_sha256
        self.gps_scope: GPSProbabilityScope | None = None
        self.gps_scope_sha256: str | None = None
        self.gps_scope_error: str | None = None

    @classmethod
    def load(cls,path: Path | str,expected_model_sha256: str, *, gps_scope_path: Path | str | None=None,
             expected_encoder_sha256: str | None=None) -> 'ProbabilityModel':
        payload=Path(path).read_bytes()
        if len(payload)>65536:
            raise ValueError('Слишком большой файл коэффициентов вероятности')
        artifact=ProbabilityArtifact.model_validate(json.loads(payload))
        if artifact.regression_model_sha256 != expected_model_sha256:
            raise ValueError('Калибратор относится к другому регрессионному артефакту')
        if expected_encoder_sha256 is not None and artifact.encoder_sha256 != expected_encoder_sha256:
            raise ValueError('Калибратор не подтверждает текущий Transformer encoder')
        model = cls(artifact,artifact_sha256=hashlib.sha256(payload).hexdigest())
        scope_path = Path(gps_scope_path) if gps_scope_path is not None else Path(__file__).with_name('probability_gps_scope.json')
        if scope_path.exists():
            try:
                scope_payload = scope_path.read_bytes()
                if len(scope_payload) > 65536:
                    raise ValueError('Слишком большой GPS-профиль вероятности')
                scope = GPSProbabilityScope.model_validate_json(scope_payload)
                if (scope.regression_model_sha256 != expected_model_sha256
                    or scope.probability_artifact_sha256 != model.artifact_sha256):
                    raise ValueError('GPS-профиль относится к другим весам/коэффициентам')
                model.gps_scope = scope
                model.gps_scope_sha256 = hashlib.sha256(scope_payload).hexdigest()
            except (OSError, ValueError, TypeError) as error:
                # Ошибка GPS-профиля не отключает подтверждённую CSV-калибровку.
                model.gps_scope_error = f'{type(error).__name__}: {error}'
        return model

    def source_allowed(self, source, *, telemetry_domain='unknown', detector_version=None, detector_sha256=None):
        if source == 'csv_snapshot':
            return True
        scope = self.gps_scope
        return bool(source == 'gps_estimate' and telemetry_domain == 'historical_real'
                    and scope is not None and detector_version == scope.detector_version
                    and detector_sha256 == scope.detector_sha256)

    def source_status(self, source, *, telemetry_domain='unknown', detector_version=None, detector_sha256=None):
        """Область проверки и явно обозначенный перенос — разные статусы.

        Перенос не расширяет source_allowed и не обходит привязку коэффициентов,
        GPS-профиля или детектора. Свежесть проверяется отдельно в predict.
        """
        if self.source_allowed(source, telemetry_domain=telemetry_domain,
                               detector_version=detector_version, detector_sha256=detector_sha256):
            return 'validated'
        scope = self.gps_scope
        if (source == 'gps_estimate' and telemetry_domain == 'historical_mixed'
                and scope is not None and detector_version == scope.detector_version
                and detector_sha256 == scope.detector_sha256):
            return 'transferred'
        return 'unavailable'

    def predict(self,predicted_delay_s, *, current_delay_s,telemetry_age_s) -> float | None:
        value,current,age=map(finite,(predicted_delay_s,current_delay_s,telemetry_age_s))
        if value is None or current is None or age is None or age<0 or age>60:
            return None
        a=self.artifact
        z=a.coefficient*((value-a.feature_offset_s)/a.feature_scale_s)+a.intercept
        # Вещественно конечный вход может переполнить произведение. Формула
        # насыщается к 0/1 без OverflowError и без чтения дополнительных данных.
        if math.isnan(z):
            return None
        return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))

    def unavailable_source_note(self, source: str | None, *, telemetry_domain='unknown') -> str:
        """Причина отсутствия p зависит от источника; это не причина задержки."""
        if source == 'gps_estimate':
            if self.gps_scope is not None:
                if telemetry_domain == 'synthetic':
                    return 'Вероятность проверена на реальной истории; перенос на синтетический генератор не подтверждён.'
                if telemetry_domain == 'live_unverified':
                    return 'Вероятность проверена на историческом GPS-потоке; для нового live-потока качество ещё не подтверждено.'
                return 'Для вероятности нужны подтверждённое происхождение GPS и точная проверенная версия детектора.'
            return ('Вероятность для GPS-оценки пока не подтверждена: ошибки распознавания '
                    'остановки меняют вход модели. Прогноз секунд доступен отдельно.')
        if source == 'door_estimate':
            return ('Для оценки по GPS и дверям нет размеченной реальной истории, '
                    'чтобы проверить вероятность. Прогноз секунд доступен отдельно.')
        return 'Вероятность проверена для выданных CSV-подсказок; для этого источника требуется отдельная проверка'

    def metadata(self) -> dict:
        a=self.artifact
        return dict(version=a.schema_version,event=a.event,artifact_sha256=self.artifact_sha256,
            regression_model_sha256=a.regression_model_sha256,
            encoder_sha256=a.encoder_sha256,training_protocol=a.training_protocol,history_protocol=a.history_protocol,
            train_rows=a.train_rows,test_rows=a.test_rows,test_brier=a.test_brier,
            test_constant_brier=a.test_constant_brier,scope=a.scope,
            allowed_current_delay_sources=['csv_snapshot']+(['gps_estimate'] if self.gps_scope else []),
            source_validation={
                'csv_snapshot': 'accepted_secondary_test',
                'gps_estimate': 'accepted_secondary_diagnostic_historical_only' if self.gps_scope else 'transfer_not_confirmed',
                'door_estimate': 'no_labeled_real_history',
            },
            transferred_scope=('historical_mixed_gps_estimate' if self.gps_scope else None),
            transfer_interpretation='Неизменный mapping на смешанном train; качество переноса не проверено, это не расширение валидированной области.',
            gps_scope=self.gps_scope.model_dump() if self.gps_scope else None,
            gps_scope_sha256=self.gps_scope_sha256,
            gps_scope_error=self.gps_scope_error,
            limitations=['Вторичная проверка на прежнем test, не новый независимый день.',
                         'Калибратор обучен на прогнозах исключённых из fit автобусов (vehicle-OOF); это не временной rolling holdout.',
                         'GPS-профиль, если подключён, ограничен реальной историей и точной проверенной версией детектора; новый live-поток требует проверки.',
                         'На смешанном train при том же детекторе возможна только приближённая оценка со статусом transferred; Brier/logloss для этого переноса не измерены.',
                         'Операционные прибытия, внешние hints, двери и синтетические источники не входят в проверенную область.',
                         'Аудит обнаружил различие выданного cur_dev_s и отклонения последнего фактически пройденного посещения.',
                         'Цвет риска по секундам и вероятность события >120с — разные величины.'])


def load_probability(expected_model_sha256: str,path: Path | str | None=None, *,
                     expected_encoder_sha256: str | None=None) -> tuple[ProbabilityModel | None,str | None]:
    """Опциональный явный путь; неверный JSON/артефакт даёт null, не падение ML."""
    configured=path if path is not None else os.environ.get('PROBABILITY_PATH')
    if not configured:
        return None,None
    try:
        return ProbabilityModel.load(configured,expected_model_sha256,expected_encoder_sha256=expected_encoder_sha256),None
    except (OSError,ValueError,TypeError) as error:
        return None,f'{type(error).__name__}: {error}'
