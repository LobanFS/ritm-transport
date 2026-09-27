"""Validation and de-correlation of exported production examples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from backend.learning_store import SCHEMA_VERSION
from common.contracts import Contract, Prediction, PredictionRequest


class ProductionLabel(Contract):
    target_delay_s: FiniteFloat
    arrived_at: AwareDatetime
    available_at: AwareDatetime
    source: Literal["arrival"]


class ProductionExample(Contract):
    schema_version: str = Field(pattern=f"^{SCHEMA_VERSION}$")
    request: PredictionRequest
    prediction: Prediction
    label: ProductionLabel

    @model_validator(mode="after")
    def causal_and_consistent(self):
        if self.prediction.method != "learned":
            raise ValueError("training export accepts learned predictions only")
        if self.request.current_delay_s is None:
            raise ValueError("learned training example requires current_delay_s")
        if (self.prediction.request_id != self.request.request_id
                or self.prediction.tr_id != self.request.tr_id
                or self.prediction.target != self.request.target):
            raise ValueError("prediction does not match request")
        if self.request.issued_at >= self.label.available_at:
            raise ValueError("label was already available when request was issued")
        expected = (self.label.arrived_at-self.request.target.scheduled_at).total_seconds()
        if abs(expected-self.label.target_delay_s) > 1e-6:
            raise ValueError("target_delay_s does not match arrival minus schedule")
        return self

    @property
    def target_key(self) -> tuple[int, str]:
        return self.request.tr_id, self.request.target.id


def load_examples(path: Path) -> list[ProductionExample]:
    examples = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                examples.append(ProductionExample.model_validate(json.loads(line)))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid production example at line {number}: {exc}") from exc
    if not examples:
        raise ValueError("production export is empty")
    return examples


def one_snapshot_per_target(examples: list[ProductionExample]) -> list[ProductionExample]:
    """Prevent high-frequency inference from over-weighting a single arrival."""
    selected: dict[tuple[int, str], ProductionExample] = {}
    for example in examples:
        lead = (example.request.target.scheduled_at-example.request.issued_at).total_seconds()
        current = selected.get(example.target_key)
        if current is None:
            selected[example.target_key] = example
            continue
        current_lead = (current.request.target.scheduled_at-current.request.issued_at).total_seconds()
        candidate_rank = (abs(lead-750), example.request.issued_at, example.request.request_id)
        current_rank = (abs(current_lead-750), current.request.issued_at, current.request.request_id)
        if candidate_rank < current_rank:
            selected[example.target_key] = example
    return sorted(selected.values(), key=lambda item: (item.label.available_at, item.target_key))


def temporal_split(examples: list[ProductionExample], validation_fraction: float = .2):
    """Split by prediction time and purge labels unavailable at the cutoff.

    Sorting by arrival/label time is insufficient: an earlier arrival may still
    be in the future of the first validation prediction. Requests issued at the
    same instant stay together, and only already known labels enter fitting.
    """
    if not 0 < validation_fraction < .5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    ordered = sorted(one_snapshot_per_target(examples),
                     key=lambda item: (item.request.issued_at, item.target_key))
    validation_size = max(1, round(len(ordered)*validation_fraction))
    if validation_size >= len(ordered):
        raise ValueError("at least two labelled targets are required")
    cutoff = ordered[-validation_size].request.issued_at
    validation = [item for item in ordered if item.request.issued_at >= cutoff]
    fit = [item for item in ordered
           if item.request.issued_at < cutoff and item.label.available_at < cutoff]
    if not fit:
        raise ValueError("no mature training labels before validation request time")
    return fit, validation
