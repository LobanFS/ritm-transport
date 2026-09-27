"""Общие пороги отображения, предупреждений и сценария подачи автобуса."""
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    amber_from_s: FiniteFloat = Field(ge=0)
    red_above_s: FiniteFloat = Field(gt=0)
    early_below_s: FiniteFloat = Field(lt=0)
    transfer_from_s: FiniteFloat = Field(ge=0)
    donor_max_delay_s: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.amber_from_s >= self.red_above_s:
            raise ValueError("Красный порог должен быть выше начала жёлтого диапазона")
        return self


RISK_POLICY = RiskPolicy.model_validate_json(
    Path(__file__).with_name("risk_policy.json").read_text(encoding="utf-8")
)


def risk_for_delay(delay_s: float | None) -> Literal["green", "amber", "red", "unknown"]:
    """Риск опоздания; опережение остаётся green, его синий цвет задаёт UI."""
    if delay_s is None or not math.isfinite(delay_s):
        return "unknown"
    if delay_s > RISK_POLICY.red_above_s:
        return "red"
    if delay_s >= RISK_POLICY.amber_from_s:
        return "amber"
    return "green"
