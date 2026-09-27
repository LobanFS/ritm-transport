"""Замороженный sequence encoder и точная runtime-проекция его входов."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


STEPS = 12
CHANNELS = 5
CONTEXT_FEATURES = 6
HIDDEN = 32
TARGET_SCALE_S = 300.0


class DelayTransformer(nn.Module):
    """Архитектура, которой соответствует frozen swiss_encoder.pt."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(CHANNELS, HIDDEN)
        self.position = nn.Parameter(torch.zeros(1, STEPS, HIDDEN))
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=4, dim_feedforward=64, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.context = nn.Sequential(nn.Linear(CONTEXT_FEATURES, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(HIDDEN + 16, 32), nn.ReLU(), nn.Dropout(0.05), nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        padding = sequence[:, :, -1].eq(0)
        encoded = self.encoder(self.input(sequence) + self.position, src_key_padding_mask=padding)
        return self.head(torch.cat([encoded[:, -1], self.context(context)], dim=1)).squeeze(1)


def load_transformer(path: Path) -> DelayTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "transformer":
        raise ValueError("encoder checkpoint имеет другую архитектуру")
    config = checkpoint.get("config", {})
    expected = {"steps": STEPS, "channels": CHANNELS, "context_features": CONTEXT_FEATURES,
                "hidden": HIDDEN, "target_scale_s": TARGET_SCALE_S}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("encoder checkpoint имеет несовместимую конфигурацию")
    model = DelayTransformer()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def sequence_row(times: Sequence[datetime], delays: Sequence[float], now: datetime) -> tuple[np.ndarray, int, float]:
    """Повторяет causal_histories из эксперимента: current включён последним."""
    if len(times) != len(delays) or not times:
        raise ValueError("sequence требует хотя бы текущее отклонение")
    if any(left >= right for left, right in zip(times, times[1:])) or times[-1] != now:
        raise ValueError("sequence timestamps должны возрастать и завершаться issued_at")
    pairs = [(stamp, float(delay)) for stamp, delay in zip(times, delays)
             if 0 <= (now - stamp).total_seconds() <= 90 * 60][-STEPS:]
    selected_times = np.asarray([stamp.timestamp() for stamp, _ in pairs], dtype=float)
    selected_delays = np.asarray([delay for _, delay in pairs], dtype=np.float32)
    result = np.zeros((STEPS, CHANNELS), dtype=np.float32)
    count = len(pairs)
    start = STEPS - count
    result[start:, 0] = np.clip(selected_delays / 300.0, -5, 5)
    if count > 1:
        result[start + 1:, 1] = np.clip(np.diff(selected_delays) / 300.0, -5, 5)
        result[start + 1:, 3] = np.clip(np.diff(selected_times) / 900.0, 0, 6)
    result[start:, 2] = np.clip((now.timestamp() - selected_times) / 3600.0, 0, 1.5)
    result[start:, 4] = 1.0
    return result, count, float(now.timestamp() - selected_times[0])


def context_row(plan_features, history_count: int) -> np.ndarray:
    return np.asarray([
        np.clip(float(plan_features["cur_dev_s"]) / 300.0, -5, 5),
        np.clip((float(plan_features["horizon_min"]) - 12.5) / 2.5, -3, 3),
        np.clip(float(plan_features["stops_ahead"]) / 10.0, 0, 5),
        np.clip(float(plan_features["state_rel_pos"]), 0, 1),
        np.clip(float(plan_features["route_len"]) / 100.0, 0, 5),
        history_count / STEPS,
    ], dtype=np.float32)


def transformer_prior(model: DelayTransformer, sequence: np.ndarray, context: np.ndarray,
                      current: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        residual = model(torch.from_numpy(sequence), torch.from_numpy(context)).cpu().numpy()
    return current.astype(float) + residual * TARGET_SCALE_S
