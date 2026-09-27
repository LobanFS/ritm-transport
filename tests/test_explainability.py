"""Математические гарантии локального объяснения; не проверка причинности."""
import json

import numpy as np
import pandas as pd
import pytest

from ml_service.explainability import FeatureGroup, GroupedShapleyExplainer, load_hybrid_explainer


class LinearEstimator:
    def predict(self, frame):
        return 2 * frame["cur_dev_s"].to_numpy() + 3 * frame["horizon_min"].to_numpy()


def explainer():
    return GroupedShapleyExplainer(
        estimator=LinearEstimator(),
        feature_names=("cur_dev_s", "horizon_min"),
        groups=(
            FeatureGroup("current_state", "Текущее отклонение", ("cur_dev_s",)),
            FeatureGroup("forecast_horizon", "Горизонт", ("horizon_min",)),
        ),
        derived_features=(),
        reference={"cur_dev_s": 1.0, "horizon_min": 2.0},
        reference_description="test reference",
    )


def test_exact_grouped_shapley_is_additive_and_batch_safe():
    frame = pd.DataFrame({"cur_dev_s": [4.0, -1.0], "horizon_min": [5.0, 2.0]})
    original = frame.copy()
    predictions, explanations = explainer().explain(frame, [3.0, -2.0])
    np.testing.assert_allclose(predictions, [23.0, 4.0])
    np.testing.assert_allclose(
        [[factor.effect_s for factor in item.factors] for item in explanations],
        [[6.0, 9.0], [-4.0, 0.0]],
    )
    for prediction, item in zip(predictions, explanations, strict=True):
        assert item.reference_prediction_s == 8.0
        assert item.reconstructed_prediction_s == pytest.approx(prediction)
        assert abs(item.reconstruction_error_s) < 1e-12
    assert explanations[0].model_adjustment_s == 20.0
    assert explanations[1].factors[1].direction == "neutral"
    pd.testing.assert_frame_equal(frame, original)


def test_groups_must_cover_each_feature_exactly_once():
    with pytest.raises(ValueError, match="ровно один раз"):
        GroupedShapleyExplainer(
            estimator=LinearEstimator(), feature_names=("cur_dev_s", "horizon_min"),
            groups=(FeatureGroup("current_state", "x", ("cur_dev_s",)),),
            derived_features=(), reference={"cur_dev_s": 1.0, "horizon_min": 2.0},
            reference_description="bad",
        )


def test_hybrid_reference_is_bound_to_both_artifact_hashes(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "schema_version": "hybrid-explanation-reference-v1",
        "model_sha256": "a" * 64,
        "encoder_sha256": "c" * 64,
        "feature_names": [], "values": {}, "description": "fixture",
    }))
    with pytest.raises(ValueError, match="другим весам"):
        load_hybrid_explainer(LinearEstimator(), path, model_sha256="b" * 64,
                              encoder_sha256="c" * 64, feature_names=())
