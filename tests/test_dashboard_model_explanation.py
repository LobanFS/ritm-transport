"""Отдельная панель сохраняет факторы и историю, которые вернул ML-сервис."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_has_model_factor_targets_and_reads_prediction_explanation():
    html = (ROOT / 'dashboard/index.html').read_text()
    script = (ROOT / 'dashboard/app.js').read_text()
    assert 'id="detail-model-factors"' in html
    assert 'id="detail-model-explanation-note"' in html
    assert 'prediction?.forecast_explanation' in script
    assert 'aria-label="Факторы модели"' in html
    assert 'transformerAnalysis?.summary' in script
    assert 'neural prior' not in script
    assert 'train-примера' not in script
    assert 'не физическая причина' in script
    assert 'снимок' not in script.lower()
