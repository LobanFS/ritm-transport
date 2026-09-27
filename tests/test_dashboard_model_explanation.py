"""Карточка причины сохраняет полезные факторы модели без отдельной панели."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_combines_observations_and_model_factors_without_extra_panel():
    html = (ROOT / "dashboard/index.html").read_text()
    script = (ROOT / "dashboard/app.js").read_text()
    assert 'id="detail-model-factors"' not in html
    assert 'id="detail-observations"' in html
    assert 'prediction?.forecast_explanation' in script
    assert "[...observations, ...modelFactors]" in script
    assert "не физическая причина" in script
    assert "neural prior" not in script
