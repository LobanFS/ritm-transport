"""Скопировать проверенный frozen team artifact в игнорируемый runtime bundle.

Из filipp/: python tools/prepare_model.py. Обучение не запускается; исходные
артефакты alexchist остаются неизменными. Веса не добавлять в Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_service.learned import ENCODER_SHA256, MODEL_SHA256, MODEL_VERSION
from ml_service.probability import ProbabilityModel


def _validate_probability(path: Path, content: bytes, model_sha: str, encoder_sha: str):
    """Проверить runtime-контракт и то, что загрузчик прочитал те же байты."""
    checked = ProbabilityModel.load(path, model_sha, expected_encoder_sha256=encoder_sha)
    if checked.artifact_sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError('Файл калибратора изменился во время проверки')


def prepare_bundle(source_dir: Path, out: Path, probability_path: Path | None = None):
    """Проверить входы, сохранить прежний bundle; калибратор задаётся явно."""
    source = source_dir / "hybrid_model.joblib"
    encoder = source_dir / "swiss_encoder.pt"
    # Сохраняем именно проверенные байты: изменение source после проверки
    # не должно подменить устанавливаемые веса.
    model_bytes, encoder_bytes = source.read_bytes(), encoder.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    encoder_digest = hashlib.sha256(encoder_bytes).hexdigest()
    if digest != MODEL_SHA256 or encoder_digest != ENCODER_SHA256:
        raise SystemExit("STOP: неподтверждённый hash HGBR / Transformer")

    probability_bytes = None
    if probability_path is not None:
        probability_path = Path(probability_path)
        try:
            probability_bytes = probability_path.read_bytes()
            _validate_probability(probability_path, probability_bytes, digest, encoder_digest)
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"STOP: калибратор не прошёл проверку: {error}") from error

    names = ("model.joblib", "encoder.pt", "manifest.json", "probability.json")
    previous = {name: (out/name).read_bytes() for name in names if (out/name).is_file()}
    stale_probability = False
    if "probability.json" in previous:
        try:
            _validate_probability(out/"probability.json", previous["probability.json"], digest, encoder_digest)
        except (OSError, ValueError, TypeError):
            stale_probability = True
    changed_weights = any(name in previous and previous[name] != content
                          for name, content in (("model.joblib", model_bytes), ("encoder.pt", encoder_bytes)))
    backup = None
    changed_probability = probability_bytes is not None and previous.get("probability.json") != probability_bytes
    if previous and (changed_weights or stale_probability or changed_probability):
        # Один и тот же прежний набор получает тот же путь; повторная подготовка
        # не плодит копии. Хеш охватывает и метаданные/калибратор, не только веса.
        snapshot = hashlib.sha256()
        for name, content in sorted(previous.items()):
            snapshot.update(name.encode()+b"\0"+hashlib.sha256(content).digest())
        backup = out / "backups" / snapshot.hexdigest()
        for name, content in previous.items():
            if (backup/name).exists() and (backup/name).read_bytes() != content:
                raise SystemExit(f"STOP: прежняя резервная копия изменена: {backup/name}")
        backup.mkdir(parents=True, exist_ok=True)
        for name, content in previous.items():
            if not (backup/name).exists():
                (backup/name).write_bytes(content)
        if stale_probability:
            # Копия уже проверена/сохранена; файл перемещается из активного
            # bundle, чтобы старую калибровку нельзя было принять за новую.
            (out/"probability.json").replace(backup/"probability.json")

    out.mkdir(parents=True, exist_ok=True)
    (out/"model.joblib").write_bytes(model_bytes)
    (out/"encoder.pt").write_bytes(encoder_bytes)
    if probability_bytes is not None:
        (out/"probability.json").write_bytes(probability_bytes)
    elif not stale_probability:
        probability_bytes = previous.get("probability.json")
    probability_sha = hashlib.sha256(probability_bytes).hexdigest() if probability_bytes is not None else None
    manifest = {
        "model_version": MODEL_VERSION, "sha256": digest, "encoder_sha256": encoder_digest,
        "source": str(source.resolve()), "encoder_source": str(encoder.resolve()),
        "moscow_training_rows": 4434, "trained_here": False,
        "instruction": "MODEL_PATH=<model.joblib>; ENCODER_PATH необязателен, по умолчанию соседний encoder.pt",
    }
    if probability_sha is not None:
        manifest["probability_sha256"] = probability_sha
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    prepared = str((out/'model.joblib').resolve())
    prepared_encoder = str((out/'encoder.pt').resolve())
    prepared_probability = str((out/'probability.json').resolve()) if probability_sha is not None else None
    environment = {"MODEL_PATH": prepared, "ENCODER_PATH": prepared_encoder}
    if prepared_probability is not None:
        environment["PROBABILITY_PATH"] = prepared_probability
    return {"status": "PASS", "model": prepared, "sha256": digest,
            "encoder": prepared_encoder, "encoder_sha256": encoder_digest,
            "probability": prepared_probability, "probability_sha256": probability_sha,
            "previous_bundle_backup": str(backup.resolve()) if backup else None,
            "stale_probability_archived": stale_probability,
            "environment": environment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "alexchist/artifacts/experiments/swiss_sequence_hybrid")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/model")
    parser.add_argument("--probability", type=Path,
                        help="Явный проверенный JSON калибратора; без аргумента сохраняется совместимый текущий")
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.source, args.out, probability_path=args.probability), ensure_ascii=False))


if __name__ == "__main__":
    main()
