"""Stable JSONL export shared by the CLI and weekly scheduler."""
from __future__ import annotations

import json
from pathlib import Path

from backend.learning_store import LearningStore, SCHEMA_VERSION


def export_examples(store_path: Path, output: Path) -> dict[str, int | str]:
    store = LearningStore(store_path)
    rows = store.examples()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "request": json.loads(row["request_json"]),
                "prediction": json.loads(row["response_json"]),
                "label": {
                    "target_delay_s": row["target_delay_s"],
                    "arrived_at": row["arrived_at"],
                    "available_at": row["label_received_at"],
                    "source": "arrival",
                },
            }
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    return {**store.stats(), "exported": len(rows), "output": str(output.resolve())}
