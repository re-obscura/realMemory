"""Append-only JSONL-журнал событий пластичности.

Назначение v0: аудит и отладка (что, когда и с какой силой писалось).
Реплей журнала в состояние сети — будущая работа; сейчас состояние L2
восстанавливается из snapshot.pkl.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, **fields: Any) -> None:
        record = {"type": event_type, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def events(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def count(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
