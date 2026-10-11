"""
CRRA — Append-only audit logger (Lab C4)

Writes one JSON object per line to logs/audit_trail.jsonl and echoes each entry
to the terminal. The file is only ever opened in append mode, so a re-run can
add to the trail but never overwrite it.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH, run_id: str | None = None, echo: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        self.echo = echo

    def log(self, node: str, contract_id: str, event: str, actor: str = "system", **details) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "contract_id": contract_id,
            "node": node,
            "event": event,
            "actor": actor,
            "details": details,
        }
        # Append-only: "a" mode never truncates existing history.
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        if self.echo:
            summary = ", ".join(f"{k}={_short(v)}" for k, v in details.items())
            print(f"  [AUDIT] {entry['timestamp']} · {contract_id} · {node} · {event} · actor={actor}"
                  + (f" · {summary}" if summary else ""))
        return entry


def _short(value, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"