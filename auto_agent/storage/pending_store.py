from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from auto_agent.models import PendingApproval, utc_now_iso


class PendingApprovalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def save(self, pending: PendingApproval) -> None:
        with self._lock:
            items = self._load_all()
            items[pending.approval_id] = pending
            pending.updated_at = utc_now_iso()
            self._write_all(items)

    def get(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._load_all().get(approval_id)

    def list_pending(self) -> list[PendingApproval]:
        with self._lock:
            return [
                item
                for item in self._load_all().values()
                if item.status == "pending"
            ]

    def _load_all(self) -> dict[str, PendingApproval]:
        if not self.path.exists():
            return {}

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            item["approval_id"]: PendingApproval.from_dict(item)
            for item in raw.get("items", [])
        }

    def _write_all(self, items: dict[str, PendingApproval]) -> None:
        payload = {"items": [item.to_dict() for item in items.values()]}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
