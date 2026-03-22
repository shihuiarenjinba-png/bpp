from __future__ import annotations

import json
from typing import Callable

from auto_agent.config import Settings
from auto_agent.models import ExecutionResult, TaskProposal, utc_now_iso


TaskHandler = Callable[[TaskProposal], ExecutionResult]


class TaskExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.handlers: dict[str, TaskHandler] = {
            "log_email_task": self._log_email_task,
            "draft_reply_stub": self._draft_reply_stub,
        }

    def supported_actions(self) -> list[str]:
        return sorted(self.handlers.keys())

    def can_execute(self, action_name: str) -> bool:
        return action_name in self.handlers

    def execute(self, proposal: TaskProposal) -> ExecutionResult:
        handler = self.handlers.get(proposal.action_name)
        if handler is None:
            return ExecutionResult(
                status="failed",
                details={"reason": f"Unsupported action: {proposal.action_name}"},
            )
        return handler(proposal)

    def _log_email_task(self, proposal: TaskProposal) -> ExecutionResult:
        log_line = {
            "timestamp": utc_now_iso(),
            "task_id": proposal.task_id,
            "title": proposal.title,
            "action_name": proposal.action_name,
            "payload": proposal.action_payload,
        }
        self.settings.execution_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.execution_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        return ExecutionResult(
            status="completed",
            details={"log_path": str(self.settings.execution_log_path)},
        )

    def _draft_reply_stub(self, proposal: TaskProposal) -> ExecutionResult:
        drafts_dir = self.settings.execution_log_path.parent / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        draft_path = drafts_dir / f"{proposal.task_id}.md"
        body = proposal.action_payload.get("draft_body", "")
        subject = proposal.action_payload.get("draft_subject", proposal.title)
        draft_path.write_text(
            f"# Draft Reply\n\nSubject: {subject}\n\n{body}\n",
            encoding="utf-8",
        )
        return ExecutionResult(
            status="completed",
            details={"draft_path": str(draft_path)},
        )
