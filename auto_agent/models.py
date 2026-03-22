from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EmailMessage:
    message_id: str
    thread_id: str
    sender: str
    subject: str
    body_text: str
    snippet: str
    received_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "sender": self.sender,
            "subject": self.subject,
            "body_text": self.body_text,
            "snippet": self.snippet,
            "received_at": self.received_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmailMessage":
        return cls(
            message_id=payload["message_id"],
            thread_id=payload["thread_id"],
            sender=payload["sender"],
            subject=payload["subject"],
            body_text=payload["body_text"],
            snippet=payload.get("snippet", ""),
            received_at=payload.get("received_at"),
        )


@dataclass(slots=True)
class TaskProposal:
    task_id: str
    title: str
    summary: str
    action_name: str
    action_payload: dict[str, Any]
    confidence: float
    requires_human_approval: bool
    risk_level: str
    rationale: str
    source_message_id: str

    @classmethod
    def new(
        cls,
        *,
        title: str,
        summary: str,
        action_name: str,
        action_payload: dict[str, Any] | None,
        confidence: float,
        requires_human_approval: bool,
        risk_level: str,
        rationale: str,
        source_message_id: str,
    ) -> "TaskProposal":
        return cls(
            task_id=str(uuid4()),
            title=title,
            summary=summary,
            action_name=action_name,
            action_payload=action_payload or {},
            confidence=confidence,
            requires_human_approval=requires_human_approval,
            risk_level=risk_level,
            rationale=rationale,
            source_message_id=source_message_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "summary": self.summary,
            "action_name": self.action_name,
            "action_payload": self.action_payload,
            "confidence": self.confidence,
            "requires_human_approval": self.requires_human_approval,
            "risk_level": self.risk_level,
            "rationale": self.rationale,
            "source_message_id": self.source_message_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskProposal":
        return cls(
            task_id=payload["task_id"],
            title=payload["title"],
            summary=payload["summary"],
            action_name=payload["action_name"],
            action_payload=payload.get("action_payload", {}),
            confidence=float(payload.get("confidence", 0.0)),
            requires_human_approval=bool(
                payload.get("requires_human_approval", True)
            ),
            risk_level=payload.get("risk_level", "medium"),
            rationale=payload.get("rationale", ""),
            source_message_id=payload["source_message_id"],
        )


@dataclass(slots=True)
class ExecutionResult:
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    executed_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "details": self.details,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionResult":
        return cls(
            status=payload["status"],
            details=payload.get("details", {}),
            executed_at=payload.get("executed_at", utc_now_iso()),
        )


@dataclass(slots=True)
class PendingApproval:
    approval_id: str
    email: EmailMessage
    task: TaskProposal
    status: str = "pending"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    decided_by: str | None = None
    note: str | None = None
    execution_result: ExecutionResult | None = None

    @classmethod
    def new(cls, *, email: EmailMessage, task: TaskProposal) -> "PendingApproval":
        return cls(approval_id=str(uuid4()), email=email, task=task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "email": self.email.to_dict(),
            "task": self.task.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "decided_by": self.decided_by,
            "note": self.note,
            "execution_result": (
                self.execution_result.to_dict() if self.execution_result else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingApproval":
        execution_result = payload.get("execution_result")
        return cls(
            approval_id=payload["approval_id"],
            email=EmailMessage.from_dict(payload["email"]),
            task=TaskProposal.from_dict(payload["task"]),
            status=payload.get("status", "pending"),
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
            decided_by=payload.get("decided_by"),
            note=payload.get("note"),
            execution_result=(
                ExecutionResult.from_dict(execution_result)
                if execution_result
                else None
            ),
        )
