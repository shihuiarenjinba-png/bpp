from __future__ import annotations

from auto_agent.models import ExecutionResult, PendingApproval
from auto_agent.services.executor import TaskExecutor
from auto_agent.storage.pending_store import PendingApprovalStore


class ApprovalService:
    def __init__(
        self,
        store: PendingApprovalStore,
        executor: TaskExecutor,
    ) -> None:
        self.store = store
        self.executor = executor

    def register(self, pending: PendingApproval) -> PendingApproval:
        self.store.save(pending)
        return pending

    def approve(
        self,
        approval_id: str,
        *,
        decided_by: str = "human",
        note: str | None = None,
    ) -> tuple[PendingApproval, ExecutionResult]:
        pending = self.store.get(approval_id)
        if pending is None:
            raise KeyError(f"Approval not found: {approval_id}")
        if pending.status not in {"pending", "approved"}:
            raise ValueError(f"Approval is already {pending.status}")

        pending.status = "approved"
        pending.decided_by = decided_by
        pending.note = note
        self.store.save(pending)

        result = self.executor.execute(pending.task)
        pending.execution_result = result
        pending.status = result.status
        self.store.save(pending)
        return pending, result

    def reject(
        self,
        approval_id: str,
        *,
        decided_by: str = "human",
        note: str | None = None,
    ) -> PendingApproval:
        pending = self.store.get(approval_id)
        if pending is None:
            raise KeyError(f"Approval not found: {approval_id}")
        if pending.status != "pending":
            raise ValueError(f"Approval is already {pending.status}")

        pending.status = "rejected"
        pending.decided_by = decided_by
        pending.note = note
        self.store.save(pending)
        return pending
