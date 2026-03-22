from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from auto_agent.clients.notifier import SlackWebhookNotifier
from auto_agent.config import Settings
from auto_agent.services.approval_service import ApprovalService
from auto_agent.services.executor import TaskExecutor
from auto_agent.storage.pending_store import PendingApprovalStore


def create_app() -> FastAPI:
    settings = Settings.from_env()
    settings.ensure_runtime_paths()
    store = PendingApprovalStore(settings.approval_store_path)
    executor = TaskExecutor(settings)
    notifier = SlackWebhookNotifier(settings)
    approval_service = ApprovalService(store, executor)

    app = FastAPI(title="Auto Agent Approval API")

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/approvals/{approval_id}/{decision}")
    def approval_link(
        approval_id: str,
        decision: str,
        token: str = Query(default=""),
        decided_by: str = Query(default="slack-link"),
        note: str | None = Query(default=None),
    ) -> PlainTextResponse:
        _ensure_token(settings, token)
        message = _handle_decision(
            approval_service,
            notifier,
            approval_id=approval_id,
            decision=decision,
            decided_by=decided_by,
            note=note,
        )
        return PlainTextResponse(message)

    @app.post("/approvals/{approval_id}/{decision}")
    def approval_post(
        approval_id: str,
        decision: str,
        x_approval_token: str | None = Header(default=None),
        decided_by: str = Query(default="api"),
        note: str | None = Query(default=None),
    ) -> JSONResponse:
        _ensure_token(settings, x_approval_token or "")
        message = _handle_decision(
            approval_service,
            notifier,
            approval_id=approval_id,
            decision=decision,
            decided_by=decided_by,
            note=note,
        )
        return JSONResponse({"message": message})

    return app


def _ensure_token(settings: Settings, supplied_token: str) -> None:
    if not settings.approval_token:
        raise HTTPException(
            status_code=500,
            detail="APPROVAL_TOKEN is not configured.",
        )
    if supplied_token != settings.approval_token:
        raise HTTPException(status_code=401, detail="Invalid approval token.")


def _handle_decision(
    approval_service: ApprovalService,
    notifier: SlackWebhookNotifier,
    *,
    approval_id: str,
    decision: str,
    decided_by: str,
    note: str | None,
) -> str:
    try:
        if decision == "approve":
            pending, result = approval_service.approve(
                approval_id,
                decided_by=decided_by,
                note=note,
            )
            notifier.send_execution_result(
                pending.task,
                result,
                approved_by=decided_by,
            )
            return f"Approved and executed: {pending.task.title} ({result.status})"
        if decision == "reject":
            pending = approval_service.reject(
                approval_id,
                decided_by=decided_by,
                note=note,
            )
            return f"Rejected: {pending.task.title}"
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    raise HTTPException(status_code=400, detail="decision must be approve or reject")


app = create_app()
