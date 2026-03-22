from __future__ import annotations

from urllib.parse import quote

import requests

from auto_agent.config import Settings
from auto_agent.models import ExecutionResult, PendingApproval, TaskProposal


class SlackWebhookNotifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_approval_request(self, pending: PendingApproval) -> None:
        if not self.settings.slack_webhook_url:
            print(self._console_approval_message(pending))
            return

        approve_url, reject_url = self._build_approval_links(pending.approval_id)
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Approval required"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*From:* {pending.email.sender}\n"
                        f"*Subject:* {pending.email.subject}\n"
                        f"*Task:* {pending.task.title}\n"
                        f"*Summary:* {pending.task.summary}\n"
                        f"*Risk:* {pending.task.risk_level}\n"
                        f"*Rationale:* {pending.task.rationale}\n"
                        f"*Approval ID:* `{pending.approval_id}`"
                    ),
                },
            },
        ]

        if approve_url and reject_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Approve"},
                            "style": "primary",
                            "url": approve_url,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Reject"},
                            "style": "danger",
                            "url": reject_url,
                        },
                    ],
                }
            )
        else:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                "Set `APPROVAL_BASE_URL` and `APPROVAL_TOKEN` to include "
                                "one-click approval links."
                            ),
                        }
                    ],
                }
            )

        self._post({"text": f"Approval required: {pending.task.title}", "blocks": blocks})

    def send_execution_result(
        self,
        proposal: TaskProposal,
        result: ExecutionResult,
        *,
        approved_by: str | None = None,
    ) -> None:
        if not self.settings.slack_webhook_url:
            print(
                "Execution result:",
                proposal.title,
                result.status,
                result.details,
            )
            return

        executor_line = (
            f"Approved by: {approved_by}" if approved_by else "Executed automatically"
        )
        payload = {
            "text": f"Task executed: {proposal.title}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Task:* {proposal.title}\n"
                            f"*Status:* {result.status}\n"
                            f"*Mode:* {executor_line}\n"
                            f"*Details:* `{result.details}`"
                        ),
                    },
                }
            ],
        }
        self._post(payload)

    def _post(self, payload: dict) -> None:
        response = requests.post(
            self.settings.slack_webhook_url,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()

    def _build_approval_links(self, approval_id: str) -> tuple[str | None, str | None]:
        if not self.settings.approval_base_url or not self.settings.approval_token:
            return None, None

        base_url = self.settings.approval_base_url.rstrip("/")
        token = quote(self.settings.approval_token)
        approve_url = f"{base_url}/approvals/{approval_id}/approve?token={token}"
        reject_url = f"{base_url}/approvals/{approval_id}/reject?token={token}"
        return approve_url, reject_url

    def _console_approval_message(self, pending: PendingApproval) -> str:
        approve_url, reject_url = self._build_approval_links(pending.approval_id)
        return (
            "Approval required\n"
            f"  task: {pending.task.title}\n"
            f"  summary: {pending.task.summary}\n"
            f"  approval_id: {pending.approval_id}\n"
            f"  approve_url: {approve_url or 'not configured'}\n"
            f"  reject_url: {reject_url or 'not configured'}"
        )
