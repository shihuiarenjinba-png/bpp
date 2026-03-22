from __future__ import annotations

from auto_agent.clients.ai_analyzer import OpenAITaskAnalyzer
from auto_agent.clients.gmail_client import GmailInboxClient
from auto_agent.clients.notifier import SlackWebhookNotifier
from auto_agent.config import Settings
from auto_agent.models import ExecutionResult, PendingApproval
from auto_agent.services.approval_service import ApprovalService
from auto_agent.services.executor import TaskExecutor
from auto_agent.storage.pending_store import PendingApprovalStore


class AutoAgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gmail_client = GmailInboxClient(settings)
        self.executor = TaskExecutor(settings)
        self.analyzer = OpenAITaskAnalyzer(
            settings,
            supported_actions=self.executor.supported_actions(),
        )
        self.notifier = SlackWebhookNotifier(settings)
        self.store = PendingApprovalStore(settings.approval_store_path)
        self.approval_service = ApprovalService(self.store, self.executor)

    def process_inbox(self) -> list[str]:
        self.settings.ensure_runtime_paths()
        summaries: list[str] = []

        for email_message in self.gmail_client.fetch_matching_messages():
            proposal = self.analyzer.analyze(email_message)

            if not self.executor.can_execute(proposal.action_name):
                proposal.requires_human_approval = True
                proposal.rationale = (
                    f"{proposal.rationale} Unsupported action requested by AI."
                ).strip()

            if (
                proposal.confidence < self.settings.auto_execute_confidence_threshold
                and not proposal.requires_human_approval
            ):
                proposal.requires_human_approval = True
                proposal.rationale = (
                    f"{proposal.rationale} Confidence below auto-execution threshold."
                ).strip()

            if proposal.requires_human_approval:
                pending = PendingApproval.new(email=email_message, task=proposal)
                self.approval_service.register(pending)
                self.notifier.send_approval_request(pending)
                summaries.append(
                    f"queued_for_approval:{email_message.message_id}:{pending.approval_id}"
                )
            else:
                result = self.executor.execute(proposal)
                self.notifier.send_execution_result(proposal, result)
                summaries.append(f"auto_executed:{email_message.message_id}:{result.status}")

            if self.settings.mark_as_read_after_handling:
                self.gmail_client.mark_as_read(email_message.message_id)

        return summaries
