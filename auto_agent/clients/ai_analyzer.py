from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from auto_agent.config import Settings
from auto_agent.models import EmailMessage, TaskProposal


class OpenAITaskAnalyzer:
    def __init__(self, settings: Settings, supported_actions: list[str]) -> None:
        self.settings = settings
        self.supported_actions = supported_actions
        self.client = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )

    def analyze(self, email_message: EmailMessage) -> TaskProposal:
        if self.client is None:
            return self._fallback_task(email_message)

        response = self.client.responses.create(
            model=self.settings.openai_model,
            instructions=self._system_instructions(),
            input=self._build_user_prompt(email_message),
        )
        parsed = self._parse_json(self._extract_text(response))

        proposal = TaskProposal.new(
            title=parsed["title"],
            summary=parsed["summary"],
            action_name=parsed["action_name"],
            action_payload=parsed.get("action_payload", {}),
            confidence=float(parsed["confidence"]),
            requires_human_approval=bool(parsed["requires_human_approval"]),
            risk_level=parsed.get("risk_level", "medium"),
            rationale=parsed.get("rationale", ""),
            source_message_id=email_message.message_id,
        )
        return proposal

    def _system_instructions(self) -> str:
        return (
            "You triage inbound emails into automation tasks. "
            "Return only valid JSON. "
            "Use auto execution only for low-risk tasks that are deterministic, reversible, "
            "and supported by the allowed action list. "
            f"Allowed action names: {', '.join(self.supported_actions)}. "
            "JSON schema: "
            "{"
            '"title": string, '
            '"summary": string, '
            '"action_name": string, '
            '"action_payload": object, '
            '"confidence": number, '
            '"requires_human_approval": boolean, '
            '"risk_level": "low"|"medium"|"high", '
            '"rationale": string'
            "}."
        )

    def _build_user_prompt(self, email_message: EmailMessage) -> str:
        return (
            "Analyze the following email and decide whether it should be auto-executed "
            "or sent for human approval.\n\n"
            f"From: {email_message.sender}\n"
            f"Subject: {email_message.subject}\n"
            f"Received At: {email_message.received_at}\n"
            f"Snippet: {email_message.snippet}\n"
            "Body:\n"
            f"{email_message.body_text}\n"
        )

    def _fallback_task(self, email_message: EmailMessage) -> TaskProposal:
        return TaskProposal.new(
            title=f"Manual review needed: {email_message.subject}",
            summary="OpenAI API key is not configured, so this email was routed to manual approval.",
            action_name="log_email_task",
            action_payload={
                "sender": email_message.sender,
                "subject": email_message.subject,
            },
            confidence=0.0,
            requires_human_approval=True,
            risk_level="medium",
            rationale="Fallback mode was used because OPENAI_API_KEY is missing.",
            source_message_id=email_message.message_id,
        )

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        return json.loads(cleaned)

    def _extract_text(self, response: Any) -> str:
        if getattr(response, "output_text", None):
            return response.output_text

        output = getattr(response, "output", []) or []
        fragments: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    fragments.append(text)

        if fragments:
            return "\n".join(fragments)
        raise ValueError("OpenAI response did not contain any text output.")
