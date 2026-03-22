from __future__ import annotations

import base64
import re
from email.utils import parsedate_to_datetime
from html import unescape

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from auto_agent.config import Settings
from auto_agent.models import EmailMessage


SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailInboxClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service = None

    def fetch_matching_messages(self) -> list[EmailMessage]:
        service = self._get_service()
        response = (
            service.users()
            .messages()
            .list(
                userId=self.settings.gmail_user_id,
                q=self.settings.gmail_query,
                maxResults=self.settings.gmail_max_results,
            )
            .execute()
        )

        messages = []
        for item in response.get("messages", []):
            full_message = (
                service.users()
                .messages()
                .get(
                    userId=self.settings.gmail_user_id,
                    id=item["id"],
                    format="full",
                )
                .execute()
            )
            messages.append(self._parse_message(full_message))
        return messages

    def mark_as_read(self, message_id: str) -> None:
        service = self._get_service()
        (
            service.users()
            .messages()
            .modify(
                userId=self.settings.gmail_user_id,
                id=message_id,
                body={"removeLabelIds": ["UNREAD"]},
            )
            .execute()
        )

    def _get_service(self):
        if self._service is None:
            credentials = self._load_credentials()
            self._service = build("gmail", "v1", credentials=credentials)
        return self._service

    def _load_credentials(self) -> Credentials:
        if not self.settings.gmail_credentials_path.exists():
            raise FileNotFoundError(
                "Gmail credentials file was not found. "
                f"Expected: {self.settings.gmail_credentials_path}"
            )

        creds = None
        if self.settings.gmail_token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self.settings.gmail_token_path), SCOPES
            )

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.settings.gmail_credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
            self.settings.gmail_token_path.write_text(creds.to_json())
        return creds

    def _parse_message(self, payload: dict) -> EmailMessage:
        headers = {
            header["name"]: header["value"]
            for header in payload["payload"]["headers"]
        }
        return EmailMessage(
            message_id=payload["id"],
            thread_id=payload["threadId"],
            sender=headers.get("From", ""),
            subject=headers.get("Subject", "(no subject)"),
            body_text=self._extract_body(payload["payload"]),
            snippet=payload.get("snippet", ""),
            received_at=self._parse_received_at(headers.get("Date")),
        )

    def _extract_body(self, payload: dict) -> str:
        text_body = self._walk_parts_for_mime(payload, "text/plain")
        if text_body:
            return text_body

        html_body = self._walk_parts_for_mime(payload, "text/html")
        if html_body:
            return _strip_html(html_body)
        return ""

    def _walk_parts_for_mime(self, payload: dict, mime_type: str) -> str:
        if payload.get("mimeType") == mime_type:
            return _decode_b64(payload.get("body", {}).get("data", ""))

        for part in payload.get("parts", []):
            nested = self._walk_parts_for_mime(part, mime_type)
            if nested:
                return nested

        if not payload.get("parts") and payload.get("mimeType") == mime_type:
            return _decode_b64(payload.get("body", {}).get("data", ""))
        return ""

    def _parse_received_at(self, date_header: str | None) -> str | None:
        if not date_header:
            return None
        try:
            return parsedate_to_datetime(date_header).isoformat()
        except (TypeError, ValueError):
            return None


def _decode_b64(value: str) -> str:
    if not value:
        return ""
    decoded = base64.urlsafe_b64decode(value.encode("utf-8"))
    return decoded.decode("utf-8", errors="ignore")


def _strip_html(html: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html)
    normalized = re.sub(r"\s+", " ", unescape(without_tags))
    return normalized.strip()
