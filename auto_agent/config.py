from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class Settings:
    gmail_user_id: str
    gmail_query: str
    gmail_max_results: int
    gmail_credentials_path: Path
    gmail_token_path: Path
    mark_as_read_after_handling: bool
    openai_api_key: str | None
    openai_model: str
    auto_execute_confidence_threshold: float
    slack_webhook_url: str | None
    approval_base_url: str | None
    approval_token: str | None
    approval_store_path: Path
    execution_log_path: Path
    poll_interval_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env")

        return cls(
            gmail_user_id=os.getenv("GMAIL_USER_ID", "me"),
            gmail_query=os.getenv(
                "GMAIL_QUERY",
                "is:unread from:support@outlier.ai newer_than:7d",
            ),
            gmail_max_results=int(os.getenv("GMAIL_MAX_RESULTS", "10")),
            gmail_credentials_path=_repo_path(
                os.getenv("GMAIL_CREDENTIALS_PATH", "secrets/gmail_credentials.json")
            ),
            gmail_token_path=_repo_path(
                os.getenv("GMAIL_TOKEN_PATH", "secrets/gmail_token.json")
            ),
            mark_as_read_after_handling=_as_bool(
                os.getenv("MARK_AS_READ_AFTER_HANDLING", "true")
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
            auto_execute_confidence_threshold=float(
                os.getenv("AUTO_EXECUTE_CONFIDENCE_THRESHOLD", "0.9")
            ),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            approval_base_url=os.getenv("APPROVAL_BASE_URL"),
            approval_token=os.getenv("APPROVAL_TOKEN"),
            approval_store_path=_repo_path(
                os.getenv(
                    "APPROVAL_STORE_PATH", "var/auto_agent/pending_approvals.json"
                )
            ),
            execution_log_path=_repo_path(
                os.getenv("EXECUTION_LOG_PATH", "var/auto_agent/execution.log")
            ),
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "60")),
        )

    def ensure_runtime_paths(self) -> None:
        self.gmail_token_path.parent.mkdir(parents=True, exist_ok=True)
        self.approval_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_log_path.parent.mkdir(parents=True, exist_ok=True)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
