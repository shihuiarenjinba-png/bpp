from __future__ import annotations

import argparse
import time

from auto_agent.config import Settings
from auto_agent.orchestrator import AutoAgentOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the email-triggered auto agent monitor."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep polling Gmail at the configured interval.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    orchestrator = AutoAgentOrchestrator(settings)

    while True:
        summaries = orchestrator.process_inbox()
        if summaries:
            for summary in summaries:
                print(summary)
        else:
            print("no_matching_messages")

        if not args.loop:
            break

        time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    main()
