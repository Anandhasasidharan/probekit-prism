"""Alert routing: Slack webhooks (httpx) + pluggable handlers."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from probekit.core.types import DriftAlert

Handler = Callable[[DriftAlert], None]


class AlertManager:
    def __init__(self, slack_webhook: str = "", min_severity: str = "warning"):
        self.slack_webhook = slack_webhook
        self.min_severity = min_severity
        self.history: list[DriftAlert] = []
        self._handlers: list[Handler] = []

    _SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

    def add_handler(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def should_fire(self, alert: DriftAlert) -> bool:
        return (
            self._SEVERITY_ORDER.get(alert.severity, 0)
            >= self._SEVERITY_ORDER.get(self.min_severity, 1)
        )

    def fire(self, alert: DriftAlert) -> bool:
        self.history.append(alert)
        if not self.should_fire(alert):
            return False
        if self.slack_webhook:
            try:
                httpx.post(
                    self.slack_webhook,
                    json={
                        "text": f"[ProbeKit:{alert.severity}] {alert.kind}\n{alert.message}"
                    },
                    timeout=10.0,
                )
            except httpx.HTTPError:
                pass  # alert delivery must never crash the monitor loop
        for h in self._handlers:
            h(alert)
        return True
