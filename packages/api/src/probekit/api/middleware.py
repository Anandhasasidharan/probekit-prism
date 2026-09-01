"""API middleware: per-IP token-bucket rate limiting + request logging."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("probekit.api")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.capacity = float(requests_per_minute)
        self.refill_rate = requests_per_minute / 60.0
        self.buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_ts)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/ready"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        tokens, last = self.buckets.get(client_ip, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_rate)
        if tokens < 1.0:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                {"detail": "rate limit exceeded"}, status_code=429
            )
        self.buckets[client_ip] = (tokens - 1.0, now)
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"
        return response
