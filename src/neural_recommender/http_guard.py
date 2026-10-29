"""Small ASGI limits and response metadata for the local research application.

The request cap is applied to streamed bytes before JSON parsing, including when
Content-Length is absent or false. This bounds application buffering; it is not
a replacement for a reverse proxy, authentication, or a server connection limit.
The same small module is vendored in each independently installable project.
"""

from __future__ import annotations

import json
import logging
import math
import time
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 256 * 1024
MAX_JSON_DEPTH = 64


def validate_json_limits(body: bytes) -> None:
    """Bound nesting before downstream validation/error rendering recurses."""

    def finite_number(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON numbers are unsupported")
        return number

    decoded = json.loads(body, parse_float=finite_number, parse_constant=finite_number)
    pending = [(decoded, 0)]
    while pending:
        value, depth = pending.pop()
        if isinstance(value, (list, dict)):
            if depth >= MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds the configured limit")
            children = value.values() if isinstance(value, dict) else value
            pending.extend(
                (child, depth + 1) for child in children if isinstance(child, (list, dict))
            )


class ResearchHTTPGuard:
    """Reject oversized bodies and attach non-sensitive request trace metadata."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("HTTP request limit must be a positive integer")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        started = time.perf_counter()
        status = 500
        response_started = False

        async def guarded_send(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-request-id", request_id.encode("ascii")),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"server-timing",
                            f"app;dur={(time.perf_counter() - started) * 1000:.2f}".encode(),
                        ),
                    ]
                )
                # Model inputs/results are private by default. Static assets keep
                # the framework's ETag behavior; no user documents enter those files.
                if not scope.get("path", "").startswith("/static/"):
                    headers.append((b"cache-control", b"no-store"))
                if scope.get("path") == "/":
                    # Generated /docs uses FastAPI's own assets; this restrictive
                    # policy applies to the dependency-free application page only.
                    headers.append(
                        (
                            b"content-security-policy",
                            b"default-src 'self'; script-src 'self'; "
                            b"style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                            b"connect-src 'self'; object-src 'none'; "
                            b"base-uri 'none'; frame-ancestors 'none'",
                        )
                    )
                message = {**message, "headers": headers}
            await send(message)

        async def reject(code: int, detail: str) -> None:
            payload = json.dumps({"detail": detail, "request_id": request_id}).encode()
            await guarded_send(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode()),
                    ],
                }
            )
            await guarded_send({"type": "http.response.body", "body": payload})

        try:
            declared = [
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ]
            if declared:
                try:
                    if len(declared) != 1 or not declared[0].isdigit():
                        raise ValueError("Invalid length")
                    length = int(declared[0])
                except ValueError:
                    await reject(400, "Malformed Content-Length")
                    return
                if length > self.max_bytes:
                    await reject(413, "Request body exceeds the configured byte limit")
                    return
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > self.max_bytes:
                    await reject(413, "Request body exceeds the configured byte limit")
                    return
                if chunk:
                    chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            content_type = next(
                (
                    value.split(b";", 1)[0].strip().lower()
                    for name, value in scope.get("headers", [])
                    if name.lower() == b"content-type"
                ),
                b"",
            )
            if body and (content_type == b"application/json" or content_type.endswith(b"+json")):
                try:
                    # Discard immediately. Catch decoder resource/encoding limits
                    # before the framework turns them into unexpected HTTP 500s.
                    validate_json_limits(body)
                except json.JSONDecodeError:
                    pass  # Ordinary malformed syntax keeps the framework's 422 response.
                except (RecursionError, UnicodeDecodeError, ValueError):
                    await reject(400, "JSON exceeds decoder limits or has invalid encoding")
                    return
            delivered = False

            async def bounded_receive() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, guarded_send)
        except Exception as error:
            if response_started:
                raise  # A streaming response cannot be replaced after its headers.
            # Keep a correlation ID even when FastAPI's outer error middleware
            # would otherwise construct an untraced 500. Never log exception text,
            # which can contain a supplied document or query.
            LOGGER.error(
                "Unhandled request failure request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            await reject(500, "Internal server error")
        finally:
            # Never log query strings, document bodies, prompts, or model outputs.
            LOGGER.info(
                "http method=%r path=%r status=%d duration_ms=%.2f request_id=%s",
                scope.get("method"),
                scope.get("path"),
                status,
                (time.perf_counter() - started) * 1000,
                request_id,
            )
