"""Exercise byte limits before parsing, including untrusted streaming headers."""

import asyncio
import json
import re

from neural_recommender.http_guard import ResearchHTTPGuard


def call_guard(chunks, headers=(), path="/answer", max_bytes=8):
    """A small ASGI harness keeps transport cases independent of HTTP libraries."""
    sent, observed = [], []
    messages = iter(
        [
            {"type": "http.request", "body": value, "more_body": index < len(chunks) - 1}
            for index, value in enumerate(chunks)
        ]
    )

    async def receive():
        return next(messages, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        observed.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    scope = {"type": "http", "path": path, "method": "POST", "headers": list(headers)}
    asyncio.run(ResearchHTTPGuard(app, max_bytes=max_bytes)(scope, receive, send))
    return sent, observed


def test_large_declared_body_is_rejected_before_downstream_read():
    sent, observed = call_guard([b"x" * 100], [(b"content-length", b"100")])
    assert sent[0]["status"] == 413
    assert not observed
    assert json.loads(sent[1]["body"])["request_id"]


def test_streamed_limit_applies_without_or_with_false_content_length():
    for headers in [[], [(b"content-length", b"2")]]:
        sent, observed = call_guard([b"12345", b"6789"], headers)
        assert sent[0]["status"] == 413
        assert not observed


def test_exact_boundary_body_and_response_privacy_metadata():
    sent, observed = call_guard([b"1234", b"", b"5678"])
    assert observed == [{"type": "http.request", "body": b"12345678", "more_body": False}]
    headers = dict(sent[0]["headers"])
    assert sent[0]["status"] == 200
    assert re.fullmatch(rb"[a-f0-9]{32}", headers[b"x-request-id"])
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"x-frame-options"] == b"DENY"


def test_malformed_or_duplicate_lengths_are_rejected():
    for headers in [
        [(b"content-length", b"-1")],
        [(b"content-length", b"NaN")],
        [(b"content-length", b"1"), (b"content-length", b"1")],
    ]:
        sent, observed = call_guard([b"x"], headers)
        assert sent[0]["status"] == 400
        assert not observed


def test_csp_is_for_app_page_without_breaking_generated_api_docs():
    home, _ = call_guard([b""], path="/")
    docs, _ = call_guard([b""], path="/docs")
    assert b"script-src 'self'" in dict(home[0]["headers"])[b"content-security-policy"]
    assert b"content-security-policy" not in dict(docs[0]["headers"])


def test_decoder_limits_return_controlled_errors_before_framework_parsing():
    for body in [
        b"[" * 2000 + b"]" * 2000,
        b"1" * 10000,
        b'"\xff"',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e999}',
    ]:
        sent, observed = call_guard(
            [body], [(b"content-type", b"application/json")], max_bytes=20000
        )
        assert sent[0]["status"] == 400
        assert not observed
    # The framework retains its own ordinary JSON syntax validation response.
    sent, observed = call_guard([b"{bad"], [(b"content-type", b"application/json")])
    assert sent[0]["status"] == 200 and observed


def test_unexpected_fastapi_failure_keeps_request_id_and_hides_sensitive_exception(caplog):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.add_middleware(ResearchHTTPGuard)

    @app.get("/crash")
    def crash():
        raise RuntimeError("private source document must not enter logs")

    response = TestClient(app, raise_server_exceptions=False).get("/crash")
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "private source document" not in caplog.text


def test_json_nesting_limit_counts_the_root_container():
    for depth, expected in [(64, 200), (65, 400)]:
        body = b"[" * depth + b"]" * depth
        sent, _ = call_guard([body], [(b"content-type", b"application/json")], max_bytes=1000)
        assert sent[0]["status"] == expected
