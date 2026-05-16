"""Unit tests for the generic webhook platform adapter.

Covers:
- HMAC signature validation (GitHub, GitLab, generic)
- Prompt rendering with dot-notation template variables
- Event type filtering
- HTTP handler behaviour (404, 202, health)
- Idempotency cache (duplicate delivery IDs)
- Rate limiting (fixed-window, per route)
- Body size limits
- INSECURE_NO_AUTH bypass
- Session isolation for concurrent webhooks
- Delivery info cleanup after send()
- connect / disconnect lifecycle
"""

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.webhook import (
    WebhookAdapter,
    _INSECURE_NO_AUTH,
    check_webhook_requirements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    routes=None,
    secret="",
    rate_limit=30,
    max_body_bytes=1_048_576,
    host="0.0.0.0",
    port=0,  # let OS pick a free port in tests
):
    """Build a PlatformConfig suitable for WebhookAdapter."""
    extra = {
        "host": host,
        "port": port,
        "routes": routes or {},
        "rate_limit": rate_limit,
        "max_body_bytes": max_body_bytes,
    }
    if secret:
        extra["secret"] = secret
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter(routes=None, **kwargs):
    """Create a WebhookAdapter with sensible defaults for testing."""
    config = _make_config(routes=routes, **kwargs)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter (without starting a full server)."""
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/linear/oauth/authorize", adapter._handle_linear_oauth_authorize)
    app.router.add_get("/linear/oauth/callback", adapter._handle_linear_oauth_callback)
    app.router.add_get("/linear/oauth/{agent}/authorize", adapter._handle_linear_oauth_authorize)
    app.router.add_get("/linear/oauth/{agent}/callback", adapter._handle_linear_oauth_callback)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _mock_request(headers=None, body=b"", content_length=None, match_info=None):
    """Build a lightweight mock aiohttp request for non-HTTP tests."""
    req = MagicMock()
    req.headers = headers or {}
    req.content_length = content_length if content_length is not None else len(body)
    req.match_info = match_info or {}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


def _generic_signature(body: bytes, secret: str) -> str:
    """Compute X-Webhook-Signature (plain HMAC-SHA256 hex) for *body*."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _linear_signature(body: bytes, secret: str) -> str:
    """Compute Linear linear-signature (plain HMAC-SHA256 hex) for *body*."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class _FakeStreamProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        if stdout:
            self.stdout.feed_data(stdout.encode())
        if stderr:
            self.stderr.feed_data(stderr.encode())
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.returncode = returncode
        self.killed = False

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


# ===================================================================
# Signature validation
# ===================================================================


class TestValidateSignature:
    """Tests for WebhookAdapter._validate_signature."""

    def test_validate_github_signature_valid(self):
        """Valid X-Hub-Signature-256 is accepted."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        sig = _github_signature(body, secret)
        req = _mock_request(headers={"X-Hub-Signature-256": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_github_signature_invalid(self):
        """Wrong X-Hub-Signature-256 is rejected."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        req = _mock_request(headers={"X-Hub-Signature-256": "sha256=deadbeef"})
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_gitlab_token(self):
        """GitLab plain-token match via X-Gitlab-Token."""
        adapter = _make_adapter()
        secret = "gl-token-value"
        req = _mock_request(headers={"X-Gitlab-Token": secret})
        assert adapter._validate_signature(req, b"{}", secret) is True

    def test_validate_gitlab_token_wrong(self):
        """Wrong X-Gitlab-Token is rejected."""
        adapter = _make_adapter()
        req = _mock_request(headers={"X-Gitlab-Token": "wrong"})
        assert adapter._validate_signature(req, b"{}", "correct") is False

    def test_validate_no_signature_with_secret_rejects(self):
        """Secret configured but no recognised signature header → reject."""
        adapter = _make_adapter()
        req = _mock_request(headers={})  # no sig headers at all
        assert adapter._validate_signature(req, b"{}", "my-secret") is False

    def test_validate_no_secret_allows_all(self):
        """When the secret is empty/falsy, the validator is never even called
        by the handler (secret check is 'if secret and secret != _INSECURE...').
        Verify that an empty secret isn't accidentally passed to the validator."""
        # This tests the semantics: empty secret means skip validation entirely.
        # The handler code does: if secret and secret != _INSECURE_NO_AUTH: validate
        # So with an empty secret, _validate_signature is never reached.
        # We just verify the code path is correct by constructing an adapter
        # with no secret and confirming the route config resolves to "".
        adapter = _make_adapter(
            routes={"test": {"prompt": "hello"}},
            secret="",
        )
        # The route has no secret, global secret is empty
        route_secret = adapter._routes["test"].get("secret", adapter._global_secret)
        assert not route_secret  # empty → validation is skipped in handler

    def test_validate_generic_signature_valid(self):
        """Valid X-Webhook-Signature (generic HMAC-SHA256 hex) is accepted."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        sig = _generic_signature(body, secret)
        req = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_linear_signature_valid(self):
        """Valid Linear linear-signature is accepted."""
        adapter = _make_adapter()
        body = b'{"type":"Comment","action":"create"}'
        secret = "linear-secret"
        sig = _linear_signature(body, secret)
        req = _mock_request(headers={"linear-signature": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_linear_signature_invalid(self):
        """Wrong Linear linear-signature is rejected."""
        adapter = _make_adapter()
        body = b'{"type":"Comment","action":"create"}'
        req = _mock_request(headers={"linear-signature": "deadbeef"})
        assert adapter._validate_signature(req, body, "linear-secret") is False

    def test_resolve_config_secret_from_env(self, monkeypatch):
        """Route secrets can be stored in env vars via env:VAR references."""
        monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "from-env")
        adapter = _make_adapter()
        assert adapter._resolve_config_secret("env:LINEAR_WEBHOOK_SECRET") == "from-env"


# ===================================================================
# Prompt rendering
# ===================================================================


class TestRenderPrompt:
    """Tests for WebhookAdapter._render_prompt."""

    def test_render_prompt_dot_notation(self):
        """Dot-notation {pull_request.title} resolves nested keys."""
        adapter = _make_adapter()
        payload = {"pull_request": {"title": "Fix bug", "number": 42}}
        result = adapter._render_prompt(
            "PR #{pull_request.number}: {pull_request.title}",
            payload,
            "pull_request",
            "github",
        )
        assert result == "PR #42: Fix bug"

    def test_render_prompt_missing_key_preserved(self):
        """{nonexistent} is left as-is when key doesn't exist in payload."""
        adapter = _make_adapter()
        result = adapter._render_prompt(
            "Hello {nonexistent}!",
            {"action": "opened"},
            "push",
            "test",
        )
        assert "{nonexistent}" in result

    def test_render_prompt_no_template_dumps_json(self):
        """Empty template → JSON dump fallback with event/route context."""
        adapter = _make_adapter()
        payload = {"key": "value"}
        result = adapter._render_prompt("", payload, "push", "my-route")
        assert "push" in result
        assert "my-route" in result
        assert "key" in result


# ===================================================================
# Delivery extra rendering
# ===================================================================


class TestRenderDeliveryExtra:
    def test_render_delivery_extra_templates(self):
        """String values in deliver_extra are rendered with payload data."""
        adapter = _make_adapter()
        extra = {"repo": "{repository.full_name}", "pr_number": "{number}", "static": 42}
        payload = {"repository": {"full_name": "org/repo"}, "number": 7}
        result = adapter._render_delivery_extra(extra, payload)
        assert result["repo"] == "org/repo"
        assert result["pr_number"] == "7"
        assert result["static"] == 42  # non-string left as-is


# ===================================================================
# Event filtering
# ===================================================================


class TestEventFilter:
    """Tests for event type filtering in _handle_webhook."""

    @pytest.mark.asyncio
    async def test_event_filter_accepts_matching(self):
        """Matching event type passes through."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "PR: {action}",
            }
        }
        adapter = _make_adapter(routes=routes)
        # Stub handle_message to avoid running the agent
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "pull_request"},
            )
            assert resp.status == 202

    @pytest.mark.asyncio
    async def test_event_filter_rejects_non_matching(self):
        """Non-matching event type returns 200 with status=ignored."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "test",
            }
        }
        adapter = _make_adapter(routes=routes)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "push"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ignored"

    @pytest.mark.asyncio
    async def test_event_filter_empty_allows_all(self):
        """No events list → accept any event type."""
        routes = {
            "all": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "got it",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/all",
                json={"action": "any"},
                headers={"X-GitHub-Event": "whatever"},
            )
            assert resp.status == 202

    @pytest.mark.asyncio
    async def test_event_filter_accepts_linear_payload_type(self):
        """Linear webhooks put the resource type in the JSON `type` field."""
        routes = {
            "linear": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["Comment"],
                "prompt": "comment: {action}",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/linear",
                json={"type": "Comment", "action": "create"},
            )
            assert resp.status == 202


# ===================================================================
# HTTP handling
# ===================================================================


class TestHTTPHandling:

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self):
        """POST to an unknown route returns 404."""
        adapter = _make_adapter(routes={"real": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/nonexistent", json={"a": 1})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_webhook_handler_returns_202(self):
        """Valid request returns 202 Accepted."""
        routes = {"test": {"secret": _INSECURE_NO_AUTH, "prompt": "hi"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/test", json={"data": "value"})
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "test"

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health returns 200 with status=ok."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "webhook"

    @pytest.mark.asyncio
    async def test_connect_starts_server(self):
        """connect() starts the HTTP listener and marks adapter as connected."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="127.0.0.1", port=0)
        # Use port 0 — the OS picks a free port, but aiohttp requires a real bind.
        # We just test that the method completes and marks connected.
        # Need to mock TCPSite to avoid actual binding.
        with patch("gateway.platforms.webhook.web.AppRunner") as MockRunner, \
             patch("gateway.platforms.webhook.web.TCPSite") as MockSite:
            mock_runner_inst = AsyncMock()
            MockRunner.return_value = mock_runner_inst
            mock_site_inst = AsyncMock()
            MockSite.return_value = mock_site_inst

            result = await adapter.connect()
            assert result is True
            assert adapter.is_connected
            mock_runner_inst.setup.assert_awaited_once()
            mock_site_inst.start.assert_awaited_once()

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """disconnect() stops the server and marks adapter disconnected."""
        adapter = _make_adapter()
        # Simulate a runner that was previously set up
        mock_runner = AsyncMock()
        adapter._runner = mock_runner
        adapter._running = True

        await adapter.disconnect()
        mock_runner.cleanup.assert_awaited_once()
        assert adapter._runner is None
        assert not adapter.is_connected


# ===================================================================
# Idempotency
# ===================================================================


class TestIdempotency:

    @pytest.mark.asyncio
    async def test_duplicate_delivery_id_returns_200(self):
        """Second request with same delivery ID returns 200 duplicate."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-123"}
            resp1 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp1.status == 202

            resp2 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp2.status == 200
            data = await resp2.json()
            assert data["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_expired_delivery_id_allows_reprocess(self):
        """After TTL expires, the same delivery ID is accepted again."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter._idempotency_ttl = 1  # 1 second TTL for test speed
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-456"}

            resp1 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp1.status == 202

            # Backdate the cache entry so it appears expired
            adapter._seen_deliveries["delivery-456"] = time.time() - 3700

            resp2 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp2.status == 202  # re-accepted


# ===================================================================
# Rate limiting
# ===================================================================


class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excess(self):
        """Exceeding the rate limit returns 429."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=2)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Two requests within limit
            for i in range(2):
                resp = await cli.post(
                    "/webhooks/limited",
                    json={"n": i},
                    headers={"X-GitHub-Delivery": f"d-{i}"},
                )
                assert resp.status == 202, f"Request {i} should be accepted"

            # Third request should be rate-limited
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 99},
                headers={"X-GitHub-Delivery": "d-99"},
            )
            assert resp.status == 429

    @pytest.mark.asyncio
    async def test_rate_limit_window_resets(self):
        """After the 60-second window passes, requests are allowed again."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=1)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 1},
                headers={"X-GitHub-Delivery": "d-a"},
            )
            assert resp.status == 202

            # Backdate all rate-limit timestamps to > 60 seconds ago
            adapter._rate_counts["limited"] = [time.time() - 120]

            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 2},
                headers={"X-GitHub-Delivery": "d-b"},
            )
            assert resp.status == 202  # allowed again


# ===================================================================
# Body size limit
# ===================================================================


class TestBodySize:

    @pytest.mark.asyncio
    async def test_oversized_payload_rejected(self):
        """Content-Length > max_body_bytes returns 413."""
        routes = {"big": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, max_body_bytes=100)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            large_payload = {"data": "x" * 200}
            resp = await cli.post(
                "/webhooks/big",
                json=large_payload,
                headers={"Content-Length": "999999"},
            )
            assert resp.status == 413


# ===================================================================
# INSECURE_NO_AUTH
# ===================================================================


class TestInsecureNoAuth:

    @pytest.mark.asyncio
    async def test_insecure_no_auth_skips_validation(self):
        """Setting secret to _INSECURE_NO_AUTH bypasses signature check."""
        routes = {"open": {"secret": _INSECURE_NO_AUTH, "prompt": "hello"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # No signature header at all — should still be accepted
            resp = await cli.post("/webhooks/open", json={"test": True})
            assert resp.status == 202


# ===================================================================
# Session isolation
# ===================================================================


class TestSessionIsolation:

    @pytest.mark.asyncio
    async def test_concurrent_webhooks_get_independent_sessions(self):
        """Two events on the same route produce different session keys."""
        routes = {"ci": {"secret": _INSECURE_NO_AUTH, "prompt": "build"}}
        adapter = _make_adapter(routes=routes)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp1 = await cli.post(
                "/webhooks/ci",
                json={"ref": "main"},
                headers={"X-GitHub-Delivery": "aaa-111"},
            )
            assert resp1.status == 202

            resp2 = await cli.post(
                "/webhooks/ci",
                json={"ref": "dev"},
                headers={"X-GitHub-Delivery": "bbb-222"},
            )
            assert resp2.status == 202

        # Wait for the async tasks to be created
        await asyncio.sleep(0.05)

        assert len(captured_events) == 2
        ids = {ev.source.chat_id for ev in captured_events}
        assert len(ids) == 2, "Each delivery must have a unique session chat_id"

    @pytest.mark.asyncio
    async def test_linear_agent_sessions_key_by_agent_session_not_delivery_id(self):
        routes = {
            "linear-agent": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["AgentSessionEvent"],
                "prompt": "session {agentSession.id}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for delivery in ("delivery-a", "delivery-b"):
                resp = await cli.post(
                    "/webhooks/linear-agent",
                    json={"type": "AgentSessionEvent", "agentSession": {"id": "session-123"}},
                    headers={"X-Linear-Delivery": delivery},
                )
                assert resp.status == 202
        await asyncio.sleep(0.05)
        assert [ev.source.chat_id for ev in captured_events] == [
            "webhook:linear-agent:linear-agent-session-123",
            "webhook:linear-agent:linear-agent-session-123",
        ]


# ===================================================================
# Delivery info cleanup
# ===================================================================


class TestDeliveryCleanup:

    @pytest.mark.asyncio
    async def test_delivery_info_survives_multiple_sends(self):
        """send() must NOT pop delivery_info.

        Interim status messages (fallback notifications, context-pressure
        warnings, etc.) flow through the same send() path as the final
        response.  If the entry were popped on the first send, the final
        response would silently downgrade to the ``log`` deliver type.
        Regression test for that bug.
        """
        adapter = _make_adapter()
        chat_id = "webhook:test:d-xyz"
        adapter._delivery_info[chat_id] = {
            "deliver": "log",
            "deliver_extra": {},
            "payload": {"x": 1},
        }
        adapter._delivery_info_created[chat_id] = time.time()

        # First send (e.g. an interim status message)
        result1 = await adapter.send(chat_id, "Status: switching to fallback")
        assert result1.success is True
        # Entry must still be present so the final send can read it
        assert chat_id in adapter._delivery_info

        # Second send (the final agent response)
        result2 = await adapter.send(chat_id, "Final agent response")
        assert result2.success is True
        assert chat_id in adapter._delivery_info

    @pytest.mark.asyncio
    async def test_delivery_info_pruned_via_ttl(self):
        """Stale delivery_info entries are dropped on the next POST."""
        adapter = _make_adapter()
        adapter._idempotency_ttl = 60  # short TTL for the test
        now = time.time()

        # Stale entry — older than TTL
        adapter._delivery_info["webhook:test:old"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:old"] = now - 120

        # Fresh entry — should survive
        adapter._delivery_info["webhook:test:new"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:new"] = now - 5

        adapter._prune_delivery_info(now)

        assert "webhook:test:old" not in adapter._delivery_info
        assert "webhook:test:old" not in adapter._delivery_info_created
        assert "webhook:test:new" in adapter._delivery_info
        assert "webhook:test:new" in adapter._delivery_info_created


# ===================================================================
# check_webhook_requirements
# ===================================================================


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_webhook_requirements() is True

    @patch("gateway.platforms.webhook.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_webhook_requirements() is False


# ===================================================================
# __raw__ template token
# ===================================================================


class TestRawTemplateToken:
    """Tests for the {__raw__} special token in _render_prompt."""

    def test_raw_resolves_to_full_json_payload(self):
        """{__raw__} in a template dumps the entire payload as JSON."""
        adapter = _make_adapter()
        payload = {"action": "opened", "number": 42}
        result = adapter._render_prompt(
            "Payload: {__raw__}", payload, "push", "test"
        )
        expected_json = json.dumps(payload, indent=2)
        assert result == f"Payload: {expected_json}"

    def test_raw_truncated_at_4000_chars(self):
        """{__raw__} output is truncated at 4000 characters for large payloads."""
        adapter = _make_adapter()
        # Build a payload whose JSON repr exceeds 4000 chars
        payload = {"data": "x" * 5000}
        result = adapter._render_prompt("{__raw__}", payload, "push", "test")
        assert len(result) <= 4000

    def test_raw_mixed_with_other_variables(self):
        """{__raw__} can be mixed with regular template variables."""
        adapter = _make_adapter()
        payload = {"action": "closed", "number": 7}
        result = adapter._render_prompt(
            "Action={action} Raw={__raw__}", payload, "push", "test"
        )
        assert result.startswith("Action=closed Raw=")
        assert '"action": "closed"' in result
        assert '"number": 7' in result


# ===================================================================
# Payload filtering
# ===================================================================


class TestPayloadFiltering:
    def test_requires_matching_payload_text(self):
        adapter = _make_adapter()
        route = {"filters": {"require": [{"path": "data.body", "regex": "(?i)\\bjarvis\\b"}]}}
        assert adapter._should_process_payload(route, {"data": {"body": "Jarvis please help"}})[0] is True
        ok, reason = adapter._should_process_payload(route, {"data": {"body": "hello"}})
        assert ok is False
        assert "required filter" in reason

    def test_rejects_loop_marker(self):
        adapter = _make_adapter()
        route = {"filters": {"reject": [{"path": "data.body", "contains": "Jarvis automated reply"}]}}
        ok, reason = adapter._should_process_payload(
            route, {"data": {"body": "Done.\n\n<sub>Jarvis automated reply</sub>"}}
        )
        assert ok is False
        assert "rejected" in reason


# ===================================================================
# Linear comment delivery
# ===================================================================


class TestLinearCommentDelivery:
    @pytest.mark.asyncio
    async def test_deliver_linear_comment_posts_graphql_comment(self, monkeypatch):
        """linear_comment delivery posts a CommentCreate mutation to Linear."""
        adapter = _make_adapter()
        monkeypatch.setenv("LINEAR_API_KEY", "lin-key")

        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "data": {
                    "commentCreate": {
                        "success": True,
                        "comment": {"id": "comment-1", "url": "https://linear.app/c/comment-1"},
                    }
                }
            }
        ).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = None

        with patch("gateway.platforms.webhook.urllib.request.urlopen", return_value=response) as mock_urlopen:
            result = await adapter._deliver_linear_comment(
                "Jarvis response",
                {"deliver_extra": {"issue_id": "issue-123"}},
            )

        assert result.success is True
        req = mock_urlopen.call_args.args[0]
        assert req.full_url == "https://api.linear.app/graphql"
        assert req.headers["Authorization"] == "lin-key"
        body = json.loads(req.data.decode("utf-8"))
        assert body["variables"]["input"] == {
            "issueId": "issue-123",
            "body": "Jarvis response\n\n<sub>Jarvis automated reply</sub>",
        }

    @pytest.mark.asyncio
    async def test_deliver_linear_comment_requires_issue_id(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("LINEAR_API_KEY", "lin-key")
        result = await adapter._deliver_linear_comment("body", {"deliver_extra": {}})
        assert result.success is False
        assert result.error and "issue_id" in result.error


# ===================================================================
# Cross-platform delivery thread_id passthrough
# ===================================================================


class TestDeliverCrossPlatformThreadId:
    """Tests for thread_id passthrough in _deliver_cross_platform."""

    def _setup_adapter_with_mock_target(self):
        """Set up a webhook adapter with a mocked gateway_runner and target adapter."""
        adapter = _make_adapter()
        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform("telegram"): mock_target}
        mock_runner.config.get_home_channel.return_value = None

        adapter.gateway_runner = mock_runner
        return adapter, mock_target

    @pytest.mark.asyncio
    async def test_thread_id_passed_as_metadata(self):
        """thread_id from deliver_extra is passed as metadata to adapter.send()."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "thread_id": "999",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "999"}
        )

    @pytest.mark.asyncio
    async def test_message_thread_id_passed_as_thread_id(self):
        """message_thread_id from deliver_extra is mapped to thread_id in metadata."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "message_thread_id": "888",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "888"}
        )

    @pytest.mark.asyncio
    async def test_no_thread_id_sends_no_metadata(self):
        """When no thread_id is present, metadata is None."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata=None
        )


class TestInsecureNoAuthSafetyRail:
    """connect() refuses to start when INSECURE_NO_AUTH is combined with a
    non-loopback bind. Guards against accidentally exposing an unauthenticated
    webhook endpoint on a public interface."""

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_public_bind(self):
        """INSECURE_NO_AUTH + 0.0.0.0 is refused before the server starts."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="0.0.0.0", port=0)
        with pytest.raises(ValueError, match="INSECURE_NO_AUTH"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_lan_ip(self):
        """A LAN IP is treated as public."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="192.168.1.50", port=0)
        with pytest.raises(ValueError, match="non-loopback"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_rejects_insecure_no_auth_on_empty_host(self):
        """Empty host is conservatively treated as non-loopback."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="", port=0)
        with pytest.raises(ValueError, match="INSECURE_NO_AUTH"):
            await adapter.connect()

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost"],
    )
    @pytest.mark.asyncio
    async def test_connect_allows_insecure_no_auth_on_loopback(self, host):
        """Recognised loopback hosts are permitted with INSECURE_NO_AUTH."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host=host, port=0)
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                result = await adapter.connect()
            assert result is True
        finally:
            await adapter.disconnect()

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost", "Localhost", "::1", "ip6-localhost", "ip6-loopback"],
    )
    def test_is_loopback_host_accepts(self, host):
        """_is_loopback_host covers all documented loopback spellings."""
        from gateway.platforms.webhook import _is_loopback_host
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.5", "10.0.0.1", "example.com", "", None],
    )
    def test_is_loopback_host_rejects(self, host):
        """_is_loopback_host treats public/LAN/empty as non-loopback."""
        from gateway.platforms.webhook import _is_loopback_host
        assert _is_loopback_host(host) is False

    @pytest.mark.asyncio
    async def test_connect_allows_real_secret_on_public_bind(self):
        """A real HMAC secret bound to 0.0.0.0 is the normal production case."""
        routes = {"r1": {"secret": "real-secret-abc123", "prompt": "x"}}
        adapter = _make_adapter(routes=routes, host="0.0.0.0", port=0)
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                result = await adapter.connect()
            assert result is True
        finally:
            await adapter.disconnect()



class TestLinearAgentSupport:
    def test_linear_agent_session_id_prefers_payload_agent_session(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "agentSession": {"id": "session-123"},
            "agentActivity": {"agentSessionId": "activity-session"},
        }
        assert adapter._linear_agent_session_id(payload) == "session-123"

    def test_linear_agent_session_id_falls_back_to_agent_activity(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "agentActivity": {"agentSessionId": "activity-session"},
        }
        assert adapter._linear_agent_session_id(payload) == "activity-session"

    def test_is_linear_agent_session_event_requires_payload_type(self):
        adapter = _make_adapter()
        assert adapter._is_linear_agent_session_event({"type": "AgentSessionEvent"}) is True
        assert adapter._is_linear_agent_session_event({"type": "Comment"}) is False

    def test_linear_agent_lifecycle_mode_defaults_assignment_to_grooming(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "action": "created",
            "agentSession": {"id": "session-1"},
            "promptContext": "Issue OTM-93 assigned to Jarvis",
        }
        assert adapter._linear_agent_lifecycle_mode(payload) == "groom"

    def test_linear_agent_lifecycle_mode_explicit_implementation_request(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "action": "created",
            "agentActivity": {"content": "Please implement this and open a PR."},
            "promptContext": "Fix the failing webhook tests",
        }
        assert adapter._linear_agent_lifecycle_mode(payload) == "implement"

    def test_linear_agent_lifecycle_mode_followup_discussion(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "action": "activityCreated",
            "agentActivity": {"content": "Can you split this into child issues?"},
        }
        assert adapter._linear_agent_lifecycle_mode(payload) == "groom"

    def test_linear_agent_bounds_block_includes_issue_team_project_and_mode(self):
        adapter = _make_adapter()
        payload = {
            "type": "AgentSessionEvent",
            "agentSession": {"id": "session-1"},
            "promptContext": "context",
            "data": {
                "issue": {
                    "id": "issue-uuid",
                    "identifier": "OTM-93",
                    "team": {"key": "OTM"},
                    "project": {"id": "project-uuid", "name": "JARVIS"},
                }
            },
        }
        block = adapter._linear_agent_bounds_block(payload, "groom")
        assert "Mode: groom" in block
        assert "Issue: OTM-93 / issue-uuid" in block
        assert "Team: OTM / (unknown)" in block
        assert "Project: JARVIS / project-uuid" in block
        assert "Allowed mutations: assigned issue and child issues only" in block
        assert "LINEAR_MUTATIONS" in block
        assert "Repository edits: forbidden unless explicitly requested" in block

    def test_validate_linear_issue_mutation_allows_current_issue_update(self):
        adapter = _make_adapter()
        bounds = {"issue_id": "issue-1", "project_id": "project-1"}
        ok, error = adapter._validate_linear_issue_mutation(
            bounds,
            {"operation": "issueUpdate", "id": "issue-1", "input": {"title": "Better title"}},
        )
        assert ok is True
        assert error == ""

    def test_validate_linear_issue_mutation_rejects_unrelated_issue(self):
        adapter = _make_adapter()
        bounds = {"issue_id": "issue-1"}
        ok, error = adapter._validate_linear_issue_mutation(
            bounds,
            {"operation": "issueUpdate", "id": "issue-2", "input": {"title": "Nope"}},
        )
        assert ok is False
        assert "outside assigned issue" in error

    def test_validate_linear_issue_mutation_rejects_state_closure(self):
        adapter = _make_adapter()
        bounds = {"issue_id": "issue-1"}
        ok, error = adapter._validate_linear_issue_mutation(
            bounds,
            {"operation": "issueUpdate", "id": "issue-1", "input": {"stateId": "done"}},
        )
        assert ok is False
        assert "stateId" in error

    def test_validate_linear_issue_mutation_rejects_child_issue_team_escape(self):
        adapter = _make_adapter()
        bounds = {"issue_id": "issue-1", "project_id": "project-1", "team_id": "team-1"}
        ok, error = adapter._validate_linear_issue_mutation(
            bounds,
            {
                "operation": "issueCreate",
                "input": {
                    "title": "Wrong team",
                    "parentId": "issue-1",
                    "projectId": "project-1",
                    "teamId": "team-2",
                },
            },
        )
        assert ok is False
        assert "teamId" in error

    def test_extract_linear_mutations_block_returns_clean_body_and_operations(self):
        adapter = _make_adapter()
        body = """Updated the issue.

LINEAR_MUTATIONS:
```json
[
  {"operation": "issueUpdate", "id": "issue-1", "input": {"title": "Better"}}
]
```

Visible summary."""
        clean, mutations = adapter._extract_linear_mutations_block(body)
        assert "LINEAR_MUTATIONS" not in clean
        assert "Visible summary." in clean
        assert mutations == [
            {"operation": "issueUpdate", "id": "issue-1", "input": {"title": "Better"}}
        ]

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_executes_bounded_mutations_before_response(self):
        adapter = _make_adapter()
        adapter._load_linear_agent_token = MagicMock(return_value="linear-token")
        adapter._post_linear_graphql = MagicMock(side_effect=[(True, "updated"), (True, "activity-id")])
        delivery = {
            "deliver_extra": {"agent_session_id": "session-1"},
            "payload": {
                "type": "AgentSessionEvent",
                "agentSession": {"id": "session-1"},
                "data": {
                    "issue": {
                        "id": "issue-1",
                        "team": {"id": "team-1", "key": "OTM"},
                        "project": {"id": "project-1", "name": "JARVIS"},
                    }
                },
            },
            "linear_agent": "jarvis",
            "linear_agent_required": True,
        }
        content = """Updated.

LINEAR_MUTATIONS:
```json
[{"operation":"issueUpdate","id":"issue-1","input":{"title":"Better"}}]
```

Done."""

        result = await adapter._deliver_linear_agent_activity(content, delivery)

        assert result.success is True
        assert adapter._post_linear_graphql.call_count == 2
        mutation_payload = adapter._post_linear_graphql.call_args_list[0].args[0]
        assert "issueUpdate" in mutation_payload["query"]
        assert mutation_payload["variables"] == {
            "id": "issue-1",
            "input": {"title": "Better"},
        }
        activity_payload = adapter._post_linear_graphql.call_args_list[1].args[0]
        activity_body = activity_payload["variables"]["input"]["content"]["body"]
        assert "LINEAR_MUTATIONS" not in activity_body
        assert "Done." in activity_body

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_rejects_out_of_bounds_mutation(self):
        adapter = _make_adapter()
        adapter._load_linear_agent_token = MagicMock(return_value="linear-token")
        adapter._post_linear_graphql = MagicMock(return_value=(True, "activity-id"))
        delivery = {
            "deliver_extra": {"agent_session_id": "session-1"},
            "payload": {
                "type": "AgentSessionEvent",
                "agentSession": {"id": "session-1"},
                "data": {"issue": {"id": "issue-1", "team": {"id": "team-1"}}},
            },
            "linear_agent_required": True,
        }
        content = """Nope.

LINEAR_MUTATIONS:
```json
[{"operation":"issueUpdate","id":"issue-2","input":{"title":"Escape"}}]
```"""

        result = await adapter._deliver_linear_agent_activity(content, delivery)

        assert result.success is False
        assert "outside assigned issue" in result.error
        adapter._post_linear_graphql.assert_not_called()

    def test_extract_linear_final_response_prefers_marker(self):
        adapter = _make_adapter()
        raw = "tool noise\nFINAL_LINEAR_RESPONSE:\nDone cleanly.\n"
        assert adapter._extract_linear_final_response(raw) == "Done cleanly."

    def test_extract_linear_final_response_strips_tool_transcript_and_balances_fence(self):
        adapter = _make_adapter()
        raw = "Running tool...\n```diff\n- bad\n+ noisy\n"
        result = adapter._extract_linear_final_response(raw)
        assert "```diff" not in result
        assert len(result) <= 4000

    def test_build_linear_oauth_authorization_url_actor_app(self):
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "client_id": "client-123",
                        "redirect_base_url": "https://example.test",
                    },
                },
            )
        )
        url = adapter._build_linear_oauth_authorization_url("state-abc")
        assert url.startswith("https://linear.app/oauth/authorize?")
        assert "client_id=client-123" in url
        assert "redirect_uri=https%3A%2F%2Fexample.test%2Flinear%2Foauth%2Fcallback" in url
        assert "actor=app" in url
        assert "app%3Aassignable" in url
        assert "app%3Amentionable" in url

    def test_build_linear_oauth_authorization_url_for_named_pi_agent(self):
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "redirect_base_url": "https://example.test",
                        "agents": {
                            "pi": {
                                "client_id": "pi-client",
                                "redirect_base_url": "https://pi.example.test/",
                            }
                        },
                    },
                },
            )
        )
        url = adapter._build_linear_oauth_authorization_url("state-pi", "pi")
        assert "client_id=pi-client" in url
        assert "redirect_uri=https%3A%2F%2Fpi.example.test%2Flinear%2Foauth%2Fpi%2Fcallback" in url
        assert "state=state-pi" in url

    def test_linear_oauth_token_and_state_paths_are_agent_specific(self):
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "token_path": "/tmp/jarvis-token.json",
                        "state_path": "/tmp/jarvis-state.json",
                        "agents": {
                            "pi": {
                                "token_path": "/tmp/pi-token.json",
                                "state_path": "/tmp/pi-state.json",
                            }
                        },
                    },
                },
            )
        )
        assert str(adapter._linear_oauth_token_path()) == "/tmp/jarvis-token.json"
        assert str(adapter._linear_oauth_state_path()) == "/tmp/jarvis-state.json"
        assert str(adapter._linear_oauth_token_path("pi")) == "/tmp/pi-token.json"
        assert str(adapter._linear_oauth_state_path("pi")) == "/tmp/pi-state.json"

    def test_named_agent_does_not_inherit_sensitive_legacy_oauth_fields(self):
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "client_id": "jarvis-client",
                        "client_secret": "jarvis-secret",
                        "token_path": "/tmp/jarvis-token.json",
                        "state_path": "/tmp/jarvis-state.json",
                        "redirect_base_url": "https://shared.example.test",
                        "agents": {"pi": {}},
                    },
                },
            )
        )
        assert adapter._linear_oauth_client_id("pi") == ""
        assert adapter._linear_oauth_client_secret("pi") == ""
        assert str(adapter._linear_oauth_token_path("pi")) != "/tmp/jarvis-token.json"
        assert str(adapter._linear_oauth_state_path("pi")) != "/tmp/jarvis-state.json"
        assert adapter._linear_oauth_redirect_base_url("pi") == "https://shared.example.test"

    def test_load_linear_agent_token_supports_access_token_env(self, monkeypatch):
        monkeypatch.setenv("LINEAR_PI_ACCESS_TOKEN", "pi-token")
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "agents": {
                            "pi": {
                                "access_token_env": "env:LINEAR_PI_ACCESS_TOKEN",
                            }
                        },
                    },
                },
            )
        )
        assert adapter._load_linear_agent_token("pi") == "pi-token"

    @pytest.mark.asyncio
    async def test_named_linear_oauth_authorize_route_writes_agent_state(self, tmp_path):
        state_path = tmp_path / "pi-state.json"
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "redirect_base_url": "https://example.test",
                        "agents": {
                            "pi": {
                                "client_id": "pi-client",
                                "state_path": str(state_path),
                            }
                        },
                    },
                },
            )
        )

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/linear/oauth/pi/authorize", allow_redirects=False)

        assert resp.status == 302
        assert "client_id=pi-client" in resp.headers["Location"]
        assert "redirect_uri=https://example.test/linear/oauth/pi/callback" in resp.headers["Location"]
        stored = json.loads(state_path.read_text(encoding="utf-8"))
        assert stored["agent"] == "pi"
        assert stored["state"]

    @pytest.mark.asyncio
    async def test_named_linear_oauth_callback_rejects_state_agent_mismatch(self, tmp_path):
        state_path = tmp_path / "pi-state.json"
        state_path.write_text(json.dumps({"state": "state-1", "agent": "jarvis"}), encoding="utf-8")
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "redirect_base_url": "https://example.test",
                        "agents": {
                            "pi": {
                                "client_id": "pi-client",
                                "client_secret": "pi-secret",
                                "state_path": str(state_path),
                            }
                        },
                    },
                },
            )
        )

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/linear/oauth/pi/callback?code=code-1&state=state-1")
            body = await resp.json()

        assert resp.status == 400
        assert body["error"] == "OAuth state agent mismatch"

    @pytest.mark.asyncio
    async def test_named_linear_oauth_callback_writes_agent_token(self, tmp_path):
        state_path = tmp_path / "pi-state.json"
        token_path = tmp_path / "pi-token.json"
        state_path.write_text(json.dumps({"state": "state-1", "agent": "pi"}), encoding="utf-8")
        adapter = WebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "routes": {},
                    "linear_oauth": {
                        "redirect_base_url": "https://example.test",
                        "agents": {
                            "pi": {
                                "client_id": "pi-client",
                                "client_secret": "pi-secret",
                                "state_path": str(state_path),
                                "token_path": str(token_path),
                            }
                        },
                    },
                },
            )
        )
        response = MagicMock()
        response.read.return_value = json.dumps({"access_token": "pi-token"}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = None

        with patch("gateway.platforms.webhook.urllib.request.urlopen", return_value=response):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.get("/linear/oauth/pi/callback?code=code-1&state=state-1")
                body = await resp.json()

        assert resp.status == 200
        assert body["agent"] == "pi"
        assert json.loads(token_path.read_text(encoding="utf-8"))["access_token"] == "pi-token"
        assert not state_path.exists()

    @pytest.mark.asyncio
    async def test_required_linear_agent_ignores_inline_deliver_extra_token(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_load_linear_agent_token", return_value="pi-token") as load_token, patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "activity-id")
        ) as post:
            result = await adapter._deliver_linear_agent_activity(
                "Pi done.",
                {
                    "deliver_extra": {"api_key": "jarvis-inline-token"},
                    "payload": {"agentSession": {"id": "session-pi"}},
                    "linear_agent": "pi",
                    "linear_agent_required": True,
                },
            )
        assert result.success is True
        load_token.assert_called_once_with("pi")
        _, token, _ = post.call_args.args
        assert token == "pi-token"

    @pytest.mark.asyncio
    async def test_explicit_default_linear_agent_does_not_fallback_to_linear_api_key(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("LINEAR_API_KEY", "jarvis-user-token")
        with patch.object(adapter, "_load_linear_agent_token", return_value=""), patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "should-not-post")
        ) as post:
            result = await adapter._deliver_linear_agent_activity(
                "Jarvis done.",
                {
                    "deliver_extra": {},
                    "payload": {"agentSession": {"id": "session-jarvis"}},
                    "linear_agent": "jarvis",
                },
            )
        assert result.success is False
        assert result.error == "Missing Linear app token for agent jarvis"
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_posts_response_activity(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_load_linear_agent_token", return_value="app-token"), patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "activity-id")
        ) as post:
            result = await adapter._deliver_linear_agent_activity(
                "Done.",
                {
                    "deliver_extra": {},
                    "payload": {"agentSession": {"id": "session-123"}},
                },
            )
        assert result.success is True
        payload, token, operation = post.call_args.args
        assert token == "app-token"
        assert operation == "agent-activity"
        activity_input = payload["variables"]["input"]
        assert activity_input["agentSessionId"] == "session-123"
        assert activity_input["content"] == {"type": "response", "body": "Done."}

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_requires_session_id(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_load_linear_agent_token", return_value="app-token"):
            result = await adapter._deliver_linear_agent_activity(
                "Done.", {"deliver_extra": {}, "payload": {}}
            )
        assert result.success is False
        assert "agent_session_id" in result.error

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_uses_named_pi_token(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_load_linear_agent_token", return_value="pi-token") as load_token, patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "activity-id")
        ) as post:
            result = await adapter._deliver_linear_agent_activity(
                "Pi done.",
                {
                    "deliver_extra": {},
                    "payload": {"agentSession": {"id": "session-pi"}},
                    "linear_agent": "pi",
                    "linear_agent_required": True,
                },
            )
        assert result.success is True
        load_token.assert_called_once_with("pi")
        _, token, _ = post.call_args.args
        assert token == "pi-token"

    @pytest.mark.asyncio
    async def test_deliver_linear_agent_activity_pi_missing_token_does_not_fallback_to_linear_api_key(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("LINEAR_API_KEY", "jarvis-user-token")
        with patch.object(adapter, "_load_linear_agent_token", return_value=""), patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "should-not-post")
        ) as post:
            result = await adapter._deliver_linear_agent_activity(
                "Pi done.",
                {
                    "deliver_extra": {},
                    "payload": {"agentSession": {"id": "session-pi"}},
                    "linear_agent": "pi",
                    "linear_agent_required": True,
                },
            )
        assert result.success is False
        assert result.error == "Missing Linear app token for agent pi"
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_activity_uses_named_pi_token(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_load_linear_agent_token", return_value="pi-token") as load_token, patch.object(
            adapter, "_post_linear_graphql", return_value=(True, "activity-id")
        ) as post:
            await adapter._post_linear_agent_status_activity(
                {
                    "deliver_extra": {},
                    "payload": {"agentSession": {"id": "session-pi"}},
                    "linear_agent": "pi",
                    "linear_agent_required": True,
                },
                "Pi is checking whether the local worker is available.",
            )
        load_token.assert_called_once_with("pi")
        payload, token, operation = post.call_args.args
        assert token == "pi-token"
        assert operation == "agent-status"
        assert payload["variables"]["input"]["content"] == {
            "type": "thought",
            "body": "Pi is checking whether the local worker is available.",
        }

    @pytest.mark.asyncio
    async def test_offline_linear_agent_route_does_not_post_late_status_activity(self):
        """Offline scaffold routes should emit only the terminal response activity."""
        routes = {
            "linear-pi-agent": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["AgentSessionEvent"],
                "deliver": "linear_agent_activity",
                "linear_agent": "pi",
                "linear_agent_required": True,
                "linear_agent_start_message": "Pi is checking whether the local worker is available.",
                "linear_agent_offline_message": "Pi is offline.",
            }
        }
        adapter = _make_adapter(routes=routes)
        app = _create_app(adapter)

        with patch.object(
            adapter,
            "_deliver_linear_agent_activity",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as deliver, patch.object(
            adapter,
            "_post_linear_agent_status_activity",
            new=AsyncMock(),
        ) as status:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/webhooks/linear-pi-agent",
                    json={
                        "type": "AgentSessionEvent",
                        "action": "created",
                        "agentSession": {"id": "session-pi"},
                    },
                )
                assert resp.status == 202
                await asyncio.sleep(0)

        deliver.assert_awaited_once()
        status.assert_not_awaited()

    def test_linear_profile_command_uses_safe_argv(self):
        adapter = _make_adapter()
        argv = adapter._linear_profile_command(
            {
                "linear_agent_profile_command": "python -m hermes_cli.main",
                "linear_agent_profile_args": ["--pass-session-id"],
            },
            "pi",
            "Do the work",
        )
        assert argv == [
            "python",
            "-m",
            "hermes_cli.main",
            "-p",
            "pi",
            "--pass-session-id",
            "chat",
            "-Q",
            "-q",
            "Do the work",
        ]

    def test_linear_profile_command_for_pi_keeps_profile_rules_loaded_by_default(self):
        adapter = _make_adapter()
        argv = adapter._linear_profile_command(
            {"linear_agent_profile_args": []},
            "pi",
            "Do the work",
        )
        assert argv[:3] == ["hermes", "-p", "pi"]
        assert "--ignore-rules" not in argv

    @pytest.mark.asyncio
    async def test_linear_agent_profile_route_runs_pi_profile_and_delivers_output(self):
        routes = {
            "linear-pi-agent": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["AgentSessionEvent"],
                "prompt": "Handle Linear session {agentSession.id}",
                "deliver": "linear_agent_activity",
                "linear_agent": "pi",
                "linear_agent_required": True,
                "linear_agent_profile": "pi",
                "linear_agent_start_message": "Pi is starting.",
                "linear_agent_profile_stream_status": False,
            }
        }
        adapter = _make_adapter(routes=routes)
        process = _FakeStreamProcess(stdout="Pi finished.\n")

        with patch("gateway.platforms.webhook.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as run, patch.object(
            adapter,
            "_post_linear_agent_status_activity",
            new=AsyncMock(),
        ) as status, patch.object(
            adapter,
            "_deliver_linear_agent_activity",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as deliver, patch.object(adapter, "handle_message", new=AsyncMock()) as handle:
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/linear-pi-agent",
                    json={
                        "type": "AgentSessionEvent",
                        "action": "created",
                        "agentSession": {"id": "session-pi"},
                    },
                )
                body = await resp.json()
                assert resp.status == 202
                assert body["profile"] == "pi"
                for _ in range(20):
                    if deliver.await_count:
                        break
                    await asyncio.sleep(0.01)

        handle.assert_not_awaited()
        status.assert_awaited_once()
        deliver.assert_awaited_once()
        assert deliver.await_args.args[0] == "Pi finished."
        argv = list(run.call_args.args)
        assert argv[:3] == ["hermes", "-p", "pi"]
        assert argv[-3:-1] == ["-Q", "-q"]
        assert argv[-1].startswith("Handle Linear session session-pi")
        assert "Linear AgentSession bounds:" in argv[-1]
        assert "Mode: groom" in argv[-1]

    @pytest.mark.asyncio
    async def test_linear_agent_profile_streams_throttled_progress_activity(self):
        adapter = _make_adapter()
        process = _FakeStreamProcess(stdout="step one\nstep two\nfinal answer\n")
        route_config = {
            "linear_agent_profile": "pi",
            "linear_agent_profile_status_interval": 0,
        }
        delivery = {
            "deliver_extra": {},
            "payload": {"agentSession": {"id": "session-pi"}},
            "linear_agent": "pi",
            "linear_agent_required": True,
        }
        with patch("gateway.platforms.webhook.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), patch.object(
            adapter,
            "_post_linear_agent_status_activity",
            new=AsyncMock(),
        ) as status, patch.object(
            adapter,
            "_deliver_linear_agent_activity",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as deliver:
            await adapter._process_linear_agent_profile(route_config, delivery, "prompt", "pi")

        assert status.await_count >= 1
        assert "pi progress (stdout)" in status.await_args_list[0].args[1]
        deliver.assert_awaited_once()
        assert deliver.await_args.args[0] == "step one\nstep two\nfinal answer"

    @pytest.mark.asyncio
    async def test_linear_agent_profile_failure_posts_error_activity(self):
        adapter = _make_adapter()
        process = _FakeStreamProcess(stderr="boom\n", returncode=2)
        route_config = {
            "linear_agent_profile": "pi",
            "linear_agent_profile_error_message": "Pi failed safely.",
        }
        delivery = {
            "deliver_extra": {},
            "payload": {"agentSession": {"id": "session-pi"}},
            "linear_agent": "pi",
            "linear_agent_required": True,
        }
        with patch("gateway.platforms.webhook.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), patch.object(
            adapter,
            "_deliver_linear_agent_activity",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as deliver:
            await adapter._process_linear_agent_profile(route_config, delivery, "prompt", "pi")

        deliver.assert_awaited_once()
        assert deliver.await_args.args[0] == "Pi failed safely."
