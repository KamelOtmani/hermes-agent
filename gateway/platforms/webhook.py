"""Generic webhook platform adapter.

Runs an aiohttp HTTP server that receives webhook POSTs from external
services (GitHub, GitLab, JIRA, Stripe, etc.), validates HMAC signatures,
transforms payloads into agent prompts, and routes responses back to the
source or to another configured platform.

Configuration lives in config.yaml under platforms.webhook.extra.routes.
Each route defines:
  - events: which event types to accept (header-based filtering)
  - secret: HMAC secret for signature validation (REQUIRED)
  - prompt: template string formatted with the webhook payload
  - skills: optional list of skills to load for the agent
  - deliver: where to send the response (github_comment, telegram, etc.)
  - deliver_extra: additional delivery config (repo, pr_number, chat_id)
  - deliver_only: if true, skip the agent — the rendered prompt IS the
    message that gets delivered.  Use for external push notifications
    (Supabase, monitoring alerts, inter-agent pings) where zero LLM cost
    and sub-second delivery matter more than agent reasoning.

Security:
  - HMAC secret is required per route (validated at startup)
  - Rate limiting per route (fixed-window, configurable)
  - Idempotency cache prevents duplicate agent runs on webhook retries
  - Body size limits checked before reading payload
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import pathlib
import re
import secrets
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"

# Hostnames/IP literals that only serve connections originating on the same
# machine. Anything else is treated as a public bind for safety-rail purposes.
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "ip6-localhost",
    "ip6-loopback",
})


def _is_loopback_host(host: str) -> bool:
    """True when `host` binds only to the local machine.

    Covers IPv4 loopback, the standard `localhost` alias, IPv6 loopback in
    both bracketed and bare form, and the common Debian-style aliases. Any
    falsy value (empty string, None) is conservatively treated as non-loopback
    because an unset host usually means the platform-default public bind.
    """
    if not host:
        return False
    return host.strip().lower() in _LOOPBACK_HOSTS


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = config.extra.get("secret", "")
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._linear_oauth: Dict[str, Any] = config.extra.get("linear_oauth", {}) or {}
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id (status messages
        # AND the final response).  Cleaned up via TTL on each POST so the
        # dict stays bounded — see _prune_delivery_info().  Do NOT pop on
        # send(), or interim status messages (e.g. fallback notifications,
        # context-pressure warnings) will consume the entry before the
        # final response arrives, causing the response to silently fall
        # back to the "log" deliver type.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, List[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        # Load agent-created subscriptions before validating
        self._reload_dynamic_routes()

        # Validate routes at startup — secret is required per route
        for name, route in self._routes.items():
            secret = self._resolve_config_secret(route.get("secret", self._global_secret))
            if not secret:
                raise ValueError(
                    f"[webhook] Route '{name}' has no HMAC secret. "
                    f"Set 'secret' on the route or globally. "
                    f"For testing without auth, set secret to '{_INSECURE_NO_AUTH}'."
                )

            # Safety rail: refuse to start if INSECURE_NO_AUTH is combined with a
            # non-loopback bind. The escape hatch is for local testing only;
            # serving an unauthenticated route on a public interface is a
            # deployment-grade footgun we'd rather crash early than ship.
            if secret == _INSECURE_NO_AUTH and not _is_loopback_host(self._host):
                raise ValueError(
                    f"[webhook] Route '{name}' uses INSECURE_NO_AUTH secret "
                    f"but is bound to non-loopback host '{self._host}'. "
                    f"INSECURE_NO_AUTH is for local testing only. "
                    f"Refusing to start to prevent accidental exposure."
                )
            # deliver_only routes bypass the agent — the POST body becomes a
            # direct push notification via the configured delivery target.
            # Validate up-front so misconfiguration surfaces at startup rather
            # than on the first webhook POST.
            if route.get("deliver_only"):
                deliver = route.get("deliver", "log")
                if not deliver or deliver == "log":
                    raise ValueError(
                        f"[webhook] Route '{name}' has deliver_only=true but "
                        f"deliver is '{deliver}'. Direct delivery requires a "
                        f"real target (telegram, discord, slack, github_comment, etc.)."
                    )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        # Backwards-compatible singleton Linear OAuth endpoints use the default
        # (Jarvis) app identity.  Agent-specific endpoints isolate Pi/Jarvis
        # client credentials, callback URLs, state files, and token stores.
        app.router.add_get("/linear/oauth/authorize", self._handle_linear_oauth_authorize)
        app.router.add_get("/linear/oauth/callback", self._handle_linear_oauth_callback)
        app.router.add_get("/linear/oauth/{agent}/authorize", self._handle_linear_oauth_authorize)
        app.router.add_get("/linear/oauth/{agent}/callback", self._handle_linear_oauth_callback)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)

        # Port conflict detection — fail fast if port is already in use
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(('127.0.0.1', self._port))
            logger.error('[webhook] Port %d already in use. Set a different port in config.yaml: platforms.webhook.port', self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host,
            self._port,
            route_names,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[webhook] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's response to the configured destination.

        chat_id is ``webhook:{route}:{delivery_id}``.  The delivery info
        stored during webhook receipt is read with ``.get()`` (not popped)
        so that interim status messages emitted before the final response
        — fallback-model notifications, context-pressure warnings, etc. —
        do not consume the entry and silently downgrade the final response
        to the ``log`` deliver type.  TTL cleanup happens on POST.
        """
        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        if deliver_type == "linear_comment":
            return await self._deliver_linear_comment(content, delivery)

        if deliver_type == "linear_agent_activity":
            return await self._deliver_linear_agent_activity(content, delivery)

        # Cross-platform delivery — any platform with a gateway adapter.
        # Check both built-in names and plugin-registered platforms.
        _BUILTIN_DELIVER_PLATFORMS = {
            "telegram", "discord", "slack", "signal", "sms", "whatsapp",
            "matrix", "mattermost", "homeassistant", "email", "dingtalk",
            "feishu", "wecom", "wecom_callback", "weixin", "bluebubbles",
            "qqbot", "yuanbao",
        }
        _is_known_platform = deliver_type in _BUILTIN_DELIVER_PLATFORMS
        if not _is_known_platform:
            try:
                from gateway.platform_registry import platform_registry
                _is_known_platform = platform_registry.is_registered(deliver_type)
            except Exception:
                pass
        if self.gateway_runner and _is_known_platform:
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("[webhook] Unknown deliver type: %s", deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than the idempotency TTL.

        Mirrors the cleanup pattern used for ``_seen_deliveries``.  Called
        on each POST so the dict size is bounded by ``rate_limit * TTL``
        even if many webhooks fire and never receive a final response.
        """
        cutoff = now - self._idempotency_ttl
        stale = [
            k
            for k, t in self._delivery_info_created.items()
            if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

    def _linear_default_agent(self) -> str:
        return str(self._linear_oauth.get("default_agent") or "jarvis").strip().lower() or "jarvis"

    def _linear_agent_key(self, agent: Optional[str] = None) -> str:
        key = str(agent or self._linear_default_agent()).strip().lower()
        if not key:
            return self._linear_default_agent()
        return re.sub(r"[^a-z0-9_-]", "-", key)

    def _linear_agent_config(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """Return Linear OAuth config for an app identity.

        The default agent keeps the legacy singleton Linear OAuth keys for
        backwards compatibility. Named non-default agents intentionally do not
        inherit sensitive singleton fields such as client credentials or token
        paths; otherwise a partially-configured Pi route could silently respond
        with Jarvis credentials. Only harmless shared defaults are inherited.
        """
        key = self._linear_agent_key(agent)
        default_agent = self._linear_default_agent()
        singleton = {k: v for k, v in self._linear_oauth.items() if k != "agents"}
        if key == default_agent:
            base = dict(singleton)
        else:
            base = {
                k: singleton[k]
                for k in ("redirect_base_url", "scopes")
                if k in singleton
            }
        agents = self._linear_oauth.get("agents") or {}
        if isinstance(agents, dict):
            specific = agents.get(key) or agents.get(agent or "") or {}
            if isinstance(specific, dict):
                base.update(specific)
        return base

    def _resolve_config_value(self, value: Any) -> str:
        """Resolve string config values, supporting env:VAR indirection."""
        if value is None:
            return ""
        text = str(value)
        if text.startswith("env:"):
            return os.getenv(text[4:], "")
        return text

    def _path_from_config(self, configured: str) -> pathlib.Path:
        from hermes_constants import get_hermes_home

        return pathlib.Path(configured) if os.path.isabs(configured) else get_hermes_home() / configured

    def _linear_oauth_token_path(self, agent: Optional[str] = None) -> pathlib.Path:
        from hermes_constants import get_hermes_home

        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        configured = self._resolve_config_value(cfg.get("token_path"))
        if not configured and key == self._linear_default_agent():
            configured = os.getenv("LINEAR_AGENT_TOKEN_PATH", "")
        if configured:
            return self._path_from_config(configured)
        if key == self._linear_default_agent():
            return get_hermes_home() / "linear_agent_oauth.json"
        return get_hermes_home() / f"linear_agent_oauth_{key}.json"

    def _linear_oauth_state_path(self, agent: Optional[str] = None) -> pathlib.Path:
        from hermes_constants import get_hermes_home

        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        configured = self._resolve_config_value(cfg.get("state_path"))
        if configured:
            return self._path_from_config(configured)
        if key == self._linear_default_agent():
            return get_hermes_home() / "linear_agent_oauth_state.json"
        return get_hermes_home() / f"linear_agent_oauth_state_{key}.json"

    def _linear_oauth_client_id(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        value = self._resolve_config_value(cfg.get("client_id"))
        if not value and key == self._linear_default_agent():
            value = os.getenv("LINEAR_CLIENT_ID", "")
        return value

    def _linear_oauth_client_secret(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        value = self._resolve_config_value(cfg.get("client_secret"))
        if not value and key == self._linear_default_agent():
            value = os.getenv("LINEAR_CLIENT_SECRET", "")
        return value

    def _linear_oauth_redirect_base_url(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        base = self._resolve_config_value(cfg.get("redirect_base_url"))
        if not base and key == self._linear_default_agent():
            base = os.getenv("LINEAR_REDIRECT_BASE_URL") or os.getenv("WEBHOOK_PUBLIC_URL") or ""
        return str(base).rstrip("/")

    def _linear_oauth_scopes(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        scopes = cfg.get("scopes")
        if scopes is None and key == self._linear_default_agent():
            scopes = os.getenv("LINEAR_AGENT_SCOPES")
        if isinstance(scopes, list):
            return ",".join(str(scope) for scope in scopes)
        if scopes:
            return str(scopes)
        return "read,write,comments:create,app:assignable,app:mentionable"

    def _linear_oauth_redirect_uri(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        base = self._linear_oauth_redirect_base_url(key)
        if not base:
            return ""
        if key == self._linear_default_agent():
            return f"{base}/linear/oauth/callback"
        return f"{base}/linear/oauth/{key}/callback"

    def _load_linear_agent_token(self, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        cfg = self._linear_agent_config(key)
        # `access_token_env` names an environment variable; it is not itself a
        # secret value.  Allow the common `env:VAR` spelling for consistency
        # with other config keys, but strip the prefix before os.getenv().
        configured_env = str(cfg.get("access_token_env") or "")
        env_var = configured_env[4:] if configured_env.startswith("env:") else configured_env
        token = os.getenv(env_var, "") if env_var else ""
        if not token and key == self._linear_default_agent():
            token = os.getenv("LINEAR_AGENT_ACCESS_TOKEN", "")
        if token:
            return token
        path = self._linear_oauth_token_path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(data.get("access_token") or "")

    def _build_linear_oauth_authorization_url(self, state: str, agent: Optional[str] = None) -> str:
        key = self._linear_agent_key(agent)
        client_id = self._linear_oauth_client_id(key)
        redirect_uri = self._linear_oauth_redirect_uri(key)
        query = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": self._linear_oauth_scopes(key),
                "state": state,
                "actor": "app",
            }
        )
        return f"https://linear.app/oauth/authorize?{query}"

    def _linear_agent_from_request(self, request: "web.Request") -> str:
        return self._linear_agent_key(request.match_info.get("agent") or request.query.get("agent"))

    async def _handle_linear_oauth_authorize(self, request: "web.Request") -> "web.Response":
        """Start Linear actor=app install flow for Jarvis/default or a named agent."""
        agent = self._linear_agent_from_request(request)
        if not self._linear_oauth_client_id(agent) or not self._linear_oauth_redirect_uri(agent):
            return web.json_response(
                {
                    "error": f"Linear OAuth is not configured for agent '{agent}'",
                    "required": ["client_id", "redirect_base_url"],
                },
                status=400,
            )
        state = secrets.token_urlsafe(32)
        path = self._linear_oauth_state_path(agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": state, "agent": agent, "created_at": time.time()}), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        raise web.HTTPFound(self._build_linear_oauth_authorization_url(state, agent))

    async def _handle_linear_oauth_callback(self, request: "web.Request") -> "web.Response":
        """Exchange Linear code and store the app token for Jarvis/default or a named agent."""
        agent = self._linear_agent_from_request(request)
        code = request.query.get("code", "")
        state = request.query.get("state", "")
        error = request.query.get("error", "")
        if error:
            return web.json_response({"error": error}, status=400)
        if not code or not state:
            return web.json_response({"error": "Missing code or state"}, status=400)
        try:
            stored = json.loads(self._linear_oauth_state_path(agent).read_text(encoding="utf-8"))
        except Exception:
            return web.json_response({"error": "Missing OAuth state"}, status=400)
        if stored.get("agent") and self._linear_agent_key(stored.get("agent")) != agent:
            return web.json_response({"error": "OAuth state agent mismatch"}, status=400)
        if not hmac.compare_digest(str(stored.get("state", "")), state):
            return web.json_response({"error": "Invalid OAuth state"}, status=400)

        client_id = self._linear_oauth_client_id(agent)
        client_secret = self._linear_oauth_client_secret(agent)
        redirect_uri = self._linear_oauth_redirect_uri(agent)
        if not client_id or not client_secret or not redirect_uri:
            return web.json_response(
                {
                    "error": f"Linear OAuth is not configured for agent '{agent}'",
                    "required": ["client_id", "client_secret", "redirect_base_url"],
                },
                status=400,
            )

        form = urllib.parse.urlencode(
            {
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")

        def _exchange() -> tuple[bool, str]:
            req = urllib.request.Request(
                "https://api.linear.app/oauth/token",
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return True, resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return False, e.read().decode("utf-8", errors="replace")
            except Exception as e:
                return False, str(e)

        ok, body = await asyncio.to_thread(_exchange)
        if not ok:
            return web.json_response({"error": "Token exchange failed", "details": body[:1000]}, status=400)
        try:
            token_data = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid token response"}, status=400)
        if not token_data.get("access_token"):
            return web.json_response({"error": "Token response missing access_token"}, status=400)

        token_path = self._linear_oauth_token_path(agent)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
        try:
            os.chmod(token_path, 0o600)
            self._linear_oauth_state_path(agent).unlink(missing_ok=True)
        except OSError:
            pass
        return web.json_response(
            {
                "status": "ok",
                "agent": agent,
                "message": f"Linear app-user OAuth token stored for agent '{agent}'.",
            }
        )

    def _reload_dynamic_routes(self) -> None:
        """Reload agent-created subscriptions from disk if the file changed."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        subs_path = hermes_home / _DYNAMIC_ROUTES_FILENAME
        if not subs_path.exists():
            if self._dynamic_routes:
                self._dynamic_routes = {}
                self._routes = dict(self._static_routes)
                logger.debug("[webhook] Dynamic subscriptions file removed, cleared dynamic routes")
            return
        try:
            mtime = subs_path.stat().st_mtime
            if mtime <= self._dynamic_routes_mtime:
                return  # No change
            data = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            # Merge: static routes take precedence over dynamic ones
            self._dynamic_routes = {
                k: v for k, v in data.items()
                if k not in self._static_routes
            }
            self._routes = {**self._dynamic_routes, **self._static_routes}
            self._dynamic_routes_mtime = mtime
            logger.info(
                "[webhook] Reloaded %d dynamic route(s): %s",
                len(self._dynamic_routes),
                ", ".join(self._dynamic_routes.keys()) or "(none)",
            )
        except Exception as e:
            logger.error("[webhook] Failed to reload dynamic routes: %s", e)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # ── Auth-before-body ─────────────────────────────────────
        # Check Content-Length before reading the full payload.
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # Read body (must be done before any validation)
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature FIRST (skip for INSECURE_NO_AUTH testing mode)
        secret = self._resolve_config_secret(route_config.get("secret", self._global_secret))
        if secret and secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # ── Rate limiting (after auth) ───────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            # Try form-encoded as fallback
            try:
                import urllib.parse

                payload = dict(
                    urllib.parse.parse_qsl(raw_body.decode("utf-8"))
                )
            except Exception:
                return web.json_response(
                    {"error": "Cannot parse body"}, status=400
                )

        # Check event type filter
        event_type = (
            request.headers.get("X-GitHub-Event", "")
            or request.headers.get("X-GitLab-Event", "")
            or payload.get("event_type", "")
            or payload.get("type", "")
            or "unknown"
        )
        allowed_events = route_config.get("events", [])
        if allowed_events and event_type not in allowed_events:
            logger.debug(
                "[webhook] Ignoring event %s for route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type}
            )

        should_process, ignore_reason = self._should_process_payload(
            route_config, payload
        )
        if not should_process:
            logger.debug(
                "[webhook] Ignoring route %s event %s: %s",
                route_name,
                event_type,
                ignore_reason,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type, "reason": ignore_reason}
            )

        # Format prompt from template
        prompt_template = route_config.get("prompt", "")
        prompt = self._render_prompt(
            prompt_template, payload, event_type, route_name
        )

        # Inject skill content if configured.
        # We call build_skill_invocation_message() directly rather than
        # using /skill-name slash commands — the gateway's command parser
        # would intercept those and break the flow.
        skills = route_config.get("skills", [])
        if skills:
            try:
                from agent.skill_commands import (
                    build_skill_invocation_message,
                    get_skill_commands,
                )

                skill_cmds = get_skill_commands()
                for skill_name in skills:
                    cmd_key = f"/{skill_name}"
                    if cmd_key in skill_cmds:
                        skill_content = build_skill_invocation_message(
                            cmd_key, user_instruction=prompt
                        )
                        if skill_content:
                            prompt = skill_content
                            break  # Load the first matching skill
                    else:
                        logger.warning(
                            "[webhook] Skill '%s' not found", skill_name
                        )
            except Exception as e:
                logger.warning("[webhook] Skill loading failed: %s", e)

        # Build a unique delivery ID
        delivery_id = request.headers.get(
            "X-GitHub-Delivery",
            request.headers.get(
                "X-Linear-Delivery",
                request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
            ),
        )

        # ── Idempotency ─────────────────────────────────────────
        # Skip duplicate deliveries (webhook retries).
        now = time.time()
        # Prune expired entries
        self._seen_deliveries = {
            k: v
            for k, v in self._seen_deliveries.items()
            if now - v < self._idempotency_ttl
        }
        if delivery_id in self._seen_deliveries:
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )
        self._seen_deliveries[delivery_id] = now

        # ── Direct delivery mode (deliver_only) ─────────────────
        # Skip the agent entirely — the rendered prompt IS the message we
        # deliver.  Use case: external services (Supabase, monitoring,
        # cron jobs, other agents) that need to push a plain notification
        # to a user's chat with zero LLM cost.  Reuses the same HMAC auth,
        # rate limiting, idempotency, and template rendering as agent mode.
        if route_config.get("deliver_only"):
            delivery = {
                "deliver": route_config.get("deliver", "log"),
                "deliver_extra": self._render_delivery_extra(
                    route_config.get("deliver_extra", {}), payload
                ),
                "payload": payload,
                "route_name": route_name,
                "linear_agent": route_config.get("linear_agent") or route_config.get("linear_oauth_agent"),
                "linear_agent_required": bool(route_config.get("linear_agent_required")),
            }
            logger.info(
                "[webhook] direct-deliver event=%s route=%s target=%s msg_len=%d delivery=%s",
                event_type,
                route_name,
                delivery["deliver"],
                len(prompt),
                delivery_id,
            )
            try:
                result = await self._direct_deliver(prompt, delivery)
            except Exception:
                logger.exception(
                    "[webhook] direct-deliver failed route=%s delivery=%s",
                    route_name,
                    delivery_id,
                )
                return web.json_response(
                    {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                    status=502,
                )

            if result.success:
                return web.json_response(
                    {
                        "status": "delivered",
                        "route": route_name,
                        "target": delivery["deliver"],
                        "delivery_id": delivery_id,
                    },
                    status=200,
                )
            # Delivery attempted but target rejected it — surface as 502
            # with a generic error (don't leak adapter-level detail).
            logger.warning(
                "[webhook] direct-deliver target rejected route=%s target=%s error=%s",
                route_name,
                delivery["deliver"],
                result.error,
            )
            return web.json_response(
                {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                status=502,
            )

        # Use delivery_id in session key so concurrent webhooks on the
        # same route get independent agent runs (not queued/interrupted).
        session_chat_id = f"webhook:{route_name}:{delivery_id}"

        # Store delivery info for send().  Read by every send() invocation
        # for this chat_id (interim status messages and the final response),
        # so we do NOT pop on send.  TTL-based cleanup keeps the dict bounded.
        deliver_config = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": self._render_delivery_extra(
                route_config.get("deliver_extra", {}), payload
            ),
            "payload": payload,
            "route_name": route_name,
            "linear_agent": route_config.get("linear_agent") or route_config.get("linear_oauth_agent"),
            "linear_agent_required": bool(route_config.get("linear_agent_required")),
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        if deliver_config.get("deliver") == "linear_agent_activity":
            offline_message = route_config.get("linear_agent_offline_message")
            if offline_message:
                result = await self._deliver_linear_agent_activity(
                    str(offline_message), deliver_config
                )
                status = 202 if result.success else 502
                return web.json_response(
                    {
                        "status": "offline" if result.success else "error",
                        "route": route_name,
                        "event": event_type,
                        "delivery_id": delivery_id,
                        "error": result.error if not result.success else None,
                    },
                    status=status,
                )

            linear_profile = str(
                route_config.get("linear_agent_profile")
                or route_config.get("hermes_profile")
                or ""
            ).strip()
            if linear_profile:
                profile_task = asyncio.create_task(
                    self._process_linear_agent_profile(
                        route_config,
                        deliver_config,
                        prompt,
                        linear_profile,
                    )
                )
                self._background_tasks.add(profile_task)
                profile_task.add_done_callback(self._background_tasks.discard)
                return web.json_response(
                    {
                        "status": "accepted",
                        "route": route_name,
                        "event": event_type,
                        "delivery_id": delivery_id,
                        "profile": linear_profile,
                    },
                    status=202,
                )

            status_task = asyncio.create_task(
                self._post_linear_agent_status_activity(
                    deliver_config,
                    route_config.get("linear_agent_start_message", "Jarvis is working on this."),
                )
            )
            self._background_tasks.add(status_task)
            status_task.add_done_callback(self._background_tasks.discard)

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )
        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
        )

        logger.info(
            "[webhook] %s event=%s route=%s prompt_len=%d delivery=%s",
            request.method,
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        # Non-blocking — return 202 Accepted immediately
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Linear AgentSession profile worker dispatch
    # ------------------------------------------------------------------

    def _linear_profile_command(self, route_config: dict, profile: str, prompt: str) -> List[str]:
        """Build a safe argv for running a Linear AgentSession through a Hermes profile."""
        configured_command = str(route_config.get("linear_agent_profile_command") or "hermes").strip()
        command = shlex.split(configured_command) if configured_command else ["hermes"]
        extra_args = route_config.get("linear_agent_profile_args") or []
        if isinstance(extra_args, str):
            extra_args = shlex.split(extra_args)
        return [
            *command,
            "-p",
            profile,
            *(str(arg) for arg in extra_args),
            "chat",
            "-Q",
            "-q",
            prompt,
        ]

    async def _process_linear_agent_profile(
        self,
        route_config: dict,
        delivery: dict,
        prompt: str,
        profile: str,
    ) -> None:
        """Run a configured Hermes profile and post its output as a Linear Agent Activity.

        This is intentionally route-scoped and fail-closed: a Pi AgentSession route
        can run the Pi profile, but if the subprocess fails we post a terminal
        error activity rather than falling back to the gateway's default Jarvis
        session or VPS-local handler.
        """
        start_message = route_config.get("linear_agent_start_message")
        if start_message:
            await self._post_linear_agent_status_activity(delivery, str(start_message))

        timeout = int(route_config.get("linear_agent_profile_timeout") or 900)
        argv = self._linear_profile_command(route_config, profile, prompt)

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        try:
            completed = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            logger.error(
                "[webhook] linear profile %s timed out after %ss",
                profile,
                timeout,
            )
            await self._deliver_linear_agent_activity(
                str(
                    route_config.get("linear_agent_profile_timeout_message")
                    or f"{profile} timed out while processing this Linear AgentSession."
                ),
                delivery,
            )
            return
        except Exception as e:
            logger.error(
                "[webhook] linear profile %s failed to start: %s",
                profile,
                e,
                exc_info=True,
            )
            await self._deliver_linear_agent_activity(
                str(
                    route_config.get("linear_agent_profile_error_message")
                    or f"{profile} could not start for this Linear AgentSession."
                ),
                delivery,
            )
            return

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            logger.error(
                "[webhook] linear profile %s exited %s: %s",
                profile,
                completed.returncode,
                stderr[-2000:],
            )
            await self._deliver_linear_agent_activity(
                str(
                    route_config.get("linear_agent_profile_error_message")
                    or f"{profile} failed while processing this Linear AgentSession. Check Hermes gateway logs."
                ),
                delivery,
            )
            return

        await self._deliver_linear_agent_activity(
            stdout or str(route_config.get("linear_agent_profile_empty_message") or f"{profile} completed with no output."),
            delivery,
        )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _resolve_config_secret(self, value: Any) -> str:
        """Resolve route secret values, supporting env:VAR indirection."""
        if value is None:
            return ""
        text = str(value)
        if text.startswith("env:"):
            return os.getenv(text[4:], "")
        return text

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (Linear, GitHub, GitLab, generic HMAC-SHA256)."""
        # Linear: linear-signature = <hex HMAC-SHA256>
        linear_sig = request.headers.get("linear-signature", "")
        if linear_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(linear_sig, expected)

        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return hmac.compare_digest(gl_token, secret)

        # Generic: X-Webhook-Signature = <hex HMAC-SHA256>
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    # ------------------------------------------------------------------
    # Payload filtering
    # ------------------------------------------------------------------

    def _payload_lookup(self, payload: dict, path: str) -> Any:
        """Resolve a dot-notation path inside a webhook payload."""
        value: Any = payload
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _matches_payload_filter(self, payload: dict, rule: dict) -> bool:
        """Return True when a payload satisfies a simple route filter rule."""
        path = str(rule.get("path", ""))
        if not path:
            return False
        value = self._payload_lookup(payload, path)
        text = "" if value is None else str(value)

        if "exists" in rule:
            return (value is not None) is bool(rule.get("exists"))
        if "eq" in rule:
            return text == str(rule.get("eq"))
        if "contains" in rule:
            return str(rule.get("contains")) in text
        if "contains_ignore_case" in rule:
            return str(rule.get("contains_ignore_case")).lower() in text.lower()
        if "regex" in rule:
            try:
                return re.search(str(rule.get("regex")), text) is not None
            except re.error as e:
                logger.warning("[webhook] Invalid payload filter regex %r: %s", rule.get("regex"), e)
                return False
        if "in" in rule:
            options = rule.get("in") or []
            return text in {str(option) for option in options}
        return bool(value)

    def _should_process_payload(self, route_config: dict, payload: dict) -> tuple[bool, str]:
        """Apply optional route-level payload filters before running the agent.

        Supported config:
          filters:
            require: [{path: "data.body", regex: "(?i)\\bjarvis\\b"}]
            reject: [{path: "data.body", contains: "Jarvis automated reply"}]
        """
        filters = route_config.get("filters") or {}
        if not isinstance(filters, dict):
            return True, "no filters"

        for rule in filters.get("reject") or []:
            if isinstance(rule, dict) and self._matches_payload_filter(payload, rule):
                return False, f"rejected by filter on {rule.get('path', '?')}"

        for rule in filters.get("require") or []:
            if not isinstance(rule, dict) or not self._matches_payload_filter(payload, rule):
                path = rule.get("path", "?") if isinstance(rule, dict) else "?"
                return False, f"required filter not matched on {path}"

        return True, "matched"

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template with the webhook payload.

        Supports dot-notation access into nested dicts:
        ``{pull_request.title}`` → ``payload["pull_request"]["title"]``

        Special token ``{__raw__}`` dumps the entire payload as indented
        JSON (truncated to 4000 chars).  Useful for monitoring alerts or
        any webhook where the agent needs to see the full payload.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"Webhook event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: re.Match) -> str:
            key = match.group(1)
            # Special token: dump the entire payload as JSON
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _render_delivery_extra(
        self, extra: dict, payload: dict
    ) -> dict:
        """Render delivery_extra template values with payload data."""
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                rendered[key] = self._render_prompt(value, payload, "", "")
            else:
                rendered[key] = value
        return rendered

    # ------------------------------------------------------------------
    # Response delivery
    # ------------------------------------------------------------------

    async def _direct_deliver(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Deliver *content* directly without invoking the agent.

        Used by ``deliver_only`` routes: the rendered template becomes the
        literal message body, and we dispatch to the same delivery helpers
        that the agent-mode ``send()`` flow uses.  All target types that
        work in agent mode work here — Telegram, Discord, Slack, GitHub
        PR comments, etc.
        """
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            # Shouldn't reach here — startup validation rejects deliver_only
            # with deliver=log — but guard defensively.
            logger.info("[webhook] direct-deliver log-only: %s", content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        if deliver_type == "linear_comment":
            return await self._deliver_linear_comment(content, delivery)

        if deliver_type == "linear_agent_activity":
            return await self._deliver_linear_agent_activity(content, delivery)

        # Fall through to the cross-platform dispatcher, which validates the
        # target name and routes via the gateway runner.
        return await self._deliver_cross_platform(
            deliver_type, content, delivery
        )

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post agent response as a GitHub PR/issue comment via ``gh`` CLI."""
        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")

        if not repo or not pr_number:
            logger.error(
                "[webhook] github_comment delivery missing repo or pr_number"
            )
            return SendResult(
                success=False, error="Missing repo or pr_number"
            )

        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "[webhook] Posted comment on %s#%s", repo, pr_number
                )
                return SendResult(success=True)
            else:
                logger.error(
                    "[webhook] gh pr comment failed: %s", result.stderr
                )
                return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "[webhook] 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery"
            )
            return SendResult(
                success=False, error="gh CLI not installed"
            )
        except Exception as e:
            logger.error("[webhook] github_comment delivery error: %s", e)
            return SendResult(success=False, error=str(e))

    async def _deliver_linear_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post the agent response as a Linear issue comment via GraphQL."""
        extra = delivery.get("deliver_extra", {})
        issue_id = extra.get("issue_id") or extra.get("issueId")
        api_key = extra.get("api_key") or os.getenv("LINEAR_API_KEY", "")

        if not issue_id:
            logger.error("[webhook] linear_comment delivery missing issue_id")
            return SendResult(success=False, error="Missing issue_id")
        if not api_key:
            logger.error("[webhook] linear_comment delivery missing LINEAR_API_KEY")
            return SendResult(success=False, error="Missing LINEAR_API_KEY")

        query = (
            "mutation($input: CommentCreateInput!) { "
            "commentCreate(input: $input) { success comment { id url } } }"
        )
        loop_marker = extra.get("loop_marker", "Jarvis automated reply")
        body_content = content
        if loop_marker and loop_marker not in body_content:
            body_content = f"{body_content}\n\n<sub>{loop_marker}</sub>"
        payload = {
            "query": query,
            "variables": {"input": {"issueId": issue_id, "body": body_content}},
        }

        def _post() -> tuple[bool, str]:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://api.linear.app/graphql",
                data=body,
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "hermes-agent-webhook-linear/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    response_body = resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                response_body = e.read().decode("utf-8", errors="replace")
                return False, f"HTTP {e.code}: {response_body}"
            except Exception as e:
                return False, str(e)

            try:
                data = json.loads(response_body)
            except json.JSONDecodeError:
                return False, f"Invalid JSON response: {response_body[:200]}"
            if data.get("errors"):
                return False, json.dumps(data["errors"])[:1000]
            result = data.get("data", {}).get("commentCreate", {})
            if result.get("success"):
                comment = result.get("comment") or {}
                logger.info(
                    "[webhook] Posted Linear comment on issue %s: %s",
                    issue_id,
                    comment.get("url") or comment.get("id") or "(no URL)",
                )
                return True, comment.get("url") or comment.get("id") or "ok"
            return False, response_body[:1000]

        ok, message = await asyncio.to_thread(_post)
        if ok:
            return SendResult(success=True)
        logger.error("[webhook] linear_comment delivery failed: %s", message)
        return SendResult(success=False, error=message)

    def _linear_delivery_agent(self, delivery: dict) -> str:
        return self._linear_agent_key(delivery.get("linear_agent"))

    def _linear_delivery_requires_app_token(self, delivery: dict) -> bool:
        agent = self._linear_delivery_agent(delivery)
        return (
            bool(delivery.get("linear_agent_required"))
            or bool(delivery.get("linear_agent"))
            or agent != self._linear_default_agent()
        )

    def _linear_delivery_access_token(self, delivery: dict) -> str:
        extra = delivery.get("deliver_extra", {})
        explicit = extra.get("access_token") or extra.get("api_key")
        requires_app_token = self._linear_delivery_requires_app_token(delivery)
        if explicit and not requires_app_token:
            return str(explicit)
        agent = self._linear_delivery_agent(delivery)
        token = self._load_linear_agent_token(agent)
        if token:
            return token
        # Legacy Jarvis/default behavior may still fall back to LINEAR_API_KEY
        # only when the route did not explicitly select a first-class agent.
        if not requires_app_token:
            return os.getenv("LINEAR_API_KEY", "")
        return ""

    async def _post_linear_agent_status_activity(self, delivery: dict, body: str) -> None:
        """Best-effort early AgentSession activity so Linear doesn't mark it unresponsive."""
        extra = delivery.get("deliver_extra", {})
        payload_data = delivery.get("payload", {}) or {}
        agent_session_id = (
            extra.get("agent_session_id")
            or extra.get("agentSessionId")
            or self._payload_lookup(payload_data, "agentSession.id")
            or self._payload_lookup(payload_data, "agentActivity.agentSessionId")
        )
        access_token = self._linear_delivery_access_token(delivery)
        if not agent_session_id or not access_token:
            return
        payload = {
            "query": (
                "mutation($input: AgentActivityCreateInput!) { "
                "agentActivityCreate(input: $input) { success agentActivity { id } } }"
            ),
            "variables": {
                "input": {
                    "agentSessionId": str(agent_session_id),
                    "content": {"type": "thought", "body": body},
                    "ephemeral": True,
                }
            },
        }
        ok, message = await asyncio.to_thread(
            self._post_linear_graphql, payload, access_token, "agent-status"
        )
        if not ok:
            logger.debug("[webhook] Linear agent status activity failed: %s", message)

    async def _deliver_linear_agent_activity(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post the agent response to a Linear AgentSession as an activity."""
        extra = delivery.get("deliver_extra", {})
        payload_data = delivery.get("payload", {}) or {}
        agent_session_id = (
            extra.get("agent_session_id")
            or extra.get("agentSessionId")
            or self._payload_lookup(payload_data, "agentSession.id")
            or self._payload_lookup(payload_data, "agentActivity.agentSessionId")
        )
        access_token = self._linear_delivery_access_token(delivery)

        if not agent_session_id:
            logger.error("[webhook] linear_agent_activity delivery missing agent_session_id")
            return SendResult(success=False, error="Missing agent_session_id")
        if not access_token:
            agent = self._linear_delivery_agent(delivery)
            logger.error("[webhook] linear_agent_activity delivery missing Linear app token for agent %s", agent)
            return SendResult(success=False, error=f"Missing Linear app token for agent {agent}")

        query = (
            "mutation($input: AgentActivityCreateInput!) { "
            "agentActivityCreate(input: $input) { success agentActivity { id } } }"
        )
        body_content = content
        loop_marker = extra.get("loop_marker")
        if loop_marker and loop_marker not in body_content:
            body_content = f"{body_content}\n\n<sub>{loop_marker}</sub>"
        request_payload = {
            "query": query,
            "variables": {
                "input": {
                    "agentSessionId": str(agent_session_id),
                    "content": {"type": "response", "body": body_content},
                }
            },
        }

        ok, message = await asyncio.to_thread(
            self._post_linear_graphql, request_payload, access_token, "agent-activity"
        )
        if ok:
            logger.info(
                "[webhook] Posted Linear agent activity on session %s: %s",
                agent_session_id,
                message,
            )
            return SendResult(success=True)
        logger.error("[webhook] linear_agent_activity delivery failed: %s", message)
        return SendResult(success=False, error=message)

    def _post_linear_graphql(self, payload: dict, api_key: str, operation: str) -> tuple[bool, str]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=body,
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json",
                "User-Agent": f"hermes-agent-webhook-linear/{operation}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            response_body = e.read().decode("utf-8", errors="replace")
            return False, f"HTTP {e.code}: {response_body}"
        except Exception as e:
            return False, str(e)

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            return False, f"Invalid JSON response: {response_body[:200]}"
        if data.get("errors"):
            return False, json.dumps(data["errors"])[:1000]
        result = data.get("data", {})
        for value in result.values():
            if isinstance(value, dict) and value.get("success"):
                created = value.get("comment") or value.get("agentActivity") or {}
                return True, created.get("url") or created.get("id") or "ok"
        return False, response_body[:1000]

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> SendResult:
        """Route response to another platform (telegram, discord, etc.)."""
        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )

        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )

        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False,
                error=f"Platform {platform_name} not connected",
            )

        # Use home channel if no specific chat_id in deliver_extra
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )

        # Pass thread_id from deliver_extra so Telegram forum topics work
        metadata = None
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}

        return await adapter.send(chat_id, content, metadata=metadata)
