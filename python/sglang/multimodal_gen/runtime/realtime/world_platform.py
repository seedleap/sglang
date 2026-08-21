# SPDX-License-Identifier: Apache-2.0
"""World platform ingress: session-token verification and lifecycle callbacks.

The authorized_generate route used with world-service, the business backend,
works as follows:
  - Credential: a compact Ed25519 JWS. The gateway only holds the public key,
    while world-service remains the sole token authority. Identity sources such
    as Clerk or biz-core can sit in front without changing this module.
  - Session payload: world-service seals the init message with AES-GCM and asks
    the browser to forward it. The gateway unseals it here. The browser never
    sees prompts, timelines, or skill instructions. GCM provides integrity, so a
    one-byte tamper fails without a separate hash binding.
  - Callbacks: started, ended, and aborted callbacks use HMAC signing compatible
    with the biz-core tracking ingest path.

Legacy showcase routes do not reference this module. If world platform settings
are absent, behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac as hmac_mod
import json
import logging
import secrets
import time
from dataclasses import dataclass

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)


class TokenError(Exception):
    """Session-token verification failed; externally surfaced as close code 1008."""


@dataclass(frozen=True)
class Principal:
    """Session principal carried by the token."""

    user_id: str  # per-run pseudonym, not linkable back to the account
    run_id: str
    max_lifetime_s: int
    allow_free_prompt: bool
    jti: str = ""  # one-time nonce to prevent two sessions from one token
    exp: int = 0  # expiry timestamp in seconds, used to age out replay cache


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def load_public_key(pub_b64: str) -> Ed25519PublicKey:
    """Build a verifier from a base64 public key; fail fast during startup."""
    raw = base64.b64decode(pub_b64)
    if len(raw) != 32:
        raise ValueError("Ed25519 public key must be 32 base64-encoded bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def verify_session_token(token: str, public_key: Ed25519PublicKey) -> Principal:
    """Verify the session token and return its principal."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "EdDSA":
            raise TokenError("only EdDSA is supported")
        public_key.verify(
            _b64url_decode(sig_b64),
            f"{header_b64}.{payload_b64}".encode(),
        )
        claims = json.loads(_b64url_decode(payload_b64))
    except TokenError:
        raise
    except (ValueError, KeyError, InvalidSignature, binascii.Error) as exc:
        raise TokenError(f"invalid token: {exc}") from exc

    if claims.get("aud") != "world-session":
        raise TokenError("aud mismatch")
    exp = int(claims.get("exp") or 0)
    if exp and time.time() > exp + 30:  # 30s clock skew tolerance
        raise TokenError("token expired")
    session = claims.get("session") or {}
    max_lifetime = int(session.get("max_lifetime_s") or 0)
    if max_lifetime <= 0:
        raise TokenError("missing session lifetime")
    return Principal(
        user_id=str(claims.get("sub") or ""),
        run_id=str(claims.get("sid") or ""),
        max_lifetime_s=max_lifetime,
        allow_free_prompt=bool(session.get("allow_free_prompt", False)),
        jti=str(claims.get("jti") or ""),
        exp=int(claims.get("exp") or 0),
    )


class SessionPayloadSealer:
    """Unseal session payloads produced by world-service pkg/seal.

    The AES-GCM key is derived from the service-to-service shared secret with
    HKDF-SHA256. The info label separates this key from HMAC callback signing.
    The associated data is the run id, so ciphertext copied to another run fails
    to unseal.
    """

    _HKDF_INFO = b"world-service/session-payload/v1"

    def __init__(self, shared_secret: str) -> None:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        self._key = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=self._HKDF_INFO
        ).derive(shared_secret.encode())

    def open(self, sealed: bytes, run_id: str) -> dict:
        """Unseal and return the init message; failures raise TokenError."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Lazy import keeps this module light and avoids loading the full sglang
        # dependency chain at import time.
        from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
            decode_message,
        )

        nonce_size = 12  # standard AES-GCM nonce length, matching the Go side
        if len(sealed) <= nonce_size:
            raise TokenError("invalid session payload format")
        try:
            plaintext = AESGCM(self._key).decrypt(
                sealed[:nonce_size], sealed[nonce_size:], run_id.encode()
            )
        except Exception as exc:  # Normalize InvalidTag and related failures.
            raise TokenError("session payload verification failed") from exc
        message = decode_message(plaintext)
        if not isinstance(message, dict) or message.get("type") != "init":
            raise TokenError("session payload is not an init message")
        return message


class TokenReplayGuard:
    """One-use token guarantee using jti.

    The in-process cache is naturally bounded because the gateway is a single
    instance and tokens expire quickly. consume returns False when the token has
    already been used to create a session.
    """

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}  # jti -> eviction timestamp

    def consume(self, principal: Principal, now: float | None = None) -> bool:
        if not principal.jti:
            return True  # Compatibility path for legacy issuers without jti.
        now = time.time() if now is None else now
        # Lazy cleanup: records past exp plus skew can no longer verify.
        expired = [j for j, until in self._seen.items() if until <= now]
        for j in expired:
            del self._seen[j]
        if principal.jti in self._seen:
            return False
        # Keep until exp+60s and for at least five minutes, leaving room beyond
        # the 30s verification skew so retention cannot be shorter than validity.
        self._seen[principal.jti] = max(float(principal.exp) + 60.0, now + 300.0)
        return True


@dataclass
class WorldPlatformConfig:
    """Configuration for the authorized_generate route."""

    public_key: Ed25519PublicKey
    callback_url: str  # world-service root, e.g. http://world-service.world-model
    callback_app_id: str
    callback_key_id: str
    callback_secret: str


class WorldCallbacks:
    """Lifecycle callback client with fire-and-forget bounded retries.

    Callback failures never affect the session; world-service has a deadline
    fallback scan.
    """

    def __init__(self, cfg: WorldPlatformConfig) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(timeout=5.0)
        self._pending: set[asyncio.Task] = set()  # strong task refs; see fire()

    def _sign(self, method: str, path: str, body: bytes) -> dict[str, str]:
        # Canonical string/signature compatible with biz-core tracking ingest.
        ts = str(int(time.time() * 1000))
        nonce = secrets.token_hex(16)
        canonical = "\n".join(
            [method.upper(), path, ts, nonce, hashlib.sha256(body).hexdigest()]
        )
        sig = base64.b64encode(
            hmac_mod.new(
                self._cfg.callback_secret.encode(), canonical.encode(), hashlib.sha256
            ).digest()
        ).decode()
        return {
            "Content-Type": "application/json",
            "X-Track-App-Id": self._cfg.callback_app_id,
            "X-Track-Key-Id": self._cfg.callback_key_id,
            "X-Track-Timestamp": ts,
            "X-Track-Nonce": nonce,
            "X-Track-Signature": sig,
            "X-Track-Signature-Version": "v1",
        }

    async def _post(self, path: str, payload: dict) -> None:
        body = json.dumps(payload).encode()
        url = self._cfg.callback_url.rstrip("/") + path
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    url, content=body, headers=self._sign("POST", path, body)
                )
                if resp.status_code < 300:
                    return
                logger.warning(
                    "world callback returned %s (%s)", resp.status_code, path
                )
            except Exception as exc:  # noqa: BLE001 - callback failures are logged.
                logger.warning(
                    "world callback failed (%s), attempt %d: %s",
                    path,
                    attempt + 1,
                    exc,
                )
            await asyncio.sleep(0.5 * (attempt + 1))

    def fire(self, path: str, payload: dict) -> None:
        """Dispatch asynchronously without blocking the session path.

        Keep a strong task reference as asyncio recommends, because the event
        loop only keeps weak references. Done callbacks remove completed tasks.
        """
        task = asyncio.get_running_loop().create_task(self._post(path, payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def rewrite(self, run_id: str, instruction: str, baseline: str) -> dict:
        """Rewrite a player instruction synchronously for the interaction path.

        world-service owns rewrite-model credentials and prompt assets. The
        gateway sends only the raw user instruction and current baseline, then
        receives a new full prompt. The 25s timeout leaves room for retries even
        though the model call itself is normally much shorter.
        """
        path = "/internal/v1/rewrite"
        body = json.dumps(
            {"run_id": run_id, "instruction": instruction, "baseline": baseline}
        ).encode()
        url = self._cfg.callback_url.rstrip("/") + path
        resp = await self._client.post(
            url,
            content=body,
            headers=self._sign("POST", path, body),
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("prompt"):
            raise ValueError("rewrite response missing prompt")
        return data

    def started(self, run_id: str, trace_id: str, max_lifetime_s: int) -> None:
        self.fire(
            "/internal/v1/runs/started",
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "admitted_at_ms": int(time.time() * 1000),
                "max_lifetime_s": max_lifetime_s,
            },
        )

    def ended(self, run_id: str, trace_id: str, reason: str) -> None:
        self.fire(
            "/internal/v1/runs/ended",
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "ended_at_ms": int(time.time() * 1000),
                "reason": reason,
            },
        )

    def aborted(self, run_id: str, trace_id: str, fault: str, reason: str) -> None:
        self.fire(
            "/internal/v1/runs/aborted",
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "fault": fault,
                "reason": reason,
            },
        )
