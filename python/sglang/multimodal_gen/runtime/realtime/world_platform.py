# SPDX-License-Identifier: Apache-2.0
"""World 平台接入：进入凭证验签 + 会话生命周期回调。

配套 world-service（业务后端）的 authorized_generate 路由使用：
  - 凭证：Ed25519 紧凑 JWS。网关只持公钥 —— world-service 是唯一令牌权威，
    前面接多少身份源（Clerk / biz-core），本模块一行不用改。
  - init 哈希：凭证携带 init 消息字节的 SHA-256，网关比对首条消息即可一次性
    锁死 prompt / 时长 / 时间轴 / 一切字段（比逐字段钳制更简单也更强）。
  - 回调：started / ended / aborted，HMAC 签名与 biz-core tracking ingest 同构。

本模块不被旧 showcase 路由引用；不配置 world 平台参数时行为零变化。
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
    """凭证校验失败（统一对外表现为 1008 关闭）。"""


@dataclass(frozen=True)
class Principal:
    """凭证里携带的会话主体。"""

    user_id: str
    run_id: str
    max_lifetime_s: int
    init_sha256: str
    model: str
    allow_free_prompt: bool


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def load_public_key(pub_b64: str) -> Ed25519PublicKey:
    """从 base64 公钥字符串构造验签对象（启动期调用，配置错误立即暴露）。"""
    raw = base64.b64decode(pub_b64)
    if len(raw) != 32:
        raise ValueError("Ed25519 公钥必须是 32 字节的 base64")
    return Ed25519PublicKey.from_public_bytes(raw)


def verify_session_token(token: str, public_key: Ed25519PublicKey) -> Principal:
    """校验进入凭证，返回会话主体。任何异常统一归为 TokenError。"""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "EdDSA":
            raise TokenError("仅支持 EdDSA")
        public_key.verify(
            _b64url_decode(sig_b64),
            f"{header_b64}.{payload_b64}".encode(),
        )
        claims = json.loads(_b64url_decode(payload_b64))
    except TokenError:
        raise
    except (ValueError, KeyError, InvalidSignature, binascii.Error) as exc:
        raise TokenError(f"凭证非法: {exc}") from exc

    if claims.get("aud") != "zing-gateway":
        raise TokenError("aud 不匹配")
    exp = int(claims.get("exp") or 0)
    if exp and time.time() > exp + 30:  # 30s 时钟容差
        raise TokenError("凭证已过期")
    zing = claims.get("zing") or {}
    max_lifetime = int(zing.get("max_lifetime_s") or 0)
    if max_lifetime <= 0:
        raise TokenError("缺少会话时长")
    init_sha = str(claims.get("init_sha256") or "")
    if len(init_sha) != 64:
        raise TokenError("缺少 init 哈希")
    return Principal(
        user_id=str(claims.get("sub") or ""),
        run_id=str(claims.get("sid") or ""),
        max_lifetime_s=max_lifetime,
        init_sha256=init_sha,
        model=str(zing.get("model") or "minwm"),
        allow_free_prompt=bool(zing.get("allow_free_prompt", False)),
    )


@dataclass
class WorldPlatformConfig:
    """authorized_generate 路由的全部配置。缺任一项则路由不注册。"""

    public_key: Ed25519PublicKey
    callback_url: str  # world-service 根地址，如 http://world-service.world-model
    callback_app_id: str
    callback_key_id: str
    callback_secret: str


class WorldCallbacks:
    """生命周期回调客户端。fire-and-forget + 有限重试：
    回调失败绝不影响会话本身（world-service 有 deadline 兜底扫描）。"""

    def __init__(self, cfg: WorldPlatformConfig) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(timeout=5.0)
        self._pending: set[asyncio.Task] = set()  # 任务强引用（见 fire）

    def _sign(self, method: str, path: str, body: bytes) -> dict[str, str]:
        # 与 biz-core tracking ingest 同构的 canonical/签名（pkg/hmacsign 的对端）
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
                    "world 回调返回 %s (%s)", resp.status_code, path
                )
            except Exception as exc:  # noqa: BLE001 —— 回调失败只记日志
                logger.warning("world 回调失败(%s) 第 %d 次: %s", path, attempt + 1, exc)
            await asyncio.sleep(0.5 * (attempt + 1))

    def fire(self, path: str, payload: dict) -> None:
        """异步派发，不阻塞会话主流程。"""
        asyncio.get_running_loop().create_task(self._post(path, payload))

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
