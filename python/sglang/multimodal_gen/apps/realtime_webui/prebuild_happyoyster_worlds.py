#!/usr/bin/env python3

"""Prebuild all packaged HappyOyster preset Worlds through the WebUI BFF."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_JS = ROOT / "app.js"
ASSET_ROOT = ROOT / "assets" / "presets" / "v1"
DEFAULT_OUTPUT = ROOT / "happyoyster_prebuilt_worlds.json"
PRESET_RE = re.compile(
    r'\{\s*name:\s*"(?P<name>[^"]+)"(?P<body>.*?)'
    r'prompt:\s*"(?P<prompt>(?:\\.|[^"\\])*)"(?P<tail>.*?)'
    r'referenceUrl:\s*`\$\{PRESET_ASSET_BASE_URL\}/(?P<asset>[^`]+)`.*?\}',
    re.DOTALL,
)


def preset_key(name: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", name.lower()))


def load_presets() -> list[dict[str, str]]:
    source = APP_JS.read_text()
    presets = []
    for match in PRESET_RE.finditer(source):
        name = match.group("name")
        prompt = json.loads(f'"{match.group("prompt")}"')
        asset = match.group("asset")
        presets.append(
            {
                "key": preset_key(name),
                "name": name,
                "prompt": prompt,
                "asset": asset,
                "perspective": (
                    "first_person"
                    if re.search(r"first[-_ ]person", prompt, re.IGNORECASE)
                    else "third_person"
                ),
            }
        )
    if len(presets) != 14:
        raise RuntimeError(f"expected 14 packaged presets, found {len(presets)}")
    return presets


async def read_json(response):
    payload = await response.json(content_type=None)
    if response.status >= 400:
        raise RuntimeError(str(payload.get("error") or payload))
    return payload


async def build_one(session, base_url: str, preset: dict[str, str], semaphore):
    async with semaphore:
        asset_path = ASSET_ROOT / preset["asset"]
        mime = "image/png" if asset_path.suffix.lower() == ".png" else "image/jpeg"
        uploaded = await read_json(
            await session.post(
                f"{base_url}/api/happyoyster/share-image",
                data=asset_path.read_bytes(),
                headers={"Content-Type": mime},
            )
        )
        created = await read_json(
            await session.post(
                f"{base_url}/api/happyoyster/worlds",
                json={
                    "prompt": preset["prompt"],
                    "firstFrameUrl": uploaded["url"],
                    "perspective": preset["perspective"],
                    "presetKey": preset["key"],
                },
            )
        )
        world_id = str(created.get("encryptedWorldId") or "")
        if not world_id:
            raise RuntimeError(f'{preset["name"]}: create did not return encryptedWorldId')
        for _attempt in range(80):
            status = await read_json(
                await session.get(
                    f"{base_url}/api/happyoyster/worlds/build-status",
                    params={"encryptedWorldId": world_id},
                )
            )
            state = str(status.get("status") or "").lower()
            if state == "ready":
                return preset["key"], {
                    "name": preset["name"],
                    "encryptedWorldId": world_id,
                }
            if state == "failed":
                raise RuntimeError(f'{preset["name"]}: build failed')
            await asyncio.sleep(3)
        raise RuntimeError(f'{preset["name"]}: build timed out')


def load_existing(output: Path) -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(output.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    worlds = payload.get("worlds") if isinstance(payload, dict) else None
    if not isinstance(worlds, dict):
        return {}
    return {
        key: value
        for key, value in worlds.items()
        if isinstance(key, str)
        and isinstance(value, dict)
        and value.get("encryptedWorldId")
    }


def persist(output: Path, worlds: dict[str, dict[str, str]]) -> dict:
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "worlds": dict(sorted(worlds.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return payload


async def run(args):
    presets = load_presets()
    if args.dry_run:
        print(json.dumps(presets, ensure_ascii=False, indent=2))
        return
    from aiohttp import ClientSession, ClientTimeout

    base_url = args.base_url.rstrip("/")
    timeout = ClientTimeout(total=360)
    semaphore = asyncio.Semaphore(args.concurrency)
    worlds = load_existing(args.output) if args.resume else {}
    pending = [preset for preset in presets if preset["key"] not in worlds]
    async with ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                build_one(session, base_url, preset, semaphore)
            )
            for preset in pending
        ]
        errors = []
        for task in asyncio.as_completed(tasks):
            try:
                key, value = await task
            except Exception as exc:
                errors.append(str(exc))
                continue
            worlds[key] = value
            persist(args.output, worlds)
            print(f"ready {key}: {value['encryptedWorldId']}")
    payload = persist(args.output, worlds)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 4:
        parser.error("--concurrency must be between 1 and 4")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
