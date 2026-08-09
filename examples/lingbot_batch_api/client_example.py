#!/usr/bin/env python3
"""Minimal caller-side backpressure and retry example."""

from __future__ import annotations

import argparse
import asyncio
import json
import random

import httpx


async def generate_with_backpressure(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    payload: dict,
) -> dict:
    attempt = 0
    while True:
        attempt += 1
        response = await client.post(
            url,
            headers={"X-API-Key": api_key},
            json=payload,
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
        retry_after = float(response.headers.get("Retry-After", "5"))
        await asyncio.sleep(
            min(60, retry_after * (1.5 ** min(attempt - 1, 5))) + random.random()
        )


async def main_async(args: argparse.Namespace) -> None:
    payload = json.loads(args.payload)
    async with httpx.AsyncClient() as client:
        result = await generate_with_backpressure(
            client, args.url, args.api_key, payload
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--payload", required=True, help="One JSON request")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
