#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def assert_latency_counts(record: dict[str, Any]) -> list[str]:
    expected = record["workload"]["measured_chunks"]
    checked: list[str] = []

    def visit(value: Any, path: str = "metrics") -> None:
        if not isinstance(value, dict):
            return
        if value.get("status") == "available" and value.get("unit") == "ms_per_chunk":
            count = value.get("value", {}).get("count")
            if count != expected:
                raise ValueError(f"{path}.value.count={count!r}, expected {expected}")
            checked.append(path)
        for key, child in value.items():
            visit(child, f"{path}.{key}")

    visit(record["metrics"])
    if not checked:
        raise ValueError("no available ms_per_chunk latency metrics found")
    return checked


def main() -> None:
    path = Path(sys.argv[1])
    record = json.loads(path.read_text(encoding="utf-8"))
    checked = assert_latency_counts(record)
    expected = record["workload"]["measured_chunks"]
    print(
        f"S1 latency-count assertion passed for {len(checked)} metrics: count={expected}"
    )


if __name__ == "__main__":
    main()
