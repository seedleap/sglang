"""Validate the MinWM image dependency graph and its pinned metadata exceptions."""

from __future__ import annotations

import json
import re
import subprocess
import sys

ALLOWED_METADATA_CONFLICTS = {
    "cutlass-base-protobuf": re.compile(
        r"^nvidia-cutlass-dsl-libs-base 4\.6\.0\.dev0 has requirement "
        r"protobuf<7,>=6\.30\.2, but you have protobuf 7\.[0-9.]+\.$"
    ),
    "cutlass-cu13-protobuf": re.compile(
        r"^nvidia-cutlass-dsl-libs-cu13 4\.6\.0\.dev0 has requirement "
        r"protobuf<7,>=6\.30\.2, but you have protobuf 7\.[0-9.]+\.$"
    ),
    "fa4-tvm-ffi": re.compile(
        r"^flash-attn-4 4\.0\.0b21 has requirement "
        r"apache-tvm-ffi<0\.2,>=0\.1\.12, but you have apache-tvm-ffi 0\.1\.11\.$"
    ),
    "sglang-fa4-overlay": re.compile(
        r"^sglang [^ ]+ has requirement flash-attn-4==4\.0\.0b15, "
        r"but you have flash-attn-4 4\.0\.0b21\.$"
    ),
    "sglang-cutlass-overlay": re.compile(
        r"^sglang [^ ]+ has requirement nvidia-cutlass-dsl\[cu13\]==4\.5\.2, "
        r"but you have nvidia-cutlass-dsl 4\.6\.0\.dev0\.$"
    ),
}


def validate_pip_check(output: str, returncode: int) -> dict:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if returncode == 0:
        if lines not in ([], ["No broken requirements found."]):
            raise ValueError(f"unexpected successful pip-check output: {lines!r}")
        return {"pip_check_clean": True, "accepted_metadata_conflicts": []}

    matched: dict[str, str] = {}
    unexpected: list[str] = []
    for line in lines:
        names = [
            name
            for name, pattern in ALLOWED_METADATA_CONFLICTS.items()
            if pattern.fullmatch(line)
        ]
        if len(names) != 1 or names[0] in matched:
            unexpected.append(line)
        else:
            matched[names[0]] = line

    missing = sorted(set(ALLOWED_METADATA_CONFLICTS) - set(matched))
    if unexpected or missing:
        raise ValueError(
            json.dumps(
                {"unexpected_pip_check_lines": unexpected, "missing": missing},
                sort_keys=True,
            )
        )
    return {
        "pip_check_clean": False,
        "accepted_metadata_conflicts": [matched[name] for name in sorted(matched)],
    }


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    try:
        result = validate_pip_check(output, completed.returncode)
    except ValueError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps({"passed": True, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
