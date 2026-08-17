#!/usr/bin/env python3
"""Render or check deterministic Loopit platform application JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from platform_config import (
    all_platform_configs,
    required_inputs_document,
    resolve_image_inputs,
    validate_configs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "platform/world-model/golden"
DEFAULT_REQUIRED_INPUTS = ROOT / "platform/world-model/required-inputs.json"


def render_documents(
    *, image_inputs: dict[str, object] | None = None
) -> dict[str, str]:
    configs = all_platform_configs()
    if image_inputs is not None:
        configs = resolve_image_inputs(configs, image_inputs)
    validate_configs(configs, require_resolved_images=image_inputs is not None)
    return {
        f"{index:02d}-{config['name']}.json": json.dumps(
            config, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
        for index, config in enumerate(configs, start=1)
    }


def render_required_inputs() -> str:
    return json.dumps(
        required_inputs_document(), ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-inputs", type=Path)
    parser.add_argument("--check-image-inputs", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_inputs = None
    if args.image_inputs:
        image_inputs = json.loads(args.image_inputs.read_text(encoding="utf-8"))
    documents = render_documents(image_inputs=image_inputs)
    if args.check_image_inputs:
        if image_inputs is None:
            raise SystemExit("--check-image-inputs requires --image-inputs")
        print(
            json.dumps(
                {"imageInputsValid": True, "executionReady": False},
                sort_keys=True,
            )
        )
        return
    if image_inputs is not None and not required_inputs_document()["executionReady"]:
        raise SystemExit(
            "deployment render blocked: platform/world-model/required-inputs.json "
            "still has unresolved hard inputs"
        )
    if args.check:
        expected_names = set(documents)
        actual_names = {path.name for path in args.output_dir.glob("*.json")}
        if actual_names != expected_names:
            raise SystemExit(
                f"golden filenames differ: expected={sorted(expected_names)} "
                f"actual={sorted(actual_names)}"
            )
        for name, content in documents.items():
            path = args.output_dir / name
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"golden is stale: {path}")
        if DEFAULT_REQUIRED_INPUTS.read_text(encoding="utf-8") != render_required_inputs():
            raise SystemExit(f"required inputs are stale: {DEFAULT_REQUIRED_INPUTS}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    DEFAULT_REQUIRED_INPUTS.write_text(render_required_inputs(), encoding="utf-8")


if __name__ == "__main__":
    main()
