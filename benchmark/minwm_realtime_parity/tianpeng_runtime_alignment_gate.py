#!/usr/bin/env python3
"""Resolve Tianpeng's immutable contract and gate MinWM runtime configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
from pathlib import Path

from tianpeng_alignment import (
    DEFAULT_ALIGNMENT_URL,
    _fetch_bytes,
    load_contract,
)

CANONICAL_FILES = ("gap12.jsonl", "input_manifest.json", "run_manifest.json")
ALIGNMENT_FIELDS = (
    "local_attn_size",
    "sink_size",
    "rope_position_mode",
    "rope_max_frame_gap",
    "prompt_first_frame_pin_enabled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--alignment-url", default=DEFAULT_ALIGNMENT_URL)
    parser.add_argument(
        "--canonical-source-url",
        help=(
            "Immutable public source URL to record when --alignment-url uses a "
            "signed or local mirror of the same bytes."
        ),
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_alignment(canonical: dict, model_config: dict) -> list[dict]:
    runtime = {
        **{field: model_config[field] for field in ALIGNMENT_FIELDS},
        "sliding_window_num_frames": model_config["sliding_window_num_frames"],
        "scene_cut_rope_offset": model_config.get("scene_cut_rope_offset", 0),
        "scene_cut_sink_enabled": model_config.get("scene_cut_sink_enabled", False),
    }
    expected = {
        **{field: canonical[field] for field in ALIGNMENT_FIELDS},
        "sliding_window_num_frames": canonical["local_attn_size"],
        "scene_cut_rope_offset": 0,
        "scene_cut_sink_enabled": False,
    }
    rows = []
    for field, expected_value in expected.items():
        actual = runtime[field]
        rows.append(
            {
                "field": field,
                "tianpeng_canonical": expected_value,
                "converted_model": actual,
                "pass": actual == expected_value,
            }
        )
    failures = [row for row in rows if not row["pass"]]
    if failures:
        raise ValueError(f"Tianpeng alignment mismatch: {failures}")
    return rows


def build_provenance(
    model_dir: Path,
    alignment_url: str,
    checkpoint_sha256: str,
    canonical_source_url: str | None = None,
) -> dict:
    source_payloads = {
        name: _fetch_bytes(urllib.parse.urljoin(alignment_url, name))
        for name in CANONICAL_FILES
    }
    contract = load_contract(alignment_url)
    canonical = {field: contract["expected"][field] for field in ALIGNMENT_FIELDS}
    model_config_path = model_dir / "transformer" / "config.json"
    model_config = json.loads(model_config_path.read_text())
    conversion_manifest_path = model_dir / "minwm_conversion_manifest.json"
    conversion_manifest = json.loads(conversion_manifest_path.read_text())
    table = validate_alignment(canonical, model_config)
    request_contract = {
        "realtime_causal_sink_size": canonical["sink_size"],
        "realtime_causal_kv_cache_num_frames": canonical["local_attn_size"],
    }
    runtime_log_prefix = (
        "MINWM_RUNTIME_ALIGNMENT local_attn_size=32 sink_size=8 window_size=32 "
        "rope_position_mode=block_relative rope_gap=12 "
        "prompt_first_frame_pin_enabled=True request_sink_size=8 "
        "request_window_size=32 allow_growth=False"
    )
    recorded_source_url = canonical_source_url or alignment_url
    return {
        "schema_version": "minwm-tianpeng-runtime-alignment/v1",
        "status": "pass",
        "canonical_source": {
            "url": recorded_source_url,
            "files": {
                name: {
                    "url": urllib.parse.urljoin(recorded_source_url, name),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
                for name, payload in source_payloads.items()
            },
            "resolver_path": str(Path(__file__).with_name("tianpeng_alignment.py")),
            "resolver_sha256": sha256_file(
                Path(__file__).with_name("tianpeng_alignment.py")
            ),
        },
        "canonical": canonical,
        "contract_expected": contract["expected"],
        "experiment_checkpoint": {
            "sha256": checkpoint_sha256,
            "conversion_source": conversion_manifest["source_checkpoint"],
            "matches_tianpeng_canonical_checkpoint": (
                checkpoint_sha256 == contract["expected"]["checkpoint_sha256"]
            ),
            "gate_scope": (
                "The requested gate aligns causal attention/cache semantics; the "
                "experiment remains on its fixed A/B checkpoint."
            ),
        },
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "converted_model": {
            field: model_config[field]
            for field in (
                *ALIGNMENT_FIELDS,
                "sliding_window_num_frames",
                "scene_cut_rope_offset",
                "scene_cut_sink_enabled",
            )
        },
        "request_contract": request_contract,
        "runtime_log_required_prefix": runtime_log_prefix,
        "alignment_table": table,
        "semantics": {
            "block_relative": (
                "RoPE positions are recomputed over the currently visible raw-K "
                "order: global sink, delayed prompt/scene pin, then continuous tail."
            ),
            "rope_gap_12": (
                "Relative temporal positions are capped to a maximum inter-block "
                "gap of 12 after cache eviction; pre-rotated K cannot be reused."
            ),
        },
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    payload = build_provenance(
        Path(args.model_dir),
        args.alignment_url,
        args.checkpoint_sha256,
        args.canonical_source_url,
    )
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
