# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_metrics():
    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "apps"
        / "realtime_webui"
        / "critical_path_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("standalone_webui_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_histogram_uses_the_shared_contract_without_external_dependencies():
    metrics = _load_metrics()

    assert metrics.observe_stage_seconds(
        "frame_encode",
        0.02,
        service="world-studio-webui",
        model="lingbot2",
        lane="default",
        result="success",
        codec="h264",
        scope="frame",
    )
    output = metrics.prometheus_latest().decode()

    assert "# TYPE world_model_critical_path_stage_duration_seconds histogram" in output
    assert 'stage="frame_encode"' in output
    assert 'service="world-studio-webui"' in output
    assert 'model="lingbot2"' in output
    assert 'result="success"' in output
    assert 'codec="h264"' in output
    assert 'scope="frame"' in output
    assert "_bucket{" in output and 'le="+Inf"} 1' in output
    assert "_sum{" in output and " 0.02" in output
    assert "_count{" in output and " 1" in output


def test_histogram_rejects_unbounded_or_invalid_labels():
    metrics = _load_metrics()

    assert not metrics.observe_stage_seconds("unknown", 1)
    assert not metrics.observe_stage_seconds("frame_encode", 1, result="retry")
    assert not metrics.observe_stage_seconds("frame_encode", 1, scope="session")
    assert not metrics.observe_stage_seconds("frame_encode", -1)
    assert metrics.prometheus_latest().decode().count("_bucket{") == 0
