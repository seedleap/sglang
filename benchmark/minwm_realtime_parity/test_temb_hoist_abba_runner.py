import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from measurement_tool import load_aggregate_records  # noqa: E402
from test_measurement import _record  # noqa: E402


def test_sp4_abba_runner_is_statically_profiler_off_only() -> None:
    wrapper = (ROOT / "run_temb_hoist_sp4_abba.sh").read_text()
    runner = (ROOT / "run_s0_measurement.sh").read_text()

    for contract in (
        "export MINWM_S0_PROFILER_OFF_ONLY=1",
        "export MINWM_S0_OFF_REPEAT_COUNT=1",
        "export MINWM_S0_KV_CACHE_NUM_FRAMES=45",
        "export MINWM_S0_SP_DEGREES=4",
    ):
        assert contract in wrapper
    assert not re.search(r"^\s*nsys\s+start(?:\s|$)", wrapper, re.MULTILINE)
    assert not re.search(
        r"^\s*run_profiler_on(?:\s|$)", wrapper, re.MULTILINE
    )
    assert re.search(
        r'if \[\[ "\$\{PROFILER_OFF_ONLY\}" != "1" \]\]; then\s+'
        r"install_nsys\s+fi",
        runner,
    )
    assert re.search(
        r'if \[\[ "\$\{PROFILER_OFF_ONLY\}" != "1" \]\]; then\s+'
        r"run_profiler_on",
        runner,
    )


def test_sp4_abba_runner_has_the_approved_order() -> None:
    wrapper = (ROOT / "run_temb_hoist_sp4_abba.sh").read_text()
    run_position = re.search(
        r"run_position\(\) \{(?P<body>.*?)\n\}", wrapper, re.DOTALL
    )
    assert run_position is not None
    body = run_position.group("body")
    audit_steps = (
        'CURRENT_LANE_DIR="${RESULT_ROOT}/${label}/sp4"',
        'bash "${SCRIPT_DIR}/run_s0_measurement.sh"',
        '"${SCRIPT_DIR}/measurement_tool.py" validate',
        '"${SCRIPT_DIR}/assert_latency_counts.py"',
        'CURRENT_LANE_DIR=""',
    )
    audit_positions = [body.index(step) for step in audit_steps]
    assert audit_positions == sorted(audit_positions)

    calls = (
        "run_position temb-hoist-abba-a1-candidate 1",
        "run_position temb-hoist-abba-b1-legacy 0",
        "run_position temb-hoist-abba-b2-legacy 0",
        "run_position temb-hoist-abba-a2-candidate 1",
    )
    positions = [wrapper.index(call) for call in calls]
    assert positions == sorted(positions)


def test_post_run_failure_marks_only_current_position_and_aggregate_excludes_it(
    tmp_path: Path,
) -> None:
    run_id = "abba-test"
    result_root = tmp_path / run_id
    labels = (
        "temb-hoist-abba-a1-candidate",
        "temb-hoist-abba-b1-legacy",
        "temb-hoist-abba-b2-legacy",
        "temb-hoist-abba-a2-candidate",
    )
    paths = {}
    for label in labels:
        path = result_root / label / "sp4" / "profiler-off-repeat1.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_record(run_id=label)), encoding="utf-8")
        paths[label] = path

    failed_label = labels[0]
    failed_lane = paths[failed_label].parent
    environment = os.environ.copy()
    environment.update(
        {
            "MINWM_RUN_ID": run_id,
            "MINWM_RESULTS_ROOT": str(tmp_path),
            "SGLANG_GIT_REF": "a" * 40,
            "SGLANG_SOURCE_ROOT": str(ROOT.parent.parent),
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{ROOT / "run_temb_hoist_sp4_abba.sh"}"\n'
                f'CURRENT_LANE_DIR="{failed_lane}"\n'
                "trap on_exit EXIT\n"
                "false\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert len(list(failed_lane.glob("invalid-marker*.json"))) == 1
    for label in labels[1:]:
        assert not list(paths[label].parent.glob("invalid-marker*.json"))

    records, excluded = load_aggregate_records(list(paths.values()))
    assert excluded == [paths[failed_label]]
    assert [record["run_id"] for record in records] == list(labels[1:])

    candidate = [paths[labels[0]], paths[labels[3]]]
    candidate_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "measurement_tool.py"),
            "aggregate",
            *(str(path) for path in candidate),
            "--output",
            str(tmp_path / "candidate-summary.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert candidate_result.returncode != 0
    assert "excluded invalid result" in candidate_result.stdout

    legacy = [paths[labels[1]], paths[labels[2]]]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "measurement_tool.py"),
            "aggregate",
            *(str(path) for path in legacy),
            "--output",
            str(tmp_path / "legacy-summary.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "legacy-summary.json").read_text())
    assert summary["run_ids"] == list(labels[1:3])
