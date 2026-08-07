import re
from pathlib import Path


ROOT = Path(__file__).parent


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
    calls = (
        "run_position temb-hoist-abba-a1-candidate 1",
        "run_position temb-hoist-abba-b1-legacy 0",
        "run_position temb-hoist-abba-b2-legacy 0",
        "run_position temb-hoist-abba-a2-candidate 1",
    )
    positions = [wrapper.index(call) for call in calls]
    assert positions == sorted(positions)

