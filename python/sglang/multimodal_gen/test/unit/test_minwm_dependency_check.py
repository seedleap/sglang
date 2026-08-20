import pytest
from sglang.multimodal_gen.tools.minwm_dependency_check import validate_pip_check

EXPECTED_CONFLICTS = """\
nvidia-cutlass-dsl-libs-cu13 4.6.0.dev0 has requirement protobuf<7,>=6.30.2, but you have protobuf 7.35.1.
nvidia-cutlass-dsl-libs-base 4.6.0.dev0 has requirement protobuf<7,>=6.30.2, but you have protobuf 7.35.1.
flash-attn-4 4.0.0b21 has requirement apache-tvm-ffi<0.2,>=0.1.12, but you have apache-tvm-ffi 0.1.11.
sglang 0.0.0.dev0 has requirement flash-attn-4==4.0.0b15, but you have flash-attn-4 4.0.0b21.
sglang 0.0.0.dev0 has requirement nvidia-cutlass-dsl[cu13]==4.5.2, but you have nvidia-cutlass-dsl 4.6.0.dev0.
"""


def test_accepts_exact_sm120_overlay_metadata_conflicts():
    result = validate_pip_check(EXPECTED_CONFLICTS, 1)

    assert result["pip_check_clean"] is False
    assert len(result["accepted_metadata_conflicts"]) == 5


def test_accepts_a_clean_dependency_graph():
    assert validate_pip_check("No broken requirements found.\n", 0) == {
        "pip_check_clean": True,
        "accepted_metadata_conflicts": [],
    }


def test_rejects_any_additional_conflict():
    with pytest.raises(ValueError, match="unexpected_pip_check_lines"):
        validate_pip_check(EXPECTED_CONFLICTS + "extra 1 has requirement x.\n", 1)
