from pathlib import Path

from render_taehv_ab_job import render_job


def _env(manifest):
    return {
        item["name"]: item.get("value")
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }


def test_rendered_ab_job_keeps_b300_topology_and_only_enables_taehv_for_candidate():
    baseline = render_job(
        name="codex-lingbot-taehv-ab-baseline",
        image="107014413969.dkr.ecr.us-west-2.amazonaws.com/leap-world/sglang-video-runner:taehv-ab",
        variant="baseline",
        run_id="20260722-a",
        output_s3_prefix="s3://leap-world-us-east-2/world-model/eval/lingbot2/taehv_ab/20260722-a",
        source_s3_uri="s3://leap-world-us-east-2/world-model/eval/lingbot2/taehv_ab/20260722-a/source.tar.gz",
        source_revision="deadbeef",
    )
    candidate = render_job(
        name="codex-lingbot-taehv-ab-taehv",
        image="107014413969.dkr.ecr.us-west-2.amazonaws.com/leap-world/sglang-video-runner:taehv-ab",
        variant="taehv",
        run_id="20260722-a",
        output_s3_prefix="s3://leap-world-us-east-2/world-model/eval/lingbot2/taehv_ab/20260722-a",
        source_s3_uri="s3://leap-world-us-east-2/world-model/eval/lingbot2/taehv_ab/20260722-a/source.tar.gz",
        source_revision="deadbeef",
    )

    baseline_pod = baseline["spec"]["template"]["spec"]
    assert baseline_pod["serviceAccountName"] == "sglang-video-job"
    assert baseline_pod["schedulerName"] == "default-scheduler"
    assert baseline_pod["nodeSelector"] == {
        "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
        "eks.amazonaws.com/nodegroup": "wan22-cb-p6b300-0715-20c",
        "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
    }
    assert (
        baseline_pod["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "8"
    )
    assert _env(baseline)["SGLANG_VIDEO_TOPOLOGY"] == "8x1"
    assert _env(baseline)["SGLANG_VIDEO_CASE_LIMIT"] == "100"
    assert "TAEHV_CHECKPOINT_PATH" not in _env(baseline)
    assert _env(candidate)["TAEHV_CHECKPOINT_PATH"] == "/opt/taehv/taew2_1.pth"
    assert baseline_pod["initContainers"][0]["name"] == "prepare-sglang-source"
    assert "install-taehv" not in {
        item["name"] for item in baseline_pod["initContainers"]
    }
    assert "install-taehv" in {
        item["name"] for item in candidate["spec"]["template"]["spec"]["initContainers"]
    }
    assert "/opt/sglang/python" in _env(candidate)["PYTHONPATH"]
    taehv_install = candidate["spec"]["template"]["spec"]["initContainers"][1]["args"][
        0
    ]
    assert "--no-deps" in taehv_install


def test_ab_runner_and_uploader_keep_presigned_input_urls_out_of_artifacts():
    root = Path(__file__).resolve().parent
    runner = (root / "run_taehv_ab_test.sh").read_text(encoding="utf-8")
    uploader = (root / "upload_taehv_ab_results.py").read_text(encoding="utf-8")

    assert "prepare_taehv_ab_inputs.py" in runner
    assert "upload_taehv_ab_results.py" in runner
    assert "SGLANG_VIDEO_CASE_LIMIT" in runner
    assert "image-urls.json" in uploader
    assert "server-cache" in uploader
