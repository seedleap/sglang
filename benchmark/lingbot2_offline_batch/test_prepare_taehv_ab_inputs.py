import json

from prepare_taehv_ab_inputs import build_fixture


def test_build_fixture_preserves_prompt_actions_and_fixed_first_cases(tmp_path):
    source = tmp_path / "messages.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"testset100_v2/G1/case-{index}",
                    "metadata": {"image_id": f"p{index:02d}"},
                    "messages": [
                        {"role": "user", "type": "text", "content": f"prompt {index}"},
                        {
                            "role": "target",
                            "type": "video",
                            "uri": f"s3://example-bucket/images/p{index:02d}.png",
                            "controls": [
                                {
                                    "type": "keyboard_direction_frame_interval",
                                    "action_keys": ["w", "a"],
                                    "actions": [[1, 0], [1, 0]],
                                }
                            ],
                        },
                    ],
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_fixture(
        source,
        tmp_path / "fixture",
        limit=2,
        presign_image_uri=lambda uri: f"https://signed.example/{uri.rsplit('/', 1)[-1]}",
    )

    fixture_rows = [
        json.loads(line)
        for line in (tmp_path / "fixture" / "fixture.jsonl").read_text().splitlines()
    ]
    image_urls = json.loads((tmp_path / "fixture" / "image-urls.json").read_text())
    metadata = json.loads((tmp_path / "fixture" / "fixture-metadata.json").read_text())

    assert result["selected_samples"] == 2
    assert [row["sample_id"] for row in fixture_rows] == [
        "testset100_v2/G1/case-0",
        "testset100_v2/G1/case-1",
    ]
    assert fixture_rows[0]["messages"][0]["content"] == "prompt 0"
    assert fixture_rows[0]["messages"][1]["controls"][0]["actions"] == [[1, 0], [1, 0]]
    assert image_urls == {
        "p00": "https://signed.example/p00.png",
        "p01": "https://signed.example/p01.png",
    }
    assert metadata["source_sha256"]
    assert metadata["fixture_sha256"]
