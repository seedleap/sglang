#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.results_root.glob("*x*gpu/summary.json")):
        data = json.loads(summary_path.read_text())
        rows.append(
            {
                "topology": summary_path.parent.name,
                **data["summary"],
            }
        )
    rows.sort(key=lambda row: row["node_videos_per_hour"], reverse=True)
    output = {"ranked_by_node_videos_per_hour": rows}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
