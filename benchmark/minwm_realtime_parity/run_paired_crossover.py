#!/usr/bin/env python3
"""Run isolated two-GPU MinWM paired-crossover benchmarks.

The runner only orchestrates servers and the RTX6000 contract client. Product
paths and feature selection remain entirely in the explicit config commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


TERMINATING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_cpu_set(value: str) -> set[int]:
    cpus = set()
    for group in value.split(","):
        bounds = [int(part) for part in group.split("-")]
        if len(bounds) == 1:
            cpus.add(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            cpus.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"invalid cpu_set: {value}")
    return cpus


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if len(config["gpu_slots"]) != 2:
        raise ValueError("paired crossover requires exactly two gpu_slots")
    if len({str(slot["gpu"]) for slot in config["gpu_slots"]}) != 2:
        raise ValueError("gpu_slots must select two distinct GPUs")
    if any(not slot.get("cpu_set") for slot in config["gpu_slots"]):
        raise ValueError("every gpu_slot requires an explicit cpu_set")
    if len({str(slot["cpu_set"]) for slot in config["gpu_slots"]}) != 2:
        raise ValueError("gpu_slots must use distinct cpu_set values")
    if parse_cpu_set(str(config["gpu_slots"][0]["cpu_set"])) & parse_cpu_set(
        str(config["gpu_slots"][1]["cpu_set"])
    ):
        raise ValueError("gpu_slots cpu_set values must not overlap")
    if int(config.get("paired_reps", 3)) < 3:
        raise ValueError("paired_reps must be at least 3")
    if not config.get("cases"):
        raise ValueError("at least one case is required")
    for case in config["cases"]:
        if case["size"] not in {"832x480", "1248x704"}:
            raise ValueError(f"unsupported size: {case['size']}")
        if case["mode"] not in {"eager", "cuda_graph"}:
            raise ValueError(f"unsupported mode: {case['mode']}")
        for variant in ("control", "candidate"):
            if not case[variant].get("command"):
                raise ValueError(f"{case['name']} {variant} command is empty")
    return config


def assignment(rep: int) -> dict[str, int]:
    return {"control": rep % 2, "candidate": 1 - (rep % 2)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def render_command(command: list[str], *, port: int, work_dir: Path) -> list[str]:
    return [part.format(port=port, work_dir=str(work_dir)) for part in command]


def isolated_env(base: dict[str, str], slot: dict, work_dir: Path) -> dict[str, str]:
    env = dict(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(slot["gpu"]),
            "HOME": str(work_dir / "home"),
            "TMPDIR": str(work_dir / "tmp"),
            "TORCHINDUCTOR_CACHE_DIR": str(work_dir / "torchinductor"),
            "TRITON_CACHE_DIR": str(work_dir / "triton"),
            "CUDA_CACHE_PATH": str(work_dir / "cuda-cache"),
            "XDG_CACHE_HOME": str(work_dir / "xdg-cache"),
        }
    )
    return env


def command_prefix(slot: dict) -> list[str]:
    prefix = []
    if slot.get("numa_node") is not None:
        prefix.extend(
            [
                "numactl",
                f"--cpunodebind={slot['numa_node']}",
                f"--membind={slot['numa_node']}",
            ]
        )
    if slot.get("cpu_set"):
        prefix.extend(["taskset", "-c", str(slot["cpu_set"])])
    return prefix


def wait_health(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become healthy")


def start_server(
    case: dict, variant: str, slot: dict, root: Path, port: int
) -> tuple[subprocess.Popen, object]:
    work_dir = root / variant
    for child in ("home", "tmp", "torchinductor", "triton", "cuda-cache", "xdg-cache"):
        (work_dir / child).mkdir(parents=True, exist_ok=True)
    log = (work_dir / "server.log").open("w")
    command = command_prefix(slot) + render_command(
        case[variant]["command"], port=port, work_dir=work_dir
    )
    env = isolated_env(os.environ, slot, work_dir)
    env.update(
        {str(key): str(value) for key, value in case[variant].get("env", {}).items()}
    )
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_client(
    config: dict, case: dict, variant: str, root: Path, port: int, chunks: int
) -> tuple[subprocess.Popen, object]:
    work_dir = root / variant
    output = work_dir / "benchmark.json"
    log = (work_dir / "client.log").open("w")
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_rtx6000_contract.py")),
        "--output",
        str(output),
        "--server-log",
        str(work_dir / "server.log"),
        "--profile-name",
        f"{case['name']}-{variant}",
        "--sglang-git-ref",
        str(config["sglang_git_ref"]),
        "--ws-url",
        f"ws://127.0.0.1:{port}/v1/realtime_video/generate",
        "--warmup-chunks",
        str(config.get("warmup_chunks", 5)),
        "--measured-chunks",
        str(chunks),
        "--steady-start-chunk",
        str(min(config.get("steady_start_chunk", 10), chunks - 1)),
        "--sizes",
        case["size"],
    ]
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def start_telemetry(
    root: Path, gpus: list[str]
) -> tuple[subprocess.Popen | None, object | None]:
    output = (root / "gpu-telemetry.csv").open("w")
    command = [
        "nvidia-smi",
        f"--id={','.join(gpus)}",
        "--query-gpu=timestamp,index,uuid,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,temperature.gpu",
        "--format=csv",
        "--loop-ms=1000",
    ]
    try:
        return subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        ), output
    except FileNotFoundError:
        output.write("nvidia-smi not found\n")
        output.close()
        return None, None


def capture_environment(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidia-smi.txt": ["nvidia-smi", "-q"],
        "topology.txt": ["nvidia-smi", "topo", "-m"],
        "p2p.txt": ["nvidia-smi", "topo", "-p2p", "r"],
        "cpu.txt": ["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE"],
        "uname.txt": ["uname", "-a"],
        "runner-git.txt": ["git", "rev-parse", "HEAD"],
    }
    for name, command in commands.items():
        with (root / name).open("w") as output:
            subprocess.run(
                command, stdout=output, stderr=subprocess.STDOUT, check=False
            )


def publish(config: dict, source: Path, destination: Path, metadata: dict) -> None:
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    atomic_json(staging / "result.json", metadata)
    (staging / "COMPLETE").write_text("complete\n")
    if destination.exists():
        if (destination / "UPLOADED.json").is_file():
            return
        shutil.rmtree(destination)
    os.replace(staging, destination)
    upload = config.get("upload_command")
    if upload:
        command = [
            part.format(source=str(destination), relative=str(destination.name))
            for part in upload
        ]
        subprocess.run(command, check=True)
    atomic_json(
        destination / "UPLOADED.json", {"completed_at": time.time(), **metadata}
    )


def flush_interrupted(
    config: dict, source: Path, case_name: str, label: str, metadata: dict
) -> None:
    if not source.exists():
        return
    destination = (
        Path(config["artifact_root"])
        / "interrupted"
        / case_name
        / f"{label}-{int(time.time())}"
    )
    staging = destination.with_name(destination.name + ".partial")
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    atomic_json(staging / "INTERRUPTED.json", metadata)
    os.replace(staging, destination)
    upload = config.get("upload_command")
    if upload:
        command = [
            part.format(
                source=str(destination),
                relative=str(destination.relative_to(config["artifact_root"])),
            )
            for part in upload
        ]
        subprocess.run(command, check=False)


def result_rates(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    result = data["results"][0]
    return {
        "client": float(result["steady_client_fps"]),
        "scheduler": float(result["steady_scheduler_output_fps"]),
    }


def concurrency_is_safe(
    slowdowns: dict[str, dict[str, float]], threshold: float
) -> bool:
    return (
        max(value for variant in slowdowns.values() for value in variant.values())
        <= threshold
    )


def run_pair(
    config: dict,
    case: dict,
    label: str,
    slot_map: dict[str, int],
    chunks: int,
    concurrent: bool,
    variants: tuple[str, ...] = ("control", "candidate"),
) -> dict:
    scratch = Path(config["nvme_root"]) / "active" / case["name"] / label
    published = Path(config["artifact_root"]) / case["name"] / label
    if (published / "UPLOADED.json").is_file():
        return json.loads((published / "UPLOADED.json").read_text())
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    processes: dict[str, subprocess.Popen | None] = {
        variant: None for variant in variants
    }
    clients: dict[str, subprocess.Popen] = {}
    logs = {}
    client_logs = {}
    ports = {
        variant: int(config["base_port"]) + slot_map[variant] for variant in processes
    }
    started = time.time()
    telemetry, telemetry_log = start_telemetry(
        scratch, [str(config["gpu_slots"][slot_map[v]]["gpu"]) for v in variants]
    )
    error: BaseException | None = None
    metadata = {}
    try:
        for variant in processes:
            processes[variant], logs[variant] = start_server(
                case,
                variant,
                config["gpu_slots"][slot_map[variant]],
                scratch,
                ports[variant],
            )
        for variant in processes:
            wait_health(
                ports[variant],
                processes[variant],
                float(config.get("health_timeout", 1200)),
            )
        if concurrent:
            for variant in processes:
                clients[variant], client_logs[variant] = run_client(
                    config, case, variant, scratch, ports[variant], chunks
                )
            statuses = {variant: client.wait() for variant, client in clients.items()}
        else:
            statuses = {}
            for variant in processes:
                clients[variant], client_logs[variant] = run_client(
                    config, case, variant, scratch, ports[variant], chunks
                )
                statuses[variant] = clients[variant].wait()
                client_logs[variant].close()
        if any(statuses.values()):
            raise RuntimeError(f"client failure: {statuses}")
        metadata = {
            "case": case["name"],
            "size": case["size"],
            "mode": case["mode"],
            "label": label,
            "concurrent": concurrent,
            "sglang_git_ref": config["sglang_git_ref"],
            "commands": {variant: case[variant]["command"] for variant in variants},
            "gpu_assignment": {
                variant: config["gpu_slots"][index]["gpu"]
                for variant, index in slot_map.items()
            },
            "status": "complete",
            "started_at": started,
            "completed_at": time.time(),
        }
    except BaseException as caught:
        error = caught
    finally:
        for client in clients.values():
            stop_server(client)
        for process in processes.values():
            stop_server(process)
        for log in logs.values():
            log.close()
        for log in client_logs.values():
            if not log.closed:
                log.close()
        stop_server(telemetry)
        if telemetry_log is not None:
            telemetry_log.close()
    if error is not None:
        interrupted = {
            "error": repr(error),
            "time": time.time(),
            "terminating": TERMINATING,
        }
        atomic_json(scratch / "INTERRUPTED.json", interrupted)
        flush_interrupted(config, scratch, case["name"], label, interrupted)
        raise error
    publish(config, scratch, published, metadata)
    return metadata


def calibrate(config: dict, case: dict) -> bool:
    chunks = int(config.get("calibration_chunks", 12))
    solo = {}
    for variant, slot_index in (("control", 0), ("candidate", 1)):
        solo[variant] = run_pair(
            config,
            case,
            f"calibration-solo-{variant}",
            {variant: slot_index},
            chunks,
            False,
            (variant,),
        )
    concurrent = run_pair(
        config,
        case,
        "calibration-concurrent",
        {"control": 0, "candidate": 1},
        chunks,
        True,
    )
    root = Path(config["artifact_root"]) / case["name"]
    ratios: dict[str, dict[str, float]] = {}
    for variant in ("control", "candidate"):
        solo_rates = result_rates(
            root / solo[variant]["label"] / variant / "benchmark.json"
        )
        concurrent_rates = result_rates(
            root / concurrent["label"] / variant / "benchmark.json"
        )
        ratios[variant] = {
            metric: (solo_rates[metric] - concurrent_rates[metric]) / solo_rates[metric]
            for metric in solo_rates
        }
    threshold = float(config.get("concurrency_threshold", 0.02))
    exploratory = not concurrency_is_safe(ratios, threshold)
    atomic_json(
        root / "calibration.json",
        {
            "slowdown": ratios,
            "threshold": threshold,
            "concurrent_exploratory": exploratory,
        },
    )
    return not exploratory


def plan(config: dict) -> dict:
    return {
        "paired_reps": int(config.get("paired_reps", 3)),
        "cases": [
            {
                "name": case["name"],
                "size": case["size"],
                "mode": case["mode"],
                "assignments": [
                    assignment(rep) for rep in range(int(config.get("paired_reps", 3)))
                ],
            }
            for case in config["cases"]
        ],
    }


def on_signal(signum, _frame) -> None:
    global TERMINATING
    TERMINATING = True
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    if args.dry_run:
        print(json.dumps(plan(config), indent=2, sort_keys=True))
        return
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    artifact_root = Path(config["artifact_root"])
    capture_environment(artifact_root / "environment")
    atomic_json(artifact_root / "resolved-config.json", config)
    for case in config["cases"]:
        concurrent = calibrate(config, case)
        for rep in range(int(config.get("paired_reps", 3))):
            run_pair(
                config,
                case,
                f"rep-{rep:02d}",
                assignment(rep),
                int(config.get("measured_chunks", 69)),
                concurrent,
            )


if __name__ == "__main__":
    main()
