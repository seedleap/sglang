"""GPU contracts for MinWM's split asynchronous Ulysses A2A.

The default CI arm uses two GPUs. Set ``MINWM_ASYNC_TEST_WORLD=4`` when
launching this file on a four-GPU node to exercise the identical SP4 contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform

_WORLD = int(os.environ.get("MINWM_ASYNC_TEST_WORLD", "2"))


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=_WORLD, ulysses_degree=_WORLD
    )

    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_begin_input_all_to_all_qk,
        _usp_begin_input_all_to_all_v,
        _usp_begin_output_all_to_all,
        _usp_input_all_to_all_qkv,
        _usp_output_all_to_all,
        _usp_pynccl_communicator,
        _usp_wait_all_to_all,
    )
    from sglang.multimodal_gen.runtime.models.dits.minwm import (
        _MinWMUlyssesWorkspace,
    )

    workspace = _MinWMUlyssesWorkspace()
    failures: list[str] = []
    batch, local_sequence, global_heads, head_size = 1, 96, 8, 32

    def inputs(seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed + rank * 1000)
        shape = batch, local_sequence, global_heads, head_size
        tensors = [
            torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
            for _ in range(3)
        ]
        return tuple(tensors)

    def split_input(q, k, v, backend: str):
        qk_lease = workspace.acquire("input_qk", q, 2 * q.numel())
        assert qk_lease is not None
        send, recv, stream, events, release = qk_lease
        qk_handle = _usp_begin_input_all_to_all_qk(
            q,
            k,
            input_buffer=send,
            output_buffer=recv,
            comm_stream=stream,
            events=events,
            backend=backend,
            release=release,
        )

        # This kernel has no dependency on Q/K or their receive buffer. Its
        # position between begin and wait is the event-ordering contract.
        independent = v.square().sum(dim=-1)

        v_lease = workspace.acquire("input_v", v, v.numel())
        assert v_lease is not None
        send, recv, stream, events, release = v_lease
        v_handle = _usp_begin_input_all_to_all_v(
            v,
            input_buffer=send,
            output_buffer=recv,
            comm_stream=stream,
            events=events,
            backend=qk_handle.backend,
            release=release,
        )
        qk = _usp_wait_all_to_all(qk_handle)
        gathered_v = _usp_wait_all_to_all(v_handle)
        return (*qk.chunk(2, dim=-1), gathered_v, independent)

    def async_output(x, backend: str):
        lease = workspace.acquire("output", x, x.numel())
        assert lease is not None
        send, recv, stream, events, release = lease
        handle = _usp_begin_output_all_to_all(
            x,
            head_dim=2,
            input_buffer=send,
            output_buffer=recv,
            comm_stream=stream,
            events=events,
            backend=backend,
            release=release,
        )
        return _usp_wait_all_to_all(handle)

    # Long enough to wrap both persistent slots repeatedly and expose stale
    # receive data or an early slot release.
    for iteration in range(12):
        q, k, v = inputs(100 + iteration)
        expected = _usp_input_all_to_all_qkv(q, k, v)
        got_q, got_k, got_v, independent = split_input(q, k, v, "process_group")
        torch.cuda.synchronize()
        if not torch.equal(torch.cat((got_q, got_k, got_v), dim=-1), expected):
            failures.append(f"process-group split input iteration {iteration}")
        if independent.shape != q.shape[:-1]:
            failures.append("independent overlap kernel returned the wrong shape")

        expected_output = _usp_output_all_to_all(got_q, head_dim=2)
        got_output = async_output(got_q, "process_group")
        torch.cuda.synchronize()
        if not torch.equal(got_output, expected_output):
            failures.append(f"process-group output iteration {iteration}")
    if workspace._busy:
        failures.append(f"workspace retained busy leases: {workspace._busy}")

    # Capture and replay the same split API. ProcessGroupNCCL intentionally
    # falls back in capture; this arm proves the raw PyNCCL selection instead.
    communicator = _usp_pynccl_communicator()
    if communicator is None:
        failures.append("PyNCCL communicator unavailable for graph contract")
    else:
        for seed in (300, 301):
            q, k, v = inputs(seed)
            split_input(q, k, v, "pynccl")
        torch.cuda.synchronize()

        static_q, static_k, static_v = inputs(302)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            graph_q, graph_k, graph_v, graph_independent = split_input(
                static_q, static_k, static_v, "pynccl"
            )
        torch.cuda.synchronize()

        for replay, seed in enumerate((400, 401, 402)):
            q, k, v = inputs(seed)
            expected = _usp_input_all_to_all_qkv(q, k, v)
            static_q.copy_(q)
            static_k.copy_(k)
            static_v.copy_(v)
            graph.replay()
            torch.cuda.synchronize()
            got = torch.cat((graph_q, graph_k, graph_v), dim=-1)
            if not torch.equal(got, expected):
                failures.append(f"PyNCCL graph replay {replay}")
            if graph_independent.shape != q.shape[:-1]:
                failures.append("captured independent kernel returned wrong shape")

    verdict = torch.tensor([len(failures)], dtype=torch.int32, device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} FAIL {failures}", flush=True)
    if rank == 0:
        print(
            f"MINWM_ASYNC_A2A_SP{_WORLD} {'FAIL' if verdict.item() else 'PASS'}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestMinWMAsyncA2A(unittest.TestCase):
    def test_split_async_a2a_contracts(self):
        if not current_platform.is_cuda():
            self.skipTest("MinWM async A2A requires CUDA")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29523",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "MinWM async A2A contract failed")
        self.assertIn(f"MINWM_ASYNC_A2A_SP{_WORLD} PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
