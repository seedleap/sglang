# SPDX-License-Identifier: Apache-2.0
"""Wire helpers for the experimental exact realtime VAE decoder."""

from __future__ import annotations

from typing import Any

import msgspec.msgpack
import torch


SCHEMA_VERSION = "sglang-realtime-vae/v1"
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"

_NAME_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_TO_NAME = {dtype: name for name, dtype in _NAME_TO_DTYPE.items()}


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    try:
        dtype_name = _DTYPE_TO_NAME[cpu.dtype]
    except KeyError as exc:
        raise TypeError(
            f"unsupported tensor dtype for remote VAE: {cpu.dtype}"
        ) from exc
    return {
        "shape": list(cpu.shape),
        "dtype": dtype_name,
        "data": cpu.view(torch.uint8).numpy().tobytes(),
    }


def payload_to_tensor(payload: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in payload["shape"])
    try:
        dtype = _NAME_TO_DTYPE[str(payload["dtype"])]
    except KeyError as exc:
        raise TypeError(
            f"unsupported tensor dtype from remote VAE: {payload['dtype']}"
        ) from exc
    flat = torch.frombuffer(bytearray(payload["data"]), dtype=dtype)
    return flat.reshape(shape).contiguous()


def packb(value: Any) -> bytes:
    return msgspec.msgpack.encode(value)


def unpackb(value: bytes) -> Any:
    return msgspec.msgpack.decode(value)
