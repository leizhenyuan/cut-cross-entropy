# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import softcapping

skip_no_xpu = pytest.mark.skipif(not torch.xpu.is_available(), reason="Test requires XPU")


def _lse(e: torch.Tensor, c: torch.Tensor, softcap: float | None) -> torch.Tensor:
    logits = e @ c.T
    if softcap is not None:
        logits = softcapping(logits, softcap)
    return torch.logsumexp(logits.float(), dim=-1)


@skip_no_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shape", [(256, 512, 128), (255, 507, 128), (255, 507, 123)])
def test_lse(dtype: torch.dtype, softcap: float | None, shape: tuple[int, int, int]):
    torch.set_float32_matmul_precision("highest")
    torch.xpu.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.xpu.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="xpu", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="xpu", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    gt = _lse(e.float(), c.float(), softcap)

    torch.set_float32_matmul_precision("highest" if dtype == torch.float32 else "high")
    ref = _lse(e, c, softcap)

    cce_lse = cce_lse_forward_kernel(e, c, softcap=softcap)

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_lse).abs()

    assert (
        cce_error <= (expected_error + 1e-5)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"
