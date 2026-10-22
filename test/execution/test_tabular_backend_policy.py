# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit attention backend policy and ambient restoration without CUDA work."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sdm.execution._tabular_icl import fit_icl


def enabled() -> tuple[bool, bool, bool, bool]:
    backend = torch.backends.cuda
    return (
        backend.flash_sdp_enabled(),
        backend.mem_efficient_sdp_enabled(),
        backend.cudnn_sdp_enabled(),
        backend.math_sdp_enabled(),
    )


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize(
    ("autocast", "dtype", "cudnn"),
    [
        (False, torch.float16, False),
        (True, torch.float16, True),
        (True, torch.bfloat16, False),
    ],
)
def test_cuda_fit_precision_policy_and_ambient_restoration(
    raises: bool,
    autocast: bool,
    dtype: torch.dtype,
    cudnn: bool,
) -> None:
    primary = RuntimeError("injected fit failure")
    full = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
    exchange = SimpleNamespace(gather=lambda *args: full)

    def model(*args: object, **kwargs: object) -> object:
        assert enabled() == (True, True, cudnn, False)
        if raises:
            raise primary
        return full

    before = enabled()
    was_enabled = torch.is_autocast_enabled("cuda")
    previous_dtype = torch.get_autocast_dtype("cuda")
    try:
        # Thread-local policy only: no CUDA allocation or initialization.
        # torch.autocast("cuda") would disable itself in CPU-only CI.
        torch.set_autocast_enabled("cuda", autocast)
        torch.set_autocast_dtype("cuda", dtype)
        with sdpa_kernel(SDPBackend.MATH):
            assert enabled() == (False, False, False, True)
            if raises:
                with pytest.raises(RuntimeError) as caught:
                    fit_icl(
                        model,
                        full,
                        full,
                        SimpleNamespace(lengths=(1,)),
                        exchange,
                    )
                assert caught.value is primary
            else:
                assert (
                    fit_icl(
                        model,
                        full,
                        full,
                        SimpleNamespace(lengths=(1,)),
                        exchange,
                    )
                    is full
                )
            assert enabled() == (False, False, False, True)
    finally:
        torch.set_autocast_enabled("cuda", was_enabled)
        torch.set_autocast_dtype("cuda", previous_dtype)
    assert enabled() == before


def test_cpu_fit_leaves_ambient_attention_policy_unchanged() -> None:
    full = SimpleNamespace(is_cuda=False)
    exchange = SimpleNamespace(gather=lambda *args: full)

    def model(*args: object, **kwargs: object) -> object:
        assert enabled() == (False, False, False, True)
        return full

    before = enabled()
    with sdpa_kernel(SDPBackend.MATH):
        assert (
            fit_icl(model, full, full, SimpleNamespace(lengths=(1,)), exchange)
            is full
        )
        assert enabled() == (False, False, False, True)
    assert enabled() == before
