# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qkv_norm_rope import mindiesd_qkv_norm_rope, prepare_rope_tables

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


@pytest.mark.parametrize("seq_len", [17, 1024, 4096])
def test_fused_qkv_matches_model_rope(seq_len):
    pytest.importorskip("torch_npu")
    pytest.importorskip("mindiesd")
    if not hasattr(torch.ops.mindiesd, "norm_rope_concat"):
        pytest.skip("MindIE-SD norm_rope_concat is not installed")
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _apply_qwen_image21_rotary_emb_native,
    )
    from vllm_omni.platforms.npu import is_a5

    if not is_a5():
        pytest.skip("This adapter is enabled only on A5")
    generator = torch.Generator().manual_seed(42)
    packed = torch.randn(1, seq_len, 3 * 4 * 128, generator=generator).to(device="npu", dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (4, 128)) for x in packed.chunk(3, -1)]
    w = torch.linspace(0.5, 1.5, 128, device="npu", dtype=q.dtype)
    phase = torch.randn(seq_len, 64, generator=generator).to("npu")
    freqs = torch.polar(torch.ones_like(phase), phase)
    actual = mindiesd_qkv_norm_rope(q, k, v, w, w, 1e-6, prepare_rope_tables(freqs, q.dtype))
    # Match the installed NPU model's RMSNorm, then its original FP32 RoPE.
    for x, out in zip((q, k), actual[:2]):
        normalized = torch.ops.npu.npu_rms_norm(x, w, epsilon=1e-6)[0]
        expected = _apply_qwen_image21_rotary_emb_native(normalized, freqs)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual[2], v, atol=0, rtol=0)
