# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression of single-frame cache lifetime and request-local gate selection."""

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21TransformerBlock,
    _select_modulation_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "prefill", "decode", "mixed"])
def test_shared_activated_gate_preserves_batch_and_condition_rows(batch, dtype, mask_kind):
    torch.manual_seed(41)
    seq, width = 17, 128
    mask = None
    if mask_kind != "none":
        mask = torch.ones(seq, dtype=torch.bool)
        if mask_kind == "prefill":
            mask[:7] = False
        elif mask_kind == "mixed":
            mask[::2] = False
    table = torch.randn(batch + (mask is not None), width, dtype=dtype)
    original = table.clone()
    residual = torch.randn(batch, seq, width, dtype=dtype)
    output = torch.randn_like(residual)
    native = residual + _select_modulation_rows(table, mask).tanh() * output
    shared = residual + _select_modulation_rows(table.tanh(), mask) * output
    torch.testing.assert_close(shared, native, rtol=0, atol=0)
    assert torch.equal(table, original)


class _Attention(torch.nn.Module):
    def forward(self, hidden_states, freqs, **kwargs):
        return hidden_states.sin()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_shared_gates_reach_both_residuals_without_leaking_between_forwards(dtype, masked):
    torch.manual_seed(9)
    block = QwenImage21TransformerBlock.__new__(QwenImage21TransformerBlock)
    torch.nn.Module.__init__(block)
    block.img_norm1 = torch.nn.LayerNorm(8, elementwise_affine=False).to(dtype)
    block.img_norm2 = torch.nn.LayerNorm(8, elementwise_affine=False).to(dtype)
    block.attn = _Attention()
    block.img_mlp = torch.nn.SiLU()
    mask = torch.tensor([False, False, True, True, True]) if masked else None
    hidden = torch.randn(2, 5, 8, dtype=dtype)
    freqs = torch.empty(0)
    for _ in range(2):
        modulation = torch.randn(2 + masked, 32, dtype=dtype)
        _, gate1, _, gate2 = modulation.chunk(4, -1)
        shared = tuple(_select_modulation_rows(gate.tanh(), mask) for gate in (gate1, gate2))
        expected = block(hidden, modulation, freqs, mask)
        actual = block(hidden, modulation, freqs, mask, shared_gates=shared)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
