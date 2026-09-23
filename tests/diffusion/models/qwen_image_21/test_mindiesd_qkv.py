# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.layers.fused_norm_rope import FusedNormRope, prepare_rope_tables
from vllm_omni.diffusion.models.qwen_image_21 import qkv_norm_rope as fusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config(enabled=False):
    # Configuration parsing here must not download checkpoint metadata.
    with patch.object(OmniDiffusionConfig, "__post_init__", return_value=None):
        return OmniDiffusionConfig(
            model="Qwen/Qwen-Image-2.1",
            additional_config={"qwen21_mindiesd_qkv_fusion": enabled},
            parallel_config=DiffusionParallelConfig(),
            enforce_eager=True,
        )


def test_disabled_does_not_probe_platform(monkeypatch):
    monkeypatch.setattr(fusion, "current_omni_platform", None)
    assert not fusion.use_mindiesd_qkv(config(), native_cache=True, unquantized=True)


@pytest.mark.parametrize("value", ["false", "true", 1, None])
def test_switch_requires_bool(value):
    with pytest.raises(ValueError, match="must be a bool"):
        fusion.use_mindiesd_qkv(config(value), native_cache=True, unquantized=True)


@pytest.mark.parametrize("field,value", [("enforce_eager", False), ("diffusion_kv_cache_dtype", "fp8")])
def test_unsupported_configuration(monkeypatch, field, value):
    monkeypatch.setattr(fusion.current_omni_platform, "is_npu", lambda: True)
    cfg = config(True)
    setattr(cfg, field, value)
    assert not fusion.use_mindiesd_qkv(cfg, native_cache=True, unquantized=True)


def test_enabled_requires_public_mindiesd_interface(monkeypatch):
    from vllm_omni.platforms import npu

    monkeypatch.setattr(fusion.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(npu, "is_a5", lambda: True)
    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace())
    assert not fusion.use_mindiesd_qkv(config(True), native_cache=True, unquantized=True)
    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(norm_rope_concat=lambda *args: None))
    assert fusion.use_mindiesd_qkv(config(True), native_cache=True, unquantized=True)


def test_rope_preserves_positions_and_pair_order():
    angles = torch.tensor([[0.1, 0.7], [1.2, -0.3]])
    freqs = torch.polar(torch.ones_like(angles), angles)
    sin, cos = prepare_rope_tables(freqs, torch.bfloat16)
    torch.testing.assert_close(sin, angles.sin().repeat_interleave(2, -1).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(cos, angles.cos().repeat_interleave(2, -1).bfloat16(), rtol=0, atol=0)


def test_adapter_contiguous_inputs_and_output_layout(monkeypatch):
    packed = torch.randn(1, 5, 3 * 2 * 128, dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (2, 128)) for x in packed.chunk(3, -1)]
    weight = torch.ones(128, dtype=q.dtype)
    tables = (torch.zeros(5, 128, dtype=q.dtype), torch.ones(5, 128, dtype=q.dtype))
    calls = []

    def op(query, key, value, **kwargs):
        calls.append(kwargs)
        assert all(x.is_contiguous() for x in (query, key, value))
        assert kwargs["norm_type"] == 4 and kwargs["rope_type"] == 1
        assert "encoder_key" not in kwargs
        return tuple(x.transpose(1, 2).contiguous() for x in (query, key, value))

    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(norm_rope_concat=op))
    outputs = FusedNormRope().forward_npu(q, k, v, weight, weight, 1e-6, tables)
    for actual, expected in zip(outputs, (q, k, v)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 1


def test_operator_failure_is_not_silently_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("device kernel failed")

    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(norm_rope_concat=fail))
    x = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)
    w = torch.ones(128, dtype=x.dtype)
    with pytest.raises(RuntimeError, match="device kernel failed"):
        FusedNormRope().forward_npu(x, x, x, w, w, 1e-6, (x[0, :, 0], x[0, :, 0]))


def test_cache_receives_rotated_prefix_once_and_keeps_cfg_branches_separate(monkeypatch):
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as model

    class PackedProjection(torch.nn.Module):
        def forward(self, x):
            return x, None

    class CaptureAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.keys = []

        def forward(self, q, k, v, metadata):
            self.keys.append(k.clone())
            return q

    attention = object.__new__(model.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 1
    attention.head_dim = 128
    attention.prefix_kv_cache_dtype = None
    attention.to_qkv = PackedProjection()
    attention.norm_q = torch.nn.RMSNorm(128, eps=1e-6)
    attention.norm_k = torch.nn.RMSNorm(128, eps=1e-6)
    attention.fused_norm_rope = FusedNormRope()
    attention.attn = CaptureAttention()
    attention.to_out = torch.nn.Identity()
    lengths = []

    def fused(q, k, v, *args):
        lengths.append(q.shape[1])
        return q + 10, k + 20, v

    monkeypatch.setattr(attention.fused_norm_rope, "_forward_method", fused)
    tables = (torch.empty(0), torch.empty(0))
    cache: dict[str, dict[str, torch.Tensor]] = {}
    attention(torch.ones(1, 5, 384), torch.empty(0), kv_cache=cache, cache_write_len=3, qkv_rope_tables=tables)
    prefix = cache["cond"]["key"].clone()
    attention(
        torch.full((1, 5, 384), 2.0),
        torch.empty(0),
        kv_cache=cache,
        cache_branch="uncond",
        cache_write_len=3,
        qkv_rope_tables=tables,
    )
    for _ in range(2):
        attention(torch.full((1, 2, 384), 3.0), torch.empty(0), kv_cache=cache, qkv_rope_tables=tables)
        torch.testing.assert_close(attention.attn.keys[-1][:, :3], prefix, rtol=0, atol=0)
        assert attention.attn.keys[-1].shape[1] == 5
    torch.testing.assert_close(cache["cond"]["key"], prefix, rtol=0, atol=0)
    assert not torch.equal(cache["cond"]["key"], cache["uncond"]["key"])
    assert lengths == [5, 5, 2, 2]
