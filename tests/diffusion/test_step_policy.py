# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared policy contracts, independent of any model pipeline or device."""

from dataclasses import FrozenInstanceError

import pytest
from torch import nn

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.step_policy import DenoiseExecutionPolicy
from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, _is_w4a8_fallback_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def attention(dtype):
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._kv_cache_dtype = dtype
    layer._kv_cache_skip_steps = {0, 2}
    layer._kv_cache_skip_layers = None
    layer._disable_kv_quant = False
    layer.layer_idx = 0
    layer.attention = FlashAttentionImpl.__new__(FlashAttentionImpl)
    return layer


@pytest.mark.parametrize("dtype", [None, "auto", "float", "mxfp8", "mxfp4", "future_dtype"])
@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_attention_explicit_choice_matches_legacy_with_no_common_step(dtype, step):
    layer = attention(dtype)
    with set_forward_context(denoise_step_idx=step):
        expected = layer._with_kv_cache_dtype(None)
    policy = DenoiseExecutionPolicy().with_consumer(layer, step)
    with set_forward_context():
        get_forward_context().denoise_execution_policy = policy
        actual = layer._with_kv_cache_dtype(None)
        assert (actual.extra if actual else None) == (expected.extra if expected else None)
        assert get_forward_context().denoise_step_idx is None


@pytest.mark.parametrize("step", [0, 1, 2, 3])
@pytest.mark.parametrize("steps", [[], [0], [2], [1, 3]])
def test_linear_explicit_choice_matches_legacy_with_no_common_step(steps, step):
    config = DiffusionMXFP4Config(w4a8_fallback_steps=steps)
    with set_forward_context(denoise_step_idx=step):
        expected = _is_w4a8_fallback_step(steps)
    policy = DenoiseExecutionPolicy().with_consumer(config, step)
    with set_forward_context():
        get_forward_context().denoise_execution_policy = policy
        assert _is_w4a8_fallback_step(steps, policy_owner=config) == expected


def test_missing_consumer_is_not_silently_reinterpreted_as_legacy():
    layer = attention("mxfp8")
    with set_forward_context(denoise_step_idx=1):
        get_forward_context().denoise_execution_policy = DenoiseExecutionPolicy()
        with pytest.raises(RuntimeError, match="no choice"):
            layer._with_kv_cache_dtype(None)


def test_policy_is_immutable_and_distinguishes_explicit_no_override():
    layer = attention(None)
    empty = DenoiseExecutionPolicy()
    policy = empty.with_consumer(layer, 1)
    assert policy != empty
    assert policy == empty.with_consumer(layer, 3)
    assert len({policy, empty.with_consumer(layer, 3)}) == 1
    with pytest.raises(FrozenInstanceError):
        policy.choices = ()


def test_legacy_unknown_linear_step_still_fails():
    with set_forward_context():
        with pytest.raises(RuntimeError, match="publishes its denoise step"):
            _is_w4a8_fallback_step([2])
