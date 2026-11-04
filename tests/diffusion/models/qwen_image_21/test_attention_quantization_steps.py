# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise step selectors through both Qwen2.1 denoising entry points."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import get_forward_context, override_forward_context, set_forward_context
from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel
from vllm_omni.diffusion.step_policy import ExactStep, resolve_step_execution_key
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, _is_w4a8_fallback_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _attention(config: OmniDiffusionConfig) -> Attention:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._kv_cache_dtype = config.diffusion_kv_cache_dtype
    layer._kv_cache_skip_steps = getattr(config, "diffusion_kv_cache_skip_step_indices", None)
    layer._kv_cache_skip_layers = None
    layer._disable_kv_quant = False
    layer.layer_idx = 0
    layer.attention = FlashAttentionImpl.__new__(FlashAttentionImpl)
    return layer


class _Transformer:
    causal_condition = False
    transformer_blocks = ()

    def __init__(self) -> None:
        self.releases = 0

    def release_request_geometry(self) -> None:
        self.releases += 1


class _Scheduler:
    def set_begin_index(self, index: int) -> None:
        assert index == 0


class _Loop(QwenImage21CFGParallelMixin):
    def __init__(self, attention: Attention, fail_at: int | None = None) -> None:
        self.attention = attention
        self.transformer = _Transformer()
        self.scheduler = _Scheduler()
        self.interrupt = False
        self._progress_bar_config = {"disable": True}
        self.seen: list[tuple[int | None, str]] = []
        self.fail_at = fail_at

    def predict_noise(self, **kwargs: object) -> torch.Tensor:
        step = get_forward_context().denoise_step_idx
        metadata = self.attention._with_kv_cache_dtype(None)
        self.seen.append((step, metadata.extra["kv_cache_dtype"]))
        if step == self.fail_at and self.fail_at is not None:
            raise RuntimeError("denoise failed")
        hidden_states = kwargs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        return hidden_states

    def scheduler_step_maybe_with_cfg(self, noise_pred, timestep, latents, do_true_cfg):
        return latents


def _diffuse(pipe: _Loop, do_true_cfg: bool = False) -> torch.Tensor:
    return pipe.diffuse(
        prompt_embeds=torch.zeros(1, 1, 4),
        prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool),
        negative_prompt_embeds=torch.zeros(1, 1, 4) if do_true_cfg else None,
        negative_prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool) if do_true_cfg else None,
        img_mask=torch.ones(1, 3, dtype=torch.bool),
        negative_img_mask=torch.ones(1, 3, dtype=torch.bool) if do_true_cfg else None,
        latents=torch.zeros(1, 2, 4),
        image_latents=None,
        img_shapes=[[(1, 1, 2)]],
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        do_true_cfg=do_true_cfg,
        true_cfg_scale=4.0 if do_true_cfg else 1.0,
    )


@pytest.mark.parametrize("do_true_cfg", [False, True])
@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_loop_publishes_step_indices_and_restores_between_requests(do_true_cfg, previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _Loop(_attention(config))
    expected = [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")]
    if do_true_cfg:
        expected = [entry for entry in expected for _ in range(2)]
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        for _ in range(2):
            pipe.seen.clear()
            _diffuse(pipe, do_true_cfg)
            assert pipe.seen == expected
            assert get_forward_context().denoise_step_idx == previous_step
    assert pipe.transformer.releases == 2


@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_loop_restores_step_context_after_failure(previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _Loop(_attention(config), fail_at=1)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        with pytest.raises(RuntimeError, match="denoise failed"):
            _diffuse(pipe)
        assert get_forward_context().denoise_step_idx == previous_step
    assert pipe.transformer.releases == 1


def _state(request_id: str, step: int, fill: float, rows: int = 1) -> StepRequestState:
    return StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(true_cfg_scale=1.0),
        prompt_embeds=torch.ones(rows, 1, 4),
        prompt_embeds_mask=torch.ones(rows, 1, dtype=torch.bool),
        latents=torch.full((rows, 2, 4), fill),
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        step_index=step,
        img_shapes=[[(1, 1, 2)]] * rows,
        extra={"image_pad_mask": torch.ones(rows, 1, dtype=torch.bool), "image_latents": None, "kv_cache": None},
    )


def _step_pipeline(config: OmniDiffusionConfig) -> QwenImage21Pipeline:
    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipe)
    pipe.od_config = config
    transformer = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    nn.Module.__init__(transformer)
    transformer.quant_config = None
    transformer.causal_condition = False
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.attn = _attention(config)
    transformer.transformer_blocks = nn.ModuleList([block])
    pipe.transformer = transformer
    pipe._interrupt = False
    pipe._attention_kwargs = {}
    return pipe


@pytest.mark.parametrize("step,expected", [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")])
def test_step_mode_publishes_request_step(monkeypatch, step, expected):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    seen = []

    def predict(*args, **kwargs):
        seen.append(
            (get_forward_context().denoise_step_idx, attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"])
        )
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch([_state("a", step, 1.0)]))
    assert result.shape == (1, 2, 4)
    assert seen == [(step, expected)]


def test_distinct_non_skipped_decode_steps_share_forward(monkeypatch):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0, rows=2), _state("c", 3, 3.0)]
    for state in states:
        state.extra["prefill_prompt_seq_len"] = 1
        state.extra["kv_cache"] = [
            {"cond": {part: torch.zeros(state.latents.shape[0], 1, 4) for part in ("key", "value")}}
        ]
    seen = []
    timesteps = []

    def predict(*args, **kwargs):
        positive = args[2]
        metadata = attention._with_kv_cache_dtype(None)
        seen.append((get_forward_context().denoise_step_idx, metadata.extra["kv_cache_dtype"]))
        timesteps.append(positive["timestep"])
        return positive["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx is None
    assert seen == [(None, "mxfp8")]
    # Each row retains its own scheduler timestep, including multi-row requests.
    torch.testing.assert_close(timesteps[0], torch.tensor([0.7, 0.2, 0.2, 0.001]))
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))


@pytest.mark.parametrize("skip_steps", [None, "0,2"])
@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_step_mode_separates_quantization_steps_without_reordering_rows(monkeypatch, skip_steps, previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps=skip_steps)
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0, rows=2), _state("c", 1, 3.0)]
    seen = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        seen.append(
            (
                get_forward_context().denoise_step_idx,
                attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"],
                hidden.shape[0],
            )
        )
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx == previous_step
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))
    expected: list[tuple[int | None, str, int]]
    if skip_steps:
        expected = [(1, "mxfp8", 2), (2, "float", 2)]
    else:
        expected = [(None, "mxfp8", 4)]
    assert seen == expected


@pytest.mark.parametrize("dtype", [None, "auto", "float"])
@pytest.mark.parametrize("different_prefix_lengths", [False, True])
def test_native_decode_ignores_step_selector_but_preserves_prefix_groups(monkeypatch, dtype, different_prefix_lengths):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype=dtype, diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0, rows=2), _state("c", 3, 3.0)]
    for state in states:
        rows = state.latents.shape[0]
        prefix_length = 2 if different_prefix_lengths and state.request_id == "b" else 1
        state.latents = state.latents.to(torch.bfloat16)
        state.extra["prefill_prompt_seq_len"] = prefix_length
        state.extra["kv_cache"] = [{"cond": {part: torch.zeros(rows, prefix_length, 4) for part in ("key", "value")}}]
    forward_rows = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        assert hidden.dtype == torch.bfloat16
        forward_rows.append(hidden[:, 0, 0].tolist())
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx is None
    assert forward_rows == ([[1.0, 3.0], [2.0, 2.0]] if different_prefix_lengths else [[1.0, 2.0, 2.0, 3.0]])
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.bfloat16))


@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_step_mode_restores_context_after_failure(monkeypatch, previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)

    def predict(*args, **kwargs):
        assert get_forward_context().denoise_step_idx == 1
        raise RuntimeError("denoise failed")

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        with pytest.raises(RuntimeError, match="denoise failed"):
            pipe.denoise_step(InputBatch.make_batch([_state("a", 1, 1.0)]))
        assert get_forward_context().denoise_step_idx == previous_step


def _entry_point(monkeypatch, config, mode, predict):
    pipe = _Loop(_attention(config)) if mode == "request" else _step_pipeline(config)
    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    if mode == "request":
        return lambda: _diffuse(pipe)
    return lambda: pipe.denoise_step(InputBatch.make_batch([_state("inner", 1, 1.0)]))


@pytest.mark.parametrize("mode", ["request", "step"])
@pytest.mark.parametrize("failure", [None, "activation", "forward"])
@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_restoration_does_not_reactivate_runtime(monkeypatch, mode, failure, previous_step):
    config = OmniDiffusionConfig()
    runtime = Mock()
    if failure == "activation":
        runtime.ensure_active.side_effect = RuntimeError("activation failed")

    def predict(*args, **kwargs):
        if failure == "forward":
            raise RuntimeError("forward failed")
        return args[2]["hidden_states"]

    run = _entry_point(monkeypatch, config, mode, predict)
    with set_forward_context(denoise_step_idx=previous_step, paged_kv_runtime=runtime):
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                run()
        else:
            run()
        assert get_forward_context().denoise_step_idx == previous_step
    expected_steps = [0, 1, 2, 3] if mode == "request" else [1]
    if failure:
        expected_steps = expected_steps[:1]
    assert [call.args[0] for call in runtime.ensure_active.call_args_list] == expected_steps


@pytest.mark.parametrize("mode", ["request", "step"])
@pytest.mark.parametrize("inner_fails", [False, True])
def test_nested_calls_restore_outer_step(monkeypatch, mode, inner_fails):
    config = OmniDiffusionConfig()

    def inner_predict(*args, **kwargs):
        assert get_forward_context().denoise_step_idx in (0, 1, 2, 3)
        if inner_fails:
            raise RuntimeError("inner failed")
        return args[2]["hidden_states"]

    inner = _entry_point(monkeypatch, config, mode, inner_predict)

    def outer_predict(*args, **kwargs):
        outer_step = get_forward_context().denoise_step_idx
        if inner_fails:
            with pytest.raises(RuntimeError, match="inner failed"):
                inner()
        else:
            inner()
        assert get_forward_context().denoise_step_idx == outer_step
        return args[2]["hidden_states"]

    outer = _entry_point(monkeypatch, config, mode, outer_predict)
    with set_forward_context(denoise_step_idx=9):
        outer()
        assert get_forward_context().denoise_step_idx == 9


@pytest.mark.parametrize("mode", ["request", "step"])
def test_entry_without_forward_context(monkeypatch, mode):
    def predict(*args, **kwargs):
        return args[2]["hidden_states"]

    run = _entry_point(monkeypatch, OmniDiffusionConfig(), mode, predict)
    with override_forward_context(None):
        assert run().shape == (1, 2, 4)


def test_step_mode_groups_mixed_prefix_lengths_and_steps(monkeypatch):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0), _state("c", 2, 3.0), _state("d", 1, 4.0, rows=2)]
    for state in states:
        prefix_length = 2 if state.request_id == "b" else 1
        state.extra["prefill_prompt_seq_len"] = prefix_length
        state.extra["kv_cache"] = [
            {"cond": {part: torch.zeros(state.latents.shape[0], prefix_length, 4) for part in ("key", "value")}}
        ]
    assert pipe._needs_decode_seq_len_split(states)
    assert len({state.step_index for state in states}) > 1
    seen = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        seen.append(
            (
                get_forward_context().denoise_step_idx,
                attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"],
                hidden[:, 0, 0].tolist(),
            )
        )
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx is None
    assert seen == [(1, "mxfp8", [1.0, 4.0, 4.0]), (2, "float", [2.0]), (2, "float", [3.0])]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0]))


def test_step_mode_accepts_config_without_parsed_step_selector(monkeypatch):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8")
    # Remove both the instance field and its dataclass default to exercise a
    # config producer that does not expose the optional parsed selector.
    monkeypatch.delattr(config, "diffusion_kv_cache_skip_step_indices")
    monkeypatch.delattr(OmniDiffusionConfig, "diffusion_kv_cache_skip_step_indices")
    pipe = _step_pipeline(config)
    calls = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        calls.append(hidden.shape[0])
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch([_state("a", 1, 1.0), _state("b", 2, 2.0)]))
    assert calls == [2]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0]))


def _decode_states():
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0, rows=2), _state("c", 3, 3.0)]
    for state in states:
        state.extra["prefill_prompt_seq_len"] = 1
        state.extra["kv_cache"] = [
            {"cond": {part: torch.zeros(state.latents.shape[0], 1, 4) for part in ("key", "value")}}
        ]
    return states


@pytest.mark.parametrize("dtype,skip", [("mxfp8", "0"), ("mxfp8", None), (None, "0"), ("float", None)])
def test_grouping_preserves_linear_fallback_independently_of_attention(monkeypatch, dtype, skip):
    from vllm_omni.quantization import mxfp4_config
    from vllm_omni.quantization.mxfp8_config import MXFPLinearMethodBase

    config = OmniDiffusionConfig(diffusion_kv_cache_dtype=dtype, diffusion_kv_cache_skip_steps=skip)
    pipe = _step_pipeline(config)
    quant = DiffusionMXFP4Config(w4a8_fallback_steps=[2])
    pipe.transformer.quant_config = quant
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    linear = mxfp4_config.NPUMxfp4LinearMethod(quant)
    layer = nn.Linear(4, 4, bias=False)
    layer.weight_scale = torch.ones(1)
    branches = []

    def a4(self, layer, x, bias, ori_dtype):
        branches.append("A4")
        return x + 4

    def a8(x, weight, scale, bias, ori_dtype):
        branches.append("A8")
        return x + 8

    # Keep the real Linear branch selection; replace only device execution.
    monkeypatch.setattr(MXFPLinearMethodBase, "_apply_inner", a4)
    monkeypatch.setattr(mxfp4_config, "_npu_w4a8_matmul", a8)
    seen = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        metadata = attention._with_kv_cache_dtype(None)
        seen.append((get_forward_context().denoise_step_idx, metadata.extra["kv_cache_dtype"] if metadata else None))
        return linear._apply_inner(layer, hidden, None, hidden.dtype)

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9):
        result = pipe.denoise_step(InputBatch.make_batch(_decode_states()))
        assert get_forward_context().denoise_step_idx == 9
    expected_dtype = "mxfp8" if dtype == "mxfp8" else None
    assert seen == [(None, expected_dtype), (2, expected_dtype)]
    assert branches == ["A4", "A8"]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([5.0, 10.0, 10.0, 7.0]))


@pytest.mark.parametrize("consumer", ["linear", "attention", "transformer", "runtime"])
def test_undeclared_consumers_keep_exact_steps(monkeypatch, consumer):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    runtime = None
    if consumer == "linear":
        pipe.transformer.quant_config = object()
    elif consumer == "attention":
        pipe.transformer.transformer_blocks[0].attn.attn.attention = object()
    elif consumer == "transformer":
        pipe.transformer = nn.Module()
    else:
        runtime = object()
    seen = []

    def predict(*args, **kwargs):
        seen.append(get_forward_context().denoise_step_idx)
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, paged_kv_runtime=runtime):
        result = pipe.denoise_step(InputBatch.make_batch(_decode_states()))
    assert seen == [1, 2, 3]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))


@pytest.mark.parametrize("disable,skip_layers", [(True, None), (False, {0})])
def test_fixed_layer_opt_out_does_not_split_steps(monkeypatch, disable, skip_layers):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    attention._disable_kv_quant = disable
    attention._kv_cache_skip_layers = skip_layers
    seen = []

    def predict(*args, **kwargs):
        seen.append((args[2]["hidden_states"].shape[0], attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"]))
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        pipe.denoise_step(InputBatch.make_batch(_decode_states()))
    assert seen == [(4, "float")]


def test_policy_queries_do_not_publish_steps_or_activate_runtime():
    class Runtime:
        def ensure_active(self, step):
            pytest.fail("policy query must not activate paged KV")

    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = pipe.transformer.transformer_blocks[0].attn.attn
    # A future backend-supported dtype passes through without changing batching.
    attention._kv_cache_dtype = "future_quant_dtype"
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9, paged_kv_runtime=Runtime()):
        first = pipe.transformer.get_execution_policy(1)
        assert first == pipe.transformer.get_execution_policy(3)
        assert first != pipe.transformer.get_execution_policy(2)
        assert get_forward_context().denoise_step_idx == 9
        assert attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"] == "future_quant_dtype"


@pytest.mark.parametrize("step", [None, -1, True, 1.5])
def test_execution_keys_reject_unknown_or_invalid_steps(step):
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_step_execution_key(None, step)


def test_step_runtime_and_unknown_backend_declare_different_contracts():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl
    from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend

    runtime = DiffusionKVModelRunnerBackend.__new__(DiffusionKVModelRunnerBackend)
    assert resolve_step_execution_key(runtime, 1) == resolve_step_execution_key(runtime, 3)
    assert AttentionImpl.get_step_execution_key(object(), 1) == ExactStep(1)
    assert AttentionImpl.get_step_execution_key(object(), 3) == ExactStep(3)


@pytest.mark.parametrize("steps", [[], [0], [2], [1, 3]])
@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_linear_query_matches_execution_selector(steps, step):
    quant = DiffusionMXFP4Config(w4a8_fallback_steps=steps)
    with set_forward_context(denoise_step_idx=step):
        assert quant.get_step_execution_key(step) == _is_w4a8_fallback_step(steps)


def test_merged_steps_share_cfg_policy_and_keep_both_timestep_vectors(monkeypatch):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    states = _decode_states()
    for state in states:
        state.do_true_cfg = True
        state.sampling.true_cfg_scale = 4.0
        state.negative_prompt_embeds = torch.ones_like(state.prompt_embeds)
        state.negative_prompt_embeds_mask = torch.ones_like(state.prompt_embeds_mask)
        state.extra["negative_image_pad_mask"] = torch.ones(state.latents.shape[0], 1, dtype=torch.bool)
        state.extra["prefill_negative_prompt_seq_len"] = 1
        state.extra["kv_cache"][0]["uncond"] = {
            part: torch.zeros(state.latents.shape[0], 1, 4) for part in ("key", "value")
        }
    seen = []

    def predict(**kwargs):
        seen.append((get_forward_context().denoise_step_idx, kwargs["cache_branch"]))
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.7, 0.2, 0.2, 0.001]))
        return kwargs["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx == 9
    assert seen == [(None, "cond"), (None, "uncond")]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))


def test_encoder_only_quantization_does_not_split_transformer_steps(monkeypatch):
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig, resolve_component_quant_config

    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    config.quantization_config = ComponentQuantizationConfig(
        {"text_encoder": DiffusionMXFP4Config(w4a8_fallback_steps=[2])}
    )
    pipe = _step_pipeline(config)
    pipe.transformer.quant_config = resolve_component_quant_config(config.quantization_config, "transformer")
    calls = []

    def predict(*args, **kwargs):
        calls.append(args[2]["hidden_states"].shape[0])
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        pipe.denoise_step(InputBatch.make_batch(_decode_states()))
    assert calls == [4]


def test_policy_grouping_does_not_admit_mixed_kv_phases():
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    states = [_state("prefill", 0, 1.0), _decode_states()[0]]
    with set_forward_context(omni_diffusion_config=config):
        with pytest.raises(ValueError, match="mixed KV-cache phases"):
            pipe.denoise_step(InputBatch.make_batch(states))


@pytest.mark.parametrize("mode", ["request", "step"])
@pytest.mark.parametrize("failure", [False, True])
def test_policy_scope_restores_outer_policy(monkeypatch, mode, failure):
    from vllm_omni.diffusion.step_policy import DenoiseExecutionPolicy

    previous = DenoiseExecutionPolicy()

    def predict(*args, **kwargs):
        assert get_forward_context().denoise_execution_policy is not previous
        if failure:
            raise RuntimeError("policy scope failed")
        return args[2]["hidden_states"]

    run = _entry_point(monkeypatch, OmniDiffusionConfig(), mode, predict)
    with set_forward_context(denoise_step_idx=9):
        context = get_forward_context()
        context.denoise_execution_policy = previous
        if failure:
            with pytest.raises(RuntimeError, match="policy scope failed"):
                run()
        else:
            run()
        assert context.denoise_execution_policy is previous
        assert context.denoise_step_idx == 9


@pytest.mark.parametrize("fallback_steps", [None, []])
def test_linear_without_step_fallback_keeps_native_mixed_batch(monkeypatch, fallback_steps):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = OmniDiffusionConfig(diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    pipe.transformer.quant_config = DiffusionMXFP4Config(w4a8_fallback_steps=fallback_steps)
    seen = []

    def predict(*args, **kwargs):
        seen.append(get_forward_context().denoise_step_idx)
        torch.testing.assert_close(args[2]["timestep"], torch.tensor([0.7, 0.2, 0.2, 0.001]))
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9):
        result = pipe.denoise_step(InputBatch.make_batch(_decode_states()))
        assert get_forward_context().denoise_step_idx == 9
    assert seen == [None]
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))


def test_step_grouping_rejects_mixed_kv_phases():
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    states = _decode_states()
    states[0].extra["kv_cache"] = None
    with set_forward_context(omni_diffusion_config=config):
        with pytest.raises(ValueError, match="mixed KV-cache phases"):
            pipe.denoise_step(InputBatch.make_batch(states))
