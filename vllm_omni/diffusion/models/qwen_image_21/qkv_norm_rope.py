# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in MindIE-SD preprocessing; prefix-cache ownership stays in Attention."""

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
RopeTables = tuple[torch.Tensor, torch.Tensor]


def use_mindiesd_qkv(od_config: OmniDiffusionConfig, *, native_cache: bool, unquantized: bool) -> bool:
    enabled = (od_config.additional_config or {}).get("qwen21_mindiesd_qkv_fusion", False)
    if not isinstance(enabled, bool):
        raise ValueError("qwen21_mindiesd_qkv_fusion must be a bool")
    if not enabled:
        return False
    parallel = od_config.parallel_config
    reason = None
    if not current_omni_platform.is_npu():
        reason = "requires A5 NPU"
    elif not od_config.enforce_eager:
        reason = "requires enforce_eager"
    elif parallel.tensor_parallel_size != 1 or parallel.sequence_parallel_size != 1:
        reason = "requires TP1/SP1"
    elif not native_cache or not unquantized or od_config.diffusion_kv_cache_dtype not in (None, "auto"):
        reason = "requires unquantized weights, native prefix cache and native attention"
    else:
        from vllm_omni.platforms.npu import is_a5

        if not is_a5():
            reason = "requires A5 NPU"
        else:
            try:
                import mindiesd  # noqa: F401
            except ImportError:
                reason = "MindIE-SD is unavailable"
            if reason is None and not hasattr(torch.ops.mindiesd, "norm_rope_concat"):
                reason = "MindIE-SD norm_rope_concat is unavailable"
    if reason is not None:
        logger.warning("Qwen2.1 MindIE-SD QKV fusion disabled: %s", reason)
        return False
    return True


def prepare_rope_tables(freqs: torch.Tensor, dtype: torch.dtype) -> RopeTables:
    # Preserve the model's three-axis positions, including the decode target offset.
    # The registered kernel uses the same dtype for Q/K/V, weights and RoPE.
    sin = freqs.imag.repeat_interleave(2, dim=-1).to(dtype).contiguous()
    cos = freqs.real.repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return sin, cos


def mindiesd_qkv_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    tables: RopeTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sin, cos = tables
    outputs = torch.ops.mindiesd.norm_rope_concat(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        norm_query_weight=q_weight.to(query.dtype).contiguous(),
        norm_key_weight=k_weight.to(key.dtype).contiguous(),
        rope_sin=sin,
        rope_cos=cos,
        norm_type=4,
        rope_type=1,
        eps=eps,
    )
    # A view restores Attention's BSND contract; do not rotate cached keys here.
    return outputs[0].transpose(1, 2), outputs[1].transpose(1, 2), outputs[2].transpose(1, 2)
