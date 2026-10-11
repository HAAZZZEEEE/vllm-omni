# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression of single-frame cache lifetime and request-local gate selection."""

import pytest
import torch

from tests.diffusion.models.qwen_image_21.test_vae import _make_vae

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("size", [8, 16])
def test_single_frame_decode_without_temporal_cache_is_exact_and_request_independent(batch, size):
    torch.manual_seed(13)
    vae = _make_vae().eval()
    torch.set_num_threads(1)
    z = torch.randn(batch, 4, 1, size, size)
    other = torch.randn_like(z)
    with torch.no_grad():
        expected = vae._decode(z, return_dict=False)[0]
        vae.skip_single_frame_cache = True
        actual = vae._decode(z, return_dict=False)[0]
        vae._decode(other, return_dict=False)
        repeated = vae._decode(z, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    assert all(value is None for value in vae._feat_map)


def test_tiled_single_frame_decode_is_unchanged():
    torch.manual_seed(7)
    vae = _make_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    z = torch.randn(1, 4, 1, 16, 16)
    with torch.no_grad():
        expected = vae.decode(z, return_dict=False)[0]
        vae.skip_single_frame_cache = True
        actual = vae.decode(z, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("skip_cache", [False, True])
def test_multiframe_input_keeps_existing_rejection(skip_cache):
    # This image VAE squeezes the singleton frame axis before its 2D convolutions.
    # The optimization does not introduce video/multiframe support.
    vae = _make_vae().eval()
    vae.skip_single_frame_cache = skip_cache
    with pytest.raises(RuntimeError, match="Expected 3D .* or 4D"):
        vae.decode(torch.randn(1, 4, 2, 8, 8), return_dict=False)
