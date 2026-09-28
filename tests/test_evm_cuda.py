"""GPU tests for evm.py: the same code run on CuPy arrays.

Requires CuPy and an NVIDIA GPU. Run via: ./test.sh gpu
"""

import contextlib
import io
import os
import sys

import cv2
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

# Add project root to path so we can import evm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evm  # noqa: E402


def test_color_roundtrip():
    frame = cp.asarray(np.random.RandomState(42).rand(16, 16, 3).astype(np.float32))
    recovered = evm.yiq_to_rgb(evm.rgb_to_yiq(frame))
    np.testing.assert_allclose(cp.asnumpy(recovered), cp.asnumpy(frame), atol=1e-6)


@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (37, 50)])
def test_pyramid_matches_opencv(shape):
    """Issue #35: the GPU pyramid used different filters from the CPU one."""
    h, w = shape
    x = np.random.RandomState(0).rand(h, w, 3).astype(np.float32)
    down = evm.pyr_down(cp.asarray(x))
    np.testing.assert_allclose(cp.asnumpy(down), cv2.pyrDown(x), atol=1e-5)
    up = evm.pyr_up(down, (h, w))
    np.testing.assert_allclose(cp.asnumpy(up), cv2.pyrUp(cv2.pyrDown(x), dstsize=(w, h)),
                               atol=1e-5)


def test_bandpass_rejects_out_of_band():
    t = cp.arange(300) / 30.0
    data = cp.sin(2 * cp.pi * 10.0 * t).astype(cp.float32).reshape(300, 1, 1, 1)
    filtered = evm.ideal_bandpass_filter(data, 30.0, 1.0, 3.0)
    assert float(cp.sum(filtered ** 2)) / float(cp.sum(data ** 2)) < 0.01


def test_gpu_matches_cpu_end_to_end():
    """Acceptance for #35: CPU vs GPU PSNR >= 60 dB."""
    rng = np.random.RandomState(3)
    video = evm.rgb_to_yiq(rng.rand(60, 48, 64, 3).astype(np.float32))
    args = (30.0, 0.5, 3.0, 20.0)
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = evm.eulerian_magnification(video.copy(), *args, lambda_c=10)
        gpu = cp.asnumpy(evm.eulerian_magnification(cp.asarray(video), *args, lambda_c=10))
    mse = np.mean((cpu - gpu) ** 2)
    assert 10 * np.log10(1.0 / max(mse, 1e-30)) >= 60
