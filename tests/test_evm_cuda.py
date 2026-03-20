"""Unit tests for evm_cuda.py — GPU Eulerian Video Magnification.

Requires CuPy and an NVIDIA GPU. Run via: ./test.sh gpu
"""

import sys
import os

import numpy as np
import pytest

# Add project root to path so we can import evm_cuda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evm_cuda

cp = pytest.importorskip("cupy")


# ---------------------------------------------------------------------------
# VRAM estimation (pure math — no GPU needed, but lives in evm_cuda)
# ---------------------------------------------------------------------------

class TestEstimateVramBytes:
    """Pure arithmetic — exact equality."""

    def test_known_values(self):
        # 100 frames, 480x640, 4 levels
        # Level 0: 100 * 480 * 640 * 12 = 368,640,000
        # Level 1: 100 * 240 * 320 * 12 = 92,160,000
        # Level 2: 100 * 120 * 160 * 12 = 23,040,000
        # Level 3: 100 * 60 * 80 * 12   = 5,760,000
        # Pyramid total = 489,600,000
        # FFT buffer (level 1): 100 * 240 * 320 * 3 * 8 = 184,320,000
        # Total = 673,920,000
        result = evm_cuda.estimate_vram_bytes(100, 480, 640, 4)
        assert result == 673_920_000

    def test_single_frame(self):
        result = evm_cuda.estimate_vram_bytes(1, 100, 100, 2)
        # Level 0: 1 * 100 * 100 * 12 = 120,000
        # Level 1: 1 * 50 * 50 * 12 = 30,000
        # FFT buffer: 1 * 50 * 50 * 24 = 60,000
        assert result == 210_000


# ---------------------------------------------------------------------------
# GPU color conversion
# ---------------------------------------------------------------------------

class TestGpuColorConversion:
    """Color conversion roundtrip on GPU."""

    @pytest.fixture(autouse=True)
    def init_gpu(self):
        evm_cuda._init_gpu_matrices()

    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        frame_np = rng.rand(16, 16, 3).astype(np.float32)
        frame_gpu = cp.asarray(frame_np)

        yiq = frame_gpu @ evm_cuda._RGB_TO_YIQ.T
        recovered = yiq @ evm_cuda._YIQ_TO_RGB.T
        recovered_np = cp.asnumpy(recovered)

        np.testing.assert_allclose(recovered_np, frame_np, atol=1e-6)


# ---------------------------------------------------------------------------
# GPU pyramid operations
# ---------------------------------------------------------------------------

class TestGpuPyramidOps:
    """GPU pyrDown/pyrUp basic sanity checks."""

    def test_pyr_down_shape(self):
        frame = cp.random.rand(64, 64, 3).astype(cp.float32)
        down = evm_cuda.gpu_pyr_down(frame)
        assert down.shape == (32, 32, 3)

    def test_pyr_up_shape(self):
        frame = cp.random.rand(16, 16, 3).astype(cp.float32)
        up = evm_cuda.gpu_pyr_up(frame, (32, 32))
        assert up.shape == (32, 32, 3)

    def test_pyr_down_values_finite(self):
        frame = cp.random.rand(32, 32, 3).astype(cp.float32)
        down = evm_cuda.gpu_pyr_down(frame)
        assert cp.all(cp.isfinite(down))

    def test_pyr_up_values_finite(self):
        frame = cp.random.rand(16, 16, 3).astype(cp.float32)
        up = evm_cuda.gpu_pyr_up(frame, (32, 32))
        assert cp.all(cp.isfinite(up))


# ---------------------------------------------------------------------------
# GPU bandpass filter
# ---------------------------------------------------------------------------

class TestGpuBandpassFilter:
    """GPU FFT bandpass filter."""

    def test_rejects_out_of_band(self):
        """A 10Hz signal with bandpass 1-3Hz should be zeroed."""
        fps = 30.0
        n_frames = 300
        t = cp.arange(n_frames) / fps

        signal = cp.sin(2 * cp.pi * 10.0 * t).astype(cp.float32)
        data = signal.reshape(n_frames, 1, 1, 1) * cp.ones((1, 1, 1, 3), dtype=cp.float32)

        filtered = evm_cuda.ideal_bandpass_filter(data, fps, 1.0, 3.0)

        input_energy = float(cp.sum(data ** 2))
        output_energy = float(cp.sum(filtered ** 2))
        assert output_energy / input_energy < 0.01

    def test_dc_rejected(self):
        """Constant signal should be rejected."""
        data = cp.ones((60, 2, 2, 3), dtype=cp.float32)
        filtered = evm_cuda.ideal_bandpass_filter(data, 30.0, 1.0, 5.0)
        cp.testing.assert_allclose(filtered, 0.0, atol=1e-6)
