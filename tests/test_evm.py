"""Unit tests for evm.py — CPU Eulerian Video Magnification."""

import sys
import os

import cv2
import numpy as np
import pytest

# Add project root to path so we can import evm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_seconds_only(self):
        assert evm.format_duration(30.0) == "30.0s"

    def test_minutes_and_seconds(self):
        assert evm.format_duration(90.5) == "1m 30.5s"

    def test_zero(self):
        assert evm.format_duration(0) == "0.0s"

    def test_exactly_60(self):
        assert evm.format_duration(60.0) == "1m 0.0s"


# ---------------------------------------------------------------------------
# Tier 1: Strict tolerance (atol=1e-6)
# ---------------------------------------------------------------------------

class TestColorConversion:
    """rgb_to_yiq / yiq_to_rgb roundtrip — should recover original."""

    def test_roundtrip_single_frame(self):
        rng = np.random.RandomState(42)
        frame = rng.rand(64, 64, 3).astype(np.float32)
        recovered = evm.yiq_to_rgb(evm.rgb_to_yiq(frame))
        np.testing.assert_allclose(recovered, frame, atol=1e-6)

    def test_roundtrip_black(self):
        frame = np.zeros((8, 8, 3), dtype=np.float32)
        recovered = evm.yiq_to_rgb(evm.rgb_to_yiq(frame))
        np.testing.assert_allclose(recovered, frame, atol=1e-6)

    def test_roundtrip_white(self):
        frame = np.ones((8, 8, 3), dtype=np.float32)
        recovered = evm.yiq_to_rgb(evm.rgb_to_yiq(frame))
        np.testing.assert_allclose(recovered, frame, atol=1e-6)

    def test_yiq_y_channel_is_luminance(self):
        """Y channel should be weighted sum of RGB (0.299R + 0.587G + 0.114B)."""
        frame = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)  # pure red
        yiq = evm.rgb_to_yiq(frame)
        assert abs(yiq[0, 0, 0] - 0.299) < 1e-6

    def test_output_dtype_is_float32(self):
        frame = np.random.rand(4, 4, 3).astype(np.float32)
        assert evm.rgb_to_yiq(frame).dtype == np.float32
        assert evm.yiq_to_rgb(frame).dtype == np.float32


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
        evm_cuda = pytest.importorskip("evm_cuda")
        result = evm_cuda.estimate_vram_bytes(100, 480, 640, 4)
        assert result == 673_920_000

    def test_single_frame(self):
        evm_cuda = pytest.importorskip("evm_cuda")
        result = evm_cuda.estimate_vram_bytes(1, 100, 100, 2)
        # Level 0: 1 * 100 * 100 * 12 = 120,000
        # Level 1: 1 * 50 * 50 * 12 = 30,000
        # FFT buffer: 1 * 50 * 50 * 24 = 60,000
        assert result == 210_000


# ---------------------------------------------------------------------------
# Tier 2: Moderate tolerance (atol=1e-4)
# ---------------------------------------------------------------------------

class TestIdealBandpassFilter:
    """Feed known-frequency signals, verify passband behavior."""

    def test_passes_in_band_signal(self):
        """A 2Hz sine wave with bandpass 1-3Hz should survive."""
        fps = 30.0
        n_frames = 300
        t = np.arange(n_frames) / fps

        # 2Hz sine wave, single pixel, 3 channels
        signal = np.sin(2 * np.pi * 2.0 * t).astype(np.float32)
        data = signal.reshape(n_frames, 1, 1, 1) * np.ones((1, 1, 1, 3), dtype=np.float32)

        filtered = evm.ideal_bandpass_filter(data, fps, 1.0, 3.0)

        # One-sided mask (matching MATLAB reference) passes ~25% of energy
        # for a pure sine: only positive freq half, real part of IFFT
        input_energy = np.sum(data ** 2)
        output_energy = np.sum(filtered ** 2)
        assert output_energy / input_energy > 0.2  # at least 20% energy preserved

    def test_rejects_out_of_band_signal(self):
        """A 10Hz sine wave with bandpass 1-3Hz should be zeroed."""
        fps = 30.0
        n_frames = 300
        t = np.arange(n_frames) / fps

        signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        data = signal.reshape(n_frames, 1, 1, 1) * np.ones((1, 1, 1, 3), dtype=np.float32)

        filtered = evm.ideal_bandpass_filter(data, fps, 1.0, 3.0)

        # Out-of-band energy should be near zero
        input_energy = np.sum(data ** 2)
        output_energy = np.sum(filtered ** 2)
        assert output_energy / input_energy < 0.01  # less than 1% leaks through

    def test_output_shape_and_dtype(self):
        data = np.random.rand(60, 4, 4, 3).astype(np.float32)
        filtered = evm.ideal_bandpass_filter(data, 30.0, 1.0, 5.0)
        assert filtered.shape == data.shape
        assert filtered.dtype == np.float32

    def test_dc_signal_rejected(self):
        """A constant (DC) signal should be completely rejected by bandpass."""
        data = np.ones((60, 2, 2, 3), dtype=np.float32)
        filtered = evm.ideal_bandpass_filter(data, 30.0, 1.0, 5.0)
        np.testing.assert_allclose(filtered, 0.0, atol=1e-6)


class TestLaplacianPyramid:
    """Build pyramid then collapse — reconstruction should approximate original."""

    def test_roundtrip_reconstruction(self):
        """Build Laplacian pyramid of one frame, collapse it, compare to original."""
        rng = np.random.RandomState(42)
        # Use power-of-2 dimensions for clean pyrDown/pyrUp
        frame = rng.rand(64, 64, 3).astype(np.float32)

        # Build Gaussian pyramid
        n_levels = 4
        gauss = [frame]
        for _ in range(1, n_levels):
            gauss.append(cv2.pyrDown(gauss[-1]))

        # Build Laplacian pyramid
        lap = []
        for i in range(n_levels - 1):
            up = cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0]))
            lap.append(gauss[i] - up)
        lap.append(gauss[-1])

        # Collapse
        reconstructed = evm.collapse_laplacian_pyramid(lap)

        # Reconstruction error should be tiny (float32 precision)
        error = np.max(np.abs(reconstructed - frame))
        assert error < 1e-4, f"Max reconstruction error: {error}"

    def test_pyramid_level_shapes(self):
        """Verify create_laplacian_video_pyramid produces correct shapes."""
        rng = np.random.RandomState(42)
        video = rng.rand(10, 32, 32, 3).astype(np.float32)
        pyramid = evm.create_laplacian_video_pyramid(video, 3)

        assert len(pyramid) == 3
        assert pyramid[0].shape == (10, 32, 32, 3)  # level 0: full res
        assert pyramid[1].shape == (10, 16, 16, 3)  # level 1: half
        assert pyramid[2].shape == (10, 8, 8, 3)    # level 2: quarter

    def test_pyramid_values_finite(self):
        """No NaN or Inf in pyramid output."""
        rng = np.random.RandomState(42)
        video = rng.rand(5, 16, 16, 3).astype(np.float32)
        pyramid = evm.create_laplacian_video_pyramid(video, 3)

        for level in pyramid:
            assert np.all(np.isfinite(level)), "Pyramid contains NaN or Inf"

    def test_pyramid_dtype(self):
        video = np.random.rand(5, 16, 16, 3).astype(np.float32)
        pyramid = evm.create_laplacian_video_pyramid(video, 2)
        for level in pyramid:
            assert level.dtype == np.float32
