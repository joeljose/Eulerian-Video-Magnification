"""Unit tests for evm.py — CPU Eulerian Video Magnification."""

import subprocess
import sys
import os
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Bug fix: load_video buffer guard
# ---------------------------------------------------------------------------

class TestLoadVideoBufferGuard:
    """Verify load_video doesn't crash when CAP_PROP_FRAME_COUNT is wrong."""

    def test_frame_count_too_low(self):
        """If reported frame_count < actual frames, should cap at reported count."""
        actual_frames = 10
        reported_count = 5
        h, w = 8, 8
        fake_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(actual_frames)]
        call_idx = [0]

        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: reported_count,
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FPS: 30.0,
        }[prop]
        mock_cap.isOpened.return_value = True

        def mock_read():
            if call_idx[0] < actual_frames:
                frame = fake_frames[call_idx[0]]
                call_idx[0] += 1
                return True, frame
            return False, None

        mock_cap.read.side_effect = mock_read

        with patch("cv2.VideoCapture", return_value=mock_cap):
            video, fps = evm.load_video("fake_path.mp4")

        # Should have exactly reported_count frames, not actual_frames
        assert video.shape[0] == reported_count
        assert fps == 30.0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

# Path to evm.py from project root
EVM_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evm.py")


def run_evm(*args):
    """Run evm.py with given args and return (returncode, stderr)."""
    result = subprocess.run(
        [sys.executable, EVM_SCRIPT] + list(args),
        capture_output=True, text=True
    )
    return result.returncode, result.stderr


@pytest.fixture
def dummy_video(tmp_path):
    """Create a tiny valid file to pass the file-exists check."""
    p = tmp_path / "dummy.mp4"
    p.write_bytes(b"\x00" * 100)
    return str(p)


class TestInputValidation:
    """Test that invalid arguments are rejected with exit code 1."""

    def test_nonexistent_input_file(self):
        code, stderr = run_evm("-i", "nonexistent_file.mp4")
        assert code == 1
        assert "not found" in stderr

    def test_freq_low_zero(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "-fl", "0")
        assert code == 1
        assert "--freq-low must be positive" in stderr

    def test_freq_low_negative(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "-fl", "-1")
        assert code == 1
        assert "--freq-low must be positive" in stderr

    def test_freq_high_less_than_freq_low(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "-fl", "5", "-fh", "2")
        assert code == 1
        assert "--freq-high must be greater" in stderr

    def test_amplification_zero(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "-a", "0")
        assert code == 1
        assert "--amplification must be positive" in stderr

    def test_pyramid_levels_one(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "--pyramid-levels", "1")
        assert code == 1
        assert "--pyramid-levels must be at least 2" in stderr

    def test_lambda_c_zero(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "--lambda-c", "0")
        assert code == 1
        assert "--lambda-c must be positive" in stderr

    def test_chrom_attenuation_above_one(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "--chrom-attenuation", "1.5")
        assert code == 1
        assert "--chrom-attenuation must be between" in stderr

    def test_chrom_attenuation_negative(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "--chrom-attenuation", "-0.1")
        assert code == 1
        assert "--chrom-attenuation must be between" in stderr
