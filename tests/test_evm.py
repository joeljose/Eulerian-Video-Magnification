"""Unit tests for evm.py — CPU Eulerian Video Magnification."""

import os
import subprocess
import sys
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
        """If reported frame_count < actual frames, all frames are still read."""
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

        assert video.shape[0] == actual_frames
        assert fps == 30.0

    def test_unopenable_raises(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError, match="cannot open"):
                evm.load_video("fake_path.mp4")

    def test_zero_fps_raises(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 8,
            cv2.CAP_PROP_FPS: 0.0,
        }[prop]
        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError, match="frame rate"):
                evm.load_video("fake_path.mp4")


# ---------------------------------------------------------------------------
# save_video must fail loudly
# ---------------------------------------------------------------------------

class TestSaveVideo:
    def test_writer_not_opened_raises(self, tmp_path):
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        video = np.zeros((2, 8, 8, 3), dtype=np.float32)
        with patch("cv2.VideoWriter", return_value=mock_writer):
            with pytest.raises(RuntimeError, match="could not open"):
                evm.save_video(video, 30.0, str(tmp_path / "out.avi"))

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "out.avi")
        video = np.full((5, 16, 16, 3), 0.5, dtype=np.float32)
        evm.save_video(evm.rgb_to_yiq(video), 30.0, path)
        loaded, fps = evm.load_video(path)
        assert loaded.shape == (5, 16, 16, 3)
        assert fps == 30.0


# ---------------------------------------------------------------------------
# Passband resolution and per-level gains
# ---------------------------------------------------------------------------

class TestPassband:
    def test_band_narrower_than_one_bin_raises(self):
        data = np.zeros((301, 1, 1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="no frequency bins"):
            evm.ideal_bandpass_filter(data, 30.0, 0.83, 0.84)

    def test_passband_freqs(self):
        # 2n-point grid: bins 0.05 Hz apart for 300 frames at 30 fps
        bins = evm.passband_freqs(300, 30.0, 0.5, 1.0)
        np.testing.assert_allclose(bins, np.arange(0.55, 0.96, 0.05))


class TestDrift:
    def test_slow_drift_does_not_become_flicker(self):
        """A linear drift has no in-band content. Without the mirrored
        extension the FFT joins the last frame to the first, and the jump
        leaks into the band (issue #36)."""
        n = 300
        drift = (np.arange(n) / n).astype(np.float32).reshape(n, 1, 1, 1)
        out = evm.ideal_bandpass_filter(drift, 30.0, 0.5, 3.0)
        assert np.abs(out).max() < 0.02


class TestRounding:
    def test_save_video_rounds_to_nearest(self, tmp_path):
        written = []
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer.write.side_effect = lambda f: written.append(f.copy())
        rgb = np.full((1, 2, 2, 3), 100.6 / 255, dtype=np.float32)
        path = tmp_path / "out.avi"
        path.write_bytes(b"x")  # the writer is mocked, so fake its output
        with patch("cv2.VideoWriter", return_value=mock_writer):
            evm.save_video(evm.rgb_to_yiq(rgb), 30.0, str(path))
        assert written[0].min() == 101


class TestFpsOverride:
    def test_fps_argument_replaces_missing_frame_rate(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 3,
            cv2.CAP_PROP_FRAME_WIDTH: 4,
            cv2.CAP_PROP_FRAME_HEIGHT: 4,
            cv2.CAP_PROP_FPS: 0.0,
        }[prop]
        frames = iter([(True, np.zeros((4, 4, 3), np.uint8))] * 3)
        mock_cap.read.side_effect = lambda: next(frames, (False, None))
        with patch("cv2.VideoCapture", return_value=mock_cap):
            video, fps = evm.load_video("fake.mp4", fps=25.0)
        assert fps == 25.0 and video.shape[0] == 3


class TestPyramidMatchesOpenCV:
    """The ndimage pyramid used on the GPU reproduces cv2.pyrDown / pyrUp,
    which the CPU uses (issue #35)."""

    @pytest.mark.parametrize("shape", [(64, 64), (63, 65), (37, 50), (5, 7)])
    def test_pyr_down(self, shape):
        x = np.random.RandomState(0).rand(*shape, 3).astype(np.float32)
        np.testing.assert_allclose(evm._ndimage_pyr_down(x), cv2.pyrDown(x), atol=1e-5)

    @pytest.mark.parametrize("shape", [(64, 64), (63, 65), (37, 50), (5, 7)])
    def test_pyr_up(self, shape):
        h, w = shape
        small = cv2.pyrDown(np.random.RandomState(1).rand(h, w, 3).astype(np.float32))
        np.testing.assert_allclose(evm._ndimage_pyr_up(small, (h, w)),
                                   cv2.pyrUp(small, dstsize=(w, h)), atol=1e-5)

    def test_batched(self):
        video = np.random.RandomState(2).rand(3, 21, 30, 3).astype(np.float32)
        down = evm._ndimage_pyr_down(video)
        np.testing.assert_allclose(evm.pyr_down(video), down, atol=1e-5)
        np.testing.assert_allclose(evm.pyr_up(down, (21, 30)),
                                   evm._ndimage_pyr_up(down, (21, 30)), atol=1e-5)


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
        result = evm.estimate_vram_bytes(100, 480, 640, 4)
        assert result == 673_920_000

    def test_single_frame(self):
        result = evm.estimate_vram_bytes(1, 100, 100, 2)
        # Level 0: 1 * 100 * 100 * 12 = 120,000
        # Level 1: 1 * 50 * 50 * 12 = 30,000
        # FFT buffer: 1 * 50 * 50 * 24 = 60,000
        assert result == 210_000


class TestVersion:
    def test_version_file_matches(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "VERSION")) as f:
            assert f.read().strip() == evm.__version__


class TestLevelAlphas:
    def test_face_defaults(self):
        # Pins current MATLAB-parity behaviour for face.mp4 (528x592) at the
        # defaults; see issue #28 before changing.
        alphas = evm.compute_level_alphas(592, 528, 4, 50, 1000)
        np.testing.assert_allclose(alphas, [0, 4.74, 11.49, 0], atol=0.01)

    def test_small_lambda_c_gives_full_alpha(self):
        alphas = evm.compute_level_alphas(592, 528, 4, 50, 10)
        assert alphas == [0.0, 50, 50, 0.0]


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

    @pytest.mark.parametrize("flag", ["-a", "-fl", "-fh", "--lambda-c", "--fps"])
    @pytest.mark.parametrize("value", ["nan", "inf"])
    def test_non_finite_values(self, dummy_video, flag, value):
        code, stderr = run_evm("-i", dummy_video, flag, value)
        assert code == 1
        assert "must be a finite number" in stderr

    def test_freq_high_above_nyquist(self, tmp_path):
        clip = str(tmp_path / "clip.avi")
        evm.save_video(np.zeros((10, 16, 16, 3), np.float32), 30.0, clip)
        code, stderr = run_evm("-i", clip, "-fl", "1", "-fh", "20")
        assert code == 1
        assert "Nyquist" in stderr

    def test_gpu_without_cupy(self, dummy_video):
        try:
            import cupy  # noqa: F401
            pytest.skip("CuPy is installed")
        except ImportError:
            pass
        code, stderr = run_evm("-i", dummy_video, "--gpu")
        assert code == 1
        assert "requires CuPy" in stderr

    def test_output_dir_missing(self, dummy_video, tmp_path):
        out = str(tmp_path / "missing" / "out.avi")
        code, stderr = run_evm("-i", dummy_video, "-o", out)
        assert code == 1
        assert "output directory does not exist" in stderr

    def test_corrupt_input(self, dummy_video, tmp_path):
        code, stderr = run_evm("-i", dummy_video, "-o", str(tmp_path / "o.avi"))
        assert code == 1
        assert "Traceback" not in stderr
        assert "cannot open video" in stderr

    def test_chrom_attenuation_negative(self, dummy_video):
        code, stderr = run_evm("-i", dummy_video, "--chrom-attenuation", "-0.1")
        assert code == 1
        assert "--chrom-attenuation must be between" in stderr
