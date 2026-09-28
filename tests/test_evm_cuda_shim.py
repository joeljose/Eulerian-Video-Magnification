"""Run evm_cuda's pipeline on the CPU, with numpy/scipy standing in for cupy.

cupyx.scipy mirrors SciPy's API, so this exercises evm_cuda's algorithm and
error handling in CI without a GPU. Real-GPU tests live in test_evm_cuda.py.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.fftpack  # noqa: E402
import scipy.ndimage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evm  # noqa: E402


def _load_evm_cuda_on_cpu():
    fake_cp = types.ModuleType("cupy")
    fake_cp.__dict__.update(
        {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
    )
    fake_cp.asarray = np.asarray
    fake_cp.asnumpy = np.asarray
    fake_cp.cuda = types.SimpleNamespace(
        Stream=types.SimpleNamespace(
            null=types.SimpleNamespace(synchronize=lambda: None)
        )
    )
    cupyx = types.ModuleType("cupyx")
    cupyx.scipy = types.ModuleType("cupyx.scipy")
    cupyx.scipy.fftpack = scipy.fftpack
    cupyx.scipy.ndimage = scipy.ndimage
    fakes = {
        "cupy": fake_cp,
        "cupyx": cupyx,
        "cupyx.scipy": cupyx.scipy,
        "cupyx.scipy.fftpack": scipy.fftpack,
        "cupyx.scipy.ndimage": scipy.ndimage,
    }
    spec = importlib.util.spec_from_file_location(
        "evm_cuda_on_cpu", os.path.join(ROOT, "evm_cuda.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fakes):
        spec.loader.exec_module(module)
    module._init_gpu_matrices()
    return module


evm_cuda = _load_evm_cuda_on_cpu()


def pulsing_blob_video(n_frames=90, fps=30.0, size=64, freq=1.0, amp=0.02):
    """YIQ video: a Gaussian blob whose brightness pulses at `freq` Hz."""
    t = np.arange(n_frames) / fps
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * 4.0 ** 2))
    rgb = 0.5 + amp * np.sin(2 * np.pi * freq * t)[:, None, None] * blob
    rgb = np.repeat(rgb[..., None], 3, axis=3).astype(np.float32)
    return evm.rgb_to_yiq(rgb), fps


@pytest.mark.parametrize("module", [evm, evm_cuda], ids=["cpu", "cuda-shim"])
def test_pipeline_amplifies_in_band_signal(module):
    video, fps = pulsing_blob_video()
    c = video.shape[1] // 2
    before = video[:, c, c, 0].std()
    out = module.eulerian_magnification(
        video.copy(), fps, 0.5, 2.0, alpha=20, lambda_c=10
    )
    after = np.asarray(out)[:, c, c, 0].std()
    assert out.shape == video.shape
    assert after > 2 * before


def test_level_alphas_match_cpu():
    args = (592, 528, 4, 50, 1000)
    assert evm_cuda.compute_level_alphas(*args) == evm.compute_level_alphas(*args)


def test_filter_raises_on_empty_band():
    data = np.zeros((301, 1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="no frequency bins"):
        evm_cuda.ideal_bandpass_filter(data, 30.0, 0.83, 0.85)


def test_save_video_raises_when_writer_not_opened(tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    with patch("cv2.VideoWriter", return_value=mock_writer):
        with pytest.raises(RuntimeError, match="could not open"):
            evm_cuda.save_video(video, 30.0, str(tmp_path / "out.avi"))


def test_load_video_roundtrip(tmp_path):
    path = str(tmp_path / "out.avi")
    video = evm.rgb_to_yiq(np.full((5, 16, 16, 3), 0.5, dtype=np.float32))
    evm_cuda.save_video(video, 30.0, path)
    loaded, fps = evm_cuda.load_video(path)
    assert loaded.shape == (5, 16, 16, 3)
    assert fps == 30.0
