"""Run evm.py's GPU branch on the CPU, with a fake `cupy`.

The fake arrays are NumPy arrays whose type claims to come from cupy, and
the fake cupy/cupyx modules wrap NumPy/SciPy so every result stays a fake
cupy array. That drives _backend() down its CuPy branch, so the GPU glue
(device transfer, --gpu setup, VRAM check, copying back to the host) is
tested in CI without a GPU. Real-GPU tests live in test_evm_cuda.py.
"""

import contextlib
import io
import os
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import scipy.fft
import scipy.ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evm  # noqa: E402


def cupy_installed():
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(cupy_installed(),
                                reason="real CuPy is installed; see test_evm_cuda.py")


class FakeCupyArray(np.ndarray):
    def get(self):
        return np.asarray(self)


FakeCupyArray.__module__ = "cupy"


def _to_fake(value):
    return value.view(FakeCupyArray) if isinstance(value, np.ndarray) else value


def _wrap(module, name):
    """A module whose functions are module's, returning fake cupy arrays."""
    fake = types.ModuleType(name)
    for attr in dir(module):
        value = getattr(module, attr)
        if callable(value) and not isinstance(value, type):
            fake.__dict__[attr] = (lambda f: lambda *a, **k: _to_fake(f(*a, **k)))(value)
        else:
            fake.__dict__[attr] = value
    return fake


@pytest.fixture
def fake_cupy():
    cupy = _wrap(np, "cupy")
    cupy.fft = _wrap(np.fft, "cupy.fft")
    device = types.SimpleNamespace(use=lambda: None, mem_info=(8 * 1024**3, 8 * 1024**3))
    cupy.cuda = types.SimpleNamespace(
        Device=lambda i: device,
        Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None)),
        runtime=types.SimpleNamespace(
            getDeviceProperties=lambda i: {"name": b"Fake GPU"},
            CUDARuntimeError=RuntimeError,
        ),
    )
    cupyx = types.ModuleType("cupyx")
    cupyx.scipy = types.ModuleType("cupyx.scipy")
    cupyx.scipy.fft = _wrap(scipy.fft, "cupyx.scipy.fft")
    cupyx.scipy.ndimage = _wrap(scipy.ndimage, "cupyx.scipy.ndimage")
    modules = {
        "cupy": cupy,
        "cupyx": cupyx,
        "cupyx.scipy": cupyx.scipy,
        "cupyx.scipy.fft": cupyx.scipy.fft,
        "cupyx.scipy.ndimage": cupyx.scipy.ndimage,
    }
    with patch.dict(sys.modules, modules):
        yield cupy


def test_gpu_branch_matches_cpu(fake_cupy):
    rng = np.random.RandomState(0)
    video = evm.rgb_to_yiq(rng.rand(40, 32, 48, 3).astype(np.float32))
    args = (30.0, 0.5, 3.0, 20.0)
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = evm.eulerian_magnification(video.copy(), *args, lambda_c=10)
        gpu = evm.eulerian_magnification(fake_cupy.asarray(video), *args, lambda_c=10)
    assert type(gpu) is FakeCupyArray  # the CuPy branch ran end to end
    np.testing.assert_allclose(gpu.get(), cpu, atol=1e-6)


def test_main_with_gpu_flag(fake_cupy, tmp_path):
    clip = str(tmp_path / "clip.avi")
    evm.save_video(evm.rgb_to_yiq(np.full((40, 16, 16, 3), 0.5, np.float32)), 30.0, clip)
    out = str(tmp_path / "out.avi")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        evm.main(["-i", clip, "-o", out, "--gpu", "--lambda-c", "10"])
    assert "Using GPU: Fake GPU" in stdout.getvalue()
    assert "Estimated VRAM needed" in stdout.getvalue()
    video, fps = evm.load_video(out)
    assert video.shape == (40, 16, 16, 3) and fps == 30.0


def test_evm_cuda_forwards_gpu_flag():
    """The deprecated evm_cuda.py runs evm.py --gpu (CuPy is missing here)."""
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "evm_cuda.py"), "-i", __file__],
        capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "requires CuPy" in result.stderr


def test_pipeline_amplifies_in_band_signal():
    """A Gaussian blob pulsing at 1 Hz comes out with a larger pulse."""
    n, size, fps = 90, 64, 30.0
    t = np.arange(n) / fps
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * 4.0 ** 2))
    rgb = 0.5 + 0.02 * np.sin(2 * np.pi * t)[:, None, None] * blob
    video = evm.rgb_to_yiq(np.repeat(rgb[..., None], 3, axis=3).astype(np.float32))
    c = size // 2
    with contextlib.redirect_stdout(io.StringIO()):
        out = evm.eulerian_magnification(video.copy(), fps, 0.5, 2.0, alpha=20, lambda_c=10)
    assert out[:, c, c, 0].std() > 2 * video[:, c, c, 0].std()
