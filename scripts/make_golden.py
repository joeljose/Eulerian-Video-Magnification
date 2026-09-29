"""Regenerate tests/data/golden_face.npz: a crop of face.mp4 and its output.

Run only when an output change is intended, and note it in CHANGELOG.md:
    python scripts/make_golden.py
"""

import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evm  # noqa: E402

# 48 frames of a 48x48 forehead crop: small enough to keep in git
FRAMES, ROWS, COLS = slice(0, 48), slice(100, 148), slice(220, 268)
FPS = 30.0
PARAMS = dict(freq_min=0.7, freq_max=3.0, alpha=20, pyramid_levels=4,
              lambda_c=10, chrom_attenuation=1.0)


def magnify(frames_bgr, xp=np):
    """Magnify uint8 BGR frames with PARAMS; returns uint8 BGR frames."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = evm.eulerian_magnification(evm.frames_to_yiq(frames_bgr, xp), FPS, **PARAMS)
    return evm.yiq_to_bgr8(out)


if __name__ == "__main__":
    frames, _ = evm.read_frames(os.path.join(ROOT, "face.mp4"))
    crop = np.ascontiguousarray(frames[FRAMES, ROWS, COLS])
    np.savez_compressed(os.path.join(ROOT, "tests", "data", "golden_face.npz"),
                        input=crop, output=magnify(crop))
