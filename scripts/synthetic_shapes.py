"""Synthetic pulsating shapes with exact ground truth, for checking the
magnification. Adapted from the harness in Motion-Magnification-Using-2D-DTCWT.

A circle (or square) of radius r(t) = r0 + r1 * sin(2 pi f t) is rendered
with a smooth, camera-like edge (Gaussian blur of `edge_sigma` px), so
sub-pixel sizes are exact. Magnifying by k should give the same shape with
radius r0 + k * r1 * sin(2 pi f t).

The edge position is measured along 64 rays from the centre, and a sine
fitted to it over time gives:
  gain         output amplitude / r1 (ideal: k)
  phase_lag    lag of that sine behind the input, in frames
  angle_spread how much the gain varies around the shape (isotropy)
For outlines (rings), `ghost_energy` is the energy of (output - ideal)
across the line relative to the ideal line's energy.
"""

import contextlib
import io
import os
import sys

import numpy as np
from scipy import ndimage, special

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evm  # noqa: E402

BG, FG = 60.0, 180.0
FPS = 30.0


def evm_magnify(alpha, freq_low=0.5, freq_high=3.0, lambda_c=10, levels=4, fps=FPS):
    """A magnify(frames) function running evm on grayscale frames (0-255).

    The frames go in as the Y channel of a YIQ video with zero chroma.
    """
    def magnify(frames):
        yiq = np.zeros(frames.shape + (3,), np.float32)
        yiq[..., 0] = frames / 255.0
        with contextlib.redirect_stdout(io.StringIO()):
            out = evm.eulerian_magnification(
                yiq, fps, freq_low, freq_high, alpha=alpha,
                pyramid_levels=levels, lambda_c=lambda_c)
        return out[..., 0] * 255.0
    return magnify


def render(shape, radii, size=128, edge_sigma=0.8, thickness=None):
    """Frames (len(radii), size, size) of a centred circle or square.

    With `thickness`, only an outline of that width (px) centred on the
    radius is drawn: a ring or a hollow square.
    """
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    if shape == "circle":
        dist = np.hypot(yy, xx)
    else:  # square: Chebyshev distance (half side = radius)
        dist = np.maximum(np.abs(yy), np.abs(xx))

    def inside(r):
        return 0.5 * special.erfc((dist - r) / (np.sqrt(2) * edge_sigma))

    if thickness is None:
        frames = [BG + (FG - BG) * inside(r) for r in radii]
    else:
        frames = [BG + (FG - BG) * (inside(r + thickness / 2) - inside(r - thickness / 2))
                  for r in radii]
    return np.stack(frames)


def _rays(size, shape, t, n_angles=64):
    c = (size - 1) / 2
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    scale = (np.ones_like(angles) if shape == "circle"
             else 1 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles))))
    return c + t * np.sin(angles) * scale, c + t * np.cos(angles) * scale


def edge_radii(frames, shape, r_guess):
    """Radius per frame and angle where the intensity crosses (BG + FG) / 2."""
    t = np.linspace(0.3, 1.7, 281)[:, None] * r_guess  # samples along each ray
    ys, xs = _rays(frames.shape[1], shape, t)
    mid = (BG + FG) / 2
    out = np.full((len(frames), ys.shape[1]), np.nan)
    for i, frame in enumerate(frames):
        prof = ndimage.map_coordinates(frame, [ys, xs], order=3)  # (len(t), angles)
        for a in range(prof.shape[1]):
            p = prof[:, a]
            # last crossing from inside (above mid) to outside (below mid)
            idx = np.nonzero((p[:-1] >= mid) & (p[1:] < mid))[0]
            if len(idx):
                j = idx[-1]
                frac = (p[j] - mid) / (p[j] - p[j + 1])
                out[i, a] = t[j, 0] + frac * (t[j + 1, 0] - t[j, 0])
    return out


def fit_sine(signal, freq, fps):
    """Least-squares amplitude, phase (radians) and residual of a sine at freq."""
    tt = np.arange(len(signal)) / fps
    basis = np.stack([np.sin(2 * np.pi * freq * tt), np.cos(2 * np.pi * freq * tt),
                      np.ones(len(signal))], axis=1)
    coef, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return np.hypot(coef[0], coef[1]), np.arctan2(coef[1], coef[0]), signal - basis @ coef


def _clips(shape, r0, r1, k, freq, n, edge_sigma, thickness, noise, magnify):
    tt = np.arange(n) / FPS
    wave = np.sin(2 * np.pi * freq * tt)
    clip = render(shape, r0 + r1 * wave, edge_sigma=edge_sigma, thickness=thickness)
    if noise:
        clip = clip + np.random.RandomState(0).normal(0, noise, clip.shape)
    if magnify is None:
        magnify = evm_magnify(k - 1)  # EVM scales changes by 1 + alpha
    out = np.asarray(magnify(clip.astype(np.float32)), dtype=np.float64)
    ideal = render(shape, r0 + k * r1 * wave, edge_sigma=edge_sigma, thickness=thickness)
    return clip, out, ideal


def analyse(shape="circle", r0=30.0, r1=0.02, k=10.0, freq=1.5, n=180, noise=0.0,
            magnify=None, edge_sigma=0.8, trim=30):
    """Magnify a pulsating filled shape and measure it against the ideal.

    Args:
        magnify: function(frames float32, 0-255) -> magnified frames.
            Default: evm_magnify(k - 1), since EVM multiplies in-band
            changes by 1 + alpha.
        trim: frames dropped at each end (temporal filter edges).
    """
    clip, out, ideal = _clips(shape, r0, r1, k, freq, n, edge_sigma, None, noise, magnify)
    keep = slice(trim, n - trim)
    r_in = edge_radii(clip[keep], shape, r0)
    r_out = edge_radii(out[keep], shape, r0)
    a_in, ph_in, _ = fit_sine(np.nanmean(r_in, axis=1), freq, FPS)
    a_out, ph_out, _ = fit_sine(np.nanmean(r_out, axis=1), freq, FPS)
    per_angle = np.array([fit_sine(r_out[:, j], freq, FPS)[0] for j in range(r_out.shape[1])])
    lag = ((ph_in - ph_out + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * freq) * FPS
    return {
        "gain": a_out / r1,
        "gain_over_k": a_out / r1 / k,
        "input_gain": a_in / r1,
        "phase_lag_frames": lag,
        "angle_spread": np.std(per_angle) / np.mean(per_angle),
        "frames": (clip, out, ideal),
    }


def analyse_outline(shape="circle", thickness=1.0, r0=30.0, r1=0.05, k=10.0, freq=1.5,
                    n=180, noise=0.0, magnify=None, edge_sigma=0.8, trim=30):
    """Magnify a pulsating ring / hollow square and measure line fidelity.

    The line position is the centroid of its brightness above the background
    along each ray (robust for thin lines and ghosts).
    """
    clip, out, ideal = _clips(shape, r0, r1, k, freq, n, edge_sigma, thickness, noise,
                              magnify)
    keep = slice(trim, n - trim)
    span = k * abs(r1) + thickness + 6
    t = r0 + np.linspace(-span, span, int(span * 20) + 1)[:, None]
    ys, xs = _rays(clip.shape[1], shape, t)

    def profiles(frames):
        return np.stack([ndimage.map_coordinates(f, [ys, xs], order=3) for f in frames])

    def position(p):
        w = np.clip(p - BG, 0, None)
        return ((t[None] * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)).mean(axis=1)

    p_in, p_out, p_ideal = profiles(clip[keep]), profiles(out[keep]), profiles(ideal[keep])
    a_in = fit_sine(position(p_in), freq, FPS)[0]
    a_out = fit_sine(position(p_out), freq, FPS)[0]
    return {
        "gain_over_k": a_out / a_in / k if a_in > 0 else np.nan,
        "ghost_energy": float(np.sum((p_out - p_ideal) ** 2) / np.sum((p_ideal - BG) ** 2)),
        "frames": (clip, out, ideal),
    }


if __name__ == "__main__":
    print("Thin ring (1 px), k = 10 (alpha 9), band 0.5-3 Hz, lambda_c 10, 4 levels")
    for freq in (0.2, 0.5, 1.0, 1.5, 3.0, 6.0):
        r = analyse_outline(freq=freq, n=240, trim=50)
        print(f"  {freq:4.1f} Hz: gain/k {r['gain_over_k']:.3f}")
    print("Magnified displacement (1 px ring, 1.5 Hz)")
    for disp in (0.25, 0.5, 1, 2, 3, 5):
        r = analyse_outline(r1=disp / 10)
        print(f"  {disp:4} px: gain/k {r['gain_over_k']:.3f}  ghost {r['ghost_energy']:.2f}")
    print("Filled shapes (0.2 px magnified displacement)")
    for shape in ("circle", "square"):
        r = analyse(shape=shape)
        print(f"  {shape}: gain/k {r['gain_over_k']:.3f}  lag {r['phase_lag_frames']:+.2f} "
              f"frames  spread {r['angle_spread']:.3f}")
