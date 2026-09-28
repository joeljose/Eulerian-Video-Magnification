"""Correctness checks on pulsating shapes with exact ground truth.

A circle, ring or square of radius r0 + r1 * sin(2 pi f t) magnified by k
should pulse k times as much. See scripts/synthetic_shapes.py and
docs/research/synthetic-validation.md.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import synthetic_shapes as ss  # noqa: E402

import evm  # noqa: E402

K = 10.0


def run(func, **kwargs):
    result = func(**kwargs)
    result.pop("frames", None)
    return result


def test_in_band_gain_is_pinned():
    """Pins current behaviour: a thin ring in the band gets ~0.38k, because
    the filter halves the signal and the finest and coarsest pyramid levels
    are not amplified (issue #28). Update when #28 changes what alpha means."""
    r = run(ss.analyse_outline, freq=1.5)
    assert 0.33 <= r["gain_over_k"] <= 0.43, r["gain_over_k"]


@pytest.mark.parametrize("freq", [0.2, 6.0])
def test_out_of_band_motion_is_unchanged(freq):
    r = run(ss.analyse_outline, freq=freq, n=240, trim=50)
    assert r["gain_over_k"] * K == pytest.approx(1.0, abs=0.05)


def test_no_phase_lag():
    r = run(ss.analyse, freq=1.5)
    assert abs(r["phase_lag_frames"]) < 0.05


@pytest.mark.parametrize("shape", ["circle", "square"])
def test_isotropic(shape):
    r = run(ss.analyse, shape=shape)
    assert r["angle_spread"] < 0.05, r["angle_spread"]


def test_tiny_alpha_is_identity():
    r = run(ss.analyse, r1=0.5, k=1.0, magnify=ss.evm_magnify(0.01))
    assert r["gain"] == pytest.approx(1.0, abs=0.01)


def test_default_lambda_c_never_shrinks_motion():
    """With lambda_c 1000 on a small frame the per-level gains used to go
    negative, so 'magnify 10x' shrank the motion to about 0.7x."""
    r = run(ss.analyse, magnify=ss.evm_magnify(K, lambda_c=1000))
    assert r["gain"] >= 0.99, r["gain"]


@pytest.mark.parametrize("size", [(128, 128), (240, 320), (480, 640), (1080, 1920)])
@pytest.mark.parametrize("alpha", [1, 10, 50])
def test_level_gains_are_never_negative(size, alpha):
    assert min(evm.compute_level_alphas(*size, 4, alpha, 1000)) >= 0
