# Synthetic validation

Checks the magnification against exact ground truth, using pulsating shapes. The harness is adapted from the one in Motion-Magnification-Using-2D-DTCWT.

- Script: [`scripts/synthetic_shapes.py`](../../scripts/synthetic_shapes.py) (`python scripts/synthetic_shapes.py` prints the tables below).
- Permanent checks: [`tests/test_synthetic_shapes.py`](../../tests/test_synthetic_shapes.py).
- Setup unless noted: asking for k = 10× (`-a 9`, since EVM multiplies in-band changes by `1 + α`), band 0.5–3 Hz, `--lambda-c 10`, 4 pyramid levels, 30 fps, 128×128 frames. Shapes are drawn as luminance (Y) with no chroma. Edges are blurred with a Gaussian of σ = 0.8 px, like a camera's point spread, so sub-pixel sizes are exact.

## Method

A shape whose radius follows `r(t) = r0 + r1·sin(2πft)` should, after magnification by k, be the same shape with radius `r0 + k·r1·sin(2πft)`. In every output frame the edge is located along 64 rays from the centre: for filled shapes, where the intensity crosses the midpoint between background and foreground; for rings, the brightness centroid across the line. A sine fitted to the radius over time gives the output amplitude, so **gain / k** = output amplitude / (k · r1), with 1.0 as the ideal. The same fit per ray gives the spread of the gain around the shape. **Ghost energy** is the energy of (output − ideal) across a ring relative to the ideal ring's energy. It includes the shortfall in gain, not just artefacts. The first and last 30–50 frames are excluded, where the temporal filter runs out of data.

## Results

**Frequency response** (1 px ring, r0 = 30 px, magnified displacement 0.5 px):

| f (Hz) | 0.2 | 0.5 | 1.0 | 1.5 | 3.0 | 6.0 |
|---|---:|---:|---:|---:|---:|---:|
| gain / k | 0.104 | 0.355 | 0.572 | 0.573 | 0.347 | 0.100 |

Outside the band the motion is unchanged (gain 1.0, i.e. 0.1k at k = 10). At the band edges, which fall exactly on the pulse frequency here, gain is about half. Inside the band it reaches about **0.57k**. The temporal filter passes in-band signal at full amplitude (since v3.0.0; before, like the MATLAB reference, it halved it and this was 0.38k), so every amplified level changes by exactly `1 + α`. The shortfall is spatial: the finest level (level 0) and the low-pass residual are never amplified, and part of the edge's motion lives there.

**Displacement** (1 px ring, 1.5 Hz):

| magnified displacement | 0.25 px | 0.5 px | 1 px | 2 px | 3 px | 5 px |
|---|---:|---:|---:|---:|---:|---:|
| gain / k | 0.633 | 0.574 | 0.490 | 0.366 | 0.275 | 0.158 |
| ghost energy | 0.01 | 0.04 | 0.15 | 0.54 | 1.05 | 2.38 |

The first-order (Taylor) approximation behind Eulerian magnification only holds for small motion. Quality falls from about 1 px of magnified motion, and ghost edges dominate from 2 px. For comparison, the phase-based DTCWT method holds up to about 3 px.

**Filled shapes** (0.2 px magnified displacement): a circle reaches 0.436k and a square 0.413k. The phase lag is 0.00 frames, and the gain varies by about 3% around the shape, so all edge directions are treated alike.

**Negative gains** (fixed in v3.0.0): with `--lambda-c 1000` (the default before v3.0.0), the reference formula gives negative per-level gains on small frames. For example, 320×240 at `-a 10` gives `[0, -1.27, -0.53, 0]`, which *shrinks* the motion: a filled circle came out at 0.71× instead of being magnified. Gains are now clamped at 0, and the same case gives 1.0×. With `--lambda-c 1000`, `-a 10` still does nothing on such frames; the v3.0.0 default `--lambda-c 16` gives those levels the full α.

**Identity:** with a tiny α (0.01) the output motion equals the input (gain 1.002).

## Reproducing

```bash
python scripts/synthetic_shapes.py
python -m pytest tests/test_synthetic_shapes.py -q
```

`analyse()` (filled shapes) and `analyse_outline()` (rings, hollow squares) take the shape, radius, amplitude, frequency, k, noise and an optional `magnify` function. They return the metrics above plus the input, output and ideal frames. `evm_magnify()` builds a `magnify` function for any EVM settings.
