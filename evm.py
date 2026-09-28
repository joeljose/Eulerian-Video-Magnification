"""
Eulerian Video Magnification — CLI tool.

Amplifies subtle temporal variations (color changes, small motions) in video
using spatial decomposition, temporal bandpass filtering, and reconstruction.

Based on: Wu et al., "Eulerian Video Magnification for Revealing Subtle
Changes in the World", SIGGRAPH 2012.

Algorithm follows the reference MATLAB implementation from MIT CSAIL.
"""

__version__ = "2.1.0"

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import scipy.fft

# YIQ/NTSC color space conversion matrices (matches MATLAB rgb2ntsc/ntsc2rgb)
_RGB_TO_YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
], dtype=np.float32)

_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ).astype(np.float32)


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def rgb_to_yiq(frame):
    """Convert an RGB float frame to YIQ color space."""
    return frame @ _RGB_TO_YIQ.T


def yiq_to_rgb(frame):
    """Convert a YIQ float frame to RGB color space."""
    return frame @ _YIQ_TO_RGB.T


def read_frames(path, fps=None):
    """Decode every frame of a video. Returns (uint8 BGR array, fps).

    Reads until the decoder stops, not until CAP_PROP_FRAME_COUNT, because
    that value is only a container estimate. `fps` overrides the
    container's frame rate. Raises ValueError for unreadable input, a
    missing frame rate, or fewer than 2 frames.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"could not determine the frame rate of {path}; "
                             f"pass --fps")

        # Preallocate from the reported count; frames past it go to a list
        frames = np.empty((max(reported, 0), height, width, 3), dtype=np.uint8)
        extra = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i < len(frames):
                frames[i] = frame
            else:
                extra.append(frame)
            i += 1
    finally:
        cap.release()

    frames = frames[:i]
    if extra:
        frames = np.concatenate([frames, np.stack(extra)])
    if i != reported:
        print(f"Warning: decoded {i} frames, container reported {reported}",
              file=sys.stderr)
    if i < 2:
        raise ValueError(f"only {i} decodable frame(s) in {path}")
    return frames, fps


def load_video(path, fps=None):
    """Load a video file and return (YIQ float32 numpy array, fps).

    The returned array has shape (num_frames, height, width, 3) in YIQ
    color space with Y in [0, 1].
    """
    frames, fps = read_frames(path, fps)
    # Convert BGR→RGB→float→YIQ
    rgb = frames[:, :, :, ::-1].astype(np.float32) / 255.0
    del frames  # free uint8 buffer
    yiq = rgb @ _RGB_TO_YIQ.T  # vectorized over all frames at once
    del rgb
    return yiq, fps


def save_video(video_yiq, fps, path):
    """Save a YIQ float video array to an AVI file with MJPG codec.

    Converts YIQ → RGB → BGR, clips to [0, 1], then writes.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    h, w = video_yiq.shape[1], video_yiq.shape[2]
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path} (MJPG)")
    try:
        for i in range(video_yiq.shape[0]):
            rgb = yiq_to_rgb(video_yiq[i])
            bgr = np.clip(np.rint(rgb[:, :, ::-1] * 255), 0, 255).astype(np.uint8)
            writer.write(bgr)
    finally:
        writer.release()
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"video writer produced no output at {path}")
    print(f"Output saved to {path}")


def create_laplacian_video_pyramid(video, pyramid_levels):
    """Decompose every frame into a Laplacian pyramid.

    Returns a list of arrays, one per pyramid level. Each array has shape
    (num_frames, level_height, level_width, 3). Level 0 is the finest
    (full resolution), level N-1 is the coarsest.
    """
    num_frames = video.shape[0]
    vid_pyramid = []

    t_start = time.time()
    for frame_idx in range(num_frames):
        # Build Gaussian pyramid for this frame
        gauss = [video[frame_idx]]
        for _ in range(1, pyramid_levels):
            gauss.append(cv2.pyrDown(gauss[-1]))

        # Build Laplacian pyramid from Gaussian
        lap = []
        for i in range(pyramid_levels - 1):
            lap.append(gauss[i] - cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0])))
        lap.append(gauss[-1])  # coarsest level

        # Allocate pyramid storage on first frame
        if frame_idx == 0:
            for level in lap:
                vid_pyramid.append(np.zeros(
                    (num_frames, level.shape[0], level.shape[1], 3),
                    dtype=np.float32
                ))

        for level_idx, level in enumerate(lap):
            vid_pyramid[level_idx][frame_idx] = level

        # Progress reporting every 10%
        if (frame_idx + 1) % max(1, num_frames // 10) == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx + 1) / num_frames
            eta = elapsed / pct * (1 - pct)
            print(
                f"  Pyramid: {frame_idx + 1}/{num_frames} frames "
                f"({pct:.0%}) — {format_duration(eta)} remaining"
            )

    return vid_pyramid


def passband_freqs(n, fps, freq_low, freq_high):
    """Return the FFT bin frequencies that the ideal filter keeps.

    The filter transforms the clip extended to 2n frames (see
    ideal_bandpass_filter), so bins are fps / (2n) apart. The real frequency
    resolution of an n-frame clip is still fps / n.
    """
    freqs = np.fft.rfftfreq(2 * n, 1.0 / fps)
    return freqs[(freqs > freq_low) & (freqs < freq_high)]


def ideal_bandpass_filter(data, fps, freq_low, freq_high):
    """Apply ideal bandpass filter along the time axis.

    Follows the reference MATLAB implementation (ideal_bandpassing.m),
    including its half-amplitude output (it keeps positive frequencies
    only). No amplification is applied; that is done per level.

    The clip is extended with its time-reversed copy before the FFT, so the
    transform sees a seamless loop instead of joining the last frame to the
    first; slow drift over the clip no longer turns into amplified flicker.

    Raises ValueError if no frequency bin falls inside the band, since the
    output would then be all zeros.
    """
    n = data.shape[0]
    freqs = np.fft.rfftfreq(2 * n, 1.0 / fps)
    keep = (freqs > freq_low) & (freqs < freq_high)
    if not keep.any():
        raise ValueError(
            f"band {freq_low}-{freq_high} Hz contains no frequency bins "
            f"(resolution {fps / n:.3f} Hz for {n} frames at {fps} fps)"
        )
    mask = keep.reshape([len(keep)] + [1] * (data.ndim - 1))

    # Clip followed by its time reverse: a seamless loop for the FFT
    extended = np.concatenate([data, data[::-1]], axis=0)
    spectrum = scipy.fft.rfft(extended, axis=0)
    del extended
    spectrum *= mask
    filtered = scipy.fft.irfft(spectrum, 2 * n, axis=0)[:n]

    # x0.5: the reference keeps positive frequencies only, which halves the
    # in-band signal. Kept for MATLAB parity; see issue #28.
    return (0.5 * filtered).astype(np.float32)


def collapse_laplacian_pyramid(image_pyramid):
    """Reconstruct an image from its Laplacian pyramid levels."""
    img = image_pyramid[-1]
    for i in range(len(image_pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(image_pyramid[i].shape[1], image_pyramid[i].shape[0])) + image_pyramid[i]
    return img


def collapse_laplacian_video_pyramid(pyramid):
    """Reconstruct a full video from its Laplacian video pyramid."""
    num_frames = pyramid[0].shape[0]
    for i in range(num_frames):
        frame_pyramid = [level[i] for level in pyramid]
        pyramid[0][i] = collapse_laplacian_pyramid(frame_pyramid)
    return pyramid[0]


def compute_level_alphas(height, width, pyramid_levels, alpha, lambda_c):
    """Per-level amplification, finest level first (MATLAB reference, Fig. 6).

    Levels whose representative wavelength is short relative to lambda_c
    get less than alpha, so a larger lambda_c means weaker amplification.
    """
    delta = lambda_c / 8.0 / (1.0 + alpha)
    exaggeration_factor = 2.0

    # Representative wavelength for the coarsest level
    lv = math.sqrt(height ** 2 + width ** 2) / 3.0

    # Compute per-level alpha from coarsest to finest
    level_alphas = [0.0] * pyramid_levels
    for i in range(pyramid_levels - 1, -1, -1):
        curr_alpha = (lv / delta / 8.0 - 1.0) * exaggeration_factor
        if i == pyramid_levels - 1 or i == 0:
            # Level 0 (finest): spatial wavelengths too short, amplification
            # would break the Taylor approximation -> artifacts.
            # Coarsest level: low-pass residual (DC/mean), not a bandpass
            # level, amplifying it shifts global brightness.
            level_alphas[i] = 0.0
        else:
            # Clamp at 0: a negative gain would shrink or invert motion
            level_alphas[i] = max(0.0, min(curr_alpha, alpha))
        lv /= 2.0
    return level_alphas


def eulerian_magnification(video, fps, freq_min, freq_max, alpha,
                           pyramid_levels=4, lambda_c=1000,
                           chrom_attenuation=1.0):
    """Run the full Eulerian Video Magnification pipeline.

    Follows the reference MATLAB implementation
    (amplify_spatial_lpyr_temporal_ideal.m):

    1. Build Laplacian video pyramid (spatial decomposition)
    2. Ideal bandpass filter each pyramid level temporally
    3. Amplify with adaptive per-level alpha based on lambda_c
    4. Apply chromatic attenuation to I/Q channels
    5. Add filtered signal back to pyramid and reconstruct

    Args:
        video: Input video in YIQ color space (num_frames, H, W, 3).
        fps: Frame rate.
        freq_min: Lower cutoff frequency (Hz).
        freq_max: Upper cutoff frequency (Hz).
        alpha: Amplification factor.
        pyramid_levels: Number of Laplacian pyramid levels.
        lambda_c: Cutoff spatial wavelength in pixels (paper Figure 6).
            Structures smaller than lambda_c get reduced amplification, so
            lower values give stronger amplification.
        chrom_attenuation: Attenuation factor for I/Q (color) channels.
            1.0 = full color amplification, 0.0 = luminance only.
    """
    total_start = time.time()
    height, width = video.shape[1], video.shape[2]
    n_levels = pyramid_levels

    print("Building Laplacian video pyramid...")
    t0 = time.time()
    vid_pyramid = create_laplacian_video_pyramid(video, n_levels)
    del video  # free original; data is now in pyramid levels
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Filtering and amplifying...")
    t0 = time.time()

    level_alphas = compute_level_alphas(height, width, n_levels, alpha,
                                        lambda_c)

    # Filter, amplify, and add back — one level at a time to limit memory
    for i in range(n_levels):
        if level_alphas[i] == 0.0:
            continue  # skip levels that would be zeroed out (saves FFT)

        filtered = ideal_bandpass_filter(
            vid_pyramid[i], fps, freq_min, freq_max
        )
        # Amplify: full alpha on Y, attenuated on I/Q
        filtered[:, :, :, 0] *= level_alphas[i]
        filtered[:, :, :, 1] *= level_alphas[i] * chrom_attenuation
        filtered[:, :, :, 2] *= level_alphas[i] * chrom_attenuation

        vid_pyramid[i] += filtered
        del filtered  # free memory immediately

    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Reconstructing video from pyramid...")
    t0 = time.time()
    result = collapse_laplacian_video_pyramid(vid_pyramid)
    print(f"  Done in {format_duration(time.time() - t0)}")

    print(f"Total processing time: "
          f"{format_duration(time.time() - total_start)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification — amplify subtle temporal "
                    "variations in video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evm.py -i face.mp4\n"
            "  python evm.py -i face.mp4 -o magnified.avi -a 50 "
            "-fl 0.83 -fh 1.0\n"
            "  python evm.py -i guitar.mp4 -fl 72 -fh 92 -a 50 "
            "--lambda-c 10 --chrom-attenuation 0"
        )
    )
    parser.add_argument(
        '--version', action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input video path'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output video path (default: <input>_magnified.avi)'
    )
    parser.add_argument(
        '-fl', '--freq-low', type=float, default=0.5,
        help='Lower cutoff frequency in Hz (default: 0.5)'
    )
    parser.add_argument(
        '-fh', '--freq-high', type=float, default=2.0,
        help='Upper cutoff frequency in Hz (default: 2.0)'
    )
    parser.add_argument(
        '-a', '--amplification', type=float, default=50,
        help='Amplification factor / alpha (default: 50)'
    )
    parser.add_argument(
        '--pyramid-levels', type=int, default=4,
        help='Number of Laplacian pyramid levels (default: 4)'
    )
    parser.add_argument(
        '--lambda-c', type=float, default=1000,
        help='Cutoff spatial wavelength in pixels (default: 1000). '
             'Structures smaller than this get reduced amplification, '
             'so lower values give stronger amplification '
             '(see paper Figure 6).'
    )
    parser.add_argument(
        '--fps', type=float, default=None,
        help='Frame rate of the input (default: read from the video)'
    )
    parser.add_argument(
        '--chrom-attenuation', type=float, default=1.0,
        help='Attenuation for color (I/Q) channels. '
             '1.0 = full color amplification, '
             '0.0 = luminance only (default: 1.0)'
    )

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    for name in ('freq_low', 'freq_high', 'amplification', 'lambda_c',
                 'chrom_attenuation', 'fps'):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            print(f"Error: --{name.replace('_', '-')} must be a finite number",
                  file=sys.stderr)
            sys.exit(1)
    if args.fps is not None and args.fps <= 0:
        print("Error: --fps must be positive", file=sys.stderr)
        sys.exit(1)

    if args.freq_low <= 0:
        print("Error: --freq-low must be positive", file=sys.stderr)
        sys.exit(1)

    if args.freq_high <= args.freq_low:
        print("Error: --freq-high must be greater than --freq-low",
              file=sys.stderr)
        sys.exit(1)

    if args.amplification <= 0:
        print("Error: --amplification must be positive", file=sys.stderr)
        sys.exit(1)

    if args.pyramid_levels < 2:
        print("Error: --pyramid-levels must be at least 2", file=sys.stderr)
        sys.exit(1)

    if args.lambda_c <= 0:
        print("Error: --lambda-c must be positive", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.chrom_attenuation <= 1.0:
        print("Error: --chrom-attenuation must be between 0.0 and 1.0",
              file=sys.stderr)
        sys.exit(1)

    # --- Default output path ---
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_magnified.avi"

    # --- Check the output location before doing any work ---
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        print(f"Error: output directory does not exist: {out_dir}",
              file=sys.stderr)
        sys.exit(1)
    if not os.access(out_dir, os.W_OK):
        print(f"Error: output directory is not writable: {out_dir}",
              file=sys.stderr)
        sys.exit(1)
    if not args.output.lower().endswith('.avi'):
        print("Warning: output is always MJPG; use a .avi extension",
              file=sys.stderr)

    # --- Load video ---
    print(f"Loading {args.input}...")
    try:
        video, fps = load_video(args.input, fps=args.fps)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    n_frames, height, width = video.shape[:3]
    print(f"  {n_frames} frames, {width}x{height}, {fps} fps")

    # --- Nyquist check ---
    nyquist = fps / 2.0
    if args.freq_high > nyquist:
        print(f"Error: --freq-high ({args.freq_high} Hz) exceeds the Nyquist "
              f"frequency ({nyquist} Hz) at {fps} fps", file=sys.stderr)
        sys.exit(1)

    # --- Effective band ---
    bins = passband_freqs(n_frames, fps, args.freq_low, args.freq_high)
    resolution = fps / n_frames
    if len(bins) == 0:
        print(f"Error: band {args.freq_low}–{args.freq_high} Hz contains no "
              f"frequency bins (resolution {resolution:.3f} Hz for "
              f"{n_frames} frames at {fps} fps). Widen the band or use a "
              f"longer clip.", file=sys.stderr)
        sys.exit(1)
    if args.freq_high - args.freq_low < resolution:
        print(f"Warning: the band is narrower than the clip's frequency "
              f"resolution ({resolution:.3f} Hz); consider a wider band or a "
              f"longer clip.", file=sys.stderr)

    level_alphas = compute_level_alphas(height, width, args.pyramid_levels,
                                        args.amplification, args.lambda_c)

    # --- Run ---
    print("\nParameters:")
    print(f"  Frequency band:      {args.freq_low}–{args.freq_high} Hz "
          f"({len(bins)} bins, {bins[0]:.3f}–{bins[-1]:.3f} Hz, "
          f"Δf={resolution:.3f} Hz)")
    print(f"  Amplification:       {args.amplification}x")
    print(f"  Level gains:         "
          f"[{', '.join(f'{a:.2f}' for a in level_alphas)}] "
          f"(x0.5 from one-sided filter)")
    print(f"  Pyramid levels:      {args.pyramid_levels}")
    print(f"  Lambda_c:            {args.lambda_c}")
    print(f"  Chrom attenuation:   {args.chrom_attenuation}\n")

    result = eulerian_magnification(
        video, fps,
        freq_min=args.freq_low,
        freq_max=args.freq_high,
        alpha=args.amplification,
        pyramid_levels=args.pyramid_levels,
        lambda_c=args.lambda_c,
        chrom_attenuation=args.chrom_attenuation,
    )

    # --- Save ---
    try:
        save_video(result, fps, args.output)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
