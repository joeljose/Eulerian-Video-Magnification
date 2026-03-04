"""
Eulerian Video Magnification — CLI tool.

Amplifies subtle temporal variations (color changes, small motions) in video
using spatial decomposition, temporal bandpass filtering, and reconstruction.

Based on: Wu et al., "Eulerian Video Magnification for Revealing Subtle
Changes in the World", SIGGRAPH 2012.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import scipy.fftpack


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def load_video(path):
    """Load a video file and return (float32 numpy array, fps).

    The returned array has shape (num_frames, height, width, 3) with
    values in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames[i] = frame
            i += 1
    finally:
        cap.release()

    # Trim to actual frame count and convert to float
    frames = frames[:i]
    return frames.astype(np.float32) / 255.0, fps


def save_video(video, fps, path):
    """Save a float [0,1] video array to an AVI file with MJPG codec."""
    out = np.clip(video * 255, 0, 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(
        path, fourcc, fps, (out.shape[2], out.shape[1]), True
    )
    try:
        for i in range(out.shape[0]):
            writer.write(out[i])
    finally:
        writer.release()
    print(f"Output saved to {path}")


def create_laplacian_video_pyramid(video, pyramid_levels):
    """Decompose every frame into a Laplacian pyramid.

    Returns a list of arrays, one per pyramid level. Each array has shape
    (num_frames, level_height, level_width, 3).
    """
    num_frames = video.shape[0]
    vid_pyramid = []

    t_start = time.time()
    for frame_idx in range(num_frames):
        # Build Gaussian pyramid for this frame
        gauss = [video[frame_idx].copy()]
        for _ in range(1, pyramid_levels):
            gauss.append(cv2.pyrDown(gauss[-1]))

        # Build Laplacian pyramid from Gaussian
        lap = []
        for i in range(pyramid_levels - 1):
            lap.append(gauss[i] - cv2.pyrUp(gauss[i + 1]))
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

        # Progress reporting every 10% or every 50 frames
        if (frame_idx + 1) % max(1, num_frames // 10) == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx + 1) / num_frames
            eta = elapsed / pct * (1 - pct)
            print(
                f"  Pyramid: {frame_idx + 1}/{num_frames} frames "
                f"({pct:.0%}) — {format_duration(eta)} remaining"
            )

    return vid_pyramid


def temporal_bandpass_filter(data, fps, freq_min, freq_max,
                             amplification_factor):
    """Apply FFT bandpass filter along the time axis and amplify."""
    fft = scipy.fftpack.fft(data, axis=0)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)

    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()

    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.abs(scipy.fftpack.ifft(fft, axis=0)).astype(np.float32)
    result *= amplification_factor
    return result


def collapse_laplacian_pyramid(image_pyramid):
    """Reconstruct an image from its Laplacian pyramid levels."""
    img = image_pyramid[-1]
    for i in range(len(image_pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img) + image_pyramid[i]
    return img


def collapse_laplacian_video_pyramid(pyramid):
    """Reconstruct a full video from its Laplacian video pyramid."""
    num_frames = pyramid[0].shape[0]
    for i in range(num_frames):
        frame_pyramid = [level[i] for level in pyramid]
        pyramid[0][i] = collapse_laplacian_pyramid(frame_pyramid)
    return pyramid[0]


def eulerian_magnification(video, fps, freq_min, freq_max, amplification,
                           pyramid_levels=4, skip_levels=1):
    """Run the full Eulerian Video Magnification pipeline.

    1. Build Laplacian video pyramid (spatial decomposition)
    2. Bandpass filter + amplify each pyramid level temporally
    3. Collapse pyramid to reconstruct magnified video
    """
    total_start = time.time()

    print("Building Laplacian video pyramid...")
    t0 = time.time()
    vid_pyramid = create_laplacian_video_pyramid(video, pyramid_levels)
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Applying temporal bandpass filter...")
    t0 = time.time()
    for i in range(len(vid_pyramid)):
        if i < skip_levels or i >= len(vid_pyramid) - 1:
            continue
        bandpassed = temporal_bandpass_filter(
            vid_pyramid[i], fps,
            freq_min=freq_min, freq_max=freq_max,
            amplification_factor=amplification
        )
        vid_pyramid[i] += bandpassed
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Reconstructing video from pyramid...")
    t0 = time.time()
    result = collapse_laplacian_video_pyramid(vid_pyramid)
    print(f"  Done in {format_duration(time.time() - t0)}")

    print(f"Total processing time: {format_duration(time.time() - total_start)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification — amplify subtle temporal "
                    "variations in video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evm.py -i face.mp4\n"
            "  python evm.py -i face.mp4 -o magnified.avi -fl 0.5 -fh 2.0 "
            "-a 15\n"
            "  python evm.py -i baby.mp4 -fl 0.1 -fh 0.5 -a 30 "
            "--pyramid-levels 6"
        )
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
        '-a', '--amplification', type=float, default=15,
        help='Amplification factor (default: 15)'
    )
    parser.add_argument(
        '--pyramid-levels', type=int, default=4,
        help='Number of Laplacian pyramid levels (default: 4)'
    )
    parser.add_argument(
        '--skip-levels', type=int, default=1,
        help='Number of finest pyramid levels to skip (default: 1)'
    )

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
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

    if args.skip_levels < 0:
        print("Error: --skip-levels must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.skip_levels >= args.pyramid_levels - 1:
        print("Error: --skip-levels must be less than pyramid-levels - 1",
              file=sys.stderr)
        sys.exit(1)

    # --- Default output path ---
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_magnified.avi"

    # --- Load video ---
    print(f"Loading {args.input}...")
    video, fps = load_video(args.input)
    print(f"  {video.shape[0]} frames, {video.shape[2]}x{video.shape[1]}, "
          f"{fps} fps")

    # --- Nyquist warning ---
    nyquist = fps / 2.0
    if args.freq_high > nyquist:
        print(f"Warning: --freq-high ({args.freq_high} Hz) exceeds Nyquist "
              f"frequency ({nyquist} Hz) for this video's frame rate ({fps} "
              f"fps). Results may be unreliable.", file=sys.stderr)

    # --- Run ---
    print(f"\nParameters:")
    print(f"  Frequency band: {args.freq_low}–{args.freq_high} Hz")
    print(f"  Amplification:  {args.amplification}x")
    print(f"  Pyramid levels: {args.pyramid_levels}")
    print(f"  Skip levels:    {args.skip_levels}\n")

    result = eulerian_magnification(
        video, fps,
        freq_min=args.freq_low,
        freq_max=args.freq_high,
        amplification=args.amplification,
        pyramid_levels=args.pyramid_levels,
        skip_levels=args.skip_levels
    )

    # --- Save ---
    save_video(result, fps, args.output)


if __name__ == '__main__':
    main()
