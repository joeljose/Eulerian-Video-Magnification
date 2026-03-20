"""
Eulerian Video Magnification — GPU-accelerated CLI tool.

Same algorithm as evm.py but runs entirely on GPU via CuPy.
Requires an NVIDIA GPU with CUDA support.

Based on: Wu et al., "Eulerian Video Magnification for Revealing Subtle
Changes in the World", SIGGRAPH 2012.
"""

__version__ = "2.0.0-cuda"

import argparse
import math
import os
import sys
import time

import cv2
import cupy as cp
import cupyx.scipy.fftpack
import cupyx.scipy.ndimage
import numpy as np

# YIQ/NTSC color space conversion matrices (matches MATLAB rgb2ntsc/ntsc2rgb)
_RGB_TO_YIQ = cp.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
], dtype=cp.float32)

_YIQ_TO_RGB = cp.linalg.inv(_RGB_TO_YIQ).astype(cp.float32)


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def load_video(path):
    """Load a video file and return (GPU YIQ float32 array, fps).

    Loads via OpenCV on CPU, converts to YIQ, then transfers to GPU once.
    The returned array has shape (num_frames, height, width, 3) in YIQ
    color space with Y in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

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

    # Trim to actual frame count, convert BGR→RGB→float on CPU
    frames = frames[:i]
    rgb = frames[:, :, :, ::-1].astype(np.float32) / 255.0
    del frames

    # Transfer to GPU and convert to YIQ
    rgb_gpu = cp.asarray(rgb)
    del rgb
    yiq = rgb_gpu @ _RGB_TO_YIQ.T
    del rgb_gpu
    return yiq, fps


def save_video(video_yiq_gpu, fps, path):
    """Save a GPU YIQ float video array to an AVI file with MJPG codec.

    Transfers back to CPU, converts YIQ → RGB → BGR, clips, then writes.
    """
    # Transfer entire result back to CPU at once
    video_yiq = cp.asnumpy(video_yiq_gpu)
    yiq_to_rgb = cp.asnumpy(_YIQ_TO_RGB)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    h, w = video_yiq.shape[1], video_yiq.shape[2]
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h), True)
    try:
        for i in range(video_yiq.shape[0]):
            rgb = video_yiq[i] @ yiq_to_rgb.T
            bgr = np.clip(rgb[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
            writer.write(bgr)
    finally:
        writer.release()
    print(f"Output saved to {path}")


def gpu_pyr_down(frame):
    """Downsample a frame by 2x using Gaussian blur + subsampling on GPU.

    Args:
        frame: CuPy array of shape (H, W, 3).
    Returns:
        CuPy array of shape (H//2, W//2, 3).
    """
    # Apply Gaussian blur to each channel (sigma ~= OpenCV pyrDown default)
    blurred = cp.empty_like(frame)
    for c in range(3):
        blurred[:, :, c] = cupyx.scipy.ndimage.gaussian_filter(
            frame[:, :, c], sigma=1.0
        )
    return blurred[::2, ::2]


def gpu_pyr_up(frame, dst_shape):
    """Upsample a frame by 2x using bilinear-style upsampling on GPU.

    Args:
        frame: CuPy array of shape (H, W, 3).
        dst_shape: Target (height, width) tuple.
    Returns:
        CuPy array of shape (dst_height, dst_width, 3).
    """
    dst_h, dst_w = dst_shape
    src_h, src_w = frame.shape[0], frame.shape[1]
    zoom_h = dst_h / src_h
    zoom_w = dst_w / src_w

    result = cp.empty((dst_h, dst_w, 3), dtype=frame.dtype)
    for c in range(3):
        result[:, :, c] = cupyx.scipy.ndimage.zoom(
            frame[:, :, c], (zoom_h, zoom_w), order=1
        )
    # Smooth after upsampling to match pyrUp behavior
    for c in range(3):
        result[:, :, c] = cupyx.scipy.ndimage.gaussian_filter(
            result[:, :, c], sigma=1.0
        )
    return result


def create_laplacian_video_pyramid(video, pyramid_levels):
    """Decompose every frame into a Laplacian pyramid on GPU.

    Returns a list of CuPy arrays, one per pyramid level.
    """
    num_frames = video.shape[0]
    vid_pyramid = []

    t_start = time.time()
    for frame_idx in range(num_frames):
        # Build Gaussian pyramid for this frame
        gauss = [video[frame_idx]]
        for _ in range(1, pyramid_levels):
            gauss.append(gpu_pyr_down(gauss[-1]))

        # Build Laplacian pyramid from Gaussian
        lap = []
        for i in range(pyramid_levels - 1):
            up = gpu_pyr_up(gauss[i + 1], (gauss[i].shape[0], gauss[i].shape[1]))
            lap.append(gauss[i] - up)
        lap.append(gauss[-1])  # coarsest level

        # Allocate pyramid storage on first frame
        if frame_idx == 0:
            for level in lap:
                vid_pyramid.append(cp.zeros(
                    (num_frames, level.shape[0], level.shape[1], 3),
                    dtype=cp.float32
                ))

        for level_idx, level in enumerate(lap):
            vid_pyramid[level_idx][frame_idx] = level

        # Progress reporting every 10%
        if (frame_idx + 1) % max(1, num_frames // 10) == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - t_start
            pct = (frame_idx + 1) / num_frames
            eta = elapsed / pct * (1 - pct)
            print(
                f"  Pyramid: {frame_idx + 1}/{num_frames} frames "
                f"({pct:.0%}) — {format_duration(eta)} remaining"
            )

    return vid_pyramid


def ideal_bandpass_filter(data, fps, freq_low, freq_high):
    """Apply ideal bandpass filter along the time axis on GPU.

    Uses CuPy's FFT (backed by cuFFT) for GPU-accelerated filtering.
    """
    n = data.shape[0]
    freqs = cp.arange(n) / n * fps
    mask = ((freqs > freq_low) & (freqs < freq_high)).astype(cp.float64)
    mask = mask.reshape([n] + [1] * (data.ndim - 1))

    fft = cupyx.scipy.fftpack.fft(data, axis=0)
    fft *= mask

    return cp.real(cupyx.scipy.fftpack.ifft(fft, axis=0)).astype(cp.float32)


def collapse_laplacian_pyramid(image_pyramid):
    """Reconstruct an image from its Laplacian pyramid levels on GPU."""
    img = image_pyramid[-1]
    for i in range(len(image_pyramid) - 2, -1, -1):
        img = gpu_pyr_up(img, (image_pyramid[i].shape[0], image_pyramid[i].shape[1])) + image_pyramid[i]
    return img


def collapse_laplacian_video_pyramid(pyramid):
    """Reconstruct a full video from its Laplacian video pyramid on GPU."""
    num_frames = pyramid[0].shape[0]
    for i in range(num_frames):
        frame_pyramid = [level[i] for level in pyramid]
        pyramid[0][i] = collapse_laplacian_pyramid(frame_pyramid)
    return pyramid[0]


def eulerian_magnification(video, fps, freq_min, freq_max, alpha,
                           pyramid_levels=4, lambda_c=1000,
                           chrom_attenuation=1.0):
    """Run the full Eulerian Video Magnification pipeline on GPU."""
    total_start = time.time()
    height, width = video.shape[1], video.shape[2]
    n_levels = pyramid_levels

    print("Building Laplacian video pyramid (GPU)...")
    t0 = time.time()
    vid_pyramid = create_laplacian_video_pyramid(video, n_levels)
    del video
    cp.cuda.Stream.null.synchronize()
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Filtering and amplifying (GPU)...")
    t0 = time.time()

    # Adaptive per-level amplification (matches MATLAB reference, Figure 6)
    delta = lambda_c / 8.0 / (1.0 + alpha)
    exaggeration_factor = 2.0

    lambda_val = math.sqrt(height ** 2 + width ** 2) / 3.0

    level_alphas = [0.0] * n_levels
    lv = lambda_val
    for i in range(n_levels - 1, -1, -1):
        curr_alpha = (lv / delta / 8.0 - 1.0) * exaggeration_factor
        if i == n_levels - 1 or i == 0:
            level_alphas[i] = 0.0
        elif curr_alpha > alpha:
            level_alphas[i] = alpha
        else:
            level_alphas[i] = curr_alpha
        lv /= 2.0

    for i in range(n_levels):
        if level_alphas[i] == 0.0:
            continue

        filtered = ideal_bandpass_filter(
            vid_pyramid[i], fps, freq_min, freq_max
        )
        filtered[:, :, :, 0] *= level_alphas[i]
        filtered[:, :, :, 1] *= level_alphas[i] * chrom_attenuation
        filtered[:, :, :, 2] *= level_alphas[i] * chrom_attenuation

        vid_pyramid[i] += filtered
        del filtered

    cp.cuda.Stream.null.synchronize()
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Reconstructing video from pyramid (GPU)...")
    t0 = time.time()
    result = collapse_laplacian_video_pyramid(vid_pyramid)
    cp.cuda.Stream.null.synchronize()
    print(f"  Done in {format_duration(time.time() - t0)}")

    print(f"Total processing time: "
          f"{format_duration(time.time() - total_start)}")
    return result


def estimate_vram_bytes(num_frames, height, width, pyramid_levels):
    """Estimate peak GPU memory usage in bytes.

    Accounts for: video array, pyramid levels, FFT buffers.
    """
    bytes_per_pixel = 3 * 4  # 3 channels × float32
    video_bytes = num_frames * height * width * bytes_per_pixel

    # Pyramid levels: each level is ~1/4 the previous
    pyramid_bytes = 0
    h, w = height, width
    for _ in range(pyramid_levels):
        pyramid_bytes += num_frames * h * w * bytes_per_pixel
        h = h // 2
        w = w // 2

    # FFT buffer: same size as one pyramid level (largest = level 0)
    fft_buffer = num_frames * height * width * 3 * 16  # complex128

    return video_bytes + pyramid_bytes + fft_buffer


def check_vram(num_frames, height, width, pyramid_levels, device_id):
    """Check if GPU has enough VRAM. Exit with error if not."""
    required = estimate_vram_bytes(num_frames, height, width, pyramid_levels)
    free, total = cp.cuda.Device(device_id).mem_info

    required_gb = required / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    print(f"  Estimated VRAM needed: {required_gb:.1f} GB")
    print(f"  GPU VRAM available:    {free_gb:.1f} GB / {total_gb:.1f} GB")

    if required > free:
        print(
            f"\nError: insufficient GPU memory. Need {required_gb:.1f} GB "
            f"but only {free_gb:.1f} GB available.\n"
            f"Try a shorter clip or lower resolution.",
            file=sys.stderr
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification (GPU) — amplify subtle "
                    "temporal variations in video using CUDA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evm_cuda.py -i face.mp4\n"
            "  python evm_cuda.py -i face.mp4 -o magnified.avi -a 50 "
            "-fl 0.83 -fh 1.0\n"
            "  python evm_cuda.py -i guitar.mp4 -fl 72 -fh 92 -a 50 "
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
        help='Cutoff spatial wavelength for adaptive amplification '
             '(default: 1000). Lower values reduce amplification at '
             'finer spatial scales (see paper Figure 6).'
    )
    parser.add_argument(
        '--chrom-attenuation', type=float, default=1.0,
        help='Attenuation for color (I/Q) channels. '
             '1.0 = full color amplification, '
             '0.0 = luminance only (default: 1.0)'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='CUDA device ID (default: 0)'
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

    if args.lambda_c <= 0:
        print("Error: --lambda-c must be positive", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.chrom_attenuation <= 1.0:
        print("Error: --chrom-attenuation must be between 0.0 and 1.0",
              file=sys.stderr)
        sys.exit(1)

    # --- Set CUDA device ---
    try:
        cp.cuda.Device(args.device).use()
    except cp.cuda.runtime.CUDARuntimeError:
        print(f"Error: CUDA device {args.device} not available",
              file=sys.stderr)
        sys.exit(1)

    device_name = cp.cuda.runtime.getDeviceProperties(args.device)['name']
    print(f"Using GPU: {device_name} (device {args.device})")

    # --- Default output path ---
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_magnified.avi"

    # --- Load video (CPU) ---
    print(f"Loading {args.input}...")
    cap = cv2.VideoCapture(args.input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"  {frame_count} frames, {width}x{height}, {fps} fps")

    # --- VRAM check ---
    check_vram(frame_count, height, width, args.pyramid_levels, args.device)

    # --- Load and transfer to GPU ---
    video, fps = load_video(args.input)
    print("  Loaded and transferred to GPU")

    # --- Nyquist warning ---
    nyquist = fps / 2.0
    if args.freq_high > nyquist:
        print(f"Warning: --freq-high ({args.freq_high} Hz) exceeds Nyquist "
              f"frequency ({nyquist} Hz) for this video's frame rate ({fps} "
              f"fps). Results may be unreliable.", file=sys.stderr)

    # --- Run ---
    print("\nParameters:")
    print(f"  Frequency band:      {args.freq_low}–{args.freq_high} Hz")
    print(f"  Amplification:       {args.amplification}x")
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

    # --- Save (transfer back to CPU) ---
    print("Transferring result to CPU and saving...")
    save_video(result, fps, args.output)


if __name__ == '__main__':
    main()
