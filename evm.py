"""
Eulerian Video Magnification — CLI tool.

Amplifies subtle temporal variations (color changes, small motions) in video
using spatial decomposition, temporal bandpass filtering, and reconstruction.

Runs on the CPU with NumPy/SciPy, or on an NVIDIA GPU with CuPy (--gpu).
Every step is written once: the array functions pick NumPy or CuPy from the
type of the array they are given (see _backend).

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
import scipy.ndimage

# YIQ/NTSC color space conversion matrices (matches MATLAB rgb2ntsc/ntsc2rgb)
_RGB_TO_YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
], dtype=np.float32)

_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ).astype(np.float32)


def _backend(array):
    """(xp, ndimage, fft) modules for an array: CuPy's for a CuPy array,
    else NumPy/SciPy's. CuPy is only imported when a CuPy array appears."""
    if type(array).__module__.split('.')[0] == 'cupy':
        import cupy
        import cupyx.scipy.fft
        import cupyx.scipy.ndimage
        return cupy, cupyx.scipy.ndimage, cupyx.scipy.fft
    return np, scipy.ndimage, scipy.fft


def _to_numpy(array):
    """Copy a CuPy array to the host; NumPy arrays pass through."""
    return array.get() if hasattr(array, 'get') else array


def _sync(xp):
    """Wait for queued GPU work, so timings are real."""
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def rgb_to_yiq(frame):
    """Convert an RGB float frame (or video) to YIQ color space."""
    xp = _backend(frame)[0]
    return frame @ xp.asarray(_RGB_TO_YIQ.T)


def yiq_to_rgb(frame):
    """Convert a YIQ float frame (or video) to RGB color space."""
    xp = _backend(frame)[0]
    return frame @ xp.asarray(_YIQ_TO_RGB.T)


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


def frames_to_yiq(frames, xp=np):
    """uint8 BGR frames to a float32 YIQ array on `xp` (numpy or cupy).

    The uint8 frames are transferred first, so a GPU copy moves a quarter
    of the bytes of the float video.
    """
    rgb = xp.asarray(frames)[:, :, :, ::-1].astype(xp.float32) / 255.0
    yiq = rgb_to_yiq(rgb)
    del rgb
    return yiq


def load_video(path, fps=None, xp=np):
    """Load a video file and return (YIQ float32 array, fps).

    The returned array is on `xp` (numpy, or cupy for the GPU), with shape
    (num_frames, height, width, 3) in YIQ color space with Y in [0, 1].
    """
    frames, fps = read_frames(path, fps)
    return frames_to_yiq(frames, xp), fps


def open_writer(path, fps, frame_size):
    """An MJPG VideoWriter for `path`, frame_size = (width, height).

    Raises RuntimeError if it can't be opened (missing codec, bad path).
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size, True)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path} (MJPG)")
    return writer


def yiq_to_bgr8(frames_yiq):
    """YIQ float frames (numpy or cupy) to rounded BGR uint8, on the host."""
    xp = _backend(frames_yiq)[0]
    rgb = yiq_to_rgb(frames_yiq)
    return _to_numpy(xp.clip(xp.rint(rgb[..., ::-1] * 255), 0, 255).astype(xp.uint8))


def write_yiq(writer, frames_yiq):
    """Write YIQ float frames (numpy or cupy) as rounded BGR uint8."""
    for frame in yiq_to_bgr8(frames_yiq):
        writer.write(np.ascontiguousarray(frame))


def close_writer(writer, path):
    """Release the writer; raise RuntimeError if it wrote nothing."""
    writer.release()
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"video writer produced no output at {path}")
    print(f"Output saved to {path}")


def save_video(video_yiq, fps, path):
    """Save a YIQ float video array (numpy or cupy) to an AVI file with
    MJPG codec."""
    writer = open_writer(path, fps, (video_yiq.shape[2], video_yiq.shape[1]))
    try:
        write_yiq(writer, video_yiq)
    except BaseException:
        writer.release()
        raise
    close_writer(writer, path)


# OpenCV's pyramid kernel: binomial [1 4 6 4 1] / 16
_PYR_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16

# Frames per pyramid batch: large enough to vectorise, small enough that the
# temporaries stay a fraction of the video
_PYR_BLOCK = 8


def _pyr_filter(x, kernel):
    """Separable filter over the two spatial axes of (..., H, W, C).

    'mirror' is OpenCV's BORDER_REFLECT_101.
    """
    xp, ndimage, _ = _backend(x)
    k = xp.asarray(kernel)
    x = ndimage.correlate1d(x, k, axis=-3, mode='mirror')
    return ndimage.correlate1d(x, k, axis=-2, mode='mirror')


def _ndimage_pyr_down(x):
    """cv2.pyrDown with ndimage: blur, then keep every second pixel."""
    return _pyr_filter(x, _PYR_KERNEL)[..., ::2, ::2, :]


def _ndimage_pyr_up(x, dst_hw):
    """cv2.pyrUp with ndimage, to size dst_hw = (H, W).

    Inserts zeros to 2h x 2w, filters with 4x the kernel (2x per axis),
    then crops to (H, W). Matches OpenCV to float precision, odd sizes too.
    """
    xp = _backend(x)[0]
    h, w = x.shape[-3:-1]
    up = xp.zeros(x.shape[:-3] + (2 * h, 2 * w, x.shape[-1]), dtype=x.dtype)
    up[..., ::2, ::2, :] = x
    return _pyr_filter(up, 2 * _PYR_KERNEL)[..., :dst_hw[0], :dst_hw[1], :]


def _per_frame(x, func):
    """Apply an OpenCV (H, W, C) function to each frame of (..., H, W, C)."""
    if x.ndim == 3:
        return func(x)
    out = np.stack([func(f) for f in x.reshape((-1,) + x.shape[-3:])])
    return out.reshape(x.shape[:-3] + out.shape[1:])


def pyr_down(x):
    """cv2.pyrDown on the spatial axes of (..., H, W, C).

    NumPy input uses OpenCV itself (about 10x faster than scipy.ndimage);
    CuPy input uses the ndimage version, which tests keep equal to OpenCV.
    """
    if _backend(x)[0] is np:
        return _per_frame(x, cv2.pyrDown)
    return _ndimage_pyr_down(x)


def pyr_up(x, dst_hw):
    """cv2.pyrUp on (..., h, w, C) to size dst_hw = (H, W); see pyr_down."""
    if _backend(x)[0] is np:
        return _per_frame(x, lambda f: cv2.pyrUp(f, dstsize=(dst_hw[1], dst_hw[0])))
    return _ndimage_pyr_up(x, dst_hw)


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

    In-band frequencies pass at full amplitude and everything else is
    removed. (The reference MATLAB ideal_bandpassing.m keeps positive
    frequencies only, which halves the in-band signal; this doesn't, so
    alpha is the true gain.) No amplification is applied here.

    The clip is extended with its time-reversed copy before the FFT, so the
    transform sees a seamless loop instead of joining the last frame to the
    first; slow drift over the clip no longer turns into amplified flicker.

    Raises ValueError if no frequency bin falls inside the band, since the
    output would then be all zeros.
    """
    xp, _, fft = _backend(data)
    n = data.shape[0]
    freqs = np.fft.rfftfreq(2 * n, 1.0 / fps)
    keep = (freqs > freq_low) & (freqs < freq_high)
    if not keep.any():
        raise ValueError(
            f"band {freq_low}-{freq_high} Hz contains no frequency bins "
            f"(resolution {fps / n:.3f} Hz for {n} frames at {fps} fps)"
        )
    mask = xp.asarray(keep.reshape([len(keep)] + [1] * (data.ndim - 1)))

    # SciPy's FFT runs on all CPU cores with workers=-1 (about 3x faster
    # here); CuPy's is already parallel on the GPU
    workers = {'workers': -1} if xp is np else {}

    # Clip followed by its time reverse: a seamless loop for the FFT
    extended = xp.concatenate([data, data[::-1]], axis=0)
    spectrum = fft.rfft(extended, axis=0, **workers)
    del extended
    spectrum *= mask
    return fft.irfft(spectrum, 2 * n, axis=0, **workers)[:n].astype(xp.float32, copy=False)


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


# Bytes of level data per temporal-filter chunk; bounds the FFT temporaries
_FILTER_CHUNK_BYTES = 32 * 1024 ** 2


def pyramid_shapes(height, width, pyramid_levels):
    """(height, width) of each pyramid level, finest first."""
    shapes = [(height, width)]
    for _ in range(1, pyramid_levels):
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    return shapes


def magnify_blocks(read_block, num_frames, height, width, fps, freq_min, freq_max,
                   alpha, pyramid_levels=4, lambda_c=16, chrom_attenuation=1.0,
                   xp=np):
    """The EVM pipeline, yielding the output in blocks of frames.

    `read_block(start, end)` returns frames [start, end) as YIQ float32 on
    `xp`; it is called twice per block (to build the pyramid, and to add
    the result back), so the input never has to be held as floats. Yields
    (start, end, magnified YIQ frames).

    Collapsing a Laplacian pyramid is linear, and only the levels with a
    non-zero gain change, so the output is
        input + collapse(gain_i * bandpass(level_i) for amplified levels).
    Only the amplified levels are stored (at most 1/3 of the video's size;
    level 0 and the low-pass residual never are).

    Steps, following the reference MATLAB implementation
    (amplify_spatial_lpyr_temporal_ideal.m):
    1. Build the amplified Laplacian levels (spatial decomposition)
    2. Ideal bandpass filter each of them over time
    3. Amplify with the adaptive per-level alpha from lambda_c
    4. Attenuate the I/Q (colour) channels by chrom_attenuation
    5. Collapse the amplified signal and add it to the input
    """
    blocks = [(a, min(a + _PYR_BLOCK, num_frames)) for a in range(0, num_frames, _PYR_BLOCK)]
    level_alphas = compute_level_alphas(height, width, pyramid_levels, alpha, lambda_c)
    amplified = [i for i, a in enumerate(level_alphas) if a > 0]
    if not amplified:
        for start, end in blocks:
            yield start, end, read_block(start, end)
        return
    shapes = pyramid_shapes(height, width, pyramid_levels)
    deepest = max(amplified)
    levels = {i: xp.empty((num_frames,) + shapes[i] + (3,), dtype=xp.float32)
              for i in amplified}

    print("Building Laplacian pyramid...")
    t0 = time.time()
    for start, end in blocks:
        gauss = read_block(start, end)
        for i in range(deepest + 1):
            down = pyr_down(gauss)
            if i in levels:
                levels[i][start:end] = gauss - pyr_up(down, shapes[i])
            gauss = down
    _sync(xp)
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Filtering and amplifying...")
    t0 = time.time()
    for i in amplified:
        gains = xp.asarray(np.float32(level_alphas[i])
                           * np.array([1, chrom_attenuation, chrom_attenuation], np.float32))
        pixels = levels[i].reshape(num_frames, -1, 3)  # a view
        step = max(1, _FILTER_CHUNK_BYTES // (num_frames * 3 * 4))
        for p in range(0, pixels.shape[1], step):
            chunk = pixels[:, p:p + step]
            chunk[...] = ideal_bandpass_filter(chunk, fps, freq_min, freq_max) * gains
    _sync(xp)
    print(f"  Done in {format_duration(time.time() - t0)}")

    print("Reconstructing...")
    t0 = time.time()
    for start, end in blocks:
        delta = levels[deepest][start:end]
        for i in range(deepest - 1, -1, -1):
            delta = pyr_up(delta, shapes[i])
            if i in levels:
                delta += levels[i][start:end]
        yield start, end, read_block(start, end) + delta
    _sync(xp)
    print(f"  Done in {format_duration(time.time() - t0)}")


def eulerian_magnification(video, fps, freq_min, freq_max, alpha,
                           pyramid_levels=4, lambda_c=16,
                           chrom_attenuation=1.0, out=None):
    """Run the Eulerian Video Magnification pipeline on a whole video.

    Args:
        video: Input video in YIQ color space (num_frames, H, W, 3), a
            NumPy array (CPU) or a CuPy array (GPU).
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
        out: Array to write the result into; pass `video` itself to
            magnify in place and save its memory. Default: a new array.

    See magnify_blocks for how it works.
    """
    xp = _backend(video)[0]
    if out is None:
        out = xp.empty_like(video)
    total_start = time.time()
    for start, end, block in magnify_blocks(
            lambda a, b: video[a:b], video.shape[0], video.shape[1], video.shape[2],
            fps, freq_min, freq_max, alpha, pyramid_levels, lambda_c,
            chrom_attenuation, xp):
        out[start:end] = block
    print(f"Total processing time: {format_duration(time.time() - total_start)}")
    return out


def estimate_vram_bytes(num_frames, height, width, pyramid_levels):
    """Estimate peak GPU memory of magnify_blocks in bytes.

    The frames stay on the host as uint8. The GPU holds the amplified
    levels (levels 1 to N-2, float32, all frames), a few blocks of
    _PYR_BLOCK full-resolution frames, and the temporal filter's FFT
    buffers and cuFFT plans, which scale with _FILTER_CHUNK_BYTES.
    """
    shapes = pyramid_shapes(height, width, pyramid_levels)
    levels = num_frames * sum(h * w for h, w in shapes[1:-1]) * 3 * 4
    block = _PYR_BLOCK * height * width * 3 * 4
    # ponytail: 4 blocks + 11 chunks fitted to cupy memory-pool peaks on an
    # RTX 4050 (face.mp4 at 60/150/301 frames, 1080p x 60, 720p x 200,
    # 240p x 900; all within 25%); re-fit if the pipeline's buffers change
    return levels + 4 * block + 11 * _FILTER_CHUNK_BYTES


def check_vram(num_frames, height, width, pyramid_levels, device_id):
    """Check if GPU has enough VRAM. Exit with error if not."""
    required = estimate_vram_bytes(num_frames, height, width, pyramid_levels)
    import cupy
    free, total = cupy.cuda.Device(device_id).mem_info

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification — amplify subtle temporal "
                    "variations in video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evm.py -i face.mp4\n"
            "  python evm.py -i face.mp4 -o pulse.avi -a 50 "
            "-fl 0.83 -fh 1.0 --lambda-c 1000\n"
            "  python evm.py -i guitar.mp4 -fl 72 -fh 92 -a 50 "
            "--lambda-c 10 --chrom-attenuation 0\n"
            "  python evm.py -i face.mp4 --gpu --device 0"
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
        '-a', '--amplification', type=float, default=10,
        help='Amplification factor alpha: in-band changes are multiplied '
             'by 1 + alpha on each amplified pyramid level (default: 10)'
    )
    parser.add_argument(
        '--pyramid-levels', type=int, default=4,
        help='Number of Laplacian pyramid levels (default: 4)'
    )
    parser.add_argument(
        '--lambda-c', type=float, default=16,
        help='Cutoff spatial wavelength in pixels (default: 16). '
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

    parser.add_argument(
        '--gpu', action='store_true',
        help='Run on an NVIDIA GPU with CuPy (install requirements-cuda.txt)'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='CUDA device ID for --gpu (default: 0)'
    )

    args = parser.parse_args(argv)

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

    # --- GPU setup ---
    xp = np
    if args.gpu:
        try:
            import cupy as xp
        except ImportError:
            print("Error: --gpu requires CuPy; install requirements-cuda.txt "
                  "or use Dockerfile.cuda", file=sys.stderr)
            sys.exit(1)
        try:
            xp.cuda.Device(args.device).use()
            name = xp.cuda.runtime.getDeviceProperties(args.device)['name']
        except xp.cuda.runtime.CUDARuntimeError:
            print(f"Error: CUDA device {args.device} not available",
                  file=sys.stderr)
            sys.exit(1)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        print(f"Using GPU: {name} (device {args.device})")

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
        frames, fps = read_frames(args.input, fps=args.fps)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    n_frames, height, width = frames.shape[:3]
    print(f"  {n_frames} frames, {width}x{height}, {fps} fps")
    if args.gpu:
        check_vram(n_frames, height, width, args.pyramid_levels, args.device)

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
          f"(finest to coarsest)")
    print(f"  Pyramid levels:      {args.pyramid_levels}")
    print(f"  Lambda_c:            {args.lambda_c}")
    print(f"  Chrom attenuation:   {args.chrom_attenuation}\n")

    # --- Run and save ---
    # The frames stay uint8 on the host; each block is converted to YIQ
    # when needed and written as soon as it is magnified.
    try:
        writer = open_writer(args.output, fps, (width, height))
        t0 = time.time()
        try:
            for _, _, block in magnify_blocks(
                    lambda a, b: frames_to_yiq(frames[a:b], xp),
                    n_frames, height, width, fps,
                    freq_min=args.freq_low,
                    freq_max=args.freq_high,
                    alpha=args.amplification,
                    pyramid_levels=args.pyramid_levels,
                    lambda_c=args.lambda_c,
                    chrom_attenuation=args.chrom_attenuation,
                    xp=xp):
                write_yiq(writer, block)
        except BaseException:
            writer.release()
            raise
        close_writer(writer, args.output)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Total processing time: {format_duration(time.time() - t0)}")


if __name__ == '__main__':
    main()
