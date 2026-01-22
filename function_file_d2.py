# function_file.py

import os
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. PREPROCESS GEL IMAGE
# ------------------------------------------------------------
def preprocess_gel(img):
    """
    Accepts either file path (str) or numpy array.
    Returns processed gel and estimated background.
    """
    # If a path is provided, read the image
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Image not found: " + img)

    # Convert to grayscale if needed
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()

    # Make bands bright
    g = cv2.bitwise_not(g)

    # Background estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)

    # Remove background
    g_proc = cv2.subtract(g, bg)

    # Light smoothing
    g_proc = cv2.medianBlur(g_proc, 3)

    return g_proc, bg


# ------------------------------------------------------------
# 2. REMOVE SMALL NOISE COMPONENTS
# ------------------------------------------------------------
def remove_small_components(img, min_pixels=10):
    """Remove specks/noise from binary or grayscale images."""
    if img.sum() == 0:
        return img

    # Ensure binary
    bin_img = img > 0

    structure = ndimage.generate_binary_structure(2, 1)
    labeled, num = ndimage.label(bin_img, structure)

    cleaned = np.zeros_like(bin_img)

    for i in range(1, num + 1):
        mask = labeled == i
        if mask.sum() >= min_pixels:
            cleaned[mask] = 1

    return (cleaned * img).astype(np.uint8)


# ------------------------------------------------------------
# 3. LANE DETECTION (threshold method)
# ------------------------------------------------------------
def detect_lanes(processed_img, min_lane_width=20, threshold_ratio=0.2):
    """
    innput a preprocessed image (background subtracted) -> then collapse images to vertical projection -> detect bright regions along x axis
    Use vertical projection to find lanes.
    Returns list of (x_start, x_end) for each lane.
    """
    col_profile = processed_img.mean(axis=0)
    col_norm = col_profile / col_profile.max()
    lane_mask = col_norm > threshold_ratio

    lanes = []
    start = None
    for i, val in enumerate(lane_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if (i - start) >= min_lane_width:
                lanes.append((start, i))
            start = None
    if start is not None and (len(lane_mask) - start) >= min_lane_width:
        lanes.append((start, len(lane_mask)))

    return lanes, col_profile


# ------------------------------------------------------------
# 3b. LANE DETECTION (peak detection method)
# ------------------------------------------------------------
def detect_lanes_peaks(processed_img, min_lane_width=20, min_height_ratio=0.2, distance=20):
    """
    Detect lanes using peaks in the vertical projection (column mean).
    Returns list of (x_start, x_end) for each lane and the column profile.
    """
    col_profile = processed_img.mean(axis=0)
    col_norm = col_profile / col_profile.max()

    # Determine minimum peak height
    min_height = min_height_ratio * col_profile.max()

    # Find peaks
    peaks, _ = find_peaks(col_profile, height=min_height, distance=distance)

    lanes = []
    for peak in peaks:
        start = max(peak - min_lane_width // 2, 0)
        end = min(peak + min_lane_width // 2, processed_img.shape[1])
        lanes.append((start, end))

    return lanes, col_profile


# ------------------------------------------------------------
# 3c. LANE DETECTION (Adaptive Threshold method)
# ------------------------------------------------------------
def detect_lanes_adaptive(proc_img, block_size=51, C=5, min_lane_width=20):
    """
    Detect lanes using adaptive thresholding for uneven illumination.
    Returns list of (x_start, x_end) and the column profile.
    """
    import cv2
    import numpy as np

    # Ensure 8-bit image
    proc_8bit = cv2.normalize(proc_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        proc_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )

    # Column sum
    col_sum = adaptive.sum(axis=0)
    lane_mask = col_sum > 10

    # Extract lane start/end indices
    lanes = []
    start = None
    for i, val in enumerate(lane_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if (i - start) >= min_lane_width:
                lanes.append((start, i))
            start = None
    if start is not None and (len(lane_mask) - start) >= min_lane_width:
        lanes.append((start, len(lane_mask)))

    return lanes, col_sum


# ------------------------------------------------------------
# 4. EXTRACT 1D DENSITY TRACE FOR EACH LANE
# ------------------------------------------------------------
def extract_lane_traces(proc_img, lanes):
    traces = []
    for (x1, x2) in lanes:
        lane_strip = proc_img[:, x1:x2]
        trace = lane_strip.mean(axis=1)
        traces.append(trace)
    return traces


# ------------------------------------------------------------
# There are 2 main intense peaks (some lanes have band 2, some have band 3 as most intense)
# I think the best way forward will be to realign on the average of the two peaks rather than on the max peak (which could be band 2 or 3)
# ------------------------------------------------------------
def align_traces_by_two_peaks(traces, target_index=None, min_distance=20):
    """
    Align traces so the midpoint of the two strongest peaks is at target_index.

    Parameters
    ----------
    traces : np.ndarray (n_lanes, n_points)
    target_index : int or None
        If None, uses the median reference across lanes
    min_distance : int
        Minimum distance between peaks (in pixels)

    Returns
    -------
    aligned_traces : np.ndarray
    peak_info : list of dicts (for diagnostics)
    """

    refs = []
    peak_info = []

    # --- Find reference location for each trace ---
    for trace in traces:
        peaks, props = find_peaks(trace, distance=min_distance)

        if len(peaks) < 2:
            refs.append(None)
            peak_info.append({"peaks": peaks})
            continue

        # Take two highest peaks
        top2 = peaks[np.argsort(trace[peaks])[-2:]]
        top2 = np.sort(top2)

        ref = int(np.mean(top2))
        refs.append(ref)

        peak_info.append({
            "peaks": top2,
            "ref": ref
        })

    # --- Choose global target index ---
    valid_refs = [r for r in refs if r is not None]
    if not valid_refs:
        raise RuntimeError("No valid traces with two peaks found")

    if target_index is None:
        target_index = int(np.median(valid_refs))

    # --- Align traces ---
    aligned = np.zeros_like(traces)

    for i, (trace, ref) in enumerate(zip(traces, refs)):
        if ref is None:
            aligned[i] = trace
            continue

        shift = target_index - ref
        aligned[i] = np.roll(trace, shift)

    return aligned, peak_info
