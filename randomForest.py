from scipy.signal import find_peaks, resample
from scipy.stats import skew, kurtosis
import numpy as np


def extract_features(trace, K=6, downsample_len=100):
    trace = np.asarray(trace)
    L = len(trace)

    # ----- Normalize -----
    trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-8)

    # ---------------------------
    # 1. Global statistics
    # ---------------------------
    features = [
        trace_norm.mean(),
        trace_norm.std(),
        skew(trace_norm),
        kurtosis(trace_norm),
        trace_norm.min(),
        trace_norm.max(),
    ]

    # ---------------------------
    # 2. Peak detection
    # ---------------------------
    peaks, props = find_peaks(trace_norm, height=0.2, distance=10)
    peak_heights = props['peak_heights'] if 'peak_heights' in props else []

    num_peaks = len(peaks)
    mean_peak_height = np.mean(peak_heights) if num_peaks > 0 else 0

    features += [
        num_peaks,
        mean_peak_height,
    ]

    # ---------------------------
    # 3. Top-K normalized peak positions
    # ---------------------------
    if num_peaks > 0:
        # sort by height (descending)
        idx_sorted = np.argsort(peak_heights)[::-1]
        top_k_peaks = peaks[idx_sorted][:K] / L
    else:
        top_k_peaks = np.zeros(K)

    # pad if fewer peaks
    if len(top_k_peaks) < K:
        top_k_peaks = np.pad(top_k_peaks, (0, K - len(top_k_peaks)))

    features += list(top_k_peaks)

    # ---------------------------
    # 4. Downsampled trace
    # ---------------------------
    down = resample(trace_norm, downsample_len)
    features += list(down)

    return np.array(features, dtype=np.float32)
