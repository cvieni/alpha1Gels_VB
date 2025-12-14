from scipy.signal import find_peaks
import numpy as np

def extract_features(trace, K=6, min_peak_height=0.2, min_peak_distance=10):
    """
    Extract peak-based features from a 1D gel lane trace.

    Features included:
    - Number of peaks
    - Mean height of peaks
    - Top-K peak heights (normalized to max intensity)
    - Top-K peak positions (normalized to trace length)
    
    Parameters
    ----------
    trace : array-like
        1D intensity profile along a gel lane
    K : int, default 6
        Number of top peaks to keep
    min_peak_height : float, default 0.2
        Minimum peak height (after normalization) to consider
    min_peak_distance : int, default 10
        Minimum distance between peaks in pixels

    Returns
    -------
    features : np.ndarray, shape=(2 + 2*K,)
        Peak-based feature vector
    """
    
    trace = np.asarray(trace)
    L = len(trace)
    
    # Normalize to [0,1]
    trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-8)
    
    # Detect peaks
    peaks, props = find_peaks(trace_norm, height=min_peak_height, distance=min_peak_distance)
    peak_heights = props['peak_heights'] if 'peak_heights' in props else []
    
    num_peaks = len(peaks)
    mean_peak_height = np.mean(peak_heights) if num_peaks > 0 else 0.0
    
    # Top-K peak heights (normalized to max intensity)
    if num_peaks > 0:
        idx_sorted = np.argsort(peak_heights)[::-1]  # descending order
        top_k_heights = peak_heights[idx_sorted][:K] / trace_norm.max()
    else:
        top_k_heights = np.zeros(K)
    
    # Top-K peak positions (normalized to trace length)
    if num_peaks > 0:
        top_k_positions = peaks[idx_sorted][:K] / L
    else:
        top_k_positions = np.zeros(K)
    
    # Pad if fewer than K peaks
    if len(top_k_heights) < K:
        top_k_heights = np.pad(top_k_heights, (0, K - len(top_k_heights)), 'constant')
        top_k_positions = np.pad(top_k_positions, (0, K - len(top_k_positions)), 'constant')
    
    # Build feature vector
    features = [num_peaks, mean_peak_height] + list(top_k_heights) + list(top_k_positions)
    
    return np.array(features, dtype=np.float32)
