from scipy.signal import resample
import numpy as np

def extract_features(trace, downsample_len=100):
    trace = np.asarray(trace)
    # ----- Normalize -----
    trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-8)
    # ---------------------------
    # Only downsampled trace
    # ---------------------------
    down = resample(trace_norm, downsample_len)
    return np.array(down, dtype=np.float32)
