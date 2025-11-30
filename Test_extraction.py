# Gel Lane Extraction & 1D Trace Labelling Template
# Requirements: numpy, matplotlib, opencv-python, pandas, scipy

# pip install opencv-python
# pip install -U scikit-learn

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import os 

# projectdirect = "C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/"
projectdirect = "/Users/cvieni/Documents/Path_alpha1Gels_VB"
working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/")
img_direct = os.path.join(working_direct, "images/")

img_filename = "25331.jpeg"  # Change this to your PDF file
img_path = os.path.join(img_direct, img_filename)


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_invert = cv2.bitwise_not(img)

# ---------------------------
# PARAMETERS
# ---------------------------
EXPECTED_LANES = 18
LANE_MIN_WIDTH_PX = 20  # minimum distance between lanes in pixels
TRACE_LENGTH = 512       # number of points in 1D trace
S_WINDOW = (0.40, 0.47) # fraction of trace for S peak
Z_WINDOW = (0.55, 0.62) # fraction of trace for Z peak

# ---------------------------
# 1. LOAD & PREPROCESS GEL IMAGE
# ---------------------------
def preprocess_gel(img_path):
    img = cv2.imread(img_path) # read array of image
    if img.ndim == 3: # 3 = color image; dimensions = 2 = grayscale
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    
    g = cv2.bitwise_not(g)  # make bands bright
    # background removal using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51))
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    g = cv2.subtract(g, bg)
    g = cv2.medianBlur(g, 3) # smooth image with Gaussian blur
    return g

# ---------------------------
# 2. LANE DETECTION
# ---------------------------
def detect_lanes(img, expected_n=EXPECTED_LANES, min_sep=LANE_MIN_WIDTH_PX):
    # Take 2D image and measure sum along the vertical row to create a 1D array of 
    # length width w/ the total intensity of all pixels in that column
    # if 3D (color image) then vx = sum of RGB values in each column
    vx = img.sum(axis=0).astype(np.float32) # axis = 0 -> sum along the vertical axis
    
    # normalize all values to center around 0 -> add 1e-6 in case Std is "0"
    # should improve peak detection as the band threshold becomes relative vs. intensity
    vx = (vx - vx.mean()) / (vx.std() + 1e-6)
    
    # find_peaks = SciPy function to take 1d array and find local maxima
    # "_" is a dictionary that we don't need here
    peaks, _ = find_peaks(vx, distance=min_sep)
    
    # pick top expected_n peaks
    if len(peaks) > expected_n:
        peaks = peaks[np.argsort(vx[peaks])[-expected_n:]]
    peaks = np.sort(peaks)
    
    lanes = []
    half_width = min_sep // 2
    for p in peaks:
        x0 = max(p - half_width, 0)
        x1 = min(p + half_width, img.shape[1]-1)
        lanes.append((x0, x1))
    return lanes

# ---------------------------
# 3. EXTRACT 1D TRACE
# ---------------------------
def lane_trace(img, x0, x1, length=TRACE_LENGTH):
    roi = img[:, x0:x1]
    trace = roi.mean(axis=1)  # collapse to 1D
    # normalize
    trace = (trace - np.percentile(trace, 5)) / (np.percentile(trace, 95) - np.percentile(trace, 5) + 1e-6)
    # resample
    trace_resampled = np.interp(np.linspace(0, len(trace)-1, length), np.arange(len(trace)), trace)
    return trace_resampled


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Optional: invert image if bands are dark
img = cv2.bitwise_not(img)

# ---------------------------
# DEFINE LANE (column range)
# ---------------------------
x0, x1 = 100, 130  # example column indices of a lane

# ---------------------------
# COMPUTE 1D TRACE
# ---------------------------
trace = lane_trace(img, x0, x1)

# ---------------------------
# OVERLAY TRACE ON IMAGE
# ---------------------------
plt.figure(figsize=(8,10))
plt.imshow(img, cmap='gray', aspect='auto')

# Scale the normalized trace to the width of the lane
lane_center = (x0 + x1) / 2
lane_width = x1 - x0
# scale trace to lane width (pixels)
trace_scaled = trace * lane_width
# x-coordinates of the trace: center + trace offset
x_overlay = lane_center - lane_width/2 + trace_scaled
# y-coordinates: linearly from top to bottom
y_overlay = np.linspace(0, img.shape[0]-1, TRACE_LENGTH)

plt.plot(x_overlay, y_overlay, color='red', linewidth=2, label='Lane intensity trace')
plt.title(f"Lane intensity overlay: columns {x0}-{x1}")
plt.xlabel("Horizontal pixels")
plt.ylabel("Vertical pixels")
plt.legend()
plt.gca().invert_yaxis()  # optional: top of gel = 0
plt.show()


# ---------------------------
# 4. PLOT TRACES WITH S/Z WINDOWS
# ---------------------------
def plot_traces(traces, s_win=S_WINDOW, z_win=Z_WINDOW):
    L = traces[0].size
    x = np.linspace(0,1,L)
    plt.figure(figsize=(12,8))
    for i, trace in enumerate(traces):
        plt.plot(x, trace + i*1.2, label=f"Lane {i+1}")  # offset each lane for clarity
    plt.axvspan(*s_win, color='green', alpha=0.2, label='S window')
    plt.axvspan(*z_win, color='red', alpha=0.2, label='Z window')
    plt.xlabel("Normalized migration distance")
    plt.ylabel("Intensity (offset per lane)")
    plt.title("Lane Traces with S/Z Windows")
    plt.legend()
    plt.show()

# ---------------------------
# 5. EXPORT CSV FOR LABELING
# ---------------------------
def export_csv(traces, output_path="lane_traces.csv"):
    rows = []
    for i, t in enumerate(traces):
        rows.append({
            "lane_id": i+1,
            "trace": t.tolist(),
            "label": ""  # placeholder for manual labeling
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

# ---------------------------
# 6. MAIN PIPELINE
# ---------------------------
if __name__ == "__main__":
    gel_path = "gel_example.png"  # replace with your gel image path
    img = preprocess_gel(gel_path)
    
    lanes = detect_lanes(img)
    print(f"Detected {len(lanes)} lanes")
    
    traces = [lane_trace(img, x0,x1) for (x0,x1) in lanes]
    
    plot_traces(traces)
    export_csv(traces)
