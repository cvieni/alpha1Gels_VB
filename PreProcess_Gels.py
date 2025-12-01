# ---------------------
# ---- Check if imports are installed -> if Not install them ------
# ---------------------
import importlib
import subprocess
import sys


def install_if_missing(package):
    """Check if a package is installed, and install it if not."""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"📦 Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["matplotlib", "numpy", "pandas", 
    "opencv-python", "Pillow", "PYMIC", "torch", 
    "scikit-image", "scikit-learn",
    "tensorboard", "torchvision"]:
    
    install_if_missing(pkg)

# ---------------------
# ---- Import required packages ------
# ---------------------

import os
import pandas as pd
import numpy as np
import cv2
import math
import importlib # used to reload my other files of function
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------
# ---- Define Project Directories ------
# ---------------------

# projectdirect = "C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/"
projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
working_direct = projectdirect
# working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/")
# C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/Wilrich_alpha1_Isofocus_machinelearning/gel_scans

img_direct = os.path.join(working_direct, "gel_scans/")
output_direct = os.path.join(working_direct, "output_pngs/")


test_image = "25331.jpg"  # Change this to your PDF file
test_img_path = os.path.join(img_direct, test_image)

test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

y_start, y_end = 800, 1400
x_start, x_end = 850, 1960
gel_crop = test_image[y_start:y_end, x_start:x_end]

# ------------------------------------------------------
#  Uncomment to see the Loaded & Cropped Images ----------
# ------------------------------------------------------
# # Check if image is properly loaded -----
# if test_image is None:
#     raise FileNotFoundError(f"Could not load: {test_img_path}")

# # Show the full image
# plt.imshow(test_image, cmap="gray")
# plt.title("Loaded Test Image")
# plt.axis("off")
# plt.show()
# # --------------------

# # Crop the image to improve alignment for lane calling: ----------
# plt.figure(figsize=(8, 10))
# plt.imshow(gel_crop, cmap="gray", aspect='auto')
# plt.title("Cropped Gel Region")
# plt.axis("off")
# plt.show()
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------



# ============================================================
# ============================================================
# MAIN: LOAD IMAGE → PROCESS → DETECT LANES → PLOT TRACE
# ============================================================
# ============================================================


# ---------------------
# ---- Load function File ------
# ---------------------
import function_file
importlib.reload(function_file)

import plot_GelVisualization_functions as viz
importlib.reload(viz)


# ---------------------
# ---- Run preprocessing ----
# ---------------------
proc, bg = function_file.preprocess_gel(gel_crop)
proc_clean = function_file.remove_small_components(proc)
proc = cv2.equalizeHist(proc)  # after preprocessing

# ---------------------
# ---- Run preprocessing ----
# ---------------------
proc, bg = function_file.preprocess_gel(gel_crop)
proc_clean = function_file.remove_small_components(proc)
proc = cv2.equalizeHist(proc)  # optional histogram equalization

# ---------------------
# ---- Detect lanes ----
# ---------------------
#  Threshold method
lanes_thresh, col_profile_thresh = function_file.detect_lanes(proc_clean)
traces_thresh = function_file.extract_lane_traces(proc_clean, lanes_thresh)

# Peak method
lanes_peak, col_profile_peak = function_file.detect_lanes_peaks(proc_clean)
traces_peak = function_file.extract_lane_traces(proc_clean, lanes_peak)

# Adaptive threshold method
lanes_adapt, col_profile_adapt = function_file.detect_lanes_adaptive(proc_clean)
traces_adapt = function_file.extract_lane_traces(proc_clean, lanes_adapt)

print("Threshold lanes:", lanes_thresh)
print("Peak lanes:", lanes_peak)
print("Adaptive lanes:", lanes_adapt)

# ---------------------
# ---- VISUALIZE: Threshold method ----
# ---------------------
viz.plot_all_lanes(traces_thresh, output_direct, prefix="thresh_")
viz.plot_lanes_grid(traces_thresh, output_direct, rows=2, prefix="thresh_")
viz.plot_single_lane(traces_thresh, lane_index=1, output_direct=output_direct, prefix="thresh_")
viz.lane_overlay_side_by_side(
    gel_img=gel_crop,
    traces=traces_thresh,
    lanes=lanes_thresh,
    lane_index=1,
    output_direct=output_direct,
    prefix="thresh_"
)
viz.visualize_lane_detection(
    gel_img=gel_crop,
    lanes=lanes_thresh,
    output_direct=output_direct,
    prefix="thresh_"
)

# ---------------------
# ---- VISUALIZE: Peak method ----
# ---------------------
viz.plot_all_lanes(traces_peak, output_direct, prefix="peak_")
viz.plot_lanes_grid(traces_peak, output_direct, rows=2, prefix="peak_")
viz.plot_single_lane(traces_peak, lane_index=1, output_direct=output_direct, prefix="peak_")
viz.lane_overlay_side_by_side(
    gel_img=gel_crop,
    traces=traces_peak,
    lanes=lanes_peak,
    lane_index=1,
    output_direct=output_direct,
    prefix="peak_"
)
viz.visualize_lane_detection(
    gel_img=gel_crop,
    lanes=lanes_peak,
    output_direct=output_direct,
    prefix="peak_"
)

# ---------------------
# ---- VISUALIZE: Adaptive threshold method ----
# ---------------------
viz.plot_all_lanes(traces_adapt, output_direct, prefix="adapt_")
viz.plot_lanes_grid(traces_adapt, output_direct, rows=2, prefix="adapt_")
viz.plot_single_lane(traces_adapt, lane_index=1, output_direct=output_direct, prefix="adapt_")
viz.visualize_lane_detection(
    gel_img=gel_crop,
    lanes=lanes_adapt,
    output_direct=output_direct,
    prefix="adapt_"
)
viz.lane_overlay_side_by_side(
    gel_img=gel_crop,
    traces=traces_adapt,
    lanes=lanes_adapt,
    lane_index=1,
    output_direct=output_direct,
    prefix="adapt_"
)

