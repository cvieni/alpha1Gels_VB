# ---------------------
# ---- Import required packages ------
# ---------------------

import os
import pandas as pd
import numpy as np
import cv2
import math
import importlib # used to reload my other files of function

import function_file
import plot_GelVisualization_functions as viz
# Reload modules to ensure latest version
importlib.reload(function_file)
importlib.reload(viz)

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
output_traces = os.path.join(working_direct, "output_traces/")
y_start, y_end = 800, 1400
x_start, x_end = 850, 1960

image_files = [f for f in os.listdir(img_direct) if f.lower().endswith(('.jpg','.png','.tif'))]

# ---------------------
# Loop through images
# ---------------------
for img_name in image_files:
    img_path = os.path.join(img_direct, img_name)
    print(f"\nProcessing: {img_name}")

    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load image {img_name}, skipping")
        continue

    # Crop
    gel_crop = img[y_start:y_end, x_start:x_end]

    # Preprocess
    proc, bg = function_file.preprocess_gel(gel_crop)
    proc_clean = function_file.remove_small_components(proc)
    # proc_clean = cv2.equalizeHist(proc_clean)

    # ---------------------
    # Detect lanes using multiple methods
    # ---------------------
    lanes_thresh, _ = function_file.detect_lanes(proc_clean)
    traces_thresh = function_file.extract_lane_traces(proc_clean, lanes_thresh)

    lanes_peak, _ = function_file.detect_lanes_peaks(proc_clean)
    traces_peak = function_file.extract_lane_traces(proc_clean, lanes_peak)

    lanes_adapt, _ = function_file.detect_lanes_adaptive(proc_clean)
    traces_adapt = function_file.extract_lane_traces(proc_clean, lanes_adapt)

    # ---------------------
    # Save traces for CNN training
    # ---------------------
    np.save(os.path.join(output_traces, f"{img_name}_traces_thresh.npy"), traces_thresh)
    np.save(os.path.join(output_traces, f"{img_name}_lanes_thresh.npy"), lanes_thresh)

    # ---------------------
    # Visualize for this image
    # ---------------------
    # Create a folder inside output_pngs for this image
    img_name_noext = os.path.splitext(img_name)[0]  # removes file extension
    save_folder = os.path.join(output_direct, f"{img_name_noext}_peak_overlay")
    os.makedirs(save_folder, exist_ok=True)

    viz.lane_overlay_side_by_side(
        gel_img=gel_crop,
        traces=traces_thresh,
        lanes=lanes_thresh,
        lane_index=1,
        output_direct=save_folder,
        prefix="thresh_"
    )
    viz.visualize_lane_detection(
        gel_img=gel_crop,
        lanes=lanes_thresh,
        output_direct=save_folder,
        prefix="thresh_"
    )

print("\n✅ Finished processing all images.")
