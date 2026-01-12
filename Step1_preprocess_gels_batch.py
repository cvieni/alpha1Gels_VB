# ---------------------
# ---- Import required packages ------
# ---------------------

import os
import pandas as pd
import numpy as np
import cv2
import math
import importlib # used to reload my other files of function

import function_file_d2 as function_file
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

    # Convert traces to np.ndarray if they are lists
    traces_thresh_arr = np.array(traces_thresh)
    # traces_peak_arr   = np.array(traces_peak)
    # traces_adapt_arr  = np.array(traces_adapt)
    # Align using top two peaks
    # traces_thresh_aligned, info_thresh = function_file.align_traces_by_two_peaks(traces_thresh_arr)
    traces_thresh_aligned, peak_info = function_file.align_traces_by_two_peaks(traces_thresh_arr)
    # traces_peak_aligned, info_peak     = function_file.align_traces_by_two_peaks(traces_peak_arr)
    # traces_adapt_aligned, info_adapt   = function_file.align_traces_by_two_peaks(traces_adapt_arr)

    # ---------------------
    # Save traces for CNN training
    # ---------------------
    np.save(os.path.join(output_traces, f"{img_name}_traces_thresh.npy"), traces_thresh)
    np.save(os.path.join(output_traces, f"{img_name}_lanes_thresh.npy"), lanes_thresh)

    # ---------------------
    # Save Aligned traces for CNN training
    # ---------------------
    np.save(os.path.join(output_traces, f"{img_name}_traces_thresh_aligned.npy"), traces_thresh_aligned)
    # np.save(os.path.join(output_traces, f"{img_name}_traces_peak_aligned.npy"), traces_peak_aligned)
    # np.save(os.path.join(output_traces, f"{img_name}_traces_adapt_aligned.npy"), traces_adapt_aligned)

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

    # Determine global alignment index (median of all valid refs)
    valid_refs = [info["ref"] for info in peak_info if info["ref"] is not None]
    if valid_refs:
        target_index = int(np.median(valid_refs))
    else:
        target_index = traces_thresh_arr.shape[1] // 2  # fallback

    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(6, 4))

    n_lanes = len(traces_thresh_aligned)
    cmap = plt.cm.coolwarm  # red → blue

    for i, t in enumerate(traces_thresh_aligned):
        color = cmap(i / (n_lanes - 1))  # normalized index for colormap
        ax.plot(t, alpha=0.8, color=color)

    # Add vertical dashed line at alignment
    ax.axvline(target_index, color='black', linestyle='--', linewidth=1.5, label='Alignment point')

    ax.set_title(f"{img_name_noext} – Aligned traces (two-peak midpoint)")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Intensity")
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Add colorbar properly
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_lanes))
    sm.set_array([])  # needed for ScalarMappable
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Lane index')

    # Save the plot
    plt_path = os.path.join(save_folder, f"{img_name_noext}_aligned_traces.png")
    plt.savefig(plt_path, dpi=300)
    # plt.show()

print("\n✅ Finished processing all images.")
