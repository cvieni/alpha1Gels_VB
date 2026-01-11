# ---------------------
# ---- Import required packages ------
# ---------------------

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------
# ---- PREPROCESSING FUNCTIONS ------
# ---------------------

def preprocess_gel(img, return_steps=False):
    """
    Preprocess gel image.
    If return_steps=True, returns a dict of intermediate steps.
    """

    # Ensure grayscale
    if img.ndim == 3:
        g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g0 = img.copy()

    # Invert (bands bright)
    g1 = cv2.bitwise_not(g0)

    # Background estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(g1, cv2.MORPH_OPEN, kernel)

    # Background subtraction
    g2 = cv2.subtract(g1, bg)

    # Smoothing
    g3 = cv2.medianBlur(g2, 3)

    if return_steps:
        return {
            "original": g0,
            "inverted": g1,
            "background": bg,
            "bg_subtracted": g2,
            "smoothed": g3
        }

    return g3, bg


def visualize_preprocessing_steps(steps, title, save_path=None):
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, (name, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()






# ---------------------
# ---- Define Project Directories ------
# ---------------------

projectdirect = r"c:\Users\m297055\OneDrive - Mayo Clinic\Documents\Research\Wilrich_alpha1_Isofocus_machinelearning\alpha1Gels_VB"
# projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
working_direct = projectdirect
# working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/")
# C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/Wilrich_alpha1_Isofocus_machinelearning/gel_scans

img_direct = os.path.join(working_direct, "gel_scans/")
output_direct = os.path.join(working_direct, "output_pngs/")
output_traces = os.path.join(working_direct, "output_traces/")
y_start, y_end = 800, 1400
x_start, x_end = 850, 1960

# ---------------------
# ---- Load FIRST image only ------
# ---------------------

image_files = sorted(
    f for f in os.listdir(img_direct)
    if f.lower().endswith((".jpg", ".png", ".tif"))
)

if not image_files:
    raise RuntimeError("No images found in img_direct")

img_name = image_files[0]
img_path = os.path.join(img_direct, img_name)

print(f"\nProcessing ONLY first image: {img_name}")

# ---------------------
# ---- Load, crop, preprocess ------
# ---------------------

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError(f"Could not load image {img_name}")

gel_crop = img[y_start:y_end, x_start:x_end]

steps = preprocess_gel(gel_crop, return_steps=True)

# ---------------------
# ---- Visualize preprocessing ------
# ---------------------

img_name_noext = os.path.splitext(img_name)[0]
save_path = os.path.join(output_direct, f"{img_name_noext}_preprocessing_steps.png")

visualize_preprocessing_steps(
    steps,
    title=f"{img_name_noext} – Preprocessing Pipeline",
    save_path=save_path
)

print("✅ Preprocessing visualization complete.")