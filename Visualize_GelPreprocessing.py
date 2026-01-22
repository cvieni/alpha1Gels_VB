# ---------------------
# ---- Import required packages ------
# ---------------------

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

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
    g1b = 255-g0 # explicit inversion (for comparison)

    # Convert once to float for math
    g1f = g1.astype(np.float64)
    
    # ============================================================
    # Background estimation (kernel = 51)
    # ============================================================
    #  create an ellipse to probe the image -> 51 pixels wide/tall
    # morphopen -> erosion to shrink bright region then dilation -> grow back to original size (with removal of small features)
    kernel_51 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg_51 = cv2.morphologyEx(g1f, cv2.MORPH_OPEN, kernel_51)
    g2_51 = g1f - bg_51


    # ============================================================
    # Background estimation via TOP-HAT (kernel = 61)
    # ============================================================
    kernel_61 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    bg_61 = cv2.morphologyEx(g1f, cv2.MORPH_OPEN, kernel_61)
    g2_61 = g1f - bg_61

    # ============================================================
    # Background estimation via TOP-HAT (kernel = 81)
    # ============================================================
    kernel_81 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    bg_81 = cv2.morphologyEx(g1f, cv2.MORPH_OPEN, kernel_81)
    g2_81 = g1f - bg_81

    # ============================================================
    # Equal-energy normalization (on background-subtracted image)
    # ============================================================
    g2c_min = g1f - g1f.min()
    den = g1f.max() - g1f.min()
    g2d_norm = (g1f - g1f.min()) / den if den > 0 else np.zeros_like(g1f)

    energy = np.sum(g1f ** 2)
    g2e_energy = g1f / np.sqrt(energy) if energy > 0 else np.zeros_like(g1f)

    # ============================================================
    # Kernel sweep (visual inspection only)
    # ============================================================
    bg_sweep = {}
    for k in [21, 31, 51, 81]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bgk = cv2.morphologyEx(g1f, cv2.MORPH_OPEN, kernel)
        bg_sweep[f"bg_kernel_{k}"] = bgk




    # ============================================================
    # Smoothing
    # ============================================================
    g2_51_u8 = cv2.normalize(g2_51, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g3 = cv2.medianBlur(g2_51_u8, 3)

    if return_steps:
        return {
            "original": g0,
            "inverted": g1,
            "inverted subtraction": g1b,

            # --- Foregrounds ---
            "bg-subtracted (k=51)": g2_51,
            "bg-subtracted (k=61)": g2_61,
            "bg-subtracted (k=81)": g2_81,

            # --- Comparisons ---
            "subtract minimum": g2c_min,
            "subtract + normalize": g2d_norm,
            "normalize to equal energy": g2e_energy,

            # --- Final ---
            "smoothed": g3,

            # --- Diagnostics ---
            **bg_sweep
        }

    return g3, bg_51


def visualize_preprocessing_steps(steps, title, save_path=None):
    n = len(steps)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, (name, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[len(steps):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()




# ---------------------
# ---- Define Project Directories ------
# ---------------------

# projectdirect = r"c:\Users\m297055\OneDrive - Mayo Clinic\Documents\Research\Wilrich_alpha1_Isofocus_machinelearning\alpha1Gels_VB"
projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
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