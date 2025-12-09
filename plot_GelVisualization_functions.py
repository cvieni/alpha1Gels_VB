# plot_GelVisualization_functions.py

import os
import math
import matplotlib.pyplot as plt

def plot_all_lanes(traces, output_direct, prefix=""):
    os.makedirs(output_direct, exist_ok=True)
    save_path = os.path.join(output_direct, f"{prefix}density_plot_all_lanes.png")

    plt.figure(figsize=(10, 8))
    for i, trace in enumerate(traces):
        plt.plot(trace, label=f"Lane {i+1}")
    plt.gca().invert_yaxis()
    plt.title("1D Density Trace per Lane")
    plt.xlabel("Vertical Position (pixels)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


# ---------------------
# 2. Plot lanes in grid
# ---------------------
def plot_lanes_grid(traces, output_direct, rows=2, prefix=""):
    os.makedirs(output_direct, exist_ok=True)
    num_lanes = len(traces)
    cols = math.ceil(num_lanes / rows)
    save_path = os.path.join(output_direct, f"{prefix}density_trace_grid.png")

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten()
    for i, trace in enumerate(traces):
        ax = axes[i]
        ax.plot(trace)
        ax.invert_yaxis()
        ax.set_title(f"Lane {i+1}")
        ax.grid(True, linestyle="--", alpha=0.3)
    for j in range(num_lanes, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)

# ---------------------
# 3. Plot single lane
# ---------------------
def plot_single_lane(traces, lane_index, output_direct, prefix=""):
    os.makedirs(output_direct, exist_ok=True)
    if lane_index >= len(traces):
        raise IndexError(f"Lane {lane_index+1} does not exist. Only {len(traces)} lanes detected.")
    lane_trace = traces[lane_index]
    save_path = os.path.join(output_direct, f"{prefix}density_trace_lane{lane_index+1}.png")

    plt.figure(figsize=(6, 8))
    plt.plot(lane_trace, color="blue", linewidth=2)
    plt.gca().invert_yaxis()
    plt.title(f"Density Trace — Lane {lane_index+1}")
    plt.xlabel("Vertical Position (pixels)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


# ---------------------
# 4. Lane overlay side by side
# ---------------------
def lane_overlay_side_by_side(gel_img, traces, lanes, lane_index, output_direct, prefix="", line_color="red", alpha=0.7):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_direct, exist_ok=True)
    if lane_index >= len(traces):
        raise IndexError(f"Lane {lane_index+1} does not exist. Only {len(traces)} lanes detected.")
    
    lane_trace = traces[lane_index]
    lane_pos = lanes[lane_index]  # this can be an int or tuple (x_start, x_end)
    save_path = os.path.join(output_direct, f"{prefix}lane{lane_index+1}_overlay.png")

    # Scale the trace to gel width
    trace_scaled = (lane_trace - lane_trace.min()) / (lane_trace.max() - lane_trace.min())
    trace_scaled = trace_scaled * gel_img.shape[1]

    plt.figure(figsize=(12, 6))

    # 1. Lane overlay on gel
    plt.subplot(1, 3, 1)
    plt.imshow(gel_img, cmap="gray", aspect="auto", alpha=alpha)
    # plt.gca().invert_yaxis()
    plt.plot(trace_scaled, range(len(lane_trace)), color=line_color, linewidth=2)
    plt.title(f"Lane {lane_index+1} Overlay")
    plt.axis("off")

    # 2. Only the single lane
    plt.subplot(1, 3, 2)
    if isinstance(lane_pos, tuple):
        x_start, x_end = lane_pos
    else:  # if it's a single x-coordinate, define a small width
        lane_width = 10
        x_start = max(lane_pos - lane_width//2, 0)
        x_end = min(lane_pos + lane_width//2, gel_img.shape[1])
    lane_img = gel_img[:, x_start:x_end]

    plt.imshow(lane_img, cmap="gray", aspect="auto")
    # plt.gca().invert_yaxis()  # flip to match overlay
    plt.axis("off")
    plt.title(f"Lane {lane_index+1} Only")

    # 3. Lane density plot
    plt.subplot(1, 3, 3)
    plt.plot(lane_trace)
    # plt.gca().invert_yaxis()
    plt.title(f"Lane {lane_index+1} Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
    print("Saved:", save_path)


# ---------------------
# 5. Visualize lane detection
# ---------------------
def visualize_lane_detection(gel_img, lanes, output_direct, prefix=""):
    os.makedirs(output_direct, exist_ok=True)
    save_path = os.path.join(output_direct, f"{prefix}lane_detection.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(gel_img, cmap="gray")
    for (x1, x2) in lanes:
        plt.gca().add_patch(plt.Rectangle((x1, 0), x2-x1, gel_img.shape[0],
                                          linewidth=2, edgecolor='red', facecolor='none'))
    plt.title("Lane Detection Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
    print("Saved:", save_path)