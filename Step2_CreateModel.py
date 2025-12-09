import os
import numpy as np
import importlib
import function_model as CreateModel
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt

# Reload in case you edit the module
importlib.reload(CreateModel)

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
output_predictions = os.path.join(projectdirect, "output_predictions/")
os.makedirs(output_predictions, exist_ok=True)


# ------------------------------------------
# --- load your traces from lane detection
# ------------------------------------------
traces_files = [f for f in os.listdir(output_traces) if f.lower().endswith('_traces_thresh.npy')]

all_traces = [] 
traces_thresh = []
file_trace_map = []  # keep track of which trace came from which file

for f in traces_files:
    file_path = os.path.join(output_traces, f)
    loaded_traces = np.load(file_path, allow_pickle=True)
    all_traces.extend(loaded_traces)
    file_trace_map.extend([f] * len(loaded_traces))  # associate each lane with its source file

print(f"Loaded {len(all_traces)} lane traces from {len(traces_files)} files.")

# Visualize a trace to ensure they are loaded correctly:# Pick a trace to visualize, e.g., the first one
# trace = all_traces[0]

# plt.figure(figsize=(8,4))
# plt.plot(trace, color='blue')
# plt.title("Lane Trace")
# plt.xlabel("Position along lane")
# plt.ylabel("Intensity / Density")
# plt.grid(True)
# plt.show()



# -----------------------------
# 1. Prepare dataset
# -----------------------------
# X, y = CreateModel.prepare_dataset(traces)
X, y = CreateModel.prepare_dataset(all_traces)

# Split into 20:80 training/validation set
training_size=0.7
validation_size = 1 - training_size
X_train, X_val, y_train, y_val, map_train, map_val = train_test_split(
    X, y, file_trace_map, test_size=training_size, random_state=42
)

print(f"Training set: {len(X_train)} lanes, Validation set: {len(X_val)} lanes")

# -----------------------------
# 2. Train CNN
# -----------------------------
model = CreateModel.train_model(X, y, num_epochs=50, batch_size=4, device='cpu')  # or 'cuda'

# -----------------------------
# 3. Predict on same data
# -----------------------------
preds_all = CreateModel.predict(model, all_traces, device='cpu')

# -----------------------------
# 5. Save predictions grouped by original file
# -----------------------------
preds_per_file = defaultdict(list)
for trace_pred, fname in zip(preds_all, file_trace_map):
    preds_per_file[fname].append(trace_pred)

for fname, preds in preds_per_file.items():
    save_path = os.path.join(output_predictions, fname.replace("_traces_thresh.npy", "_cnn_pred.npy"))
    np.save(save_path, np.array(preds, dtype=object))  # object array for variable-length lanes
    print(f"Saved CNN predictions for {fname} → {save_path}")

# -----------------------------
# 6. Optional: visualize first lane of first file
# -----------------------------
CreateModel.visualize_prediction(all_traces[0], preds_all[0])