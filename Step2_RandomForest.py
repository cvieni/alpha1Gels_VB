import os
import numpy as np
import importlib
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your feature extraction functions
import randomforest as modelfncs
importlib.reload(modelfncs)

# -------------------------
# ---- Define Directories
# -------------------------

projectdirect = "C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/"
# projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
working_direct = projectdirect
# working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/")
# C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/Wilrich_alpha1_Isofocus_machinelearning/gel_scans


img_direct = os.path.join(working_direct, "gel_scans/")
output_direct = os.path.join(working_direct, "output_pngs/")
output_traces = os.path.join(working_direct, "output_traces/")
output_predictions = os.path.join(projectdirect, "output_predictions/")
os.makedirs(output_predictions, exist_ok=True)
label_csv_path = os.path.join(img_direct, "Results_gels_25331_25350.csv") 

# -------------------------
# ---- Load Label CSV
# -------------------------

print("Loading label CSV:", label_csv_path)
df_labels = pd.read_csv(label_csv_path)

# Normalize column names (just in case)
df_labels.columns = [c.lower() for c in df_labels.columns]

if not {"gel", "lane", "label"}.issubset(df_labels.columns):
    raise ValueError("CSV must contain columns: gel, lane, label")

print(f"Loaded {len(df_labels)} label entries.")

# -------------------------
# ---- Load all traces
# -------------------------

print("Loading lane traces...")

trace_files = [
    f for f in os.listdir(output_traces)
    if f.lower().endswith("_traces_thresh.npy")
]

all_traces = []
file_trace_map = []

for f in trace_files:
    file_path = os.path.join(output_traces, f)
    traces = np.load(file_path, allow_pickle=True)

    # Extract gel ID from filename
    # Example: "gel_12_traces_thresh.npy" → gel = "12"
    gel_id = f.replace("_traces_thresh.npy", "").replace("gel_", "")

    for idx, trace in enumerate(traces):
        # match CSV entry
        # lane indexing: CSV may use 1-based lanes, traces likely 0-based
        match = df_labels[
            (df_labels["gel"].astype(str) == str(gel_id)) &
            (df_labels["lane"].astype(int) == idx)  # change +1 if CSV is 1-based
        ]

        if len(match) == 0:
            raise ValueError(
                f"No label found in CSV for gel={gel_id}, lane={idx}. "
                "Check if CSV uses 1-based lane numbering."
            )

        label = match["label"].values[0]

        all_traces.append(trace)
        all_labels.append(label)
        file_trace_map.append(f)

print(f"Loaded {len(all_traces)} traces with labels.")


# -------------------------
# ---- Feature extraction
# -------------------------

print("Extracting features...")

X = np.array([modelfncs.extract_features(t) for t in all_traces])
y = np.array(all_labels)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of labels: {len(y)}")

# -------------------------
# Train / validation split
# -------------------------

training_set_size = 0.8 

X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
    X, y, file_trace_map, test_size=training_set_size, random_state=42, stratify=y
)

# -------------------------
# ---- Train Random Forest
# -------------------------

print("Training Random Forest classifier...")

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

print("Training complete.")
print(f"Training accuracy: {clf.score(X_train, y_train):.3f}")

print(f"Validation accuracy: {clf.score(X_val, y_val):.3f}")

# -------------------------
# ---- Unknown Detection Threshold
# -------------------------
# We use maximum class probability as uncertainty measure

print("\nEstimating optimal Unknown detection threshold...")

val_probs = clf.predict_proba(X_val)
val_maxp = val_probs.max(axis=1)

# Since we don't have true "unknowns", we approximate:
# threshold = 5th percentile of known validation confidence
UNKNOWN_THRESHOLD = np.percentile(val_maxp, 5)

print(f"Auto-selected UNKNOWN_THRESHOLD = {UNKNOWN_THRESHOLD:.3f}")

# -------------------------
# Predict all
# -------------------------

print("\nRunning predictions on all traces...")

all_probs = clf.predict_proba(X)
all_maxp = all_probs.max(axis=1)
all_preds_raw = clf.predict(X)

# apply unknown threshold
final_preds = []

for raw_label, maxp in zip(all_preds_raw, all_maxp):
    if maxp < UNKNOWN_THRESHOLD:
        final_preds.append("Unknown")
    else:
        final_preds.append(raw_label)

# ---------------------------------------------
# 1. Feature Importance Plot
# ---------------------------------------------

print("\nPlotting feature importance...")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 10))
plt.title("Random Forest Feature Importances")
plt.barh(range(20), importances[indices][:20])
plt.yticks(range(20), [f"F{idx}" for idx in indices[:20]])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_predictions, "feature_importance.png"), dpi=300)
plt.close()

print("Saved feature_importance.png")

# ---------------------------------------------
# 2. Confusion Matrix (Known labels only)
# ---------------------------------------------

print("Generating confusion matrix...")

mask_known = np.array([not u for u in is_unknown])
y_true_known = y[mask_known]
y_pred_known = np.array(final_pred)[mask_known]

unique_labels = sorted(list(set(y)))
cm = confusion_matrix(y_true_known, y_pred_known, labels=unique_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (excluding Unknown)")
plt.tight_layout()
plt.savefig(os.path.join(output_predictions, "confusion_matrix.png"), dpi=300)
plt.close()

print("Saved confusion_matrix.png")

# ---------------------------------------------
# 3. Output Spreadsheet with Predictions
# ---------------------------------------------

print("\nSaving predictions.csv...")

pred_df = pd.DataFrame({
    "gel": [g for g, l in lane_map],
    "lane": [l for g, l in lane_map],
    "true_label": y,
    "predicted_label": final_pred,
    "max_probability": maxp_all,
    "is_unknown": is_unknown
})

pred_csv_path = os.path.join(output_predictions, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)

print("Saved predictions.csv")

# ---------------------------------------------
# 4. Save per-file .npy predictions (same as before)
# ---------------------------------------------

preds_per_file = defaultdict(list)
for p, prob, fname in zip(final_pred, probs_all, file_trace_map):
    preds_per_file[fname].append((p, prob))

for fname, preds in preds_per_file.items():
    out_name = fname.replace("_traces_thresh.npy", "_rf_predictions.npy")
    out_path = os.path.join(output_predictions, out_name)
    np.save(out_path, np.array(preds, dtype=object))
    print("Saved:", out_path)

print("\nAll tasks completed successfully.")


