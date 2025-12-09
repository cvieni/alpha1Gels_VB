import os
import numpy as np
import importlib
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Load your feature extraction functions
import randomForest as modelfncs
importlib.reload(modelfncs)

# -------------------------
# ---- Define Directories
# -------------------------

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
label_csv_path = os.path.join(img_direct, "Results_gels_25331_25350.csv") 

# -------------------------
# ---- Load Label CSV
# -------------------------

print("Loading label CSV:", label_csv_path)
df_labels = pd.read_csv(label_csv_path)

# Normalize column names (just in case)
df_labels.columns = [c.lower() for c in df_labels.columns]

required_cols = {"image", "lane", "result"}

# Check if all required columns exist
if not required_cols.issubset(df_labels.columns):
    raise ValueError(f"Error ****** CSV must contain columns: {required_cols} *******")

# Keep only the required columns
df_filtered = df_labels[list(required_cols)]
# Rename column headers --------
df_filtered = df_filtered.rename(columns={"result": "label"})

print(f"Loaded {len(df_labels)} label entries.")
print(f"Saved a variable called df_filtered with only column titles:", required_cols)

# -------------------------
# ---- Load all traces
# -------------------------

print("Loading lane traces...")

trace_files = [
    f for f in os.listdir(output_traces)
    if f.lower().endswith("_traces_thresh.npy")
]

trace_data = []

for f in trace_files:
    gel_id = f.replace("_traces_thresh.npy", "").replace("gel_", "")
    traces = np.load(os.path.join(output_traces, f), allow_pickle=True)
    for idx, trace in enumerate(traces):
        trace_data.append({"image": gel_id, "lane": idx, "trace": trace, "file": f})

trace_df = pd.DataFrame(trace_data)
# Remove file extension from trace_df["image"]

trace_df["image"] = trace_df["image"].str.replace(".jpg","")
trace_df["image"] = trace_df["image"].astype(int)  # convert to int
trace_df["lane"] = trace_df["lane"] + 1


# -------------------------
# ---- Merge with labels on gel AND lane
# -------------------------
trace_df = trace_df.merge(df_filtered, on=["image", "lane"], how="left")

if trace_df["label"].isnull().any():
    raise ValueError("Some traces do not have matching labels in CSV!")

# Extract lists
all_traces = trace_df["trace"].tolist()
all_labels = trace_df["label"].tolist()
file_trace_map = trace_df["file"].tolist()

print(f"Loaded {len(all_traces)} traces with labels.")

# I noticed that some gels the calling is for 19 total lanes (in an 18 well gell) which are just cause of a bad crop
# Wondering if these should be removed for training the model??? 
# Maybe discuss with Dr. Vb, could also want "bad" lanes for comparison with the model

# Count number of lanes per image/gel
lane_counts = trace_df.groupby("image")["lane"].count()
# Show images with exactly 19 lanes
images_19_lanes = lane_counts[lane_counts == 19]
print("These images have an extra lane called due to cropping issue:")
print(images_19_lanes)


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

# remove stratify=y so that the training data is split randomly ------
# X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
#     X, y, file_trace_map, test_size=training_set_size, random_state=42
# )


# ----- For test purposes remove any classes with too low counts -------

# Count examples per class
class_counts = pd.Series(y).value_counts()
print(class_counts[class_counts < 2])

# Keep only classes with >= 2 examples
valid_classes = class_counts[class_counts >= 2].index
mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]
file_trace_map = [f for i, f in enumerate(file_trace_map) if mask[i]]

X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
    X, y, file_trace_map, test_size=training_set_size, random_state=42, stratify=y
)

# -------------------------
# ---- Train Random Forest
# -------------------------

print("Training Random Forest classifier...")

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=5,
    # max_features='sqrt',
    min_samples_leaf=2, # ensure leaf has at least 2 samples
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

print("Training complete.")
print(f"Training accuracy: {clf.score(X_train, y_train):.3f}")

print(f"Validation accuracy: {clf.score(X_val, y_val):.3f}")


# cross-validation to better estimate generalization
scores = cross_val_score(clf, X, y, cv=5)
print("CV mean accuracy:", scores.mean())

# Grid search to fine tune parameters:

param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None]
}

grid = GridSearchCV(RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
                    param_grid, 
                    cv=5,
                    scoring='accuracy',
                    verbose=3  # 1-5; higher = more info
                    )
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)

# Output from 12/8/25 --------
# >>> print(grid.best_params_, grid.best_score_)
# {'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 1} 0.6620624048706241
best_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    max_features='log2',
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

best_clf.fit(X, y)

# -------------------------
# ---- Unknown Detection Threshold
# -------------------------
# We use maximum class probability as uncertainty measure

print(f"Estimating optimal Unknown detection threshold...")

# returns the probability for each class for every sample in X_val (i.e. if trace is predicited as 0.1-cls1, 0.1-cls2, 0.8-cls3-> model is 80% confident its class 3)
val_probs = best_clf.predict_proba(X_val)
# now just save max probability class for each sample
val_maxp = val_probs.max(axis=1)

# Since we don't have true "unknowns", we approximate:
# threshold = 5th percentile of known validation confidence
UNKNOWN_THRESHOLD = np.percentile(val_maxp, 5)
# ^^^^ Unknown above means an "unknown class". I.e. a band pattern that is not represented in our training data (ex. if class A,B,C,D in training, this would be class E)

print(f"Auto-selected UNKNOWN_THRESHOLD = {UNKNOWN_THRESHOLD:.3f}")

# # -------------------------
# # Predict all
# # -------------------------

# print("\nRunning predictions on all traces...")

# all_probs = best_clf.predict_proba(X)
# all_maxp = all_probs.max(axis=1)
# all_preds_raw = best_clf.predict(X)

# # apply unknown threshold
# final_preds = []

# for raw_label, maxp in zip(all_preds_raw, all_maxp):
#     if maxp < UNKNOWN_THRESHOLD:
#         final_preds.append("Unknown")
#     else:
#         final_preds.append(raw_label)

# -------------------------
# Predict Validation data set
# -------------------------
# Predict class probabilities on validation set
val_probs = best_clf.predict_proba(X_val)  # shape: [num_val_samples, num_classes]

# Get maximum probability per trace (model confidence)
val_maxp = val_probs.max(axis=1)

# Get predicted classes for each validation trace
val_preds_raw = best_clf.predict(X_val)

# Apply unknown threshold if you want to flag low-confidence predictions
final_val_preds = []
for raw_label, maxp in zip(val_preds_raw, val_maxp):
    if maxp < UNKNOWN_THRESHOLD:
        final_val_preds.append("Unknown")
    else:
        final_val_preds.append(raw_label)

# Optionally, check validation accuracy (ignoring Unknowns)
mask_known = [pred != "Unknown" for pred in final_val_preds]
val_accuracy = np.mean(np.array(val_preds_raw)[mask_known] == np.array(y_val)[mask_known])
print(f"Validation accuracy (excluding Unknowns): {val_accuracy:.3f}")

val_raw_accuracy = np.mean(val_preds_raw == y_val)
print(f"Raw validation accuracy (no masking): {val_raw_accuracy:.3f}")


# ---------------------------------------------
# 1. Feature Importance Plot for all features
# ---------------------------------------------

print("\nPlotting feature importance...")

importances = best_clf.feature_importances_
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
# 1. Feature Importance Plot for a single validation sample
# ---------------------------------------------

# Create SHAP explainer for your Random Forest
explainer = shap.TreeExplainer(best_clf)

# Pick a validation sample index
sample_idx = 0
X_sample = X_val[sample_idx].reshape(1, -1)

# Get SHAP values
shap_values = explainer.shap_values(X_sample)

# Predicted class for this sample
pred_class = best_clf.predict(X_sample)[0]
class_idx = list(best_clf.classes_).index(pred_class)

# Plot SHAP values for this sample
shap.initjs()
shap.force_plot(
    explainer.expected_value[class_idx],
    shap_values[class_idx],
    X_sample,
    matplotlib=True
)


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


