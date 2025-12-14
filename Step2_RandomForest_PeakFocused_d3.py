%reset -sf

import os
import numpy as np
import importlib
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load your feature extraction functions
import randomForest_PeakFocused_d2 as modelfncs
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

tmp_image_direct= os.path.join(working_direct, "tmp_output_pngs/")

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
df_filtered["image"] = df_filtered["image"].astype(int)
df_filtered["lane"] = df_filtered["lane"].astype(int)
# ensure order of dataframe
df_filtered = df_filtered[["image", "lane", "label"]]

df_filtered["label"].unique()
print(df_filtered['label'].value_counts())

classes_vect = ["M","MM1", "MZ","MS"]
df_M_unknown = df_filtered.copy()
df_M_unknown["label"] = df_filtered["label"].where(df_filtered["label"].isin(classes_vect), "Other")

# Optional: check counts
print(df_M_unknown['label'].value_counts())




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

# Count number of lanes per image/gel
lane_counts = trace_df.groupby("image")["lane"].count()
# Show images with exactly 19 lanes
images_19_lanes = lane_counts[lane_counts == 19]
print("These images have an extra lane called due to cropping issue:")
print(images_19_lanes)

# -------------------------
# Remove specific lanes or remove all lanes 1 and 19 from any lane that has > 18 lanes
# -------------------------

miscalled_lanes = [
    (25337, 19),
    (25340, 19),
    (25341, 19),
    (25343, 1),
    (25346, 19),
]

# Keep track of which images had lanes removed
images_to_renumber = set(img for img, lane in miscalled_lanes)

trace_df = trace_df[
    ~trace_df[["image", "lane"]].apply(tuple, axis=1).isin(miscalled_lanes)
]

# Only re-number lanes for affected images
def renumber_lanes(group):
    if group.name in images_to_renumber:
        group["lane"] = range(1, len(group) + 1)
    return group

trace_df = trace_df.groupby("image", group_keys=False).apply(renumber_lanes)

# trace_df = trace_df[~(
#     (trace_df["image"].isin(images_19_lanes)) &
#     (trace_df["lane"] == 19)
# )]

print(df_M_unknown.columns)


# -------------------------
# ---- Merge with labels on gel AND lane
# -------------------------
# Merge with M / All labels
# trace_df = trace_df.merge(df_filtered, on=["image", "lane"], how="left")
# Merge with M / Unknown
trace_df = trace_df.merge(df_M_unknown, on=["image", "lane"], how="left")

if trace_df["label"].isnull().any():
    raise ValueError("Some traces do not have matching labels in CSV!")

# I noticed that some gels the calling is for 19 total lanes (in an 18 well gell) which are just cause of a bad crop
# Wondering if these should be removed for training the model??? 
# Maybe discuss with Dr. Vb, could also want "bad" lanes for comparison with the model

missing_labels = trace_df[trace_df["label"].isnull()]
print("Traces without labels:")
print(missing_labels[["image", "lane", "file"]])


# -------------------------
# ---- Feature extraction
# -------------------------

# Extract lists
all_traces = trace_df["trace"].tolist()
all_labels = trace_df["label"].tolist()
file_trace_map = trace_df["file"].tolist()

print(f"Loaded {len(all_traces)} traces with labels.")

print("Extracting features...")

NumPeaksConsidered_K = 6
min_peak_height=0.2 # after normalizing from 0-1
min_peak_distance=10 # in pixels (of 600)

X = np.array([modelfncs.extract_features(t, NumPeaksConsidered_K, min_peak_height, min_peak_distance) for t in all_traces])
y = np.array(all_labels)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of labels: {len(y)}")

# -------------------------
# Train / validation split
# -------------------------

training_set_size = 0.8 

# X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
#     X, y, file_trace_map, test_size=training_set_size, random_state=42, stratify=y
# )

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


# X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
#     X, y, file_trace_map, test_size=training_set_size, random_state=42, stratify=y
# )


# Encode labels as integers (M=1, Other=0) -----> this will allow for ROC and other stats for the model
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
    X, y_encoded, file_trace_map,
    test_size=training_set_size,
    random_state=42,
    stratify=y_encoded
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


train_sizes, train_scores, val_scores = learning_curve(
    clf, X, y_encoded, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)

# Save the figure
tmp_image = os.path.join(tmp_image_direct, "learning_curve_PeakFocused_moreClsses.png")
plt.savefig(tmp_image, dpi=300)
print(f"Learning curve saved to: {tmp_image}")


# -------------------------
# ---- Feature Importance
# -------------------------

# Example lane
example_lane = all_traces[0]
trace_len = len(example_lane)
print("Trace length:", trace_len)

# Extract features for this lane
features = modelfncs.extract_features(example_lane, K=6)

# Extract indices in feature vector
num_peaks = int(features[0])
top_k_heights = features[2:8]           # K=6
top_k_positions = features[8:14]        # K=6

# Random Forest feature importances
feat_importances = clf.feature_importances_

# Map feature importances to peaks only
# Feature vector: [num_peaks, mean_peak_height, top_k_heights (6), top_k_positions (6)]
peak_indices = np.arange(2, 2 + 6)       # indices corresponding to top_k_heights
peak_importances = feat_importances[peak_indices]

# Map normalized peak positions to actual pixels
peak_pixels = (top_k_positions * trace_len).astype(int)

# Create figure with twin y-axis
fig, ax1 = plt.subplots(figsize=(6,8))

# Left y-axis: Feature importance
ax1.bar(peak_pixels, peak_importances, width=3, color='red', alpha=0.7, label='Peak feature importance')
ax1.set_ylabel("Feature importance", color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(0, max(peak_importances)*1.2)

# Right y-axis: Lane intensity
ax2 = ax1.twinx()
ax2.plot(example_lane, color='blue', alpha=0.3, label='Lane intensity')
ax2.set_ylabel("Lane intensity", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# X-axis
ax1.set_xlabel("Pixel along lane (vertical)")
plt.title("Gel lane with peak-based feature importance overlay")

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Save and show
feature_imp_file = os.path.join(tmp_image_direct, "feature_importance_PeakFocused_4cls.png")
plt.savefig(feature_imp_file, dpi=300)
plt.show()
print(f"Feature importance plot saved to: {feature_imp_file}")




# ------------------------------------------
# ------------------------------------------
# Now evaluate model:
# ------------------------------------------
# ------------------------------------------

# ---------------------------------------------------------------------------------
# -------Classification Matrix -> Precision/Recal/F1 Score/Support --------------
# ---------------------------------------------------------------------------------
y_val_pred = clf.predict(X_val)
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# Confusion Matrix --------
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
ConfusionMatrix_file = os.path.join(tmp_image_direct, "ConfusionMatrix_PeakFocused_4cls.png")
plt.savefig(ConfusionMatrix_file, dpi=300)
plt.show()
