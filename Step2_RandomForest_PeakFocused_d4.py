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

# -------------------------
# ---- Load all UNALIGNED traces
# -------------------------
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


# -------------------------
# ---- Load all ALIGNED traces
# -------------------------
trace_files_aligned = [
    f for f in os.listdir(output_traces)
    if f.lower().endswith("_traces_thresh_aligned.npy")
]

trace_data_aligned = []

for f in trace_files_aligned:
    gel_id = f.replace("_traces_thresh_aligned.npy", "").replace("gel_", "")
    traces = np.load(os.path.join(output_traces, f), allow_pickle=True)
    for idx, trace in enumerate(traces):
        trace_data_aligned.append({
            "image": gel_id,
            "lane": idx + 1,   # keep lane numbering consistent
            "trace": trace,
            "file": f
        })

trace_df_aligned = pd.DataFrame(trace_data_aligned)

# Remove file extensions if present
# ---- unaligned traces ------
trace_df["image"] = trace_df["image"].str.replace(".jpg","")
trace_df["image"] = trace_df["image"].astype(int)  # convert to int
trace_df["lane"] = trace_df["lane"] + 1
# ---- aligned traces ------
trace_df_aligned["image"] = trace_df_aligned["image"].str.replace(".jpg","").str.replace(".png","").str.replace(".tif","")
trace_df_aligned["image"] = trace_df_aligned["image"].astype(int)

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


# ---- unaligned traces ------
trace_df = trace_df[
    ~trace_df[["image", "lane"]].apply(tuple, axis=1).isin(miscalled_lanes)
]
# Only re-number lanes for affected images
def renumber_lanes(group):
    if group.name in images_to_renumber:
        group["lane"] = range(1, len(group) + 1)
    return group

trace_df = trace_df.groupby("image", group_keys=False).apply(renumber_lanes)

# ---- aligned traces ------
trace_df_aligned = trace_df_aligned[
    ~trace_df_aligned[["image", "lane"]].apply(tuple, axis=1).isin(miscalled_lanes)
]
trace_df_aligned = trace_df_aligned.groupby("image", group_keys=False).apply(renumber_lanes)

print(df_M_unknown.columns)


# -------------------------
# ---- Merge with labels on gel AND lane
# -------------------------
# Merge with M / All labels
# trace_df = trace_df.merge(df_filtered, on=["image", "lane"], how="left")
# Merge with M / Unknown

# ---- unaligned traces ------
trace_df = trace_df.merge(df_M_unknown, on=["image", "lane"], how="left")
if trace_df["label"].isnull().any():
    raise ValueError("Some traces do not have matching labels in CSV!")

# ---- aligned traces ------
trace_df_aligned = trace_df_aligned.merge(df_M_unknown, on=["image", "lane"], how="left")
if trace_df_aligned["label"].isnull().any():
    raise ValueError("Some traces do not have matching labels in CSV!")


# I noticed that some gels the calling is for 19 total lanes (in an 18 well gell) which are just cause of a bad crop
# Wondering if these should be removed for training the model??? 
# Maybe discuss with Dr. Vb, could also want "bad" lanes for comparison with the model

missing_labels = trace_df[trace_df["label"].isnull()]
print("Traces without labels:")
print(missing_labels[["image", "lane", "file"]])


# -------------------------
# ---- Add a marker so we can distinguish aligned vs unaligned
# -------------------------
trace_df["type"] = "unaligned"
trace_df_aligned["type"] = "aligned"
all_traces_df = pd.concat([trace_df, trace_df_aligned], ignore_index=True)


# -------------------------
# ---- Feature extraction
# -------------------------
# Create a unique lane identifier
all_traces_df["lane_id"] = all_traces_df["image"].astype(str) + "_" + all_traces_df["lane"].astype(str)

# Get unique lanes
unique_lanes = all_traces_df["lane_id"].unique()

# # Extract lists
# all_traces = trace_df["trace"].tolist()
# all_labels = trace_df["label"].tolist()
# file_trace_map = trace_df["file"].tolist()

print(f"Loaded {len(all_traces_df)} traces with labels.")

print("Extracting features...")

# -------------------------
# Train / validation split
# -------------------------

training_set_size = 0.8 

# Split lane IDs into train / validation
train_lanes, val_lanes = train_test_split(
    unique_lanes,
    test_size=training_set_size,
    random_state=42,
    shuffle=True
)

# Filter traces according to lane split
train_df = all_traces_df[all_traces_df["lane_id"].isin(train_lanes)].copy()
val_df   = all_traces_df[all_traces_df["lane_id"].isin(val_lanes)].copy()

# Now extract features from train/val separately
NumPeaksConsidered_K = 10
# NumPeaksConsidered_K = 6
min_peak_height = 0.2
min_peak_distance = 10

X_train = np.array([
    modelfncs.extract_features(t, NumPeaksConsidered_K, min_peak_height, min_peak_distance)
    for t in train_df["trace"]
])
y_train = train_df["label"].to_numpy()

X_val = np.array([
    modelfncs.extract_features(t, NumPeaksConsidered_K, min_peak_height, min_peak_distance)
    for t in val_df["trace"]
])
y_val = val_df["label"].to_numpy()

# Optional: store which file each trace came from
ftm_train = train_df["file"].tolist()
ftm_val = val_df["file"].tolist()

print(f"Training feature matrix shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation feature matrix shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")



# ----- For test purposes remove any classes with too low counts -------
# # Count examples per class
# class_counts = pd.Series(y).value_counts()
# print(class_counts[class_counts < 2])

# # Keep only classes with >= 2 examples
# valid_classes = class_counts[class_counts >= 2].index
# mask = np.isin(y, valid_classes)

# X = X[mask]
# y = y[mask]
# file_trace_map = [f for i, f in enumerate(file_trace_map) if mask[i]]


# # Encode labels as integers (M=1, Other=0) -----> this will allow for ROC and other stats for the model
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# X_train, X_val, y_train, y_val, ftm_train, ftm_val = train_test_split(
#     X, y_encoded, file_trace_map,
#     test_size=training_set_size,
#     random_state=42,
#     stratify=y_encoded
# )

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)  # use the same encoder

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
    clf, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
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
example_lane = all_traces_df.iloc[0]["trace"]
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
# plt.show()
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
# plt.show()






# ------------------------------------------
# ------------------------------------------
# Plot feature importance overlaid on a gel lane
# ------------------------------------------
# ------------------------------------------

example_lane = all_traces_df.iloc[0]["trace"]
trace_len = len(example_lane)

# Create zero importance vector
importance_vector = np.zeros(trace_len)

# Map top_k_positions * trace_len -> indices
top_k_positions = features[8:14]  # normalized positions
top_k_heights = features[2:8]     # heights
peak_importances = clf.feature_importances_[2:8]  # corresponding importances

# Assign feature importance to a small window around each peak
window = 3  # pixels
for pos, imp in zip((top_k_positions * trace_len).astype(int), peak_importances):
    start = max(pos - window, 0)
    end   = min(pos + window, trace_len)
    importance_vector[start:end] = imp

# Normalize importance for plotting
importance_vector /= importance_vector.max()

# Plot
plt.figure(figsize=(12,4))
plt.plot(example_lane, color='blue', alpha=0.5, label='Lane intensity')
plt.fill_between(range(trace_len), 0, importance_vector * example_lane.max(),
                 color='red', alpha=0.3, label='Feature importance overlay')
plt.xlabel('Pixel along lane')
plt.ylabel('Intensity / Importance')
plt.title('Lane intensity with per-pixel feature importance overlay')
plt.legend()
FeatureOverlay_file = os.path.join(tmp_image_direct, "PerPixel_FeatureImportance.png")
plt.savefig(FeatureOverlay_file, dpi=300)
# plt.show()




# Select the first 15 lanes
num_lanes_to_plot = 15
lanes_to_plot = all_traces_df.iloc[:num_lanes_to_plot]

# Create figure
plt.figure(figsize=(12, 5))

# Loop through lanes
for i, row in lanes_to_plot.iterrows():
    lane = row["trace"]
    trace_len = len(lane)
    
    # Optional: scale alpha by lane index for visibility
    alpha = 0.4
    
    plt.plot(lane, color='blue', alpha=alpha)

# Compute feature importance overlay from first lane (or average across lanes if desired)
example_lane = lanes_to_plot.iloc[0]["trace"]
trace_len = len(example_lane)

# Feature vector from your extraction function
features = modelfncs.extract_features(example_lane, K=6)
top_k_positions = features[8:14]  # normalized positions
peak_importances = clf.feature_importances_[2:8]  # corresponding importances

# Create zero importance vector
importance_vector = np.zeros(trace_len)
window = 3  # pixels around each peak
for pos, imp in zip((top_k_positions * trace_len).astype(int), peak_importances):
    start = max(pos - window, 0)
    end   = min(pos + window, trace_len)
    importance_vector[start:end] = imp

# Normalize importance
importance_vector /= importance_vector.max()

# Overlay importance as shaded area
plt.fill_between(range(trace_len), 0, importance_vector * max([len(row["trace"]) for idx,row in lanes_to_plot.iterrows()]),
                 color='red', alpha=0.3, label='Feature importance overlay')

plt.xlabel('Pixel along lane')
plt.ylabel('Intensity / Importance')
plt.title(f'Overlay of first {num_lanes_to_plot} lanes with feature importance')
plt.legend()
FeatureOverlay_file = os.path.join(tmp_image_direct, "PerPixel_FeatureImportance_15overlay.png")
plt.savefig(FeatureOverlay_file, dpi=300)
plt.show()