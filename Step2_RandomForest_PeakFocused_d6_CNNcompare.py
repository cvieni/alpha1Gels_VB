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

# for CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


# Load your feature extraction functions
import randomForest_PeakFocused_d2 as modelfncs
importlib.reload(modelfncs)

# python -m pip install <package>


# -------------------------
# ---- Define Directories
# -------------------------

# Home Computer ---------
projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
working_direct = projectdirect

# Work Laptop ---------
# projectdirect = "C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/"
# working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/")
# C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/Wilrich_alpha1_Isofocus_machinelearning/gel_scans


img_direct = os.path.join(working_direct, "gel_scans/")
output_direct = os.path.join(working_direct, "output_pngs/")
output_traces = os.path.join(working_direct, "output_traces/")
output_predictions = os.path.join(projectdirect, "output_predictions/")
os.makedirs(output_predictions, exist_ok=True)
# label_csv_path = os.path.join(img_direct, "Results_gels_25331_25350.csv") 
label_csv_path = os.path.join(img_direct, "Results_gels_25331_25350_fixRepeat.csv") 

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
print("Print label counts before dropping repeat")
print(df_filtered['label'].value_counts())
print("")

# Drop "repeat" labels BEFORE classification
df_filtered = df_filtered[df_filtered["label"].str.lower() != "repeat"]
# print("filtered", df_filtered.head())




classes_vect = ["M","MM1", "MZ","MS"]
df_M_unknown = df_filtered.copy()
df_M_unknown["label"] = df_filtered["label"].where(df_filtered["label"].isin(classes_vect), "Other")

print("")
print("Print label counts after dropping repeat")
print("\nClass distribution:")
for label, count in df_filtered["label"].value_counts().items():
    print(f"{label}: {count}")




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
# --> too strict led to crashing ----
# if trace_df["label"].isnull().any():
#     raise ValueError("Some traces do not have matching labels in CSV!")
n_missing = trace_df["label"].isnull().sum()
if n_missing > 0:
    print(f"Unaligned: Dropping {n_missing} traces with no label (e.g., repeats)")
trace_df = trace_df.dropna(subset=["label"]).reset_index(drop=True)


# ---- aligned traces ------
trace_df_aligned = trace_df_aligned.merge(df_M_unknown, on=["image", "lane"], how="left")
# if trace_df_aligned["label"].isnull().any():
#     raise ValueError("Some traces do not have matching labels in CSV!")
n_missing_align = trace_df_aligned["label"].isnull().sum()
if n_missing_align > 0:
    print(f"Aligned: Dropping {n_missing_align} traces with no label (e.g., repeats)")
trace_df_aligned = trace_df_aligned.dropna(subset=["label"]).reset_index(drop=True)


# I noticed that some gels the calling is for 19 total lanes (in an 18 well gell) which are just cause of a bad crop
# Wondering if these should be removed for training the model??? 
# Maybe discuss with Dr. Vb, could also want "bad" lanes for comparison with the model

missing_labels = trace_df[trace_df["label"].isnull()]
print("Traces without labels:")
print(missing_labels[["image", "lane", "file"]])


# -------------------------
# ---- Add a marker so we can distinguish aligned vs unaligned
# -------------------------

# trace_df["type"] = "unaligned"
# trace_df_aligned["type"] = "aligned"
# all_traces_df = pd.concat([trace_df, trace_df_aligned], ignore_index=True)

# We only want the aligned lanes
all_traces_df = trace_df_aligned.copy()
all_traces_df["type"] = "aligned"




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

training_set_size = 0.7 

# Split lane IDs into train / validation
train_lanes, val_lanes = train_test_split(
    unique_lanes,
    train_size=training_set_size,
    random_state=42,
    shuffle=True
)

print("Training set length:", len(train_lanes))
print("Validation set length:", len(val_lanes))

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
# ---- Prepare data for 1D CNN
# -------------------------

def normalize_trace(t):
    t = np.asarray(t)
    return (t - t.mean()) / (t.std() + 1e-8)

# Use RAW traces (not extracted features)
X_train_cnn = np.array([normalize_trace(t) for t in train_df["trace"]])
X_val_cnn   = np.array([normalize_trace(t) for t in val_df["trace"]])

# Add channel dimension: (samples, length, channels)
X_train_cnn = X_train_cnn[..., np.newaxis]
X_val_cnn   = X_val_cnn[..., np.newaxis]

# One-hot encode labels
y_train_cnn = to_categorical(y_train_encoded)
y_val_cnn   = to_categorical(y_val_encoded)

print("CNN training data shape:", X_train_cnn.shape)
print("CNN validation data shape:", X_val_cnn.shape)



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

print("Random Forest Training complete.")
print(f"Random Forest Training accuracy: {clf.score(X_train, y_train):.3f}")
print(f"Random Forest Validation accuracy: {clf.score(X_val, y_val):.3f}")


# -------------------------
# ---- Train 1D CNN
# -------------------------

print("Training 1D CNN classifier...")
epoch_num = 25

# EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_loss',      # stop based on validation loss
    patience=5,              # wait this many epochs without improvement
    restore_best_weights=True # use the best weights from training
)

cnn = Sequential([
    Conv1D(32, kernel_size=5, activation='relu',
           input_shape=X_train_cnn.shape[1:]),
    MaxPooling1D(2),

    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(le.classes_), activation='softmax')
])

cnn.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = cnn.fit(
    X_train_cnn, y_train_cnn,
    validation_data=(X_val_cnn, y_val_cnn),
    epochs=epoch_num,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]

)

cnn_train_acc = history.history["accuracy"][-1]
cnn_val_acc   = history.history["val_accuracy"][-1]

print("CNN training complete.")

# -------------------------
# ---- Model Comparison
# -------------------------

rf_train_acc = clf.score(X_train, y_train)
rf_val_acc   = clf.score(X_val, y_val)

print("\n=== Model Comparison ===")
print(f"Random Forest | Train Acc: {rf_train_acc:.3f} | Val Acc: {rf_val_acc:.3f}")
print(f"1D CNN        | Train Acc: {cnn_train_acc:.3f} | Val Acc: {cnn_val_acc:.3f}")


# -------------------------
# ---- CNN Learning Curve
# -------------------------

print("Computing CNN learning curve...")

train_fracs = np.linspace(0.1, 1.0, 6)

l2_value=0.1

cnn_train_scores = []
cnn_val_scores = []


for frac in train_fracs:
    n_samples = int(len(X_train_cnn) * frac)

    X_sub = X_train_cnn[:n_samples]
    y_sub = y_train_cnn[:n_samples]

    # Reinitialize model each time
    cnn_tmp = Sequential([
        Conv1D(32, kernel_size=5, activation='relu',
               input_shape=X_train_cnn.shape[1:]),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_value)),
        Dropout(0.4),
        Dense(len(le.classes_), activation='softmax')
    ])

    cnn_tmp.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = cnn_tmp.fit(
        X_sub, y_sub,
        validation_data=(X_val_cnn, y_val_cnn),
        epochs=epoch_num,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]   # <--- add the callback here
    )

    cnn_train_scores.append(history.history["accuracy"][-1])
    cnn_val_scores.append(history.history["val_accuracy"][-1])

print("CNN learning curve complete.")





# -------------------------
# ---- Visualizations
# -------------------------

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
# ---- Learning Curves: RF vs CNN
# -------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ---- Random Forest ----
axes[0].plot(train_sizes, train_scores_mean, 'o-', label='Train')
axes[0].plot(train_sizes, val_scores_mean, 'o-', label='Validation')
axes[0].set_title("Random Forest Learning Curve")
axes[0].set_xlabel("Training set size")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# ---- CNN ----
axes[1].plot(train_fracs * len(X_train_cnn), cnn_train_scores, 'o-', label='Train')
axes[1].plot(train_fracs * len(X_train_cnn), cnn_val_scores, 'o-', label='Validation')
axes[1].set_title("1D CNN Learning Curve")
axes[1].set_xlabel("Training set size")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Learning Curves: Random Forest vs 1D CNN")
plt.tight_layout(rect=[0, 0, 1, 0.95])

tmp_image = os.path.join(tmp_image_direct, "learning_curve_RF_vs_CNN.png")
plt.savefig(tmp_image, dpi=300)
print(f"Learning curve comparison saved to: {tmp_image}")


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
print("")
print("Table of Random Forest Model Stats")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# Confusion Matrix --------
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix Random Forest Model")
ConfusionMatrix_file = os.path.join(tmp_image_direct, "ConfusionMatrix_RF_4cls.png")
plt.savefig(ConfusionMatrix_file, dpi=300)
# plt.show()

# -------------------------
# Confusion Matrix for CNN
# -------------------------
y_val_pred_cnn = np.argmax(cnn.predict(X_val_cnn), axis=1)
y_val_true_cnn = np.argmax(y_val_cnn, axis=1)

# Create confusion matrix
cm_cnn = confusion_matrix(y_val_true_cnn, y_val_pred_cnn)
# Display
disp_cnn = ConfusionMatrixDisplay(cm_cnn, display_labels=le.classes_)
disp_cnn.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - 1D CNN")
# Save figure
ConfusionMatrix_CNN_file = os.path.join(tmp_image_direct, "ConfusionMatrix_CNN_4cls.png")
plt.savefig(ConfusionMatrix_CNN_file, dpi=300)
print(f"CNN Confusion Matrix saved to: {ConfusionMatrix_CNN_file}")
# plt.show()  # optional if you want to display




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
# plt.show()



# ----------------
# Save Models
# ----------------
import joblib

# Save
rf_model_path = os.path.join(output_predictions, "rf_model_peakfocused.pkl")
joblib.dump(clf, rf_model_path)
print(f"Random Forest saved to: {rf_model_path}")

# Load later
# clf_loaded = joblib.load(rf_model_path)

# Save the CNN
cnn_model_path = os.path.join(output_predictions, "cnn_1D_lane_model.h5")
cnn.save(cnn_model_path)
print(f"1D CNN saved to: {cnn_model_path}")

# Load later
from tensorflow.keras.models import load_model
# cnn_loaded = load_model(cnn_model_path)

# You can now evaluate
# cnn_loaded.evaluate(X_val_cnn, y_val_cnn)




# -----------------------------
# Random Forest report
# -----------------------------
y_val_pred_rf = clf.predict(X_val)
report_rf = classification_report(
    y_val, y_val_pred_rf, target_names=le.classes_, output_dict=True
)
df_rf = pd.DataFrame(report_rf).transpose()
df_rf["Model"] = "Random Forest"

# -----------------------------
# CNN report
# -----------------------------
# Make predictions
y_val_pred_cnn_probs = cnn.predict(X_val_cnn)          # softmax outputs
y_val_pred_cnn = np.argmax(y_val_pred_cnn_probs, axis=1)
y_val_true_cnn = np.argmax(y_val_cnn, axis=1)

report_cnn = classification_report(
    y_val_true_cnn, y_val_pred_cnn, target_names=le.classes_, output_dict=True
)
df_cnn = pd.DataFrame(report_cnn).transpose()
df_cnn["Model"] = "1D CNN"

# -----------------------------
# Combine into one table
# -----------------------------
# Add a 'Model' column to each report
df_rf["Model"] = "Random Forest"
df_cnn["Model"] = "1D CNN"

# The index of classification_report (e.g., 'M', 'MM1', 'accuracy') is the class
df_rf = df_rf.reset_index().rename(columns={"index": "Class"})
df_cnn = df_cnn.reset_index().rename(columns={"index": "Class"})

# Keep only relevant columns
cols = ["Model", "Class", "precision", "recall", "f1-score", "support"]
df_rf = df_rf[cols]
df_cnn = df_cnn[cols]

# Round numeric columns
numeric_cols = ["precision", "recall", "f1-score", "support"]
df_rf[numeric_cols] = df_rf[numeric_cols].apply(pd.to_numeric, errors='coerce').round(2)
df_cnn[numeric_cols] = df_cnn[numeric_cols].apply(pd.to_numeric, errors='coerce').round(2)

# Combine into one table
df_compare = pd.concat([df_rf, df_cnn], ignore_index=True)

# Show table
print("=== Model Comparison Table ===")
print(df_compare)

# Save to CSV
compare_file = os.path.join(tmp_image_direct, "Model_Comparison_RF_CNN.csv")
df_compare.to_csv(compare_file, index=False)
print(f"Model comparison table saved to: {compare_file}")




print("Rows in val_df:", len(val_df))
print("Unique lanes in val_df:", val_df["lane_id"].nunique())
print(val_df["type"].value_counts())


# True labels
y_val_true = y_val

# RF predictions
y_val_pred_rf = clf.predict(X_val)

# CNN predictions
y_val_pred_cnn_idx = np.argmax(cnn.predict(X_val_cnn), axis=1)
y_val_pred_cnn = le.inverse_transform(y_val_pred_cnn_idx)



y_val_pred_cnn_probs = cnn.predict(X_val_cnn)
y_val_pred_cnn_conf = np.max(y_val_pred_cnn_probs, axis=1)


df_predictions = pd.DataFrame({
    "True_Label": y_val_true,
    "RF_Prediction": y_val_pred_rf,
    "CNN_Prediction": y_val_pred_cnn,
    "CNN_Confidence": y_val_pred_cnn_conf
})


# -----------------------------
# Build prediction dataframe (aligned with val_df)
# -----------------------------

# True labels
y_val_true = val_df["label"].values

# RF predictions
y_val_pred_rf = clf.predict(X_val)

# CNN predictions
y_val_pred_cnn_probs = cnn.predict(X_val_cnn)
y_val_pred_cnn_idx   = np.argmax(y_val_pred_cnn_probs, axis=1)
y_val_pred_cnn       = le.inverse_transform(y_val_pred_cnn_idx)
y_val_pred_cnn_conf  = np.max(y_val_pred_cnn_probs, axis=1)

# Metadata (already aligned!)
df_predictions = pd.DataFrame({
    "image": val_df["image"].values,
    "lane": val_df["lane"].values,
    "True_Label": y_val_true,
    "RF_Prediction": y_val_pred_rf,
    "CNN_Prediction": y_val_pred_cnn,
    "CNN_Confidence": y_val_pred_cnn_conf
})

# Add disagreement flag
df_predictions["Disagree"] = (
    df_predictions["RF_Prediction"] != df_predictions["CNN_Prediction"]
)

# Save
pred_file = os.path.join(output_predictions, "RF_CNN_Predictions.csv")
df_predictions.to_csv(pred_file, index=False)

print(f"Saved predictions to: {pred_file}")
print("\nDisagreements:")
print(df_predictions[df_predictions["Disagree"]].head())


df_predictions["RF_Agree"]  = (
    df_predictions["True_Label"] == df_predictions["RF_Prediction"]
)

df_predictions["CNN_Agree"] = (
    df_predictions["True_Label"] == df_predictions["CNN_Prediction"]
)

rf_summary = (
    df_predictions
    .groupby("True_Label")["RF_Agree"]
    .agg(
        Agree_Count="sum",
        Total="count"
    )
    .reset_index()
)

rf_summary["Disagree_Count"] = rf_summary["Total"] - rf_summary["Agree_Count"]
rf_summary["Accuracy"] = (rf_summary["Agree_Count"] / rf_summary["Total"]).round(2)
rf_summary["Model"] = "Random Forest"


cnn_summary = (
    df_predictions
    .groupby("True_Label")["CNN_Agree"]
    .agg(
        Agree_Count="sum",
        Total="count"
    )
    .reset_index()
)

cnn_summary["Disagree_Count"] = cnn_summary["Total"] - cnn_summary["Agree_Count"]
cnn_summary["Accuracy"] = (cnn_summary["Agree_Count"] / cnn_summary["Total"]).round(2)
cnn_summary["Model"] = "1D CNN"

summary_all = pd.concat([rf_summary, cnn_summary], ignore_index=True)

# Reorder columns nicely
summary_all = summary_all[
    ["Model", "True_Label", "Agree_Count", "Disagree_Count", "Total", "Accuracy"]
]
summary_file = os.path.join(tmp_image_direct, "PerClass_Agreement_RF_CNN.csv")
summary_all.to_csv(summary_file, index=False)
print(f"Saved per-class agreement summary to: {summary_file}")



print("\n=== Validation Set Class Distribution ===")
val_dist = pd.Series(y_val_true).value_counts().sort_index()
print(val_dist)

print("\nPercentages:")
print((val_dist / val_dist.sum() * 100).round(1))


print("")
print("Save df of results")


