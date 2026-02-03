import os
import numpy as np
import importlib
from collections import defaultdict
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load your feature extraction functions
import randomForest_PeakFocused_d2 as modelfncs
importlib.reload(modelfncs)

# python -m pip install <package>


# -------------------------
# ---- Define Directories
# -------------------------

projectdirect = "C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/"
# projectdirect = "/Users/cvieni/Documents/Pathology/alpha1Gels_VB"
# working_direct = projectdirect
working_direct = os.path.join(projectdirect, "Wilrich_alpha1_Isofocus_machinelearning/alpha1Gels_VB/")
# C:/Users/M297055/OneDrive - Mayo Clinic/Documents/Research/Wilrich_alpha1_Isofocus_machinelearning/gel_scans


img_direct = os.path.join(working_direct, "gel_scans/")
output_direct = os.path.join(working_direct, "output_pngs/")
output_traces = os.path.join(working_direct, "output_traces/")
output_predictions = os.path.join(projectdirect, "output_predictions/")
os.makedirs(output_predictions, exist_ok=True)
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


classes_vect = ["M","MM1", "MZ","MS"]
df_M_unknown = df_filtered.copy()
df_M_unknown["label"] = df_filtered["label"].where(df_filtered["label"].isin(classes_vect), "Other")

# Optional: check counts
print(df_M_unknown['label'].value_counts())



print(f"Loaded {len(df_labels)} label entries.")
print(f"Saved a variable called df_filtered with only column titles:", required_cols)


print("")
print("Print label counts after dropping repeat")
print("\nClass distribution:")
for label, count in df_filtered["label"].value_counts().items():
    print(f"{label}: {count}")
