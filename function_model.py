# ================================
# function_model.py
# ================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ---------------------
# Helper: Create labels from peaks
# ---------------------
def create_peak_labels(trace, height_ratio=0.3, distance=10):
    """
    Convert 1D density trace to binary labels for peak detection.
    """
    peaks, _ = find_peaks(trace, height=height_ratio*np.max(trace), distance=distance)
    labels = np.zeros_like(trace, dtype=np.float32)
    labels[peaks] = 1.0
    return labels

# ---------------------
# Prepare dataset for CNN
# ---------------------
def prepare_dataset(traces):
    X = []
    y = []

    for trace in traces:
        trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
        X.append(trace_norm[:, np.newaxis])  # shape: [height, 1]
        y.append(create_peak_labels(trace))

    X = np.array(X, dtype=np.float32)       # [num_lanes, height, 1]
    y = np.array(y, dtype=np.float32)       # [num_lanes, height]

    return X, y

# ---------------------
# Define 1D CNN model
# ---------------------
class LanePeakCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch, channel=1, height]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x.squeeze(1)  # [batch, height]

# ---------------------
# Train function
# ---------------------
def train_model(X, y, num_epochs=50, batch_size=4, lr=1e-3, device='cpu'):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LanePeakCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for xb, yb in loader:
            xb = xb.transpose(1, 2).to(device)  # [batch, 1, height]
            yb = yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model

# ---------------------
# Predict peaks
# ---------------------
def predict(model, traces, device='cpu'):
    model.eval()
    X, _ = prepare_dataset(traces)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor.transpose(1, 2)).cpu().numpy()
    return preds

# ---------------------
# Optional: visualize prediction
# ---------------------
def visualize_prediction(trace, pred):
    plt.figure(figsize=(6,6))
    plt.plot(trace, label="Original Density")
    plt.plot(pred*trace.max(), label="CNN Predicted Peaks", color="red")
    plt.legend()
    plt.title("Lane Peak Prediction")
    plt.show()

# ================================
# End of CreateModel.py
# ================================
