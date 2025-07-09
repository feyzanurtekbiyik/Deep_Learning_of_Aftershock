import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# --- 1. Read data ---
train_dir = "C:/Users/PC/OneDrive/Masaüstü/traindata"
test_dir = "C:/Users/PC/OneDrive/Masaüstü/testdata"

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)


# --- 2. Features and target ---
base_features = [
    'stresses_full_xx', 'stresses_full_yy', 'stresses_full_zz',
    'stresses_full_xy', 'stresses_full_yz', 'stresses_full_xz', 'euclidean_distance'
]

pca_features = [
    'x_neg_y_zero_z_zero_stresses_full_xx', 'x_zero_y_neg_z_zero_stresses_full_xx',
    'x_zero_y_zero_z_neg_stresses_full_xx', 'x_pos_y_zero_z_zero_stresses_full_xx',
    'x_zero_y_pos_z_zero_stresses_full_xx', 'x_zero_y_zero_z_pos_stresses_full_xx',
    'x_neg_y_zero_z_zero_stresses_full_yy', 'x_zero_y_neg_z_zero_stresses_full_yy',
    'x_zero_y_zero_z_neg_stresses_full_yy', 'x_pos_y_zero_z_zero_stresses_full_yy',
    'x_zero_y_pos_z_zero_stresses_full_yy', 'x_zero_y_zero_z_pos_stresses_full_yy',
    'x_neg_y_zero_z_zero_stresses_full_zz', 'x_zero_y_neg_z_zero_stresses_full_zz',
    'x_zero_y_zero_z_neg_stresses_full_zz', 'x_pos_y_zero_z_zero_stresses_full_zz',
    'x_zero_y_pos_z_zero_stresses_full_zz', 'x_zero_y_zero_z_pos_stresses_full_zz',
    'x_neg_y_zero_z_zero_stresses_full_xy', 'x_zero_y_neg_z_zero_stresses_full_xy',
    'x_zero_y_zero_z_neg_stresses_full_xy', 'x_pos_y_zero_z_zero_stresses_full_xy',
    'x_zero_y_pos_z_zero_stresses_full_xy', 'x_zero_y_zero_z_pos_stresses_full_xy',
    'x_neg_y_zero_z_zero_stresses_full_xz', 'x_zero_y_neg_z_zero_stresses_full_xz',
    'x_zero_y_zero_z_neg_stresses_full_xz', 'x_pos_y_zero_z_zero_stresses_full_xz',
    'x_zero_y_pos_z_zero_stresses_full_xz', 'x_zero_y_zero_z_pos_stresses_full_xz',
    'x_neg_y_zero_z_zero_stresses_full_yz', 'x_zero_y_neg_z_zero_stresses_full_yz',
    'x_zero_y_zero_z_neg_stresses_full_yz', 'x_pos_y_zero_z_zero_stresses_full_yz',
    'x_zero_y_pos_z_zero_stresses_full_yz', 'x_zero_y_zero_z_pos_stresses_full_yz',
    "x_neg_y_neg_z_zero_stresses_full_xx", "x_neg_y_neg_z_zero_stresses_full_xy", "x_neg_y_neg_z_zero_stresses_full_xz",
    "x_neg_y_neg_z_zero_stresses_full_yy", "x_neg_y_neg_z_zero_stresses_full_yz", "x_neg_y_neg_z_zero_stresses_full_zz",
    "x_neg_y_zero_z_neg_stresses_full_xx", "x_neg_y_zero_z_neg_stresses_full_xy", "x_neg_y_zero_z_neg_stresses_full_xz",
    "x_neg_y_zero_z_neg_stresses_full_yy", "x_neg_y_zero_z_neg_stresses_full_yz", "x_neg_y_zero_z_neg_stresses_full_zz",
    "x_zero_y_neg_z_neg_stresses_full_xx", "x_zero_y_neg_z_neg_stresses_full_xy", "x_zero_y_neg_z_neg_stresses_full_xz",
    "x_zero_y_neg_z_neg_stresses_full_yy", "x_zero_y_neg_z_neg_stresses_full_yz", "x_zero_y_neg_z_neg_stresses_full_zz",
    "x_pos_y_zero_z_pos_stresses_full_xx", "x_pos_y_zero_z_pos_stresses_full_xy", "x_pos_y_zero_z_pos_stresses_full_xz",
    "x_pos_y_zero_z_pos_stresses_full_yy", "x_pos_y_zero_z_pos_stresses_full_yz", "x_pos_y_zero_z_pos_stresses_full_zz",
    "x_pos_y_pos_z_zero_stresses_full_xx", "x_pos_y_pos_z_zero_stresses_full_xy", "x_pos_y_pos_z_zero_stresses_full_xz",
    "x_pos_y_pos_z_zero_stresses_full_yy", "x_pos_y_pos_z_zero_stresses_full_yz", "x_pos_y_pos_z_zero_stresses_full_zz",
    "x_zero_y_pos_z_pos_stresses_full_xx", "x_zero_y_pos_z_pos_stresses_full_xy", "x_zero_y_pos_z_pos_stresses_full_xz",
    "x_zero_y_pos_z_pos_stresses_full_yy", "x_zero_y_pos_z_pos_stresses_full_yz", "x_zero_y_pos_z_pos_stresses_full_zz",
    "x_neg_y_pos_z_zero_stresses_full_xx", "x_neg_y_pos_z_zero_stresses_full_xy", "x_neg_y_pos_z_zero_stresses_full_xz",
    "x_neg_y_pos_z_zero_stresses_full_yy", "x_neg_y_pos_z_zero_stresses_full_yz", "x_neg_y_pos_z_zero_stresses_full_zz",
    "x_pos_y_neg_z_zero_stresses_full_xx", "x_pos_y_neg_z_zero_stresses_full_xy", "x_pos_y_neg_z_zero_stresses_full_xz",
    "x_pos_y_neg_z_zero_stresses_full_yy", "x_pos_y_neg_z_zero_stresses_full_yz", "x_pos_y_neg_z_zero_stresses_full_zz",
    "x_neg_y_zero_z_pos_stresses_full_xx", "x_neg_y_zero_z_pos_stresses_full_xy", "x_neg_y_zero_z_pos_stresses_full_xz",
    "x_neg_y_zero_z_pos_stresses_full_yy", "x_neg_y_zero_z_pos_stresses_full_yz", "x_neg_y_zero_z_pos_stresses_full_zz",
    "x_pos_y_zero_z_neg_stresses_full_xx", "x_pos_y_zero_z_neg_stresses_full_xy", "x_pos_y_zero_z_neg_stresses_full_xz",
    "x_pos_y_zero_z_neg_stresses_full_yy", "x_pos_y_zero_z_neg_stresses_full_yz", "x_pos_y_zero_z_neg_stresses_full_zz",
    "x_zero_y_neg_z_pos_stresses_full_xx", "x_zero_y_neg_z_pos_stresses_full_xy", "x_zero_y_neg_z_pos_stresses_full_xz",
    "x_zero_y_neg_z_pos_stresses_full_yy", "x_zero_y_neg_z_pos_stresses_full_yz", "x_zero_y_neg_z_pos_stresses_full_zz",
    "x_zero_y_pos_z_neg_stresses_full_xx", "x_zero_y_pos_z_neg_stresses_full_xy", "x_zero_y_pos_z_neg_stresses_full_xz",
    "x_zero_y_pos_z_neg_stresses_full_yy", "x_zero_y_pos_z_neg_stresses_full_yz", "x_zero_y_pos_z_neg_stresses_full_zz",
    "x_neg_y_neg_z_neg_stresses_full_xx", "x_neg_y_neg_z_neg_stresses_full_xy", "x_neg_y_neg_z_neg_stresses_full_xz",
    "x_neg_y_neg_z_neg_stresses_full_yy", "x_neg_y_neg_z_neg_stresses_full_yz", "x_neg_y_neg_z_neg_stresses_full_zz",
    "x_pos_y_pos_z_pos_stresses_full_xx", "x_pos_y_pos_z_pos_stresses_full_xy", "x_pos_y_pos_z_pos_stresses_full_xz",
    "x_pos_y_pos_z_pos_stresses_full_yy", "x_pos_y_pos_z_pos_stresses_full_yz", "x_pos_y_pos_z_pos_stresses_full_zz",
    "x_neg_y_neg_z_pos_stresses_full_xx", "x_neg_y_neg_z_pos_stresses_full_xy", "x_neg_y_neg_z_pos_stresses_full_xz",
    "x_neg_y_neg_z_pos_stresses_full_yy", "x_neg_y_neg_z_pos_stresses_full_yz", "x_neg_y_neg_z_pos_stresses_full_zz",
    "x_neg_y_pos_z_pos_stresses_full_xx", "x_neg_y_pos_z_pos_stresses_full_xy", "x_neg_y_pos_z_pos_stresses_full_xz",
    "x_neg_y_pos_z_pos_stresses_full_yy", "x_neg_y_pos_z_pos_stresses_full_yz", "x_neg_y_pos_z_pos_stresses_full_zz",
    "x_pos_y_neg_z_neg_stresses_full_xx", "x_pos_y_neg_z_neg_stresses_full_xy", "x_pos_y_neg_z_neg_stresses_full_xz",
    "x_pos_y_neg_z_neg_stresses_full_yy", "x_pos_y_neg_z_neg_stresses_full_yz", "x_pos_y_neg_z_neg_stresses_full_zz",
    "x_neg_y_pos_z_neg_stresses_full_xx", "x_neg_y_pos_z_neg_stresses_full_xy", "x_neg_y_pos_z_neg_stresses_full_xz",
    "x_neg_y_pos_z_neg_stresses_full_yy", "x_neg_y_pos_z_neg_stresses_full_yz", "x_neg_y_pos_z_neg_stresses_full_zz",
    "x_pos_y_neg_z_pos_stresses_full_xx", "x_pos_y_neg_z_pos_stresses_full_xy", "x_pos_y_neg_z_pos_stresses_full_xz",
    "x_pos_y_neg_z_pos_stresses_full_yy", "x_pos_y_neg_z_pos_stresses_full_yz", "x_pos_y_neg_z_pos_stresses_full_zz",
    "x_pos_y_pos_z_neg_stresses_full_xx", "x_pos_y_pos_z_neg_stresses_full_xy", "x_pos_y_pos_z_neg_stresses_full_xz",
    "x_pos_y_pos_z_neg_stresses_full_yy", "x_pos_y_pos_z_neg_stresses_full_yz", "x_pos_y_pos_z_neg_stresses_full_zz"
]

target = 'grid_aftershock_count'

# --- 3. Drop NaN Values ---
train_df = train_df.dropna(subset=base_features + pca_features + [target])
test_df = test_df.dropna(subset=base_features + pca_features + [target])

# --- 4. Normalization ---
all_features = base_features + pca_features
scaler = StandardScaler()
X_train_all = scaler.fit_transform(train_df[all_features].values)
X_test_all = scaler.transform(test_df[all_features].values)

# --- 5. PCA on pca_features---
pca_indices = [all_features.index(f) for f in pca_features]
X_train_pca_input = X_train_all[:, pca_indices]
X_test_pca_input = X_test_all[:, pca_indices]

pca = PCA(n_components=110)
X_train_pca = pca.fit_transform(X_train_pca_input)
X_test_pca = pca.transform(X_test_pca_input)

# --- 6. Final input (base + pca) ---
base_indices = [all_features.index(f) for f in base_features]
X_train_base = X_train_all[:, base_indices]
X_test_base = X_test_all[:, base_indices]

X_train = np.hstack([X_train_base, X_train_pca])
X_test = np.hstack([X_test_base, X_test_pca])
y_train = train_df[target].values
y_test = test_df[target].values

# --- 7. Dataset ---
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 8. Model (Regression) ---
class AftershockRegressor(nn.Module):
    def __init__(self):
        super(AftershockRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(117, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x).squeeze()

# --- 9. Train Preparation ---
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = AftershockRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 10. Train Loop ---
epochs = 300
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 11.Estimate and metrics ---
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(test_X_tensor).numpy()

# --- 12. Regression Metrics ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# --- 13. Forecasting vs Real Values Visualization ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Real Count")
plt.ylabel("Estimated Count")
plt.title("Aftershock Count Estimate")
plt.grid(True)
plt.tight_layout()
plt.show()


"""
MAE: 0.0304
RMSE: 0.2937
R² Score: -0.0108

"""
