import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# === 1. READ DATA ===
train_dir = '/Users/PC/OneDrive/Masaüstü/traindata/'
test_dir = '/Users/PC/OneDrive/Masaüstü/testdata/'

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

# === 2. FEATURES ===
features = [
    'stresses_full_xx', 'stresses_full_yy', 'stresses_full_zz',
    'stresses_full_xy', 'stresses_full_yz', 'stresses_full_xz', 'euclidean_distance',
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
    "x_zero_y_pos_z_neg_stresses_full_yy", "x_zero_y_pos_z_neg_stresses_full_yz", "x_zero_y_pos_z_neg_stresses_full_zz"
]



target = 'grid_aftershock_count'

# === 3. DROPNA ===
train_df = train_df.dropna(subset=features + [target])
test_df = test_df.dropna(subset=features + [target])

# === 4. UNDERSAMPLING ===
df_0 = train_df[train_df[target] == 0]
df_1 = train_df[train_df[target] == 1]

undersample_0 = df_0.sample(n=len(df_1) * 2, random_state=42)  # 1 sınıfının 2 katı kadar 0 sınıfı
balanced_train_df = pd.concat([undersample_0, df_1]).sample(frac=1, random_state=42)


# === 5. PREPARE X AND Y ===
X_train = balanced_train_df[features].values
y_train = balanced_train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

# === 6. STANDARDIZATION ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 7. PYTORCH DATASET ===
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === 8. MODEL CLASS ===
class AftershockRegressor(nn.Module):
    def __init__(self):
        super(AftershockRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(len(features), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# === 9. TRAIN PREPERATION ===
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = AftershockRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# === 10. TRAIN LOOP ===
best_rmse = float('inf')
epochs = 300
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
        val_rmse = np.sqrt(mean_squared_error(y_test, val_preds))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), 'best_regression_model.pth')
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, RMSE: {val_rmse:.4f}")

# === 11. LOAD BEST MODEL ===
model.load_state_dict(torch.load('best_regression_model.pth'))
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(test_X_tensor).squeeze().numpy()

# === 12. METRICS ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(datetime.now().strftime("%d-%m-%Y %H:%M"))

# === 13. GRAPHICS ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Aftershock Count')
plt.grid(True)
plt.show()

"""

MAE: 0.2761
RMSE: 0.4054
R² Score: -0.9263
"""
