import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
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

# --- 2. Inputs and output ---
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
    'x_zero_y_pos_z_zero_stresses_full_yz', 'x_zero_y_zero_z_pos_stresses_full_yz'
]

target = 'is_aftershock'

# --- 3. Drop NAN ---
train_df = train_df.dropna(subset=base_features + pca_features + [target])
test_df = test_df.dropna(subset=base_features + pca_features + [target])

# --- 4. Normalization + PCA ---
all_features = base_features + pca_features
scaler = StandardScaler()
X_train_all = scaler.fit_transform(train_df[all_features].values)
X_test_all = scaler.transform(test_df[all_features].values)

# PCA on pca_features 
pca_indices = [all_features.index(f) for f in pca_features]
X_train_pca_input = X_train_all[:, pca_indices]
X_test_pca_input = X_test_all[:, pca_indices]

pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train_pca_input)
X_test_pca = pca.transform(X_test_pca_input)

# base_features normalize 
base_indices = [all_features.index(f) for f in base_features]
X_train_base = X_train_all[:, base_indices]
X_test_base = X_test_all[:, base_indices]

# Final input = base + PCA features
X_train = np.hstack([X_train_base, X_train_pca])
X_test = np.hstack([X_test_base, X_test_pca])
y_train = train_df[target].values
y_test = test_df[target].values

# --- 5. PyTorch Dataset ---
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 6. Model ---
class AftershockClassifier(nn.Module):
    def __init__(self):
        super(AftershockClassifier, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Sigmoid yok burada çünkü loss BCEWithLogits
        return x.squeeze()

# --- 7. Train ---
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = AftershockClassifier()

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 8. Traşn loop ---
epochs = 200
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 9. Metrics ---
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    logits = model(test_X_tensor)
    y_pred_probs = torch.sigmoid(logits).numpy()
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {auc_score:.4f}")
print(datetime.now().strftime("%d-%m-%Y %H:%M"))

# --- 10. ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# --- 11. Confusion Matrix Heatmap ---
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



"""

Confusion Matrix:
 [[744478 365965]
 [  4615  18902]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.67      0.80   1110443
           1       0.05      0.80      0.09     23517

    accuracy                           0.67   1133960
   macro avg       0.52      0.74      0.45   1133960
weighted avg       0.97      0.67      0.79   1133960


ROC AUC Score: 0.7995

"""
