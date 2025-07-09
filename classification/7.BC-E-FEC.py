import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Read data ---
train_dir = "C:/Users/PC/OneDrive/Masaüstü/traindata"
test_dir = "C:/Users/PC/OneDrive/Masaüstü/testdata"

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

# --- 2. Features ---
features = [
    "stresses_full_xx", "stresses_full_yy", "stresses_full_zz",
    "stresses_full_xy", "stresses_full_yz", "stresses_full_xz",
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
    "x_pos_y_pos_z_neg_stresses_full_yy", "x_pos_y_pos_z_neg_stresses_full_yz", "x_pos_y_pos_z_neg_stresses_full_zz",'euclidean_distance'
]
target = 'is_aftershock'

train_df = train_df.dropna(subset=features + [target])
test_df = test_df.dropna(subset=features + [target])

X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- 3. Dataset class ---
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 4. Model class ---
class AftershockClassifier(nn.Module):
    def __init__(self):
        super(AftershockClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(163, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# --- 5. Train  ---
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = AftershockClassifier()

# Positive class (inverse frequency)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

# Weighted BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 6. Train loop ---
epochs =200
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        logits = model(batch_X).squeeze()
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 7. Test and metrics ---
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    logits = model(test_X_tensor).squeeze().numpy()
    y_pred_probs = 1 / (1 + np.exp(-logits))  # sigmoid
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

# --- 8. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:\n", cm)

# --- 9. Classification Report (Precision, Recall, F1) ---
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# --- 10. ROC AUC ---
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {auc_score:.4f}")

# --- 11. ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# --- 12. Confusion Matrix Heatmap ---
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""
Confusion Matrix:
 [[742307 368136]
 [  5427  18090]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.67      0.80   1110443
           1       0.05      0.77      0.09     23517

    accuracy                           0.67   1133960
   macro avg       0.52      0.72      0.44   1133960
weighted avg       0.97      0.67      0.78   1133960


ROC AUC Score: 0.7689

"""





