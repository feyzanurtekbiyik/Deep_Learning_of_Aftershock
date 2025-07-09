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
from datetime import datetime

# --- 1. Verileri oku ---
train_dir = "C:/Users/PC/OneDrive/Masaüstü/traindata"
test_dir = "C:/Users/PC/OneDrive/Masaüstü/testdata"

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

# --- 2. Özellikler ve hedef değişken ---
features = [
    'stresses_full_xx', 'stresses_full_yy', 'stresses_full_zz',
    'stresses_full_xy', 'stresses_full_yz', 'stresses_full_xz', 'euclidean_distance'
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


# --- 3. Dataset sınıfı ---
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 4. Model sınıfı ---
class AftershockClassifier(nn.Module):
    def __init__(self):
        super(AftershockClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )



    def forward(self, x):
        return self.model(x)


# --- 5. Train ---
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = AftershockClassifier()

# Positive train set (inverse frequency)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

# Weighted BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 6. Training loop---
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
    sigmoid = torch.nn.Sigmoid()
    y_pred_probs = sigmoid(torch.tensor(logits)).numpy()
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

# --- 8. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:\n", cm)

# --- 9. Classification Report (Precision, Recall, F1) ---
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# --- 10. ROC AUC ---
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {auc_score:.4f}")


simdi=datetime.now()
print(simdi.strftime("%d-%m-%Y %H:%M"))

# --- 11. ROC Eğrisi ---
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
 [[799730 310713]
 [  6032  17485]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.72      0.83   1110443
           1       0.05      0.74      0.10     23517

    accuracy                           0.72   1133960
   macro avg       0.52      0.73      0.47   1133960
weighted avg       0.97      0.72      0.82   1133960


ROC AUC Score: 0.7962
"""
