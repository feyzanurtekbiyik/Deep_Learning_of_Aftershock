

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

# --- 1. Load Data ---
train_dir = "C:/Users/PC/OneDrive/Masaüstü/traindata"
test_dir = "C:/Users/PC/OneDrive/Masaüstü/testdata"

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

# --- 2. Features and Target ---
features = [
    'stresses_full_xx', 'stresses_full_yy', 'stresses_full_zz',
    'stresses_full_xy', 'stresses_full_yz', 'stresses_full_xz'
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

# --- 3. Dataset Class ---
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 4. Model Class (Manual with Sigmoid) ---
class AftershockClassifier(nn.Module):
    def __init__(self):
        super(AftershockClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # sigmoid is applied here
        return x.squeeze()

# --- 5. Training Prep ---
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = AftershockClassifier()

# Use standard binary cross-entropy 
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 6. Training Loop ---
epochs = 200
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 7. Evaluation ---
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred_probs = model(test_X_tensor).numpy()
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

# --- 8. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:\n", cm)

# --- 9. Classification Report ---
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# --- 10. ROC AUC ---
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {auc_score:.4f}")

print(datetime.now().strftime("%d-%m-%Y %H:%M"))

# --- 11. ROC Curve ---
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
 [[686786 423657]
 [  4813  18704]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.62      0.76   1110443
           1       0.04      0.80      0.08     23517

    accuracy                           0.62   1133960
   macro avg       0.52      0.71      0.42   1133960
weighted avg       0.97      0.62      0.75   1133960


ROC AUC Score: 0.7783


"""
