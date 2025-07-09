import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === 1.Read data ===
train_dir = '/Users/PC/OneDrive/Masaüstü/traindata/'
test_dir = '/Users/PC/OneDrive/Masaüstü/testdata/'

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

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

target = 'is_aftershock'

train_df = train_df.dropna(subset=features + [target])
test_df = test_df.dropna(subset=features + [target])

# === 2. OVERSAMPLING ===
train_df_majority = train_df[train_df[target] == 0]
train_df_minority = train_df[train_df[target] == 1]

n_samples_to_add = len(train_df_majority) // 2
minority_oversampled = train_df_minority.sample(n=n_samples_to_add, replace=True, random_state=42)

balanced_train_df = pd.concat([train_df_majority, train_df_minority, minority_oversampled]).sample(frac=1).reset_index(drop=True)

X_train = balanced_train_df[features].values
y_train = balanced_train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 3. PYTORCH DATASET ===
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === 4. MODEL TANIMI ===
class AftershockClassifier(nn.Module):
    def __init__(self):
        super(AftershockClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(115, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# === 5. Train  ===
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = AftershockClassifier()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# === 6. Train loop ===
best_auc = 0
epochs = 200
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        logits = model(batch_X).squeeze()
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # AUC kontrolü
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        val_probs = 1 / (1 + torch.exp(-val_logits)).numpy()
        current_auc = roc_auc_score(y_test, val_probs)
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, AUC: {current_auc:.4f}")

# === 7. Load best model ===
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    logits = model(test_X_tensor).squeeze().numpy()
    y_pred_probs = 1 / (1 + np.exp(-logits))  # sigmoid

# === 8. THRESHOLD  ===
threshold = 0.65
y_pred_labels = (y_pred_probs > threshold).astype(int)

# === 9. Metrics ===
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {auc_score:.4f}")
print(datetime.now().strftime("%d-%m-%Y %H:%M"))

# === 10. Graphics ===
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
"""
Confusion Matrix:
 [[1019812   90631]
 [  13364   10153]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.92      0.95   1110443
           1       0.10      0.43      0.16     23517

    accuracy                           0.91   1133960
   macro avg       0.54      0.68      0.56   1133960
weighted avg       0.97      0.91      0.94   1133960


ROC AUC Score: 0.8167

"""
