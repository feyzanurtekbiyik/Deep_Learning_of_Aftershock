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

# === 1. Read data ===
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

# === 2. UNDERSAMPLING ===
df_0 = train_df[train_df[target] == 0]
df_1 = train_df[train_df[target] == 1]

undersample_0 = df_0.sample(n=len(df_1) * 2, random_state=42)  # 1 sınıfının 2 katı kadar 0 sınıfı
balanced_train_df = pd.concat([undersample_0, df_1]).sample(frac=1, random_state=42)

X_train = balanced_train_df[features].values
y_train = balanced_train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 4. PYTORCH DATASET ===
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === 5. MODEL TANIMI ===
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

# === 6. Train ===
train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = AftershockClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

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

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        val_probs = 1 / (1 + torch.exp(-val_logits)).numpy()
        current_auc = roc_auc_score(y_test, val_probs)
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), 'best_model_undersample.pth')
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, AUC: {current_auc:.4f}")

# === 7. Analysis ===
model.load_state_dict(torch.load('best_model_undersample.pth'))
model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    logits = model(test_X_tensor).squeeze().numpy()
    y_pred_probs = 1 / (1 + np.exp(-logits))

threshold = 0.65
y_pred = (y_pred_probs > threshold).astype(int)

# === 8. Metrics ===
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_probs):.4f}")
print("Tarih:", datetime.now().strftime("%d-%m-%Y %H:%M"))

# === 9. Graphics ===
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_probs):.2f})')
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
 [[1025609   84834]
 [  14545    8972]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.92      0.95   1110443
           1       0.10      0.38      0.15     23517

    accuracy                           0.91   1133960
   macro avg       0.54      0.65      0.55   1133960
weighted avg       0.97      0.91      0.94   1133960

ROC AUC Score: 0.8087


"""
