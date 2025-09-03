import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
from tqdm import tqdm

from CCPNet import CCPNet

class ChurnDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        num_features = self.X_num[idx]
        cat_features = {k: v[idx] for k, v in self.X_cat.items()}
        label = self.y[idx]
        return num_features, cat_features, label

df = pd.read_csv("churn-dataset-small.csv")

numeric_features = ["age", "daily_watch_hours", "tenure", "last_login", "promotions_used"]
categorical_features = ["gender", "subscription_type", "profiles", "genre_preference", "region"]

# Encode categorical features as codes
for col in categorical_features:
    df[col] = df[col].astype("category").cat.codes

X = df[numeric_features + categorical_features].values
y = df["churn"].values

# Split before resampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply ADASYN
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

print("Class distribution before ADASYN:", np.bincount(y_train))
print("Class distribution after ADASYN:", np.bincount(y_train_res))

# Convert resampled training data back to DataFrame
df_train_res = pd.DataFrame(X_train_res, columns=numeric_features + categorical_features)
df_train_res["churn"] = y_train_res

# Convert validation and test into DataFrames (for consistency)
df_val = pd.DataFrame(X_val, columns=numeric_features + categorical_features)
df_val["churn"] = y_val

df_test = pd.DataFrame(X_test, columns=numeric_features + categorical_features)
df_test["churn"] = y_test

def df_to_tensors(df):
    X_num = torch.tensor(df[numeric_features].values, dtype=torch.float32)
    X_cat = {col: torch.tensor(df[col].astype(int).values, dtype=torch.long) for col in categorical_features}
    y = torch.tensor(df["churn"].values, dtype=torch.float32)
    return X_num, X_cat, y

X_train_num, X_train_cat, y_train = df_to_tensors(df_train_res)
X_val_num, X_val_cat, y_val = df_to_tensors(df_val)
X_test_num, X_test_cat, y_test = df_to_tensors(df_test)

train_dataset = ChurnDataset(X_train_num, X_train_cat, y_train)
val_dataset = ChurnDataset(X_val_num, X_val_cat, y_val)
test_dataset = ChurnDataset(X_test_num, X_test_cat, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

categorical_vocab_sizes = {col: df[col].nunique() for col in categorical_features}
model = CCPNet(numeric_dim=len(numeric_features), categorical_vocab_sizes=categorical_vocab_sizes)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()


n_epochs = 25

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    for X_num_batch, X_cat_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        optimizer.zero_grad()
        outputs = model(X_num_batch, X_cat_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in val_loader:
            outputs = model(X_num_batch, X_cat_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * y_batch.size(0)

            preds = (outputs >= 0.5).float()
            val_correct += (preds == y_batch).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
