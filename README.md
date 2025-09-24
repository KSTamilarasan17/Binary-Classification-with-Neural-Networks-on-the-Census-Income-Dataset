# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

# NAME : TAMILARASAN K S.

# REG NO: 212223100056

# Requirements
```
pandas
numpy
torch
scikit-learn
jupyter
```
# Steps:

# 1. Data Preparation

Loaded dataset (income.csv).

Separated categorical, continuous, and label columns.

Encoded categorical features with LabelEncoder.

Scaled continuous features with StandardScaler.

Split into 25,000 training samples and 5,000 testing samples.

# 2. Model Design

Defined TabularModel class with:

Embeddings for categorical variables.

BatchNorm for continuous variables.

One hidden layer (50 neurons, ReLU, Dropout p=0.4).

Output layer with 2 classes (<=50K, >50K).

# 3. Training

Loss: CrossEntropyLoss

Optimizer: Adam (lr=0.001)

Epochs: 300

Tracked training loss and accuracy.

# 4. Evaluation

Evaluated on test set (5,000 samples).

Reported loss and accuracy.

# Program:
```
# ========================
# 1. Imports
# ========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
```
```
# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```
```
# ========================
# 2. Load Data
# ========================
df = pd.read_csv("income (1).csv")   # <- make sure file is in same folder
print("Dataset shape:", df.shape)
print(df.head())
```
```
# ========================
# 3. Separate categorical, continuous, label
# ========================
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("income")   # target
continuous_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
label_col = "income"

print("Categorical:", categorical_cols)
print("Continuous:", continuous_cols)
```
```
# Encode labels
label_enc = LabelEncoder()
df[label_col] = label_enc.fit_transform(df[label_col])  # <=50K=0, >50K=1
```
```
# Encode categoricals
cat_encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    cat_encoders[col] = enc
```
```
# Scale continuous
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
```
```
# ========================
# 4. Train-test split
# ========================
train_df, test_df = train_test_split(df, train_size=25000, test_size=5000, random_state=SEED)

X_train_cats = torch.tensor(train_df[categorical_cols].values, dtype=torch.int64)
X_train_conts = torch.tensor(train_df[continuous_cols].values, dtype=torch.float)
y_train = torch.tensor(train_df[label_col].values, dtype=torch.long)

X_test_cats = torch.tensor(test_df[categorical_cols].values, dtype=torch.int64)
X_test_conts = torch.tensor(test_df[continuous_cols].values, dtype=torch.float)
y_test = torch.tensor(test_df[label_col].values, dtype=torch.long)
```
```
# ========================
# 5. Model Definition
# ========================
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, hidden_units=50, dropout=0.4):
        super().__init__()
        # Embeddings for categorical variables
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(dropout)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Fully connected layers
        n_emb = sum([nf for ni, nf in emb_szs])
        self.fc1 = nn.Linear(n_emb + n_cont, hidden_units)
        self.fc2 = nn.Linear(hidden_units, out_sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cats, x_conts):
        # Categorical embeddings
        x = [emb(x_cats[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)

        # Continuous variables
        x_conts = self.bn_cont(x_conts)

        # Concatenate
        x = torch.cat([x, x_conts], 1)

        # Hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```
```
# Embedding sizes
cat_sizes = [int(df[col].nunique()) for col in categorical_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_sizes]

model = TabularModel(emb_szs, n_cont=len(continuous_cols), out_sz=2)
```
```
# ========================
# 6. Training
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_cats, X_train_conts)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        _, predicted = torch.max(y_pred, 1)
        acc = (predicted == y_train).sum().item() / len(y_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")
```
```
# ========================
# 7. Evaluation
# ========================
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_cats, X_test_conts)
    loss = criterion(y_test_pred, y_test)
    _, predicted = torch.max(y_test_pred, 1)
    acc = (predicted == y_test).sum().item() / len(y_test)

print(f"\nTest Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")
```
```
# ========================
# 8. Bonus: Prediction Function
# ========================
def predict_new(sample_dict):
    # Convert categorical
    cat_vals = []
    for col in categorical_cols:
        val = cat_encoders[col].transform([sample_dict[col]])[0]
        cat_vals.append(val)

    # Convert continuous
    cont_vals = [sample_dict[col] for col in continuous_cols]
    cont_vals = scaler.transform([cont_vals])[0]

    x_cats = torch.tensor([cat_vals], dtype=torch.int64)
    x_conts = torch.tensor([cont_vals], dtype=torch.float)

    with torch.no_grad():
        pred = model(x_cats, x_conts)
        prob = F.softmax(pred, dim=1)
        result = label_enc.inverse_transform([torch.argmax(prob).item()])[0]
        return result, prob.numpy()
```
```
# Example
print("\nExample prediction:")
sample = {
    "workclass": "Private",
    "education": "Bachelors",
    "marital_status": "Never-married",  
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native_country": "United-States",
    "age": 30,
    "fnlwgt": 200000,
    "education_num": 13,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 45
}
print(predict_new(sample))

```
# OUTPUT:
<img width="1226" height="526" alt="image" src="https://github.com/user-attachments/assets/1a2c1460-0980-4ea0-bc8e-d8e590a9cc45" />

# RESULT:
 Thus,The Binary-Classification-with-Neural-Networks-on-the-Census was successfully developed and trained using PyTorch.
