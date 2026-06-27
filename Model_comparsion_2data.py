# model_comparison_r2_scatter.py - 9个模型的R²散点图对比（3×3布局）
# 所有模型使用 Data.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设置绘图样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== 数据加载和预处理 ==========
print("Loading dataset (Data.csv)...")
data = pd.read_csv("Data.csv")

feature_columns = [
    'SA (m2/g)', 'Dav (nm)', 'VTot (cm3/g)', 'C (wt%)',
    'O/C', '(O+N)/C', 'pH', 'T (K)', 'C0 (mg/L)', 'SLR (g/L)'
]
target_column = 'Qe (mg/g)'

X = data[feature_columns].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 用于PyTorch模型
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=64, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 存储所有模型的预测结果和真实值
predictions = {}
true_values = {}

# ========== 1. Decision Tree ==========
print("\n[1/9] Training Decision Tree...")
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10,
                                 min_samples_leaf=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)
predictions['Decision Tree'] = dt_model.predict(X_test_scaled)
true_values['Decision Tree'] = y_test

# ========== 2. Random Forest ==========
print("[2/9] Training Random Forest...")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                 min_samples_split=10, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf_model.predict(X_test_scaled)
true_values['Random Forest'] = y_test

# ========== 3. GBDT ==========
print("[3/9] Training GBDT...")
from sklearn.ensemble import GradientBoostingRegressor

gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=5, random_state=42)
gbdt_model.fit(X_train_scaled, y_train)
predictions['GBDT'] = gbdt_model.predict(X_test_scaled)
true_values['GBDT'] = y_test

# ========== 4. Linear Regression ==========
print("[4/9] Training Linear Regression...")
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
predictions['Linear Regression'] = lr_model.predict(X_test_scaled)
true_values['Linear Regression'] = y_test

# ========== 5. FNN ==========
print("[5/9] Training FNN...")


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


fnn_model = FNN(input_dim=X_train_scaled.shape[1]).to(device)
optimizer = torch.optim.Adam(fnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    fnn_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = fnn_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

fnn_model.eval()
with torch.no_grad():
    predictions['FNN'] = fnn_model(X_test_tensor.to(device)).cpu().numpy().flatten()
    true_values['FNN'] = y_test

# ========== 6. FNN + Attention ==========
print("[6/9] Training FNN + Attention...")


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention_fc(x)
        weighted_x = x * attention_weights
        return weighted_x, attention_weights


class FNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2):
        super(FNNWithAttention, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, attention_weights = self.attention(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x, attention_weights


fnn_att_model = FNNWithAttention(input_dim=X_train_scaled.shape[1]).to(device)
optimizer = torch.optim.Adam(fnn_att_model.parameters(), lr=0.001)

for epoch in range(100):
    fnn_att_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs, _ = fnn_att_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

fnn_att_model.eval()
with torch.no_grad():
    fnn_att_pred, _ = fnn_att_model(X_test_tensor.to(device))
    predictions['FNN + Attention'] = fnn_att_pred.cpu().numpy().flatten()
    true_values['FNN + Attention'] = y_test

# ========== 7. Mamba ==========
print("[7/9] Training Mamba...")


class MambaBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


class MambaRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            MambaBlock(hidden_dim, dropout=dropout),
            MambaBlock(hidden_dim, dropout=dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.encoder(x)


mamba_model = MambaRegressor(input_dim=X_train_scaled.shape[1], dropout=0.1).to(device)
optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=0.001)

for epoch in range(100):
    mamba_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = mamba_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

mamba_model.eval()
with torch.no_grad():
    predictions['Mamba'] = mamba_model(X_test_tensor.to(device)).cpu().numpy().flatten()
    true_values['Mamba'] = y_test

# ========== 8. Mamba + Dropout ==========
print("[8/9] Training Mamba + Dropout...")
mamba_dropout_model = MambaRegressor(input_dim=X_train_scaled.shape[1], dropout=0.2).to(device)
optimizer = torch.optim.AdamW(mamba_dropout_model.parameters(), lr=0.001)

for epoch in range(100):
    mamba_dropout_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = mamba_dropout_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

mamba_dropout_model.eval()
with torch.no_grad():
    predictions['Mamba + Dropout'] = mamba_dropout_model(X_test_tensor.to(device)).cpu().numpy().flatten()
    true_values['Mamba + Dropout'] = y_test

# ========== 9. Mamba + Attention ==========
print("[9/9] Training Mamba + Attention...")


class MambaAttentionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.attention = AttentionLayer(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            MambaBlock(hidden_dim, dropout=dropout),
            MambaBlock(hidden_dim, dropout=dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x, attention_weights = self.attention(x)
        output = self.encoder(x)
        return output, attention_weights


mamba_att_model = MambaAttentionRegressor(input_dim=X_train_scaled.shape[1]).to(device)
optimizer = torch.optim.AdamW(mamba_att_model.parameters(), lr=0.001)

for epoch in range(100):
    mamba_att_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs, _ = mamba_att_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

mamba_att_model.eval()
with torch.no_grad():
    mamba_att_pred, _ = mamba_att_model(X_test_tensor.to(device))
    predictions['Mamba + Attention'] = mamba_att_pred.cpu().numpy().flatten()
    true_values['Mamba + Attention'] = y_test

# ========== 计算R²并打印结果 ==========
print("\n" + "=" * 70)
print("Model R² Scores")
print("=" * 70)
for model_name in predictions.keys():
    y_pred = predictions[model_name]
    y_true = true_values[model_name]
    r2 = r2_score(y_true, y_pred)
    # 标注使用的数据集
    dataset_info = " (Data.csv)"
    print(f"{model_name:25s}: R² = {r2:.4f}{dataset_info}")
print("=" * 70)

# ========== 绘制3×3 R²散点图 ==========
print("\nGenerating 3×3 R² scatter plots...")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, model_name in enumerate(predictions.keys()):
    ax = axes[idx]

    y_pred = predictions[model_name]
    y_true = true_values[model_name]

    # 计算R²
    r2 = r2_score(y_true, y_pred)

    # 绘制散点
    ax.scatter(y_true, y_pred, alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.5)

    # 绘制理想拟合线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Ideal fit')

    # 设置标题和标签
    title = f'({chr(97 + idx)}) {model_name}\nR² = {r2:.3f}'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Experimental Qe', fontsize=10)
    ax.set_ylabel('Predicted Qe', fontsize=10)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 设置坐标轴范围一致
    ax.set_xlim([min_val - 10, max_val + 10])
    ax.set_ylim([min_val - 10, max_val + 10])

    # 设置刻度
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('model_comparison_r2_scatter_3x3.png', dpi=300, bbox_inches='tight')
print("✓ 3×3 R² scatter plot saved as: model_comparison_r2_scatter_3x3.png")
plt.show()

print("\n========== All Done! ==========")