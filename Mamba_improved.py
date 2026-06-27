# Mamba 模型 - 生物炭吸附铀数据
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ===== Mamba 模块（参考官方 repo 简化结构） =====
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


# ===== Mamba 回归模型 =====
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


# 读取CSV数据
data = pd.read_csv("REAL_biochar_adsorption_ECs_mapped.csv")

# 查看数据信息
print("Data shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())

# 特征列和目标列
feature_columns = [
    'SA (m2/g)', 'Dav (nm)', 'VTot (cm3/g)', 'C (wt%)',
    'O/C', '(O+N)/C', 'pH', 'T (K)', 'C0 (mg/L)', 'SLR (g/L)'
]
target_column = 'Qe (mg/g)'

# 提取特征和目标变量
X = data[feature_columns].values
y = data[target_column].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# 划分训练测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=64,
    shuffle=True
)

# 训练Mamba模型
print("\n========== Training Mamba Model ==========")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MambaRegressor(input_dim=X_train_scaled.shape[1], hidden_dim=64, dropout=0.2)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
n_epochs = 100
train_losses = []
val_losses = []

print(f"Starting training for {n_epochs} epochs...")

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # 前向传播
        pred = model(xb)
        loss = criterion(pred, yb)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 计算平均训练损失
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证损失
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)
        val_outputs = model(X_test_device)
        val_loss = criterion(val_outputs, y_test_device).item()
        val_losses.append(val_loss)

    # 每20个epoch打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Model training completed!")

# 模型预测
model.eval()
with torch.no_grad():
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    y_train_pred_tensor = model(X_train_tensor).cpu().numpy().flatten()
    y_test_pred_tensor = model(X_test_tensor).cpu().numpy().flatten()

# ========== 训练集评估 ==========
print("\n========== Training Set Performance Metrics ==========")
mae_train = mean_absolute_error(y_train, y_train_pred_tensor)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_tensor))
r2_train = r2_score(y_train, y_train_pred_tensor)

print(f"MAE  (Mean Absolute Error):  {mae_train:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse_train:.2f}")
print(f"R²   (R-squared):            {r2_train:.2f}")

# ========== 测试集评估 ==========
print("\n========== Test Set Performance Metrics ==========")
mae_test = mean_absolute_error(y_test, y_test_pred_tensor)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_tensor))
r2_test = r2_score(y_test, y_test_pred_tensor)

print(f"MAE  (Mean Absolute Error):  {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse_test:.2f}")
print(f"R²   (R-squared):            {r2_test:.2f}")

# ========== 对比分析 ==========
print("\n========== Training vs Test Set Comparison ==========")
comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Training': [f'{mae_train:.2f}', f'{rmse_train:.2f}', f'{r2_train:.2f}'],
    'Test': [f'{mae_test:.2f}', f'{rmse_test:.2f}', f'{r2_test:.2f}'],
    'Difference': [f'{mae_test - mae_train:.2f}',
                   f'{rmse_test - rmse_train:.2f}',
                   f'{r2_test - r2_train:.2f}']
})
print(comparison_df.to_string(index=False))

# ========== 模型信息 ==========
print("\n========== Mamba Model Information ==========")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Hidden dimension: 64")
print(f"Number of Mamba blocks: 2")
print(f"Dropout rate: 0.2 (20%)")

# ========== 模型评估总结 ==========
print("\n========== Model Evaluation Summary ==========")
if abs(r2_train - r2_test) < 0.05:
    print("✓ Good generalization, training and test performance are close")
elif r2_train > r2_test + 0.05:
    print("⚠ Possible slight overfitting, test performance slightly lower than training")
    print("  Suggestion: Increase dropout rate or reduce model complexity")
else:
    print("! Abnormal model performance, check data quality")

print(f"\nFinal Test Set Results:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")
print(f"  R²:   {r2_test:.2f}")

# ========== 预测示例 ==========
print("\n========== Test Set Prediction Examples (First 5) ==========")
prediction_df = pd.DataFrame({
    'Actual': y_test[:5],
    'Predicted': y_test_pred_tensor[:5],
    'Error': y_test[:5] - y_test_pred_tensor[:5],
    'Absolute Error': np.abs(y_test[:5] - y_test_pred_tensor[:5])
})
print(prediction_df.to_string(index=False))

# ========== 最终评估指标汇总 ==========
print("\n" + "=" * 60)
print("【Final Evaluation Metrics】")
print("=" * 60)
print(f"MAE  (Mean Absolute Error):       {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error):   {rmse_test:.2f}")
print(f"R²   (R-squared Score):            {r2_test:.2f}")
print("=" * 60)

# ========== 绘制收敛曲线 ==========
print("\n========== Generating Training Convergence Curve ==========")
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss', linewidth=2, color='blue')
plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss', linewidth=2, color='red')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Mamba Model Training Convergence Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Mamba_convergence_curve.png', dpi=300, bbox_inches='tight')
print("Convergence curve saved as: Mamba_convergence_curve.png")
plt.show()

print("\n========== Mamba Model Training Completed ==========")