# Linear Regression 模型 - 生物炭吸附铀数据
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取CSV数据
data = pd.read_csv("../data/REAL_biochar_adsorption_ECs_mapped.csv")

# 查看数据信息
print("数据形状:", data.shape)
print("\n列名:")
print(data.columns.tolist())
print("\n前5行数据:")
print(data.head())

# 特征列和目标列
# 特征: SA (m2/g), Dav (nm), VTot (cm3/g), C (wt%), O/C, (O+N)/C, pH, T (K), C0 (mg/L), SLR (g/L)
# 目标: Qe (mg/g) - 吸附量
feature_columns = [
    'SA (m2/g)', 'Dav (nm)', 'VTot (cm3/g)', 'C (wt%)',
    'O/C', '(O+N)/C', 'pH', 'T (K)', 'C0 (mg/L)', 'SLR (g/L)'
]
target_column = 'Qe (mg/g)'

# 提取特征和目标变量
X = data[feature_columns].values
y = data[target_column].values

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 划分训练测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练线性回归模型
print("\n========== 训练线性回归模型 ==========")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("模型训练完成！")

# 预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# ========== 训练集评估 ==========
print("\n========== 训练集性能指标 ==========")
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

print(f"MAE  (平均绝对误差):  {mae_train:.2f}")
print(f"RMSE (均方根误差):     {rmse_train:.2f}")
print(f"R²   (决定系数):       {r2_train:.2f}")

# ========== 测试集评估 ==========
print("\n========== 测试集性能指标 ==========")
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"MAE  (平均绝对误差):  {mae_test:.2f}")
print(f"RMSE (均方根误差):     {rmse_test:.2f}")
print(f"R²   (决定系数):       {r2_test:.2f}")

# ========== 对比分析 ==========
print("\n========== 训练集 vs 测试集对比 ==========")
comparison_df = pd.DataFrame({
    '指标': ['MAE', 'RMSE', 'R²'],
    '训练集': [f'{mae_train:.2f}', f'{rmse_train:.2f}', f'{r2_train:.2f}'],
    '测试集': [f'{mae_test:.2f}', f'{rmse_test:.2f}', f'{r2_test:.2f}'],
    '差异': [f'{mae_test - mae_train:.2f}',
             f'{rmse_test - rmse_train:.2f}',
             f'{r2_test - r2_train:.2f}']
})
print(comparison_df.to_string(index=False))

# ========== 特征重要性分析 ==========
print("\n========== 特征系数（标准化后）==========")
coefficients = pd.DataFrame({
    '特征': feature_columns,
    '系数': model.coef_
}).sort_values('系数', key=abs, ascending=False)
print(coefficients.to_string(index=False))
print(f"\n截距: {model.intercept_:.4f}")

# ========== 模型评估总结 ==========
print("\n========== 模型评估总结 ==========")
if abs(r2_train - r2_test) < 0.05:
    print("✓ 模型泛化能力良好，训练集和测试集性能接近")
elif r2_train > r2_test + 0.05:
    print("⚠ 可能存在轻微过拟合，测试集性能略低于训练集")
else:
    print("! 模型表现异常，需要检查数据质量")

print(f"\n最终测试集结果:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")
print(f"  R²:   {r2_test:.2f}")

# ========== 预测示例 ==========
print("\n========== 测试集预测示例（前5个）==========")
prediction_df = pd.DataFrame({
    '实际值': y_test[:5],
    '预测值': y_test_pred[:5],
    '误差': y_test[:5] - y_test_pred[:5],
    '绝对误差': np.abs(y_test[:5] - y_test_pred[:5])
})
print(prediction_df.to_string(index=False))

# ========== 最终评估指标汇总 ==========
print("\n" + "="*60)
print("【最终评估指标】")
print("="*60)
print(f"MAE  (Mean Absolute Error):       {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error):   {rmse_test:.2f}")
print(f"R²   (R-squared Score):            {r2_test:.2f}")
print("="*60)