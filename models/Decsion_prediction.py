# Decision Tree 模型 - 生物炭吸附铀数据
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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

print(f"\n训练集大小: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练决策树模型
print("\n========== 训练决策树模型 ==========")
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("模型训练完成!")

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
print("\n========== 特征重要性(决策树)==========")
feature_importance = pd.DataFrame({
    '特征': feature_columns,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print(feature_importance.to_string(index=False))

# ========== 模型信息 ==========
print("\n========== 决策树模型信息 ==========")
print(f"树的深度: {model.get_depth()}")
print(f"叶节点数量: {model.get_n_leaves()}")

# ========== 模型评估总结 ==========
print("\n========== 模型评估总结 ==========")
if abs(r2_train - r2_test) < 0.05:
    print("✓ 模型泛化能力良好,训练集和测试集性能接近")
elif r2_train > r2_test + 0.05:
    print("⚠ 可能存在过拟合,测试集性能明显低于训练集")
    print("  建议: 降低树的深度(max_depth)或增加min_samples_split参数")
else:
    print("! 模型表现异常,需要检查数据质量")

print(f"\n最终测试集结果:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")
print(f"  R²:   {r2_test:.2f}")

# ========== 预测示例 ==========
print("\n========== 测试集预测示例(前5个)==========")
prediction_df = pd.DataFrame({
    '实际值': y_test[:5],
    '预测值': y_test_pred[:5],
    '误差': y_test[:5] - y_test_pred[:5],
    '绝对误差': np.abs(y_test[:5] - y_test_pred[:5])
})
print(prediction_df.to_string(index=False))

# ========== 最终评估指标汇总 ==========
print("\n" + "=" * 60)
print("【最终评估指标】")
print("=" * 60)
print(f"MAE  (Mean Absolute Error):       {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error):   {rmse_test:.2f}")
print(f"R²   (R-squared Score):            {r2_test:.2f}")
print("=" * 60)


# ========== 导出预测结果到Excel ==========
def export_predictions_to_excel(y_test, y_test_pred, filename='prediction_results.xlsx'):
    """
    将测试集的预测结果导出到Excel文件

    参数:
        y_test: 测试集实际值
        y_test_pred: 测试集预测值
        filename: 输出的Excel文件名
    """
    # 计算误差和绝对误差
    errors = y_test - y_test_pred
    absolute_errors = np.abs(errors)

    # 创建DataFrame
    results_df = pd.DataFrame({
        '样本编号': range(1, len(y_test) + 1),
        '实际值': y_test,
        '预测值': y_test_pred,
        '误差': errors,
        '绝对误差': absolute_errors,
        '相对误差(%)': (absolute_errors / y_test) * 100
    })

    # 保留2位小数
    results_df = results_df.round(2)

    # 导出到Excel
    results_df.to_excel(filename, index=False, sheet_name='预测结果')

    print(f"\n预测结果已成功导出到: {filename}")
    print(f"共导出 {len(y_test)} 条数据")

    # 显示统计信息
    print("\n========== 预测结果统计 ==========")
    print(f"平均绝对误差:     {absolute_errors.mean():.2f}")
    print(f"最大绝对误差:     {absolute_errors.max():.2f}")
    print(f"最小绝对误差:     {absolute_errors.min():.2f}")
    print(f"绝对误差标准差:   {absolute_errors.std():.2f}")

    return results_df


# 调用函数导出结果
prediction_results = export_predictions_to_excel(y_test, y_test_pred, '../prediction/DT_prediction_results.xlsx')

# 显示前10行预览
print("\n========== 预测结果预览(前10行) ==========")
print(prediction_results.head(10).to_string(index=False))