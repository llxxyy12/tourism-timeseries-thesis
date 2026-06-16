#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月度旅游接待人次季节时序建模与预测 - 完整分析流程

Author: 时间序列分析课程
Date: 2026-06-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime

print("\n" + "="*60)
print("  月度旅游接待人次季节时序建模与预测")
print("  完整分析流程")
print("="*60)

# 创建结果目录
if not os.path.exists('结果'):
    os.makedirs('结果')

# ========== 步骤1：加载数据 ==========
print("\n[步骤1] 加载数据...")
df = pd.read_csv('数据/旅游数据_2015_2024.csv', index_col='date', parse_dates=True)
print(f"✓ 数据加载成功")
print(f"  时间范围：{df.index[0].strftime('%Y年%m月')} - {df.index[-1].strftime('%Y年%m月')}")
print(f"  样本量：{len(df)}个月")
print(f"  平均值：{df['visitors'].mean():.4f} 亿人次")
print(f"  标准差：{df['visitors'].std():.4f} 亿人次")

# ========== 步骤2：数据可视化 ==========
print("\n[步骤2] 生成时间序列图...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['visitors'], 'b-', linewidth=2)
ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('接待人次（亿人次）', fontsize=12)
ax.set_title('月度旅游接待人次时间序列图（2015-2024）', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('结果/01_时间序列图.png', dpi=300, bbox_inches='tight')
print("✓ 时间序列图已保存：结果/01_时间序列图.png")
plt.close()

# ========== 步骤3：时间序列分解 ==========
print("\n[步骤3] 执行时间序列分解...")
result = seasonal_decompose(df['visitors'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
axes[0].plot(result.observed.index, result.observed, 'b-', linewidth=1.5)
axes[0].set_ylabel('原始序列', fontsize=11)
axes[0].set_title('时间序列分解（加法模型）', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(result.trend.index, result.trend, 'g-', linewidth=1.5)
axes[1].set_ylabel('趋势成分', fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(result.seasonal.index, result.seasonal, 'r-', linewidth=1.5)
axes[2].set_ylabel('季节成分', fontsize=11)
axes[2].grid(True, alpha=0.3)

axes[3].plot(result.resid.index, result.resid, 'k-', linewidth=1)
axes[3].set_ylabel('随机成分', fontsize=11)
axes[3].set_xlabel('时间', fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('结果/02_时间序列分解.png', dpi=300, bbox_inches='tight')
print("✓ 时间序列分解图已保存")
plt.close()

# ========== 步骤4：平稳性检验 ==========
print("\n[步骤4] 执行平稳性检验（ADF）...")
adf_result = adfuller(df['visitors'].dropna())
print(f"  原始序列ADF统计量：{adf_result[0]:.6f}")
print(f"  p值：{adf_result[1]:.6f}")
print(f"  结论：序列{'非平稳' if adf_result[1] > 0.05 else '平稳'}")

diff1 = df['visitors'].diff().dropna()
adf_result_d1 = adfuller(diff1)
print(f"  一阶差分后ADF统计量：{adf_result_d1[0]:.6f}")
print(f"  p值：{adf_result_d1[1]:.6f}")
print(f"  结论：差分后序列{'非平稳' if adf_result_d1[1] > 0.05 else '平稳'}")

fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(df.index, df['visitors'], 'b-', linewidth=1.5)
axes[0].set_ylabel('原始序列', fontsize=11)
axes[0].set_title('平稳性检验：原始序列与一阶差分', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(diff1.index, diff1, 'r-', linewidth=1)
axes[1].set_ylabel('一阶差分', fontsize=11)
axes[1].set_xlabel('时间', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('结果/03_平稳性检验.png', dpi=300, bbox_inches='tight')
print("✓ 平稳性检验图已保存")
plt.close()

# ========== 步骤5：ACF/PACF分析 ==========
print("\n[步骤5] 执行自相关性分析（ACF/PACF）...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(df['visitors'], lags=40, ax=axes[0, 0], title='原始序列ACF')
plot_pacf(df['visitors'], lags=40, ax=axes[0, 1], title='原始序列PACF')
plot_acf(diff1, lags=40, ax=axes[1, 0], title='一阶差分后ACF')
plot_pacf(diff1, lags=40, ax=axes[1, 1], title='一阶差分后PACF')

plt.tight_layout()
plt.savefig('结果/04_ACF_PACF分析.png', dpi=300, bbox_inches='tight')
print("✓ ACF/PACF分析图已保存")
plt.close()

# ========== 步骤6：参数自动选择 ==========
print("\n[步骤6] 搜索最优SARIMA参数...")
print("  (这可能需要1-2分钟，请耐心等待...)")

best_model = auto_arima(
    df['visitors'],
    seasonal=True,
    m=12,
    max_p=5, max_q=5,
    max_P=2, max_Q=2,
    max_d=2, max_D=1,
    stepwise=True,
    trace=False,
    error_action='ignore',
    suppress_warnings=True
)

print(f"✓ 最优模型找到！")
print(f"  模型：SARIMA{best_model.order}{best_model.seasonal_order}")
print(f"  AIC：{best_model.aic():.2f}")
print(f"  BIC：{best_model.bic():.2f}")

# ========== 步骤7：模型拟合 ==========
print("\n[步骤7] 拟合SARIMA模型...")

train_size = int(len(df) * 0.8)
train_data = df['visitors'][:train_size]
test_data = df['visitors'][train_size:]

print(f"  训练集：{len(train_data)}个月")
print(f"  测试集：{len(test_data)}个月")

model = SARIMAX(
    train_data,
    order=best_model.order,
    seasonal_order=best_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False, maxiter=1000)
print(f"✓ 模型拟合完成")
print(f"  Log-Likelihood: {results.llf:.2f}")
print(f"  AIC: {results.aic:.2f}")
print(f"  BIC: {results.bic:.2f}")

# ========== 步骤8：残差诊断 ==========
print("\n[步骤8] 进行残差诊断...")
fig = results.plot_diagnostics(figsize=(14, 10))
plt.savefig('结果/05_残差诊断.png', dpi=300, bbox_inches='tight')
print("✓ 残差诊断图已保存")
plt.close()

# ========== 步骤9：预测与评估 ==========
print("\n[步骤9] 执行样本外预测...")

forecast = results.get_forecast(steps=len(test_data))
forecast_ci = forecast.conf_int()
forecast_mean = forecast.predicted_mean

mae = mean_absolute_error(test_data, forecast_mean)
rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
mape = np.mean(np.abs((test_data - forecast_mean) / test_data)) * 100

print(f"✓ 预测完成")
print(f"  MAE（平均绝对误差）：{mae:.4f} 亿人次")
print(f"  RMSE（均方根误差）：{rmse:.4f} 亿人次")
print(f"  MAPE（平均绝对百分比误差）：{mape:.2f}%")

# ========== 步骤10：预测结果可视化 ==========
print("\n[步骤10] 绘制预测结果图...")

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(train_data.index, train_data, 'b-', label='训练集', linewidth=2)
ax.plot(test_data.index, test_data, 'g-', label='测试集', linewidth=2)
ax.plot(forecast_mean.index, forecast_mean, 'r--', label='预测值', linewidth=2)
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.2, color='red', label='95%置信区间'
)

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('接待人次（亿人次）', fontsize=12)
ax.set_title('SARIMA模型预测结果（样本外预测）', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('结果/06_预测结果.png', dpi=300, bbox_inches='tight')
print("✓ 预测结果图已保存")
plt.close()

# ========== 步骤11：未来预测 ==========
print("\n[步骤11] 进行未来12个月预测...")

model_full = SARIMAX(
    df['visitors'],
    order=best_model.order,
    seasonal_order=best_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results_full = model_full.fit(disp=False, maxiter=1000)

future_forecast = results_full.get_forecast(steps=12)
future_ci = future_forecast.conf_int()
future_mean = future_forecast.predicted_mean

last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)

print(f"✓ 未来预测完成")
print("\n  未来12个月预测结果：")
print("  " + "-"*58)
for date, value in zip(future_dates, future_mean):
    print(f"  {date.strftime('%Y年%m月'):12s} 预计接待人次：{value:.4f} 亿人次")

# ========== 步骤12：完整预测图 ==========
print("\n[步骤12] 绘制完整预测图...")

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df.index, df['visitors'], 'b-', label='历史数据', linewidth=2)
ax.plot(future_mean.index, future_mean, 'r--', label='未来预测', linewidth=2)
ax.fill_between(
    future_ci.index,
    future_ci.iloc[:, 0],
    future_ci.iloc[:, 1],
    alpha=0.2, color='red', label='95%置信区间'
)
ax.axvline(x=df.index[-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('接待人次（亿人次）', fontsize=12)
ax.set_title('月度旅游接待人次历史与预测（2015-2025）', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('结果/07_完整预测图.png', dpi=300, bbox_inches='tight')
print("✓ 完整预测图已保存")
plt.close()

# ========== 步骤13：保存结果 ==========
print("\n[步骤13] 保存分析结果...")

future_results = pd.DataFrame({
    '月份': future_dates.strftime('%Y年%m月'),
    '预测值': future_mean.values,
    '下界': future_ci.iloc[:, 0].values,
    '上界': future_ci.iloc[:, 1].values
})

future_results.to_csv('结果/预测结果_2025年.csv', index=False, encoding='utf-8-sig')
print("✓ 预测结果已保存：结果/预测结果_2025年.csv")

# 保存模型总结
with open('结果/模型总结.txt', 'w', encoding='utf-8') as f:
    f.write("月度旅游接待人次季节时序建模与预测 - 模型总结\n")
    f.write("="*60 + "\n\n")
    f.write(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
    f.write("【数据信息】\n")
    f.write(f"时间范围：{df.index[0].strftime('%Y年%m月')} - {df.index[-1].strftime('%Y年%m月')}\n")
    f.write(f"样本量：{len(df)}个月\n")
    f.write(f"平均值：{df['visitors'].mean():.4f} 亿人次\n")
    f.write(f"标准差：{df['visitors'].std():.4f} 亿人次\n\n")
    
    f.write("【最优模型】\n")
    f.write(f"SARIMA{best_model.order}{best_model.seasonal_order}\n")
    f.write(f"AIC：{best_model.aic():.2f}\n")
    f.write(f"BIC：{best_model.bic():.2f}\n\n")
    
    f.write("【样本外预测精度】\n")
    f.write(f"MAE：{mae:.4f} 亿人次\n")
    f.write(f"RMSE：{rmse:.4f} 亿人次\n")
    f.write(f"MAPE：{mape:.2f}%\n\n")
    
    f.write(str(results.summary()))

print("✓ 模型总结已保存：结果/模型总结.txt")

print("\n" + "="*60)
print("  ✓ 分析完全完成！")
print("="*60)
print("\n生成的文件：")
print("  图表：")
print("    - 结果/01_时间序列图.png")
print("    - 结果/02_时间序列分解.png")
print("    - 结果/03_平稳性检验.png")
print("    - 结果/04_ACF_PACF分析.png")
print("    - 结果/05_残差诊断.png")
print("    - 结果/06_预测结果.png")
print("    - 结果/07_完整预测图.png")
print("  数据：")
print("    - 结果/预测结果_2025年.csv")
print("    - 结果/模型总结.txt")
print("\n")