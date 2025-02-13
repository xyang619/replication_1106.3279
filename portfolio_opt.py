import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 参数设置
# np.random.seed(42)
n_assets = 10                  # 资产数量
n_periods = 5000               # 时间点数量
rebalance_freq = 240           # 再平衡周期
annual_vol_range = (0.1, 0.5)  # 年化波动率范围
corr_level = 0.2               # 资产间相关系数
mu_daily = 0.0002              # 日收益率均值

# 生成真实波动率向量
annual_vols = np.linspace(annual_vol_range[0], annual_vol_range[1], n_assets)
daily_vols = annual_vols / np.sqrt(252)

# 创建真实相关系数矩阵
corr_matrix = np.full((n_assets, n_assets), corr_level)
np.fill_diagonal(corr_matrix, 1)

# 创建真实协方差矩阵
cov_matrix = np.outer(daily_vols, daily_vols) * corr_matrix

# 生成对数收益率数据（使用真实协方差矩阵）
log_returns = np.random.multivariate_normal(
    mean=mu_daily * np.linspace(0.1, 2, n_assets),
    cov=cov_matrix,
    size=n_periods
)

# 初始化净值数组
nav_rp = np.ones(n_periods + 1)  # 风险平价组合
nav_mv = np.ones(n_periods + 1)  # Markowitz组合
nav_ew = np.ones(n_periods + 1)  # 等权重组合

# 改进后的风险平价权重优化
def risk_parity_weights(cov_matrix, maxiter=1000):
    n = cov_matrix.shape[0]
    def objective(w):
        sigma = np.sqrt(w @ cov_matrix @ w)
        if sigma < 1e-8:
            return 0
        rc = (w * (cov_matrix @ w)) / sigma
        return np.sum((rc - rc.mean())**2)
    
    # 使用波动率倒数作为初始值
    vol = np.sqrt(np.diag(cov_matrix))
    init_weights = (1/vol)/np.sum(1/vol)
    
    res = minimize(
        objective,
        x0=init_weights,
        method='SLSQP',
        bounds=[(0,1)]*n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w)-1},
        options={'maxiter': maxiter}
    )
    print(res)
    return res.x if res.success else init_weights

# Markowitz优化函数保持不变
def markowitz_weights(mu, cov_matrix):
    n = len(mu)
    def objective(w):
        return - (w @ mu) / np.sqrt(w @ cov_matrix @ w)
    
    res = minimize(
        objective,
        x0=np.ones(n)/n,
        method='SLSQP',
        bounds=[(0,1)]*n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w)-1}
    )
    print(res)
    return res.x if res.success else np.ones(n)/n

# 主循环处理再平衡（使用真实协方差矩阵验证）
rebalance_days = list(range(rebalance_freq, n_periods, rebalance_freq))

for rb in rebalance_days:
    # 使用真实协方差矩阵替代历史估计
    hist_data = log_returns[rb-rebalance_freq:rb]
    hist_cov = np.cov(hist_data.T)
    hist_mu = np.mean(log_returns[rb-rebalance_freq:rb], axis=0)
    
    # 计算权重
    w_rp = risk_parity_weights(hist_cov)
    w_mv = markowitz_weights(hist_mu, hist_cov)
    w_ew = np.ones(n_assets)/n_assets
    
    # 应用权重到持有期
    start, end = rb, min(rb+rebalance_freq, n_periods)
    for t in range(start, end):
        simple_ret = np.exp(log_returns[t]) - 1  # 转换为简单收益率
        nav_rp[t+1] = nav_rp[t] * (1 + w_rp @ simple_ret)
        nav_mv[t+1] = nav_mv[t] * (1 + w_mv @ simple_ret)
        nav_ew[t+1] = nav_ew[t] * (1 + w_ew @ simple_ret)

# 显示最新权重分配示例
print("示例风险平价权重（最后一次再平衡）:")
print(np.round(w_rp, 3))

# 后续统计和绘图代码保持不变...
# 计算统计指标
def calculate_stats(nav):
    ret = nav[1:]/nav[:-1] - 1
    annual_ret = nav[-1]**(252/n_periods) - 1
    annual_vol = np.std(ret) * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
    
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav)/peak
    return {
        'Return': annual_ret,
        'Volatility': annual_vol,
        'Sharpe': sharpe,
        'MaxDD': np.max(drawdown)
    }

stats_rp = calculate_stats(nav_rp)
stats_mv = calculate_stats(nav_mv)
stats_ew = calculate_stats(nav_ew)

# 打印结果
print(f"{'Metric':<15}{'Risk Parity':>15}{'Markowitz':>15}{'Equal Weight':>15}")
for k in stats_rp:
    print(f"{k:<15}{stats_rp[k]:>15.4f}{stats_mv[k]:>15.4f}{stats_ew[k]:>15.4f}")

# 绘制净值曲线
plt.figure(figsize=(12, 6))
plt.plot(nav_rp, label='Risk Parity')
plt.plot(nav_mv, label='Markowitz')
plt.plot(nav_ew, label='Equal Weight')
plt.legend()
plt.title('Portfolio NAV Comparison')
plt.xlabel('Days')
plt.ylabel('Net Asset Value')
plt.show()

# 绘制有效边界
last_hist = log_returns[-rebalance_freq:]
mu_eff = np.mean(np.exp(last_hist)-1, axis=0)
cov_eff = np.cov(last_hist.T)

target_returns = np.linspace(mu_eff.min(), mu_eff.max(), 50)
eff_volatility = []

for ret in target_returns:
    cons = (
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        {'type': 'eq', 'fun': lambda w: w @ mu_eff - ret}
    )
    res = minimize(
        lambda w: np.sqrt(w @ cov_eff @ w),
        x0=np.ones(n_assets)/n_assets,
        method='SLSQP',
        bounds=[(0,1)]*n_assets,
        constraints=cons
    )
    if res.success:
        eff_volatility.append(np.sqrt(res.fun))

# plt.figure(figsize=(12, 6))
# plt.plot(np.array(eff_volatility), target_returns, label='Efficient Frontier')
# plt.scatter(stats_rp['Volatility'], stats_rp['Return'], label='Risk Parity')
# plt.scatter(stats_mv['Volatility'], stats_mv['Return'], label='Markowitz')
# plt.scatter(stats_ew['Volatility'], stats_ew['Return'], label='Equal Weight')
# plt.xlabel('Annualized Volatility')
# plt.ylabel('Annualized Return')
# plt.legend()
# plt.title('Efficient Frontier Comparison')
# plt.show()
