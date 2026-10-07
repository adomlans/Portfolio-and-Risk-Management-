# -*- coding: utf-8 -*-
"""
投资组合与风险管理 · Day14-15 实战脚本
模拟数据兜底：不依赖任何网络，可直接跑。
输出：efficient_frontier.png + 控制台打印全部报告数字（供 Word 报告引用）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "/home/user/Doubao/chats/38440457112078338/portfolio_report"
FIG = f"{OUT_DIR}/efficient_frontier.png"

# ============ 1. 资产数据（4 只资产，模拟月收益 120 个月） ============
np.random.seed(2026)
n_assets, n_months = 4, 120
mu = np.array([0.012, 0.008, 0.006, 0.010])        # 月均收益（模拟）
sig = np.array([0.20, 0.16, 0.12, 0.17]) / np.sqrt(12)  # 月波动（年化 20%/16%/12%/17%）
# 构造协方差矩阵：对角为方差，非对角由相关系数决定
rho = np.array([[1.00, 0.45, 0.25, 0.35],
                [0.45, 1.00, 0.30, 0.40],
                [0.25, 0.30, 1.00, 0.20],
                [0.35, 0.40, 0.20, 1.00]])
cov = np.diag(sig) @ rho @ np.diag(sig)

# ============ 2. 权重扫描（画点云） ============
N_SIM = 30000
np.random.seed(7)
w = np.random.dirichlet(np.ones(n_assets), N_SIM)
port_mu = w @ mu
port_var = np.einsum("ij,jk,ik->i", w, cov, w)
port_sig = np.sqrt(port_var)

# ============ 3. 优化：GMV + 最大夏普 + 有效前沿 ============
bnds = [(0, 1)] * n_assets
x0 = np.ones(n_assets) / n_assets
cons_sum = {"type": "eq", "fun": lambda ww: np.sum(ww) - 1}

def var_obj(ww):  # 组合方差
    return ww @ cov @ ww

# GMV（全局最小方差）
res_gmv = minimize(var_obj, x0, bounds=bnds, constraints=cons_sum)
w_gmv = res_gmv.x
gmv_mu, gmv_sig = w_gmv @ mu, np.sqrt(res_gmv.fun)

# 最大夏普（无风险月利率 0.2%）
rf_m = 0.002
def sharpe_neg(ww):
    return -(ww @ mu - rf_m) / np.sqrt(ww @ cov @ ww)
cons_alloc = cons_sum
res_sh = minimize(sharpe_neg, x0, bounds=bnds, constraints=cons_alloc)
w_sh = res_sh.x
sh_mu, sh_sig = w_sh @ mu, np.sqrt(w_sh @ cov @ w_sh)
sharpe = (sh_mu - rf_m) / sh_sig

# 有效前沿（给定目标收益，最小化方差）
frontier = []
for target in np.linspace(mu.min(), mu.max(), 50):
    cons = [cons_sum, {"type": "eq", "fun": lambda ww, t=target: ww @ mu - t}]
    r = minimize(var_obj, x0, bounds=bnds, constraints=cons)
    if r.success:
        frontier.append((r.x @ mu, np.sqrt(r.fun)))

# ============ 4. 画图 ============
plt.figure(figsize=(7.2, 5.6))
plt.scatter(port_sig * np.sqrt(12), port_mu * 12, s=2, c="lightblue", label="随机组合（3万组）")
fx = np.array([p[1] for p in frontier]) * np.sqrt(12)
fy = np.array([p[0] for p in frontier]) * 12
plt.plot(fx, fy, "r-", lw=2.2, label="有效前沿")
plt.scatter([gmv_sig * np.sqrt(12)], [gmv_mu * 12], c="green", s=120, marker="*", label="最小方差组合 GMV")
plt.scatter([sh_sig * np.sqrt(12)], [sh_mu * 12], c="orange", s=120, marker="D", label="最大夏普组合")
plt.xlabel("年化波动率"); plt.ylabel("年化预期收益")
plt.title("有效前沿 · 最小方差组合 · 最大夏普组合（模拟数据）")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG, dpi=150)
print(f"[OK] 图已保存: {FIG}")

# ============ 5. 组合日收益：VaR / CVaR / 波动 / 夏普 ============
# 日度参数与月度参数自洽：日收益≈月收益/21，日波动≈月波动/sqrt(21)
np.random.seed(99)
n_days = 500
daily_mu = mu / 21.0
daily_sig = sig / np.sqrt(21.0)
daily = np.random.normal(daily_mu, daily_sig, size=(n_days, n_assets))
port_daily = daily @ w_sh                      # 用最大夏普组合权重
var95 = np.percentile(port_daily, 5)
cvar95 = port_daily[port_daily <= var95].mean()
ann_sig = np.std(port_daily, ddof=1) * np.sqrt(250)
ann_ret = port_daily.mean() * 250
sharpe_d = (port_daily.mean() / np.std(port_daily, ddof=1)) * np.sqrt(250)

# ============ 6. 压力测试（示意情景） ============
shocks = {"温和熊市": -0.10, "金融危机": -0.30, "极端黑天鹅": -0.50}

# ============ 7. 输出报告数字 ============
print("\n===== 组合概况（最大夏普组合） =====")
print(f"权重: A={w_sh[0]:.2%}  B={w_sh[1]:.2%}  C={w_sh[2]:.2%}  D={w_sh[3]:.2%}")
print(f"年化预期收益≈{sh_mu*12:.2%}   年化波动≈{sh_sig*np.sqrt(12):.2%}   夏普≈{sharpe:.2f}")
print("\n===== GMV（最小方差组合） =====")
print(f"权重: A={w_gmv[0]:.2%}  B={w_gmv[1]:.2%}  C={w_gmv[2]:.2%}  D={w_gmv[3]:.2%}")
print(f"年化预期收益≈{gmv_mu*12:.2%}   年化波动≈{gmv_sig*np.sqrt(12):.2%}")
print("\n===== 风险度量（最大夏普组合 · 日口径·历史模拟法） =====")
print(f"95% VaR(1日)≈{var95:.2%}   95% CVaR(1日)≈{cvar95:.2%}")
print(f"年化波动≈{ann_sig:.2%}   年化收益≈{ann_ret:.2%}   夏普≈{sharpe_d:.2f}")
print("\n===== 压力测试（示意情景） =====")
for name, s in shocks.items():
    action = "减仓 / 对冲" if s <= -0.2 else "持有观察"
    print(f"{name}: 组合单日损失≈{s:.0%}  → 应对: {action}")

# 保存数字供报告使用
import json
summary = {
    "w_sh": [round(float(x), 4) for x in w_sh],
    "w_gmv": [round(float(x), 4) for x in w_gmv],
    "sh_mu_y": round(float(sh_mu * 12), 4), "sh_sig_y": round(float(sh_sig * np.sqrt(12)), 4),
    "sharpe": round(float(sharpe), 2),
    "gmv_mu_y": round(float(gmv_mu * 12), 4), "gmv_sig_y": round(float(gmv_sig * np.sqrt(12)), 4),
    "var95": round(float(var95), 4), "cvar95": round(float(cvar95), 4),
    "ann_sig": round(float(ann_sig), 4), "ann_ret": round(float(ann_ret), 4),
    "sharpe_d": round(float(sharpe_d), 2),
}
with open(f"{OUT_DIR}/portfolio_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n[OK] 数字已保存: portfolio_summary.json")
#（注：内容由AI生成）
