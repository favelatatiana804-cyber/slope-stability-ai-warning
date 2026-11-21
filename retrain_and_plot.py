# -*- coding: utf-8 -*-
"""
快速复现脚本：读取已生成的 1192 组 LEM 边坡数据集，修复中文字体注册，
完成多模型训练、引入土力学无量纲特征优化决策树易发性制图、卓越计划分析与7张高清图表生成。
"""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from slope_lem import search_critical_slip, slope_surface_y
from generate_and_train import (
    RandomForestReg, RegularizedGBDTReg, MLPReg, DecisionTreeClf,
    reg_metrics, fs_to_susceptibility
)

# 显式注册 Noto Sans CJK SC 字体，杜绝中文方块警告
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
CJK_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = [CJK_NAME, "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = [CJK_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def add_physics_features(X):
    """基于第八章泰勒稳定数与无限边坡理论，构造2个物理无量纲特征：
    1) 稳定数特征 Ns = c / (gamma * H)
    2) 有效摩擦-坡度比 Nf = (1 - ru) * tan(phi) / tan(beta)
    """
    H, beta_deg, gamma, c, phi_deg, ru = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    ns = c / np.maximum(gamma * H, 1e-6)
    nf = (1.0 - ru) * np.tan(np.deg2rad(phi_deg)) / np.maximum(np.tan(np.deg2rad(beta_deg)), 1e-6)
    return np.column_stack([X, ns, nf])


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # 1. 读取已生成的 LEM 数据集
    df = pd.read_csv(os.path.join(BASE_DIR, "slope_dataset.csv"))
    with open(os.path.join(BASE_DIR, "results_summary.json"), "r", encoding="utf-8") as f:
        prev = json.load(f)
    bench_results = prev["benchmarks"]
    grade_names = {0: "低易发(稳定)", 1: "中易发(基本稳定)", 2: "高易发(欠稳定)", 3: "极高易发(不稳定)"}

    # 图1：典型边坡最危险滑动面与三种条分法对比
    b0 = search_critical_slip(12.0, 35.0, 19.0, 20.0, 20.0, 0.25, n_slices=24, grid_density="fine")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    H0, beta0 = 12.0, np.deg2rad(35.0)
    x_crest0 = H0 / np.tan(beta0)
    xs_g = np.linspace(-8, x_crest0 + 14, 300)
    ys_g = slope_surface_y(xs_g, H0, beta0)
    ax1.fill_between(xs_g, -5, ys_g, color="#E8DFD1", alpha=0.88)
    ax1.plot(xs_g, ys_g, color="#3E2723", lw=2.2, label="边坡地表线")
    xs_s = np.linspace(b0["x_left"], b0["x_right"], 120)
    ys_s = b0["yc"] - np.sqrt(np.maximum(b0["R"] ** 2 - (xs_s - b0["xc"]) ** 2, 0.0))
    ax1.plot(xs_s, ys_s, color="#C62828", lw=2.5, ls="--", label=f"最危险滑弧 (Fs={b0['fs_bishop']:.3f})")
    for xk in np.linspace(b0["x_left"], b0["x_right"], 15):
        yb = b0["yc"] - np.sqrt(max(b0["R"] ** 2 - (xk - b0["xc"]) ** 2, 0.0))
        yt = float(slope_surface_y(xk, H0, beta0))
        ax1.plot([xk, xk], [yb, yt], color="#5D4037", lw=0.75, alpha=0.65)
    ax1.scatter([b0["xc"]], [b0["yc"]], color="#1565C0", s=55, zorder=5, label=f"圆心 O({b0['xc']:.1f}, {b0['yc']:.1f})")
    ax1.plot([b0["xc"], b0["x_left"]], [b0["yc"], float(slope_surface_y(b0["x_left"], H0, beta0))], color="#1565C0", lw=1.0, ls=":")
    ax1.plot([b0["xc"], b0["x_right"]], [b0["yc"], float(slope_surface_y(b0["x_right"], H0, beta0))], color="#1565C0", lw=1.0, ls=":")
    ax1.set_xlim(-6, x_crest0 + 12)
    ax1.set_ylim(-4, b0["yc"] + 3)
    ax1.set_xlabel("水平距离 x (m)")
    ax1.set_ylabel("高程 y (m)")
    ax1.set_title("(a) 典型边坡条分与最危险滑动面搜索", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax1.grid(True, ls=":", alpha=0.45)

    labels_b = ["算例1\n干燥黏土", "算例2\n含地下水", "算例3\n高陡边坡", "算例4\n软土缓坡"]
    x_pos = np.arange(len(labels_b))
    w = 0.25
    ax2.bar(x_pos - w, [r["Fs_Swedish"] for r in bench_results], w, label="瑞典条分法", color="#78909C")
    ax2.bar(x_pos, [r["Fs_Bishop"] for r in bench_results], w, label="简化Bishop法", color="#1565C0")
    ax2.bar(x_pos + w, [r["Fs_Janbu"] for r in bench_results], w, label="简化Janbu法", color="#2E7D32")
    for i, r in enumerate(bench_results):
        ax2.text(i, r["Fs_Bishop"] + 0.04, f"{r['Fs_Bishop']:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1565C0")
    ax2.axhline(1.0, color="#C62828", ls="--", lw=1.0, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_b, fontsize=9)
    ax2.set_ylabel("最小安全系数 Fs")
    ax2.set_title("(b) 三种条分法经典算例对比", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(r["Fs_Bishop"] for r in bench_results) + 0.45)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, axis="y", ls=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_lem_slip_surface.png"), bbox_inches="tight")
    plt.close()

    # 图2：数据集分布与相关性热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    fs_vals = df["Fs_Bishop"].values
    ax1.hist(fs_vals, bins=32, color="#1976D2", alpha=0.80, edgecolor="white")
    for thr, col, txt in [(1.05, "#C62828", "1.05"), (1.20, "#EF6C00", "1.20"), (1.35, "#2E7D32", "1.35")]:
        ax1.axvline(thr, color=col, ls="--", lw=1.4)
        ax1.text(thr + 0.04, ax1.get_ylim()[1] * 0.87, txt, color=col, fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("简化 Bishop 法安全系数 Fs")
    ax1.set_ylabel("样本数量 (组)")
    ax1.set_title(f"(a) {len(df)}组边坡样本安全系数分布", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", ls=":", alpha=0.45)

    corr_cols = ["H_m", "beta_deg", "gamma_kNm3", "c_kPa", "phi_deg", "ru", "Fs_Bishop"]
    corr_labels = ["坡高H", "坡角β", "重度γ", "黏聚力c", "内摩擦角φ", "孔压比ru", "安全系数Fs"]
    corr = df[corr_cols].corr().values
    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr_labels)))
    ax2.set_yticks(range(len(corr_labels)))
    ax2.set_xticklabels(corr_labels, rotation=30, ha="right", fontsize=8.5)
    ax2.set_yticklabels(corr_labels, fontsize=8.5)
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            val = corr[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if abs(val) > 0.55 else "#222222", fontsize=7.8)
    ax2.set_title("(b) 输入参数与安全系数 Pearson 相关性热力图", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_dataset_corr.png"), bbox_inches="tight")
    plt.close()

    # 2. 多模型训练（复用同一划分种子）
    feat_cols = ["H_m", "beta_deg", "gamma_kNm3", "c_kPa", "phi_deg", "ru"]
    feat_cn = ["坡高 H (m)", "坡角 β (°)", "重度 γ (kN/m³)", "黏聚力 c (kPa)", "内摩擦角 φ (°)", "孔压比 ru"]
    X = df[feat_cols].values
    y = df["Fs_Bishop"].values
    n_all = len(df)
    perm = rng.permutation(n_all)
    n_train = int(0.8 * n_all)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    rf_m = RandomForestReg(n_estimators=60, max_depth=10, min_samples_leaf=3, max_features=4, seed=2026).fit(X_tr, y_tr)
    gbdt_m = RegularizedGBDTReg(n_estimators=130, learning_rate=0.08, max_depth=5, min_samples_leaf=4, reg_lambda=1.5, seed=2026).fit(X_tr, y_tr)
    mlp_m = MLPReg(hidden=(64, 32), lr=0.008, epochs=450, batch_size=64, l2=1e-4, seed=2026).fit(X_tr, y_tr)

    trained_models = {"随机森林 (RF)": rf_m, "梯度提升树 (XGBoost风格)": gbdt_m, "神经网络 (MLP)": mlp_m}
    model_results = prev["regression_models"]
    test_preds = {k: m.predict(X_te) for k, m in trained_models.items()}
    best_model_name = "神经网络 (MLP)"
    best_model = mlp_m

    # 图3：模型预测精度散点对比
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    colors_m = ["#2E7D32", "#1565C0", "#D84315"]
    for ax, (m_name, col) in zip(axes, zip(trained_models.keys(), colors_m)):
        yp = test_preds[m_name]
        met = model_results[m_name]
        ax.scatter(y_te, yp, s=18, alpha=0.68, color=col, edgecolors="none")
        lims = [min(y_te.min(), yp.min()) - 0.1, max(y_te.max(), yp.max()) + 0.1]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("条分法真实 Fs (Bishop)")
        ax.set_ylabel("模型预测 Fs")
        ax.set_title(f"{m_name}\nR²={met['R2']:.4f} | RMSE={met['RMSE']:.4f}", fontsize=10, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_model_comparison.png"), bbox_inches="tight")
    plt.close()

    # 图4：特征重要性对比
    perm_imp_norm = np.array([prev["feature_importance_perm_pct"][k] for k in feat_cn]) / 100.0
    gbdt_imp = np.array([prev["feature_importance_gbdt_pct"][k] for k in feat_cn]) / 100.0
    order = np.argsort(perm_imp_norm)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y_pos = np.arange(len(feat_cn))
    ax.barh(y_pos + 0.18, perm_imp_norm[order] * 100, height=0.35, color="#1565C0", label="排列重要性 (MLP神经网络)")
    ax.barh(y_pos - 0.18, gbdt_imp[order] * 100, height=0.35, color="#26A69A", label="分裂增益重要性 (GBDT)")
    for i, idx_f in enumerate(order):
        ax.text(perm_imp_norm[idx_f] * 100 + 0.6, i + 0.18, f"{perm_imp_norm[idx_f]*100:.1f}%", va="center", fontsize=8.5, color="#1565C0", fontweight="bold")
        ax.text(gbdt_imp[idx_f] * 100 + 0.6, i - 0.18, f"{gbdt_imp[idx_f]*100:.1f}%", va="center", fontsize=8.5, color="#00695C")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_cn[i] for i in order], fontsize=9.5)
    ax.set_xlabel("归一化特征贡献度 (%)")
    ax.set_title("边坡安全系数输入特征重要性对比分析", fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, max(perm_imp_norm.max(), gbdt_imp.max()) * 120)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_feature_importance.png"), bbox_inches="tight")
    plt.close()

    # 3. 引入土力学无量纲特征的决策树滑坡易发性分类器
    X_phys = add_physics_features(X)
    y_cls = df["Susceptibility_Class"].values
    dt_clf = DecisionTreeClf(max_depth=7, min_samples_leaf=4).fit(X_phys[tr_idx], y_cls[tr_idx], n_classes=4)
    y_cls_pred = dt_clf.predict(X_phys[te_idx])
    acc = float(np.mean(y_cls_pred == y_cls[te_idx]))
    cm = np.zeros((4, 4), dtype=int)
    for yt, yp in zip(y_cls[te_idx], y_cls_pred):
        cm[yt, yp] += 1
    f1s = []
    for k in range(4):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    clf_metrics = {"Accuracy": round(acc, 4), "Macro_F1": round(float(np.mean(f1s)), 4), "Confusion_Matrix": cm.tolist()}
    print("Physics-Enhanced Decision Tree Classifier:", clf_metrics)

    # 区域滑坡易发性制图
    gx, gy = np.meshgrid(np.linspace(0, 2.0, 100), np.linspace(0, 2.0, 100))
    H_map = 12.0 + 14.0 * np.exp(-((gx - 0.65) ** 2 + (gy - 1.35) ** 2) / 0.35) + 9.0 * np.exp(-((gx - 1.5) ** 2 + (gy - 0.6) ** 2) / 0.45)
    beta_map = np.clip(24.0 + 24.0 * np.exp(-((gx - 0.7) ** 2 + (gy - 1.25) ** 2) / 0.30) + 14.0 * np.sin(gx * 1.8) * np.cos(gy * 1.5), 20.0, 56.0)
    gamma_map = 18.8 + 0.8 * np.sin(gx + gy)
    c_map = 28.0 - 11.0 * np.exp(-((gx - 0.8) ** 2 + (gy - 1.1) ** 2) / 0.40) + 4.0 * np.cos(gx * 2.0)
    phi_map = 24.0 - 7.0 * np.exp(-((gx - 0.75) ** 2 + (gy - 1.15) ** 2) / 0.45) + 3.0 * np.sin(gy * 2.2)
    ru_map = np.clip(0.10 + 0.32 * np.exp(-((gx - 0.85) ** 2 + (gy - 1.15) ** 2) / 0.25), 0.05, 0.45)

    X_grid = np.column_stack([H_map.ravel(), beta_map.ravel(), gamma_map.ravel(), c_map.ravel(), phi_map.ravel(), ru_map.ravel()])
    susc_map = dt_clf.predict(add_physics_features(X_grid)).reshape(100, 100)
    area_pct = {
        "低易发区": round(float(np.mean(susc_map == 0) * 100), 1),
        "中易发区": round(float(np.mean(susc_map == 1) * 100), 1),
        "高易发区": round(float(np.mean(susc_map == 2) * 100), 1),
        "极高易发区": round(float(np.mean(susc_map == 3) * 100), 1),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    cmap_s = ListedColormap(["#4CAF50", "#FDD835", "#FB8C00", "#D32F2F"])
    norm_s = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_s.N)
    im1 = ax1.imshow(susc_map, origin="lower", extent=[0, 2.0, 0, 2.0], cmap=cmap_s, norm=norm_s)
    cs_cnt = ax1.contour(gx, gy, H_map, levels=[15, 20, 25], colors="#263238", linewidths=0.8, alpha=0.65)
    ax1.clabel(cs_cnt, fmt="H=%dm", fontsize=7.5)
    cbar = fig.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["低易发", "中易发", "高易发", "极高易发"], fontsize=8.5)
    ax1.set_xlabel("东西向坐标 X (km)")
    ax1.set_ylabel("南北向坐标 Y (km)")
    ax1.set_title("(a) 区域滑坡易发性空间分区图 (物理增强决策树)", fontsize=10.5, fontweight="bold")

    ax2.imshow(cm, cmap="Blues")
    cls_names = ["低易发", "中易发", "高易发", "极高易发"]
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(cls_names, fontsize=8.5)
    ax2.set_yticklabels(cls_names, fontsize=8.5)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "#111111", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("决策树预测等级")
    ax2.set_ylabel("条分法真实等级")
    ax2.set_title(f"(b) 测试集混淆矩阵 (Acc={acc*100:.1f}%, F1={np.mean(f1s):.3f})", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_susceptibility_map.png"), bbox_inches="tight")
    plt.close()

    # 4. 卓越计划：地下水与施工扰动分析（重新绘制清晰中文版图6）
    base_slope = {"H": 16.0, "beta": 38.0, "gamma": 19.2, "c": 22.0, "phi": 20.0}
    ru_grid = np.linspace(0.0, 0.50, 11)
    fs_ru_lem = [search_critical_slip(base_slope["H"], base_slope["beta"], base_slope["gamma"],
                                      base_slope["c"], base_slope["phi"], rv, n_slices=20)["fs_bishop"] for rv in ru_grid]
    fs_ru_ai = [float(best_model.predict(np.array([[base_slope["H"], base_slope["beta"], base_slope["gamma"],
                                                    base_slope["c"], base_slope["phi"], rv]]))[0]) for rv in ru_grid]

    dh_vals = np.linspace(0.0, 6.0, 7)
    q_vals = np.linspace(0.0, 60.0, 7)
    fs_coupling = np.zeros((len(q_vals), len(dh_vals)))
    for iq, qv in enumerate(q_vals):
        for ih, dh in enumerate(dh_vals):
            fs_coupling[iq, ih] = search_critical_slip(
                base_slope["H"] + dh, base_slope["beta"], base_slope["gamma"],
                base_slope["c"], base_slope["phi"], 0.25, q_surcharge=qv, n_slices=18
            )["fs_bishop"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax1.plot(ru_grid, fs_ru_lem, "o-", color="#1565C0", lw=2.2, label="简化Bishop法解析计算")
    ax1.plot(ru_grid, fs_ru_ai, "s--", color="#D84315", lw=1.8, label="AI模型预测 (MLP)")
    ax1.axhline(1.20, color="#EF6C00", ls=":", lw=1.3, label="黄色预警线 (Fs=1.20)")
    ax1.axhline(1.05, color="#C62828", ls="--", lw=1.3, label="红色预警线 (Fs=1.05)")
    ax1.set_xlabel("孔隙水压力比 ru (模拟降雨/水位抬升)")
    ax1.set_ylabel("边坡安全系数 Fs")
    ax1.set_title("(a) 地下水变化对边坡稳定性的非线性影响", fontsize=10.8, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.2)
    ax1.grid(True, ls=":", alpha=0.45)

    cf = ax2.contourf(dh_vals, q_vals, fs_coupling, levels=12, cmap="RdYlGn")
    cs2 = ax2.contour(dh_vals, q_vals, fs_coupling, levels=[0.85, 0.90, 0.95], colors="#111111", linewidths=1.2)
    ax2.clabel(cs2, fmt="Fs=%.2f", fontsize=8.5)
    fig.colorbar(cf, ax=ax2, label="安全系数 Fs")
    ax2.set_xlabel("坡脚开挖加深量 ΔH (m)")
    ax2.set_ylabel("坡顶施工堆载 q (kPa)")
    ax2.set_title("(b) 施工开挖与坡顶堆载耦合安全系数等值图 (ru=0.25)", fontsize=10.2, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_construction_groundwater.png"), bbox_inches="tight")
    plt.close()

    # 5. 卓越计划：典型临界高风险边坡 (H=15m, beta=36°, gamma=19.0, c=22kPa, phi=22°, ru=0.35) AI加固优选
    risk_case = [15.0, 36.0, 19.0, 22.0, 22.0, 0.35]
    fs_orig = float(best_model.predict(np.array([risk_case]))[0])
    schemes = [
        {"name": "方案A：削坡减载\n(坡角36°→26°)", "params": [15.0, 26.0, 19.0, 22.0, 22.0, 0.35], "cost": 1.85, "fs_bonus": 0.00},
        {"name": "方案B：截排水降压\n(ru: 0.35→0.12)", "params": [15.0, 36.0, 19.0, 22.0, 22.0, 0.12], "cost": 1.20, "fs_bonus": 0.00},
        {"name": "方案C：预应力锚索\n(等效Δc=+12kPa)", "params": [15.0, 36.0, 19.0, 34.0, 22.0, 0.35], "cost": 2.90, "fs_bonus": 0.08},
        {"name": "方案D：抗滑桩支护\n(提供抗滑力)", "params": [15.0, 36.0, 19.0, 36.0, 24.0, 0.35], "cost": 4.20, "fs_bonus": 0.15},
        {"name": "方案E：综合治理\n(排水+锚索协同)", "params": [15.0, 36.0, 19.0, 30.0, 22.0, 0.12], "cost": 3.10, "fs_bonus": 0.05},
    ]
    reinf_summary = []
    for sc in schemes:
        fs_after = float(best_model.predict(np.array([sc["params"]]))[0]) + sc["fs_bonus"]
        delta_fs = fs_after - fs_orig
        eff_index = delta_fs / sc["cost"]
        reinf_summary.append({
            "scheme": sc["name"].replace("\n", " "),
            "Fs_before": round(fs_orig, 3),
            "Fs_after": round(fs_after, 3),
            "Delta_Fs": round(delta_fs, 3),
            "Cost_w_per_m": sc["cost"],
            "Efficiency_per_w": round(eff_index, 3),
            "Meets_1_35": fs_after >= 1.35,
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    s_names = [s["name"] for s in schemes]
    fs_afters = [r["Fs_after"] for r in reinf_summary]
    costs = [r["Cost_w_per_m"] for r in reinf_summary]
    effs = [r["Efficiency_per_w"] for r in reinf_summary]
    bar_cols = ["#42A5F5", "#26A69A", "#FFA726", "#7E57C2", "#2E7D32"]
    bars = ax1.bar(range(len(s_names)), fs_afters, color=bar_cols, width=0.58)
    ax1.axhline(fs_orig, color="#C62828", ls="--", lw=1.3, label=f"加固前 Fs={fs_orig:.2f}")
    ax1.axhline(1.35, color="#1B5E20", ls="-.", lw=1.4, label="规范目标值 Fs=1.35")
    for b_bar, val in zip(bars, fs_afters):
        ax1.text(b_bar.get_x() + b_bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax1.set_xticks(range(len(s_names)))
    ax1.set_xticklabels(s_names, fontsize=8.2)
    ax1.set_ylabel("加固后安全系数 Fs")
    ax1.set_ylim(0, max(fs_afters) + 0.35)
    ax1.set_title("(a) 不同加固方案安全系数提升效果", fontsize=10.8, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.2)
    ax1.grid(True, axis="y", ls=":", alpha=0.45)

    ax2.scatter(costs, fs_afters, s=[e * 900 for e in effs], c=bar_cols, alpha=0.85, edgecolors="#222222", lw=1.0)
    ax2.axhline(1.35, color="#1B5E20", ls="-.", lw=1.2, alpha=0.8)
    for i, r in enumerate(reinf_summary):
        short_lbl = ["方案A 削坡", "方案B 排水", "方案C 锚索", "方案D 抗滑桩", "方案E 综合(推荐)"][i]
        ax2.annotate(f"{short_lbl}\n(效费比={r['Efficiency_per_w']:.2f})",
                     (costs[i], fs_afters[i]),
                     xytext=(0, 12 if i != 3 else -26), textcoords="offset points",
                     ha="center", fontsize=8.2, fontweight="bold" if i == 4 else "normal")
    ax2.set_xlabel("估算工程造价 (万元/延米)")
    ax2.set_ylabel("加固后安全系数 Fs")
    ax2.set_xlim(0.7, 4.9)
    ax2.set_ylim(min(fs_afters) - 0.15, max(fs_afters) + 0.28)
    ax2.set_title("(b) 安全系数-造价帕累托优选分析 (气泡大小=效费比)", fontsize=10.2, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_reinforcement_opt.png"), bbox_inches="tight")
    plt.close()

    # 更新 results_summary.json
    prev["susceptibility_clf"] = clf_metrics
    prev["susceptibility_area_pct"] = area_pct
    prev["reinforcement_case"] = {"H": 15.0, "beta": 36.0, "gamma": 19.0, "c": 22.0, "phi": 22.0, "ru": 0.35, "Fs_before": round(fs_orig, 3)}
    prev["reinforcement_schemes"] = reinf_summary
    with open(os.path.join(BASE_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)

    print(f"Retrain & Replot finished cleanly in {time.time() - t0:.1f}s! CJK font: {CJK_NAME}")
    print("Susceptibility:", clf_metrics, area_pct)
    print("Reinforcement:", reinf_summary)


if __name__ == "__main__":
    main()
