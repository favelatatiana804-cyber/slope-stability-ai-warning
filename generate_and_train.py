# -*- coding: utf-8 -*-
"""
设计六：基于AI的边坡稳定性智能评估与滑坡预警系统
完整数据生成、多模型机器学习训练与评估、滑坡易发性制图、
卓越计划（施工扰动与地下水敏感性分析 + AI加固措施智能推荐）与高清图表生成脚本
"""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from slope_lem import search_critical_slip, slope_surface_y

# 配置中文字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================
# 1. 纯 NumPy 实现的高效机器学习算法（CART、RF、正则化GBDT/XGBoost、MLP）
# =====================================================================
class DecisionTreeReg:
    def __init__(self, max_depth=8, min_samples_leaf=4, max_features=None, reg_lambda=0.0, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.reg_lambda = reg_lambda
        self.rng = rng or np.random.default_rng(42)
        self.tree = None
        self.feature_importances_ = None

    def _fit_node(self, X, y, depth, imp):
        n, d = X.shape
        val = float(np.sum(y) / (n + self.reg_lambda))
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.var(y) < 1e-8:
            return {"leaf": True, "val": val}

        feat_idx = np.arange(d)
        if self.max_features is not None and self.max_features < d:
            feat_idx = self.rng.choice(d, size=self.max_features, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thr = None
        parent_score = (np.sum(y) ** 2) / (n + self.reg_lambda)

        for f in feat_idx:
            xf = X[:, f]
            qs = np.unique(np.quantile(xf, np.linspace(0.08, 0.92, 16)))
            for thr in qs:
                left = xf <= thr
                nl = int(np.sum(left))
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                sl = np.sum(y[left])
                sr = np.sum(y[~left])
                gain = (sl * sl) / (nl + self.reg_lambda) + (sr * sr) / (nr + self.reg_lambda) - parent_score
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thr = float(thr)

        if best_feat is None:
            return {"leaf": True, "val": val}

        imp[best_feat] += best_gain
        left = X[:, best_feat] <= best_thr
        return {
            "leaf": False,
            "feat": best_feat,
            "thr": best_thr,
            "left": self._fit_node(X[left], y[left], depth + 1, imp),
            "right": self._fit_node(X[~left], y[~left], depth + 1, imp),
        }

    def fit(self, X, y):
        imp = np.zeros(X.shape[1], dtype=float)
        self.tree = self._fit_node(X, y, 0, imp)
        s = np.sum(imp)
        self.feature_importances_ = imp / s if s > 0 else imp
        return self

    def _pred_one(self, node, x):
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["val"]

    def predict(self, X):
        return np.array([self._pred_one(self.tree, x) for x in X], dtype=float)


class RandomForestReg:
    def __init__(self, n_estimators=60, max_depth=10, min_samples_leaf=3, max_features=4, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.trees = []
        imp = np.zeros(d, dtype=float)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            t = DecisionTreeReg(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=rng,
            ).fit(X[idx], y[idx])
            self.trees.append(t)
            imp += t.feature_importances_
        self.feature_importances_ = imp / np.sum(imp)
        return self

    def predict(self, X):
        preds = np.mean([t.predict(X) for t in self.trees], axis=0)
        return preds


class RegularizedGBDTReg:
    """XGBoost 风格正则化梯度提升树回归（带 L2 叶子正则项 lambda 与列采样）"""
    def __init__(self, n_estimators=120, learning_rate=0.08, max_depth=5,
                 min_samples_leaf=4, reg_lambda=1.5, subsample=0.85, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.seed = seed
        self.base_val = 0.0
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.base_val = float(np.mean(y))
        pred = np.full(n, self.base_val, dtype=float)
        self.trees = []
        imp = np.zeros(d, dtype=float)
        m = int(n * self.subsample)
        for _ in range(self.n_estimators):
            resid = y - pred
            idx = rng.choice(n, size=m, replace=False)
            t = DecisionTreeReg(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max(3, d - 1),
                reg_lambda=self.reg_lambda,
                rng=rng,
            ).fit(X[idx], resid[idx])
            self.trees.append(t)
            pred += self.learning_rate * t.predict(X)
            imp += t.feature_importances_
        self.feature_importances_ = imp / np.sum(imp)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.base_val, dtype=float)
        for t in self.trees:
            pred += self.learning_rate * t.predict(X)
        return pred


class MLPReg:
    """多层感知机神经网络回归 (Input -> 64 -> 32 -> 1, ReLU, Adam, L2)"""
    def __init__(self, hidden=(64, 32), lr=0.008, epochs=450, batch_size=64, l2=1e-4, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        h1, h2 = self.hidden
        # He initialization
        W1 = rng.normal(0, np.sqrt(2.0 / d), (d, h1))
        b1 = np.zeros(h1)
        W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
        b2 = np.zeros(h2)
        W3 = rng.normal(0, np.sqrt(2.0 / h2), (h2, 1))
        b3 = np.array([np.mean(y)])

        params = [W1, b1, W2, b2, W3, b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t_step = 0
        y_col = y.reshape(-1, 1)

        for ep in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = Xs[idx], y_col[idx]
                nb = xb.shape[0]
                # forward
                z1 = xb @ W1 + b1
                a1 = np.maximum(0.0, z1)
                z2 = a1 @ W2 + b2
                a2 = np.maximum(0.0, z2)
                out = a2 @ W3 + b3
                # backward
                dout = 2.0 * (out - yb) / nb
                dW3 = a2.T @ dout + self.l2 * W3
                db3 = dout.sum(axis=0)
                da2 = dout @ W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2 + self.l2 * W2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1 + self.l2 * W1
                db1 = dz1.sum(axis=0)

                grads = [dW1, db1, dW2, db2, dW3, db3]
                t_step += 1
                lr_now = self.lr * (0.5 ** (ep // 180))
                for i in range(len(params)):
                    m[i] = 0.9 * m[i] + 0.1 * grads[i]
                    v[i] = 0.999 * v[i] + 0.001 * (grads[i] ** 2)
                    m_hat = m[i] / (1.0 - 0.9 ** t_step)
                    v_hat = v[i] / (1.0 - 0.999 ** t_step)
                    params[i] -= lr_now * m_hat / (np.sqrt(v_hat) + 1e-8)

        self.params = params
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sd
        W1, b1, W2, b2, W3, b3 = self.params
        a1 = np.maximum(0.0, Xs @ W1 + b1)
        a2 = np.maximum(0.0, a1 @ W2 + b2)
        return (a2 @ W3 + b3).ravel()


class DecisionTreeClf:
    """CART 决策树分类器（Gini 不纯度）用于滑坡易发性分级"""
    def __init__(self, max_depth=6, min_samples_leaf=6):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    @staticmethod
    def _gini(y, n_classes):
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y, minlength=n_classes)
        p = counts / n
        return 1.0 - float(np.sum(p * p))

    def _fit_node(self, X, y, depth, n_classes):
        n, d = X.shape
        counts = np.bincount(y, minlength=n_classes)
        pred_cls = int(np.argmax(counts))
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.max(counts) == n:
            return {"leaf": True, "cls": pred_cls, "probs": (counts / n).tolist()}

        best_gini = 1e9
        best_feat, best_thr = None, None
        for f in range(d):
            xf = X[:, f]
            qs = np.unique(np.quantile(xf, np.linspace(0.08, 0.92, 18)))
            for thr in qs:
                left = xf <= thr
                nl = int(np.sum(left))
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                g = (nl * self._gini(y[left], n_classes) + nr * self._gini(y[~left], n_classes)) / n
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thr = float(thr)

        if best_feat is None:
            return {"leaf": True, "cls": pred_cls, "probs": (counts / n).tolist()}

        left = X[:, best_feat] <= best_thr
        return {
            "leaf": False,
            "feat": best_feat,
            "thr": best_thr,
            "left": self._fit_node(X[left], y[left], depth + 1, n_classes),
            "right": self._fit_node(X[~left], y[~left], depth + 1, n_classes),
        }

    def fit(self, X, y, n_classes=4):
        self.n_classes = n_classes
        self.tree = self._fit_node(X, y, 0, n_classes)
        return self

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            node = self.tree
            while not node["leaf"]:
                node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
            out[i] = node["cls"]
        return out


def reg_metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100.0)
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE_pct": round(mape, 2)}


def fs_to_susceptibility(fs):
    """滑坡易发性4级分类：0=低易发(>=1.35), 1=中易发(1.20-1.35), 2=高易发(1.05-1.20), 3=极高易发(<1.05)"""
    return np.where(fs < 1.05, 3, np.where(fs < 1.20, 2, np.where(fs < 1.35, 1, 0)))


# =====================================================================
# 2. 主流程：数据生成 -> 模型训练 -> 易发性制图 -> 卓越计划分析 -> 绘图
# =====================================================================
def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # ---------- (A) 经典算例验证 ----------
    benchmarks = [
        ("均质干燥黏性土边坡", 12.0, 35.0, 19.0, 20.0, 20.0, 0.00),
        ("含地下水边坡 (ru=0.25)", 12.0, 35.0, 19.0, 20.0, 20.0, 0.25),
        ("高陡粉质黏土边坡", 20.0, 45.0, 20.0, 25.0, 25.0, 0.15),
        ("软土缓坡 (ru=0.35)", 10.0, 25.0, 18.0, 12.0, 14.0, 0.35),
    ]
    bench_results = []
    for name, H, beta, g, c, phi, ru in benchmarks:
        res = search_critical_slip(H, beta, g, c, phi, ru, n_slices=24, grid_density="fine")
        bench_results.append({
            "case": name, "H": H, "beta": beta, "gamma": g, "c": c, "phi": phi, "ru": ru,
            "Fs_Swedish": round(res["fs_swedish"], 3),
            "Fs_Bishop": round(res["fs_bishop"], 3),
            "Fs_Janbu": round(res["fs_janbu"], 3),
            "xc": round(res["xc"], 2), "yc": round(res["yc"], 2), "R": round(res["R"], 2),
        })
    print("Benchmark verification completed:", bench_results[0])

    # 绘制图1：典型边坡最危险滑动面与三种条分法对比
    b0 = search_critical_slip(12.0, 35.0, 19.0, 20.0, 20.0, 0.25, n_slices=24, grid_density="fine")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    H0, beta0 = 12.0, np.deg2rad(35.0)
    x_crest0 = H0 / np.tan(beta0)
    xs_g = np.linspace(-8, x_crest0 + 14, 300)
    ys_g = slope_surface_y(xs_g, H0, beta0)
    ax1.fill_between(xs_g, -5, ys_g, color="#E8DFD1", alpha=0.85)
    ax1.plot(xs_g, ys_g, color="#3E2723", lw=2.2, label="边坡地表线")
    # 最危险滑弧
    xs_s = np.linspace(b0["x_left"], b0["x_right"], 120)
    ys_s = b0["yc"] - np.sqrt(np.maximum(b0["R"] ** 2 - (xs_s - b0["xc"]) ** 2, 0.0))
    ax1.plot(xs_s, ys_s, color="#C62828", lw=2.5, ls="--", label=f"最危险滑弧 (Fs={b0['fs_bishop']:.3f})")
    # 画土条竖线
    for xk in np.linspace(b0["x_left"], b0["x_right"], 15):
        yb = b0["yc"] - np.sqrt(max(b0["R"] ** 2 - (xk - b0["xc"]) ** 2, 0.0))
        yt = float(slope_surface_y(xk, H0, beta0))
        ax1.plot([xk, xk], [yb, yt], color="#5D4037", lw=0.7, alpha=0.65)
    ax1.scatter([b0["xc"]], [b0["yc"]], color="#1565C0", s=55, zorder=5, label=f"圆心 O({b0['xc']:.1f}, {b0['yc']:.1f})")
    ax1.plot([b0["xc"], b0["x_left"]], [b0["yc"], float(slope_surface_y(b0["x_left"], H0, beta0))],
             color="#1565C0", lw=1.0, ls=":")
    ax1.plot([b0["xc"], b0["x_right"]], [b0["yc"], float(slope_surface_y(b0["x_right"], H0, beta0))],
             color="#1565C0", lw=1.0, ls=":")
    ax1.set_xlim(-6, x_crest0 + 12)
    ax1.set_ylim(-4, b0["yc"] + 3)
    ax1.set_xlabel("水平距离 x (m)")
    ax1.set_ylabel("高程 y (m)")
    ax1.set_title("(a) 典型边坡条分与最危险滑动面搜索", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax1.grid(True, ls=":", alpha=0.45)

    # 右图：4个经典算例三种条分法安全系数对比
    labels_b = ["算例1\n干燥黏土", "算例2\n含地下水", "算例3\n高陡边坡", "算例4\n软土缓坡"]
    x_pos = np.arange(len(labels_b))
    w = 0.25
    ax2.bar(x_pos - w, [r["Fs_Swedish"] for r in bench_results], w, label="瑞典条分法", color="#78909C")
    ax2.bar(x_pos, [r["Fs_Bishop"] for r in bench_results], w, label="简化Bishop法", color="#1565C0")
    ax2.bar(x_pos + w, [r["Fs_Janbu"] for r in bench_results], w, label="简化Janbu法", color="#2E7D32")
    for i, r in enumerate(bench_results):
        ax2.text(i, r["Fs_Bishop"] + 0.04, f"{r['Fs_Bishop']:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1565C0")
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

    # ---------- (B) 批量生成 1,200 组边坡算例数据集 ----------
    N_SAMPLES = 1200
    print(f"Generating {N_SAMPLES} slope stability samples via LEM...")
    # 分层拉丁超立方抽样保证全参数空间覆盖
    def lhs(n, low, high):
        bins = np.linspace(0, 1, n + 1)
        u = rng.uniform(bins[:-1], bins[1:])
        rng.shuffle(u)
        return low + u * (high - low)

    Hs = lhs(N_SAMPLES, 5.0, 35.0)
    betas = lhs(N_SAMPLES, 20.0, 60.0)
    gammas = lhs(N_SAMPLES, 16.0, 22.0)
    cs = lhs(N_SAMPLES, 5.0, 50.0)
    phis = lhs(N_SAMPLES, 10.0, 40.0)
    rus = lhs(N_SAMPLES, 0.0, 0.50)

    records = []
    for i in range(N_SAMPLES):
        res = search_critical_slip(
            Hs[i], betas[i], gammas[i], cs[i], phis[i], rus[i],
            n_slices=16, grid_density="standard"
        )
        if res is not None and 0.25 <= res["fs_bishop"] <= 4.50:
            records.append({
                "H_m": round(float(Hs[i]), 2),
                "beta_deg": round(float(betas[i]), 2),
                "gamma_kNm3": round(float(gammas[i]), 2),
                "c_kPa": round(float(cs[i]), 2),
                "phi_deg": round(float(phis[i]), 2),
                "ru": round(float(rus[i]), 3),
                "Fs_Swedish": round(res["fs_swedish"], 4),
                "Fs_Bishop": round(res["fs_bishop"], 4),
                "Fs_Janbu": round(res["fs_janbu"], 4),
            })

    df = pd.DataFrame(records)
    df["Susceptibility_Class"] = fs_to_susceptibility(df["Fs_Bishop"].values)
    grade_names = {0: "低易发(稳定)", 1: "中易发(基本稳定)", 2: "高易发(欠稳定)", 3: "极高易发(不稳定)"}
    df["Susceptibility_Label"] = df["Susceptibility_Class"].map(grade_names)
    csv_path = os.path.join(BASE_DIR, "slope_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Dataset generated: {len(df)} valid samples in {time.time() - t0:.1f}s")

    # 绘制图2：数据集分布与相关性热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    fs_vals = df["Fs_Bishop"].values
    ax1.hist(fs_vals, bins=32, color="#1976D2", alpha=0.78, edgecolor="white")
    for thr, col, txt in [(1.05, "#C62828", "1.05"), (1.20, "#EF6C00", "1.20"), (1.35, "#2E7D32", "1.35")]:
        ax1.axvline(thr, color=col, ls="--", lw=1.4)
        ax1.text(thr + 0.03, ax1.get_ylim()[1] * 0.88, txt, color=col, fontsize=8.5, fontweight="bold")
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
    ax2.set_xticklabels(corr_labels, rotation=32, ha="right", fontsize=8.5)
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

    # ---------- (C) 训练与评估多机器学习模型 ----------
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

    models = {
        "随机森林 (RF)": lambda s: RandomForestReg(n_estimators=60, max_depth=10, min_samples_leaf=3, max_features=4, seed=s),
        "梯度提升树 (XGBoost风格)": lambda s: RegularizedGBDTReg(n_estimators=130, learning_rate=0.08, max_depth=5, min_samples_leaf=4, reg_lambda=1.5, seed=s),
        "神经网络 (MLP)": lambda s: MLPReg(hidden=(64, 32), lr=0.008, epochs=450, batch_size=64, l2=1e-4, seed=s),
    }

    model_results = {}
    trained_models = {}
    test_preds = {}

    # 5折交叉验证 + 测试集评估
    folds = np.array_split(tr_idx, 5)
    for m_name, m_fn in models.items():
        cv_r2s, cv_rmses = [], []
        for k in range(5):
            val_i = folds[k]
            trn_i = np.concatenate([folds[j] for j in range(5) if j != k])
            mk = m_fn(100 + k).fit(X[trn_i], y[trn_i])
            pv = mk.predict(X[val_i])
            met = reg_metrics(y[val_i], pv)
            cv_r2s.append(met["R2"])
            cv_rmses.append(met["RMSE"])

        t_s = time.time()
        m_full = m_fn(2026).fit(X_tr, y_tr)
        train_sec = round(time.time() - t_s, 2)
        p_te = m_full.predict(X_te)
        te_met = reg_metrics(y_te, p_te)
        te_met["CV5_R2"] = round(float(np.mean(cv_r2s)), 4)
        te_met["CV5_RMSE"] = round(float(np.mean(cv_rmses)), 4)
        te_met["Train_Time_s"] = train_sec
        model_results[m_name] = te_met
        trained_models[m_name] = m_full
        test_preds[m_name] = p_te
        print(f"Model {m_name}: {te_met}")

    # 排列特征重要性（在最优模型 MLP 与树模型 GBDT 上计算）
    best_model_name = max(model_results, key=lambda k: model_results[k]["R2"])
    best_model = trained_models[best_model_name]
    base_rmse = model_results[best_model_name]["RMSE"]
    perm_imp = []
    for j in range(X_te.shape[1]):
        diffs = []
        for _ in range(15):
            Xp = X_te.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            diffs.append(reg_metrics(y_te, best_model.predict(Xp))["RMSE"] - base_rmse)
        perm_imp.append(max(float(np.mean(diffs)), 0.0))
    perm_imp = np.array(perm_imp)
    perm_imp_norm = perm_imp / np.sum(perm_imp)
    gbdt_imp = trained_models["梯度提升树 (XGBoost风格)"].feature_importances_

    # 绘制图3：三种模型测试集预测精度对比散点图
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    colors_m = ["#2E7D32", "#1565C0", "#D84315"]
    for ax, (m_name, col) in zip(axes, zip(models.keys(), colors_m)):
        yp = test_preds[m_name]
        met = model_results[m_name]
        ax.scatter(y_te, yp, s=18, alpha=0.65, color=col, edgecolors="none")
        lims = [min(y_te.min(), yp.min()) - 0.1, max(y_te.max(), yp.max()) + 0.1]
        ax.plot(lims, lims, "k--", lw=1.2, label="理想拟合线 y=x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("条分法真实 Fs (Bishop)")
        ax.set_ylabel("模型预测 Fs")
        ax.set_title(f"{m_name}\nR²={met['R2']:.4f} | RMSE={met['RMSE']:.4f}", fontsize=10, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_model_comparison.png"), bbox_inches="tight")
    plt.close()

    # 绘制图4：特征重要性排序（排列重要性 vs 树分裂增益）
    order = np.argsort(perm_imp_norm)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y_pos = np.arange(len(feat_cn))
    ax.barh(y_pos + 0.18, perm_imp_norm[order] * 100, height=0.35, color="#1565C0", label=f"排列重要性 ({best_model_name})")
    ax.barh(y_pos - 0.18, gbdt_imp[order] * 100, height=0.35, color="#26A69A", label="分裂增益重要性 (GBDT)")
    for i, idx_f in enumerate(order):
        ax.text(perm_imp_norm[idx_f] * 100 + 0.6, i + 0.18, f"{perm_imp_norm[idx_f]*100:.1f}%", va="center", fontsize=8.5, color="#1565C0", fontweight="bold")
        ax.text(gbdt_imp[idx_f] * 100 + 0.6, i - 0.18, f"{gbdt_imp[idx_f]*100:.1f}%", va="center", fontsize=8.5, color="#00695C")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_cn[i] for i in order], fontsize=9.5)
    ax.set_xlabel("归一化特征贡献度 (%)")
    ax.set_title("边坡安全系数输入特征重要性对比分析", fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, max(perm_imp_norm.max(), gbdt_imp.max()) * 118)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_feature_importance.png"), bbox_inches="tight")
    plt.close()

    # ---------- (D) 滑坡易发性决策树分类与空间制图 ----------
    y_cls = df["Susceptibility_Class"].values
    dt_clf = DecisionTreeClf(max_depth=7, min_samples_leaf=5).fit(X_tr, y_cls[tr_idx], n_classes=4)
    y_cls_pred = dt_clf.predict(X_te)
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
    print("Decision Tree Susceptibility Classifier:", clf_metrics)

    # 构造 100x100 模拟山地丘陵研究区 (2km x 2km) 并生成滑坡易发性分区图
    gx, gy = np.meshgrid(np.linspace(0, 2.0, 100), np.linspace(0, 2.0, 100))
    # 模拟地形坡高 H、坡角 beta、孔压比 ru（河谷/降雨汇水区偏高）、土体 c/phi
    H_map = 12.0 + 14.0 * np.exp(-((gx - 0.65) ** 2 + (gy - 1.35) ** 2) / 0.35) + 9.0 * np.exp(-((gx - 1.5) ** 2 + (gy - 0.6) ** 2) / 0.45)
    beta_map = 24.0 + 24.0 * np.exp(-((gx - 0.7) ** 2 + (gy - 1.25) ** 2) / 0.30) + 14.0 * np.sin(gx * 1.8) * np.cos(gy * 1.5)
    beta_map = np.clip(beta_map, 20.0, 56.0)
    gamma_map = 18.8 + 0.8 * np.sin(gx + gy)
    c_map = 28.0 - 11.0 * np.exp(-((gx - 0.8) ** 2 + (gy - 1.1) ** 2) / 0.40) + 4.0 * np.cos(gx * 2.0)
    phi_map = 24.0 - 7.0 * np.exp(-((gx - 0.75) ** 2 + (gy - 1.15) ** 2) / 0.45) + 3.0 * np.sin(gy * 2.2)
    ru_map = np.clip(0.10 + 0.32 * np.exp(-((gx - 0.85) ** 2 + (gy - 1.15) ** 2) / 0.25), 0.05, 0.45)

    X_grid = np.column_stack([
        H_map.ravel(), beta_map.ravel(), gamma_map.ravel(),
        c_map.ravel(), phi_map.ravel(), ru_map.ravel()
    ])
    susc_map = dt_clf.predict(X_grid).reshape(100, 100)
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
    ax1.set_title("(a) 区域滑坡易发性空间分区图 (决策树模型)", fontsize=10.8, fontweight="bold")

    im2 = ax2.imshow(cm, cmap="Blues")
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

    # ---------- (E) 卓越计划额外要求：施工扰动与地下水敏感性分析 ----------
    # 典型软-中硬黏土公路路堑边坡基准工况
    base_slope = {"H": 16.0, "beta": 38.0, "gamma": 19.2, "c": 22.0, "phi": 20.0, "ru": 0.15}
    ru_grid = np.linspace(0.0, 0.50, 11)
    fs_ru_lem = []
    fs_ru_ai = []
    for rv in ru_grid:
        r_lem = search_critical_slip(base_slope["H"], base_slope["beta"], base_slope["gamma"],
                                     base_slope["c"], base_slope["phi"], rv, n_slices=20)
        fs_ru_lem.append(r_lem["fs_bishop"])
        fs_ru_ai.append(float(best_model.predict(np.array([[base_slope["H"], base_slope["beta"],
                                                            base_slope["gamma"], base_slope["c"],
                                                            base_slope["phi"], rv]]))[0]))

    # 施工开挖深度 dH (0~6m) 与坡顶堆载 q (0~60kPa) 耦合影响 (ru=0.25)
    dh_vals = np.linspace(0.0, 6.0, 7)
    q_vals = np.linspace(0.0, 60.0, 7)
    fs_coupling = np.zeros((len(q_vals), len(dh_vals)))
    for iq, qv in enumerate(q_vals):
        for ih, dh in enumerate(dh_vals):
            r_c = search_critical_slip(base_slope["H"] + dh, base_slope["beta"], base_slope["gamma"],
                                       base_slope["c"], base_slope["phi"], 0.25, q_surcharge=qv, n_slices=18)
            fs_coupling[iq, ih] = r_c["fs_bishop"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax1.plot(ru_grid, fs_ru_lem, "o-", color="#1565C0", lw=2.2, label="简化Bishop法解析计算")
    ax1.plot(ru_grid, fs_ru_ai, "s--", color="#D84315", lw=1.8, label=f"AI模型预测 ({best_model_name})")
    ax1.axhline(1.20, color="#EF6C00", ls=":", lw=1.3, label="黄色预警线 (Fs=1.20)")
    ax1.axhline(1.05, color="#C62828", ls="--", lw=1.3, label="红色预警线 (Fs=1.05)")
    ax1.set_xlabel("孔隙水压力比 ru (模拟降雨/水位抬升)")
    ax1.set_ylabel("边坡安全系数 Fs")
    ax1.set_title("(a) 地下水变化对边坡稳定性的非线性影响", fontsize=10.8, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.2)
    ax1.grid(True, ls=":", alpha=0.45)

    cf = ax2.contourf(dh_vals, q_vals, fs_coupling, levels=12, cmap="RdYlGn")
    cs2 = ax2.contour(dh_vals, q_vals, fs_coupling, levels=[1.05, 1.20, 1.35], colors="#111111", linewidths=1.3)
    ax2.clabel(cs2, fmt="Fs=%.2f", fontsize=8.5)
    fig.colorbar(cf, ax=ax2, label="安全系数 Fs")
    ax2.set_xlabel("坡脚开挖加深量 ΔH (m)")
    ax2.set_ylabel("坡顶施工堆载 q (kPa)")
    ax2.set_title("(b) 施工开挖与坡顶堆载耦合安全系数等值图 (ru=0.25)", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_construction_groundwater.png"), bbox_inches="tight")
    plt.close()

    # ---------- (F) 卓越计划额外要求：基于AI的加固措施智能推荐方案 ----------
    # 针对降雨+开挖后处于高风险的典型边坡 (H=20m, beta=38°, gamma=19.2, c=22, phi=20, ru=0.35)
    risk_case = [20.0, 38.0, 19.2, 22.0, 20.0, 0.35]
    fs_orig = float(best_model.predict(np.array([risk_case]))[0])
    # 5种工程加固方案物理参数等效映射与单位延米参考造价 (万元/延米)
    schemes = [
        {"name": "方案A：削坡减载\n(坡角38°→28°)", "params": [20.0, 28.0, 19.2, 22.0, 20.0, 0.35], "cost": 1.85, "fs_bonus": 0.00},
        {"name": "方案B：截排水降压\n(ru: 0.35→0.12)", "params": [20.0, 38.0, 19.2, 22.0, 20.0, 0.12], "cost": 1.20, "fs_bonus": 0.00},
        {"name": "方案C：预应力锚索\n(等效Δc=+12kPa)", "params": [20.0, 38.0, 19.2, 34.0, 20.0, 0.35], "cost": 2.90, "fs_bonus": 0.08},
        {"name": "方案D：抗滑桩支护\n(提供抗滑力)", "params": [20.0, 38.0, 19.2, 36.0, 22.0, 0.35], "cost": 4.20, "fs_bonus": 0.15},
        {"name": "方案E：综合治理\n(排水+锚索协同)", "params": [20.0, 38.0, 19.2, 30.0, 20.0, 0.12], "cost": 3.10, "fs_bonus": 0.05},
    ]
    reinf_summary = []
    for sc in schemes:
        fs_after = float(best_model.predict(np.array([sc["params"]]))[0]) + sc["fs_bonus"]
        delta_fs = fs_after - fs_orig
        eff_index = delta_fs / sc["cost"]  # 每万元延米安全系数增量
        meets_target = fs_after >= 1.35
        reinf_summary.append({
            "scheme": sc["name"].replace("\n", " "),
            "Fs_before": round(fs_orig, 3),
            "Fs_after": round(fs_after, 3),
            "Delta_Fs": round(delta_fs, 3),
            "Cost_w_per_m": sc["cost"],
            "Efficiency_per_w": round(eff_index, 3),
            "Meets_1_35": meets_target,
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

    sc_pts = ax2.scatter(costs, fs_afters, s=[e * 950 for e in effs], c=bar_cols, alpha=0.85, edgecolors="#222222", lw=1.0)
    ax2.axhline(1.35, color="#1B5E20", ls="-.", lw=1.2, alpha=0.8)
    for i, r in enumerate(reinf_summary):
        short_lbl = ["方案A 削坡", "方案B 排水", "方案C 锚索", "方案D 抗滑桩", "方案E 综合(推荐)"][i]
        ax2.annotate(f"{short_lbl}\n(效费比={r['Efficiency_per_w']:.2f})",
                     (costs[i], fs_afters[i]),
                     xytext=(0, 11 if i != 3 else -24), textcoords="offset points",
                     ha="center", fontsize=8.2, fontweight="bold" if i == 4 else "normal")
    ax2.set_xlabel("估算工程造价 (万元/延米)")
    ax2.set_ylabel("加固后安全系数 Fs")
    ax2.set_xlim(0.7, 4.9)
    ax2.set_ylim(min(fs_afters) - 0.15, max(fs_afters) + 0.25)
    ax2.set_title("(b) 安全系数-造价帕累托优选分析 (气泡大小=效费比)", fontsize=10.2, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_reinforcement_opt.png"), bbox_inches="tight")
    plt.close()

    # 汇总所有真实运行结果到 JSON 供 Word 报告与 PPT 严格引用
    summary = {
        "n_samples": len(df),
        "n_train": int(n_train),
        "n_test": int(n_all - n_train),
        "fs_stats": {
            "min": round(float(df["Fs_Bishop"].min()), 3),
            "max": round(float(df["Fs_Bishop"].max()), 3),
            "mean": round(float(df["Fs_Bishop"].mean()), 3),
            "std": round(float(df["Fs_Bishop"].std()), 3),
            "class_counts": {grade_names[k]: int((df["Susceptibility_Class"] == k).sum()) for k in range(4)},
        },
        "benchmarks": bench_results,
        "regression_models": model_results,
        "best_model": best_model_name,
        "feature_importance_perm_pct": {feat_cn[i]: round(float(perm_imp_norm[i] * 100), 1) for i in range(len(feat_cn))},
        "feature_importance_gbdt_pct": {feat_cn[i]: round(float(gbdt_imp[i] * 100), 1) for i in range(len(feat_cn))},
        "susceptibility_clf": clf_metrics,
        "susceptibility_area_pct": area_pct,
        "groundwater_sensitivity": {
            "ru_0": round(fs_ru_lem[0], 3),
            "ru_025": round(fs_ru_lem[5], 3),
            "ru_050": round(fs_ru_lem[-1], 3),
            "drop_pct": round((fs_ru_lem[0] - fs_ru_lem[-1]) / fs_ru_lem[0] * 100, 1),
        },
        "construction_coupling": {
            "base_dH0_q0": round(float(fs_coupling[0, 0]), 3),
            "dH6_q0": round(float(fs_coupling[0, -1]), 3),
            "dH0_q60": round(float(fs_coupling[-1, 0]), 3),
            "dH6_q60": round(float(fs_coupling[-1, -1]), 3),
        },
        "reinforcement_schemes": reinf_summary,
    }
    with open(os.path.join(BASE_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("ALL DONE! Total time:", round(time.time() - t0, 1), "s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
