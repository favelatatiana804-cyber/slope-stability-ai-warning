# -*- coding: utf-8 -*-
"""
第八章 土质边坡稳定性分析 — 极限平衡条分法核心计算模块
支持：瑞典条分法 (Fellenius)、简化 Bishop 法、简化 Janbu 法
支持附加条件：孔隙水压力比 r_u、坡顶均布超载 q
"""
import numpy as np


def slope_surface_y(x, H, beta_rad):
    """给定横坐标 x，计算均质单级边坡地表高程 y(x)。
    坡脚位于 (0, 0)，坡顶位于 (H / tan(beta), H)。
    """
    x_crest = H / np.tan(beta_rad)
    return np.where(x <= 0.0, 0.0, np.where(x >= x_crest, H, x * np.tan(beta_rad)))


def evaluate_circle_lem(H, beta_deg, gamma, c, phi_deg, ru, xc, yc, R,
                        n_slices=24, q_surcharge=0.0):
    """计算单个圆弧滑动面 (xc, yc, R) 的三种条分法安全系数。
    若圆弧未有效切割坡体，返回 None。
    """
    beta = np.deg2rad(beta_deg)
    phi = np.deg2rad(phi_deg)
    tan_phi = np.tan(phi)
    x_crest = H / np.tan(beta)

    # 寻找圆弧与地表交点 x_left (坡脚或坡脚左侧) 和 x_right (坡顶或坡面)
    # 先在 [xc - R, xc + R] 细网格上搜索圆弧位于地表以下的区间
    xs_probe = np.linspace(xc - R * 0.999, xc + R * 0.999, 200)
    y_circ = yc - np.sqrt(np.maximum(R * R - (xs_probe - xc) ** 2, 0.0))
    y_surf = slope_surface_y(xs_probe, H, beta)
    inside = y_surf > y_circ + 1e-3

    if np.count_nonzero(inside) < 8:
        return None

    idx = np.where(inside)[0]
    # 确保只取单个连续滑体
    x_left = xs_probe[idx[0]]
    x_right = xs_probe[idx[-1]]

    # 有效滑动面应跨越坡体（出口在坡脚附近或坡面下部，入口在坡面或坡顶）
    if x_right <= 0.15 * x_crest or x_left >= 0.95 * x_crest:
        return None
    if (x_right - x_left) < 0.25 * H:
        return None

    # 划分土条
    edges = np.linspace(x_left, x_right, n_slices + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    b = edges[1:] - edges[:-1]

    y_top = slope_surface_y(x_mid, H, beta)
    y_bot = yc - np.sqrt(np.maximum(R * R - (x_mid - xc) ** 2, 1e-12))
    h_slice = np.maximum(y_top - y_bot, 0.0)

    if np.max(h_slice) < 0.05 * H:
        return None

    # 条底倾角 alpha：sin(alpha) = (x_mid - xc) / R
    sin_a = np.clip((x_mid - xc) / R, -0.96, 0.96)
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    tan_a = sin_a / cos_a
    l_slice = b / cos_a

    # 土条自重 + 坡顶超载 (仅作用于坡顶区域 x >= x_crest)
    W_soil = gamma * h_slice * b
    W_q = np.where(x_mid >= x_crest, q_surcharge * b, 0.0)
    W = W_soil + W_q

    # 孔隙水压力 u = r_u * gamma * h_slice
    u = ru * gamma * h_slice

    # 滑动力矩/滑动力项
    driving_moment = np.sum(W * sin_a)
    if driving_moment <= 1e-4:
        return None

    # 1) 瑞典条分法 (Fellenius / Ordinary Method of Slices)
    normal_eff = np.maximum(W * cos_a - u * l_slice, 0.0)
    resisting_swedish = np.sum(c * l_slice + normal_eff * tan_phi)
    fs_swedish = resisting_swedish / driving_moment

    # 2) 简化 Bishop 法 (迭代求解)
    fs_bishop = max(fs_swedish, 0.3)
    for _ in range(40):
        m_alpha = cos_a + (sin_a * tan_phi) / fs_bishop
        if np.any(m_alpha <= 0.15):
            return None
        num = np.sum((c * b + np.maximum(W - u * b, 0.0) * tan_phi) / m_alpha)
        fs_new = num / driving_moment
        if abs(fs_new - fs_bishop) < 1e-5:
            fs_bishop = fs_new
            break
        fs_bishop = 0.5 * fs_bishop + 0.5 * fs_new

    # 3) 简化 Janbu 法 (带修正系数 f0)
    driving_janbu = np.sum(W * tan_a)
    if driving_janbu <= 1e-4:
        fs_janbu = fs_bishop * 0.96
    else:
        fs_j0 = max(fs_bishop, 0.3)
        for _ in range(40):
            m_alpha = cos_a + (sin_a * tan_phi) / fs_j0
            if np.any(m_alpha <= 0.15):
                return None
            num_j = np.sum((c * b + np.maximum(W - u * b, 0.0) * tan_phi) / (cos_a * m_alpha))
            fs_new = num_j / driving_janbu
            if abs(fs_new - fs_j0) < 1e-5:
                fs_j0 = fs_new
                break
            fs_j0 = 0.5 * fs_j0 + 0.5 * fs_new

        # Janbu 经验修正系数 f0 (考虑条间剪力，c-phi 土 b1 ≈ 0.50)
        L_chord = np.hypot(x_right - x_left, slope_surface_y(x_right, H, beta) - slope_surface_y(x_left, H, beta))
        # 最大弧深 d
        y_line = (slope_surface_y(x_left, H, beta) +
                  (slope_surface_y(x_right, H, beta) - slope_surface_y(x_left, H, beta)) *
                  (x_mid - x_left) / max(x_right - x_left, 1e-6))
        d_max = np.max(np.maximum(y_line - y_bot, 0.0))
        d_over_L = np.clip(d_max / max(L_chord, 1e-6), 0.0, 0.45)
        b1 = 0.50 if (c > 0 and phi_deg > 0) else (0.69 if phi_deg == 0 else 0.31)
        f0 = 1.0 + b1 * (d_over_L - 1.4 * d_over_L ** 2)
        fs_janbu = f0 * fs_j0

    if not (0.15 < fs_bishop < 10.0 and 0.15 < fs_swedish < 10.0 and 0.15 < fs_janbu < 10.0):
        return None

    return {
        "fs_swedish": float(fs_swedish),
        "fs_bishop": float(fs_bishop),
        "fs_janbu": float(fs_janbu),
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R),
        "x_left": float(x_left),
        "x_right": float(x_right),
    }


def search_critical_slip(H, beta_deg, gamma, c, phi_deg, ru,
                         q_surcharge=0.0, n_slices=20, grid_density="standard"):
    """网格搜索最危险圆弧滑动面（以简化 Bishop 法安全系数最小为判据）。"""
    beta = np.deg2rad(beta_deg)
    x_crest = H / np.tan(beta)

    if grid_density == "fine":
        n_xc, n_yc, n_r = 10, 10, 8
    else:
        n_xc, n_yc, n_r = 7, 7, 5

    xc_vals = np.linspace(-0.25 * H, 0.85 * x_crest, n_xc)
    yc_vals = np.linspace(1.05 * H, 2.35 * H, n_yc)

    best = None
    for xc in xc_vals:
        for yc in yc_vals:
            # 经过坡脚 (0,0) 的半径为基准，上下扩展覆盖坡面圆与深层圆
            R_toe = np.hypot(xc, yc)
            R_min = max(R_toe * 0.85, yc - 0.85 * H)
            R_max = R_toe * 1.25
            for R in np.linspace(R_min, R_max, n_r):
                res = evaluate_circle_lem(
                    H, beta_deg, gamma, c, phi_deg, ru, xc, yc, R,
                    n_slices=n_slices, q_surcharge=q_surcharge
                )
                if res is not None:
                    if best is None or res["fs_bishop"] < best["fs_bishop"]:
                        best = res

    # 在粗网格最优点附近做局部精细搜索
    if best is not None:
        xc0, yc0, R0 = best["xc"], best["yc"], best["R"]
        for dxc in np.linspace(-0.12 * H, 0.12 * H, 5):
            for dyc in np.linspace(-0.12 * H, 0.12 * H, 5):
                for dR in np.linspace(-0.08 * R0, 0.08 * R0, 5):
                    res = evaluate_circle_lem(
                        H, beta_deg, gamma, c, phi_deg, ru,
                        xc0 + dxc, yc0 + dyc, R0 + dR,
                        n_slices=n_slices, q_surcharge=q_surcharge
                    )
                    if res is not None and res["fs_bishop"] < best["fs_bishop"]:
                        best = res

    return best


if __name__ == "__main__":
    # 经典土力学算例自检
    cases = [
        ("算例1：干燥均质黏性土边坡", 12.0, 35.0, 19.0, 20.0, 20.0, 0.0),
        ("算例2：含地下水边坡 (ru=0.25)", 12.0, 35.0, 19.0, 20.0, 20.0, 0.25),
        ("算例3：高陡粉质黏土边坡", 20.0, 45.0, 20.0, 25.0, 25.0, 0.15),
    ]
    for name, H, beta, g, c, phi, ru in cases:
        res = search_critical_slip(H, beta, g, c, phi, ru, grid_density="fine")
        print(f"{name}: Bishop={res['fs_bishop']:.3f}, Janbu={res['fs_janbu']:.3f}, Swedish={res['fs_swedish']:.3f}")
