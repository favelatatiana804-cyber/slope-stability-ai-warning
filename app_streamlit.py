# -*- coding: utf-8 -*-
"""
边坡稳定性智能预警与加固推荐原型系统 (Streamlit Web App)
运行方式：streamlit run app_streamlit.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from slope_lem import search_critical_slip

st.set_page_config(page_title="边坡稳定性智能评估与滑坡预警系统", page_icon="⛰️", layout="wide")
st.title("⛰️ 边坡稳定性智能评估与滑坡预警系统")
st.caption("《土力学（双语）》第八章课程设计 · 集成极限平衡条分法、机器学习代理模型与加固智能推荐 · 作者：5120243437")

# 侧边栏：现场监测与边坡参数输入
st.sidebar.header("1. 边坡几何与土体参数")
H = st.sidebar.slider("边坡高度 H (m)", 5.0, 35.0, 15.0, 0.5)
beta = st.sidebar.slider("坡角 β (°)", 20.0, 60.0, 36.0, 1.0)
gamma = st.sidebar.slider("土体重度 γ (kN/m³)", 16.0, 22.0, 19.0, 0.2)
c = st.sidebar.slider("有效黏聚力 c (kPa)", 5.0, 50.0, 22.0, 1.0)
phi = st.sidebar.slider("有效内摩擦角 φ (°)", 10.0, 40.0, 22.0, 0.5)

st.sidebar.header("2. 现场动态监测数据（预警输入）")
ru = st.sidebar.slider("孔隙水压力比 ru（地下水位/降雨）", 0.00, 0.50, 0.35, 0.01)
dh = st.sidebar.slider("施工坡脚开挖加深 ΔH (m)", 0.0, 6.0, 0.0, 0.5)
q = st.sidebar.slider("坡顶施工堆载 q (kPa)", 0.0, 60.0, 0.0, 5.0)
disp_rate = st.sidebar.slider("地表位移速率 v (mm/d)", 0.0, 20.0, 2.5, 0.5)

# 实时计算（调用简化 Bishop / 瑞典 / Janbu 条分法）
res = search_critical_slip(H + dh, beta, gamma, c, phi, ru, q_surcharge=q, n_slices=20)
fs_bishop = res["fs_bishop"]
fs_swedish = res["fs_swedish"]
fs_janbu = res["fs_janbu"]

# 综合风险预警判别（安全系数 + 现场位移速率双指标融合）
if fs_bishop < 1.05 or disp_rate >= 10.0:
    level, color, action = "Ⅰ级 · 红色预警（极高风险/失稳临界）", "error", "立即撤离坡脚人员设备，启动应急抢险与临时反压/截水措施！"
elif fs_bishop < 1.20 or disp_rate >= 5.0:
    level, color, action = "Ⅱ级 · 橙色预警（高风险/欠稳定）", "warning", "加密 GNSS 与孔压监测频次至 1 次/2h，停止坡脚开挖并实施应急加固。"
elif fs_bishop < 1.35 or disp_rate >= 2.0:
    level, color, action = "Ⅲ级 · 黄色关注（中风险/基本稳定）", "info", "保持常规巡检与雨季排水疏通，严禁坡顶违规堆载。"
else:
    level, color, action = "Ⅳ级 · 绿色正常（低风险/稳定）", "success", "边坡处于安全稳定状态，按正常周期监测即可。"

c1, c2, c3, c4 = st.columns(4)
c1.metric("简化 Bishop 安全系数", f"{fs_bishop:.3f}", delta=f"{fs_bishop - 1.35:+.3f} vs 规范1.35")
c2.metric("简化 Janbu 安全系数", f"{fs_janbu:.3f}")
c3.metric("瑞典条分法安全系数", f"{fs_swedish:.3f}")
c4.metric("现场位移速率", f"{disp_rate:.1f} mm/d")

getattr(st, color)(f"**当前预警等级：{level}**  \n处置建议：{action}")

# 卓越计划：AI 加固措施智能推荐
st.subheader("🛠️ 卓越计划：基于安全-经济帕累托分析的加固措施智能推荐")
if fs_bishop >= 1.35:
    st.success("当前边坡已满足规范安全系数要求（Fs ≥ 1.35），无需额外结构加固，建议做好坡面防冲刷与截水沟维护。")
else:
    candidates = [
        ("方案A：削坡减载（放缓坡角 10°）", search_critical_slip(H + dh, max(beta - 10, 20), gamma, c, phi, ru, q)["fs_bishop"], 1.85),
        ("方案B：截排水降压（ru 降至 0.12）", search_critical_slip(H + dh, beta, gamma, c, phi, min(ru, 0.12), q)["fs_bishop"], 1.20),
        ("方案C：预应力锚索（等效 Δc=+12kPa）", search_critical_slip(H + dh, beta, gamma, c + 12, phi, ru, q)["fs_bishop"] + 0.08, 2.90),
        ("方案D：抗滑桩支护（高抗滑储备）", search_critical_slip(H + dh, beta, gamma, c + 14, phi + 2, ru, q)["fs_bishop"] + 0.15, 4.20),
        ("方案E：综合治理（排水+锚索协同）", search_critical_slip(H + dh, beta, gamma, c + 8, phi, min(ru, 0.12), q)["fs_bishop"] + 0.05, 3.10),
    ]
    rows = []
    for name, fs_new, cost in candidates:
        dfs = fs_new - fs_bishop
        rows.append({
            "加固方案": name,
            "加固后 Fs": round(fs_new, 3),
            "提升量 ΔFs": round(dfs, 3),
            "估算造价 (万元/延米)": cost,
            "效费比 (ΔFs/万元)": round(dfs / cost, 3),
            "是否满足 Fs≥1.35": "✅ 达标" if fs_new >= 1.35 else "❌ 未达标",
        })
    res_df = pd.DataFrame(rows)
    st.dataframe(res_df, use_container_width=True)
    qualified = res_df[res_df["加固后 Fs"] >= 1.35]
    if len(qualified) > 0:
        best_row = qualified.sort_values("效费比 (ΔFs/万元)", ascending=False).iloc[0]
        st.info(f"**最优推荐**：{best_row['加固方案']}（加固后 Fs = {best_row['加固后 Fs']}，效费比最高达 {best_row['效费比 (ΔFs/万元)']} ΔFs/万元）。")
