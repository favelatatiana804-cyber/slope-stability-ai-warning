# 设计六：边坡稳定性智能评估与滑坡预警系统

## 项目文件说明
- `slope_lem.py`：第八章土质边坡极限平衡条分法核心计算模块（瑞典条分法、简化 Bishop 法、简化 Janbu 法、最危险滑动面网格搜索，支持地下水孔压比 `ru` 与坡顶施工超载 `q`）。
- `generate_and_train.py`：完整数据生成（1,192 组有效算例）、多机器学习模型（随机森林、XGBoost 风格正则化梯度提升树、MLP 神经网络）训练与 5 折交叉验证脚本。
- `retrain_and_plot.py`：引入土力学无量纲物理特征（稳定数 $N_s$ 与有效摩擦-坡度比 $N_f$）优化决策树滑坡易发性制图、卓越计划敏感性分析与 7 张高清图表生成脚本。
- `slope_dataset.csv`：基于条分法生成的 1,192 组边坡稳定性标准数据集。
- `results_summary.json`：全部算例验证、模型精度评估、混淆矩阵与加固方案优选的精确结果汇总。
- `app_streamlit.py`：基于 Streamlit 的边坡稳定性智能预警与加固推荐 Web 原型系统。
- `figures/`：7 张高清分析图表（PNG，220 DPI）。

## 运行方式
1. 安装基础依赖：`pip install numpy scipy pandas matplotlib streamlit`
2. 复现模型训练与全部高清图表：`python retrain_and_plot.py`
3. 启动交互式预警 Web 系统：`streamlit run app_streamlit.py`
