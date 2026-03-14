import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# 设置页面标题
st.set_page_config(page_title="昏迷苏醒概率预测", layout="wide")
st.title("🧠 重症脑损伤昏迷患者苏醒概率预测")
st.markdown("基于XGBoost模型，输入以下5个特征即可获得苏醒概率及解释")

# 加载模型和特征（使用缓存，只加载一次）
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_model.pkl')
    features = joblib.load('final_features.pkl')
    median_dict = joblib.load('median_dict.pkl')
    return model, features, median_dict

model, features, median_dict = load_model()

# 创建输入面板
st.sidebar.header("输入特征")
input_data = {}

for feat in features:
    if feat == 'Pupillary_reflex':
        # 瞳孔反射：三分类
        reflex_options = {'消失': 0, '迟钝': 1, '灵敏': 2}
        reflex_text = st.sidebar.selectbox('瞳孔反射', list(reflex_options.keys()))
        value = reflex_options[reflex_text]
    elif feat == 'thalamic':
        # 丘脑损伤评分：四分类（无/轻度/中度/重度）
        thalamic_options = {'无': 0, '轻度': 1, '中度': 2, '重度': 3}
        thalamic_text = st.sidebar.selectbox('丘脑损伤', list(thalamic_options.keys()))
        value = thalamic_options[thalamic_text]
    elif feat == 'mechanical_ventilation':
        # 机械通气：二分类
        value = st.sidebar.selectbox('机械通气', [0, 1], format_func=lambda x: '是' if x == 1 else '否')
    else:
        # 连续变量：数字输入框，默认值为训练集的中位数
        default_val = median_dict[feat]
        value = st.sidebar.number_input(
            feat,
            value=float(default_val),
            step=0.1,
            format="%.2f"
        )
    input_data[feat] = value

# 将输入转换为 DataFrame
input_df = pd.DataFrame([input_data])

# 预测按钮
if st.sidebar.button("计算苏醒概率"):
    with st.spinner("正在计算，请稍候..."):
        # 预测概率
        prob = model.predict_proba(input_df)[0, 1]
        prob_percent = prob * 100

        # 显示结果
        st.subheader("预测结果")
        col1, col2, col3 = st.columns(3)
        col1.metric("苏醒概率", f"{prob_percent:.1f}%")
        col2.metric("预测类别", "苏醒" if prob >= 0.5 else "未苏醒")
        col3.metric("置信水平", "高" if prob > 0.8 or prob < 0.2 else "中")

        # 显示输入特征值（映射为可读文本）
        st.subheader("输入特征值")
        display_dict = input_data.copy()
        # 瞳孔反射映射
        reflex_rev = {0: '消失', 1: '迟钝', 2: '灵敏'}
        if 'Pupillary_reflex' in display_dict:
            display_dict['瞳孔反射'] = reflex_rev[display_dict.pop('Pupillary_reflex')]
        # 机械通气映射
        if 'mechanical_ventilation' in display_dict:
            display_dict['机械通气'] = '是' if display_dict.pop('mechanical_ventilation') == 1 else '否'
        # 丘脑损伤映射
        thalamic_rev = {0: '无', 1: '轻度', 2: '中度', 3: '重度'}
        if 'thalamic' in display_dict:
            display_dict['丘脑损伤'] = thalamic_rev[display_dict.pop('thalamic')]
        st.json(display_dict)

        # SHAP 解释
        st.subheader("模型解释 (SHAP瀑布图)")
        # 缓存解释器，避免重复计算
        @st.cache_resource
        def get_explainer(model):
            return shap.TreeExplainer(model)

        explainer = get_explainer(model)
        shap_values = explainer.shap_values(input_df)

        # 绘制瀑布图
        plt.figure(figsize=(10, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0].values,
                feature_names=features
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt)

        # 生成简要报告
        st.subheader("简要临床报告")
        report = f"""
        **患者预测报告**  
        根据当前输入，模型预测该患者**3个月内苏醒的概率为 {prob_percent:.1f}%**。

        **关键影响因素**：
        """
        shap_dict = dict(zip(features, shap_values[0]))
        pos = {k: v for k, v in shap_dict.items() if v > 0}
        neg = {k: v for k, v in shap_dict.items() if v < 0}
        if pos:
            report += "\n\n✅ **正向因素（增加苏醒概率）：**"
            for k, v in sorted(pos.items(), key=lambda x: -x[1]):
                report += f"\n- {k} (贡献 +{v:.2f})"
        if neg:
            report += "\n\n❌ **负向因素（降低苏醒概率）：**"
            for k, v in sorted(neg.items(), key=lambda x: x[1]):
                report += f"\n- {k} (贡献 {v:.2f})"

        st.markdown(report)

        # 提供下载按钮
        st.download_button(
            label="下载报告 (文本)",
            data=report,
            file_name="苏醒预测报告.txt",
            mime="text/plain"
        )
else:
    st.info("请在左侧边栏输入特征值，然后点击「计算苏醒概率」。")