# -*- coding: utf-8 -*-
"""
Streamlit Web 应用：重症脑损伤昏迷患者苏醒概率预测
增强版：包含 SHAP 贡献条形图、详细文字报告、风险分级及治疗建议
"""

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
st.markdown("基于 XGBoost 模型，输入以下 5 个特征即可获得苏醒概率及解释")

# 加载模型和特征（使用缓存）
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_model.pkl')
    features = joblib.load('final_features.pkl')
    median_dict = joblib.load('median_dict.pkl')
    return model, features, median_dict

model, features, median_dict = load_model()

# 加载训练集特征均值（用于文字比较）
@st.cache_data
def load_train_mean():
    df_train = pd.read_excel('lasso.xlsx', sheet_name='Sheet1')
    return df_train[features].mean().to_dict()

train_mean = load_train_mean()

# 特征名称映射（用于显示）
feature_display = {
    'Delta_GCS': 'ΔGCS (分/天)',
    'P/F': '氧合指数 (mmHg)',
    'mechanical_ventilation': '机械通气',
    'Pupillary_reflex': '瞳孔反射',
    'thalamic': '丘脑损伤'
}

# 特征选项映射
reflex_options = {'消失': 0, '迟钝': 1, '灵敏': 2}
reflex_rev = {0: '消失', 1: '迟钝', 2: '灵敏'}
thalamic_options = {'无': 1, '轻度': 2, '重度': 3}   # 训练数据只有1,2,3
thalamic_rev = {1: '无', 2: '轻度', 3: '重度'}

# 定义风险分级与治疗建议
def get_risk_grade(prob):
    """根据苏醒概率返回风险等级、颜色及治疗建议"""
    if prob >= 0.8:
        grade = "低风险（苏醒可能性高）"
        color = "green"
        advice = (
            "✅ **治疗建议**：患者苏醒概率较高，建议积极促醒治疗，包括：\n"
            "• 多模态感觉刺激（听觉、视觉、触觉）\n"
            "• 早期康复介入（被动活动、体位管理）\n"
            "• 神经调控治疗（如经颅磁刺激、右正中神经电刺激）\n"
            "• 与家属沟通，鼓励参与康复过程"
        )
    elif prob <= 0.2:
        grade = "高风险（苏醒可能性低）"
        color = "red"
        advice = (
            "⚠️ **治疗建议**：患者苏醒可能性较低，建议：\n"
            "• 与家属进行充分预后沟通，共同制定治疗目标\n"
            "• 评估是否存在可逆性损伤（如脑积水、癫痫持续状态）\n"
            "• 若符合脑死亡标准，按规范进行脑死亡判定\n"
            "• 考虑姑息治疗与安宁疗护"
        )
    else:
        grade = "中风险（苏醒可能性中等）"
        color = "orange"
        advice = (
            "🔍 **治疗建议**：患者苏醒概率中等，建议：\n"
            "• 密切监测神经功能变化（每日GCS、EEG）\n"
            "• 优化脑灌注与氧合，维持内环境稳定\n"
            "• 每72小时复评，动态调整治疗方案\n"
            "• 结合家属意愿，制定个体化治疗计划"
        )
    return grade, color, advice

# 输入面板
st.sidebar.header("输入特征")
input_data = {}
for feat in features:
    if feat == 'Pupillary_reflex':
        reflex_text = st.sidebar.selectbox('瞳孔反射', list(reflex_options.keys()))
        value = reflex_options[reflex_text]
    elif feat == 'thalamic':
        thalamic_text = st.sidebar.selectbox('丘脑损伤', list(thalamic_options.keys()))
        value = thalamic_options[thalamic_text]
    elif feat == 'mechanical_ventilation':
        value = st.sidebar.selectbox('机械通气', [0, 1], format_func=lambda x: '是' if x == 1 else '否')
    else:
        default_val = median_dict[feat]
        value = st.sidebar.number_input(feature_display[feat], value=float(default_val), step=0.1, format="%.2f")
    input_data[feat] = value

input_df = pd.DataFrame([input_data])

# 预测按钮
if st.sidebar.button("计算苏醒概率"):
    with st.spinner("正在计算，请稍候..."):
        prob = model.predict_proba(input_df)[0, 1]
        prob_percent = prob * 100

        # 获取风险分级与建议
        grade, grade_color, advice = get_risk_grade(prob)

        # 显示预测结果卡片
        st.subheader("📊 预测结果")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("苏醒概率", f"{prob_percent:.1f}%")
        col2.metric("预测类别", "苏醒" if prob >= 0.5 else "未苏醒")
        col3.metric("置信水平", "高" if prob > 0.8 or prob < 0.2 else "中")
        col4.metric("风险分级", grade)

        # 根据风险分级显示不同颜色的提示框
        if grade_color == "green":
            st.success(advice)
        elif grade_color == "orange":
            st.warning(advice)
        else:
            st.error(advice)

        # 显示输入特征值（中文）
        st.subheader("📋 输入特征值")
        display_dict = {}
        for feat in features:
            if feat == 'Pupillary_reflex':
                display_dict['瞳孔反射'] = reflex_rev[input_data[feat]]
            elif feat == 'thalamic':
                display_dict['丘脑损伤'] = thalamic_rev[input_data[feat]]
            elif feat == 'mechanical_ventilation':
                display_dict['机械通气'] = '是' if input_data[feat] == 1 else '否'
            else:
                display_dict[feature_display[feat]] = input_data[feat]
        st.json(display_dict)

        # SHAP 计算
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # ========== 1. SHAP 瀑布图 ==========
        st.subheader("🔍 模型解释 (SHAP 瀑布图)")
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

        # ========== 2. SHAP 贡献条形图 ==========
        st.subheader("📈 各特征贡献详情")
        shap_vals = shap_values[0]
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        sorted_feats = [features[i] for i in sorted_idx]
        sorted_vals = [shap_vals[i] for i in sorted_idx]
        colors = ['#e41a1c' if v > 0 else '#377eb8' for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(range(len(sorted_feats)), sorted_vals, color=colors, edgecolor='black')
        ax.set_yticks(range(len(sorted_feats)))
        ax.set_yticklabels([feature_display.get(f, f) for f in sorted_feats])
        ax.set_xlabel('SHAP value (贡献值，对数几率尺度)')
        ax.set_title('特征对预测的贡献（正值为促进苏醒，负值为抑制苏醒）')
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        for bar, val in zip(bars, sorted_vals):
            ax.text(bar.get_width() + (0.02 if val > 0 else -0.05), bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', ha='left' if val > 0 else 'right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # ========== 3. 详细文字报告 ==========
        st.subheader("📝 详细临床报告")
        report_lines = []
        report_lines.append(f"**患者苏醒概率预测：{prob_percent:.1f}%**")
        report_lines.append(f"**风险分级：{grade}**")
        report_lines.append("")
        report_lines.append("**各因素具体影响**：")

        for idx in sorted_idx:
            feat = features[idx]
            shap_val = shap_vals[idx]
            user_val = input_data[feat]
            mean_val = train_mean.get(feat, 0)
            direction = "增加" if shap_val > 0 else "降低"
            if feat == 'Delta_GCS':
                desc = f"ΔGCS 为 {user_val:.2f} 分/天，{direction}苏醒概率 {abs(shap_val):.2f}"
                if user_val > mean_val:
                    desc += "（高于平均水平）"
                elif user_val < mean_val:
                    desc += "（低于平均水平）"
            elif feat == 'P/F':
                desc = f"氧合指数为 {user_val:.0f} mmHg，{direction}苏醒概率 {abs(shap_val):.2f}"
                if user_val > mean_val:
                    desc += "（优于平均）"
                else:
                    desc += "（低于平均）"
            elif feat == 'mechanical_ventilation':
                status = "是" if user_val == 1 else "否"
                desc = f"机械通气：{status}，{direction}苏醒概率 {abs(shap_val):.2f}"
                if status == "是":
                    desc += "（不利因素）"
                else:
                    desc += "（有利因素）"
            elif feat == 'Pupillary_reflex':
                reflex_text = reflex_rev[user_val]
                desc = f"瞳孔反射：{reflex_text}，{direction}苏醒概率 {abs(shap_val):.2f}"
                if reflex_text in ['灵敏', '迟钝']:
                    desc += "（保护因素）"
                else:
                    desc += "（危险因素）"
            elif feat == 'thalamic':
                thalamic_text = thalamic_rev[user_val]
                desc = f"丘脑损伤：{thalamic_text}，{direction}苏醒概率 {abs(shap_val):.2f}"
                if thalamic_text == '无':
                    desc += "（有利因素）"
                else:
                    desc += "（不利因素）"
            else:
                desc = f"{feature_display.get(feat, feat)} 为 {user_val:.2f}，{direction}苏醒概率 {abs(shap_val):.2f}"
            report_lines.append(f"- {desc}")

        report_lines.append("")
        report_lines.append(advice)  # 复用前面定义的完整建议

        report_text = "\n".join(report_lines)
        st.markdown(report_text)

        # 提供下载按钮
        st.download_button(
            label="📄 下载完整报告 (文本)",
            data=report_text,
            file_name="苏醒预测报告.txt",
            mime="text/plain"
        )
else:
    st.info("👈 请在左侧边栏输入特征值，然后点击「计算苏醒概率」。")