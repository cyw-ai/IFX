    # ifx_predictor
    import streamlit as st
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import traceback
    
    # 配置页面
    st.set_page_config(
        page_title="IFX药物浓度预测系统",
        page_icon="💊",
        layout="wide"
    )
    
    # 加载预训练组件
    @st.cache_resource
    def load_components():
        try:
            model = joblib.load(r'C:\Users\cyw\ifx_ensemble_model.pkl')
            scaler = joblib.load(r'C:\Users\cyw\ifx_scaler.pkl')
            return model, scaler
        except FileNotFoundError:
            st.error("模型文件未找到，请确认以下文件存在：\n- ifx_ensemble_model.pkl\n- ifx_scaler.pkl")
            return None, None
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}\n{traceback.format_exc()}")
            return None, None
    
    # 初始化session状态
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    
    # 医疗特征定义（包含完整的数据类型和临床约束）
    features = {
        'Fg': ('纤维蛋白原(g/L)', (1.8, 4.5)),
        'CDAI': ('Crohn\'s疾病活动指数', (0, 600)),
        'APTT': ('活化部分凝血活酶时间(秒)', 20.0),
        'eGFR': ('估算肾小球滤过率(ml/min/1.73m²)', (30.0, 120.0)),
        'D-Dimer': ('D-二聚体(mg/L)', 0.5),
        'ALB': ('白蛋白(g/L)', (35.0, 55.0)),
        'Dose': ('药物剂量(mg)', (200.0, 1000.0)),  # 扩展剂量范围
        'WBC': ('白细胞计数(×10⁹/L)', (4.0, 10.0)),
        'Age': ('年龄', (18.0, 80.0)),
        'AST': ('谷草转氨酶(U/L)', (8.0, 40.0)),
        'ALT': ('谷丙转氨酶(U/L)', (7.0, 56.0)),
        'ADA': ('腺苷脱氨酶(U/L)', 4.0),
        'Lesion site': ('病变部位数', (1, 5))  # 整数类型处理
    }
    
    # 侧边栏输入界面
    with st.sidebar:
        st.header("患者信息输入")
        
        inputs = {}
        for key, (desc, default) in features.items():
            # 特殊处理整数型病变部位数
            if key == 'Lesion site':
                min_val, max_val = default
                current_val = (min_val + max_val) // 2
                inputs[key] = st.slider(
                    label=desc,
                    min_value=min_val,
                    max_value=max_val,
                    value=current_val,
                    step=1,
                    format="%d",
                    key=key
                )
            # 处理范围型参数
            elif isinstance(default, tuple):
                min_val, max_val = default
                current_val = (min_val + max_val) / 2
                inputs[key] = st.slider(
                    label=desc,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(current_val),
                    key=key
                )
            # 处理单值型参数
            else:
                inputs[key] = st.number_input(
                    label=desc,
                    value=float(default),
                    min_value=0.0,
                    format="%.1f",
                    key=key
                )
    
    # 主界面布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 输入参数展示
        st.subheader("输入参数概览")
        input_df = pd.DataFrame([inputs]).T.reset_index()
        input_df.columns = ['参数', '输入值']
        st.dataframe(
            input_df, 
            use_container_width=True,
            column_config={
                "参数": st.column_config.TextColumn(width="medium"),
                "输入值": st.column_config.NumberColumn(
                    format="%.2f",
                    width="small"
                )
            }
        )
    
    with col2:
        # 预测功能区域
        st.subheader("预测操作")
        if st.button("开始预测", use_container_width=True, type="primary"):
            model, scaler = load_components()
            
            if model and scaler:
                try:
                    # 严格保证特征顺序
                    feature_order = [
                        'Fg', 'CDAI', 'APTT', 'eGFR', 'D-Dimer',
                        'ALB', 'Dose', 'WBC', 'Age', 'AST',
                        'ALT', 'ADA', 'Lesion site'
                    ]
                    
                    # 转换为DataFrame并转换数据类型
                    input_data = pd.DataFrame([inputs])[feature_order].astype(float)
                    
                    # 标准化处理
                    scaled_data = scaler.transform(input_data)
                    
                    # 执行预测
                    proba = model.predict_proba(scaled_data)[0]
                    prediction = model.predict(scaled_data)[0]
                    
                    # 显式获取对应概率
                    probability = proba[1] if prediction == 1 else proba[0]
                    
                    # 存储结果
                    st.session_state.prediction = {
                        'class': "治疗浓度 (≥3 μg/ml)" if prediction == 1 else "低浓度 (<3 μg/ml)",
                        'probability': probability
                    }
                    
                except Exception as e:
                    st.error(f"预测错误：{str(e)}\n详细追踪：\n{traceback.format_exc()}")
    
    # 显示预测结果
    if st.session_state.prediction:
        st.markdown("---")
        result = st.session_state.prediction
        
        st.subheader("预测结果")
        # 动态样式显示
        if result['class'].startswith("治疗"):
            status_icon = "✅"
            color = "#28a745"
            advice_icon = "🩺"
        else:
            status_icon = "⚠️"
            color = "#dc3545"
            advice_icon = "📉"
        
        st.markdown(f"""
        <div style="border-radius: 10px; padding: 20px; background-color: {color}10; border-left: 5px solid {color}; margin: 20px 0;">
            <h3 style="color:{color}; margin-top:0;">
                {status_icon} {result['class']}
            </h3>
            <p style="font-size: 16px;">置信度：<strong>{result['probability']*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 临床建议卡片
        st.markdown(f"""
        <div style="border-radius: 10px; padding: 15px; background-color: #f8f9fa; margin: 20px 0;">
            <h4 style="color:#2c3e50; margin-top:0;">{advice_icon} 临床建议指南</h4>
            {"""
            ✅ <strong>治疗浓度指导：</strong>
            - 维持当前治疗方案
            - 每8周监测药物浓度
            - 常规炎症指标监测
            
            ⚠️ <strong>低浓度应对策略：</strong>
            - 检查用药依从性
            - 考虑剂量优化（+10-20%）
            - 检测抗药抗体
            - 缩短监测间隔（2-4周）
            """ if result['class'].startswith("治疗") else ""}
        </div>
        """, unsafe_allow_html=True)
    
    # 侧边栏增强说明
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        **临床操作指南**
        
        📌 参数采集标准：
        1. CDAI：评估前1周数据
        2. 生化指标：用药前空腹采集
        3. 病变部位：影像学确认
        
        ⏳ 最佳实践：
        - 首次用药后第14天监测
        - 剂量调整后需重新评估
        - 合并感染时暂缓评估
        """)
    
    # 启动说明
    if __name__ == '__main__':
        # 开发环境调试用
        import os
        if not os.path.exists(r'C:\Users\cyw\ifx_ensemble_model.pkl'):
            st.warning("开发提示：需要训练模型文件")
    

