import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import google.generativeai as genai
import io
import warnings
import requests
import json

# 忽略计算过程中的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面与默认缓存配置
# ==========================================
st.set_page_config(page_title="出海游戏 | 数据预估与诊断系统", page_icon="📈", layout="wide")
st.title("📈 游戏核心指标自动预估与 AI 诊断系统")

# 初始化缓存映射字典 (缓存用户习惯)
default_mapping = {
    'col_date': '日期',
    'exclude_word': '汇总',
    'split_keys': '按天,按天（净收）',
    'block_order': '注册留存,付费留存,净收ROI,净收LTV'
}
for k, v in default_mapping.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key_input = st.text_input("请输入 Gemini API Key", type="password")
    st.markdown("---")
    st.markdown("### 工具说明\n- **留存预测**：幂函数\n- **变现预测**：对数函数")
    if api_key_input:
        genai.configure(api_key=api_key_input)

# ==========================================
# 2. 核心数学预估与样式函数
# ==========================================
def power_curve(x, a, b): return a * np.power(x, b)
def log_curve(x, a, b): return a * np.log(x) + b

def predict_and_fill(df, model_func, is_retention=False):
    df_calc = df.set_index(df.columns[0]).astype(float)
    filled_data = []
    for date, row in df_calc.iterrows():
        y = row.values
        x = np.arange(1, len(y) + 1)
        mask = ~np.isnan(y)
        x_train, y_train = x[mask], y[mask]
        if len(x_train) >= 3:
            try:
                popt, _ = curve_fit(model_func, x_train, y_train, maxfev=10000)
                y_pred = model_func(x, *popt)
                y_final = np.where(mask, y, y_pred)
                if is_retention: y_final = np.clip(y_final, 0, 1)
                else: y_final = np.maximum(y_final, 0)
                filled_data.append(y_final)
            except:
                filled_data.append(y)
        else:
            filled_data.append(y)
    df_filled = pd.DataFrame(filled_data, index=df_calc.index, columns=df_calc.columns)
    df_filled = df_filled.round(4).reset_index()
    df_filled[df_filled.columns[0]] = pd.to_datetime(df_filled[df_filled.columns[0]]).dt.strftime('%Y/%m/%d')
    return df_filled

def highlight_predicted_cells(df_filled, df_raw):
    styles = pd.DataFrame('', index=df_filled.index, columns=df_filled.columns)
    raw_values = df_raw.set_index(df_raw.columns[0]).astype(float)
    is_nan_mask = raw_values.isna()
    for col in is_nan_mask.columns:
        if col in styles.columns:
            styles[col] = np.where(is_nan_mask[col].values, 'background-color: #D3D3D3', '')
    return styles

# ==========================================
# 3. 主页面：文件上传与动态映射面板
# ==========================================
uploaded_file = st.file_uploader("📂 请上传游戏分天业务数据 (Excel格式)", type=["xlsx", "xls"])

if uploaded_file is not None:
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    
    with st.expander("🛠️ 报表字段动态映射 (系统会自动记住选择)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 整体大盘数据配置**")
            sheet_overall = st.selectbox("大盘数据所在 Sheet", sheet_names, index=0)
            df_preview = pd.read_excel(uploaded_file, sheet_name=sheet_overall, nrows=0)
            available_cols = list(df_preview.columns)
            default_date_idx = available_cols.index(st.session_state.col_date) if st.session_state.col_date in available_cols else 0
            st.selectbox("【日期】所在列名", available_cols, index=default_date_idx, key="col_date")
            st.text_input("需剔除的无效行(如: 汇总)", key="exclude_word")

        with col2:
            st.markdown("**📅 分天预估数据配置**")
            sheet_daily = st.selectbox("分天数据所在 Sheet", sheet_names, index=1 if len(sheet_names)>1 else 0)
            st.text_input("堆叠表的切割关键字(英文逗号分隔)", key="split_keys")
            st.text_input("从上到下的表格顺序(英文逗号分隔)", key="block_order")

    if not api_key_input:
        st.warning("👈 请先在左侧边栏输入您的 Gemini API Key。")
    else:
        if "data_processed" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.data_processed = False
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = [] 

        if st.button("🚀 开始分析与预估", type="primary") or st.session_state.data_processed:
            st.session_state.data_processed = True
            
            try:
                # ==========================================
                # 4. 数据处理与预估
                # ==========================================
                if 'df_overall' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                    with st.spinner('⏳ 正在依据自定义映射解析数据...'):
                        df_overall = pd.read_excel(uploaded_file, sheet_name=sheet_overall)
                        df_overall = df_overall[df_overall[st.session_state.col_date] != st.session_state.exclude_word].copy()
                        df_overall[st.session_state.col_date] = pd.to_datetime(df_overall[st.session_state.col_date]).dt.strftime('%Y/%m/%d')
                        st.session_state.df_overall = df_overall
                        
                        raw_daily = pd.read_excel(uploaded_file, sheet_name=sheet_daily, header=None)
                        split_keywords = [k.strip() for k in st.session_state.split_keys.split(',')]
                        header_indices = raw_daily[raw_daily[0].isin(split_keywords)].index.tolist()
                        table_names = [n.strip() for n in st.session_state.block_order.split(',')]
                        tables = {}
                        
                        for i in range(len(header_indices)):
                            start_idx = header_indices[i]
                            end_idx = header_indices[i+1] if i + 1 < len(header_indices) else len(raw_daily)
                            temp_df = raw_daily.iloc[start_idx:end_idx].copy()
                            temp_df.columns = temp_df.iloc[0]
                            temp_df = temp_df[1:].dropna(how='all').reset_index(drop=True)
                            current_name = table_names[i] if i < len(table_names) else f"未知表格_{i}"
                            tables[current_name] = temp_df
                            
                    with st.spinner('📐 正在运行数学模型填补空值...'):
                        df_reg_retention = tables.get(table_names[0])
                        df_pay_retention = tables.get(table_names[1])
                        df_roi = tables.get(table_names[2])
                        df_ltv = tables.get(table_names[3])

                        df_reg_retention_filled = predict_and_fill(df_reg_retention, power_curve, is_retention=True)
                        df_pay_retention_filled = predict_and_fill(df_pay_retention, power_curve, is_retention=True)
                        df_ltv_filled = predict_and_fill(df_ltv, log_curve, is_retention=False)
                        df_roi_filled = predict_and_fill(df_roi, log_curve, is_retention=False)
                        
                        # 把分天预测数据也存入 Session State，供 ChatBI 使用
                        st.session_state.df_reg = df_reg_retention_filled
                        st.session_state.df_pay = df_pay_retention_filled
                        st.session_state.df_roi = df_roi_filled
                        st.session_state.df_ltv = df_ltv_filled
                        st.session_state.table_names = table_names

                    with st.spinner('🎨 正在渲染高亮表格...'):
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            st.session_state.df_overall.to_excel(writer, sheet_name='整体数据', index=False)
                            df_reg_retention_filled.style.apply(lambda x: highlight_predicted_cells(df_reg_retention_filled, df_reg_retention), axis=None).to_excel(writer, sheet_name=table_names[0], index=False)
                            df_pay_retention_filled.style.apply(lambda x: highlight_predicted_cells(df_pay_retention_filled, df_pay_retention), axis=None).to_excel(writer, sheet_name=table_names[1], index=False)
                            df_roi_filled.style.apply(lambda x: highlight_predicted_cells(df_roi_filled, df_roi), axis=None).to_excel(writer, sheet_name=table_names[2], index=False)
                            df_ltv_filled.style.apply(lambda x: highlight_predicted_cells(df_ltv_filled, df_ltv), axis=None).to_excel(writer, sheet_name=table_names[3], index=False)
                        
                        st.session_state.excel_data = excel_buffer.getvalue()

                        latest_date = st.session_state.df_overall.iloc[-1][st.session_state.col_date]
                        st.session_state.latest_date = latest_date
                        
                        try:
                            d30_reg_retention = df_reg_retention_filled[df_reg_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_pay_retention = df_pay_retention_filled[df_pay_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_ltv = df_ltv_filled[df_ltv_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_roi = df_roi_filled[df_roi_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                        except:
                            d30_reg_retention, d30_pay_retention, d30_ltv, d30_roi = 0, 0, 0, 0

                        st.session_state.d30_metrics = {
                            "reg_retention": d30_reg_retention, "pay_retention": d30_pay_retention,
                            "ltv": d30_ltv, "roi": d30_roi
                        }

                st.success("🎉 数据处理完毕！")
                st.download_button(
                    label="📥 下载动态预测报表 (Excel)",
                    data=st.session_state.excel_data,
                    file_name="AI_Dynamic_Prediction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # ==========================================
                # 5. AI 分层诊断报告 & 6. 飞书推送 & 7. ChatBI
                # ==========================================
                st.markdown("---")
                tab1, tab2 = st.tabs(["📑 标准业务诊断", "💬 ChatBI 交互查询"])
                
                with tab1:
                    st.header("🤖 AI 游戏运营总监 诊断报告")
                    overall_data_md = st.session_state.df_overall.tail(7).to_markdown(index=False)
                    
                    if 'basic_report' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                        prompt_basic = f"""
                        你是一位海外游戏发行运营。请根据以下数据，输出一份极其精炼的【基础分析】。
                        【数据】：最新日期 {st.session_state.latest_date}。30日预估：注册留存 {st.session_state.d30_metrics['reg_retention'] * 100:.2f}%，付费留存 {st.session_state.d30_metrics['pay_retention'] * 100:.2f}%，LTV ${st.session_state.d30_metrics['ltv']:.2f}，ROI {st.session_state.d30_metrics['roi'] * 100:.2f}%。
                        附近7天大盘：\n{overall_data_md}\n
                        要求：采用要点式输出。分析大盘趋势与回本结论。控制在150字左右。
                        """
                        model = genai.GenerativeModel('gemini-2.5-flash-lite')
                        with st.spinner('🚀 正在提炼基础诊断...'):
                            st.session_state.basic_report = model.generate_content(prompt_basic).text
                            st.session_state.deep_report = None 

                    st.info(f"📅 **诊断日期：** {st.session_state.latest_date}")
                    st.markdown("### 📊 基础数据速览")
                    st.markdown(st.session_state.basic_report)

                    if st.session_state.deep_report is None:
                        if st.button("🔍 需要更深度的业务剖析？", type="primary"):
                            prompt_deep = f"基于最新数据{st.session_state.latest_date}，请从UA买量、留存变现双层结构、长线LTV三个维度进行专业发行深度剖析。大盘数据如下：\n{overall_data_md}"
                            with st.spinner('🔬 正在生成深度剖析...'):
                                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                                st.session_state.deep_report = model.generate_content(prompt_deep).text
                                st.rerun()

                    if st.session_state.deep_report is not None:
                        st.markdown("---")
                        st.markdown("### 🔬 深度业务剖析")
                        st.markdown(st.session_state.deep_report)
                        if st.button("收起深度报告"):
                            st.session_state.deep_report = None
                            st.rerun()

                    st.markdown("---")
                    if st.button("🚀 一键推送到飞书群"):
                        webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/20a3f60d-36f2-4a73-9879-3058c697a7b8"
                        msg_content = f"📅 【游戏数据预估与诊断日报】 {st.session_state.latest_date}\n\n📊 --- 基础数据速览 ---\n{st.session_state.basic_report}"
                        if st.session_state.deep_report:
                            msg_content += f"\n\n🔬 --- 深度业务剖析 ---\n{st.session_state.deep_report}"
                        payload = {"msg_type": "text", "content": {"text": msg_content}}
                        headers = {'Content-Type': 'application/json'}
                        try:
                            res = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
                            if res.status_code == 200: st.success("✅ 推送成功！")
                            else: st.error("❌ 推送失败")
                        except:
                            st.error("网络请求错误")

                with tab2:
                    st.subheader("💬 对话式数据查询")
                    st.markdown("可以直接向 AI 询问有关当前大盘表格或预测表格的数据细节。")
                    
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if user_query := st.chat_input("例如：帮我对比最后两天的首日ARPU，以及7月24日目前的30日预测LTV是多少？"):
                        with st.chat_message("user"):
                            st.markdown(user_query)
                        st.session_state.chat_history.append({"role": "user", "content": user_query})

                        # 整合大盘数据与分天预估数据 (取最近14天避免Token超限)
                        overall_md = st.session_state.df_overall.to_markdown(index=False)
                        reg_md = st.session_state.df_reg.tail(14).to_markdown(index=False)
                        pay_md = st.session_state.df_pay.tail(14).to_markdown(index=False)
                        roi_md = st.session_state.df_roi.tail(14).to_markdown(index=False)
                        ltv_md = st.session_state.df_ltv.tail(14).to_markdown(index=False)

                        full_context_data = f"""
                        【1. 大盘整体核心数据】
                        {overall_md}
                        
                        【2. 核心预估指标分天明细 (最近14天)】
                        --- {st.session_state.table_names[0]} ---
                        {reg_md}
                        --- {st.session_state.table_names[1]} ---
                        {pay_md}
                        --- {st.session_state.table_names[2]} ---
                        {roi_md}
                        --- {st.session_state.table_names[3]} ---
                        {ltv_md}
                        """
                        
                        chat_prompt = f"""
                        你现在是一位资深的出海游戏数据分析师。请仔细阅读以下我刚刚上传的游戏业务数据（包含大盘与核心分天预估数据），并回答我的提问。
                        
                        {full_context_data}
                        
                        【用户的具体问题】：
                        {user_query}
                        
                        回答要求：
                        1. 语气像专业的数据分析同事。
                        2. 必须结合提供的表格数据进行精确的数值计算或对比。
                        3. 如果问题超出了所给的数据范围，请如实告知无法计算。
                        """

                        with st.chat_message("assistant"):
                            with st.spinner("🤔 AI 分析师正在计算数据..."):
                                chat_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                                response = chat_model.generate_content(chat_prompt)
                                st.markdown(response.text)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"处理过程中发生错误，请检查表单映射。错误详情：{e}")
