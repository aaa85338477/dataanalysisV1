import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import io
import warnings
import requests
import json
import time

warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面与默认缓存配置
# ==========================================
st.set_page_config(page_title="出海游戏 | 智能归因与数据预估中台", page_icon="🧠", layout="wide")
st.title("🧠 游戏核心指标 AutoML 预估与 RCA 诊断系统")

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
    api_key_input = st.text_input("请输入中转站 API Key (sk-...)", type="password")
    st.markdown("---")
    st.markdown("""
    ### 🚀 系统功能演进
    
    **📍 V1.0 基础预估架构**
    - 30日留存与LTV曲线拟合。
    
    **📍 V2.0 动态适配与协同**
    - 万能映射自适应任意格式报表。
    - ChatBI 助理支持自然语言追问。

    **📍 V3.1 智能决策中枢**
    - AutoML 算法池：动态选用最优模型。
    - RCA 智能归因：皮尔逊相关性扫描。
    
    **📍 V4.0 无界限引擎 (当前版本)**
    - **中转站直连**：彻底告别 429 频控报错。
    - **gemini-3.1-flash-lite-preview 驱动**：极速推理与强大并发支撑。
    """)

# ==========================================
# 🌟 核心：基于中转站 API 的大模型安全调用函数
# ==========================================
def safe_generate(prompt, api_key, max_retries=3):
    """带自动重试机制的中转站 API 调用"""
    url = "https://api.bltcy.ai/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}', 
        'User-Agent': 'DMXAPI/1.0.0',
        'Content-Type': 'application/json'
    }
    
    payload = json.dumps({
        "model": "gemini-3.1-flash-lite-preview", # 可随时无缝切换为 gpt-4o 或 claude-3-5-sonnet
        "messages": [
            {
                "role": "system",
                "content": "你是一位资深的出海游戏数据分析师和发行运营专家。你的回答需要专业、克制且直击要害。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    for attempt in range(max_retries):
        try:
            # 发送请求，设置 60 秒超时防止死等
            response = requests.post(url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()  # 检查 HTTP 状态码
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_seconds = 5  # 中转站一般只需短时间冷却
                st.warning(f"⏳ API 网络波动，系统正在自动重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                error_details = response.text if 'response' in locals() else str(e)
                raise Exception(f"API 请求彻底失败: {error_details}")

# ==========================================
# 2. AutoML：动态算法池与核心预测引擎
# ==========================================
def power_curve(x, a, b): return a * np.power(x, b)
def log_curve(x, a, b): return a * np.log(x) + b
def exp_curve(x, a, b): return a * np.exp(b * x)
def linear_curve(x, a, b): return a * x + b

RETENTION_MODELS = {'幂函数(Power)': power_curve, '指数函数(Exponential)': exp_curve}
REVENUE_MODELS = {'对数函数(Logarithmic)': log_curve, '线性函数(Linear)': linear_curve, '幂函数(Power)': power_curve}

def predict_and_fill_automl(df, candidate_models, is_retention=False):
    df_calc = df.set_index(df.columns[0]).astype(float)
    filled_data = []
    best_model_for_latest = "未知" 
    
    for date, row in df_calc.iterrows():
        y = row.values
        x = np.arange(1, len(y) + 1)
        mask = ~np.isnan(y)
        x_train, y_train = x[mask], y[mask]
        
        if len(x_train) >= 3:
            best_mse = float('inf')
            best_y_final = y
            best_model_name = "未拟合"
            
            for model_name, model_func in candidate_models.items():
                try:
                    popt, _ = curve_fit(model_func, x_train, y_train, maxfev=10000)
                    y_pred_train = model_func(x_train, *popt)
                    mse = np.mean((y_train - y_pred_train) ** 2) 
                    
                    if mse < best_mse:
                        best_mse = mse
                        y_pred_full = model_func(x, *popt)
                        y_final = np.where(mask, y, y_pred_full)
                        if is_retention: y_final = np.clip(y_final, 0, 1)
                        else: y_final = np.maximum(y_final, 0)
                        
                        best_y_final = y_final
                        best_model_name = model_name
                except:
                    continue
            
            filled_data.append(best_y_final)
            best_model_for_latest = best_model_name
        else:
            filled_data.append(y)
            
    df_filled = pd.DataFrame(filled_data, index=df_calc.index, columns=df_calc.columns)
    df_filled = df_filled.round(4).reset_index()
    df_filled[df_filled.columns[0]] = pd.to_datetime(df_filled[df_filled.columns[0]]).dt.strftime('%Y/%m/%d')
    return df_filled, best_model_for_latest

def highlight_predicted_cells(df_filled, df_raw):
    styles = pd.DataFrame('', index=df_filled.index, columns=df_filled.columns)
    raw_values = df_raw.set_index(df_raw.columns[0]).astype(float)
    is_nan_mask = raw_values.isna()
    for col in is_nan_mask.columns:
        if col in styles.columns:
            styles[col] = np.where(is_nan_mask[col].values, 'background-color: #D3D3D3', '')
    return styles

# ==========================================
# 3. RCA：智能归因分析引擎 
# ==========================================
def calculate_rca_correlations(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty: return "数据不足，无法计算相关性。"
    
    corr_matrix = numeric_df.corr()
    strong_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                trend = "正相关(同涨同跌)" if corr_val > 0 else "负相关(此消彼长)"
                strong_pairs.append(f"- 【{col1}】与【{col2}】呈强烈的 **{trend}** (r={corr_val:.2f})")
                
    if not strong_pairs:
        return "近期各项指标相对独立，未发现显著的强关联特征。"
    return "\n".join(strong_pairs)


# ==========================================
# 4. 主页面：文件上传与动态映射面板
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
        st.warning("👈 请先在左侧边栏输入您的 API Key 以激活 AI 引擎。")
    else:
        # 确保传入的 Key 是干净的
        clean_api_key = api_key_input.strip()
        
        if "data_processed" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.data_processed = False
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = [] 

        if st.button("🚀 启动 AutoML 分析与归因", type="primary") or st.session_state.data_processed:
            st.session_state.data_processed = True
            
            try:
                # ----------------- 数据处理 -----------------
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
                            
                    # ----------------- AutoML 预估 -----------------
                    with st.spinner('📐 AutoML 引擎正在扫描最优算法并填补空值...'):
                        df_reg = tables.get(table_names[0])
                        df_pay = tables.get(table_names[1])
                        df_roi = tables.get(table_names[2])
                        df_ltv = tables.get(table_names[3])

                        df_reg_filled, best_reg_model = predict_and_fill_automl(df_reg, RETENTION_MODELS, is_retention=True)
                        df_pay_filled, best_pay_model = predict_and_fill_automl(df_pay, RETENTION_MODELS, is_retention=True)
                        df_roi_filled, best_roi_model = predict_and_fill_automl(df_roi, REVENUE_MODELS, is_retention=False)
                        df_ltv_filled, best_ltv_model = predict_and_fill_automl(df_ltv, REVENUE_MODELS, is_retention=False)
                        
                        st.session_state.df_reg = df_reg_filled
                        st.session_state.df_pay = df_pay_filled
                        st.session_state.df_roi = df_roi_filled
                        st.session_state.df_ltv = df_ltv_filled
                        st.session_state.table_names = table_names
                        
                        st.session_state.automl_models = {
                            "注册留存": best_reg_model, "付费留存": best_pay_model,
                            "ROI": best_roi_model, "LTV": best_ltv_model
                        }
                        
                    # ----------------- RCA 归因 -----------------
                    with st.spinner('🔍 正在计算大盘指标的底层相关性 (RCA)...'):
                        st.session_state.rca_context = calculate_rca_correlations(st.session_state.df_overall.tail(14))

                    with st.spinner('🎨 正在渲染高亮表格...'):
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            st.session_state.df_overall.to_excel(writer, sheet_name='整体数据', index=False)
                            df_reg_filled.style.apply(lambda x: highlight_predicted_cells(df_reg_filled, df_reg), axis=None).to_excel(writer, sheet_name=table_names[0], index=False)
                            df_pay_filled.style.apply(lambda x: highlight_predicted_cells(df_pay_filled, df_pay), axis=None).to_excel(writer, sheet_name=table_names[1], index=False)
                            df_roi_filled.style.apply(lambda x: highlight_predicted_cells(df_roi_filled, df_roi), axis=None).to_excel(writer, sheet_name=table_names[2], index=False)
                            df_ltv_filled.style.apply(lambda x: highlight_predicted_cells(df_ltv_filled, df_ltv), axis=None).to_excel(writer, sheet_name=table_names[3], index=False)
                        
                        st.session_state.excel_data = excel_buffer.getvalue()

                        latest_date = st.session_state.df_overall.iloc[-1][st.session_state.col_date]
                        st.session_state.latest_date = latest_date
                        
                        try:
                            d30_reg = df_reg_filled[df_reg_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_pay = df_pay_filled[df_pay_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_ltv = df_ltv_filled[df_ltv_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                            d30_roi = df_roi_filled[df_roi_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                        except:
                            d30_reg, d30_pay, d30_ltv, d30_roi = 0, 0, 0, 0

                        st.session_state.d30_metrics = {
                            "reg_retention": d30_reg, "pay_retention": d30_pay,
                            "ltv": d30_ltv, "roi": d30_roi
                        }

                # ----------------- UI 结果展示 -----------------
                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    st.download_button("📥 下载 AutoML 预测报表 (Excel)", data=st.session_state.excel_data, file_name="AI_Dynamic_Prediction_V4.xlsx")
                with col_info:
                    st.success(f"🤖 **AutoML 最新日 ({st.session_state.latest_date}) 模型：** LTV采用 [{st.session_state.automl_models['LTV']}]，留存采用 [{st.session_state.automl_models['注册留存']}]")

                st.markdown("---")
                tab1, tab2 = st.tabs(["📑 RCA 深度诊断报告", "💬 ChatBI 交互查询"])
                
                with tab1:
                    overall_data_md = st.session_state.df_overall.tail(7).to_markdown(index=False)
                    
                    if 'basic_report' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                        prompt_basic = f"""
                        【数据】：最新日期 {st.session_state.latest_date}。30日预估：注册留存 {st.session_state.d30_metrics['reg_retention'] * 100:.2f}%，付费留存 {st.session_state.d30_metrics['pay_retention'] * 100:.2f}%，LTV ${st.session_state.d30_metrics['ltv']:.2f}，ROI {st.session_state.d30_metrics['roi'] * 100:.2f}%。
                        附近7天大盘：\n{overall_data_md}\n
                        要求：采用要点式输出。分析大盘趋势与回本结论。控制在150字左右。
                        """
                        with st.spinner('🚀 正在提炼基础诊断...'):
                            # 调用中转站接口
                            st.session_state.basic_report = safe_generate(prompt_basic, clean_api_key)
                            st.session_state.deep_report = None 

                    st.info(f"📅 **诊断日期：** {st.session_state.latest_date}")
                    st.markdown("### 📊 基础数据速览")
                    st.markdown(st.session_state.basic_report)

                    # ----------------- 触发含有 RCA 上下文的深度诊断 -----------------
                    if st.session_state.deep_report is None:
                        if st.button("🔍 结合 RCA 相关性进行深度归因剖析？", type="primary"):
                            prompt_deep = f"""
                            请基于以下整体数据与 RCA（根因分析）相关性结果，输出一份专业的【深度业务剖析】。
                            
                            【大盘数据(近7天)】：\n{overall_data_md}\n
                            【RCA 底层指标相关性扫描结果】：\n{st.session_state.rca_context}\n
                            
                            要求摒弃套话，结合 RCA 线索，从以下三个维度展开：
                            1. **UA与初期付费**：评估首日付费率与新增ARPU。
                            2. **RCA 归因诊断**：重点解读提供的“强相关指标对”，解释同涨同跌的底层业务逻辑。
                            3. **落地调优建议**：给出至少 2 条能直接执行的建议。
                            """
                            with st.spinner('🔬 AI 正在解读 RCA 相关矩阵，撰写深度归因分析...'):
                                # 调用中转站接口
                                st.session_state.deep_report = safe_generate(prompt_deep, clean_api_key)
                                st.rerun()

                    if st.session_state.deep_report is not None:
                        st.markdown("---")
                        with st.expander("🛠️ 展开查看底层 RCA 相关系数矩阵", expanded=False):
                            st.markdown(st.session_state.rca_context)
                        st.markdown("### 🔬 RCA 深度业务归因剖析")
                        st.markdown(st.session_state.deep_report)
                        if st.button("收起深度报告"):
                            st.session_state.deep_report = None
                            st.rerun()

                    st.markdown("---")
                    if st.button("🚀 一键推送到飞书群"):
                        webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/20a3f60d-36f2-4a73-9879-3058c697a7b8"
                        msg_content = f"📅 【游戏数据预估与诊断日报】 {st.session_state.latest_date}\n\n📊 --- 基础数据速览 ---\n{st.session_state.basic_report}"
                        if st.session_state.deep_report:
                            msg_content += f"\n\n🔬 --- RCA 深度业务剖析 ---\n{st.session_state.deep_report}"
                        payload = {"msg_type": "text", "content": {"text": msg_content}}
                        try:
                            res = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
                            if res.status_code == 200: st.success("✅ 推送成功！")
                            else: st.error("❌ 推送失败")
                        except:
                            st.error("网络请求错误")

                with tab2:
                    st.subheader("💬 对话式数据查询")
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if user_query := st.chat_input("向全能 AI 提问..."):
                        with st.chat_message("user"): st.markdown(user_query)
                        st.session_state.chat_history.append({"role": "user", "content": user_query})

                        overall_md = st.session_state.df_overall.to_markdown(index=False)
                        reg_md = st.session_state.df_reg.tail(14).to_markdown(index=False)
                        ltv_md = st.session_state.df_ltv.tail(14).to_markdown(index=False)

                        full_context_data = f"【1. 大盘数据】\n{overall_md}\n【2. 注册留存(近14天)】\n{reg_md}\n【3. 净收LTV(近14天)】\n{ltv_md}"
                        chat_prompt = f"根据以下数据回答。\n{full_context_data}\n【用户问题】：{user_query}"

                        with st.chat_message("assistant"):
                            with st.spinner("🤔 gemini-3.1-flash-lite-preview 正在急速推理..."):
                                # 调用中转站接口
                                response_text = safe_generate(chat_prompt, clean_api_key)
                                st.markdown(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"处理错误：{e}")
