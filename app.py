import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import google.generativeai as genai
import io
import warnings

# 忽略计算过程中的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="出海游戏 | 数据预估与诊断系统", page_icon="📈", layout="wide")
st.title("📈 游戏核心指标自动预估与 AI 诊断系统")
st.markdown("上传从后台导出的固定格式报表，自动完成 **30日留存/LTV/ROI** 的曲线拟合并填补空值，同时生成 AI 诊断报告。")

# ==========================================
# 2. 侧边栏：配置 API Key
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key_input = st.text_input("请输入 Gemini API Key", type="password", help="为了安全，Key仅在当前网页生效，不会被保存。")
    st.markdown("---")
    st.markdown("### 工具说明")
    st.markdown("- **留存预测模型**：幂函数衰减曲线\n- **变现预测模型**：对数增长曲线\n- **AI 诊断大模型**：Gemini 2.5 Flash-Lite")
    
    if api_key_input:
        genai.configure(api_key=api_key_input)

# ==========================================
# 3. 核心预测与样式函数定义
# ==========================================
def power_curve(x, a, b): 
    return a * np.power(x, b)

def log_curve(x, a, b): 
    return a * np.log(x) + b

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
                
                if is_retention:
                    y_final = np.clip(y_final, 0, 1)
                else:
                    y_final = np.maximum(y_final, 0)
                filled_data.append(y_final)
            except:
                filled_data.append(y)
        else:
            filled_data.append(y)
            
    df_filled = pd.DataFrame(filled_data, index=df_calc.index, columns=df_calc.columns)
    df_filled = df_filled.round(4).reset_index()
    # 格式化日期，去除 0:00:00
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
# 4. 主页面：文件上传与业务流处理
# ==========================================
uploaded_file = st.file_uploader("📂 请上传游戏分天业务数据 (Excel格式)", type=["xlsx", "xls"])

if uploaded_file is not None:
    if not api_key_input:
        st.warning("👈 请先在左侧边栏输入您的 Gemini API Key 才能启动诊断引擎。")
    else:
        st.success("✅ 文件读取成功！正在启动自动化分析工作流...")
        
        try:
            with st.spinner('⏳ 正在解析业务表单与切分数据...'):
                # 读取 Sheet1 整体数据
                df_overall = pd.read_excel(uploaded_file, sheet_name=0)
                df_overall = df_overall[df_overall['日期'] != '汇总'].copy()
                df_overall['日期'] = pd.to_datetime(df_overall['日期']).dt.strftime('%Y/%m/%d')
                
                # 读取 Sheet2 分天数据并智能切分
                raw_daily = pd.read_excel(uploaded_file, sheet_name=1, header=None)
                header_indices = raw_daily[raw_daily[0].isin(['按天', '按天（净收）'])].index.tolist()
                
                tables = {}
                table_names = ['注册留存', '付费留存', '净收ROI', '净收LTV']
                for i in range(len(header_indices)):
                    start_idx = header_indices[i]
                    end_idx = header_indices[i+1] if i + 1 < len(header_indices) else len(raw_daily)
                    temp_df = raw_daily.iloc[start_idx:end_idx].copy()
                    temp_df.columns = temp_df.iloc[0]
                    temp_df = temp_df[1:].dropna(how='all').reset_index(drop=True)
                    tables[table_names[i]] = temp_df
                    
                df_reg_retention = tables['注册留存']
                df_pay_retention = tables['付费留存']
                df_roi = tables['净收ROI']
                df_ltv = tables['净收LTV']

            with st.spinner('📐 正在运行数学模型，预估 30 天核心指标并填补空值...'):
                df_reg_retention_filled = predict_and_fill(df_reg_retention, power_curve, is_retention=True)
                df_pay_retention_filled = predict_and_fill(df_pay_retention, power_curve, is_retention=True)
                df_ltv_filled = predict_and_fill(df_ltv, log_curve, is_retention=False)
                df_roi_filled = predict_and_fill(df_roi, log_curve, is_retention=False)

            with st.spinner('🎨 正在渲染高亮表格并生成 Excel...'):
                # 使用 BytesIO 在内存中生成 Excel，不落盘直接供下载
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_overall.to_excel(writer, sheet_name='整体数据', index=False)
                    df_reg_retention_filled.style.apply(lambda x: highlight_predicted_cells(df_reg_retention_filled, df_reg_retention), axis=None).to_excel(writer, sheet_name='注册留存(含预测)', index=False)
                    df_pay_retention_filled.style.apply(lambda x: highlight_predicted_cells(df_pay_retention_filled, df_pay_retention), axis=None).to_excel(writer, sheet_name='付费留存(含预测)', index=False)
                    df_roi_filled.style.apply(lambda x: highlight_predicted_cells(df_roi_filled, df_roi), axis=None).to_excel(writer, sheet_name='净收ROI(含预测)', index=False)
                    df_ltv_filled.style.apply(lambda x: highlight_predicted_cells(df_ltv_filled, df_ltv), axis=None).to_excel(writer, sheet_name='净收LTV(含预测)', index=False)
                
                excel_data = excel_buffer.getvalue()

            st.success("🎉 数据处理完毕！点击下方按钮即可获取填补好灰底的预测表格。")
            
            # 渲染下载按钮
            st.download_button(
                label="📥 下载预测报表 (AI_Prediction_Report.xlsx)",
                data=excel_data,
                file_name="AI_Prediction_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("---")
            
            # ==========================================
            # 5. AI 诊断报告请求与展示
            # ==========================================
            st.header("🤖 AI 游戏运营总监 诊断报告")
            
            with st.spinner('🚀 正在呼叫 Gemini 模型撰写业务诊断...'):
                overall_data_md = df_overall.tail(7).to_markdown(index=False)
                latest_date = df_overall.iloc[-1]['日期']
                
                try:
                    d30_reg_retention = df_reg_retention_filled[df_reg_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                    d30_pay_retention = df_pay_retention_filled[df_pay_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                    d30_ltv = df_ltv_filled[df_ltv_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                    d30_roi = df_roi_filled[df_roi_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                except:
                    d30_reg_retention, d30_pay_retention, d30_ltv, d30_roi = 0, 0, 0, 0
                    
                prompt = f"""
                你是一位资深的海外游戏发行运营总监。请审阅最新业务数据并给出诊断。
                
                【近期整体运营数据（近7天）】
                {overall_data_md}
                
                【最新一日（{latest_date}）新增用户的 30天长线预估】
                - 预估 30日注册留存率：{d30_reg_retention * 100:.2f}%
                - 预估 30日付费留存率：{d30_pay_retention * 100:.2f}%
                - 预估 30日净收 LTV：${d30_ltv:.2f}
                - 预估 30日净收 ROI：{d30_roi * 100:.2f}%
                
                请输出分析报告，包含：
                1. 📈 **大盘健康度诊断**（分析注册、ARPU、付费率等核心趋势）
                2. 🔮 **长线变现与回本分析**（基于预估指标评估模型压力，ROI超100%为回本）
                3. 💡 **运营与调优建议**（给出3条可落地的发行干货建议）
                """

                # 调用大模型
                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                response = model.generate_content(prompt)
                
                # 在网页容器中展示结果
                st.info(f"📅 **诊断日期：** {latest_date}")
                st.markdown(response.text)

        except Exception as e:
            st.error(f"处理过程中发生错误，请检查表单格式是否匹配。错误详情：{e}")
