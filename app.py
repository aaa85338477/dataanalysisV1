# ==========================================
            # 5. 分层 AI 诊断报告请求与展示 (Session State 管理)
            # ==========================================
            st.header("🤖 AI 游戏运营总监 诊断报告")
            
            # 使用 session_state 记住当前上传的文件，避免重复点击时反复刷新
            if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
                st.session_state.last_file = uploaded_file.name
                st.session_state.basic_report = None
                st.session_state.deep_report = None

            # 提取近期数据与预估指标
            overall_data_md = df_overall.tail(7).to_markdown(index=False)
            latest_date = df_overall.iloc[-1]['日期']
            
            try:
                d30_reg_retention = df_reg_retention_filled[df_reg_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                d30_pay_retention = df_pay_retention_filled[df_pay_retention_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                d30_ltv = df_ltv_filled[df_ltv_filled.iloc[:, 0] == latest_date].iloc[0, -1]
                d30_roi = df_roi_filled[df_roi_filled.iloc[:, 0] == latest_date].iloc[0, -1]
            except:
                d30_reg_retention, d30_pay_retention, d30_ltv, d30_roi = 0, 0, 0, 0

            # --- 基础分析 (默认生成) ---
            prompt_basic = f"""
            你是一位海外游戏发行运营。请根据以下数据，输出一份极其精炼的【基础分析】。
            【数据】：最新日期 {latest_date}。30日预估：注册留存 {d30_reg_retention * 100:.2f}%，付费留存 {d30_pay_retention * 100:.2f}%，LTV ${d30_ltv:.2f}，ROI {d30_roi * 100:.2f}%。
            附近7天大盘：
            {overall_data_md}
            
            要求：不要任何废话和客套话，采用要点式(Bullet points)输出。
            1. 大盘趋势：近7天活跃、ARPU是涨是跌。
            2. 回本结论：基于预估ROI，这批用户30天能否回本？
            控制在 150 字左右。
            """

            model = genai.GenerativeModel('gemini-2.5-flash-lite')

            if st.session_state.basic_report is None:
                with st.spinner('🚀 正在提炼基础诊断...'):
                    st.session_state.basic_report = model.generate_content(prompt_basic).text

            st.info(f"📅 **诊断日期：** {latest_date}")
            st.markdown("### 📊 基础数据速览")
            st.markdown(st.session_state.basic_report)

            # --- 深度分析 (点击按钮后生成) ---
            if st.session_state.deep_report is None:
                # 只有还没生成深度报告时，才显示这个按钮
                if st.button("🔍 需要更深度的业务剖析？", type="primary"):
                    prompt_deep = f"""
                    你是一位资深的海外游戏发行运营专家。请基于以下整体数据与预估数据，输出一份专业的【深度业务剖析】。
                    
                    【数据】：最新日期 {latest_date}。30日预估：注册留存 {d30_reg_retention * 100:.2f}%，付费留存 {d30_pay_retention * 100:.2f}%，LTV ${d30_ltv:.2f}，ROI {d30_roi * 100:.2f}%。
                    近7天大盘：
                    {overall_data_md}
                    
                    要求摒弃套话，直击核心痛点，请着重从以下三个专业维度展开：
                    1. **UA（用户获取）友好度与买量策略**：评估首日付费率与新增ARPU，分析当前数据形态对买量团队是否友好，初期的出价空间如何。
                    2. **留存与变现的“双层结构”**：结合预估的注册留存与付费留存的衰减情况，诊断游戏当前的“获客-留存”双层结构是否健康，指出长线漏斗可能存在的流失风险。
                    3. **LTV 与 ROI 压力诊断**：结合 30 日 ROI 预估值，评估当前商业化变现深度的压力，并给出至少 2 条能直接落地的针对性调优建议。
                    """
                    with st.spinner('🔬 正在结合 UA 与双层留存结构生成深度剖析...'):
                        st.session_state.deep_report = model.generate_content(prompt_deep).text
                        # 生成完毕后，强制页面重新加载一次以显示报告，并隐藏按钮
                        st.rerun() 

            if st.session_state.deep_report is not None:
                st.markdown("---")
                st.markdown("### 🔬 深度业务剖析")
                st.markdown(st.session_state.deep_report)
                
                # 可选：提供一个收起深度报告的按钮
                if st.button("收起深度报告"):
                    st.session_state.deep_report = None
                    st.rerun()
