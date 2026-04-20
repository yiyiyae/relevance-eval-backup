import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚙️ 核心配置与引擎导入
# ==========================================
try:
    # 确保 search_rel_eval.py 中已经按之前的建议增加了 learn 的 source 参数
    # 以及将 auto_extract_rule 缩进到了 RagSearchEvalAgent 类内部
    from search_rel_eval import RagSearchEvalAgent, API_KEY, TAG_DICT, OUTPUT_CSV, MODEL_NAME
except ImportError:
    st.error("❌ 核心引擎加载失败：请检查 search_rel_eval.py 是否在同级目录。")
    st.stop()

# 页面配置
st.set_page_config(page_title="SearchEval Pro | AI 质量中心", layout="wide")

# 自定义 CSS 样式
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E9ECEF; }
    .review-card { background: white; padding: 25px; border-radius: 12px; border-left: 8px solid #EF4444; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .logic-box { background: #f1f3f5; padding: 12px; border-radius: 8px; border-left: 4px solid #adb5bd; margin-top: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    # 这里的路径需与 search_rel_eval.py 中的一致
    return RagSearchEvalAgent("./agent_vector_db")

agent = load_agent()

# --- 初始化 Session State ---
if 'results_data' not in st.session_state: st.session_state.results_data = []
if 'processed' not in st.session_state: st.session_state.processed = False
if 'rule_draft' not in st.session_state: st.session_state.rule_draft = ""
if 'smart_summary' not in st.session_state: st.session_state.smart_summary = ""
if 'active_tag' not in st.session_state: st.session_state.active_tag = None
if 'last_id' not in st.session_state: st.session_state.last_id = None

# --- 辅助函数：断点接续逻辑 ---
def load_existing_progress():
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
            history = []
            for _, row in df.iterrows():
                history.append({
                    "Query": str(row.get('query', '')),
                    "Title": str(row.get('result_title', '')),
                    "Summary": str(row.get('result_summary', '')),
                    "Tag": str(row.get('llm_tag', 'Error')),
                    "Confidence": int(row.get('confidence', 0)),
                    "Reason": str(row.get('llm_reason', '')),
                    "Reviewed": True  # 已在结果文件中的视为已复核
                })
            st.session_state.results_data = history
            st.session_state.processed = True
            return True
        except Exception as e:
            st.error(f"加载失败: {e}")
            return False
    return False

# --- 侧边栏：控制中心 ---
with st.sidebar:
    st.title("🛡️ 决策控制中心")
    st.caption("PM/策略专家的人机协作边界")
    st.divider()
    
    confidence_threshold = st.slider("人工复核触发阈值", 0, 100, 95)
    st.info(f"💡 置信度 < {confidence_threshold} 的数据将拦截至复核中心。")
    
    st.divider()
    # 按钮：加载昨日进度
    if st.button("📂 恢复断点：加载历史进度", use_container_width=True):
        if load_existing_progress():
            st.success("✅ 进度已成功恢复")
            st.rerun()
        else:
            st.warning("📭 未找到历史输出文件。")

    if st.button("🗑️ 彻底清空当前任务状态", use_container_width=True):
        st.session_state.results_data = []
        st.session_state.processed = False
        st.session_state.rule_draft = ""
        st.session_state.active_tag = None
        st.session_state.last_id = None
        st.rerun()

# --- 主界面 Tabs ---
tab_eval, tab_human, tab_admin = st.tabs(["🚀 自动化诊断", "👨‍🏫 专家复核中心", "🧠 记忆治理"])

# ==========================================
# Tab 1: 自动化诊断 (并发加速 + 多维看板)
# ==========================================
with tab_eval:
    if not st.session_state.processed:
        st.subheader("待评估任务导入")
        file = st.file_uploader("上传 CSV 评测文件", type="csv")
        if file:
            df_in = pd.read_csv(file, encoding="utf-8-sig")
            if st.button("🔥 启动 AI 引擎并发诊断", type="primary"):
                p_bar = st.progress(0)
                p_text = st.empty()
                results = []
                df_list = df_in.to_dict("records")
                total = len(df_list)

                # 并发执行逻辑
                def process_row_task(row):
                    q, t, s = [str(row.get(k, "")) for k in ['query', 'result_title', 'result_summary']]
                    eval_res = agent.evaluate(q, t, s)
                    return {
                        "Query": q, "Title": t, "Summary": s,
                        "Tag": eval_res.get("bad_case_tag", "Error"),
                        "Confidence": int(eval_res.get("confidence_score", 0)),
                        "Reason": eval_res.get("thought_process", ""),
                        "Reviewed": False
                    }

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(process_row_task, r): i for i, r in enumerate(df_list)}
                    for i, future in enumerate(as_completed(futures)):
                        results.append(future.result())
                        if i % 5 == 0 or i == total - 1:
                            p_bar.progress((i + 1) / total)
                            p_text.markdown(f"🚀 大规模并发诊断中: `{i+1}/{total}`")
                
                st.session_state.results_data = results
                st.session_state.processed = True
                st.rerun()
    else:
        # 1. 战报提示
        df_res = pd.DataFrame(st.session_state.results_data)
        total_count = len(df_res)
        to_review = df_res[df_res['Confidence'] < confidence_threshold]
        review_count = len(to_review)

        col_h, col_r = st.columns([5, 1])
        col_h.success(f"✅ 机审完毕！处理 {total_count} 条，其中 {review_count} 条待人工复核。")
        if col_r.button("🔄 新任务"):
            st.session_state.processed = False
            st.rerun()

        # 2. 统计指标
        c1, c2, c3 = st.columns(3)
        c1.metric("自动放行率", f"{((total_count-review_count)/total_count)*100:.1f}%")
        c2.metric("待复核疑难件", f"{review_count} 条", delta=f"-{review_count}", delta_color="inverse")
        c3.metric("整体置信均值", f"{df_res['Confidence'].mean():.1f}")

        # 3. 可视化看板
        st.divider()
        st.subheader("📊 质量分布看板")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            tag_counts = df_res['Tag'].value_counts().reset_index()
            tag_counts.columns = ['缺陷类型', '数量']
            st.plotly_chart(px.pie(tag_counts, values='数量', names='缺陷类型', title="缺陷类型占比", hole=0.4), use_container_width=True)
        with col_chart2:
            st.plotly_chart(px.histogram(df_res, x="Confidence", title="系统置信度分布图", nbins=10), use_container_width=True)
        
        st.divider()
        st.write("### 📋 诊断明细表")
        st.dataframe(df_res, use_container_width=True)

# ==========================================
# Tab 2: 专家复核中心 (智能摘要 + 自动提炼)
# ==========================================
with tab_human:
    st.subheader("👨‍🏫 疑难 Case 决策中心")
    if not st.session_state.processed:
        st.info("💡 暂无任务，请先在自动化诊断页上传数据。")
    else:
        queue = [r for r in st.session_state.results_data if r['Confidence'] < confidence_threshold and not r['Reviewed']]
        if not queue:
            st.success("🎉 复核队列已全部清空！所有结果已校准。")
        else:
            item = queue[0]
            curr_case_id = f"{item['Query']}_{item['Title']}"
            
            # --- 智能切换逻辑：当进入新 Case 时 ---
            if st.session_state.last_id != curr_case_id:
                st.session_state.last_id = curr_case_id
                st.session_state.rule_draft = ""
                st.session_state.active_tag = None
                
                # 🌟 智能逻辑摘要 (非截断)
                with st.spinner("🧠 正在梳理原判逻辑摘要..."):
                    try:
                        summary_prompt = f"请将下方的诊断理由缩减至40字内，保留其核心判定因果：\n{item['Reason']}"
                        resp = agent.client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": summary_prompt}]
                        )
                        st.session_state.smart_summary = resp.choices[0].message.content.strip()
                    except:
                        st.session_state.smart_summary = item['Reason'][:80] + "..."

            # 1. 增强型专家看板
            st.markdown(f"""
                <div class="review-card">
                    <div style="font-size:1.1em; margin-bottom:8px;"><b>🔍 搜索词:</b> <span style="color:#1E88E5;">{item['Query']}</span></div>
                    <div style="margin-bottom:8px;"><b>📄 结果标题:</b> {item['Title']}</div>
                    <div style="margin-bottom:8px;"><b>🤖 系统原判:</b> <code>{item['Tag']}</code> <small>(置信度: {item['Confidence']})</small></div>
                    <div class="logic-box">
                        <small><b>💡 机审核心逻辑 (AI 摘要)：</b></small><br>
                        <span style="font-size:0.95em; color:#495057;">{st.session_state.smart_summary}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 2. 自动化纠偏流
            with st.container(border=True):
                st.write("**👨‍⚖️ 专家最终裁定**")
                tags_list = list(TAG_DICT.values())
                new_tag = st.selectbox("第一步：选择正确标签", tags_list, index=tags_list.index(item['Tag']) if item['Tag'] in tags_list else 0)
                
                # 🌟 自动化监听：标签一换，AI 自动重写规则
                if new_tag != st.session_state.active_tag:
                    with st.spinner("🪄 AI 正在同步提炼新规则..."):
                        st.session_state.rule_draft = agent.auto_extract_rule(
                            item['Query'], item['Title'], item['Summary'], new_tag, item['Tag']
                        )
                        st.session_state.active_tag = new_tag

                # 展示规则草稿，人可以直接点提交（跳过输入）
                final_rule = st.text_area("第二步：判定准则 (AI 已自动生成，确认无误可直接提交)", value=st.session_state.rule_draft, height=100)

                if st.button("🚀 确认提交并同步至记忆库", type="primary"):
                    source = "Human" if final_rule != st.session_state.rule_draft else "AI"
                    # 存入 RAG 库
                    agent.learn(item['Query'], new_tag, final_rule, source=source)
                    # 更新状态
                    for r in st.session_state.results_data:
                        if r['Query'] == item['Query'] and r['Title'] == item['Title']:
                            r['Tag'], r['Reviewed'], r['Confidence'] = new_tag, True, 100
                            r['Reason'] = f"【人工纠偏-{source}】{final_rule}"
                    
                    st.session_state.rule_draft = ""
                    st.session_state.active_tag = None
                    st.success("✨ 经验已成功沉淀。")
                    time.sleep(0.5)
                    st.rerun()

# ==========================================
# Tab 3: 记忆治理 (RAG 知识库)
# ==========================================
with tab_admin:
    st.subheader("🧠 RAG 长期记忆治理")
    st.caption("这里存储了所有人类专家教导过的判定准则，它们将直接影响未来的 AI 判断。")
    
    data = agent.collection.get()
    if data['ids']:
        # 整理展示数据
        kb_df = pd.DataFrame({
            "ID": data['ids'],
            "来源": [m.get('source', '👤 Human') for m in data['metadatas']],
            "Query": [m.get('query', '') for m in data['metadatas']],
            "准则内容": [m.get('human_rule', '') for m in data['metadatas']],
            "所属标签": [m.get('correct_tag', '') for m in data['metadatas']]
        })
        
        # 增加来源过滤器
        src_filter = st.multiselect("过滤来源", ["🤖 AI", "👤 Human"], default=["🤖 AI", "👤 Human"])
        display_df = kb_df[kb_df['来源'].isin(src_filter)]
        
        st.dataframe(display_df, use_container_width=True)
        
        # 删除操作
        st.divider()
        col_del1, col_del2 = st.columns([3, 1])
        target_id = col_del1.selectbox("选择要物理删除的规则 ID", ["--"] + data['ids'])
        if target_id != "--" and col_del2.button("🗑️ 物理删除", type="secondary", use_container_width=True):
            agent.collection.delete(ids=[target_id])
            st.warning(f"规则 {target_id} 已从向量库移除。")
            time.sleep(1)
            st.rerun()
    else:
        st.info("📭 记忆库目前还是空的，快去复核中心教导 AI 吧！")