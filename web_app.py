import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚙️ 核心引擎导入
# ==========================================
try:
    from search_rel_eval import RagSearchEvalAgent, API_KEY, TAG_DICT
except ImportError:
    st.error("❌ 核心引擎加载失败：请确保 search_rel_eval.py 存在。")
    st.stop()

# 页面配置
st.set_page_config(page_title="SearchEval Pro | AI 质量决策中心", layout="wide")

# 注入增强 CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E9ECEF; }
    .review-card { background: white; padding: 20px; border-radius: 12px; border-left: 8px solid #EF4444; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return RagSearchEvalAgent("./agent_vector_db")

agent = load_agent()

# 初始化 Session State
if 'results_data' not in st.session_state: st.session_state.results_data = []
if 'processed' not in st.session_state: st.session_state.processed = False
if 'rule_draft' not in st.session_state: st.session_state.rule_draft = ""
if 'active_tag' not in st.session_state: st.session_state.active_tag = None

# --- 侧边栏 ---
with st.sidebar:
    st.title("🛡️ 决策控制中心")
    st.caption("作为 PM/策略，你可以在此调整人机协作边界")
    st.divider()
    
    confidence_threshold = st.slider("人工复核触发阈值", 0, 100, 95)
    st.info(f"💡 置信度低于 {confidence_threshold} 的数据将拦截到复核中心。")
    
    st.divider()
    mem_size = agent.collection.count()
    st.metric("知识资产规模", f"{mem_size} 条规则")
    
    if st.button("🗑️ 清空当前任务", use_container_width=True):
        st.session_state.results_data = []
        st.session_state.processed = False
        st.rerun()

# --- 主界面 ---
tab_eval, tab_human, tab_admin = st.tabs(["🚀 自动化诊断", "👨‍🏫 专家复核中心", "🧠 记忆治理"])

# ==========================================
# Tab 1: 自动化诊断 (并发加速 + 可视化看板)
# ==========================================
with tab_eval:
    if not st.session_state.processed:
        st.subheader("待评估任务导入")
        file = st.file_uploader("上传 CSV", type="csv")
        if file:
            df_in = None
            for enc in ["utf-8-sig", "gbk", "utf-8"]:
                try: 
                    file.seek(0)
                    df_in = pd.read_csv(file, encoding=enc)
                    break
                except: continue
            
            if df_in is not None:
                st.success(f"已加载 {len(df_in)} 条数据")
                if st.button("🔥 启动并发加速诊断", type="primary"):
                    p_bar = st.progress(0)
                    p_text = st.empty()
                    results = []
                    df_list = df_in.to_dict("records")
                    total = len(df_list)

                    def process_task(row):
                        q, t, s = [str(row.get(k, "")) for k in ['query', 'result_title', 'result_summary']]
                        res = agent.evaluate(q, t, s)
                        return {
                            "Query": q, "Title": t, "Summary": s,
                            "Tag": res.get("bad_case_tag", "Error"),
                            "Confidence": int(res.get("confidence_score", 0)),
                            "Reason": res.get("thought_process", ""),
                            "Reviewed": False
                        }

                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = {executor.submit(process_task, r): i for i, r in enumerate(df_list)}
                        for i, future in enumerate(as_completed(futures)):
                            results.append(future.result())
                            if i % 5 == 0 or i == total - 1:
                                p_bar.progress((i + 1) / total)
                                p_text.markdown(f"🚀 已完成: `{i+1}/{total}`")
                    
                    st.session_state.results_data = results
                    st.session_state.processed = True
                    st.rerun()
    else:
        # 1. 任务战报
        total = len(st.session_state.results_data)
        to_review = [r for r in st.session_state.results_data if r['Confidence'] < confidence_threshold]
        review_count = len(to_review)
        
        st.success(f"✅ 诊断完毕！共处理 {total} 条数据。")
        c1, c2, c3 = st.columns(3)
        c1.metric("自动放行量", f"{total - review_count} 条", f"{((total-review_count)/total)*100:.1f}%")
        c2.metric("待人工审核", f"{review_count} 条", delta=f"-{review_count}", delta_color="inverse")
        c3.metric("平均置信度", f"{pd.DataFrame(st.session_state.results_data)['Confidence'].mean():.1f}")

        # 2. 数据看板
        st.divider()
        st.subheader("📊 诊断结果看板")
        df_plot = pd.DataFrame(st.session_state.results_data)
        tag_stats = df_plot['Tag'].value_counts().reset_index()
        tag_stats.columns = ['缺陷类型', '数量']
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(px.pie(tag_stats, values='数量', names='缺陷类型', title="缺陷分布占比", hole=0.4), use_container_width=True)
        with col_p2:
            st.plotly_chart(px.histogram(df_plot, x="Confidence", nbins=10, title="置信度分布趋势"), use_container_width=True)
        
        st.plotly_chart(px.bar(tag_stats, x='缺陷类型', y='数量', text='数量', title="分类统计", color='缺陷类型'), use_container_width=True)
        
        st.divider()
        st.dataframe(df_plot, use_container_width=True)

# ==========================================
# Tab 2: 专家复核中心 (自动化闭环)
# ==========================================
with tab_human:
    st.subheader("👨‍🏫 专家决策中心")
    if st.session_state.processed:
        queue = [r for r in st.session_state.results_data if r['Confidence'] < confidence_threshold and not r['Reviewed']]
        if not queue:
            st.success("🎉 暂无疑难 Case，所有结果均已通过机审。")
        else:
            item = queue[0]
            # 自动 Case 识别与重置
            curr_id = f"{item['Query']}_{item['Title']}"
            if 'last_id' not in st.session_state or st.session_state.last_id != curr_id:
                st.session_state.last_id = curr_id
                st.session_state.rule_draft = ""
                st.session_state.active_tag = None

            # 增强信息卡片
            reason_short = item['Reason'][:120] + "..." if len(item['Reason']) > 120 else item['Reason']
            st.markdown(f"""
                <div class="review-card">
                    <b>🔍 Query</b>: {item['Query']}<br>
                    <b>📄 Title</b>: {item['Title']}<br>
                    <b>🤖 系统建议</b>: {item['Tag']} <span style="color:#999;">({item['Confidence']}分)</span><br>
                    <small>💡 机审理由: {reason_short}</small>
                </div>
            """, unsafe_allow_html=True)

            with st.container(border=True):
                st.write("**🧠 自动归因流**")
                tags = list(TAG_DICT.values())
                new_tag = st.selectbox("1. 确认正确标签", tags, index=tags.index(item['Tag']) if item['Tag'] in tags else 0)
                
                # 核心自动化：监听标签变化触发 AI 总结
                if new_tag != st.session_state.active_tag:
                    with st.spinner("🤖 AI 正在自动总结逻辑..."):
                        st.session_state.rule_draft = agent.auto_extract_rule(item['Query'], item['Title'], item['Summary'], new_tag, item['Tag'])
                        st.session_state.active_tag = new_tag

                final_rule = st.text_area("2. 判定准则 (AI 已自动填充，合理可直接提交)", value=st.session_state.rule_draft, height=100)

                if st.button("🚀 确认并同步记忆", type="primary"):
                    source = "Human" if final_rule != st.session_state.rule_draft else "AI"
                    agent.learn(item['Query'], new_tag, final_rule, source=source)
                    for r in st.session_state.results_data:
                        if r['Query'] == item['Query'] and r['Title'] == item['Title']:
                            r['Tag'], r['Reviewed'], r['Confidence'] = new_tag, True, 100
                            r['Reason'] = f"【人工纠偏-{source}】{final_rule}"
                    st.session_state.rule_draft = ""
                    st.session_state.active_tag = None
                    st.success("✅ 已学习")
                    time.sleep(0.5)
                    st.rerun()

# ==========================================
# Tab 3: 记忆治理 (显示来源)
# ==========================================
with tab_admin:
    st.subheader("🧠 RAG 知识库管理")
    data = agent.collection.get()
    if data['ids']:
        kb_df = pd.DataFrame({
            "ID": data['ids'],
            "来源": [m.get('source', '👤 Human') for m in data['metadatas']],
            "Query": [m.get('query', '') for m in data['metadatas']],
            "规则": [m.get('human_rule', '') for m in data['metadatas']],
            "标签": [m.get('correct_tag', '') for m in data['metadatas']]
        })
        st.dataframe(kb_df, use_container_width=True)
        
        target_id = st.selectbox("选择要删除的规则 ID", ["--"] + data['ids'])
        if target_id != "--" and st.button("🗑️ 物理删除"):
            agent.collection.delete(ids=[target_id])
            st.warning("已删除。")
            time.sleep(0.4)
            st.rerun()