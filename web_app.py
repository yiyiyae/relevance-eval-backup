import importlib.util
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent


# ==========================================
# ⚙️ 核心配置与引擎导入
# ==========================================
def load_engine_module():
    candidates = [
        APP_DIR / "search_rel_eval.py",
        APP_DIR / "search_rel_eval_v2.py",
        APP_DIR / "search_rel_eval_fixed.py",
        APP_DIR / "search_rel_eval(5).py",
        APP_DIR / "search_rel_eval(4).py",
        APP_DIR / "search_rel_eval(2).py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("search_rel_eval_runtime", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise ImportError("未找到 search_rel_eval 引擎文件")


try:
    engine = load_engine_module()
    RagSearchEvalAgent = engine.RagSearchEvalAgent
    TAG_DICT = engine.TAG_DICT
    MODEL_NAME = engine.MODEL_NAME
    OUTPUT_CSV = engine.OUTPUT_CSV
    build_case_key = engine.build_case_key
    read_csv_with_fallback = engine.read_csv_with_fallback
    append_to_output = engine.append_to_output
    load_output_df = engine.load_output_df
    SOURCE_AI = getattr(engine, "SOURCE_AI", "AI")
    SOURCE_HUMAN = getattr(engine, "SOURCE_HUMAN", "Human")
except Exception as exc:
    st.error(f"❌ 核心引擎加载失败：{exc}")
    st.stop()

st.set_page_config(page_title="SearchEval Pro | AI 质量中心", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E9ECEF; }
    .review-card { background: white; padding: 25px; border-radius: 12px; border-left: 8px solid #EF4444; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .logic-box { background: #f1f3f5; padding: 12px; border-radius: 8px; border-left: 4px solid #adb5bd; margin-top: 10px; }
    .lookup-card { background: white; padding: 18px; border-radius: 12px; border: 1px solid #E9ECEF; margin-top: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agent():
    return RagSearchEvalAgent(str(APP_DIR / "agent_vector_db"))


agent = load_agent()


# --- Session State ---
def init_state():
    defaults = {
        "results_data": [],
        "processed": False,
        "rule_draft": "",
        "smart_summary": "",
        "active_tag": None,
        "last_id": None,
        "uploaded_filename": None,
        "current_input_total": 0,
        "history_loaded": False,
        "manual_query": "",
        "manual_title": "",
        "manual_summary": "",
        "bili_enrich_result": None,
        "bili_summary_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
RESULT_FLUSH_BATCH_SIZE = 10

# ==========================================
# 🧰 数据辅助
# ==========================================
def output_row_to_ui_record(row: pd.Series, confidence_threshold: int) -> dict:
    query = str(row.get("query", ""))
    title = str(row.get("result_title", ""))
    summary = str(row.get("result_summary", ""))
    confidence = int(row.get("confidence", 0) or 0)
    reviewed_raw = row.get("reviewed", None)
    if pd.isna(reviewed_raw):
        reviewed = confidence >= confidence_threshold
    else:
        reviewed = str(reviewed_raw).strip().lower() in {"true", "1", "yes"}

    return {
        "CaseKey": str(row.get("case_key", build_case_key(query, title, summary))),
        "Query": query,
        "Title": title,
        "Summary": summary,
        "Tag": str(row.get("llm_tag", "Error")),
        "Confidence": confidence,
        "Reason": str(row.get("llm_reason", "")),
        "Reviewed": reviewed,
    }



def ui_record_to_output_row(record: dict) -> dict:
    return {
        "case_key": record["CaseKey"],
        "query": record["Query"],
        "result_title": record["Title"],
        "result_summary": record["Summary"],
        "llm_tag": record["Tag"],
        "llm_reason": record["Reason"],
        "confidence": int(record["Confidence"]),
        "reviewed": bool(record["Reviewed"]),
    }



def merge_records(records: list[dict]) -> list[dict]:
    by_key = {}
    for rec in records:
        by_key[rec["CaseKey"]] = rec
    return list(by_key.values())



def get_history_map(confidence_threshold: int) -> dict:
    df_out = load_output_df()
    if df_out.empty:
        return {}
    return {
        str(row["case_key"]): output_row_to_ui_record(row, confidence_threshold)
        for _, row in df_out.iterrows()
    }



def prepare_input_df(file_obj):
    df_in = read_csv_with_fallback(file_obj)
    required = {"query", "result_title", "result_summary"}
    missing = required - set(df_in.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {', '.join(sorted(missing))}")
    df_in = df_in.copy()
    df_in["case_key"] = df_in.apply(
        lambda r: build_case_key(r["query"], r["result_title"], r["result_summary"]), axis=1
    )
    return df_in



def sync_history_only(confidence_threshold: int):
    history_map = get_history_map(confidence_threshold)
    st.session_state.results_data = list(history_map.values())
    st.session_state.processed = bool(history_map)
    st.session_state.history_loaded = bool(history_map)
    return bool(history_map)



def run_dynamic_resume(df_in: pd.DataFrame, confidence_threshold: int):
    history_map = get_history_map(confidence_threshold)
    input_keys = set(df_in["case_key"].astype(str).tolist())
    matched_history = [history_map[k] for k in input_keys if k in history_map]
    done_keys = {r["CaseKey"] for r in matched_history}
    pending_df = df_in[~df_in["case_key"].astype(str).isin(done_keys)].copy()

    results = list(matched_history)
    total = len(df_in)
    done_count = len(matched_history)

    progress_info = {
        "total": total,
        "already_done": done_count,
        "pending": len(pending_df),
    }

    if pending_df.empty:
        st.session_state.results_data = merge_records(results)
        st.session_state.processed = True
        st.session_state.current_input_total = total
        return progress_info

    p_bar = st.progress(done_count / total if total else 0)
    p_text = st.empty()

    def process_row_task(row):
        q = str(row.get("query", ""))
        t = str(row.get("result_title", ""))
        s = str(row.get("result_summary", ""))
        eval_res = agent.evaluate(q, t, s)
        record = {
            "CaseKey": str(row.get("case_key", build_case_key(q, t, s))),
            "Query": q,
            "Title": t,
            "Summary": s,
            "Tag": eval_res.get("bad_case_tag", "Error"),
            "Confidence": int(eval_res.get("confidence_score", 0)),
            "Reason": eval_res.get("thought_process", ""),
            "Reviewed": int(eval_res.get("confidence_score", 0)) >= confidence_threshold,
        }
        append_to_output([ui_record_to_output_row(record)])
        return record

    pending_records = pending_df.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_row_task, row): i for i, row in enumerate(pending_records, start=1)}
        for finished_count, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            results.append(rec)
            current_done = done_count + finished_count
            if total:
                p_bar.progress(current_done / total)
            p_text.markdown(f"🚀 动态接续中：已完成 `{current_done}/{total}`，历史命中 {done_count} 条")

    st.session_state.results_data = merge_records(results)
    st.session_state.processed = True
    st.session_state.current_input_total = total
    return progress_info



def persist_single_review(item: dict, final_rule: str, source: str):
    item["Reason"] = f"【人工纠偏-{source}】{final_rule}"
    item["Confidence"] = 100
    item["Reviewed"] = True
    append_to_output([ui_record_to_output_row(item)])



def render_bili_lookup_result(payload: dict):
    if not payload:
        return
    current_title = st.session_state.get("manual_title", "").strip()
    if payload.get("search_title", "").strip() != current_title:
        return

    matched = payload.get("matched", False)
    stage = payload.get("lookup_stage", "")
    stage_label_map = {
        "empty_title": "输入为空",
        "search_failed": "搜索接口失败",
        "no_candidates": "候选解析失败",
        "title_mismatch": "标题未精确命中",
        "fetch_meta_failed": "视频详情读取失败",
        "completed": "回查完成",
    }
    stage_label = stage_label_map.get(stage, stage or "未知阶段")

    if matched:
        st.success(payload.get("message", "B站标题精确回查成功。"))
    else:
        st.warning(payload.get("message", "B站标题回查失败。"))

    tags = payload.get("tags", []) or []
    tags_text = "、".join(tags[:8]) if tags else "-"
    desc = payload.get("desc", "") or "-"
    candidate_titles = payload.get("candidate_titles", []) or []
    keyword_variants = payload.get("keyword_variants", []) or []
    search_route = payload.get("search_route", "") or "-"
    debug_detail = payload.get("debug_detail", "") or "-"

    st.markdown(
        f"""
        <div class="lookup-card">
            <div><b>搜索标题：</b>{payload.get('search_title', '')}</div>
            <div><b>当前阶段：</b>{stage_label}</div>
            <div><b>搜索通道：</b>{search_route}</div>
            <div><b>命中标题：</b>{payload.get('matched_title', '') or '-'}</div>
            <div><b>UP 主：</b>{payload.get('uploader', '') or '-'}</div>
            <div><b>标签：</b>{tags_text}</div>
            <div><b>简介：</b>{desc}</div>
            <div><b>摘要来源：</b>B站标题规范化回查 + 视频简介/标签</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if payload.get("video_url"):
        st.link_button("打开匹配视频", payload["video_url"])
    with st.expander("查看回查细节"):
        st.write("候选标题：", candidate_titles if candidate_titles else ["-"])
        st.write("检索变体：", keyword_variants if keyword_variants else ["-"])
        st.write("调试信息：", debug_detail)
        if payload.get("search_url"):
            st.write("搜索页：", payload["search_url"])


# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.title("🛡️ 决策控制中心")
    st.caption("PM / 策略专家的人机协作边界")
    st.divider()

    confidence_threshold = st.slider("人工复核触发阈值", 0, 100, 95)
    st.info(f"💡 置信度 < {confidence_threshold} 的数据会进入专家复核队列。")

    st.divider()
    if st.button("📂 仅同步历史输出文件", use_container_width=True):
        if sync_history_only(confidence_threshold):
            st.success("✅ 已载入历史输出。上传新的 CSV 后会自动按 case_key 动态接续。")
            st.rerun()
        else:
            st.warning("📭 暂未发现历史输出文件。")

    if st.button("🗑️ 清空当前页面状态", use_container_width=True):
        for key in [
            "results_data", "processed", "rule_draft", "smart_summary", "active_tag",
            "last_id", "uploaded_filename", "current_input_total", "history_loaded",
            "manual_query", "manual_title", "manual_summary", "bili_enrich_result",
        ]:
            st.session_state.pop(key, None)
        init_state()
        st.rerun()

    history_df = load_output_df()
    if history_df.empty:
        st.caption("历史输出：0 条")
    else:
        st.caption(f"历史输出：{len(history_df)} 条")

# --- 主界面 Tabs ---
tab_eval, tab_human, tab_admin = st.tabs(["🚀 自动化诊断", "👨‍🏫 专家复核中心", "🧠 记忆治理"])


# ==========================================
# Tab 1: 自动化诊断 + 动态断点续跑
# ==========================================
with tab_eval:
    st.subheader("待评估任务导入")
    input_mode = st.radio(
        "选择输入方式",
        ["CSV 批量上传", "单条手动输入"],
        horizontal=True,
    )

    df_in = None
    source_label = None

    if input_mode == "CSV 批量上传":
        uploaded_file = st.file_uploader("上传 CSV 评测文件", type="csv")
        if uploaded_file is not None:
            try:
                df_in = prepare_input_df(uploaded_file)
                source_label = uploaded_file.name
            except Exception as exc:
                st.error(f"❌ 文件处理失败：{exc}")
    else:
        with st.container(border=True):
            st.caption("适合现场 demo：可以只输入 Query 和结果标题，再一键从 B 站回查并自动补全摘要。")
            st.text_input("Query", key="manual_query", placeholder="例如：苹果手机壳")
            st.text_input("结果标题", key="manual_title", placeholder="例如：iPhone 15 透明防摔手机壳")

            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button("✨ 根据 B 站标题自动补全摘要", use_container_width=True):
                    title = st.session_state.get("manual_title", "").strip()
                    if not title:
                        st.warning("请先输入结果标题。")
                    else:
                        with st.spinner("正在回查 B 站并生成摘要..."):
                            enrich_result = agent.enrich_summary_from_bilibili_title(title)
                        st.session_state.bili_enrich_result = enrich_result
                        if enrich_result.get("matched") and enrich_result.get("summary"):
                            st.session_state.manual_summary = enrich_result["summary"]
                        st.rerun()
            with col_action2:
                if st.button("🧹 清空摘要", use_container_width=True):
                    st.session_state.manual_summary = ""
                    st.session_state.bili_enrich_result = None
                    st.rerun()

            st.text_area(
                "结果摘要",
                key="manual_summary",
                placeholder="可手动填写，或点击上面的按钮自动补全",
                height=120,
            )
            render_bili_lookup_result(st.session_state.get("bili_enrich_result"))

            if st.session_state.get("manual_query", "").strip() and st.session_state.get("manual_title", "").strip():
                if st.session_state.get("manual_summary", "").strip():
                    df_in = pd.DataFrame([{
                        "query": st.session_state.manual_query.strip(),
                        "result_title": st.session_state.manual_title.strip(),
                        "result_summary": st.session_state.manual_summary.strip(),
                    }])
                    df_in["case_key"] = df_in.apply(
                        lambda r: build_case_key(r["query"], r["result_title"], r["result_summary"]), axis=1
                    )
                    source_label = "单条手动输入"
                else:
                    st.info("填写 Query 和结果标题后，可以手动填写摘要，或点击上方按钮自动补全摘要。")

    if df_in is not None:
        total_cases = len(df_in)
        history_map = get_history_map(confidence_threshold)
        already_done = int(df_in["case_key"].astype(str).isin(set(history_map.keys())).sum())
        pending = total_cases - already_done

        c1, c2, c3 = st.columns(3)
        c1.metric("当前输入总数", total_cases)
        c2.metric("已命中历史断点", already_done)
        c3.metric("待继续处理", pending)

        st.caption("这里的断点是动态的：根据当前输入的 case_key 和历史输出文件实时比对，而不是单纯把旧结果静态读出来。")

        if input_mode == "CSV 批量上传":
            btn_label = "♻️ 从断点继续" if already_done > 0 else "🔥 启动 AI 引擎诊断"
        else:
            btn_label = "♻️ 复用历史结果" if already_done > 0 else "🎯 诊断这一条"

        if st.button(btn_label, type="primary"):
            info = run_dynamic_resume(df_in, confidence_threshold)
            st.session_state.uploaded_filename = source_label
            st.success(
                f"✅ 完成。总计 {info['total']} 条，历史命中 {info['already_done']} 条，"
                f"本轮新处理 {info['pending']} 条。"
            )
            st.rerun()

    if st.session_state.processed and st.session_state.results_data:
        df_res = pd.DataFrame(st.session_state.results_data)
        total_count = len(df_res)
        review_df = df_res[(df_res["Confidence"] < confidence_threshold) & (~df_res["Reviewed"])]
        review_count = len(review_df)

        st.divider()
        col_h, col_r = st.columns([5, 1])
        file_hint = st.session_state.uploaded_filename or "当前会话"
        col_h.success(f"✅ 已载入/处理 {total_count} 条样本（来源：{file_hint}），待人工复核 {review_count} 条。")
        if col_r.button("🔄 重置展示"):
            st.session_state.results_data = []
            st.session_state.processed = False
            st.rerun()

        c1, c2, c3 = st.columns(3)
        auto_pass_rate = ((total_count - review_count) / total_count) * 100 if total_count else 0
        avg_conf = df_res["Confidence"].mean() if total_count else 0
        c1.metric("自动放行率", f"{auto_pass_rate:.1f}%")
        c2.metric("待复核疑难件", f"{review_count} 条")
        c3.metric("整体置信均值", f"{avg_conf:.1f}")

        st.divider()
        st.subheader("📊 质量分布看板")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            tag_counts = df_res["Tag"].value_counts().reset_index()
            tag_counts.columns = ["缺陷类型", "数量"]
            st.plotly_chart(
                px.pie(tag_counts, values="数量", names="缺陷类型", title="缺陷类型占比", hole=0.4),
                use_container_width=True,
            )
        with col_chart2:
            st.plotly_chart(
                px.histogram(df_res, x="Confidence", title="系统置信度分布图", nbins=10),
                use_container_width=True,
            )

        st.divider()
        st.write("### 📋 诊断明细表")
        st.dataframe(df_res.sort_values(by=["Reviewed", "Confidence"], ascending=[True, True]), use_container_width=True)


# ==========================================
# Tab 2: 专家复核中心
# ==========================================
with tab_human:
    st.subheader("👨‍🏫 疑难 Case 决策中心")
    if not st.session_state.results_data:
        st.info("💡 暂无任务，请先上传 CSV 并启动诊断。")
    else:
        queue = [
            r for r in st.session_state.results_data
            if r["Confidence"] < confidence_threshold and not r["Reviewed"]
        ]
        if not queue:
            st.success("🎉 复核队列已全部清空！所有结果已校准。")
        else:
            item = queue[0]
            curr_case_id = item["CaseKey"]

            if st.session_state.last_id != curr_case_id:
                st.session_state.last_id = curr_case_id
                st.session_state.rule_draft = ""
                st.session_state.active_tag = None
                with st.spinner("🧠 正在梳理原判逻辑摘要..."):
                    try:
                        summary_prompt = f"请将下方的诊断理由缩减至40字内，保留其核心判定因果：\n{item['Reason']}"
                        resp = agent.client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": summary_prompt}],
                        )
                        st.session_state.smart_summary = resp.choices[0].message.content.strip()
                    except Exception:
                        st.session_state.smart_summary = item["Reason"][:80] + "..."

            st.markdown(
                f"""
                <div class="review-card">
                    <div style="font-size:1.1em; margin-bottom:8px;"><b>🔍 搜索词:</b> <span style="color:#1E88E5;">{item['Query']}</span></div>
                    <div style="margin-bottom:8px;"><b>📄 结果标题:</b> {item['Title']}</div>
                    <div style="margin-bottom:8px;"><b>🤖 系统原判:</b> <code>{item['Tag']}</code> <small>(置信度: {item['Confidence']})</small></div>
                    <div class="logic-box">
                        <small><b>💡 机审核心逻辑 (AI 摘要)：</b></small><br>
                        <span style="font-size:0.95em; color:#495057;">{st.session_state.smart_summary}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.write("**👨‍⚖️ 专家最终裁定**")
                tags_list = list(TAG_DICT.values())
                new_tag = st.selectbox(
                    "第一步：选择正确标签",
                    tags_list,
                    index=tags_list.index(item["Tag"]) if item["Tag"] in tags_list else 0,
                )

                if new_tag != st.session_state.active_tag:
                    with st.spinner("🪄 AI 正在同步提炼新规则..."):
                        st.session_state.rule_draft = agent.auto_extract_rule(
                            item["Query"], item["Title"], item["Summary"], new_tag, item["Tag"]
                        )
                        st.session_state.active_tag = new_tag

                final_rule = st.text_area(
                    "第二步：判定准则 (AI 已自动生成，确认无误可直接提交)",
                    value=st.session_state.rule_draft,
                    height=100,
                )

                if st.button("🚀 确认提交并同步至记忆库", type="primary"):
                    source = SOURCE_HUMAN if final_rule != st.session_state.rule_draft else SOURCE_AI
                    agent.learn(item["Query"], new_tag, final_rule, source=source)

                    for r in st.session_state.results_data:
                        if r["CaseKey"] == item["CaseKey"]:
                            r["Tag"] = new_tag
                            persist_single_review(r, final_rule, source)

                    st.session_state.rule_draft = ""
                    st.session_state.active_tag = None
                    st.success("✨ 经验已成功沉淀，输出文件也已同步更新。")
                    time.sleep(0.5)
                    st.rerun()


# ==========================================
# Tab 3: 记忆治理
# ==========================================
with tab_admin:
    st.subheader("🧠 RAG 长期记忆治理")
    st.caption("这里存储了所有人类专家教导过的判定准则，它们会直接影响后续判定。")

    data = agent.collection.get()
    ids = data.get("ids", [])
    metas = data.get("metadatas", [])

    if ids:
        kb_df = pd.DataFrame({
            "ID": ids,
            "来源": [str((m or {}).get("source", SOURCE_HUMAN)) for m in metas],
            "Query": [str((m or {}).get("query", "")) for m in metas],
            "准则内容": [str((m or {}).get("human_rule", "")) for m in metas],
            "所属标签": [str((m or {}).get("correct_tag", "")) for m in metas],
            "创建时间": [int((m or {}).get("created_at_ms", 0) or 0) for m in metas],
        }).sort_values(by="创建时间", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("记忆总数", len(kb_df))
        c2.metric("AI 规则", int((kb_df["来源"] == SOURCE_AI).sum()))
        c3.metric("Human 规则", int((kb_df["来源"] == SOURCE_HUMAN).sum()))

        available_sources = sorted(kb_df["来源"].dropna().unique().tolist())
        search_kw = st.text_input("搜索记忆", placeholder="支持 Query / 规则 / 标签 / 来源")
        src_filter = st.multiselect("过滤来源", available_sources, default=available_sources)
        tag_filter = st.multiselect("过滤标签", sorted(kb_df["所属标签"].dropna().unique().tolist()))

        display_df = kb_df[kb_df["来源"].isin(src_filter)]
        if tag_filter:
            display_df = display_df[display_df["所属标签"].isin(tag_filter)]
        if search_kw.strip():
            kw = search_kw.strip().lower()
            mask = display_df.apply(
                lambda r: kw in "\n".join([str(v) for v in r.values]).lower(), axis=1
            )
            display_df = display_df[mask]

        st.dataframe(display_df, use_container_width=True)

        st.divider()
        col_del1, col_del2 = st.columns([3, 1])
        target_id = col_del1.selectbox("选择要物理删除的规则 ID", ["--"] + kb_df["ID"].tolist())
        if target_id != "--" and col_del2.button("🗑️ 物理删除", type="secondary", use_container_width=True):
            agent.collection.delete(ids=[target_id])
            st.warning(f"规则 {target_id} 已从向量库移除。")
            time.sleep(0.5)
            st.rerun()
    else:
        st.info("📭 记忆库目前还是空的，快去复核中心教导 AI 吧！")