#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# 引入 RAG 核心组件
import chromadb

# =========================
# ⚙️ 核心配置区
# =========================
API_KEY = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 10 

_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "test_cases_new.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "new_full_results_output.csv")

# 【RAG 升级】从单一的 json 文件变成本地向量数据库文件夹
CHROMA_DB_PATH = str(_SCRIPT_DIR / "agent_vector_db") 

TAG_DICT = {
    "1": "完全不相关", "2": "丢词搜不准", "3": "query理解有误", 
    "4": "推荐同领域内容", "5": "推荐场景衍生内容"
}

# =========================
# 🛠️ 断点续传辅助
# =========================
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gbk", "utf-8", "gb18030"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError(f"无法读取文件 {path}")

def get_processed_queries() -> set:
    if os.path.exists(OUTPUT_CSV):
        try:
            df = read_csv_with_fallback(OUTPUT_CSV)
            if "query" in df.columns:
                return set(df["query"].astype(str).tolist())
        except: pass
    return set()

def append_to_output(results_list: list):
    if not results_list: return
    df_new = pd.DataFrame(results_list)
    if "needs_intervention" in df_new.columns:
        df_new.drop(columns=["needs_intervention"], inplace=True)
    if os.path.exists(OUTPUT_CSV):
        df_old = read_csv_with_fallback(OUTPUT_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# =========================
# 🧠 具备 RAG 长期记忆的 Agent 核心类
# =========================
class RagSearchEvalAgent:
    def __init__(self, db_path: str):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # 1. 初始化本地向量数据库
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # 2. 获取或创建名为 "eval_rules" 的集合（类似于数据库的表）
        self.collection = self.chroma_client.get_or_create_collection(name="eval_rules")
        print(f"📦 [向量库就绪] 当前知识库已包含 {self.collection.count()} 条人类规则。")

    # search_rel_eval.py 中的 learn 函数修改
    def learn(self, query: str, correct_tag: str, human_rule: str, source: str = "Human"):
        """将规则存入向量库，增加 source 标记 (已确认缩进)"""
        doc_id = f"rule_{int(time.time() * 1000)}"
        
        # 将元数据存入，包含规则来源
        self.collection.add(
            documents=[query],
            metadatas=[{
                "query": query, 
                "correct_tag": correct_tag, 
                "human_rule": human_rule,
                "source": source  # 🤖 AI 或 👤 Human
            }],
            ids=[doc_id]
        )
        print(f"✨ [Agent 顿悟] 规则 ({source}) 已写入记忆库！")

    def auto_extract_rule(self, query: str, title: str, summary: str, correct_tag: str, wrong_tag: str) -> str:
        """AI 自动从人类的纠偏动作中提炼抽象规则"""
        
        prompt = f"""
你是一个搜索质量专家。系统最近将一个 Case 判定错了，请根据人类的修正，总结出一条【高度抽象、可迁移】的判定规则。

[案例信息]
Query: {query}
标题: {title}
摘要: {summary}

[纠偏记录]
机器原判: {wrong_tag}
人类修正为: {correct_tag}

# 任务要求：
1. 解释为什么该 Case 属于 {correct_tag} 而非 {wrong_tag}。
2. ⚠️ 严禁提及具体的词汇（如“苹果”、“iPhone”）。
3. 必须使用抽象描述（如“主语品牌平替”、“核心意图偏移”、“修饰词丢失”）。
4. 长度控制在 30 字以内。

# 输出格式 (JSON)
{{
  "abstract_rule": "总结的抽象准则"
}}
"""
        try:
            # 适当调高温度，增加归纳能力
            res = self._call_llm_with_retry(prompt, temperature=0.5)
            return res.get("abstract_rule", f"识别到{correct_tag}特征，修正原判{wrong_tag}")
        except:
            return f"人类专家强制判定为{correct_tag}"



    def _retrieve_relevant_rules(self, query: str, top_k: int = 3) -> List[dict]:
        """RAG 核心：根据当前的 Query，检索最相似的历史规则"""
        if self.collection.count() == 0:
            return []
            
        # 限制 k 值不能超过数据库里的实际总量
        k = min(top_k, self.collection.count())
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        # 返回检索到的元数据列表
        return results['metadatas'][0] if results['metadatas'] else []

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try: return json.loads(re.search(r"\{.*\}", text or "", flags=re.DOTALL).group(0))
        except: return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_with_retry(self, prompt: str, temperature: float = 0.0) -> dict:
        # ⚠️ 注意这里新增了 temperature 参数
        resp = self.client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature, # 使用传入的温度
            response_format={"type": "json_object"}
        )
        return self._safe_json_loads(resp.choices[0].message.content)

    def evaluate(self, query: str, title: str, summary: str) -> dict:
        base_prompt = r"""
# Role
你是一个极其严谨的搜索质量策略专家兼风控排雷大师。你的任务是诊断搜索 Bad Case 的缺陷类型，并在遇到模糊边界时主动标记低置信度。

# 核心原则
【严格区分"同领域"与"场景衍生"】
- "同领域"必须是：用户的核心任务目标完全一致，仅具体对象发生同级替换（包括竞品替换）。
- "场景衍生"是：宏观领域一致，但用户要解决的问题（任务/动作）发生了偏移。

# 判定大纲与经典判例（请严格作为你的分类标尺！）
【第一步：宏观领域跨界与多义词排雷】
1. 判断双方是否属于同一个【超大行业/超大类目】。
2. 若完全跨界（毫无交集），必须严格区分以下两种情况：
   - 存在“一词多义”导致的概念偷换（如"苹果"水果变手机，"小米"谷物变数码，"潜伏"动作变电视剧） → 判【query理解有误】
   - 纯粹的跨界且无核心多义词歧义（如"蔡徐坤"变"原神"，或仅仅是"回家"和"家中"这种无关紧要的字面重合） → 判【完全不相关】

【第二步：核心诉求与实体的二维判断】
若宏观领域一致，请按以下标准区分两种 1 分泛化：
- 类别A【推荐同领域内容（平行实体替换）】：
  定义：用户的【核心动作/诉求】没变，仅仅是【核心实体】被替换成了同分类下的其他兄弟实体（换人、换物、换剧、换竞品）。
  - 判例1：搜“猫咪吐毛球” → 给“狗狗护理”。（实体：猫变狗。判定：同领域）
  - 判例2：搜“周杰伦演唱会” → 给“林俊杰演唱会”。（实体：周杰伦变林俊杰。判定：同领域）
  - 判例3：搜“剪映教程” → 给“PR教程”。（实体：剪映变PR，竞品替换。判定：同领域）

- 类别B【推荐场景衍生内容（诉求/形式偏移）】：
  定义：【核心实体】可能没变，但用户要做的【核心动作/诉求/形式】发生了明显的偏移。
  - 判例4：搜“流浪地球2导演” → 给“郭帆作品合集”。（诉求：从找特定人物 变 找作品集。判定：场景衍生）
  - 判例5：搜“周生如故结局” → 给“周生如故幕后花絮”。（诉求：从看正片 变 看花絮。判定：场景衍生）
  - 判例6：搜“王者荣耀妲己皮肤” → 给“英雄联盟妲己角色解析”。（诉求：从看外观 变 解析技能。判定：场景衍生）

【第三步：严苛修饰词丢失检查】
- 类别C【丢词搜不准（纵向降维）】：
  定义：主语没变，动作没变，仅仅丢失了修饰该主语的【具体限制词/定语】（如具体年份、具体型号、具体地域）。
  - 判例7：2024跨年 变 2023跨年。（丢了正确年份。判定：丢词搜不准）
  - 判例8：婴儿浴巾 变 家用成人浴巾。（丢了“婴儿”限制词。判定：丢词搜不准）
  🚨 丢词防坑红线：绝对不要把主语（如：Python、护照、电影、主题曲）当成限制词/定语！丢了主语算场景衍生或不相关，绝对不算丢词搜不准。
"""
        
        # ==================================================
        # 🌟 动态 RAG 注入区：只加载最相关的 Top 3 记忆
        # ==================================================
        relevant_memory = self._retrieve_relevant_rules(query, top_k=3)
        memory_section = ""
        
        if relevant_memory:
            memory_section = "\n=========================================\n# 💡 高优先级相关经验（由外部向量库召回）\n请高度重视以下人类历史上针对类似 Query 的判定规则：\n"
            for i, mem in enumerate(relevant_memory):
                memory_section += f"- 相似历史案例{i+1}: 搜【{mem['query']}】时，适用规则【{mem['human_rule']}】 -> 结论是【{mem['correct_tag']}】\n"

        tail_prompt = f"""
=========================================
# 归因标准字典（仅限以下5选1）
- 完全不相关
- 丢词搜不准
- query理解有误
- 推荐同领域内容
- 推荐场景衍生内容

# Output Format (严格JSON，请勿输出额外字段)
{{
  "thought_process": "你的思考步骤",
  "bad_case_tag": "必须是5个标准标签之一"
}}
[当前任务]
[Query]: {query}
[Title]: {title}
[Summary]: {summary}
"""
        try:
            import concurrent.futures
            from collections import Counter
            
            final_prompt = base_prompt + memory_section + tail_prompt
            results = []
            
            # 开启 3 个线程，对同一个 Case 采样 3 次，温度设为 0.6 增加多样性
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self._call_llm_with_retry, final_prompt, 0.6) for _ in range(3)]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if "bad_case_tag" in res:
                        results.append(res)
            
            if not results:
                return {"bad_case_tag": "Error", "confidence_score": 0, "thought_process": "API全量失败"}

            # --- 统计学一致性投票 ---
            tags = [r.get("bad_case_tag", "Unknown") for r in results]
            tag_counts = Counter(tags)
            
            # 找出得票数最多的标签
            most_common_tag, count = tag_counts.most_common(1)[0]
            
            # 制定客观置信度
            if count == 3:
                confidence = 100  # 3次全一致，极度确信
            elif count == 2:
                confidence = 66   # 2:1 分歧，模糊边界
            else:
                confidence = 33   # 1:1:1 彻底混乱

            # 提取获胜标签的任意一个思维链
            final_reason = ""
            for r in results:
                if r.get("bad_case_tag") == most_common_tag:
                    final_reason = r.get("thought_process", "")
                    break
                    
            # 💡 神来之笔：如果出现分歧，把分歧记录追加到理由里给前端看
            if count < 3:
                final_reason += f"\n\n🚨 [系统提示]：AI 内部产生分歧，投票分布为 {dict(tag_counts)}"

            return {
                "thought_process": final_reason,
                "bad_case_tag": most_common_tag,
                "confidence_score": confidence
            }
            
        except Exception as e:
            return {"bad_case_tag": "Error", "confidence_score": 0, "thought_process": str(e)}
            
# =========================
# ⚙️ 并发 Worker 函数
# =========================
def process_row(index, row, agent):
    q = str(row.get("query", ""))
    t = str(row.get("result_title", ""))
    s = str(row.get("result_summary", ""))
    exp_tag = str(row.get("expected_tag", "")).strip().lower()
    
    res = agent.evaluate(q, t, s)
    pred_tag = res.get("bad_case_tag", "Unknown")
    confidence = res.get("confidence_score", 0)
    
    return {
        "index": index, "query": q, "result_title": t, "result_summary": s, "expected_tag": exp_tag,
        "llm_tag": pred_tag, "llm_reason": res.get("thought_process", ""), "confidence": confidence,
        "is_correct": re.sub(r"\s+", "", pred_tag).lower() == re.sub(r"\s+", "", exp_tag),
        "needs_intervention": confidence < 95
    }

# =========================
# 🏁 主程序
# =========================
def main():
    if not API_KEY: return print("[错误] 未设置 API KEY")
    df_all = read_csv_with_fallback(INPUT_CSV)
    
    processed_queries = get_processed_queries()
    df_todo = df_all[~df_all['query'].astype(str).isin(processed_queries)].copy()
    
    if df_todo.empty:
        return print(f"\n🎉 恭喜！{INPUT_CSV} 中的所有数据都已处理完毕。结果在 {OUTPUT_CSV}")
        
    print(f"\n{'='*50}")
    print(f"🚀 V50.0 RAG 架构启动 (剩余 {len(df_todo)} 条待处理)")
    print(f"{'='*50}")

    agent = RagSearchEvalAgent(CHROMA_DB_PATH)
    results_auto = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, i, r, agent): i for i, r in enumerate(df_todo.to_dict("records"))}
        for future in tqdm(as_completed(futures), total=len(df_todo), desc="机审进度"):
            res = future.result()
            results_auto.append(res)

    high_conf_results = [r for r in results_auto if not r["needs_intervention"]]
    intervention_queue = [r for r in results_auto if r["needs_intervention"]]
    
    if high_conf_results:
        append_to_output(high_conf_results)

    print(f"\n✅ RAG 机审阶段完成！耗时: {time.time() - start_time:.2f}s")
    print(f"⚠️ 发现 {len(intervention_queue)} 条低置信度数据，即将进入人工教学...")
    time.sleep(1)
    
    if intervention_queue:
        print(f"\n{'='*50}\n👨‍🏫 阶段二：集中人工教学 (按 0 随时保存退出)\n{'='*50}")
        for item in intervention_queue:
            print(f"\n🔍 Query: 【{item['query']}】")
            print(f"   [Title]: {item['result_title']}")
            print(f"   [Summary]: {item['result_summary']}")
            print(f"   [机器判定]: {item['llm_tag']} (思路: {item['llm_reason']})")
            
            print("\n   🎯 [1]完全不相关 [2]丢词搜不准 [3]query理解有误 [4]同领域 [5]衍生 [回车]原判 [0]退出")
            choice = input(">> 选择: ").strip()
            
            if choice == "0":
                print("\n👋 保存进度，安全退出！")
                break
                
            elif choice in TAG_DICT:
                human_tag = TAG_DICT[choice]
                human_rule = input(f">> 已选【{human_tag}】，请输入判别规则: ").strip()
                if human_rule:
                    agent.learn(item['query'], human_tag, human_rule)
                
                item['llm_tag'] = human_tag
                item['is_correct'] = re.sub(r"\s+", "", human_tag).lower() == re.sub(r"\s+", "", item['expected_tag']).lower()

            append_to_output([item])
            print("-" * 40)

    print(f"\n{'='*50}")
    print(f"✅ 任务完毕！当前向量库容量: {agent.collection.count()} 条经验")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()