#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

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
MEMORY_FILE = str(_SCRIPT_DIR / "agent_memory.json")

# 标准标签库（用于快捷键映射）
TAG_DICT = {
    "1": "完全不相关",
    "2": "丢词搜不准",
    "3": "query理解有误",
    "4": "推荐同领域内容",
    "5": "推荐场景衍生内容"
}

# =========================
# 🛠️ 断点续传与文件辅助
# =========================
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gbk", "utf-8", "gb18030"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError(f"无法读取文件 {path}")

def get_processed_queries() -> set:
    """读取已完成的进度，实现断点续传"""
    if os.path.exists(OUTPUT_CSV):
        try:
            df = read_csv_with_fallback(OUTPUT_CSV)
            if "query" in df.columns:
                return set(df["query"].astype(str).tolist())
        except:
            pass
    return set()

def append_to_output(results_list: list):
    """增量保存，防止中途退出丢失数据"""
    if not results_list: return
    df_new = pd.DataFrame(results_list)
    # 清理中间状态字段
    if "needs_intervention" in df_new.columns:
        df_new.drop(columns=["needs_intervention"], inplace=True)
        
    if os.path.exists(OUTPUT_CSV):
        df_old = read_csv_with_fallback(OUTPUT_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# =========================
# 🧠 Agent 核心类
# =========================
class SearchEvalAgent:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.memory = self._load_memory()

    def _load_memory(self) -> list:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return []

    def learn(self, query: str, correct_tag: str, human_rule: str):
        new_knowledge = {"example_query": query, "correct_tag": correct_tag, "human_rule": human_rule}
        self.memory.append(new_knowledge)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
        print(f"✨ [Agent 顿悟] 已将规则写入记忆库：{human_rule}")

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try: return json.loads(re.search(r"\{.*\}", text or "", flags=re.DOTALL).group(0))
        except: return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_with_retry(self, prompt: str) -> dict:
        resp = self.client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
            temperature=0, response_format={"type": "json_object"}
        )
        return self._safe_json_loads(resp.choices[0].message.content)

    def evaluate(self, query: str, title: str, summary: str) -> dict:
        base_prompt = r"""
# Role
你是一个极其严谨的搜索质量策略专家兼风控排雷大师。你的任务是诊断搜索 Bad Case 的缺陷类型，并在遇到模糊边界时主动标记需要人工复核。

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
        memory_section = ""
        if self.memory:
            memory_section = "\n=========================================\n# 💡 你的绝对行动纲领（前人总结的经验，优先级最高）\n请优先参考以下真实业务的纠偏记录，如果当前的 Query 与其中情况类似，直接套用人类指定的规则和标签：\n"
            for i, mem in enumerate(self.memory):
                memory_section += f"- 案例{i+1}: 搜【{mem['example_query']}】时，人类判定的规则是：{mem['human_rule']} -> 必须判为【{mem['correct_tag']}】\n"

        tail_prompt = f"""
=========================================
# 归因标准字典（仅限以下5选1）
- 完全不相关
- 丢词搜不准
- query理解有误
- 推荐同领域内容
- 推荐场景衍生内容

# Output Format (严格JSON)
{{
  "thought_process": "你的思考步骤",
  "bad_case_tag": "必须是5个标准标签之一",
  "confidence_score": 85
}}
[Query]: {query}
[Title]: {title}
[Summary]: {summary}
"""
        try:
            return self._call_llm_with_retry(base_prompt + memory_section + tail_prompt)
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
        "needs_intervention": confidence < 85
    }

# =========================
# 🏁 主程序 (断点续传 + 快捷键交互)
# =========================
def main():
    if not API_KEY: return print("[错误] 未设置 API KEY")
    df_all = read_csv_with_fallback(INPUT_CSV)
    
    # 🌟 断点续传逻辑：过滤掉已经处理过的数据
    processed_queries = get_processed_queries()
    df_todo = df_all[~df_all['query'].astype(str).isin(processed_queries)].copy()
    
    if df_todo.empty:
        return print(f"\n🎉 恭喜！{INPUT_CSV} 中的所有数据都已处理完毕。结果在 {OUTPUT_CSV}")
        
    print(f"\n{'='*50}")
    print(f"🚀 V40.2 启动 (已跳过 {len(processed_queries)} 条历史进度，剩余 {len(df_todo)} 条待处理)")
    print(f"{'='*50}")

    agent = SearchEvalAgent(MEMORY_FILE)
    results_auto = [] # 存放高置信度结果

    # 第一阶段：极速并发机审
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, i, r, agent): i for i, r in enumerate(df_todo.to_dict("records"))}
        for future in tqdm(as_completed(futures), total=len(df_todo), desc="机审进度"):
            res = future.result()
            results_auto.append(res)

    # 分离出高低置信度
    high_conf_results = [r for r in results_auto if not r["needs_intervention"]]
    intervention_queue = [r for r in results_auto if r["needs_intervention"]]
    
    # 将不需要干预的数据立刻落盘，保障进度！
    if high_conf_results:
        append_to_output(high_conf_results)

    print(f"\n✅ 机审阶段完成！耗时: {time.time() - start_time:.2f}s")
    print(f"💾 高置信度数据已保存入库。")
    print(f"⚠️ 发现 {len(intervention_queue)} 条低置信度数据，即将进入集中教学模式...")
    time.sleep(1)
    
    # 第二阶段：集中人工教学（支持快捷键与随时退出）
    if intervention_queue:
        print(f"\n{'='*50}\n👨‍🏫 阶段二：集中人工教学 (按 0 随时保存退出)\n{'='*50}")
        
        for item in intervention_queue:
            print(f"\n🔍 Query: 【{item['query']}】")
            print(f"   [Title]: {item['result_title']}")
            print(f"   [Summary]: {item['result_summary']}")
            print(f"   [机器判定]: {item['llm_tag']} (思路: {item['llm_reason']})")
            
            # 🌟 多选交互界面
            print("\n   🎯 请选择正确标签 (输入数字):")
            print("   [1] 完全不相关     [2] 丢词搜不准      [3] query理解有误")
            print("   [4] 推荐同领域内容 [5] 推荐场景衍生内容")
            print("   [回车] 保持机器原判  [0] 💾 保存进度并下班！")
            
            choice = input(">> 你的选择: ").strip()
            
            if choice == "0":
                print("\n👋 收到！正在保存当前进度，明天见！")
                break # 跳出循环，结束程序
                
            elif choice in TAG_DICT:
                human_tag = TAG_DICT[choice]
                human_rule = input(f">> 已选【{human_tag}】，请输入一条判别规则教给它 (例: '找正片变成看花絮判衍生'): ").strip()
                if human_rule:
                    agent.learn(item['query'], human_tag, human_rule)
                
                # 覆写结果
                item['llm_tag'] = human_tag
                item['is_correct'] = re.sub(r"\s+", "", human_tag).lower() == re.sub(r"\s+", "", item['expected_tag']).lower()
            elif choice != "":
                print("[警告] 输入无效，默认保持机器原判。")

            # 用户每处理完一条，立刻将这一条落盘保存！
            append_to_output([item])
            print("-" * 40)

    print(f"\n{'='*50}")
    print(f"✅ 本次任务执行完毕！")
    print(f"🧠 Agent 最新记忆容量: {len(agent.memory)} 条经验")
    print(f"📁 累计所有结果已安全保存在: {OUTPUT_CSV}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()