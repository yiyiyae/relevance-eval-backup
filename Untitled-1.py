#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple, Any
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
_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "test_cases.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "full_results_output.csv")
REVIEW_CSV = str(_SCRIPT_DIR / "human_review_needed.csv")

REQUEST_TIMEOUT = 60
MAX_WORKERS = 10  # 建议先调小一点进行测试

# =========================
# 👨‍⚖️ Agent A: 独立法官 (The Judge)
# =========================
JUDGE_PROMPT_TEMPLATE = r"""
# Role
你是一个极其严谨的搜索质量策略专家。你的任务是诊断搜索 Bad Case，判断其"相关性"缺陷的类型。

# 核心原则
【严格区分"同领域"与"场景衍生"】
- "同领域"必须是：用户的核心任务目标完全一致，仅具体对象发生同级替换。
- "场景衍生"是：宏观领域一致，但用户要解决的问题（任务/动作）发生了偏移。

# Task
输入：[Query]、[result_Title]、[result_Summary]
请严格按照下方的【判定优先级原则】进行诊断。

=========================================
# 判定优先级原则（严格按顺序！）

【第一步：宏观领域大类判断】
1. 判断双方是否属于同一个【超大行业/超大类目】（如编程、教育、影视、游戏等）。
2. 若完全跨界（如"苹果手机"→"苹果种植"），检查字面诱导：
   - 存在字面同音/同字诱导 → 【0分：query理解有误】
   - 毫无交集 → 【0分：完全不相关】

【第二步：核心诉求与实体的二维判断（关键！）】
若宏观领域一致，请按以下标准严格区分两种1分泛化：

- 类别A【推荐同领域内容（平行实体替换）】：
  定义：用户的【核心动作/诉求】没变，仅仅是【核心实体】被替换成了同分类下的其他兄弟实体。
  💡 核心特征：换了人、换了物、换了剧。
  - 判例1：搜“猫咪吐毛球” → 给“狗狗护理”。（实体：猫变狗。判定：同领域）
  - 判例2：搜“周杰伦演唱会” → 给“林俊杰演唱会”。（实体：周杰伦变林俊杰。判定：同领域）
  - 判例3：搜“剪映教程” → 给“PR教程”。（实体：剪映变PR。判定：同领域）
  - 判例4：搜“红楼梦讲解” → 给“西游记讲解”。（实体：红楼梦变西游记。判定：同领域）

- 类别B【推荐场景衍生内容（诉求/形式偏移）】：
  定义：【核心实体】可能没变，但用户要做的【核心动作/诉求/形式】发生了明显的偏移（如：看正片变成了看花絮、看外观变成了看评测、买东西变成了看维修）。
  💡 核心特征：人/物没变，但讨论的角度变了。
  - 判例5：搜“流浪地球2导演” → 给“郭帆作品合集”。（诉求：从特定电影导演 偏移到了 导演个人作品合集。判定：场景衍生）
  - 判例6：搜“周生如故结局” → 给“周生如故幕后花絮”。（诉求：从正片结局 偏移到了 幕后娱乐。判定：场景衍生）
  - 判例7：搜“王者荣耀妲己皮肤” → 给“英雄联盟妲己角色解析”。（诉求：从看皮肤外观 偏移到了 解析战斗技能与游戏对比。判定：场景衍生）
  - 判例8：搜“泰坦尼克号主题曲” → 给“泰坦尼克号特效技术”。（诉求：从听音乐 偏移到了 看技术解析。判定：场景衍生）
  
【第三步：修饰词丢失检查】
- 类别C【丢词搜不准（纵向降维）】：
  定义：Query 的【核心主语】没有发生横向替换，用户的【动作】也没有偏移，仅仅是丢失了修饰该主语的【具体限制词/定语】（如：年份、型号、特定版本、适用人群）。
  💡 判例1：2024跨年 变 2023跨年（丢了正确年份）。
  💡 判例2：婴儿浴巾 变 家用成人浴巾（丢了“婴儿”这个适用人群限制词）。
  💡 判例3：iPhone 15 Pro 变 iPhone 15（丢了“Pro”这个型号限制词）。

=========================================
# 归因标准字典（仅限以下5选1）
- 完全不相关
- 丢词搜不准
- query理解有误
- 推荐同领域内容
- 推荐场景衍生内容

=========================================
# Output Format (严格JSON)
{
  "step1_domain": "宏观领域判断结果",
  "step2_core_entity": "Query核心实体 vs Result核心实体",
  "step2_core_task": "Query核心任务 vs Result核心任务",
  "step2_is_same_level": "是否满足同级替换严格标准（true/false/不适用）",
  "score": 0,
  "bad_case_tag": "必须是5个标准标签之一",
  "final_reason": "简述判断依据，需明确说明核心实体和核心任务是否一致"
}

=========================================
# 待诊断输入数据
[Query]: {query}
[result_Title]: {result_title}
[result_Summary]: {result_summary}
""".strip()

# =========================
# 🕵️‍♂️ Agent B: 独立对账员 (The Independent Auditor)
# =========================
AUDITOR_PROMPT_TEMPLATE = r"""
# Role
你是一个极度死板、专门负责“强行寻找联系”的对账员。你的任务不是追求判对，而是追求【拦截法官的滥杀】。

# 核心判定红线（你必须执行最宽松的关联标准）

【红线1：行业大类兜底（拦截 0 分的关键）】
- 逻辑：只要 Q 和 R 属于同一个大行业，你【绝对不准】判定为“完全不相关”或“理解有误”。
- 行业定义：
  - 办公：Word, Excel, PPT, PDF, 钉钉等。
  - 教育：英语, 数学, 考研, 公考等。
  - ACG：海贼王, 死神, 火影, 咒术回战等。
  - 运动：足球, 篮球, 健身, 减肥等。
- 动作：只要属于上述大行业，哪怕主语换了（如：Excel变Word），你必须判定为【推荐同领域内容】。

【红线2：核心动作一致（拦截 0 分）】
- 逻辑：剥离 Query 的动作（学、买、看、查、做、连）。
- 动作：只要 Result 也是在做同样的动作，哪怕主语平替（如：A星变B星、A药变B药），你必须判定为【推荐同领域内容】。

【红线3：核心主语保留（拦截 1 分偏移）】
- 逻辑：只要 Result 标题里还保留着 Query 的核心主语（如：黑神话、iPhone）。
- 动作：如果 Result 只是没包含具体年份、地点、价格，你必须判定为【丢词搜不准】。

# 归因标签（必须 5 选 1）
- 完全不相关
- 丢词搜不准
- query理解有误
- 推荐同领域内容
- 推荐场景衍生内容

# 执行指令
法官喜欢判 0 分（不相关），你的天职是尝试寻找证据判 1 分（同领域/衍生/丢词）。只要你给了 1 分而法官给了 0 分，我们就成功拦截了一个潜在错误！

# Output Format (严格JSON)
{
  "bad_case_tag": "必须是5个标准标签之一",
  "independent_reason": "简述你发现的行业或动作联系"
}

[Query]: {query}
[Title]: {result_title}
[Summary]: {result_summary}
"""

# =========================
# 🛠️ 辅助工具函数
# =========================
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gbk", "utf-8", "gb18030"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError(f"无法读取文件 {path}")

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try: return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return {}

def normalize_tag(tag: Any) -> str:
    return re.sub(r"\s+", "", str(tag or "")).lower()

# =========================
# ⚙️ 双盲并联执行核心
# =========================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def single_agent_call(client, prompt):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,
    )
    return safe_json_loads(resp.choices[0].message.content if resp.choices else "")

def call_parallel_agents(client, query, title, summary):
    # 使用 .replace 而不是 .format，防止 JSON 大括号导致的 KeyError
    p1 = (JUDGE_PROMPT_TEMPLATE
          .replace("{query}", query)
          .replace("{result_title}", title)
          .replace("{result_summary}", summary))
    
    p2 = (AUDITOR_PROMPT_TEMPLATE
          .replace("{query}", query)
          .replace("{result_title}", title)
          .replace("{result_summary}", summary))

    with ThreadPoolExecutor(max_workers=2) as inner_exec:
        future_j = inner_exec.submit(single_agent_call, client, p1)
        future_a = inner_exec.submit(single_agent_call, client, p2)
        
        judge_res = future_j.result()
        audit_res = future_a.result()
        
    return judge_res, audit_res

def process_row(index, row, client):
    q, t, s = str(row.get("query", "")), str(row.get("result_title", "")), str(row.get("result_summary", ""))
    exp_tag = str(row.get("expected_tag", ""))
    
    try:
        # 【第一步：先执行并联请求】
        judge, audit = call_parallel_agents(client, q, t, s)
        
        # 【第二步：提取 LLM 返回的标签内容】
        j_tag_raw = judge.get("bad_case_tag", "Unknown")
        a_tag_raw = audit.get("bad_case_tag", "Unknown")

        # 【第三步：归一化处理】
        judge_tag = normalize_tag(j_tag_raw)
        auditor_tag = normalize_tag(a_tag_raw)

        # 1. 标签映射表（解决“不相关”vs“完全不相关”的字面差异）
        alias_map = {
            "不相关": "完全不相关",
            "理解有误": "query理解有误",
            "平替": "推荐同领域内容",
            "衍生": "推荐场景衍生内容"
        }
        j_final = alias_map.get(judge_tag, judge_tag)
        a_final = alias_map.get(auditor_tag, auditor_tag)

        # 2. 势能区划分（防止在“平替”和“衍生”之间过度报警）
        zero_zone = ["完全不相关", "query理解有误"]
        one_zone = ["丢词搜不准", "推荐同领域内容", "推荐场景衍生内容"]

        # 3. 核心碰撞判定逻辑
        if j_final == a_final:
            is_collision = False
        # 只有在“0分”和“1分”之间产生分歧，才触发报警（碰撞）
        elif (j_final in zero_zone and a_final in one_zone) or (j_final in one_zone and a_final in zero_zone):
            is_collision = True
        else:
            # 都在 1 分区（如同领域 vs 衍生），不报警，信任法官
            is_collision = False
        
        return {
            "index": index,
            "llm_tag": j_tag_raw,
            "llm_reason": judge.get("final_reason", ""),
            "auditor_tag": a_tag_raw,
            "auditor_reason": audit.get("independent_reason", ""),
            "is_collision": is_collision,
            "is_correct": normalize_tag(exp_tag) == normalize_tag(j_tag_raw)
        }
    except Exception as e:
        print(f"\n[Error at index {index}]: {traceback.format_exc()}")
        return {
            "index": index, "llm_tag": "Error", "llm_reason": str(e),
            "auditor_tag": "Error", "auditor_reason": "逻辑处理失败",
            "is_collision": True, "is_correct": False
        }

# =========================
# 🏁 主程序
# =========================
def main():
    if not API_KEY: return print("[错误] 未设置 API KEY")
    if not os.path.exists(INPUT_CSV): return print(f"[错误] 找不到输入文件: {INPUT_CSV}")

    df = read_csv_with_fallback(INPUT_CSV)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = [None] * len(df)

    print(f"\n[开始] 双盲并联评估 (总数: {len(df)}, 并发: {MAX_WORKERS})...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, i, r, client): i for i, r in enumerate(df.to_dict("records"))}
        for future in tqdm(as_completed(futures), total=len(df), desc="进度"):
            res = future.result()
            results[res["index"]] = res

    # 结果整合
    res_df = df.copy()
    for col in ["llm_tag", "llm_reason", "auditor_tag", "auditor_reason", "is_collision", "is_correct"]:
        res_df[col] = [results[i][col] for i in range(len(df))]

    res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    res_df[res_df["is_collision"] == True].to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time

    # =========================
    # 📈 高阶并联指标统计
    # =========================
    total_cases = len(res_df)

    # 1. 法官独立表现
    judge_correct_count = res_df['is_correct'].sum()
    judge_accuracy = judge_correct_count / total_cases
    judge_wrong_df = res_df[res_df['is_correct'] == False]
    total_judge_errors = len(judge_wrong_df)

    # 2. 盲审碰撞效能
    # 成功召回：法官错了，且两人标签不同（碰撞成功）
    successfully_caught = judge_wrong_df['is_collision'].sum()
    collision_recall = successfully_caught / total_judge_errors if total_judge_errors > 0 else 0
    
    # 3. 剩余风险 (静默错误)
    # 漏网之鱼：法官错了，但两人标签一样（碰撞失败）
    silent_errors_count = total_judge_errors - successfully_caught
    residual_error_rate = silent_errors_count / total_cases

    print(f"\n{'='*50}")
    print(f"📊 并联对账系统效能报告")
    print(f"{'='*50}")
    print(f"完成！耗时: {elapsed:.2f}s")
    print(f"1. [基准] 法官独立准确率: {judge_accuracy:.2%}")
    print(f"   - 在 {total_cases} 条数据中，法官判对了 {judge_correct_count} 条")
    print(f"   - 原始错误存量: {total_judge_errors} 条")
    print(f"-"*50)
    print(f"2. [核心] 盲审召回率 (Recall): {collision_recall:.2%}")
    print(f"   - 成功通过“标签分歧”拦截了 {successfully_caught} 个法官错误")
    print(f"   - 拦截后的人工复核池大小: {res_df['is_collision'].sum()} 条")
    print(f"-"*50)
    print(f"3. [风险] 剩余漏网率 (Residual Error): {residual_error_rate:.2%}")
    print(f"   - 共有 {silent_errors_count} 条错误因两人“错得一致”而逃逸")
    print(f"   - 最终入库数据的潜在纯度约为: {1 - residual_error_rate:.2%}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()