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
_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "new_100_test_cases.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "new_full_results_output.csv")
REVIEW_CSV = str(_SCRIPT_DIR / "new_human_review_needed.csv")

MAX_WORKERS = 10 

# =========================
# 🧠 全知法官 (The Self-Reflective Judge)
# =========================
JUDGE_PROMPT = r"""
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
  "thought_process": "1.排查跨界与多义词... 2.套用8个经典判例比对... 3.得出最终结论",
  "bad_case_tag": "必须是5个标准标签之一",
  "confidence_score": 85,
  "needs_human_review": true/false
}

※ 填表说明：
- thought_process: 必须强制写出你的思考步骤。
- confidence_score: 填写 1-100 的整数。表示你对本次判定的绝对把握。
- needs_human_review: 如果 confidence_score 低于 85，或者你在（同领域 vs 衍生）等模糊边界难以抉择，必须填 true，交由人类复核。

[Query]: {query}
[Title]: {result_title}
[Summary]: {result_summary}
"""

# =========================
# 🛠️ 辅助函数
# =========================
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gbk", "utf-8", "gb18030"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError(f"无法读取文件 {path}")

def safe_json_loads(text: str) -> Dict[str, Any]:
    try: return json.loads(re.search(r"\{.*\}", text or "", flags=re.DOTALL).group(0))
    except: return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(client, prompt):
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0, response_format={"type": "json_object"}
    )
    return safe_json_loads(resp.choices[0].message.content)

def process_row_v30(index, row, client):
    q = str(row.get("query", ""))
    t = str(row.get("result_title", ""))
    s = str(row.get("result_summary", ""))
    exp_tag = str(row.get("expected_tag", "")).strip().lower()
    
    try:
        prompt = JUDGE_PROMPT.replace("{query}", q).replace("{result_title}", t).replace("{result_summary}", s)
        res = call_llm(client, prompt)
        
        pred_tag = res.get("bad_case_tag", "Unknown")
        thought = res.get("thought_process", "")
        needs_review = str(res.get("needs_human_review", "false")).lower() == "true"
        
        return {
            "index": index,
            "llm_tag": pred_tag,
            "llm_reason": thought,           # 记录详细的思考过程
            "confidence": res.get("confidence_score", 0),
            "is_collision": needs_review,    # 直接映射为是否需要复核
            "is_correct": re.sub(r"\s+", "", pred_tag).lower() == re.sub(r"\s+", "", exp_tag)
        }
    except Exception as e:
        print(f"\n[Error at index {index}]: {traceback.format_exc()}")
        return {"index": index, "llm_tag": "Error", "llm_reason": str(e), "is_collision": True, "is_correct": False}

# =========================
# 🏁 主程序
# =========================
def main():
    if not API_KEY: return print("[错误] 未设置 API KEY")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig') 
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = [None] * len(df)

    print(f"\n[开始] 全知法官评估 (总数: {len(df)}, 并发: {MAX_WORKERS})...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row_v30, i, r, client): i for i, r in enumerate(df.to_dict("records"))}
        for future in tqdm(as_completed(futures), total=len(df), desc="处理进度"):
            res = future.result()
            results[res["index"]] = res

    res_df = df.copy()
    for col in ["llm_tag", "llm_reason", "confidence", "is_collision", "is_correct"]:
        res_df[col] = [results[i].get(col, "") for i in range(len(df))]

    res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    # 导出需要复核的记录
    review_df = res_df[res_df["is_collision"] == True]
    review_df.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    total = len(res_df)
    collisions = res_df["is_collision"].sum()
    final_accuracy = res_df["is_correct"].sum() / total

    print(f"\n{'='*50}\n📊 V30 全知法官系统报告\n{'='*50}")
    print(f"总耗时: {time.time() - start_time:.2f}s (速度大幅提升！)")
    print(f"🌟 系统最终纯净度 (Accuracy): {final_accuracy:.2%} 🌟")
    print(f"人工复核触发率: {collisions/total:.2%} (共 {collisions} 条机器不自信的Case)")
    print(f"复核数据已导出至: {REVIEW_CSV}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()