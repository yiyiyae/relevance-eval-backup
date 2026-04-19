#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线批量评估搜索相关性脚本（V6 终极并发+任务一致性版）
- 采用多线程并发架构
- 引入 JTBD (用户任务) 决策树 Prompt
- 严格校验 JSON Schema
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# =========================
# ⚙️ 核心配置区
# =========================
API_KEY = (os.environ.get("DEEPSEEK_API_KEY") or "你的API_KEY填写在这里").strip()
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"  
_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "test_cases.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "results_output.csv")

REQUEST_TIMEOUT = 60  
MAX_WORKERS = 10  # 并发线程数

# =========================
# 🧠 V6 终极业务对齐 Prompt
# =========================
PROMPT_TEMPLATE = r"""
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

【第三步：实体缺失检查】
若核心实体未被替换为兄弟实体，核心诉求也未偏离，仅仅是丢失了具体的限制词（如年份2024、特定型号）：
- 【0分：丢词搜不准】
（注：2024年跨年 变成 2023年跨年，属于丢掉了2024的限制词，判为丢词搜不准）

=========================================
# 归因标准字典（仅限以下5选1）
- 完全不相关
- 丢词搜不准
- query理解有误
- 推荐同领域内容
- 推荐场景衍生内容

=========================================
# Output Format (严格JSON，请确保 key 名称一致)
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
# 🛠️ 辅助函数
# =========================
def build_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def normalize_text(s: Any) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()

def is_tag_match(expected_tag: str, llm_tag: str) -> bool:
    exp, pred = normalize_text(expected_tag), normalize_text(llm_tag)
    if not exp or not pred: return False
    return exp == pred or exp in pred or pred in exp

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    
    try: return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError: return {}

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"无法读取 CSV 文件: {path}")

# =========================
# ⚙️ 核心引擎执行层
# =========================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_llm_for_row(client: OpenAI, query: str, result_title: str, result_summary: str) -> Tuple[str, str, Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.replace("{query}", str(query)).replace("{result_title}", str(result_title)).replace("{result_summary}", str(result_summary))

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个严谨的搜索质量诊断助手。请严格输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,
    )

    data = safe_json_loads(resp.choices[0].message.content if resp.choices else "")
    return str(data.get("bad_case_tag", "")).strip(), str(data.get("final_reason", "")).strip(), data

def process_single_row(index: int, row: dict, client: OpenAI) -> tuple:
    try:
        pred_tag, pred_reason, _ = call_llm_for_row(client, row.get("query", ""), row.get("result_title", ""), row.get("result_summary", ""))
    except Exception as e:
        pred_tag, pred_reason = "", f"API调用失败: {e}"
    
    correct = is_tag_match(str(row.get("expected_tag", "")), pred_tag)
    return index, pred_tag, pred_reason, correct

# =========================
# 🏁 主程序入口
# =========================
def main():
    if not API_KEY or API_KEY == "你的API_KEY填写在这里":
        print("⚠️ 报错：请在代码最上方配置你的 API_KEY")
        return

    df = read_csv_with_fallback(INPUT_CSV)
    client = build_client()

    llm_tags, llm_reasons, is_correct_list = [""] * len(df), [""] * len(df), [False] * len(df)

    print(f"\n🚀 开始执行 V6 终极引擎并发诊断 (并发数: {MAX_WORKERS})...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        records = df.to_dict('records')
        futures = [executor.submit(process_single_row, i, record, client) for i, record in enumerate(records)]
        
        for future in tqdm(as_completed(futures), total=len(df), desc="评估进度"):
            idx, pred_tag, pred_reason, correct = future.result()
            llm_tags[idx], llm_reasons[idx], is_correct_list[idx] = pred_tag, pred_reason, correct

    df["llm_tag"], df["llm_reason"], df["is_correct"] = llm_tags, llm_reasons, is_correct_list
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    accuracy = (sum(is_correct_list) / len(is_correct_list)) if is_correct_list else 0.0
    
    print("\n✅ 诊断圆满完成！")
    print(f"⏱️ 耗时: {time.time() - start_time:.2f} 秒")
    print(f"📊 最终对齐率 (Accuracy): {accuracy:.2%} ({sum(is_correct_list)}/{len(is_correct_list)})")
    print(f"📁 详细结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()