#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
API_KEY = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "test_cases.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "full_results_output.csv")
REVIEW_CSV = str(_SCRIPT_DIR / "human_review_needed.csv")  # 导出需人工复核的 case

REQUEST_TIMEOUT = 60
MAX_WORKERS = 20

# =========================
# 🧠 Agent A: 原始 V6 法官 (The Judge)
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
# 🧠 Agent B: 审计专家 (The Auditor)
# =========================
AUDITOR_PROMPT_TEMPLATE = r"""
# Role
你是一个极度死板、铁面无私的搜索质量合规质检员。你的唯一工作是执行“字面逻辑核对”和“常识常理纠偏”。严禁过度解读用户的潜在意图。

# 输入信息
[Query]: {query}
[Result]: {result_title}
[法官判定标签]: {judge_tag}
[法官判定理由]: {judge_reason}

# 合规审查清单（严格按顺序执行）

【红线1：法官是否在“跨界领域”上撒谎？】（抓漏网之鱼）
- 检查动作：如果法官的标签是【完全不相关】或【query理解有误】，你必须用自己的常识看一眼 Query 和 Result。
- 触发条件：如果它们明明属于同一个大众认知的行业/圈层（如：iPhone与华为同属手机、考研英语与数学同属教育、海贼王与死神同属动漫），法官却说“完全跨界无交集/完全不相关”，这是法官在撒谎！
- 判决：必须质疑 (Low)，理由写：“同属XX领域，法官误判为完全不相关”。

【红线2：法官的“理由”和“标签”是否字面冲突？】（抓逻辑断裂）
- 检查动作：只看文字表面，不加主观理解。
- 触发条件 A：法官理由写了“任务/诉求/形式/动作发生了偏移”，但最后给的标签是【同领域】。
- 触发条件 B：法官理由写了“仅仅是对象/实体发生了同级替换”，但最后给的标签是【场景衍生】。
- 判决：必须质疑 (Medium)，理由写：“理由与标签字面冲突”。

【红线3：防杠精绝对纪律】（防误伤）
- 只要不触发红线1和红线2，你【绝对禁止】去质疑法官的分类。
- 严禁自行脑补：“我认为猫看病和狗洗澡意图不同”、“我认为这算丢词”。
- 只要法官的文字逻辑是自洽的，哪怕你觉得有点牵强，也必须无条件放行！

# Output Format (严格JSON)
{
  "confidence_level": "High/Medium/Low",
  "uncertainty_reason": "只填触发了哪条红线及简述 / 未触发红线填'无'",
  "is_audit_agree": "true/false"
}
"""
# =========================
# 辅助函数
# =========================
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    # 依次尝试：带签名的UTF8 -> 中文GBK -> 标准UTF8 -> 扩展中文
    for enc in ["utf-8-sig", "gbk", "utf-8", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件 {path}，请确认文件格式是否正确。")


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = re.sub(r"\s+", "", t)
    return t.lower()


def is_tag_match(expected_tag: str, llm_tag: str) -> bool:
    exp, pred = normalize_text(expected_tag), normalize_text(llm_tag)
    if not exp or not pred:
        return False
    return exp == pred or exp in pred or pred in exp


# =========================
# ⚙️ 并发流水线逻辑
# =========================

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_agents(client: OpenAI, query: str, title: str, summary: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p1 = (
        JUDGE_PROMPT_TEMPLATE.replace("{query}", query)
        .replace("{result_title}", title)
        .replace("{result_summary}", summary)
    )
    resp1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": p1}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,
    )
    raw1 = resp1.choices[0].message.content if resp1.choices else ""
    judge_res = safe_json_loads(raw1)

    p2 = (
        AUDITOR_PROMPT_TEMPLATE.replace("{query}", query)
        .replace("{result_title}", title)
        .replace("{judge_tag}", str(judge_res.get("bad_case_tag", "")))
        .replace("{judge_reason}", str(judge_res.get("final_reason", "")))
    )
    resp2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": p2}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,
    )
    raw2 = resp2.choices[0].message.content if resp2.choices else ""
    audit_res = safe_json_loads(raw2)

    return judge_res, audit_res

def process_single_row(index, row, client):
    q, t, s = str(row.get("query", "")), str(row.get("result_title", "")), str(row.get("result_summary", ""))
    try:
        judge, audit = call_agents(client, q, t, s)
        pred_tag = str(judge.get("bad_case_tag", "")).strip()
        exp_tag = str(row.get("expected_tag", ""))
        return {
            "index": index,
            "llm_tag": pred_tag,
            "llm_reason": judge.get("final_reason", ""),
            "confidence_level": str(audit.get("confidence_level", "High")),
            "uncertainty_reason": str(audit.get("uncertainty_reason", "无")),
            "is_audit_agree": str(audit.get("is_audit_agree", "true")),
            "is_correct": is_tag_match(exp_tag, pred_tag),
        }
    except Exception as e:
        # 修复了异常字典漏字段导致 Pandas KeyError 的 BUG
        return {
            "index": index, 
            "llm_tag": "Error", 
            "llm_reason": "程序执行异常",
            "confidence_level": "Low", # 遇到错误强行标记为 Low 以便人工排查
            "uncertainty_reason": f"代码报错: {str(e)}",
            "is_audit_agree": "false",
            "is_correct": False
        }

# =========================
# 🏁 主程序
# =========================
def main():
    if not API_KEY:
        print("[错误] 请设置环境变量 DEEPSEEK_API_KEY 后重试。")
        return

    df = read_csv_with_fallback(INPUT_CSV)
    required = ["query", "result_title", "result_summary", "expected_tag"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"输入 CSV 缺少列: {miss}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results: list = [None] * len(df)

    print("\n[开始] 双 Agent 串联评估 (V6 Judge + Auditor)...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        records = df.to_dict("records")
        futures = {executor.submit(process_single_row, i, r, client) for i, r in enumerate(records)}
        for future in tqdm(as_completed(futures), total=len(df), desc="进度"):
            res = future.result()
            results[res["index"]] = res

    if any(r is None for r in results):
        raise RuntimeError("部分行未返回结果，请检查并发与异常逻辑。")

    res_df = df.copy().reset_index(drop=True)
    for col in (
        "llm_tag",
        "llm_reason",
        "confidence_level",
        "uncertainty_reason",
        "is_audit_agree",
        "is_correct",
    ):
        res_df[col] = [results[i][col] for i in range(len(df))]

    res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    conf_norm = res_df["confidence_level"].astype(str).str.strip().str.lower()
    audit_disagree = res_df["is_audit_agree"].astype(str).str.lower() == "false"
    review_needed = res_df[(conf_norm != "high") | audit_disagree]
    review_needed.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    # =========================
    # 📈 新增：多维高阶评估指标
    # =========================
    total_cases = len(res_df)
    
    # 1. 法官指标 (Judge Metrics)
    judge_correct = res_df['is_correct'].sum()
    judge_accuracy = judge_correct / total_cases if total_cases > 0 else 0
    judge_wrong_total = total_cases - judge_correct

    # 2. 审计员指标 (Auditor Metrics)
    flagged_total = len(review_needed) # 审计员举手的总次数
    
    # 有效拦截：法官判错了，且审计员成功举手了 (True Positive)
    true_interception = len(review_needed[review_needed['is_correct'] == False])
    
    # 误伤：法官其实判对了，但审计员觉得有问题 (False Positive)
    false_interception = len(review_needed[review_needed['is_correct'] == True])
    
    # 漏网之鱼：法官判错了，但审计员没发现 (False Negative)
    missed_errors = judge_wrong_total - true_interception

    sep = "=" * 40
    elapsed = time.time() - start_time
    print(f"\n[完成] 耗时: {elapsed:.2f} 秒")
    print(sep)
    print(" [Agent A 法官]")
    print(f"    - 法官对齐率: {judge_accuracy:.2%} ({judge_correct}/{total_cases})")
    print(f"    - 法官判错数: {judge_wrong_total}")
    print(sep)
    print(" [Agent B 审计]")
    print(f"    - 质疑条数: {flagged_total}")
    tp_rate = (true_interception / judge_wrong_total) if judge_wrong_total else 0.0
    fp_rate = (false_interception / judge_correct) if judge_correct else 0.0
    print(f"    - 错案中被标记: {tp_rate:.2%} ({true_interception}/{judge_wrong_total})")
    print(f"    - 对案中被误标: {fp_rate:.2%} ({false_interception}/{judge_correct})")
    print(f"    - 错案未标记数: {missed_errors}")
    print(sep)
    print(f" 待复核列表: {REVIEW_CSV} (共 {flagged_total} 条)")

    accuracy = float(res_df["is_correct"].mean())
    print(f"\n全量对齐率: {accuracy:.2%}")
    print(f"待复核导出: {REVIEW_CSV} (共 {len(review_needed)} 条)")

if __name__ == "__main__":
    main()