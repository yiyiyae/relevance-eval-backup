#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import html
import json
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import chromadb
import pandas as pd
import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# =========================
# ⚙️ 核心配置区
# =========================
API_KEY = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 4
VOTE_SAMPLES = 3

_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(_SCRIPT_DIR / "test_cases_new.csv")
OUTPUT_CSV = str(_SCRIPT_DIR / "new_full_results_output.csv")
CHROMA_DB_PATH = str(_SCRIPT_DIR / "agent_vector_db")

TAG_DICT = {
    "1": "完全不相关",
    "2": "丢词搜不准",
    "3": "query理解有误",
    "4": "推荐同领域内容",
    "5": "推荐场景衍生内容",
}

SOURCE_AI = "AI"
SOURCE_HUMAN = "Human"

BILI_SEARCH_API = "https://api.bilibili.com/x/web-interface/search/type"
BILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILI_TAG_API = "https://api.bilibili.com/x/tag/archive/tags"
BILI_SEARCH_URL_TEMPLATE = "https://search.bilibili.com/video?keyword={keyword}"
BILI_CANDIDATE_SCAN_LIMIT = 30
BILI_PREFETCH_META_LIMIT = 20
GENERIC_BILI_TAGS = {
    "哔哩哔哩", "bilibili", "b站", "高清视频", "在线观看", "弹幕", "视频", "原创",
}


# =========================
# 🛠️ 通用辅助
# =========================
def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_title_for_match(value: Any) -> str:
    text = clean_text(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"_哔哩哔哩_bilibili$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[-_\s]*哔哩哔哩(?:_bilibili)?$", "", text, flags=re.IGNORECASE)
    chars: List[str] = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if ch.isspace():
            continue
        if cat.startswith("P") or cat.startswith("S"):
            continue
        chars.append(ch)
    return "".join(chars).strip()


def build_case_key(query: Any, title: Any, summary: Any) -> str:
    return " || ".join([
        normalize_text(query),
        normalize_text(title),
        normalize_text(summary),
    ])


def read_csv_with_fallback(path_or_buffer) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV: {last_error}")


def ensure_case_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "case_key" not in df.columns:
        required = {"query", "result_title", "result_summary"}
        if required.issubset(df.columns):
            df["case_key"] = df.apply(
                lambda r: build_case_key(r["query"], r["result_title"], r["result_summary"]), axis=1
            )
    return df


def load_output_df() -> pd.DataFrame:
    if not os.path.exists(OUTPUT_CSV):
        return pd.DataFrame()
    try:
        return ensure_case_key(read_csv_with_fallback(OUTPUT_CSV))
    except Exception:
        return pd.DataFrame()


def get_processed_case_keys() -> set:
    df = load_output_df()
    if df.empty:
        return set()
    if "case_key" in df.columns:
        return set(df["case_key"].astype(str).tolist())
    if "query" in df.columns:
        return set(df["query"].astype(str).tolist())
    return set()


def append_to_output(results_list: list):
    if not results_list:
        return

    df_new = pd.DataFrame(results_list)
    if "needs_intervention" in df_new.columns:
        df_new = df_new.drop(columns=["needs_intervention"])
    if "reviewed" not in df_new.columns:
        df_new["reviewed"] = False
    df_new = ensure_case_key(df_new)

    df_old = load_output_df()
    if df_old.empty:
        df_combined = df_new
    else:
        df_combined = pd.concat([df_old, df_new], ignore_index=True)

    if "case_key" in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=["case_key"], keep="last")

    df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


# =========================
# 🎬 B 站标题回查 + 摘要助手
# =========================
class BilibiliVideoSummaryHelper:
    def __init__(self, client: OpenAI, model_name: str = MODEL_NAME, timeout: int = 10):
        self.client = client
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._primed = False

    def _prime_session(self):
        if self._primed:
            return
        try:
            self.session.get("https://www.bilibili.com/", timeout=self.timeout)
        except Exception:
            pass
        self._primed = True

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try:
            return json.loads(re.search(r"\{.*\}", text or "", flags=re.DOTALL).group(0))
        except Exception:
            return {}

    def _keyword_variants(self, title: str) -> List[str]:
        raw = clean_text(title)
        variants: List[str] = []
        candidates = [
            raw,
            raw.strip("《》〈〉【】[]()（）"),
            raw.replace("《", "").replace("》", ""),
            re.sub(r"\s+", " ", raw),
            re.sub(r"[《》〈〉【】\[\]()（）]", " ", raw),
            re.sub(r"[!！?？:：·•,，.。/\\\-—_~`'\"“”‘’]", " ", raw),
        ]
        for item in candidates:
            item = clean_text(item)
            if item and item not in variants:
                variants.append(item)
        return variants[:5]

    def _http_get_json(self, url: str, params: Optional[dict] = None) -> dict:
        self._prime_session()
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _http_get_text(self, url: str, params: Optional[dict] = None) -> str:
        self._prime_session()
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        if not resp.encoding:
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    def _search_videos_via_api(self, keyword: str) -> List[dict]:
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": 1,
            "order": "totalrank",
        }
        payload = self._http_get_json(BILI_SEARCH_API, params=params)
        result_list = (((payload or {}).get("data") or {}).get("result") or [])
        rows: List[dict] = []
        for item in result_list:
            title = clean_text(item.get("title", ""))
            arcurl = clean_text(item.get("arcurl", ""))
            bvid = clean_text(item.get("bvid", ""))
            if not bvid and arcurl:
                m = re.search(r"/video/(BV[0-9A-Za-z]+)", arcurl)
                if m:
                    bvid = m.group(1)
            rows.append({
                "video_title": title,
                "video_url": arcurl,
                "bvid": bvid,
                "desc": clean_text(item.get("description", "")),
                "uploader": clean_text(item.get("author", "")),
                "tags": [t for t in clean_text(item.get("tag", "")).split(",") if clean_text(t)],
                "source": "api_search",
            })
        return rows

    def _search_videos_via_html(self, keyword: str) -> List[dict]:
        search_url = BILI_SEARCH_URL_TEMPLATE.format(keyword=quote(keyword))
        html_text = self._http_get_text(search_url)
        if "验证码" in html_text and "哔哩哔哩" in html_text:
            raise RuntimeError("搜索页触发验证码")

        rows: List[dict] = []
        seen = set()
        pattern = re.compile(
            r'<a[^>]+href=["\'](?P<href>(?:https:)?//www\.bilibili\.com/video/(?P<bvid>BV[0-9A-Za-z]+)[^"\']*)["\'][^>]*?(?:title=["\'](?P<title1>.*?)["\'])?[^>]*>(?P<body>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html_text):
            href = clean_text(match.group("href"))
            if href.startswith("//"):
                href = "https:" + href
            bvid = clean_text(match.group("bvid"))
            title = clean_text(match.group("title1") or match.group("body"))
            key = bvid or href
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({
                "video_title": title,
                "video_url": href,
                "bvid": bvid,
                "desc": "",
                "uploader": "",
                "tags": [],
                "source": "html_search",
            })
        return rows

    def _fetch_video_meta(self, candidate: dict) -> dict:
        bvid = clean_text(candidate.get("bvid", ""))
        video_url = clean_text(candidate.get("video_url", ""))
        title = clean_text(candidate.get("video_title", ""))
        if not bvid and video_url:
            m = re.search(r"/video/(BV[0-9A-Za-z]+)", video_url)
            if m:
                bvid = m.group(1)

        meta = {
            "video_title": title,
            "video_url": video_url,
            "bvid": bvid,
            "desc": clean_text(candidate.get("desc", "")),
            "uploader": clean_text(candidate.get("uploader", "")),
            "tags": list(candidate.get("tags", []) or []),
            "source": candidate.get("source", "unknown"),
        }

        if bvid:
            try:
                payload = self._http_get_json(BILI_VIEW_API, params={"bvid": bvid})
                data = (payload or {}).get("data") or {}
                meta["video_title"] = clean_text(data.get("title", "")) or meta["video_title"]
                meta["desc"] = clean_text(data.get("desc", "")) or meta["desc"]
                owner = data.get("owner") or {}
                meta["uploader"] = clean_text(owner.get("name", "")) or meta["uploader"]
                if not meta["video_url"]:
                    meta["video_url"] = f"https://www.bilibili.com/video/{bvid}"
                aid = data.get("aid")
                if aid:
                    try:
                        tag_payload = self._http_get_json(BILI_TAG_API, params={"aid": aid})
                        tag_data = (tag_payload or {}).get("data") or []
                        tags = [clean_text(x.get("tag_name", "")) for x in tag_data if clean_text(x.get("tag_name", ""))]
                        if tags:
                            meta["tags"] = tags
                    except Exception:
                        pass
            except Exception:
                pass

        if meta["video_url"] and (not meta["video_title"] or not meta["tags"]):
            try:
                page = self._http_get_text(meta["video_url"])
                if not meta["video_title"]:
                    m = re.search(r"<title>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        meta["video_title"] = clean_text(m.group(1)).replace("_哔哩哔哩_bilibili", "")
                if not meta["desc"]:
                    m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', page, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        meta["desc"] = clean_text(m.group(1))
                if not meta["uploader"]:
                    m = re.search(r'<meta[^>]+name=["\']author["\'][^>]+content=["\'](.*?)["\']', page, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        meta["uploader"] = clean_text(m.group(1))
                if not meta["tags"]:
                    m = re.search(r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\'](.*?)["\']', page, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        tags = [clean_text(t) for t in clean_text(m.group(1)).split(",") if clean_text(t)]
                        tags = [t for t in tags if normalize_title_for_match(t) not in {normalize_title_for_match(meta["video_title"]), ""} and t.lower() not in GENERIC_BILI_TAGS]
                        if tags:
                            meta["tags"] = tags[:8]
            except Exception:
                pass

        meta["tags"] = [t for t in [clean_text(x) for x in meta["tags"]] if t and t.lower() not in GENERIC_BILI_TAGS]
        return meta

    def _merge_candidates(self, rows: List[dict]) -> List[dict]:
        merged: List[dict] = []
        seen = set()
        for item in rows:
            key = clean_text(item.get("bvid", "")) or clean_text(item.get("video_url", "")) or normalize_title_for_match(item.get("video_title", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def summarize_from_meta(self, title: str, desc: str, tags: List[str], uploader: str = "") -> str:
        prompt = f"""
你是一个视频内容摘要助手。
请根据以下信息生成 30-60 字的简短摘要，仅用于辅助判断搜索结果相关性。

要求：
1. 只能依据提供的信息，不要脑补。
2. 输出简洁、客观，不要套话。
3. 优先概括视频主题、对象、核心内容。

[视频标题]
{title}

[视频简介]
{desc}

[视频标签]
{', '.join(tags)}

[UP主]
{uploader}

请输出 JSON：{{"summary": "..."}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = self._safe_json_loads(resp.choices[0].message.content)
            return clean_text(data.get("summary", ""))
        except Exception:
            parts = [clean_text(title), clean_text(desc)] + [clean_text(t) for t in tags[:3]]
            parts = [p for p in parts if p]
            return "；".join(parts)[:80]

    def enrich_summary_from_bilibili_title(self, result_title: str) -> Dict[str, Any]:
        raw_title = clean_text(result_title)
        cache_key = normalize_title_for_match(raw_title)
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        search_link = BILI_SEARCH_URL_TEMPLATE.format(keyword=quote(raw_title))
        keyword_variants = self._keyword_variants(raw_title)
        routes = []
        debug_detail = ""
        all_candidates: List[dict] = []

        for keyword in keyword_variants:
            try:
                api_rows = self._search_videos_via_api(keyword)
                if api_rows:
                    all_candidates.extend(api_rows)
                    routes.append(f"api:{keyword}")
            except Exception as exc:
                debug_detail += f"api[{keyword}]={exc}; "

            if len(all_candidates) < BILI_CANDIDATE_SCAN_LIMIT:
                try:
                    html_rows = self._search_videos_via_html(keyword)
                    if html_rows:
                        all_candidates.extend(html_rows)
                        routes.append(f"html:{keyword}")
                except Exception as exc:
                    debug_detail += f"html[{keyword}]={exc}; "

        candidates = self._merge_candidates(all_candidates)
        target_norm = normalize_title_for_match(raw_title)

        result: Dict[str, Any] = {
            "matched": False,
            "summary": "",
            "search_title": raw_title,
            "matched_title": "",
            "video_url": "",
            "desc": "",
            "tags": [],
            "uploader": "",
            "message": "",
            "lookup_stage": "search_failed",
            "debug_detail": debug_detail.strip(),
            "candidate_titles": [clean_text(x.get("video_title", "")) for x in candidates[:10] if clean_text(x.get("video_title", ""))],
            "keyword_variants": keyword_variants,
            "search_route": " | ".join(routes) if routes else "none",
            "search_link": search_link,
            "scanned_candidates": min(len(candidates), BILI_CANDIDATE_SCAN_LIMIT),
        }

        if not candidates:
            result["lookup_stage"] = "no_candidates"
            result["message"] = "B站回查失败：未找到任何候选视频。"
            self._cache[cache_key] = dict(result)
            return result

        scanned = candidates[:BILI_CANDIDATE_SCAN_LIMIT]
        exact_match = None
        for item in scanned:
            if normalize_title_for_match(item.get("video_title", "")) == target_norm:
                exact_match = item
                break

        prefetched_candidates = []
        for item in scanned[:BILI_PREFETCH_META_LIMIT]:
            meta = self._fetch_video_meta(item)
            prefetched_candidates.append(meta)
            if normalize_title_for_match(meta.get("video_title", "")) == target_norm:
                exact_match = meta
                break

        if exact_match is None:
            result["lookup_stage"] = "title_mismatch"
            result["candidate_titles"] = [clean_text(x.get("video_title", "")) for x in prefetched_candidates[:10] if clean_text(x.get("video_title", ""))] or result["candidate_titles"]
            result["message"] = f"B站回查失败：已扫描前 {result['scanned_candidates']} 个候选，仍未命中规范化精确标题。"
            self._cache[cache_key] = dict(result)
            return result

        meta = self._fetch_video_meta(exact_match)
        summary = self.summarize_from_meta(
            meta.get("video_title", raw_title),
            meta.get("desc", ""),
            meta.get("tags", []),
            meta.get("uploader", ""),
        )

        result.update({
            "matched": True,
            "summary": summary,
            "matched_title": meta.get("video_title", raw_title),
            "video_url": meta.get("video_url", ""),
            "desc": meta.get("desc", ""),
            "tags": meta.get("tags", []),
            "uploader": meta.get("uploader", ""),
            "message": "B站标题回查成功，已自动生成摘要。",
            "lookup_stage": "completed",
            "candidate_titles": [clean_text(x.get("video_title", "")) for x in prefetched_candidates[:10] if clean_text(x.get("video_title", ""))] or result["candidate_titles"],
        })
        self._cache[cache_key] = dict(result)
        return result


# =========================
# 🧠 具备 RAG 长期记忆的 Agent 核心类
# =========================
class RagSearchEvalAgent:
    def __init__(self, db_path: str):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="eval_rules")
        self.video_helper = BilibiliVideoSummaryHelper(self.client, MODEL_NAME)
        print(f"📦 [向量库就绪] 当前知识库已包含 {self.collection.count()} 条规则。")

    def learn(self, query: str, correct_tag: str, human_rule: str, source: str = SOURCE_HUMAN):
        created_at_ms = int(time.time() * 1000)
        doc_id = f"rule_{uuid.uuid4().hex}"
        rule_text = normalize_text(human_rule)
        self.collection.add(
            documents=[f"query: {normalize_text(query)}\nrule: {rule_text}"],
            metadatas=[{
                "query": normalize_text(query),
                "correct_tag": normalize_text(correct_tag),
                "human_rule": rule_text,
                "source": normalize_text(source) or SOURCE_HUMAN,
                "created_at_ms": created_at_ms,
            }],
            ids=[doc_id],
        )
        print(f"✨ [Agent 顿悟] 规则 ({source}) 已写入记忆库！")

    def auto_extract_rule(self, query: str, title: str, summary: str, correct_tag: str, wrong_tag: str) -> str:
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
2. ⚠️ 严禁提及具体词汇。
3. 必须使用抽象描述。
4. 长度控制在 30 字以内。

# 输出格式 (JSON)
{{
  "abstract_rule": "总结的抽象准则"
}}
"""
        try:
            res = self._call_llm_with_retry(prompt, temperature=0.5)
            return res.get("abstract_rule", f"识别到{correct_tag}特征，修正原判{wrong_tag}")
        except Exception:
            return f"人类专家强制判定为{correct_tag}"

    def enrich_summary_from_bilibili_title(self, result_title: str) -> Dict[str, Any]:
        return self.video_helper.enrich_summary_from_bilibili_title(result_title)

    def _retrieve_relevant_rules(self, query: str, top_k: int = 3) -> List[dict]:
        total = self.collection.count()
        if total == 0:
            return []
        k = min(top_k, total)
        results = self.collection.query(query_texts=[normalize_text(query)], n_results=k)
        return results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try:
            return json.loads(re.search(r"\{.*\}", text or "", flags=re.DOTALL).group(0))
        except Exception:
            return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_with_retry(self, prompt: str, temperature: float = 0.0) -> dict:
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
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

# 判定大纲
【第一步：宏观领域跨界与多义词排雷】
1. 判断双方是否属于同一个【超大行业/超大类目】。
2. 若完全跨界（毫无交集），必须严格区分：
   - 存在“一词多义”导致概念偷换 → 判【query理解有误】
   - 纯粹跨界且无多义词歧义 → 判【完全不相关】

【第二步：核心诉求与实体的二维判断】
若宏观领域一致：
- 【推荐同领域内容】= 核心动作不变，仅核心实体平行替换。
- 【推荐场景衍生内容】= 核心实体可能没变，但动作/诉求/形式发生偏移。

【第三步：严苛修饰词丢失检查】
- 【丢词搜不准】= 主语没变、动作没变，只丢了修饰主语的具体限制词。
- 🚨 不要把主语当成限制词；丢了主语绝对不算丢词搜不准。
"""

        relevant_memory = self._retrieve_relevant_rules(query, top_k=3)
        memory_section = ""
        if relevant_memory:
            memory_section = (
                "\n=========================================\n"
                "# 💡 高优先级相关经验（由外部向量库召回）\n"
                "请高度重视以下历史规则：\n"
            )
            for i, mem in enumerate(relevant_memory, start=1):
                memory_section += (
                    f"- 相似历史案例{i}: 搜【{mem.get('query', '')}】时，"
                    f"适用规则【{mem.get('human_rule', '')}】 -> 结论是【{mem.get('correct_tag', '')}】\n"
                )

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
            final_prompt = base_prompt + memory_section + tail_prompt
            results = []
            for _ in range(VOTE_SAMPLES):
                res = self._call_llm_with_retry(final_prompt, temperature=0.6)
                if "bad_case_tag" in res:
                    results.append(res)

            if not results:
                return {"bad_case_tag": "Error", "confidence_score": 0, "thought_process": "API全量失败"}

            tags = [r.get("bad_case_tag", "Unknown") for r in results]
            tag_counts = Counter(tags)
            most_common_tag, count = tag_counts.most_common(1)[0]
            confidence = 100 if count == VOTE_SAMPLES else (66 if count == 2 else 33)
            final_reason = next((r.get("thought_process", "") for r in results if r.get("bad_case_tag") == most_common_tag), "")
            if count < VOTE_SAMPLES:
                final_reason += f"\n\n🚨 [系统提示]：AI 内部产生分歧，投票分布为 {dict(tag_counts)}"
            return {
                "thought_process": final_reason,
                "bad_case_tag": most_common_tag,
                "confidence_score": confidence,
            }
        except Exception as exc:
            return {"bad_case_tag": "Error", "confidence_score": 0, "thought_process": str(exc)}


# =========================
# ⚙️ 并发 Worker 函数
# =========================
def process_row(index, row, agent):
    q = normalize_text(row.get("query", ""))
    t = normalize_text(row.get("result_title", ""))
    s = normalize_text(row.get("result_summary", ""))
    exp_tag = normalize_text(row.get("expected_tag", "")).lower()

    res = agent.evaluate(q, t, s)
    pred_tag = res.get("bad_case_tag", "Unknown")
    confidence = int(res.get("confidence_score", 0))

    return {
        "index": index,
        "case_key": build_case_key(q, t, s),
        "query": q,
        "result_title": t,
        "result_summary": s,
        "expected_tag": exp_tag,
        "llm_tag": pred_tag,
        "llm_reason": res.get("thought_process", ""),
        "confidence": confidence,
        "is_correct": re.sub(r"\s+", "", pred_tag).lower() == re.sub(r"\s+", "", exp_tag),
        "needs_intervention": confidence < 95,
        "reviewed": confidence >= 95,
    }


# =========================
# 🏁 主程序
# =========================
def main():
    if not API_KEY:
        print("[错误] 未设置 API KEY")
        return

    df_all = ensure_case_key(read_csv_with_fallback(INPUT_CSV))
    processed_case_keys = get_processed_case_keys()
    df_todo = df_all[~df_all["case_key"].astype(str).isin(processed_case_keys)].copy() if processed_case_keys else df_all.copy()

    if df_todo.empty:
        print(f"\n🎉 恭喜！{INPUT_CSV} 中的所有数据都已处理完毕。结果在 {OUTPUT_CSV}")
        return

    print(f"\n{'=' * 50}")
    print(f"🚀 RAG 架构启动 (剩余 {len(df_todo)} 条待处理)")
    print(f"{'=' * 50}")

    agent = RagSearchEvalAgent(CHROMA_DB_PATH)
    results_auto = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, i, r, agent): i for i, r in enumerate(df_todo.to_dict("records"))}
        for future in tqdm(as_completed(futures), total=len(df_todo), desc="机审进度"):
            results_auto.append(future.result())

    high_conf_results = [r for r in results_auto if not r["needs_intervention"]]
    intervention_queue = [r for r in results_auto if r["needs_intervention"]]

    if high_conf_results:
        append_to_output(high_conf_results)

    print(f"\n✅ RAG 机审阶段完成！耗时: {time.time() - start_time:.2f}s")
    print(f"⚠️ 发现 {len(intervention_queue)} 条低置信度数据，即将进入人工教学...")
    time.sleep(1)

    if intervention_queue:
        print(f"\n{'=' * 50}\n👨‍🏫 阶段二：集中人工教学 (按 0 随时保存退出)\n{'=' * 50}")
        for item in intervention_queue:
            print(f"\n🔍 Query: 【{item['query']}】")
            print(f"   [Title]: {item['result_title']}")
            print(f"   [Summary]: {item['result_summary']}")
            print(f"   [机器判定]: {item['llm_tag']} (思路: {item['llm_reason']})")
            print("\n   🎯 [1]完全不相关 [2]丢词搜不准 [3]query理解有误 [4]同领域 [5]衍生 [回车]接受原判 [0]退出")
            choice = input(">> 选择: ").strip()

            if choice == "0":
                print("\n👋 保存进度，安全退出！")
                break

            if choice in TAG_DICT:
                human_tag = TAG_DICT[choice]
                human_rule = input(f">> 已选【{human_tag}】，请输入判别规则: ").strip()
                if human_rule:
                    agent.learn(item["query"], human_tag, human_rule, source=SOURCE_HUMAN)
                    item["llm_tag"] = human_tag
                    item["llm_reason"] = f"【人工纠偏-Human】{human_rule}"
                    item["confidence"] = 100
                    item["reviewed"] = True
                    item["is_correct"] = re.sub(r"\s+", "", human_tag).lower() == re.sub(r"\s+", "", item["expected_tag"]).lower()

            append_to_output([item])
            print("-" * 40)

    print(f"\n{'=' * 50}")
    print(f"✅ 任务完毕！当前向量库容量: {agent.collection.count()} 条经验")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
