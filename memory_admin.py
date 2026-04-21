import os
from pathlib import Path

import chromadb

_SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(_SCRIPT_DIR / "agent_vector_db")


def sort_memory_payload(data):
    rows = list(zip(data.get("ids", []), data.get("metadatas", []), data.get("documents", [])))
    rows.sort(key=lambda x: ((x[1] or {}).get("created_at_ms", 0), x[0]), reverse=True)
    return {
        "ids": [r[0] for r in rows],
        "metadatas": [r[1] for r in rows],
        "documents": [r[2] for r in rows],
    }


def search_memories(collection, keyword: str):
    keyword = keyword.strip().lower()
    if not keyword:
        return {"ids": [], "metadatas": [], "documents": []}

    raw = collection.get()
    hits = []
    for memory_id, meta, doc in zip(raw.get("ids", []), raw.get("metadatas", []), raw.get("documents", [])):
        meta = meta or {}
        haystack = "\n".join([
            str(doc or ""),
            str(meta.get("query", "")),
            str(meta.get("human_rule", "")),
            str(meta.get("correct_tag", "")),
            str(meta.get("source", "")),
        ]).lower()
        if keyword in haystack:
            hits.append((memory_id, meta, doc))

    return sort_memory_payload({
        "ids": [h[0] for h in hits],
        "metadatas": [h[1] for h in hits],
        "documents": [h[2] for h in hits],
    })


def _display_memories(data):
    ids = data.get("ids", [])
    metas = data.get("metadatas", [])
    if not ids:
        print("📭 没有找到任何记忆。")
        return

    print("-" * 90)
    for i, memory_id in enumerate(ids):
        meta = metas[i] or {}
        print(f"🆔 [ID]: {memory_id}")
        print(f"🕒 [创建时间ms]: {meta.get('created_at_ms', '未知')}")
        print(f"🤖 [来源]: {meta.get('source', '未知')}")
        print(f"🔍 [Query]: {meta.get('query', '')}")
        print(f"🏷️  [判定为]: {meta.get('correct_tag', '')}")
        print(f"💡 [教导规则]: {meta.get('human_rule', '')}")
        print("-" * 90)


def main():
    print(f"\n{'=' * 50}")
    print("🧠 欢迎进入 Agent 记忆管理后台 (Memory Admin)")
    print(f"{'=' * 50}")

    if not os.path.exists(CHROMA_DB_PATH):
        print("⚠️ 找不到向量数据库文件夹！请先运行 Agent 并进行至少一次人工教学。")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name="eval_rules")
    except Exception:
        print("📭 记忆库目前是空的，没有任何规则。")
        return

    while True:
        print(f"\n📊 当前记忆总数: {collection.count()} 条")
        print("1. 📖 查看所有记忆 (按创建时间倒序)")
        print("2. 🔍 关键词搜索记忆 (支持 Query / 规则 / 标签 / 来源)")
        print("3. 🗑️ 删除单条记忆 (输入 ID)")
        print("4. ⚠️ 清空所有记忆 (危险！)")
        print("0. 👋 安全退出")

        choice = input("\n>> 请选择操作 (0-4): ").strip()
        if choice == "0":
            print("\n👋 已退出管理后台。")
            break
        elif choice == "1":
            _display_memories(sort_memory_payload(collection.get()))
        elif choice == "2":
            keyword = input(">> 请输入要搜索的 Query / 规则关键词 / 标签 / 来源: ").strip()
            _display_memories(search_memories(collection, keyword))
        elif choice == "3":
            del_id = input(">> 🎯 请输入要删除的记忆 ID: ").strip()
            if del_id:
                try:
                    collection.delete(ids=[del_id])
                    print(f"✅ 成功删除记忆: {del_id}")
                except Exception as exc:
                    print(f"❌ 删除失败: {exc}")
        elif choice == "4":
            confirm = input(">> 🚨 警告：此操作将永久清空 Agent 的大脑！确认请输入 'YES': ").strip()
            if confirm == "YES":
                all_ids = collection.get().get("ids", [])
                if all_ids:
                    collection.delete(ids=all_ids)
                    print("✅ 记忆已被彻底清空！")
                else:
                    print("📭 记忆库本来就是空的。")
            else:
                print("🛑 已取消清空操作。")
        else:
            print("⚠️ 无效输入，请重新选择。")


if __name__ == "__main__":
    main()
