import os
import chromadb
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(_SCRIPT_DIR / "agent_vector_db")

def main():
    print(f"\n{'='*50}")
    print("🧠 欢迎进入 Agent 记忆管理后台 (Memory Admin)")
    print(f"{'='*50}")

    if not os.path.exists(CHROMA_DB_PATH):
        print("⚠️ 找不到向量数据库文件夹！请先运行 Agent 并进行至少一次人工教学。")
        return

    # 连接到本地的 Chroma 数据库
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name="eval_rules")
    except Exception:
        print("📭 记忆库目前是空的，没有任何规则。")
        return

    while True:
        count = collection.count()
        print(f"\n📊 当前记忆总数: {count} 条")
        print("1. 📖 查看所有记忆 (按时间排序)")
        print("2. 🔍 关键词搜索记忆 (查重/查错)")
        print("3. 🗑️ 删除单条记忆 (输入 ID)")
        print("4. ⚠️ 清空所有记忆 (危险！)")
        print("0. 👋 安全退出")
        
        choice = input("\n>> 请选择操作 (0-4): ").strip()
        
        if choice == '0':
            print("\n👋 已退出管理后台。")
            break
            
        elif choice == '1':
            data = collection.get()
            _display_memories(data)
            
        elif choice == '2':
            keyword = input(">> 请输入要搜索的 Query 或 规则关键词: ").strip()
            if keyword:
                # 使用 ChromaDB 的 where_document 进行模糊过滤
                data = collection.get(where_document={"$contains": keyword})
                print(f"\n🔍 包含 '{keyword}' 的搜索结果：")
                _display_memories(data)
                
        elif choice == '3':
            del_id = input(">> 🎯 请输入要删除的记忆 ID (例如 rule_1710000000000): ").strip()
            if del_id:
                try:
                    collection.delete(ids=[del_id])
                    print(f"✅ 成功删除记忆: {del_id}")
                except Exception as e:
                    print(f"❌ 删除失败，可能 ID 不存在。错误: {e}")
                    
        elif choice == '4':
            confirm = input(">> 🚨 警告：此操作将永久清空 Agent 的大脑！确认请输入 'YES': ").strip()
            if confirm == 'YES':
                # 获取所有 ID 并删除
                all_ids = collection.get()['ids']
                if all_ids:
                    collection.delete(ids=all_ids)
                    print("✅ 记忆已被彻底清空！Agent 重置为出厂状态。")
                else:
                    print("📭 记忆库本来就是空的。")
            else:
                print("🛑 已取消清空操作。")
        else:
            print("⚠️ 无效输入，请重新选择。")

def _display_memories(data):
    ids = data.get('ids', [])
    metadatas = data.get('metadatas', [])
    
    if not ids:
        print("📭 没有找到任何记忆。")
        return
        
    print("-" * 50)
    for i in range(len(ids)):
        m = metadatas[i]
        print(f"🆔 [ID]: {ids[i]}")
        print(f"🔍 [Query]: {m.get('query')}")
        print(f"🏷️  [判定为]: {m.get('correct_tag')}")
        print(f"💡 [教导规则]: {m.get('human_rule')}")
        print("-" * 50)

if __name__ == "__main__":
    main()