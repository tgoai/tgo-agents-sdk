"""
简化的Session和Memory测试

这个测试专注于验证Session和Memory的基本功能，
不涉及复杂的多智能体任务执行。
"""

import asyncio
from src.core.models import Session
from src.core.enums import SessionType
from src.memory.in_memory_session_manager import InMemorySessionManager
from src.memory.in_memory_memory_manager import InMemoryMemoryManager


async def test_session_basic():
    """测试Session基本功能"""
    print("🧪 测试Session基本功能")
    print("-" * 30)
    
    # 创建Session管理器
    session_manager = InMemorySessionManager()
    
    # 测试创建单聊会话
    session = await session_manager.create_session(
        session_id="test_session_001",
        user_id="test_user_123",
        session_type=SessionType.SINGLE_CHAT
    )
    
    print(f"✅ 创建单聊会话成功")
    print(f"   Session ID: {session.session_id}")
    print(f"   User ID: {session.user_id}")
    print(f"   Session Type: {session.session_type}")
    print(f"   Is Group Chat: {session.is_group_chat()}")
    print(f"   Is Active: {session.is_active()}")
    
    # 测试创建群聊会话
    group_session = await session_manager.create_session(
        session_id="test_group_session_001",
        user_id="test_user_123",
        session_type=SessionType.GROUP_CHAT
    )
    
    print(f"\n✅ 创建群聊会话成功")
    print(f"   Session ID: {group_session.session_id}")
    print(f"   Session Type: {group_session.session_type}")
    print(f"   Is Group Chat: {group_session.is_group_chat()}")
    
    # 测试获取会话
    retrieved_session = await session_manager.get_session("test_session_001")
    print(f"\n✅ 检索会话成功: {retrieved_session.session_id if retrieved_session else 'None'}")
    
    # 测试列出用户会话
    user_sessions = await session_manager.list_user_sessions("test_user_123")
    print(f"✅ 用户会话列表: {len(user_sessions)} 个会话")
    
    # 测试统计
    stats = session_manager.get_stats()
    print(f"✅ 会话统计: {stats}")
    
    await session_manager.shutdown()
    print("✅ Session测试完成\n")


async def test_memory_basic():
    """测试Memory基本功能"""
    print("🧪 测试Memory基本功能")
    print("-" * 30)
    
    # 创建Memory管理器
    memory_manager = InMemoryMemoryManager()
    
    session_id = "test_memory_session"
    session_type = SessionType.SINGLE_CHAT
    
    # 测试存储不同类型的记忆
    memory_types = [
        ("conversation", "用户询问了关于Python的问题", 0.6),
        ("fact", "Python是一种编程语言", 0.8),
        ("preference", "用户喜欢详细的代码示例", 0.9),
        ("context", "当前讨论的主题是Python编程", 0.7)
    ]
    
    stored_memories = []
    for memory_type, content, importance in memory_types:
        memory = await memory_manager.store_memory(
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            session_type=session_type,
            importance=importance
        )
        stored_memories.append(memory)
        print(f"✅ 存储{memory_type}记忆: {content[:30]}...")
    
    # 测试检索所有记忆
    all_memories = await memory_manager.retrieve_memories(
        session_id=session_id,
        session_type=session_type
    )
    print(f"\n✅ 检索所有记忆: {len(all_memories)} 条")
    
    # 测试按类型检索记忆
    for memory_type in ["conversation", "fact", "preference", "context"]:
        type_memories = await memory_manager.retrieve_memories(
            session_id=session_id,
            session_type=session_type,
            memory_type=memory_type
        )
        print(f"✅ {memory_type}类型记忆: {len(type_memories)} 条")
    
    # 测试搜索记忆
    search_results = await memory_manager.search_memories(
        session_id=session_id,
        query="Python",
        session_type=session_type
    )
    print(f"\n✅ 搜索'Python'结果: {len(search_results)} 条")
    for memory in search_results:
        print(f"   - {memory.content[:50]}...")
    
    # 测试记忆访问更新
    if stored_memories:
        memory = stored_memories[0]
        original_access_count = memory.access_count
        memory.update_access()
        print(f"✅ 记忆访问更新: {original_access_count} -> {memory.access_count}")
    
    # 测试统计
    stats = memory_manager.get_stats()
    print(f"✅ 记忆统计: {stats}")
    
    await memory_manager.shutdown()
    print("✅ Memory测试完成\n")


async def test_session_memory_integration():
    """测试Session和Memory集成"""
    print("🧪 测试Session和Memory集成")
    print("-" * 30)
    
    # 创建管理器
    session_manager = InMemorySessionManager()
    memory_manager = InMemoryMemoryManager()
    
    # 创建会话
    session = await session_manager.create_session(
        session_id="integration_test_session",
        user_id="integration_user",
        session_type=SessionType.SINGLE_CHAT
    )
    
    print(f"✅ 创建集成测试会话: {session.session_id}")
    
    # 为会话存储记忆
    memories_data = [
        ("用户开始了一个新的对话", "conversation", 0.5),
        ("用户的名字是张三", "fact", 0.8),
        ("用户喜欢简洁的回答", "preference", 0.9),
        ("当前讨论的是AI技术", "context", 0.7)
    ]
    
    for content, memory_type, importance in memories_data:
        await memory_manager.store_memory(
            session_id=session.session_id,
            content=content,
            memory_type=memory_type,
            session_type=session.session_type,
            importance=importance
        )
        print(f"✅ 为会话存储记忆: {content[:30]}...")
    
    # 检索会话的所有记忆
    session_memories = await memory_manager.retrieve_memories(
        session_id=session.session_id,
        session_type=session.session_type
    )
    
    print(f"\n✅ 会话记忆总数: {len(session_memories)}")
    print("📚 会话记忆详情:")
    for i, memory in enumerate(session_memories, 1):
        print(f"   {i}. [{memory.memory_type}] {memory.content}")
        print(f"      重要性: {memory.importance}, 访问次数: {memory.access_count}")
    
    # 搜索特定内容
    search_query = "用户"
    search_results = await memory_manager.search_memories(
        session_id=session.session_id,
        query=search_query,
        session_type=session.session_type
    )
    
    print(f"\n🔍 搜索'{search_query}'的结果: {len(search_results)} 条")
    for memory in search_results:
        print(f"   - {memory.content}")
    
    # 更新会话活动
    session.update_activity()
    print(f"✅ 更新会话活动时间: {session.last_activity}")
    
    # 清理
    await session_manager.shutdown()
    await memory_manager.shutdown()
    print("✅ 集成测试完成\n")


async def test_memory_manager_injection():
    """测试Memory Manager注入机制"""
    print("🧪 测试Memory Manager注入机制")
    print("-" * 30)
    
    from src.adapters.google_adk_adapter import GoogleADKAdapter
    from src.memory.in_memory_memory_manager import InMemoryMemoryManager
    
    # 创建适配器和记忆管理器
    adapter = GoogleADKAdapter()
    memory_manager = InMemoryMemoryManager()
    
    print(f"✅ 创建适配器和记忆管理器")
    print(f"   适配器初始memory_manager: {adapter.get_memory_manager()}")
    
    # 测试注入
    adapter.set_memory_manager(memory_manager)
    print(f"✅ 注入memory_manager")
    print(f"   适配器注入后memory_manager: {adapter.get_memory_manager() is not None}")
    print(f"   是同一个实例: {adapter.get_memory_manager() is memory_manager}")
    
    # 测试清除
    adapter.set_memory_manager(None)
    print(f"✅ 清除memory_manager")
    print(f"   适配器清除后memory_manager: {adapter.get_memory_manager()}")
    
    await memory_manager.shutdown()
    print("✅ 注入机制测试完成\n")


async def main():
    """主测试函数"""
    print("🚀 Session和Memory功能测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        await test_session_basic()
        await test_memory_basic()
        
        # 集成测试
        await test_session_memory_integration()
        
        # 注入机制测试
        await test_memory_manager_injection()
        
        print("🎉 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
