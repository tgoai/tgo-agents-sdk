"""
Session and Memory功能使用示例

这个示例展示了如何使用新增的Session和Memory功能：
1. 创建和管理用户会话
2. 存储和检索对话记忆
3. 在多智能体任务执行中使用会话和记忆
"""

import asyncio
import logging
from typing import Dict, Any

# 导入核心组件
from src.registry.adapter_registry import AdapterRegistry
from src.coordinator.multi_agent_coordinator import MultiAgentCoordinator
from src.adapters.google_adk_adapter import GoogleADKAdapter

# 导入Session和Memory相关组件
from src.core.models import Session, SessionConfig, MemoryConfig
from src.core.enums import SessionType
from src.memory.in_memory_session_manager import InMemorySessionManager
from src.memory.in_memory_memory_manager import InMemoryMemoryManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数：演示Session和Memory功能的完整使用流程"""
    
    print("🚀 Session和Memory功能演示")
    print("=" * 50)
    
    # 1. 创建会话和记忆管理器
    print("\n📋 步骤1: 初始化Session和Memory管理器")
    
    session_config = SessionConfig(
        session_timeout_minutes=60,
        max_sessions_per_user=5,
        enable_persistence=True
    )
    
    memory_config = MemoryConfig(
        max_memories_per_session=100,
        memory_retention_days=7,
        enable_semantic_search=True
    )
    
    session_manager = InMemorySessionManager(session_config)
    memory_manager = InMemoryMemoryManager(memory_config)
    
    print(f"✅ Session管理器已初始化 (超时: {session_config.session_timeout_minutes}分钟)")
    print(f"✅ Memory管理器已初始化 (最大记忆数: {memory_config.max_memories_per_session})")
    
    # 2. 创建用户会话
    print("\n📋 步骤2: 创建用户会话")
    
    # 创建单聊会话
    single_session = await session_manager.create_session(
        session_id="session_001",
        user_id="user_123",
        session_type=SessionType.SINGLE_CHAT
    )

    # 创建群聊会话
    group_session = await session_manager.create_session(
        session_id="group_session_001",
        user_id="user_123",
        session_type=SessionType.GROUP_CHAT
    )
    
    print(f"✅ 单聊会话已创建: {single_session.session_id}")
    print(f"✅ 群聊会话已创建: {group_session.session_id}")
    
    # 3. 存储一些初始记忆
    print("\n📋 步骤3: 存储初始记忆")
    
    # 存储用户偏好
    await memory_manager.store_memory(
        session_id=single_session.session_id,
        content="用户偏好详细的技术分析和代码示例",
        memory_type="preference",
        session_type=single_session.session_type,
        importance=0.9,
        tags=["user_preference", "technical"]
    )
    
    # 存储上下文信息
    await memory_manager.store_memory(
        session_id=single_session.session_id,
        content="用户正在开发一个多智能体协调系统",
        memory_type="context",
        session_type=single_session.session_type,
        importance=0.8,
        tags=["project_context", "development"]
    )
    
    print("✅ 已存储用户偏好和上下文记忆")
    
    # 4. 设置多智能体系统
    print("\n📋 步骤4: 配置多智能体系统")
    
    # 注册适配器
    registry = AdapterRegistry()
    google_adapter = GoogleADKAdapter()  # memory_manager由coordinator自动注入
    registry.register("google-adk", google_adapter)
    
    # 创建协调器
    coordinator = MultiAgentCoordinator(
        registry=registry,
        session_manager=session_manager,
        memory_manager=memory_manager
    )
    
    # 配置智能体（简化版本）
    config = {
        "framework": "google-adk",
        "agents": [
            {
                "agent_id": "manager_001",
                "name": "Task Manager",
                "agent_type": "manager",
                "capabilities": ["reasoning", "delegation"],
                "model": "gemini-2.0-flash"
            },
            {
                "agent_id": "expert_001",
                "name": "AI Expert",
                "agent_type": "expert",
                "capabilities": ["tool_calling", "reasoning"],
                "model": "gemini-2.0-flash"
            }
        ],
        "workflow": {
            "workflow_type": "hierarchical",
            "execution_strategy": "fail_fast"
        }
    }
    
    print("✅ 多智能体系统已配置")
    
    # 5. 执行任务（带会话和记忆支持）
    print("\n📋 步骤5: 执行任务（利用会话记忆）")
    
    task = {
        "task_id": "task_001",
        "title": "分析AI智能体协调架构",
        "description": "请分析多智能体协调系统的架构设计，重点关注Session和Memory管理",
        "task_type": "complex",
        "priority": "high",
        "input_data": {"topic": "multi-agent coordination", "focus": "session_memory"}
    }
    
    # 执行任务，自动利用会话记忆
    print("🔄 正在执行任务...")
    print("⚠️  注意：此示例需要完整的模型定义，这里仅演示Session和Memory的创建")
    # try:
    #     result = await coordinator.execute_task(config, task, session=single_session)
    #     print(f"✅ 任务执行完成: {result.success}")
    #     if result.success:
    #         print(f"📄 结果摘要: {str(result.result)[:200]}...")
    # except Exception as e:
    #     print(f"❌ 任务执行失败: {e}")
    
    # 6. 检索和展示记忆
    print("\n📋 步骤6: 检索会话记忆")
    
    # 检索所有记忆
    all_memories = await memory_manager.retrieve_memories(
        session_id=single_session.session_id,
        session_type=single_session.session_type,
        limit=10
    )
    
    print(f"📚 会话中共有 {len(all_memories)} 条记忆:")
    for i, memory in enumerate(all_memories, 1):
        print(f"  {i}. [{memory.memory_type}] {memory.content[:100]}...")
        print(f"     重要性: {memory.importance:.2f}, 访问次数: {memory.access_count}")
    
    # 7. 搜索特定记忆
    print("\n📋 步骤7: 搜索相关记忆")
    
    search_results = await memory_manager.search_memories(
        session_id=single_session.session_id,
        query="技术分析",
        session_type=single_session.session_type,
        limit=5
    )
    
    print(f"🔍 搜索'技术分析'找到 {len(search_results)} 条相关记忆:")
    for memory in search_results:
        print(f"  - {memory.content[:80]}...")
    
    # 8. 会话统计
    print("\n📋 步骤8: 系统统计")
    
    session_stats = session_manager.get_stats()
    memory_stats = memory_manager.get_stats()
    
    print("📊 会话统计:")
    print(f"  - 总会话数: {session_stats['total_sessions']}")
    print(f"  - 活跃会话数: {session_stats['active_sessions']}")
    print(f"  - 总用户数: {session_stats['total_users']}")
    
    print("📊 记忆统计:")
    print(f"  - 总记忆数: {memory_stats['total_memories']}")
    print(f"  - 涉及会话数: {memory_stats['total_sessions']}")
    print(f"  - 平均每会话记忆数: {memory_stats['avg_memories_per_session']:.1f}")
    
    # 9. 清理资源
    print("\n📋 步骤9: 清理资源")
    
    await session_manager.shutdown()
    await memory_manager.shutdown()
    
    print("✅ 资源清理完成")
    print("\n🎉 Session和Memory功能演示完成！")


async def demonstrate_advanced_features():
    """演示高级功能"""
    print("\n🔬 高级功能演示")
    print("-" * 30)
    
    # 创建管理器
    session_manager = InMemorySessionManager()
    memory_manager = InMemoryMemoryManager()
    
    # 创建会话
    session = await session_manager.create_session(
        session_id="advanced_session",
        user_id="advanced_user",
        session_type=SessionType.SINGLE_CHAT
    )
    
    # 存储不同类型的记忆
    memory_types = [
        ("conversation", "用户询问了关于Python异步编程的问题", 0.6),
        ("fact", "Python asyncio库用于编写异步代码", 0.8),
        ("preference", "用户喜欢看到具体的代码示例", 0.9),
        ("context", "当前讨论的主题是异步编程最佳实践", 0.7)
    ]
    
    for memory_type, content, importance in memory_types:
        await memory_manager.store_memory(
            session_id=session.session_id,
            content=content,
            memory_type=memory_type,
            session_type=session.session_type,
            importance=importance
        )
    
    # 按类型检索记忆
    for memory_type in ["conversation", "fact", "preference", "context"]:
        memories = await memory_manager.retrieve_memories(
            session_id=session.session_id,
            session_type=session.session_type,
            memory_type=memory_type
        )
        print(f"📝 {memory_type}类型记忆: {len(memories)}条")
    
    # 清理
    await session_manager.shutdown()
    await memory_manager.shutdown()


if __name__ == "__main__":
    print("Session和Memory功能完整演示")
    print("请确保已安装Google ADK依赖")
    
    try:
        # 运行主演示
        asyncio.run(main())
        
        # 运行高级功能演示
        asyncio.run(demonstrate_advanced_features())
        
    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
