"""
基本Session和Memory使用示例

这个示例展示了修改后的基本用法，符合我们之前讨论的需求：
- session_id和user_id为必填
- session_type可选，默认为1（单聊）
- memory_manager.store_memory支持session_type
- MultiAgentCoordinator不处理记忆，记忆由adapters层处理
"""

import asyncio
from src.registry.adapter_registry import AdapterRegistry
from src.coordinator.multi_agent_coordinator import MultiAgentCoordinator
from src.adapters.google_adk_adapter import GoogleADKAdapter
from src.core.models import (
    Task, AgentConfig, MultiAgentConfig, WorkflowConfig, Session
)
from src.core.enums import (
    AgentType, TaskType, TaskPriority, WorkflowType,
    ExecutionStrategy, SessionType
)
from src.memory.in_memory_session_manager import InMemorySessionManager
from src.memory.in_memory_memory_manager import InMemoryMemoryManager


async def basic_usage_example():
    """基本用法示例（符合最终需求）"""
    
    print("🚀 基本Session和Memory用法示例")
    print("=" * 40)
    
    # 注册适配器
    registry = AdapterRegistry()
    
    # 创建会话和记忆管理器（可选）
    session_manager = InMemorySessionManager()
    memory_manager = InMemoryMemoryManager()
    
    # 创建适配器（memory_manager由coordinator自动注入）
    google_adapter = GoogleADKAdapter()
    registry.register("google-adk", google_adapter)
    
    # 创建协调器
    coordinator = MultiAgentCoordinator(
        registry=registry,
        session_manager=session_manager,  # 可选
        memory_manager=memory_manager     # 可选
    )
    
    # 创建会话（session_id和user_id必填，session_type可选）
    session = Session(
        session_id="session_001",  # 必填
        user_id="user_123",        # 必填
        session_type=SessionType.SINGLE_CHAT  # 可选，默认为"1"（单聊）
    )
    
    print(f"✅ 创建会话: {session.session_id}")
    print(f"   用户ID: {session.user_id}")
    print(f"   会话类型: {session.session_type} ({'单聊' if session.session_type == SessionType.SINGLE_CHAT else '群聊'})")
    
    # 配置多智能体系统
    config = MultiAgentConfig(
        framework="google-adk",
        agents=[
            AgentConfig(
                agent_id="manager_001",
                name="Task Manager",
                description="A task management agent",
                agent_type=AgentType.MANAGER,
                capabilities=["reasoning", "delegation"],
                model="gemini-2.0-flash",
                timeout_seconds=300
            ),
            AgentConfig(
                agent_id="expert_001",
                name="Research Expert",
                description="A research expert agent",
                agent_type=AgentType.EXPERT,
                capabilities=["tool_calling", "reasoning"],
                model="gemini-2.0-flash",
                timeout_seconds=300
            )
        ],
        workflow=WorkflowConfig(
            workflow_type=WorkflowType.HIERARCHICAL,
            execution_strategy=ExecutionStrategy.FAIL_FAST,
            manager_agent_id="manager_001",
            timeout_seconds=600
        ),
        security={},
        monitoring={}
    )
    
    # 执行任务
    task = Task(
        task_id="task_001",
        title="Research AI trends",
        description="Research latest AI development trends",
        task_type=TaskType.COMPLEX,
        priority=TaskPriority.HIGH,
        input_data={"topic": "AI agents"},
        output_data={},
        timeout_seconds=300,
        started_at=None,
        completed_at=None,
        parent_task_id=None
    )
    
    print(f"\n🔄 执行任务: {task.title}")
    
    # 执行任务，Session信息会传递给adapters层
    # 记忆处理由GoogleADKAdapter自动处理
    try:
        result = await coordinator.execute_task(config, task, session=session)
        print(f"✅ 任务执行完成: {result.success}")
    except Exception as e:
        print(f"❌ 任务执行失败: {e}")
    
    # 手动存储记忆（演示memory_manager的session_type支持）
    print(f"\n📝 手动存储记忆")
    await memory_manager.store_memory(
        session.session_id,
        content="用户偏好详细的技术分析",
        memory_type="preference",
        session_type=session.session_type,  # 支持session_type
        importance=0.9
    )
    
    # 检索会话记忆
    memories = await memory_manager.retrieve_memories(
        session.session_id,
        session_type=session.session_type
    )
    print(f"📚 检索到 {len(memories)} 条记忆")
    
    # 清理
    await session_manager.shutdown()
    await memory_manager.shutdown()
    
    print("✅ 示例完成")


async def group_chat_example():
    """群聊示例"""
    
    print("\n🚀 群聊会话示例")
    print("=" * 40)
    
    session_manager = InMemorySessionManager()
    memory_manager = InMemoryMemoryManager()
    
    # 创建群聊会话
    group_session = Session(
        session_id="group_session_001",
        user_id="user_123",  # 创建者
        session_type=SessionType.GROUP_CHAT  # 群聊
    )
    
    print(f"✅ 创建群聊会话: {group_session.session_id}")
    print(f"   会话类型: {group_session.session_type} (群聊)")
    
    # 存储群聊记忆
    await memory_manager.store_memory(
        group_session.session_id,
        content="群组讨论了项目进度和下一步计划",
        memory_type="conversation",
        session_type=group_session.session_type,  # 群聊类型
        importance=0.7
    )
    
    # 检索群聊记忆
    group_memories = await memory_manager.retrieve_memories(
        group_session.session_id,
        session_type=group_session.session_type
    )
    
    print(f"📚 群聊记忆: {len(group_memories)} 条")
    for memory in group_memories:
        print(f"   - {memory.content}")
    
    await session_manager.shutdown()
    await memory_manager.shutdown()


async def memory_types_example():
    """不同记忆类型示例"""
    
    print("\n🚀 记忆类型示例")
    print("=" * 40)
    
    memory_manager = InMemoryMemoryManager()
    
    session_id = "demo_session"
    session_type = SessionType.SINGLE_CHAT
    
    # 存储不同类型的记忆
    memory_examples = [
        ("conversation", "用户询问了关于AI的问题", 0.6),
        ("fact", "AI是人工智能的缩写", 0.8),
        ("preference", "用户喜欢简洁明了的回答", 0.9),
        ("context", "当前讨论主题是人工智能技术", 0.7)
    ]
    
    for memory_type, content, importance in memory_examples:
        await memory_manager.store_memory(
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            session_type=session_type,
            importance=importance
        )
        print(f"✅ 存储{memory_type}记忆: {content}")
    
    # 按类型检索
    for memory_type in ["conversation", "fact", "preference", "context"]:
        memories = await memory_manager.retrieve_memories(
            session_id=session_id,
            session_type=session_type,
            memory_type=memory_type
        )
        print(f"📝 {memory_type}: {len(memories)}条")
    
    await memory_manager.shutdown()


if __name__ == "__main__":
    print("Session和Memory基本用法演示")
    
    try:
        # 基本用法
        asyncio.run(basic_usage_example())
        
        # 群聊示例
        asyncio.run(group_chat_example())
        
        # 记忆类型示例
        asyncio.run(memory_types_example())
        
        print("\n🎉 所有示例完成！")
        
    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()
