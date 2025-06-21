"""
Memory Manager注入机制演示

这个示例展示了优雅的memory_manager传递方式：
1. GoogleADKAdapter构造函数不需要memory_manager参数
2. MultiAgentCoordinator自动将memory_manager注入到adapter中
3. 通过BaseFrameworkAdapter的set_memory_manager方法实现
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


async def demonstrate_memory_injection():
    """演示memory_manager的优雅注入机制"""
    
    print("🚀 Memory Manager注入机制演示")
    print("=" * 40)
    
    # 1. 创建管理器
    print("\n📋 步骤1: 创建Session和Memory管理器")
    session_manager = InMemorySessionManager()
    memory_manager = InMemoryMemoryManager()
    print("✅ 管理器创建完成")
    
    # 2. 创建适配器（不传入memory_manager）
    print("\n📋 步骤2: 创建适配器（无需传入memory_manager）")
    registry = AdapterRegistry()
    google_adapter = GoogleADKAdapter()  # 注意：构造函数不需要memory_manager
    registry.register("google-adk", google_adapter)
    
    # 验证adapter初始时没有memory_manager
    print(f"✅ 适配器创建完成")
    print(f"   初始memory_manager状态: {google_adapter.get_memory_manager()}")
    
    # 3. 创建协调器（传入memory_manager）
    print("\n📋 步骤3: 创建协调器")
    coordinator = MultiAgentCoordinator(
        registry=registry,
        session_manager=session_manager,
        memory_manager=memory_manager  # 只在coordinator中设置
    )
    print("✅ 协调器创建完成")
    
    # 4. 配置任务
    print("\n📋 步骤4: 配置任务")
    config = MultiAgentConfig(
        framework="google-adk",
        agents=[
            AgentConfig(
                agent_id="test_agent",
                name="Test Agent",
                description="A test agent",
                agent_type=AgentType.EXPERT,
                capabilities=["reasoning"],
                model="gemini-2.0-flash",
                timeout_seconds=300
            )
        ],
        workflow=WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST,
            timeout_seconds=600
        ),
        security={},
        monitoring={}
    )
    
    session = Session(
        session_id="demo_session",
        user_id="demo_user",
        session_type=SessionType.SINGLE_CHAT
    )

    task = Task(
        task_id="demo_task",
        title="测试Memory注入",
        description="这是一个测试memory_manager注入机制的任务",
        task_type=TaskType.SIMPLE,
        priority=TaskPriority.MEDIUM,
        input_data={},
        output_data={},
        timeout_seconds=300,
        started_at=None,
        completed_at=None,
        parent_task_id=None
    )
    
    print("✅ 任务配置完成")
    
    # 5. 执行任务（这时会触发memory_manager注入）
    print("\n📋 步骤5: 执行任务（触发memory_manager注入）")
    
    try:
        # 在execute_task过程中，coordinator会调用_get_framework_adapter
        # 该方法会自动将memory_manager注入到adapter中
        result = await coordinator.execute_task(config, task, session=session)
        
        # 验证adapter现在有了memory_manager
        print(f"✅ 任务执行完成")
        print(f"   注入后memory_manager状态: {google_adapter.get_memory_manager() is not None}")
        print(f"   注入的是同一个实例: {google_adapter.get_memory_manager() is memory_manager}")
        
    except Exception as e:
        print(f"❌ 任务执行失败: {e}")
        # 即使失败，也应该已经注入了memory_manager
        print(f"   注入后memory_manager状态: {google_adapter.get_memory_manager() is not None}")
    
    # 6. 验证记忆功能
    print("\n📋 步骤6: 验证记忆功能")
    
    # 手动存储一些记忆
    await memory_manager.store_memory(
        session_id=session.session_id,
        content="这是通过注入的memory_manager存储的记忆",
        memory_type="test",
        session_type=session.session_type,
        importance=0.8
    )
    
    # 检索记忆
    memories = await memory_manager.retrieve_memories(
        session_id=session.session_id,
        session_type=session.session_type
    )
    
    print(f"✅ 记忆功能验证完成")
    print(f"   存储的记忆数量: {len(memories)}")
    for memory in memories:
        print(f"   - {memory.content}")
    
    # 7. 清理
    print("\n📋 步骤7: 清理资源")
    await session_manager.shutdown()
    await memory_manager.shutdown()
    print("✅ 清理完成")


async def demonstrate_multiple_adapters():
    """演示多个adapter的memory_manager注入"""
    
    print("\n🔬 多适配器注入演示")
    print("-" * 30)
    
    # 创建管理器
    memory_manager = InMemoryMemoryManager()
    
    # 创建多个适配器
    registry = AdapterRegistry()
    
    google_adapter1 = GoogleADKAdapter()
    google_adapter2 = GoogleADKAdapter()
    
    registry.register("google-adk-1", google_adapter1)
    registry.register("google-adk-2", google_adapter2)
    
    # 创建协调器
    coordinator = MultiAgentCoordinator(
        registry=registry,
        memory_manager=memory_manager
    )
    
    print("✅ 创建了2个适配器和1个协调器")
    print(f"   适配器1初始memory_manager: {google_adapter1.get_memory_manager()}")
    print(f"   适配器2初始memory_manager: {google_adapter2.get_memory_manager()}")
    
    # 模拟获取不同的适配器
    adapter1 = await coordinator._get_framework_adapter("google-adk-1")
    adapter2 = await coordinator._get_framework_adapter("google-adk-2")
    
    print("✅ 获取适配器后:")
    print(f"   适配器1注入后memory_manager: {adapter1.get_memory_manager() is not None}")
    print(f"   适配器2注入后memory_manager: {adapter2.get_memory_manager() is not None}")
    print(f"   两个适配器使用同一个memory_manager: {adapter1.get_memory_manager() is adapter2.get_memory_manager()}")
    
    await memory_manager.shutdown()


if __name__ == "__main__":
    print("Memory Manager注入机制演示")
    print("展示优雅的memory_manager传递方式")
    
    try:
        # 基本注入演示
        asyncio.run(demonstrate_memory_injection())
        
        # 多适配器演示
        asyncio.run(demonstrate_multiple_adapters())
        
        print("\n🎉 演示完成！")
        print("\n💡 关键要点:")
        print("   1. GoogleADKAdapter构造函数不需要memory_manager参数")
        print("   2. MultiAgentCoordinator在获取adapter时自动注入memory_manager")
        print("   3. 通过BaseFrameworkAdapter的set_memory_manager方法实现")
        print("   4. 所有adapter共享同一个memory_manager实例")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
