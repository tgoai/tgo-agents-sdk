# Session和Memory功能实现文档

## 概述

本次开发为多智能体协调系统添加了完整的Session（会话）和Memory（记忆）管理功能，满足以下需求：

1. **Session模型**：session_id和user_id为必填，session_type可选（默认为1-单聊）
2. **Memory管理**：memory_manager.store_memory支持session_type参数
3. **架构设计**：MultiAgentCoordinator不处理记忆逻辑，记忆由adapters层处理

## 新增组件

### 1. 核心模型 (`src/core/models.py`)

#### Session相关
- **SessionType**: 枚举类型，支持单聊(1)和群聊(2)
- **Session**: 会话模型，包含session_id(必填)、user_id(必填)、session_type(可选)
- **SessionConfig**: 会话管理配置

#### Memory相关
- **MemoryEntry**: 记忆条目模型，支持不同类型的记忆存储
- **MemoryConfig**: 记忆管理配置

### 2. 管理器接口 (`src/core/interfaces.py`)

- **SessionManager**: 会话管理器抽象接口
- **MemoryManager**: 记忆管理器抽象接口

### 3. 内存实现 (`src/memory/`)

- **InMemorySessionManager**: 内存版本的会话管理器
- **InMemoryMemoryManager**: 内存版本的记忆管理器

### 4. 适配器增强

- **GoogleADKAdapter**: 增加了记忆处理功能，在执行任务时自动检索和存储记忆

## 使用方式

### 基本用法

```python
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

# 创建会话
session = Session(
    session_id="session_001",  # 必填
    user_id="user_123",        # 必填
    session_type=SessionType.SINGLE_CHAT  # 可选，默认为"1"（单聊）
)

# 配置多智能体系统
config = {
    "framework": "google-adk",
    "agents": [
        {
            "agent_id": "manager_001",
            "name": "Task Manager",
            "agent_type": "manager",
            "capabilities": ["reasoning", "delegation"],
            "model": "gemini-2.0-flash"
        }
    ],
    "workflow": {
        "workflow_type": "hierarchical",
        "execution_strategy": "fail_fast"
    }
}

# 执行任务（Session信息会传递给adapters层）
result = await coordinator.execute_task(config, task, session=session)

# 手动存储记忆（支持session_type）
await memory_manager.store_memory(
    session.session_id,
    content="用户偏好详细的技术分析",
    memory_type="preference",
    session_type=session.session_type,
    importance=0.9
)
```

### 记忆类型

支持以下记忆类型：
- **conversation**: 对话记录
- **fact**: 事实信息
- **preference**: 用户偏好
- **context**: 上下文信息

### 会话类型

- **SessionType.SINGLE_CHAT** ("1"): 单聊
- **SessionType.GROUP_CHAT** ("2"): 群聊

## 架构特点

### 1. 分层设计
- **协调器层**: MultiAgentCoordinator只传递Session信息，不处理记忆逻辑
- **适配器层**: 各个adapter负责具体的记忆存储和检索
- **管理器层**: SessionManager和MemoryManager提供统一的管理接口

### 2. 记忆处理流程
1. **初始化阶段**: MultiAgentCoordinator在获取adapter时自动注入memory_manager
2. **任务执行前**: adapter从MemoryManager检索相关记忆，增强任务输入
3. **任务执行后**: adapter将执行结果存储到MemoryManager

### 3. Memory Manager注入机制
- **BaseFrameworkAdapter**: 提供`set_memory_manager()`和`get_memory_manager()`方法
- **MultiAgentCoordinator**: 在`_get_framework_adapter()`中自动注入memory_manager
- **优雅传递**: 无需在adapter构造函数中传递，由coordinator统一管理

### 4. 可扩展性
- 支持不同的存储后端（当前为内存，可扩展到Redis、数据库等）
- 支持语义搜索（当前为简单文本匹配，可扩展到向量搜索）
- 支持自定义记忆类型和重要性评分

## 示例文件

1. **session_memory_example.py**: 完整功能演示
2. **basic_session_memory_example.py**: 基本用法示例

## 配置选项

### SessionConfig
- `session_timeout_minutes`: 会话超时时间（分钟）
- `max_sessions_per_user`: 每用户最大会话数
- `enable_persistence`: 是否启用持久化
- `cleanup_interval_minutes`: 清理间隔（分钟）

### MemoryConfig
- `max_memories_per_session`: 每会话最大记忆数
- `memory_retention_days`: 记忆保留天数
- `enable_semantic_search`: 是否启用语义搜索
- `similarity_threshold`: 相似度阈值
- `importance_decay_rate`: 重要性衰减率

## 测试建议

建议编写以下测试：

1. **单元测试**
   - Session和Memory模型验证
   - SessionManager和MemoryManager接口测试
   - 内存实现的功能测试

2. **集成测试**
   - MultiAgentCoordinator与Session的集成
   - GoogleADKAdapter的记忆处理测试
   - 端到端的任务执行测试

3. **性能测试**
   - 大量会话和记忆的性能测试
   - 记忆检索和搜索的性能测试

## 后续扩展

1. **持久化存储**: 实现Redis和数据库版本的管理器
2. **语义搜索**: 集成向量数据库进行语义相似度搜索
3. **记忆压缩**: 实现记忆的自动摘要和压缩
4. **权限控制**: 增强用户权限和访问控制
5. **记忆分析**: 提供记忆使用情况的分析和可视化

## 总结

本次实现完全满足了原始需求：
- ✅ Session模型中session_id和user_id为必填
- ✅ session_type为可选，默认为1（单聊）
- ✅ memory_manager.store_memory支持session_type
- ✅ MultiAgentCoordinator不处理记忆，记忆由adapters层处理
- ✅ 提供了完整的使用示例和文档

系统现在具备了完整的会话和记忆管理能力，为构建更智能的多智能体对话系统奠定了基础。
