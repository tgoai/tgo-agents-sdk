# Multi-Agent System Architecture

This document describes the refactored multi-agent system architecture that supports multiple AI frameworks with a unified interface.

## Architecture Overview

The system is built using the **Adapter Pattern** + **Strategy Pattern** + **Factory Pattern** combination to achieve:

- **Framework Agnostic**: Support for Google ADK, LangGraph, CrewAI, and easy extension to new frameworks
- **Dynamic Switching**: Runtime framework switching with automatic fallback
- **Multiple Workflows**: Hierarchical, sequential, parallel, and custom workflow execution
- **Unified Interface**: Consistent API regardless of underlying framework

## Directory Structure

```
src/
├── core/                          # Core abstractions
│   ├── __init__.py
│   ├── interfaces.py              # Core interfaces and protocols
│   ├── models.py                  # Data models and schemas
│   ├── enums.py                   # Enumerations
│   └── exceptions.py              # Exception classes
├── registry/                      # Adapter registry
│   ├── __init__.py
│   └── adapter_registry.py        # Framework adapter registry
├── adapters/                      # Framework adapters
│   ├── __init__.py
│   ├── base_adapter.py            # Base adapter implementation
│   ├── google_adk_adapter.py      # Google ADK integration
│   ├── langgraph_adapter.py       # LangGraph integration
│   └── crewai_adapter.py          # CrewAI integration
├── coordinator/                   # Multi-agent coordination
│   ├── __init__.py
│   ├── multi_agent_coordinator.py # Main coordinator
│   ├── workflow_engine.py         # Workflow execution engine
│   ├── task_executor.py           # Task execution logic
│   └── result_aggregator.py       # Result aggregation
├── workflows/                     # Workflow definitions (future)
├── utils/                         # Utility functions (future)
├── __init__.py                    # Package exports
├── simple_example.py              # Basic usage example
└── README.md                      # This file
```

## Key Components

### 1. AdapterRegistry
Manages registration and discovery of AI framework adapters:

```python
registry = AdapterRegistry()
registry.register("google-adk", GoogleADKAdapter())
registry.register("langgraph", LangGraphAdapter())
registry.register("crewai", CrewAIAdapter())
```

### 2. MultiAgentCoordinator
Orchestrates multi-agent task execution:

```python
coordinator = MultiAgentCoordinator(registry)
result = await coordinator.execute_task(config, task)
```

### 3. Framework Adapters
Provide unified interface to different AI frameworks:

- **GoogleADKAdapter**: Google Agent Development Kit integration
- **LangGraphAdapter**: LangGraph framework integration  
- **CrewAIAdapter**: CrewAI framework integration

### 4. Workflow Engine
Supports different execution patterns:

- **Single**: Single agent execution
- **Hierarchical**: Manager-expert coordination
- **Sequential**: Pipeline-style execution
- **Parallel**: Concurrent execution
- **Custom**: User-defined workflows

## Usage Examples

### Basic Usage (Matching Requirements)

```python
# 注册适配器
registry = AdapterRegistry()
registry.register("google-adk", GoogleADKAdapter())
registry.register("langgraph", LangGraphAdapter())
registry.register("crewai", CrewAIAdapter())

# 创建协调器
coordinator = MultiAgentCoordinator(registry)

# 配置多智能体系统
config = {
    "framework": "google-adk",  # 可以动态切换
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
            "name": "Research Expert",
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

# 执行任务
task = Task(
    task_id="task_001",
    title="Research AI trends",
    description="Research latest AI development trends",
    task_type="complex",
    priority="high",
    input_data={"topic": "AI agents"}
)

result = await coordinator.execute_task(config, task)
```

### Advanced Features

#### Framework Switching with Fallback
```python
config = MultiAgentConfig(
    framework="google-adk",
    fallback_frameworks=["langgraph", "crewai"],
    # ... other config
)
```

#### Streaming Execution
```python
async for update in coordinator.execute_task_stream(config, task):
    print(f"Update: {update}")
```

#### Batch Processing
```python
results = await coordinator.execute_batch_tasks(config, [task1, task2, task3])
```

## Key Design Decisions

### 1. Adapter Pattern
- **Why**: Provides unified interface across different AI frameworks
- **Benefit**: Easy to add new frameworks without changing existing code

### 2. Registry Pattern  
- **Why**: Centralized management of framework adapters
- **Benefit**: Dynamic discovery and switching of frameworks

### 3. Strategy Pattern for Workflows
- **Why**: Different execution strategies for different use cases
- **Benefit**: Flexible workflow execution without tight coupling

### 4. Pydantic Models
- **Why**: Type safety and validation
- **Benefit**: Catch errors early and provide clear interfaces

### 5. Async/Await Throughout
- **Why**: Non-blocking execution for better performance
- **Benefit**: Handle multiple agents and tasks concurrently

## Extension Points

### Adding New Frameworks
1. Create new adapter inheriting from `BaseFrameworkAdapter`
2. Implement required abstract methods
3. Register with the registry

### Adding New Workflow Types
1. Add new workflow type to `WorkflowType` enum
2. Implement handler in `WorkflowEngine`
3. Update coordinator to support new type

### Adding New Capabilities
1. Add capability to `FrameworkCapability` enum
2. Update adapters to declare support
3. Use capability checks in coordination logic

## Migration from Old Architecture

The old architecture used a single `GoogleADKAdapter` with tight coupling. The new architecture:

1. **Abstracts frameworks** behind unified interfaces
2. **Separates concerns** with dedicated components
3. **Enables extensibility** through adapter pattern
4. **Provides type safety** with Pydantic models
5. **Supports multiple workflows** beyond single-agent execution

## Testing

Run the test suite to verify functionality:

```bash
# Run basic tests
python src/tests/test_multi_agent_system.py

# Run simple example
python src/simple_example.py
```

## Migration from Old Architecture

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed migration instructions from the old single-framework architecture.

## Performance and Monitoring

The new architecture provides:

- **Execution Metrics**: Detailed timing and resource usage
- **Health Monitoring**: Adapter and system health status
- **Concurrent Execution**: Parallel task processing
- **Resource Management**: Automatic cleanup and lifecycle management

## Troubleshooting

### Common Issues

1. **Framework Not Available**: Ensure adapters are registered and initialized
2. **Configuration Errors**: Use `ConfigValidator` to check configurations
3. **Execution Failures**: Check logs and adapter status

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Workflow Definitions**: YAML/JSON workflow definitions ✅
2. **Agent Factories**: Dynamic agent creation based on requirements ✅
3. **Monitoring Integration**: Detailed metrics and observability ✅
4. **Caching Layer**: Result caching for improved performance
5. **Security Layer**: Authentication and authorization
6. **Configuration Management**: Environment-based configuration
