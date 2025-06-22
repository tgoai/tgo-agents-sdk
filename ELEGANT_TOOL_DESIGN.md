# 🎯 Elegant Tool Configuration Design

## 概述

我们重新设计了TGO多Agent系统的工具配置方式，使其更加优雅、简洁和直观。新设计将函数工具和MCP工具统一到一个`tools`数组中，通过对象类型自动识别工具类型。

## 🔄 设计变更

### 之前的复杂设计
```python
# 复杂且分散的配置
AgentConfig(
    agent_id="agent_001",
    tools=["tool1", "tool2"],           # 字符串工具名
    mcp_servers=["server1", "server2"], # MCP服务器配置
    mcp_tools=["mcp_tool1"],            # 特定MCP工具
    mcp_auto_approve=True               # MCP自动批准
)
```

### 现在的优雅设计
```python
# 优雅统一的配置
AgentConfig(
    agent_id="agent_001",
    tools=[
        my_function_tool,    # 函数工具 - 直接传入函数对象
        my_mcp_tool_object,  # MCP工具 - 传入MCPTool对象
        "legacy_tool"        # 字符串工具 - 向后兼容
    ]
)
```

## ✨ 核心优势

### 1. 🎯 统一接口
- **单一配置点**: 所有工具都在`tools`数组中
- **类型自动识别**: 系统自动区分函数工具和MCP工具
- **直观明了**: 开发者一眼就能看出Agent有哪些工具

### 2. 🔧 类型安全
```python
# 函数工具 - 编译时类型检查
def calculate_sum(a: float, b: float) -> float:
    return a + b

# MCP工具 - 结构化定义
mcp_tool = MCPTool(
    name="web_search",
    description="Search the web",
    input_schema={...},
    server_id="web_api"
)

# 混合使用
tools=[calculate_sum, mcp_tool]  # 类型安全且清晰
```

### 3. 🚀 开发体验
- **即插即用**: 函数工具直接传入，无需额外配置
- **智能检测**: 自动识别同步/异步函数
- **向后兼容**: 支持原有的字符串工具名

### 4. 🔍 工具分析
```python
agent_config = AgentConfig(tools=[func1, func2, mcp_tool1, mcp_tool2])

# 便捷的工具分析方法
function_tools = agent_config.get_function_tools()  # [func1, func2]
mcp_tools = agent_config.get_mcp_tools()           # [mcp_tool1, mcp_tool2]
has_mcp = agent_config.has_mcp_tools()             # True
has_functions = agent_config.has_function_tools()  # True
```

## 🛠️ 实现细节

### 1. AgentConfig扩展
```python
class AgentConfig(BaseModel):
    tools: List[Any] = Field(default_factory=list, description="Available tools (functions or MCPTool objects)")
    
    def get_function_tools(self) -> List[Any]:
        """获取函数工具"""
        return [tool for tool in self.tools if callable(tool) and not hasattr(tool, 'server_id')]
    
    def get_mcp_tools(self) -> List[Any]:
        """获取MCP工具"""
        return [tool for tool in self.tools if hasattr(tool, 'server_id') and hasattr(tool, 'name')]
    
    def get_string_tools(self) -> List[str]:
        """获取字符串工具名（向后兼容）"""
        return [tool for tool in self.tools if isinstance(tool, str)]
```

### 2. GoogleADKAdapter适配
```python
async def _process_agent_tools(self, config: AgentConfig) -> List[Any]:
    """处理统一工具数组"""
    all_tools = []
    
    for tool in config.tools:
        if isinstance(tool, str):
            # 字符串工具名 - 传统处理方式
            string_tools = await self._get_tools_for_agent([tool])
            all_tools.extend(string_tools)
        elif callable(tool):
            # 函数工具 - 包装为ADK工具
            adk_tool = self._create_adk_function_tool_wrapper(tool)
            all_tools.append(adk_tool)
        elif hasattr(tool, 'server_id'):
            # MCP工具对象 - 转换为ADK格式
            adk_tool = await self._create_adk_mcp_tool_from_object(tool)
            all_tools.append(adk_tool)
    
    return all_tools
```

### 3. 工具包装器
```python
def _create_adk_function_tool_wrapper(self, func) -> Any:
    """为函数工具创建ADK包装器"""
    try:
        from google.adk.tools import FunctionTool
        return FunctionTool(func=func)
    except ImportError:
        # 降级到模拟工具
        return MockFunctionTool(func)

async def _create_adk_mcp_tool_from_object(self, mcp_tool) -> Any:
    """从MCP工具对象创建ADK包装器"""
    async def mcp_tool_function(**kwargs):
        # 通过MCP工具管理器调用实际工具
        result = await self._mcp_tool_manager.call_tool(
            tool_name=mcp_tool.name,
            arguments=kwargs,
            context=context,
            user_approved=True
        )
        return result.content if result.success else f"Error: {result.error_message}"
    
    mcp_tool_function.__name__ = mcp_tool.name
    mcp_tool_function.__doc__ = mcp_tool.description
    
    return FunctionTool(func=mcp_tool_function)
```

## 📝 使用示例

### 基础示例
```python
# 定义函数工具
def calculate_sum(a: float, b: float) -> float:
    """计算两个数的和"""
    return a + b

async def fetch_data(url: str) -> str:
    """异步获取数据"""
    # 模拟异步操作
    await asyncio.sleep(0.1)
    return f"Data from {url}"

# 定义MCP工具
web_search = MCPTool(
    name="web_search",
    description="搜索网络信息",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    },
    server_id="web_api"
)

# 优雅配置
config = MultiAgentConfig(
    framework="google-adk",
    agents=[
        AgentConfig(
            agent_id="research_agent",
            name="研究助手",
            tools=[calculate_sum, fetch_data, web_search]  # 混合工具类型
        )
    ]
)
```

### 高级示例
```python
class AdvancedToolExample:
    def create_calculation_tools(self):
        """创建计算工具集"""
        def add(a: float, b: float) -> float:
            return a + b
        
        def multiply(a: float, b: float) -> float:
            return a * b
        
        def calculate_percentage(value: float, total: float) -> float:
            return (value / total) * 100
        
        return [add, multiply, calculate_percentage]
    
    def create_mcp_tools(self):
        """创建MCP工具集"""
        return [
            MCPTool(name="read_file", description="读取文件", ...),
            MCPTool(name="write_file", description="写入文件", ...),
            MCPTool(name="query_db", description="查询数据库", ...)
        ]
    
    def create_agent_config(self):
        """创建Agent配置"""
        calc_tools = self.create_calculation_tools()
        mcp_tools = self.create_mcp_tools()
        
        return AgentConfig(
            agent_id="advanced_agent",
            name="高级助手",
            tools=calc_tools + mcp_tools,  # 优雅组合
            instructions="你可以进行计算、文件操作和数据库查询。"
        )
```

## 🔄 迁移指南

### 从旧配置迁移
```python
# 旧配置
AgentConfig(
    agent_id="agent_001",
    tools=["tool1", "tool2"],
    mcp_servers=["filesystem", "database"],
    mcp_tools=["read_file", "query_db"],
    mcp_auto_approve=True
)

# 新配置
AgentConfig(
    agent_id="agent_001",
    tools=[
        "tool1", "tool2",           # 保持字符串工具（向后兼容）
        read_file_mcp_tool,         # MCP工具对象
        query_db_mcp_tool           # MCP工具对象
    ]
)
```

### 最佳实践
1. **优先使用对象**: 尽量使用函数对象和MCPTool对象而不是字符串
2. **类型注解**: 为函数工具添加完整的类型注解
3. **文档字符串**: 为所有工具提供清晰的文档字符串
4. **工具分组**: 将相关工具组织在一起，便于管理

## 🎉 总结

这个优雅的工具配置设计带来了：

- **🎯 简化配置**: 一个数组搞定所有工具
- **🔧 类型安全**: 编译时检查，运行时稳定
- **🚀 开发效率**: 直观易用，减少配置错误
- **🔄 向后兼容**: 平滑迁移，不破坏现有代码
- **📈 可扩展性**: 易于添加新的工具类型

这种设计让开发者能够更专注于业务逻辑，而不是复杂的配置管理。
