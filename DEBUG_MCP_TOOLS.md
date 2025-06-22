# MCP工具调试指南

本文档说明如何使用增强的`debug_example.py`来调试MCP（Model Context Protocol）工具功能。

## 🔧 调试功能概览

增强的调试示例包含以下测试：

### 1. 基础组件测试
- **Registry功能**: 测试适配器注册和检索
- **Memory管理**: 测试记忆存储和检索
- **Session管理**: 测试会话创建和管理

### 2. MCP工具测试
- **MCP工具管理器**: 测试初始化和配置
- **MCP服务器注册**: 测试服务器配置和注册
- **MCP工具调用**: 测试实际的工具调用功能
- **安全控制**: 测试权限检查和参数验证

### 3. 集成测试
- **单Agent执行**: 测试配置了MCP工具的Agent执行
- **工具注入**: 测试MCP工具管理器注入到适配器
- **端到端流程**: 测试完整的MCP工具使用流程

## 🚀 运行调试示例

```bash
# 进入项目目录
cd tgo-agent-coordinator

# 运行调试示例
python debug_example.py
```

## 📊 输出解释

### 成功输出示例
```
🔧 TGO Multi-Agent Coordinator - Debug Example with MCP Tools
============================================================
🧪 Running component tests...
✅ Component tests passed.
🧪 Running MCP tools test...
✅ All MCP tests passed.
🧪 Running full integration test with MCP tools...
🎉 SUCCESS! Task completed successfully
📋 Tests completed:
  ✅ Registry functionality
  ✅ Memory management
  ✅ MCP tools functionality
  ✅ MCP tool calling
  ✅ Single agent execution with MCP tools
⏱️  Total execution time: 2.78 seconds
============================================================
```

### 关键日志信息

1. **MCP工具管理器初始化**
   ```
   MCPToolManager initialized
   Initializing MCP Tool Manager
   ```

2. **MCP服务器注册**
   ```
   Registered MCP server: debug_tools (Debug Tools Server)
   ```

3. **MCP工具调用**
   ```
   MCP tool call completed: debug_echo_tool (success: True)
   ```

4. **安全审计**
   ```
   Audit log entry: permission_granted - Permission granted
   Audit log entry: parameters_validated - Parameters validated successfully
   ```

## 🔍 调试特定问题

### 1. MCP连接问题
如果看到连接错误，检查：
- MCP服务器配置是否正确
- 传输类型（stdio/websocket/sse）是否支持
- 命令和参数是否有效

### 2. 工具调用失败
如果工具调用失败，检查：
- 工具是否正确注册
- 参数是否符合schema
- 权限是否允许调用

### 3. 安全策略问题
如果权限被拒绝，检查：
- 安全策略配置
- Agent的MCP服务器访问权限
- 工具的信任级别设置

## 🛠️ 自定义调试

### 添加自定义MCP服务器测试

```python
# 在debug_example.py中添加自定义服务器配置
custom_server_config = MCPServerConfig(
    server_id="my_custom_server",
    name="My Custom Server",
    description="Custom server for testing",
    transport_type="stdio",
    command="your_command_here",
    args=["arg1", "arg2"],
    trusted=True
)

await mcp_manager.register_server(custom_server_config)
```

### 添加自定义工具测试

```python
# 创建自定义工具进行测试
custom_tool = MCPTool(
    name="my_custom_tool",
    description="Custom tool for testing",
    input_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        },
        "required": ["param"]
    },
    server_id="my_custom_server"
)

# 添加到工具管理器
mcp_manager._tools_by_name["my_custom_tool"] = custom_tool
```

### 自定义安全策略测试

```python
from tgo.agents.tools.mcp_security_manager import SecurityPolicy, SecurityLevel

# 创建自定义安全策略
custom_policy = SecurityPolicy(
    allowed_tools={"safe_tool1", "safe_tool2"},
    denied_tools={"dangerous_tool"},
    max_calls_per_minute=5,
    security_level=SecurityLevel.HIGH,
    require_approval_for_untrusted=True
)

# 应用到特定Agent
security_manager.set_policy("test_agent", custom_policy)
```

## 📝 调试日志级别

调试示例使用`DEBUG`级别日志，提供详细信息：

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 重要日志类别

- **INFO**: 主要步骤和成功操作
- **DEBUG**: 详细的内部操作
- **WARNING**: 非致命问题
- **ERROR**: 错误和异常

## 🔧 故障排除

### 常见问题

1. **导入错误**
   - 确保所有依赖已安装
   - 检查Python路径配置

2. **MCP服务器启动失败**
   - 检查命令是否存在
   - 验证参数格式
   - 确认环境变量设置

3. **工具调用超时**
   - 增加超时时间设置
   - 检查网络连接
   - 验证服务器响应

4. **权限被拒绝**
   - 检查安全策略配置
   - 验证Agent权限设置
   - 确认工具信任级别

### 获取更多调试信息

```python
# 启用更详细的日志
logging.getLogger('tgo.agents.tools').setLevel(logging.DEBUG)
logging.getLogger('tgo.agents.adapters').setLevel(logging.DEBUG)
```

## 📚 相关文档

- [README.md](README.md) - 主要项目文档
- [README_CN.md](README_CN.md) - 中文项目文档
- [examples/mcp_config_example.py](examples/mcp_config_example.py) - MCP配置示例
- [example.py](example.py) - 完整使用示例

## 🤝 贡献

如果发现调试功能的问题或有改进建议，请：

1. 创建Issue描述问题
2. 提供详细的错误日志
3. 说明复现步骤
4. 提交Pull Request（如果有修复方案）

---

**注意**: 调试示例使用模拟的MCP服务器，不需要实际的外部MCP服务器。这使得调试更加简单和可靠。
