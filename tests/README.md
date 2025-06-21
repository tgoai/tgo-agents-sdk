# 测试文档

本文档描述了TGO多智能体框架的测试套件，包括如何运行测试、测试结构和最佳实践。

## 测试结构

```
tests/
├── __init__.py                 # 测试包初始化
├── conftest.py                # Pytest配置和共享fixtures
├── test_config.py             # 测试配置和工具
├── test_core_models.py        # 核心模型单元测试
├── test_adapter_registry.py   # 适配器注册表测试
├── test_base_adapter.py       # 基础适配器测试
├── test_google_adk_adapter.py # Google ADK适配器测试
├── test_workflow_engine.py    # 工作流引擎测试
├── test_multi_agent_coordinator.py # 多智能体协调器测试
├── test_hierarchical_workflow.py   # 层级工作流测试
├── test_integration.py        # 集成测试
├── data/                      # 测试数据目录
├── temp/                      # 临时文件目录
└── README.md                  # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
# 使用Poetry安装开发依赖
poetry install

# 或使用pip
pip install -e ".[dev]"
```

### 2. 运行所有测试

```bash
# 使用测试运行脚本
python run_tests.py all

# 或直接使用pytest
pytest tests/
```

### 3. 运行特定类型的测试

```bash
# 单元测试
python run_tests.py unit

# 集成测试
python run_tests.py integration

# 带覆盖率的测试
python run_tests.py all --coverage
```

## 测试命令

### 使用测试运行脚本

```bash
# 运行单元测试
python run_tests.py unit

# 运行集成测试
python run_tests.py integration

# 运行所有测试
python run_tests.py all

# 运行特定测试文件
python run_tests.py specific -t tests/test_core_models.py

# 运行代码检查
python run_tests.py lint

# 运行类型检查
python run_tests.py type-check

# 修复代码格式
python run_tests.py format

# 运行质量检查
python run_tests.py quality

# 安装依赖
python run_tests.py install
```

### 直接使用pytest

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/ -m "unit or not integration"

# 运行集成测试
pytest tests/test_integration.py

# 运行特定测试文件
pytest tests/test_core_models.py

# 运行特定测试函数
pytest tests/test_core_models.py::TestTask::test_task_creation_with_defaults

# 详细输出
pytest tests/ -v

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 并行运行测试
pytest tests/ -n auto
```

## 测试标记

测试使用以下标记进行分类：

- `unit`: 单元测试
- `integration`: 集成测试
- `slow`: 慢速测试
- `mock`: 使用模拟的测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 排除慢速测试
pytest -m "not slow"

# 运行模拟测试
pytest -m mock
```

## 测试配置

### 环境变量

- `TEST_DEBUG`: 设置为"true"启用调试模式
- `TEST_INTEGRATION`: 设置为"true"运行集成测试
- `TEST_TIMEOUT`: 测试超时时间（秒）
- `TEST_MOCK_RESPONSES`: 设置为"false"禁用模拟响应

```bash
# 启用调试模式
TEST_DEBUG=true pytest tests/

# 运行集成测试
TEST_INTEGRATION=true pytest tests/test_integration.py

# 设置超时时间
TEST_TIMEOUT=60 pytest tests/
```

### pytest配置

项目根目录的`pytest.ini`文件包含了pytest的默认配置：

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
asyncio_mode = auto
```

## 编写测试

### 测试文件命名

- 测试文件以`test_`开头
- 测试类以`Test`开头
- 测试函数以`test_`开头

### 使用Fixtures

项目提供了丰富的fixtures，定义在`conftest.py`中：

```python
def test_task_creation(sample_task):
    """使用sample_task fixture的测试."""
    assert sample_task.title == "Test Task"
    assert sample_task.task_type == TaskType.SIMPLE

def test_agent_config(sample_agent_config):
    """使用sample_agent_config fixture的测试."""
    assert sample_agent_config.name == "Test Agent"
    assert sample_agent_config.agent_type == AgentType.EXPERT
```

### 异步测试

使用`@pytest.mark.asyncio`装饰器标记异步测试：

```python
@pytest.mark.asyncio
async def test_async_function():
    """异步测试示例."""
    result = await some_async_function()
    assert result is not None
```

### 模拟和Mock

使用unittest.mock进行模拟：

```python
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    """使用mock的测试示例."""
    with patch('src.module.function') as mock_func:
        mock_func.return_value = "mocked result"
        result = await function_under_test()
        assert result == "mocked result"
        mock_func.assert_called_once()
```

### 参数化测试

使用pytest.mark.parametrize进行参数化测试：

```python
@pytest.mark.parametrize("input_value,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
    ("test3", "result3"),
])
def test_parametrized(input_value, expected):
    """参数化测试示例."""
    result = function_under_test(input_value)
    assert result == expected
```

## 覆盖率报告

### 生成覆盖率报告

```bash
# HTML报告
pytest tests/ --cov=src --cov-report=html

# 终端报告
pytest tests/ --cov=src --cov-report=term-missing

# XML报告（用于CI）
pytest tests/ --cov=src --cov-report=xml
```

### 查看覆盖率报告

HTML报告会生成在`htmlcov/`目录中，打开`htmlcov/index.html`查看详细报告。

## 持续集成

项目使用GitHub Actions进行持续集成，配置文件位于`.github/workflows/tests.yml`。

CI流程包括：
1. 代码格式检查
2. 类型检查
3. 单元测试
4. 集成测试
5. 覆盖率报告
6. 安全检查

## 最佳实践

### 1. 测试隔离
- 每个测试应该独立运行
- 使用fixtures提供测试数据
- 清理测试产生的副作用

### 2. 测试命名
- 使用描述性的测试名称
- 测试名称应该说明测试的内容和预期结果

### 3. 断言
- 使用具体的断言而不是通用的assertTrue
- 提供有意义的断言消息

### 4. 模拟
- 模拟外部依赖和网络调用
- 使用适当的模拟级别（单元测试vs集成测试）

### 5. 异步测试
- 正确处理异步代码
- 使用适当的超时设置

### 6. 测试数据
- 使用fixtures提供测试数据
- 避免硬编码测试数据

## 故障排除

### 常见问题

1. **导入错误**: 确保项目已正确安装 (`pip install -e .`)
2. **异步测试失败**: 检查是否使用了`@pytest.mark.asyncio`装饰器
3. **覆盖率低**: 检查是否有未测试的代码路径
4. **测试超时**: 增加超时时间或优化测试代码

### 调试测试

```bash
# 启用详细输出
pytest tests/ -v -s

# 在第一个失败处停止
pytest tests/ -x

# 启用调试模式
TEST_DEBUG=true pytest tests/ -v -s

# 运行特定测试并显示输出
pytest tests/test_core_models.py::TestTask::test_task_creation_with_defaults -v -s
```

## 贡献指南

1. 为新功能编写测试
2. 确保测试覆盖率不低于80%
3. 运行所有测试确保没有回归
4. 遵循现有的测试模式和命名约定
5. 更新相关文档
