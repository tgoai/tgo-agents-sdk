#!/usr/bin/env python3
"""
简单的测试运行示例

这个脚本展示了如何运行TGO多智能体框架的测试套件。
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """运行命令并显示结果."""
    print(f"\n{'='*50}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} 成功完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失败，退出码: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ 命令未找到: {cmd[0]}")
        return False


def main():
    """主函数."""
    print("🚀 TGO多智能体框架测试套件演示")
    print("="*60)
    
    # 确保在正确的目录中
    project_root = Path(__file__).parent
    print(f"项目根目录: {project_root}")
    
    # 运行核心模型测试
    print("\n📋 运行核心模型测试...")
    success = run_command(
        ["python3", "-m", "pytest", "tests/test_core_models.py", "-v"],
        "核心模型测试"
    )
    
    if not success:
        print("\n💥 核心模型测试失败!")
        sys.exit(1)
    
    # 运行适配器注册表测试
    print("\n🔧 运行适配器注册表测试...")
    success = run_command(
        ["python3", "-m", "pytest", "tests/test_adapter_registry.py::TestAdapterRegistry::test_registry_initialization", "-v"],
        "适配器注册表基础测试"
    )
    
    if not success:
        print("\n💥 适配器注册表测试失败!")
        sys.exit(1)
    
    # 显示测试覆盖率
    print("\n📊 运行测试覆盖率分析...")
    success = run_command(
        ["python3", "-m", "pytest", "tests/test_core_models.py", "--cov=src", "--cov-report=term-missing"],
        "测试覆盖率分析"
    )
    
    if not success:
        print("\n⚠️  覆盖率分析失败，但这不影响核心功能")
    
    print("\n🎉 测试套件演示完成!")
    print("\n📚 更多测试命令:")
    print("  • 运行所有测试: python3 -m pytest tests/")
    print("  • 运行特定测试: python3 -m pytest tests/test_core_models.py::TestTask::test_task_creation_with_defaults")
    print("  • 详细输出: python3 -m pytest tests/ -v")
    print("  • 覆盖率报告: python3 -m pytest tests/ --cov=src --cov-report=html")
    print("  • 使用测试脚本: python3 run_tests.py unit")
    
    print("\n📖 查看测试文档: tests/README.md")


if __name__ == "__main__":
    main()
