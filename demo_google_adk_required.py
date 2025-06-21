#!/usr/bin/env python3
"""
演示脚本：验证 Google ADK 作为必需依赖项的功能

这个脚本演示了修改后的 Google ADK 适配器如何工作，
现在 Google ADK 是必需的依赖项而不是可选的。
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.google_adk_adapter import GoogleADKAdapter
from src.core.models import AgentConfig
from src.core.enums import AgentType


async def main():
    """主演示函数"""
    print("🚀 Google ADK 必需依赖项演示")
    print("=" * 50)
    
    try:
        # 1. 创建适配器实例
        print("1. 创建 GoogleADKAdapter 实例...")
        adapter = GoogleADKAdapter()
        print(f"   ✅ 适配器名称: {adapter.name}")
        print(f"   ✅ 版本信息: {adapter.version_info}")
        print(f"   ✅ 支持的能力: {len(adapter._capabilities)} 个")
        
        # 2. 验证 Google ADK 始终可用
        print("\n2. 验证 Google ADK 可用性...")
        print("   ✅ Google ADK 现在是必需依赖项，始终可用")
        
        # 3. 初始化适配器
        print("\n3. 初始化适配器...")
        await adapter.initialize()
        print(f"   ✅ 适配器已初始化: {adapter.is_initialized}")
        
        # 4. 创建测试配置
        print("\n4. 创建测试 Agent 配置...")
        config = AgentConfig(
            agent_id="demo_agent",
            name="demo_agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash",
            description="演示用的专家智能体",
            temperature=0.7,
            max_iterations=3
        )
        print(f"   ✅ 配置创建完成: {config.name}")
        
        # 5. 创建运行配置
        print("\n5. 创建 ADK 运行配置...")
        run_config = adapter._create_run_config(config)
        print(f"   ✅ 运行配置创建完成: {type(run_config).__name__}")
        
        # 6. 获取工具
        print("\n6. 获取可用工具...")
        tools = await adapter._get_tools_for_agent(["google_search"])
        print(f"   ✅ 获取到 {len(tools)} 个工具")
        
        # 7. 创建不同类型的 Agent
        print("\n7. 创建不同类型的 Agent...")
        
        # 管理者 Agent
        manager_config = AgentConfig(
            agent_id="manager_agent",
            name="manager_agent",
            agent_type=AgentType.MANAGER,
            model="gemini-2.0-flash"
        )
        manager_agent = await adapter._create_manager_agent(manager_config, tools)
        print(f"   ✅ 管理者 Agent: {type(manager_agent).__name__}")
        
        # 专家 Agent
        expert_agent = await adapter._create_expert_agent(config, tools)
        print(f"   ✅ 专家 Agent: {type(expert_agent).__name__}")
        
        # LLM Agent
        llm_config = AgentConfig(
            agent_id="llm_agent",
            name="llm_agent",
            agent_type=AgentType.CUSTOM,
            model="gemini-2.0-flash"
        )
        llm_agent = await adapter._create_llm_agent(llm_config, tools)
        print(f"   ✅ LLM Agent: {type(llm_agent).__name__}")
        
        # 8. 验证默认指令
        print("\n8. 验证默认指令...")
        manager_instructions = adapter._get_default_manager_instructions()
        expert_instructions = adapter._get_default_expert_instructions()
        print(f"   ✅ 管理者指令长度: {len(manager_instructions)} 字符")
        print(f"   ✅ 专家指令长度: {len(expert_instructions)} 字符")
        
        # 9. 清理
        print("\n9. 清理资源...")
        await adapter.cleanup()
        print(f"   ✅ 适配器已清理: {not adapter.is_initialized}")
        
        print("\n🎉 演示完成！Google ADK 作为必需依赖项工作正常。")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("   这表明 Google ADK 未正确安装或配置")
        return 1
        
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return 1
    
    return 0


def check_google_adk_availability():
    """检查 Google ADK 的可用性"""
    print("🔍 检查 Google ADK 模块...")
    
    try:
        from google.adk.agents import LlmAgent, RunConfig
        from google.adk.tools import google_search
        print("   ✅ google.adk.agents.LlmAgent")
        print("   ✅ google.adk.agents.RunConfig")
        print("   ✅ google.adk.tools.google_search")
        return True
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False


if __name__ == "__main__":
    print("Google ADK 必需依赖项演示脚本")
    print("=" * 50)
    
    # 首先检查 Google ADK 可用性
    if not check_google_adk_availability():
        print("\n❌ Google ADK 不可用，请确保已正确安装")
        sys.exit(1)
    
    print("\n✅ Google ADK 模块检查通过")
    print("\n开始演示...")
    
    # 运行主演示
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
