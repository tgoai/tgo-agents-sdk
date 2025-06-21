#!/usr/bin/env python3
"""
Test Basic Usage Example

This script tests the basic usage example from README.md to ensure it works correctly.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgo.agents import (
    MultiAgentCoordinator, AdapterRegistry, GoogleADKAdapter,
    InMemoryMemoryManager, InMemorySessionManager
)
from tgo.agents.core.models import (
    MultiAgentConfig, AgentConfig, Task, WorkflowConfig, Session
)
from tgo.agents.core.enums import (
    AgentType, WorkflowType, ExecutionStrategy, SessionType
)

async def test_basic_usage():
    """Test the basic usage example from README.md"""
    print("🧪 Testing Basic Usage Example from README.md")
    print("=" * 50)
    
    try:
        # 1. Initialize system components
        print("📝 Step 1: Initializing system components...")
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()
        registry = AdapterRegistry()
        registry.register("google-adk", GoogleADKAdapter())
        
        coordinator = MultiAgentCoordinator(
            registry=registry,
            memory_manager=memory_manager,
            session_manager=session_manager
        )
        print("✅ System components initialized")

        # 2. Create session
        print("📝 Step 2: Creating session...")
        await session_manager.create_session("session_001", "user_123", SessionType.SINGLE_CHAT)
        session = Session(session_id="session_001", user_id="user_123", session_type=SessionType.SINGLE_CHAT)
        print("✅ Session created")

        # 3. Configure multi-agent team (Manager + Experts)
        print("📝 Step 3: Configuring multi-agent team...")
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                # Manager Agent - coordinates the team
                AgentConfig(
                    agent_id="project_manager",
                    name="Project Manager",
                    agent_type=AgentType.MANAGER,
                    model="gemini-2.0-flash",
                    instructions="You coordinate tasks between expert agents and synthesize their results."
                ),
                # Research Expert
                AgentConfig(
                    agent_id="researcher",
                    name="Research Specialist", 
                    agent_type=AgentType.EXPERT,
                    model="gemini-2.0-flash",
                    instructions="You are a research expert. Provide thorough market analysis and data insights."
                ),
                # Writing Expert
                AgentConfig(
                    agent_id="writer",
                    name="Content Writer",
                    agent_type=AgentType.EXPERT,
                    model="gemini-2.0-flash", 
                    instructions="You are a content writer. Create clear, engaging reports from research data."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.HIERARCHICAL,  # Manager coordinates experts
                execution_strategy=ExecutionStrategy.FAIL_FAST,
                manager_agent_id="project_manager",
                expert_agent_ids=["researcher", "writer"]
            )
        )
        print("✅ Multi-agent team configured")
        print(f"   👥 Team: {len(config.agents)} agents")
        print(f"   🏢 Workflow: {config.workflow.workflow_type.value}")

        # 4. Create task for the team
        print("📝 Step 4: Creating task...")
        task = Task(
            title="AI Market Analysis Report",
            description="Create a comprehensive report on current AI market trends, including key players, growth projections, and emerging technologies."
        )
        print("✅ Task created")

        # 5. Execute multi-agent workflow
        print("📝 Step 5: Executing multi-agent workflow...")
        print("🚀 Starting multi-agent collaboration...")
        
        result = await coordinator.execute_task(config, task, session)
        
        if result.is_successful():
            print("✅ Multi-agent task completed successfully!")
            print(f"📊 Final Result: {str(result.result)[:200]}...")
            print(f"👥 Agents involved: {', '.join(result.agents_used)}")
            print(f"⏱️  Execution time: {result.total_execution_time_ms}ms")
            return True
        else:
            print(f"❌ Task failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 TGO Multi-Agent Coordinator - Basic Usage Test")
    print("=" * 60)
    
    success = await test_basic_usage()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 BASIC USAGE TEST PASSED!")
        print("✅ The README.md basic usage example works correctly")
    else:
        print("❌ BASIC USAGE TEST FAILED!")
        print("⚠️  The README.md basic usage example needs fixing")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
