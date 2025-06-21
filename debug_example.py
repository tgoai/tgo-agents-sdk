#!/usr/bin/env python3
"""
Simple Debug Example for TGO Multi-Agent Coordinator

This is a minimal example designed for debugging and testing the basic functionality
of the multi-agent system. It focuses on a single agent execution with minimal
configuration to help identify and fix issues quickly.

Usage:
    python debug_example.py
"""

import asyncio
import logging
from datetime import datetime, timezone

# Core imports - using tgo.agents structure
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgo.agents import MultiAgentCoordinator, AdapterRegistry, GoogleADKAdapter
from tgo.agents.core.models import (
    MultiAgentConfig, AgentConfig, Task, WorkflowConfig, Session
)
from tgo.agents.core.enums import (
    AgentType, WorkflowType, ExecutionStrategy, TaskType,
    TaskPriority, SessionType
)
from tgo.agents.memory.in_memory_memory_manager import InMemoryMemoryManager
from tgo.agents.memory.in_memory_session_manager import InMemorySessionManager

# Configure simple logging
logging.basicConfig(
    level=logging.DEBUG,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def debug_single_agent():
    """Simple single agent execution for debugging."""
    logger.info("🔧 Starting Debug Example - Single Agent")
    
    try:
        # Step 1: Initialize memory and session managers
        logger.info("📝 Step 1: Initializing memory and session managers...")
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()
        
        # Step 2: Create adapter registry
        logger.info("📝 Step 2: Creating adapter registry...")
        registry = AdapterRegistry()
        
        # Step 3: Register Google ADK adapter
        logger.info("📝 Step 3: Registering Google ADK adapter...")
        google_adapter = GoogleADKAdapter()
        registry.register("google-adk", google_adapter, is_default=True)
        
        # Step 4: Create coordinator
        logger.info("📝 Step 4: Creating multi-agent coordinator...")
        coordinator = MultiAgentCoordinator(
            registry=registry,
            memory_manager=memory_manager,
            session_manager=session_manager
        )
        
        # Step 5: Create session
        logger.info("📝 Step 5: Creating session...")
        session_id = "debug_session_001"
        user_id = "debug_user"
        session_type = SessionType.SINGLE_CHAT

        await session_manager.create_session(session_id, user_id, session_type)

        # Create session object for coordinator
        session = Session(
            session_id=session_id,
            user_id=user_id,
            session_type=session_type
        )
        logger.info(f"✅ Session created: {session.session_id}")
        
        # Step 6: Configure single agent
        logger.info("📝 Step 6: Configuring agent...")
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="debug_agent_001",
                    name="Debug Agent",
                    agent_type=AgentType.EXPERT,
                    description="Simple agent for debugging purposes",
                    model="gemini-2.0-flash",
                    instructions="You are a helpful assistant. Provide clear and concise responses."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.SINGLE,
                execution_strategy=ExecutionStrategy.FAIL_FAST,
                timeout_seconds=60
            )
        )
        logger.info(f"✅ Agent configured: {config.agents[0].name}")
        
        # Step 7: Create simple task
        logger.info("📝 Step 7: Creating task...")
        task = Task(
            title="Simple Test Task",
            description="Please respond with 'Hello, this is a debug test!' to confirm the system is working.",
            task_type=TaskType.SIMPLE,
            priority=TaskPriority.MEDIUM
        )
        logger.info(f"✅ Task created: {task.title}")
        
        # Step 8: Execute task
        logger.info("📝 Step 8: Executing task...")
        logger.info("⏳ This may take a moment...")
        
        start_time = datetime.now(timezone.utc)
        result = await coordinator.execute_task(config, task, session)
        end_time = datetime.now(timezone.utc)
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Step 9: Check results
        logger.info("📝 Step 9: Checking results...")
        
        if result.is_successful():
            logger.info("🎉 SUCCESS! Task completed successfully")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"📊 Result: {result.result}")
            
            # Show execution metrics if available
            if hasattr(result, 'total_execution_time_ms') and result.total_execution_time_ms:
                logger.info(f"📈 Metrics: {result.total_execution_time_ms}ms total")
            
            return True
        else:
            logger.error("❌ FAILED! Task execution failed")
            logger.error(f"💥 Error: {result.error_message}")
            # Additional error details would be in the result object itself
            return False
            
    except Exception as e:
        logger.error(f"💥 EXCEPTION during debug execution: {e}")
        logger.exception("Full exception details:")
        return False


async def debug_memory_test():
    """Simple memory functionality test."""
    logger.info("🧠 Testing Memory Functionality...")
    
    try:
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()
        
        # Create session
        session_id = "memory_debug_session"
        user_id = "debug_user"
        session_type = SessionType.SINGLE_CHAT

        await session_manager.create_session(session_id, user_id, session_type)
        
        # Store a memory
        await memory_manager.store_memory(
            session_id="memory_debug_session",
            content="This is a debug memory entry",
            memory_type="test",
            session_type=SessionType.SINGLE_CHAT,
            importance=0.8
        )
        logger.info("✅ Memory stored successfully")
        
        # Retrieve memories
        memories = await memory_manager.retrieve_memories(
            session_id="memory_debug_session",
            session_type=SessionType.SINGLE_CHAT,
            limit=5
        )
        
        logger.info(f"✅ Retrieved {len(memories)} memories")
        for memory in memories:
            logger.info(f"  📝 {memory.memory_type}: {memory.content}")
            
        return True
        
    except Exception as e:
        logger.error(f"💥 Memory test failed: {e}")
        logger.exception("Memory test exception details:")
        return False


async def debug_registry_test():
    """Simple registry functionality test."""
    logger.info("📋 Testing Registry Functionality...")
    
    try:
        registry = AdapterRegistry()
        
        # Register adapter
        google_adapter = GoogleADKAdapter()
        registry.register("google-adk", google_adapter, is_default=True)
        logger.info("✅ Adapter registered successfully")
        
        # List adapters
        adapter_names = registry.list_adapters()
        logger.info(f"✅ Available adapters: {adapter_names}")
        
        # Get adapter
        adapter = registry.get_adapter("google-adk")
        if adapter:
            logger.info(f"✅ Retrieved adapter: {type(adapter).__name__}")
        else:
            logger.error("❌ Failed to retrieve adapter")
            return False
            
        # Check health
        health = registry.get_health_status()
        logger.info(f"✅ Registry health: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Registry test failed: {e}")
        logger.exception("Registry test exception details:")
        return False


async def main():
    """Main debug function - runs all tests."""
    print("🔧 TGO Multi-Agent Coordinator - Debug Example")
    print("=" * 50)
    
    start_time = datetime.now(timezone.utc)
    
    # Run basic component tests first
    logger.info("🧪 Running component tests...")
    
    registry_ok = await debug_registry_test()
    memory_ok = await debug_memory_test()
    
    if not (registry_ok and memory_ok):
        logger.error("❌ Component tests failed. Stopping.")
        return
    
    logger.info("✅ Component tests passed. Running full integration test...")
    
    # Run full integration test
    success = await debug_single_agent()
    
    end_time = datetime.now(timezone.utc)
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 DEBUG EXAMPLE COMPLETED SUCCESSFULLY!")
        logger.info("✅ All tests passed")
    else:
        print("❌ DEBUG EXAMPLE FAILED!")
        logger.error("❌ Some tests failed")
    
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Debug example interrupted by user")
    except Exception as e:
        print(f"\n💥 Debug example failed with error: {e}")
        logging.exception("Detailed error information:")
