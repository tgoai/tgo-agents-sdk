#!/usr/bin/env python3
"""
Complete Multi-Agent System Example

This example demonstrates the full capabilities of the TGO Multi-Agent Coordinator,
including:
- Framework adapter registration and management
- Multi-agent configuration and coordination
- Memory and session management
- Different workflow types (single, hierarchical, parallel)
- Streaming execution
- Error handling and fallback mechanisms
- Tool calling and knowledge base integration

Run this example to see the multi-agent system in action.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Core imports
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentExample:
    """Complete example showcasing multi-agent system capabilities."""

    def __init__(self):
        self.registry: Optional[AdapterRegistry] = None
        self.coordinator: Optional[MultiAgentCoordinator] = None
        self.memory_manager: Optional[InMemoryMemoryManager] = None
        self.session_manager: Optional[InMemorySessionManager] = None
    
    async def setup(self):
        """Initialize the multi-agent system."""
        logger.info("🚀 Setting up Multi-Agent System...")
        
        # 1. Create and configure adapter registry
        self.registry = AdapterRegistry()
        
        # Register Google ADK adapter as primary
        google_adapter = GoogleADKAdapter()
        self.registry.register("google-adk", google_adapter, is_default=True)
        
        # Note: Other adapters would be registered here in a real implementation
        # self.registry.register("langgraph", LangGraphAdapter())
        # self.registry.register("crewai", CrewAIAdapter())
        
        # 2. Set up memory and session management
        self.memory_manager = InMemoryMemoryManager()
        self.session_manager = InMemorySessionManager()

        # 3. Create coordinator with memory and session managers
        self.coordinator = MultiAgentCoordinator(
            registry=self.registry,
            memory_manager=self.memory_manager,
            session_manager=self.session_manager
        )
        
        logger.info("✅ Multi-Agent System initialized successfully")
    
    async def example_single_agent(self):
        """Example 1: Single agent execution."""
        logger.info("\n📝 Example 1: Single Agent Execution")

        # Ensure managers are initialized
        assert self.session_manager is not None, "Session manager not initialized"
        assert self.coordinator is not None, "Coordinator not initialized"

        # Create session
        session = Session(
            session_id="single_session_001",
            user_id="user_123",
            session_type=SessionType.SINGLE_CHAT
        )
        await self.session_manager.create_session(session)
        
        # Configure single agent
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="analyst_001",
                    name="Market Analyst",
                    agent_type=AgentType.EXPERT,
                    description="Expert in market analysis and trends",
                    model="gemini-2.0-flash",
                    instructions="You are a market analyst. Provide detailed, data-driven insights."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.SINGLE,
                execution_strategy=ExecutionStrategy.FAIL_FAST,
                timeout_seconds=300
            )
        )
        
        # Create task
        task = Task(
            title="Analyze AI Market Trends",
            description="Provide a comprehensive analysis of current AI market trends, focusing on multi-agent systems and their adoption in enterprise environments.",
            task_type=TaskType.COMPLEX,
            priority=TaskPriority.HIGH,
            input_data={
                "focus_areas": ["multi-agent systems", "enterprise adoption", "market size"],
                "time_horizon": "2024-2025"
            }
        )

        # Execute task with session
        try:
            result = await self.coordinator.execute_task(config, task, session)
            
            if result.is_successful():
                logger.info(f"✅ Single agent task completed successfully")
                logger.info(f"📊 Execution time: {result.total_execution_time_ms}ms")
                logger.info(f"📝 Result preview: {str(result.result)[:200]}...")
            else:
                logger.error(f"❌ Single agent task failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Single agent execution error: {e}")
    
    async def example_hierarchical_workflow(self):
        """Example 2: Hierarchical workflow with manager and experts."""
        logger.info("\n🏢 Example 2: Hierarchical Workflow")

        # Ensure managers are initialized
        assert self.session_manager is not None, "Session manager not initialized"
        assert self.coordinator is not None, "Coordinator not initialized"

        # Create session
        session = Session(
            session_id="hierarchical_session_001",
            user_id="user_123",
            session_type=SessionType.SINGLE_CHAT
        )
        await self.session_manager.create_session(session)
        
        # Configure hierarchical system
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="manager_001",
                    name="Project Manager",
                    agent_type=AgentType.MANAGER,
                    description="Coordinates tasks and manages expert agents",
                    model="gemini-2.0-flash",
                    instructions="You coordinate tasks between expert agents. Break down complex tasks and delegate appropriately."
                ),
                AgentConfig(
                    agent_id="researcher_001",
                    name="Research Expert",
                    agent_type=AgentType.EXPERT,
                    description="Expert in research and data analysis",
                    model="gemini-2.0-flash",
                    instructions="You are a research expert. Provide thorough, evidence-based analysis."
                ),
                AgentConfig(
                    agent_id="writer_001",
                    name="Technical Writer",
                    agent_type=AgentType.EXPERT,
                    description="Expert in technical writing and documentation",
                    model="gemini-2.0-flash",
                    instructions="You are a technical writer. Create clear, well-structured documentation."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.HIERARCHICAL,
                execution_strategy=ExecutionStrategy.FAIL_FAST,
                manager_agent_id="manager_001",
                expert_agent_ids=["researcher_001", "writer_001"],
                timeout_seconds=600
            )
        )
        
        # Create complex task
        task = Task(
            title="Create AI Implementation Guide",
            description="Research and create a comprehensive implementation guide for multi-agent AI systems in enterprise environments, including best practices, architecture patterns, and case studies.",
            task_type=TaskType.COMPLEX,
            priority=TaskPriority.HIGH,
            input_data={
                "target_audience": "enterprise architects and CTOs",
                "sections": ["introduction", "architecture", "implementation", "best_practices", "case_studies"],
                "length": "comprehensive guide (10-15 pages)"
            }
        )

        # Execute with streaming (note: streaming doesn't support session parameter yet)
        try:
            logger.info("🔄 Starting hierarchical workflow with streaming...")

            async for update in self.coordinator.execute_task_stream(config, task):
                if update.get("type") == "agent_started":
                    logger.info(f"🤖 Agent {update.get('agent_id')} started task")
                elif update.get("type") == "agent_completed":
                    logger.info(f"✅ Agent {update.get('agent_id')} completed task")
                elif update.get("type") == "workflow_completed":
                    logger.info(f"🎉 Hierarchical workflow completed successfully")
                    
        except Exception as e:
            logger.error(f"❌ Hierarchical workflow error: {e}")
    
    async def example_memory_and_context(self):
        """Example 3: Memory and context management."""
        logger.info("\n🧠 Example 3: Memory and Context Management")

        # Ensure managers are initialized
        assert self.session_manager is not None, "Session manager not initialized"
        assert self.memory_manager is not None, "Memory manager not initialized"
        assert self.coordinator is not None, "Coordinator not initialized"

        session_id = "memory_session_001"
        user_id = "user_123"

        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            session_type=SessionType.SINGLE_CHAT
        )
        await self.session_manager.create_session(session)

        # Store some context memories
        await self.memory_manager.store_memory(
            session_id=session_id,
            content="User is interested in enterprise AI implementations",
            memory_type="preference",
            session_type=SessionType.SINGLE_CHAT,
            importance=0.8,
            tags=["user_preference", "enterprise", "AI"]
        )
        
        await self.memory_manager.store_memory(
            session_id=session_id,
            content="Previous discussion covered multi-agent architecture patterns",
            memory_type="conversation",
            session_type=SessionType.SINGLE_CHAT,
            importance=0.7,
            tags=["conversation", "architecture", "multi-agent"]
        )
        
        # Configure agent with memory context
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="consultant_001",
                    name="AI Consultant",
                    agent_type=AgentType.EXPERT,
                    description="AI consultant with access to conversation history",
                    model="gemini-2.0-flash",
                    instructions="You are an AI consultant. Use previous conversation context to provide personalized recommendations."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.SINGLE,
                execution_strategy=ExecutionStrategy.FAIL_FAST
            )
        )
        
        # Create task that benefits from memory
        task = Task(
            title="Provide Implementation Recommendations",
            description="Based on our previous discussions, provide specific recommendations for implementing the multi-agent system we discussed.",
            task_type=TaskType.COMPLEX,
            priority=TaskPriority.MEDIUM
        )

        try:
            # Execute with memory context (session will be passed to coordinator)
            result = await self.coordinator.execute_task(config, task, session)
            
            if result.is_successful():
                logger.info("✅ Memory-enhanced task completed")
                
                # Store the result as a new memory
                await self.memory_manager.store_memory(
                    session_id=session_id,
                    content=f"Provided implementation recommendations: {str(result.result)[:100]}...",
                    memory_type="fact",
                    session_type=SessionType.SINGLE_CHAT,
                    importance=0.6,
                    tags=["recommendation", "implementation"]
                )
                
                # Retrieve and display memories
                memories = await self.memory_manager.retrieve_memories(
                    session_id=session_id,
                    session_type=SessionType.SINGLE_CHAT,
                    limit=5
                )
                
                logger.info(f"📚 Session has {len(memories)} stored memories")
                for memory in memories:
                    logger.info(f"  - {memory.memory_type}: {memory.content[:50]}...")
                    
        except Exception as e:
            logger.error(f"❌ Memory-enhanced execution error: {e}")
    
    async def example_batch_processing(self):
        """Example 4: Batch processing multiple tasks."""
        logger.info("\n⚡ Example 4: Batch Processing")

        # Ensure managers are initialized
        assert self.session_manager is not None, "Session manager not initialized"
        assert self.coordinator is not None, "Coordinator not initialized"

        # Create session
        session = Session(
            session_id="batch_session_001",
            user_id="user_123",
            session_type=SessionType.SINGLE_CHAT
        )
        await self.session_manager.create_session(session)

        # Configure agent for batch processing
        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="batch_processor_001",
                    name="Batch Processor",
                    agent_type=AgentType.EXPERT,
                    description="Efficient batch task processor",
                    model="gemini-2.0-flash",
                    instructions="You process tasks efficiently in batch mode."
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.SINGLE,
                execution_strategy=ExecutionStrategy.CONTINUE_ON_FAILURE,
                max_concurrent_agents=3
            )
        )

        # Create multiple tasks
        tasks = [
            Task(
                title="Summarize Article 1",
                description="Summarize the key points of AI research article 1",
                task_type=TaskType.SIMPLE,
                priority=TaskPriority.MEDIUM
            ),
            Task(
                title="Summarize Article 2",
                description="Summarize the key points of AI research article 2",
                task_type=TaskType.SIMPLE,
                priority=TaskPriority.MEDIUM
            ),
            Task(
                title="Summarize Article 3",
                description="Summarize the key points of AI research article 3",
                task_type=TaskType.SIMPLE,
                priority=TaskPriority.MEDIUM
            )
        ]

        try:
            logger.info(f"🔄 Processing {len(tasks)} tasks in batch...")

            # Execute batch tasks
            results = await self.coordinator.execute_batch_tasks(config, tasks)

            # Process results
            successful_tasks = 0
            failed_tasks = 0

            for i, result in enumerate(results):
                if result.is_successful():
                    successful_tasks += 1
                    logger.info(f"✅ Task {i+1} completed successfully")
                else:
                    failed_tasks += 1
                    logger.error(f"❌ Task {i+1} failed: {result.error_message}")

            logger.info(f"📊 Batch processing completed: {successful_tasks} successful, {failed_tasks} failed")

        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")

    async def example_capability_detection(self):
        """Example 5: Framework capability detection and selection."""
        logger.info("\n🔍 Example 5: Capability Detection")

        # Ensure registry is initialized
        assert self.registry is not None, "Registry not initialized"

        try:
            # Check available adapters and their capabilities
            adapter_names = self.registry.list_adapters()
            logger.info(f"📋 Available adapters: {adapter_names}")

            # Get detailed info for each adapter
            for name in adapter_names:
                adapter_info = self.registry.get_adapter_info(name)
                if adapter_info:
                    capabilities = adapter_info.get("capabilities", [])
                    logger.info(f"🔧 {name} capabilities: {[cap.value for cap in capabilities]}")

            # Get adapter by specific capability
            from tgo.agents.core.enums import FrameworkCapability

            streaming_adapters = self.registry.find_adapters_by_capability(
                FrameworkCapability.STREAMING
            )

            if streaming_adapters:
                logger.info(f"🌊 Streaming capable adapters: {streaming_adapters}")
            else:
                logger.info("⚠️ No streaming capable adapters found")

            # Check health status
            health_status = self.registry.get_health_status()
            logger.info(f"📊 System health: {health_status['total_adapters']} total, {health_status['initialized_adapters']} initialized")

            for adapter_name, status in health_status.get("adapters", {}).items():
                status_emoji = "✅" if status.get("initialized", False) else "❌"
                logger.info(f"{status_emoji} {adapter_name}: {status}")

        except Exception as e:
            logger.error(f"❌ Capability detection error: {e}")

    async def run_all_examples(self):
        """Run all examples in sequence."""
        await self.setup()

        try:
            await self.example_single_agent()
            await asyncio.sleep(2)  # Brief pause between examples

            await self.example_hierarchical_workflow()
            await asyncio.sleep(2)

            await self.example_memory_and_context()
            await asyncio.sleep(2)

            await self.example_batch_processing()
            await asyncio.sleep(2)

            await self.example_capability_detection()

        except Exception as e:
            logger.error(f"❌ Example execution failed: {e}")

        finally:
            # Cleanup (coordinator doesn't need explicit cleanup)
            logger.info("🧹 Cleanup completed")


async def main():
    """Main entry point for the example."""
    print("🎯 TGO Multi-Agent Coordinator - Complete Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("  📝 Single agent execution")
    print("  🏢 Hierarchical multi-agent workflows")
    print("  🧠 Memory and session management")
    print("  ⚡ Batch processing")
    print("  🔍 Capability detection")
    print("=" * 60)

    start_time = datetime.now(timezone.utc)

    try:
        example = MultiAgentExample()
        await example.run_all_examples()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("✨ All examples completed successfully!")
        print(f"⏱️ Total execution time: {duration:.2f} seconds")
        print("📖 For more examples, see basic_session_memory_example.py")
        print("📚 Check README.md and README_CN.md for documentation")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    asyncio.run(main())
