"""
Base framework adapter implementation.

This module provides the base implementation for all AI framework adapters,
defining common functionality and the interface that all adapters must implement.
"""

import logging
import asyncio
from abc import abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from ..core.interfaces import BaseFrameworkAdapter as IBaseFrameworkAdapter, MemoryManager
from ..core.models import (
    Task, AgentConfig, AgentInstance, ExecutionContext,
    AgentExecutionResult, ToolCallResult, KnowledgeBaseQueryResult,
)
from ..tools.fastmcp_tool_manager import (
    MCPToolManager
)
from ..core.enums import FrameworkCapability, AgentStatus, WorkflowType
from ..core.exceptions import (
    FrameworkError, FrameworkInitializationError,
    AgentNotFoundError, AgentCreationError, AgentExecutionError
)


class BaseFrameworkAdapter(IBaseFrameworkAdapter):
    """Enhanced base framework adapter with common functionality.
    
    This class extends the core interface with practical implementations
    of common functionality that most framework adapters will need.
    """
    
    def __init__(self, framework_name: str, version: str):
        super().__init__(framework_name, version)
        self._logger = logging.getLogger(f"{__name__}.{framework_name}")
        self._agents: Dict[str, AgentInstance] = {}
        self._agent_locks: Dict[str, asyncio.Lock] = {}
        self._execution_contexts: Dict[str, ExecutionContext] = {}

        # Memory manager will be injected by coordinator
        self._memory_manager: Optional[MemoryManager] = None

        # MCP tool manager will be injected by coordinator
        self._mcp_tool_manager: Optional[MCPToolManager] = None  # Avoid circular import

        # Default capabilities - subclasses should override
        self._capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.TOOL_CALLING,
        ]
    
    def _get_agent_lock(self, agent_id: str) -> asyncio.Lock:
        """Get or create a lock for an agent."""
        if agent_id not in self._agent_locks:
            self._agent_locks[agent_id] = asyncio.Lock()
        return self._agent_locks[agent_id]

    def set_memory_manager(self, memory_manager: Optional[MemoryManager]) -> None:
        """Set the memory manager for this adapter.

        This method is called by the coordinator to inject the memory manager.

        Args:
            memory_manager: Memory manager instance or None
        """
        self._memory_manager = memory_manager
        self._logger.debug(f"Memory manager {'set' if memory_manager else 'cleared'} for {self.framework_name}")

    def get_memory_manager(self) -> Optional[MemoryManager]:
        """Get the current memory manager.

        Returns:
            Memory manager instance or None if not set
        """
        return self._memory_manager

    def set_mcp_tool_manager(self, mcp_tool_manager: Optional[Any]) -> None:
        """Set the MCP tool manager for this adapter.

        This method is called by the coordinator to inject the MCP tool manager.

        Args:
            mcp_tool_manager: MCP tool manager instance or None
        """
        self._mcp_tool_manager = mcp_tool_manager
        self._logger.debug(f"MCP tool manager {'set' if mcp_tool_manager else 'cleared'} for {self.framework_name}")

    def get_mcp_tool_manager(self) -> Optional[Any]:
        """Get the current MCP tool manager.

        Returns:
            MCP tool manager instance or None if not set
        """
        return self._mcp_tool_manager
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        return self._agents.get(agent_id)
    
    async def list_agents(self) -> List[AgentInstance]:
        """List all agent instances."""
        return list(self._agents.values())
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        agent.status = status
        agent.update_activity()
        return True
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent instance."""
        async with self._get_agent_lock(agent_id):
            if agent_id not in self._agents:
                return False
            
            # Clean up agent-specific resources
            await self._cleanup_agent_resources(agent_id)
            
            # Remove from tracking
            del self._agents[agent_id]
            self._agent_locks.pop(agent_id, None)
            self._execution_contexts.pop(agent_id, None)
            
            self._logger.info(f"Deleted agent: {agent_id}")
            return True
    
    async def _cleanup_agent_resources(self, agent_id: str) -> None:
        """Clean up resources associated with an agent.
        
        Subclasses can override this to perform framework-specific cleanup.
        """
        pass
    
    async def _validate_agent_config(self, config: AgentConfig) -> None:
        """Validate agent configuration.
        
        Args:
            config: Agent configuration to validate
            
        Raises:
            AgentCreationError: If configuration is invalid
        """
        if not config.name or not config.name.strip():
            raise AgentCreationError("Agent name cannot be empty")
        
        if not config.model or not config.model.strip():
            raise AgentCreationError("Agent model cannot be empty")
        
        if config.max_iterations <= 0:
            raise AgentCreationError("Max iterations must be positive")
        
        if config.timeout_seconds is not None and config.timeout_seconds <= 0:
            raise AgentCreationError("Timeout must be positive")
        
        if not (0.0 <= config.temperature <= 2.0):
            raise AgentCreationError("Temperature must be between 0.0 and 2.0")
    
    async def _create_agent_instance(self, config: AgentConfig) -> AgentInstance:
        """Create an agent instance from configuration.
        
        This method handles the common logic for creating agent instances.
        Subclasses should call this after creating their framework-specific agents.
        """
        await self._validate_agent_config(config)
        
        instance = AgentInstance(
            agent_id=config.agent_id,
            config=config,
            status=AgentStatus.IDLE,
            created_at=datetime.now(timezone.utc),
            current_task_id=None,
            last_activity=datetime.now(timezone.utc)
        )
        
        self._agents[config.agent_id] = instance
        self._logger.info(f"Created agent instance: {config.agent_id}")
        
        return instance
    
    async def _prepare_execution_context(
        self, 
        agent_id: str, 
        task: Task
    ) -> ExecutionContext:
        """Prepare execution context for a task.
        
        Args:
            agent_id: ID of the executing agent
            task: Task to execute
            
        Returns:
            Execution context
        """
        context = ExecutionContext(
            task_id=task.task_id,
            agent_id=agent_id,
            framework_name=self.framework_name,
            created_at=datetime.now(timezone.utc),
            user_id="system",
            session_id=f"session_{agent_id}_{task.task_id}",
            workflow_type=WorkflowType.SINGLE
        )
        
        self._execution_contexts[context.execution_id] = context
        return context
    
    async def _cleanup_execution_context(self, context: ExecutionContext) -> None:
        """Clean up execution context after task completion."""
        self._execution_contexts.pop(context.execution_id, None)
    
    async def _handle_execution_error(
        self, 
        error: Exception, 
        agent_id: str, 
        task: Task
    ) -> AgentExecutionResult:
        """Handle execution errors and create appropriate result.
        
        Args:
            error: The exception that occurred
            agent_id: ID of the agent that failed
            task: Task that failed
            
        Returns:
            AgentExecutionResult with error information
        """
        error_message = str(error)
        self._logger.error(f"Agent {agent_id} execution failed for task {task.task_id}: {error_message}")
        
        # Update agent status
        await self.update_agent_status(agent_id, AgentStatus.ERROR)
        
        now = datetime.now(timezone.utc)
        return AgentExecutionResult(
            success=False,
            result=None,
            error_message=error_message,
            execution_time_ms=0,
            started_at=now,
            completed_at=now
        )
    
    @asynccontextmanager
    async def _agent_execution_context(self, agent_id: str, task: Task):
        """Context manager for agent execution.
        
        Handles common setup and cleanup for agent execution.
        """
        # Update agent status to busy
        await self.update_agent_status(agent_id, AgentStatus.BUSY)
        
        # Prepare execution context
        context = await self._prepare_execution_context(agent_id, task)
        
        try:
            yield context
        finally:
            # Clean up execution context
            await self._cleanup_execution_context(context)
            
            # Reset agent status to idle
            await self.update_agent_status(agent_id, AgentStatus.IDLE)
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    async def _create_framework_agent(self, config: AgentConfig,context: ExecutionContext) -> Any:
        """Create a framework-specific agent instance.
        
        Args:
            config: Agent configuration
            
        Returns:
            Framework-specific agent instance
        """
        pass
    
    @abstractmethod
    async def _execute_framework_task(
        self, 
        framework_agent: Any,
        task: Task, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Execute a task using the framework-specific agent.
        
        Args:
            framework_agent: Framework-specific agent instance
            task: Task to execute
            context: Execution context
            
        Returns:
            Execution result
        """
        pass
    
    # Default implementations that can be overridden
    async def initialize(self) -> None:
        """Initialize the framework adapter."""
        if self._initialized:
            return
        
        try:
            await self._initialize_framework()
            self._initialized = True
            self._logger.info(f"Initialized {self.framework_name} adapter")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize {self.framework_name} adapter: {e}")
            raise FrameworkInitializationError(
                f"Failed to initialize {self.framework_name}: {e}",
                framework_name=self.framework_name
            )
    
    async def cleanup(self) -> None:
        """Clean up framework resources."""
        if not self._initialized:
            return
        
        try:
            # Clean up all agents
            agent_ids = list(self._agents.keys())
            for agent_id in agent_ids:
                await self.delete_agent(agent_id)
            
            # Framework-specific cleanup
            await self._cleanup_framework()
            
            self._initialized = False
            self._logger.info(f"Cleaned up {self.framework_name} adapter")
            
        except Exception as e:
            self._logger.error(f"Error during cleanup of {self.framework_name} adapter: {e}")
    
    async def create_agent(self, config: AgentConfig,context: ExecutionContext) -> AgentInstance:
        """Create an agent instance."""
        if not self._initialized:
            raise FrameworkError(f"{self.framework_name} adapter not initialized")

        if config.agent_id in self._agents:
            raise AgentCreationError(f"Agent {config.agent_id} already exists")

        async with self._get_agent_lock(config.agent_id):
            try:
                # Create framework-specific agent
                framework_agent = await self._create_framework_agent(config,context)

                # Create agent instance
                instance = await self._create_agent_instance(config)

                # Store framework agent reference
                instance.session_data['framework_agent'] = framework_agent

                return instance

            except Exception as e:
                # Clean up on failure - but don't use delete_agent to avoid deadlock
                self._agents.pop(config.agent_id, None)
                self._execution_contexts.pop(config.agent_id, None)
                await self._cleanup_agent_resources(config.agent_id)
                raise AgentCreationError(f"Failed to create agent {config.agent_id}: {e}")
    
    async def execute_task(
        self,
        agent_id: str,
        task: Task,
        context: Optional[ExecutionContext] = None
    ) -> AgentExecutionResult:
        """Execute a task with an agent."""
        if not self._initialized:
            raise FrameworkError(f"{self.framework_name} adapter not initialized")
        
        agent = self._agents.get(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")
        
        if not agent.is_available():
            raise AgentExecutionError(f"Agent {agent_id} is not available")
        
        async with self._agent_execution_context(agent_id, task) as exec_context:
            try:
                framework_agent = agent.session_data.get('framework_agent')
                if not framework_agent:
                    raise AgentExecutionError(f"Framework agent not found for {agent_id}")
                
                result = await self._execute_framework_task(framework_agent, task, exec_context)
                
                if result.is_successful():
                    self._logger.info(f"Agent {agent_id} successfully executed task {task.task_id}")
                else:
                    self._logger.warning(f"Agent {agent_id} failed to execute task {task.task_id}: {result.error_message}")
                
                return result
                
            except Exception as e:
                return await self._handle_execution_error(e, agent_id, task)
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    async def query_knowledge_base(
        self,
        agent_id: str,
        kb_id: str,
        kb_name: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None
    ) -> KnowledgeBaseQueryResult:
        """Query a knowledge base through an agent."""
        pass

    # Framework-specific methods that subclasses can override
    async def _initialize_framework(self) -> None:
        """Initialize framework-specific resources."""
        pass

    async def _cleanup_framework(self) -> None:
        """Clean up framework-specific resources."""
        pass
