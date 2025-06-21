"""
Google ADK framework adapter.

This module provides integration with Google Agent Development Kit (ADK),
supporting both single-agent and multi-agent execution patterns.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


# Google ADK imports - required dependencies
from google.adk.agents import LlmAgent
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .base_adapter import BaseFrameworkAdapter
from ..core.interfaces import MemoryManager
from ..core.models import (
    Task, AgentConfig, ExecutionContext,
    AgentExecutionResult, ToolCallResult, KnowledgeBaseQueryResult,
)
from ..core.enums import FrameworkCapability, AgentType, SessionType
from ..core.exceptions import AgentCreationError
# Note: tool_manager and kb_manager imports removed for testing
# These would be imported from the actual tool and knowledge base modules
# from ...tools.tool_manager import tool_manager
# from ...knowledge.kb_manager import kb_manager

logger = logging.getLogger(__name__)


class GoogleADKAdapter(BaseFrameworkAdapter):
    """Google ADK framework adapter.
    
    Provides integration with Google Agent Development Kit, supporting:
    - Single agent execution
    - Multi-agent coordination (hierarchical, sequential, parallel)
    - Tool calling integration
    - Knowledge base querying
    - Streaming execution (if supported by ADK)
    """
    
    def __init__(self):
        super().__init__("google-adk", "1.4.1")

        # Set capabilities
        self._capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.MULTI_AGENT,
            FrameworkCapability.TOOL_CALLING,
            FrameworkCapability.KNOWLEDGE_BASE,
            FrameworkCapability.MEMORY,
            FrameworkCapability.STREAMING,
        ]

        # Memory manager will be injected by coordinator
        self._memory_manager: Optional[MemoryManager] = None
        

        # ADK-specific storage
        self._adk_agents: Dict[str, Any] = {}  # agent_id -> ADK agent instance
        self._run_configs: Dict[str, AgentConfig] = {}  # agent_id -> RunConfig

        # Configuration
        self._default_timeout = 300  # 5 minutes
        self._max_iterations = 10
        self._retry_attempts = 3
    
    async def _initialize_framework(self) -> None:
        """Initialize Google ADK framework."""
        logger.info("Google ADK is available and ready")

        # Initialize any ADK-specific resources here
        # For example, authentication, configuration, etc.
    
    async def _cleanup_framework(self) -> None:
        """Clean up Google ADK framework resources."""
        # Clean up ADK-specific resources
        self._adk_agents.clear()
        self._run_configs.clear()

    async def _cleanup_agent_resources(self, agent_id: str) -> None:
        """Clean up resources associated with a specific agent."""
        # Remove from ADK-specific tracking
        self._adk_agents.pop(agent_id, None)
        self._run_configs.pop(agent_id, None)

    async def _create_framework_agent(self, config: AgentConfig) -> Any:
        """Create a Google ADK agent instance."""
        try:
            # Get tools for the agent
            tools = await self._get_tools_for_agent(config.tools)
            
            # Create ADK agent based on type
            if config.agent_type == AgentType.MANAGER:
                adk_agent = await self._create_manager_agent(config, tools)
            elif config.agent_type == AgentType.EXPERT:
                adk_agent = await self._create_expert_agent(config, tools)
            else:
                adk_agent = await self._create_llm_agent(config, tools)
            # Create run configuration
            run_config = self._create_run_config(config)
            
            # Store ADK-specific instances
            self._adk_agents[config.agent_id] = adk_agent
            self._run_configs[config.agent_id] = run_config
            
            logger.info(f"Created Google ADK agent: {config.agent_id}")
            return adk_agent
            
        except Exception as e:
            logger.error(f"Failed to create Google ADK agent: {e}")
            raise AgentCreationError(f"Failed to create Google ADK agent: {e}")
    
    async def _create_manager_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create a manager agent for coordination."""
        instructions = config.instructions or self._get_default_manager_instructions()

        return LlmAgent(
            name=config.agent_id,
            model=config.model,
            instruction=instructions,
            description=config.description or "Manager agent for task coordination",
            tools=tools
        )
    
    async def _create_expert_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create an expert agent for specialized tasks."""
        instructions = config.instructions or self._get_default_expert_instructions()

        return LlmAgent(
            name=config.agent_id,
            model=config.model,
            instruction=instructions,
            description=config.description or "Expert agent for specialized tasks",
            tools=tools
        )
    
    async def _create_llm_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create a general LLM agent."""
        instructions = config.instructions or "You are a helpful AI assistant."

        return LlmAgent(
            name=config.name,
            model=config.model,
            instruction=instructions,
            description=config.description or "General purpose AI agent",
            tools=tools
        )
    
    def _create_run_config(self, config: AgentConfig) -> Any:
        """Create ADK run configuration."""

        return config
    
    async def _get_tools_for_agent(self, tool_ids: List[str]) -> List[Any]:
        """Get ADK tools for an agent."""
        tools: List[Any] = []

        # Add Google Search
        if google_search:
            tools.append(google_search)

        # Add custom tools through tool manager
        for tool_id in tool_ids:
            try:
                # This would integrate with the tool manager
                # For now, we'll skip custom tool integration
                pass
            except Exception as e:
                logger.warning(f"Failed to load tool {tool_id}: {e}")

        return tools
    
    async def _execute_framework_task(
        self,
        framework_agent: Any,
        task: Task,
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Execute a task using Google ADK agent."""
        start_time = datetime.now(timezone.utc)
        try:
            # Retrieve relevant memories if session context is available
            enhanced_task_input = await self._prepare_task_input_with_memory(task, context)

            # Get run configuration
            run_config = self._run_configs.get(context.agent_id or "")
            # Execute with ADK
            result = await self._execute_adk_task(framework_agent, enhanced_task_input, run_config,context)

            # Process result
            execution_result = await self._process_execution_result(
                result, task, context, start_time
            )

            # Store execution result in memory if successful and session context is available
            if execution_result.success and context.session_id and self._memory_manager:
                await self._store_execution_memory(task, execution_result, context)

            return execution_result

        except Exception as e:
            logger.error(f"Google ADK execution failed: {e}")
            end_time = datetime.now(timezone.utc)
            return AgentExecutionResult(
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=int((end_time - start_time).total_seconds() * 1000),
                started_at=start_time,
                completed_at=end_time
            )
    
    def _prepare_task_input(self, task: Task) -> str:
        """Prepare task input for ADK execution."""
        input_parts = [task.title]

        if task.description:
            input_parts.append(f"Description: {task.description}")

        if task.input_data:
            input_parts.append(f"Input data: {task.input_data}")

        return "\n".join(input_parts)

    async def _prepare_task_input_with_memory(self, task: Task, context: ExecutionContext) -> str:
        """Prepare task input enhanced with relevant memories."""
        # Start with basic task input
        task_input = self._prepare_task_input(task)

        # Add memories if available
        if context.session_id and self._memory_manager:
            try:
                # Determine session type (default to single chat if not available)
                session_type = SessionType.SINGLE_CHAT  # Could be enhanced to get from context

                # Retrieve relevant memories
                memories = await self._memory_manager.retrieve_memories(
                    session_id=context.session_id,
                    session_type=session_type,
                    agent_id=context.agent_id,
                    limit=5,
                    min_importance=0.3
                )

                if memories:
                    memory_context = "\n".join([
                        f"- {memory.content}" for memory in memories
                    ])
                    task_input += f"\n\nRelevant context from previous interactions:\n{memory_context}"

                    logger.debug(f"Enhanced task input with {len(memories)} memories")

            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        return task_input

    async def _store_execution_memory(
        self,
        task: Task,
        execution_result: AgentExecutionResult,
        context: ExecutionContext
    ) -> None:
        """Store execution result in memory."""
        if not self._memory_manager or not context.session_id:
            return

        try:
            # Determine session type (default to single chat if not available)
            session_type = SessionType.SINGLE_CHAT  # Could be enhanced to get from context

            # Create memory content
            result_summary = ""
            if execution_result.result:
                if hasattr(execution_result.result, 'get'):
                    result_summary = execution_result.result.get("response", str(execution_result.result))
                else:
                    result_summary = str(execution_result.result)
            else:
                result_summary = str(execution_result.result)

            memory_content = f"Task '{task.title}' completed successfully. Result: {result_summary}"

            # Store memory
            await self._memory_manager.store_memory(
                session_id=context.session_id,
                content=memory_content,
                memory_type="fact",
                session_type=session_type,
                agent_id=context.agent_id,
                importance=0.7,  # Execution results are fairly important
                tags=["task_execution", "result"],
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type.value if task.task_type else "unknown",
                    "execution_time_ms": execution_result.execution_time_ms
                }
            )

            logger.debug(f"Stored execution memory for task: {task.task_id}")

        except Exception as e:
            logger.warning(f"Failed to store execution memory: {e}")

    async def _execute_adk_task(self, agent: BaseAgent, task_input: str, config: AgentConfig | None, context: ExecutionContext) -> Event | None:
        """Execute task with ADK agent."""
        # Note: run_config parameter available for future use

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="tgo",
            session_id=context.session_id,
            user_id=context.user_id,
        )
        runner = Runner(
           app_name= "tgo",
           agent= agent,
           session_service= session_service
       )
        user_content = types.Content(role='user', parts=[types.Part(text=task_input)])
        
        for event in runner.run(user_id=context.user_id, session_id=context.session_id, new_message=user_content):
            if event.is_final_response() and event.content and event.content.parts:
                return event

        
    
    async def _process_execution_result(
        self,
        adk_result: Event,
        task: Task,
        context: ExecutionContext,
        start_time: datetime
    ) -> AgentExecutionResult:
        """Process ADK execution result."""
        # Note: task and context parameters available for future use
        end_time = datetime.now(timezone.utc)
        execution_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract result data
        result_data: Dict[str, Any]
        if adk_result and adk_result.content and adk_result.content.parts:
            result_data = {"response": adk_result.content.parts[0].text}
        else:
            result_data = {"response": "No final response received."}
        
        functionResponses = adk_result.get_function_responses()
        toolCalls: List[ToolCallResult] = []
        if functionResponses:
            for functionResponse in functionResponses:
                toolCalls.append(ToolCallResult(
                    tool_name=functionResponse.name,
                    tool_id=functionResponse.name,
                    success=True,
                    result=functionResponse.response,
                    error_message=None,
                    execution_time_ms=execution_time
                ))

        return AgentExecutionResult(
            success=True,
            result=result_data,
            error_message=None,
            execution_time_ms=execution_time,
            started_at=start_time,
            completed_at=end_time,
            tool_calls=toolCalls,  # TODO: Extract tool calls from ADK result
            kb_queries=[],  # TODO: Extract KB queries from ADK result
            reasoning_steps=[],  # TODO: Extract reasoning steps
            intermediate_results=[]
        )
    
    def _get_default_manager_instructions(self) -> str:
        """Get default instructions for manager agents."""
        return """You are a manager agent responsible for coordinating and delegating tasks to expert agents.
        
Your responsibilities:
1. Analyze complex tasks and break them down into subtasks
2. Assign subtasks to appropriate expert agents
3. Monitor progress and coordinate between agents
4. Synthesize results from multiple agents
5. Ensure quality and completeness of final results

Always think step by step and provide clear reasoning for your decisions."""
    
    def _get_default_expert_instructions(self) -> str:
        """Get default instructions for expert agents."""
        return """You are an expert agent specialized in executing specific tasks within your domain of expertise.
        
Your responsibilities:
1. Execute assigned tasks with high quality and accuracy
2. Use available tools and knowledge bases effectively
3. Provide detailed and well-reasoned responses
4. Ask for clarification when task requirements are unclear
5. Report progress and any issues encountered

Focus on delivering excellent results within your area of specialization."""
    
    # Tool calling implementation
    async def call_tool(
        self,
        agent_id: str,
        tool_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> ToolCallResult:
        """Call a tool through the agent."""
        # Note: agent_id and context parameters available for future use
        start_time = datetime.now(timezone.utc)
        
        try:
            # Mock tool execution for testing
            # In real implementation, this would use the tool manager
            result = {
                "tool_result": f"Mock result from tool {tool_name}",
                "parameters_used": parameters
            }
            
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolCallResult(
                tool_name=tool_name,
                tool_id=tool_id,
                success=True,
                result=result,
                error_message=None,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolCallResult(
                tool_name=tool_name,
                tool_id=tool_id,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    # Knowledge base querying implementation
    async def query_knowledge_base(
        self,
        agent_id: str,
        kb_id: str,
        kb_name: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None
    ) -> KnowledgeBaseQueryResult:
        """Query a knowledge base through the agent."""
        # Note: agent_id, parameters, and context parameters available for future use
        start_time = datetime.now(timezone.utc)
        
        try:
            # Mock knowledge base query for testing
            # In real implementation, this would use the knowledge base manager
            result = {
                "results": [
                    {"content": f"Mock KB result for query: {query}", "score": 0.9}
                ]
            }
            
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return KnowledgeBaseQueryResult(
                kb_name=kb_name,
                kb_id=kb_id,
                query=query,
                success=True,
                results=result.get("results", []),
                results_count=len(result.get("results", [])),
                error_message=None,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return KnowledgeBaseQueryResult(
                kb_name=kb_name,
                kb_id=kb_id,
                query=query,
                success=False,
                results=[],
                results_count=0,
                error_message=str(e),
                execution_time_ms=execution_time
            )
