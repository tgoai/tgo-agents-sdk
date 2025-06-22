"""
Google ADK framework adapter.

This module provides integration with Google Agent Development Kit (ADK),
supporting both single-agent and multi-agent execution patterns with
optimized MCP tool integration and memory management.
"""

import asyncio
import concurrent.futures
import inspect
import logging
from typing import Dict, Any, List, Optional, Callable, Type
from datetime import datetime, timezone
from dataclasses import dataclass

# Google ADK imports - required dependencies
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from .base_adapter import BaseFrameworkAdapter
from ..core.interfaces import MemoryManager
from ..core.models import (
    Task, AgentConfig, ExecutionContext,
    AgentExecutionResult, ToolCallResult, KnowledgeBaseQueryResult, MCPToolCallResult
)
from ..core.enums import FrameworkCapability, AgentType, SessionType
from ..core.exceptions import AgentCreationError

logger = logging.getLogger(__name__)


@dataclass
class ADKConfig:
    """Configuration for Google ADK adapter."""
    timeout_seconds: int = 300  # 5 minutes
    max_iterations: int = 10
    retry_attempts: int = 3
    mcp_tool_timeout: int = 30  # seconds
    memory_limit: int = 5  # number of memories to retrieve
    memory_importance_threshold: float = 0.3


# Type aliases for better readability
ToolProcessor = Callable[[Any, ExecutionContext], Any]
AgentCreator = Callable[[AgentConfig, List[Any]], Any]

class GoogleADKAdapter(BaseFrameworkAdapter):
    """Google ADK framework adapter.

    Provides integration with Google Agent Development Kit, supporting:
    - Single agent execution
    - Multi-agent coordination (hierarchical, sequential, parallel)
    - Tool calling integration with optimized MCP support
    - Knowledge base querying
    - Memory management integration
    - Streaming execution (if supported by ADK)
    """

    def __init__(self, config: Optional[ADKConfig] = None):
        super().__init__("google-adk", "1.4.1")

        # Configuration
        self._config = config or ADKConfig()

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

        # Tool processors mapping
        self._tool_processors: Dict[str, ToolProcessor] = {
            'string': self._process_string_tool,
            'function': self._process_function_tool,
            'mcp': self._process_mcp_tool
        }

        # Agent creators mapping
        self._agent_creators: Dict[AgentType, AgentCreator] = {
            AgentType.MANAGER: self._create_manager_agent,
            AgentType.EXPERT: self._create_expert_agent
        }

    @property
    def config(self) -> ADKConfig:
        """Get the current configuration."""
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about managed agents."""
        return {
            "total_agents": len(self._adk_agents),
            "agent_ids": list(self._adk_agents.keys()),
            "config": {
                "timeout_seconds": self._config.timeout_seconds,
                "mcp_tool_timeout": self._config.mcp_tool_timeout,
                "memory_limit": self._config.memory_limit
            }
        }
    
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

    async def _create_framework_agent(self, config: AgentConfig, context: ExecutionContext) -> Any:
        """Create a Google ADK agent instance with optimized tool processing."""
        try:
            logger.debug(f"Creating Google ADK agent: {config.agent_id}")

            # Process tools from the unified tools array
            all_tools = await self._process_agent_tools(config, context)

            # Create ADK agent based on type using mapping
            creator = self._agent_creators.get(config.agent_type, self._create_llm_agent)
            adk_agent = await creator(config, all_tools)

            # Create and store run configuration
            run_config = self._create_run_config(config)
            self._adk_agents[config.agent_id] = adk_agent
            self._run_configs[config.agent_id] = run_config

            logger.info(f"✅ Created Google ADK agent: {config.agent_id} with {len(all_tools)} tools")
            return adk_agent

        except Exception as e:
            error_msg = f"Failed to create Google ADK agent {config.agent_id}: {e}"
            logger.error(error_msg)
            raise AgentCreationError(error_msg)
    
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

        # Add custom tools through tool manager
        for tool_id in tool_ids:
            try:
                # This would integrate with the tool manager
                # For now, we'll skip custom tool integration
                pass
            except Exception as e:
                logger.warning(f"Failed to load tool {tool_id}: {e}")

        return tools

    async def _process_agent_tools(self, config: AgentConfig, context: ExecutionContext) -> List[Any]:
        """Process tools from the unified tools array with improved error handling."""
        all_tools = []

        for tool in config.tools:
            try:
                processed_tool = await self._process_single_tool(tool, context)
                if processed_tool:
                    all_tools.append(processed_tool)
            except Exception as e:
                logger.error(f"Failed to process tool {tool}: {e}")
                # Continue processing other tools instead of failing completely

        logger.info(f"Processed {len(all_tools)} tools for agent {config.agent_id}")
        return all_tools

    async def _process_single_tool(self, tool: Any, context: ExecutionContext) -> Optional[Any]:
        """Process a single tool with type detection and appropriate processor."""
        tool_type = self._identify_tool_type(tool)
        processor = self._tool_processors.get(tool_type)

        if not processor:
            logger.warning(f"Unknown tool type: {type(tool)} for tool: {tool}")
            return None

        return await processor(tool, context)

    def _identify_tool_type(self, tool) -> str:
        """Identify the type of tool."""
        if isinstance(tool, str):
            return 'string'
        elif callable(tool):
            return 'function'
        elif hasattr(tool, 'server_id') and hasattr(tool, 'name'):
            return 'mcp'
        else:
            return 'unknown'

    async def _process_string_tool(self, tool: str,context: ExecutionContext) -> Optional[Any]:
        """Process string tool name."""
        string_tools = await self._get_tools_for_agent([tool])
        return string_tools[0] if string_tools else None

    async def _process_function_tool(self, tool: Callable,context: ExecutionContext) -> Any:
        """Process function tool."""
        return self._create_adk_function_tool_wrapper(tool)

    async def _process_mcp_tool(self, tool,context: ExecutionContext) -> Any:
        """Process MCP tool object."""
        return await self._create_adk_mcp_tool_from_object(tool,context)

    def _create_adk_function_tool_wrapper(self, func) -> Any:
        """Create an ADK tool wrapper for a function tool."""
        # Create Google ADK FunctionTool from function
        adk_tool = FunctionTool(func=func)
        return adk_tool

    async def _create_adk_mcp_tool_from_object(self, mcp_tool, context: ExecutionContext) -> Any:
        """Create an ADK tool wrapper from an MCP tool object with optimized async handling."""
        try:
            from google.adk.tools import FunctionTool

            def mcp_tool_function(**kwargs) -> str:
                """Optimized MCP tool function wrapper (sync)."""
                logger.debug(f"MCP tool '{mcp_tool.name}' called with args: {kwargs}")

                if not self._mcp_tool_manager:
                    logger.warning(f"No MCP tool manager available for {mcp_tool.name}")
                    return f"MCP tool {mcp_tool.name} unavailable (no manager)"

                return self._execute_mcp_tool_sync(mcp_tool, kwargs, context)

            # Set function attributes for Google ADK compatibility
            self._set_function_attributes(mcp_tool_function, mcp_tool)

            # Create Google ADK FunctionTool
            return FunctionTool(func=mcp_tool_function)

        except ImportError:
            logger.warning("Google ADK FunctionTool not available, using mock tool")
            return self._create_mock_mcp_tool(mcp_tool)

    def _execute_mcp_tool_sync(self, mcp_tool, kwargs: Dict[str, Any], context: ExecutionContext) -> str:
        """Execute MCP tool synchronously using thread pool."""
        try:
            def run_async_call():
                """Run the async MCP tool call in a separate thread with its own event loop."""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    if self._mcp_tool_manager:
                        return new_loop.run_until_complete(
                            self._mcp_tool_manager.call_tool(mcp_tool, kwargs, context, True)
                        )
                    return None
                finally:
                    new_loop.close()

            # Execute with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_call)
                try:
                    result = future.result(timeout=self._config.mcp_tool_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"MCP tool '{mcp_tool.name}' timed out after {self._config.mcp_tool_timeout}s")
                    return f"Tool '{mcp_tool.name}' timed out"

            return self._extract_mcp_result(result, mcp_tool.name)

        except Exception as e:
            logger.error(f"MCP tool '{mcp_tool.name}' execution failed: {e}")
            return f"Tool '{mcp_tool.name}' error: {e}"

    def _extract_mcp_result(self, result: MCPToolCallResult, tool_name: str) -> str:
        """Extract result content from MCP tool call result."""
        if result.success:
            return result.text or f"Tool '{tool_name}' executed successfully, but no text"
        return f"Tool '{tool_name}' executed failed"

    def _set_function_attributes(self, func: Callable, mcp_tool) -> None:
        """Set function attributes for Google ADK compatibility."""
        func.__name__ = mcp_tool.name
        func.__doc__ = mcp_tool.description

        # Apply function signature from MCP input_schema
        try:
            function_signature = self._create_function_signature_from_schema(mcp_tool)
            if function_signature:
                # Use setattr to avoid type checker warnings
                setattr(func, '__signature__', function_signature)
                setattr(func, '__annotations__', self._extract_annotations_from_schema(mcp_tool))
        except Exception as e:
            logger.warning(f"Failed to set function signature: {e}")

    def _create_mock_mcp_tool(self, mcp_tool):
        """Create a mock MCP tool for testing."""
        class MockMCPTool:
            def __init__(self, mcp_tool_obj):
                self.mcp_tool = mcp_tool_obj
                self.name = mcp_tool_obj.name
                self.__name__ = self.name
                self.description = mcp_tool_obj.description
                self.__doc__ = self.description

            async def __call__(self, **kwargs):
                return f"Mock MCP tool {self.name} called with {kwargs}"

        return MockMCPTool(mcp_tool)

    def _create_function_signature_from_schema(self, mcp_tool) -> Any:
        """Create a Python function signature from MCP tool input schema."""
        try:
            import inspect

            if not mcp_tool.input_schema or not isinstance(mcp_tool.input_schema, dict):
                return None

            schema = mcp_tool.input_schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            parameters = []

            for param_name, param_def in properties.items():
                param_type = param_def.get("type", "string")
                param_default = param_def.get("default", inspect.Parameter.empty)

                # Convert JSON schema types to Python types
                python_type = self._json_type_to_python_type(param_type)

                # Determine if parameter is required
                if param_name in required and param_default == inspect.Parameter.empty:
                    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                else:
                    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                    if param_default == inspect.Parameter.empty:
                        param_default = None

                param = inspect.Parameter(
                    name=param_name,
                    kind=kind,
                    default=param_default,
                    annotation=python_type
                )
                parameters.append(param)

            return inspect.Signature(parameters)

        except Exception as e:
            logger.warning(f"Failed to create function signature from MCP schema: {e}")
            return None

    def _extract_annotations_from_schema(self, mcp_tool) -> dict:
        """Extract type annotations from MCP tool schema."""
        try:
            if not mcp_tool.input_schema or not isinstance(mcp_tool.input_schema, dict):
                return {}

            schema = mcp_tool.input_schema
            properties = schema.get("properties", {})
            annotations = {}

            for param_name, param_def in properties.items():
                param_type = param_def.get("type", "string")
                python_type = self._json_type_to_python_type(param_type)
                annotations[param_name] = python_type

            # Add return type annotation
            annotations['return'] = str

            return annotations

        except Exception as e:
            logger.warning(f"Failed to extract annotations from MCP schema: {e}")
            return {}

    def _json_type_to_python_type(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        return type_mapping.get(json_type, str)

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
        """Prepare task input enhanced with relevant memories using optimized retrieval."""
        task_input = self._prepare_task_input(task)

        if not (context.session_id and self._memory_manager):
            return task_input

        try:
            memories = await self._retrieve_relevant_memories(context)
            if memories:
                memory_context = self._format_memory_context(memories)
                task_input += f"\n\nRelevant context from previous interactions:\n{memory_context}"
                logger.debug(f"Enhanced task input with {len(memories)} memories")
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")

        return task_input

    async def _retrieve_relevant_memories(self, context: ExecutionContext) -> List[Any]:
        """Retrieve relevant memories for the current context."""
        if not self._memory_manager:
            return []

        return await self._memory_manager.retrieve_memories(
            session_id=context.session_id,
            session_type=SessionType.SINGLE_CHAT,  # Could be enhanced to get from context
            agent_id=context.agent_id,
            limit=self._config.memory_limit,
            min_importance=self._config.memory_importance_threshold
        )

    def _format_memory_context(self, memories: List[Any]) -> str:
        """Format memories into a readable context string."""
        return "\n".join([f"- {memory.content}" for memory in memories])

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
        """Process ADK execution result with improved data extraction."""
        end_time = datetime.now(timezone.utc)
        execution_time = int((end_time - start_time).total_seconds() * 1000)

        # Extract result data
        result_data = self._extract_response_data(adk_result)

        # Extract tool calls
        tool_calls = self._extract_tool_calls(adk_result, execution_time)

        return AgentExecutionResult(
            success=True,
            result=result_data,
            error_message=None,
            execution_time_ms=execution_time,
            started_at=start_time,
            completed_at=end_time,
            tool_calls=tool_calls,
            kb_queries=[],  # TODO: Extract KB queries from ADK result
            reasoning_steps=[],  # TODO: Extract reasoning steps
            intermediate_results=[]
        )

    def _extract_response_data(self, adk_result: Event) -> Dict[str, Any]:
        """Extract response data from ADK result."""
        if adk_result and adk_result.content and adk_result.content.parts:
            return {"response": adk_result.content.parts[0].text}
        return {"response": "No final response received."}

    def _extract_tool_calls(self, adk_result: Event, execution_time: int) -> List[ToolCallResult]:
        """Extract tool calls from ADK result."""
        tool_calls = []

        if not adk_result:
            return tool_calls

        function_responses = adk_result.get_function_responses()
        if function_responses:
            for function_response in function_responses:
                tool_calls.append(ToolCallResult(
                    tool_name=function_response.name,
                    tool_id=function_response.name,
                    success=True,
                    result=function_response.response,
                    error_message=None,
                    execution_time_ms=execution_time
                ))

        return tool_calls
    
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
