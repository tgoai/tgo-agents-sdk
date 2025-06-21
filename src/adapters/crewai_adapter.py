"""
CrewAI framework adapter.

This module provides integration with CrewAI framework,
supporting crew-based multi-agent collaboration and task execution.
"""

import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone

# CrewAI imports with fallback
try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError as e:
    # Mock classes for when CrewAI is not available
    CREWAI_AVAILABLE = False
    logging.warning(f"CrewAI not available: {e}")
    
    class MockCrewAgent:
        def __init__(self, *args, **kwargs):
            self.role = kwargs.get('role', 'mock_agent')
            self.goal = kwargs.get('goal', 'mock_goal')
            self.backstory = kwargs.get('backstory', 'mock_backstory')
            self.tools = kwargs.get('tools', [])
    
    class MockCrewTask:
        def __init__(self, *args, **kwargs):
            self.description = kwargs.get('description', 'mock_task')
            self.agent = kwargs.get('agent')
    
    class MockCrew:
        def __init__(self, *args, **kwargs):
            self.agents = kwargs.get('agents', [])
            self.tasks = kwargs.get('tasks', [])
        
        async def kickoff(self):
            return "Mock CrewAI execution result"
    
    class MockBaseTool:
        def __init__(self, *args, **kwargs):
            pass
    
    CrewAgent = MockCrewAgent
    CrewTask = MockCrewTask
    Crew = MockCrew
    BaseTool = MockBaseTool

from .base_adapter import BaseFrameworkAdapter
from ..core.models import (
    Task, AgentConfig, AgentInstance, ExecutionContext,
    AgentExecutionResult, ToolCallResult, KnowledgeBaseQueryResult
)
from ..core.enums import FrameworkCapability, AgentType
from ..core.exceptions import AgentCreationError, AgentExecutionError

logger = logging.getLogger(__name__)


class CrewAIAdapter(BaseFrameworkAdapter):
    """CrewAI framework adapter.
    
    Provides integration with CrewAI framework, supporting:
    - Crew-based multi-agent collaboration
    - Role-based agent specialization
    - Task delegation and coordination
    - Tool integration
    - Hierarchical and collaborative workflows
    """
    
    def __init__(self):
        super().__init__("crewai", "0.1.0")
        
        # Set capabilities
        self._capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.MULTI_AGENT,
            FrameworkCapability.TOOL_CALLING,
            FrameworkCapability.KNOWLEDGE_BASE,
        ]
        
        # CrewAI-specific storage
        self._crew_agents: Dict[str, Any] = {}  # agent_id -> CrewAI agent
        self._crews: Dict[str, Any] = {}  # crew_id -> CrewAI crew
        self._agent_tools: Dict[str, List[Any]] = {}  # agent_id -> tools
    
    async def _initialize_framework(self) -> None:
        """Initialize CrewAI framework."""
        if not CREWAI_AVAILABLE:
            logger.warning("CrewAI not available, using mock implementation")
        else:
            logger.info("CrewAI is available and ready")
    
    async def _cleanup_framework(self) -> None:
        """Clean up CrewAI framework resources."""
        self._crew_agents.clear()
        self._crews.clear()
        self._agent_tools.clear()
    
    async def _create_framework_agent(self, config: AgentConfig) -> Any:
        """Create a CrewAI agent instance."""
        try:
            # Get tools for the agent
            tools = await self._get_tools_for_agent(config.tools)
            
            # Create CrewAI agent based on type
            if config.agent_type == AgentType.MANAGER:
                crew_agent = await self._create_manager_agent(config, tools)
            elif config.agent_type == AgentType.EXPERT:
                crew_agent = await self._create_expert_agent(config, tools)
            else:
                crew_agent = await self._create_general_agent(config, tools)
            
            # Store CrewAI-specific instances
            self._crew_agents[config.agent_id] = crew_agent
            self._agent_tools[config.agent_id] = tools
            
            logger.info(f"Created CrewAI agent: {config.agent_id}")
            return crew_agent
            
        except Exception as e:
            logger.error(f"Failed to create CrewAI agent: {e}")
            raise AgentCreationError(f"Failed to create CrewAI agent: {e}")
    
    async def _create_manager_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create a manager agent for crew coordination."""
        role = "Project Manager"
        goal = "Coordinate team members and ensure successful project completion"
        backstory = config.instructions or self._get_default_manager_backstory()
        
        if CREWAI_AVAILABLE:
            return CrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                verbose=True,
                allow_delegation=True,
                max_iter=config.max_iterations,
                memory=True
            )
        else:
            return MockCrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools
            )
    
    async def _create_expert_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create an expert agent for specialized tasks."""
        # Determine role based on capabilities
        role = self._determine_expert_role(config.capabilities)
        goal = f"Execute {role.lower()} tasks with high quality and expertise"
        backstory = config.instructions or self._get_default_expert_backstory(role)
        
        if CREWAI_AVAILABLE:
            return CrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                verbose=True,
                allow_delegation=False,
                max_iter=config.max_iterations,
                memory=True
            )
        else:
            return MockCrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools
            )
    
    async def _create_general_agent(self, config: AgentConfig, tools: List[Any]) -> Any:
        """Create a general-purpose agent."""
        role = config.name or "General Assistant"
        goal = "Assist with various tasks and provide helpful responses"
        backstory = config.instructions or "You are a helpful AI assistant."
        
        if CREWAI_AVAILABLE:
            return CrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                verbose=True,
                max_iter=config.max_iterations
            )
        else:
            return MockCrewAgent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools
            )
    
    def _determine_expert_role(self, capabilities: List[str]) -> str:
        """Determine expert role based on capabilities."""
        capability_roles = {
            "research": "Research Specialist",
            "writing": "Content Writer",
            "analysis": "Data Analyst",
            "translation": "Language Translator",
            "coding": "Software Developer",
            "design": "UI/UX Designer",
            "marketing": "Marketing Specialist",
            "finance": "Financial Analyst"
        }
        
        for capability in capabilities:
            if capability.lower() in capability_roles:
                return capability_roles[capability.lower()]
        
        return "Domain Expert"
    
    def _get_default_manager_backstory(self) -> str:
        """Get default backstory for manager agents."""
        return """You are an experienced project manager with a track record of successfully 
        coordinating diverse teams and delivering complex projects on time. You excel at breaking 
        down complex tasks, assigning work to the right team members, and ensuring quality outcomes. 
        You are collaborative, decisive, and always focused on achieving the best results for the team."""
    
    def _get_default_expert_backstory(self, role: str) -> str:
        """Get default backstory for expert agents."""
        return f"""You are a highly skilled {role} with years of experience in your field. 
        You are known for your attention to detail, deep expertise, and ability to deliver 
        high-quality work. You take pride in your craft and always strive for excellence 
        in everything you do. You work well with others and are always willing to share 
        your knowledge to help the team succeed."""
    
    async def _get_tools_for_agent(self, tool_ids: List[str]) -> List[Any]:
        """Get CrewAI tools for an agent."""
        tools = []
        
        # Mock tool creation - in real implementation, 
        # this would integrate with the tool manager
        for tool_id in tool_ids:
            try:
                if CREWAI_AVAILABLE:
                    # Would create actual CrewAI tools here
                    # tools.append(CustomCrewAITool(tool_id))
                    pass
                else:
                    # Mock tool
                    tools.append(MockBaseTool())
            except Exception as e:
                logger.warning(f"Failed to load tool {tool_id}: {e}")
        
        return tools
    
    async def _execute_framework_task(
        self, 
        framework_agent: Any,
        task: Task, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Execute a task using CrewAI agent."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create CrewAI task
            crew_task = self._create_crew_task(task, framework_agent)
            
            # Create crew with single agent for individual execution
            crew = self._create_single_agent_crew(framework_agent, crew_task)
            
            # Execute crew
            result = await self._execute_crew(crew)
            
            # Process result
            execution_result = await self._process_crew_result(
                result, task, context, start_time
            )
            
            return execution_result
            
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            return AgentExecutionResult(
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now(timezone.utc)
            )
    
    def _create_crew_task(self, task: Task, agent: Any) -> Any:
        """Create a CrewAI task from our task model."""
        description = f"{task.title}\n"
        if task.description:
            description += f"Description: {task.description}\n"
        if task.input_data:
            description += f"Input data: {task.input_data}\n"
        
        if CREWAI_AVAILABLE:
            return CrewTask(
                description=description,
                agent=agent,
                expected_output="A comprehensive response addressing the task requirements"
            )
        else:
            return MockCrewTask(
                description=description,
                agent=agent
            )
    
    def _create_single_agent_crew(self, agent: Any, task: Any) -> Any:
        """Create a crew with a single agent for individual task execution."""
        if CREWAI_AVAILABLE:
            return Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                memory=True
            )
        else:
            return MockCrew(
                agents=[agent],
                tasks=[task]
            )
    
    async def _execute_crew(self, crew: Any) -> Any:
        """Execute a CrewAI crew."""
        if CREWAI_AVAILABLE and hasattr(crew, 'kickoff'):
            # Use async kickoff if available
            if hasattr(crew.kickoff, '__call__'):
                return await crew.kickoff()
            else:
                return crew.kickoff()
        else:
            return await crew.kickoff()
    
    async def _process_crew_result(
        self, 
        crew_result: Any, 
        task: Task, 
        context: ExecutionContext,
        start_time: datetime
    ) -> AgentExecutionResult:
        """Process CrewAI execution result."""
        end_time = datetime.now(timezone.utc)
        execution_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract result data
        if isinstance(crew_result, str):
            result_data = {"response": crew_result}
        elif isinstance(crew_result, dict):
            result_data = crew_result
        else:
            result_data = {"response": str(crew_result)}
        
        return AgentExecutionResult(
            success=True,
            result=result_data,
            execution_time_ms=execution_time,
            started_at=start_time,
            completed_at=end_time,
            tool_calls=[],  # TODO: Extract tool calls from CrewAI result
            kb_queries=[],  # TODO: Extract KB queries from CrewAI result
            reasoning_steps=[],  # TODO: Extract reasoning steps
            intermediate_results=[]
        )
    
    # Tool calling implementation
    async def call_tool(
        self, 
        agent_id: str, 
        tool_id: str, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> ToolCallResult:
        """Call a tool through CrewAI agent."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get agent tools
            tools = self._agent_tools.get(agent_id, [])
            
            # Find and execute tool
            tool_result = None
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    if CREWAI_AVAILABLE:
                        tool_result = await tool.run(**parameters)
                    else:
                        tool_result = f"Mock tool {tool_name} result"
                    break
            
            if tool_result is None:
                raise Exception(f"Tool {tool_name} not found for agent {agent_id}")
            
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolCallResult(
                tool_name=tool_name,
                tool_id=tool_id,
                success=True,
                result={"output": tool_result},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolCallResult(
                tool_name=tool_name,
                tool_id=tool_id,
                success=False,
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
        """Query a knowledge base through CrewAI integration."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Mock implementation - would integrate with knowledge base manager
            result = {
                "results": [
                    {"content": f"CrewAI KB result for: {query}", "score": 0.85}
                ]
            }
            # Use parameters if provided
            if parameters:
                result["parameters_used"] = parameters
            
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return KnowledgeBaseQueryResult(
                kb_name=kb_name,
                kb_id=kb_id,
                query=query,
                success=True,
                results=result["results"],
                results_count=len(result["results"]),
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
                error_message=str(e),
                execution_time_ms=execution_time
            )
