"""
Agent factory for creating agents based on requirements.

This module provides utilities for dynamically creating agents
based on task requirements and available capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
import uuid as uuid_lib

from ..core.models import AgentConfig, Task
from ..core.enums import AgentType, FrameworkCapability
from ..registry import get_registry

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agents based on requirements."""
    
    def __init__(self):
        self._registry = get_registry()
        
        # Capability to agent type mapping
        self._capability_mappings = {
            "planning": AgentType.MANAGER,
            "coordination": AgentType.MANAGER,
            "delegation": AgentType.MANAGER,
            "management": AgentType.MANAGER,
            "research": AgentType.EXPERT,
            "writing": AgentType.EXPERT,
            "analysis": AgentType.EXPERT,
            "translation": AgentType.EXPERT,
            "coding": AgentType.EXPERT,
            "design": AgentType.EXPERT,
            "finance": AgentType.EXPERT,
            "marketing": AgentType.EXPERT,
        }
        
        # Default models for different frameworks
        self._default_models = {
            "google-adk": "gemini-2.0-flash",
            "langgraph": "gpt-4",
            "crewai": "claude-3-sonnet"
        }
    
    def create_agents_for_task(
        self,
        task: Task,
        framework: str,
        agent_count: Optional[int] = None,
        workflow_type: Optional[str] = None
    ) -> List[AgentConfig]:
        """Create agents automatically based on task requirements.
        
        Args:
            task: Task to analyze
            framework: AI framework to use
            agent_count: Number of agents to create (optional)
            workflow_type: Type of workflow (optional)
            
        Returns:
            List of agent configurations
        """
        logger.info(f"Creating agents for task: {task.title}")
        
        # Analyze task to determine required capabilities
        required_capabilities = self._analyze_task_requirements(task)
        
        # Determine workflow type if not specified
        if not workflow_type:
            workflow_type = self._suggest_workflow_type(required_capabilities, agent_count)
        
        # Create agents based on workflow type
        if workflow_type == "hierarchical":
            return self._create_hierarchical_agents(required_capabilities, framework)
        elif workflow_type == "parallel":
            return self._create_parallel_agents(required_capabilities, framework, agent_count)
        elif workflow_type == "sequential":
            return self._create_sequential_agents(required_capabilities, framework)
        else:
            return self._create_single_agent(required_capabilities, framework)
    
    def _analyze_task_requirements(self, task: Task) -> List[str]:
        """Analyze task to determine required capabilities."""
        capabilities = []
        
        # Analyze task title and description
        text = f"{task.title} {task.description or ''}".lower()
        
        # Keyword-based capability detection
        capability_keywords = {
            "research": ["research", "investigate", "study", "analyze", "explore"],
            "writing": ["write", "document", "report", "article", "content"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "review"],
            "translation": ["translate", "language", "localize"],
            "coding": ["code", "program", "develop", "software", "script"],
            "design": ["design", "ui", "ux", "interface", "visual"],
            "finance": ["financial", "budget", "cost", "revenue", "profit"],
            "marketing": ["market", "promote", "advertise", "campaign"],
            "planning": ["plan", "strategy", "coordinate", "manage"],
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in text for keyword in keywords):
                capabilities.append(capability)
        
        # Default capabilities if none detected
        if not capabilities:
            capabilities = ["reasoning", "general"]
        
        logger.info(f"Detected capabilities: {capabilities}")
        return capabilities
    
    def _suggest_workflow_type(
        self, 
        capabilities: List[str], 
        agent_count: Optional[int]
    ) -> str:
        """Suggest workflow type based on capabilities and agent count."""
        # If planning/coordination capabilities detected, suggest hierarchical
        if any(cap in ["planning", "coordination", "management"] for cap in capabilities):
            return "hierarchical"
        
        # If multiple diverse capabilities, suggest parallel
        if len(capabilities) > 2:
            return "parallel"
        
        # If agent count specified and > 1, suggest sequential
        if agent_count and agent_count > 1:
            return "sequential"
        
        # Default to single agent
        return "single"
    
    def _create_hierarchical_agents(
        self, 
        capabilities: List[str], 
        framework: str
    ) -> List[AgentConfig]:
        """Create agents for hierarchical workflow."""
        agents = []
        
        # Create manager agent
        manager_config = AgentConfig(
            agent_id=f"manager_{uuid_lib.uuid4().hex[:8]}",
            name="Task Manager",
            agent_type=AgentType.MANAGER,
            capabilities=["planning", "coordination", "delegation"],
            model=self._default_models.get(framework, "gemini-2.0-flash"),
            instructions="You are a manager agent responsible for coordinating and delegating tasks to expert agents."
        )
        agents.append(manager_config)
        
        # Create expert agents for each capability
        for capability in capabilities:
            if capability not in ["planning", "coordination", "management"]:
                expert_config = AgentConfig(
                    agent_id=f"expert_{capability}_{uuid_lib.uuid4().hex[:8]}",
                    name=f"{capability.title()} Expert",
                    agent_type=AgentType.EXPERT,
                    capabilities=[capability, "reasoning"],
                    model=self._default_models.get(framework, "gemini-2.0-flash"),
                    instructions=f"You are an expert agent specialized in {capability}."
                )
                agents.append(expert_config)
        
        return agents
    
    def _create_parallel_agents(
        self, 
        capabilities: List[str], 
        framework: str,
        agent_count: Optional[int]
    ) -> List[AgentConfig]:
        """Create agents for parallel workflow."""
        agents = []
        
        # Determine number of agents
        target_count = agent_count or min(len(capabilities), 4)
        
        # Create specialized agents
        for i, capability in enumerate(capabilities[:target_count]):
            agent_config = AgentConfig(
                agent_id=f"parallel_{capability}_{uuid_lib.uuid4().hex[:8]}",
                name=f"{capability.title()} Specialist",
                agent_type=AgentType.EXPERT,
                capabilities=[capability, "reasoning"],
                model=self._default_models.get(framework, "gemini-2.0-flash"),
                instructions=f"You are a specialist in {capability}. Work independently to provide your expert perspective."
            )
            agents.append(agent_config)
        
        return agents
    
    def _create_sequential_agents(
        self, 
        capabilities: List[str], 
        framework: str
    ) -> List[AgentConfig]:
        """Create agents for sequential workflow."""
        agents = []
        
        # Create pipeline stages based on logical flow
        pipeline_order = self._determine_pipeline_order(capabilities)
        
        for i, capability in enumerate(pipeline_order):
            agent_config = AgentConfig(
                agent_id=f"stage_{i+1}_{capability}_{uuid_lib.uuid4().hex[:8]}",
                name=f"Stage {i+1}: {capability.title()}",
                agent_type=AgentType.EXPERT,
                capabilities=[capability, "reasoning"],
                model=self._default_models.get(framework, "gemini-2.0-flash"),
                instructions=f"You are stage {i+1} in the pipeline, specializing in {capability}."
            )
            agents.append(agent_config)
        
        return agents
    
    def _create_single_agent(
        self, 
        capabilities: List[str], 
        framework: str
    ) -> List[AgentConfig]:
        """Create a single agent for simple tasks."""
        agent_config = AgentConfig(
            agent_id=f"single_{uuid_lib.uuid4().hex[:8]}",
            name="General Assistant",
            agent_type=AgentType.EXPERT,
            capabilities=capabilities + ["reasoning"],
            model=self._default_models.get(framework, "gemini-2.0-flash"),
            instructions="You are a general-purpose AI assistant capable of handling various tasks."
        )
        
        return [agent_config]
    
    def _determine_pipeline_order(self, capabilities: List[str]) -> List[str]:
        """Determine logical order for pipeline execution."""
        # Define typical pipeline order
        pipeline_priority = {
            "research": 1,
            "analysis": 2,
            "planning": 3,
            "writing": 4,
            "design": 5,
            "coding": 6,
            "translation": 7,
            "marketing": 8
        }
        
        # Sort capabilities by pipeline priority
        ordered_capabilities = sorted(
            capabilities,
            key=lambda cap: pipeline_priority.get(cap, 999)
        )
        
        return ordered_capabilities
    
    def create_agent_from_template(
        self,
        template_name: str,
        framework: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> AgentConfig:
        """Create an agent from a predefined template.
        
        Args:
            template_name: Name of the template to use
            framework: AI framework to use
            customizations: Optional customizations to apply
            
        Returns:
            Agent configuration
        """
        templates = {
            "researcher": {
                "name": "Research Specialist",
                "agent_type": AgentType.EXPERT,
                "capabilities": ["research", "analysis", "reasoning"],
                "instructions": "You are a research specialist focused on gathering and analyzing information."
            },
            "writer": {
                "name": "Content Writer",
                "agent_type": AgentType.EXPERT,
                "capabilities": ["writing", "editing", "reasoning"],
                "instructions": "You are a content writer specialized in creating clear, engaging content."
            },
            "analyst": {
                "name": "Data Analyst",
                "agent_type": AgentType.EXPERT,
                "capabilities": ["analysis", "data_processing", "reasoning"],
                "instructions": "You are a data analyst focused on extracting insights from data."
            },
            "manager": {
                "name": "Project Manager",
                "agent_type": AgentType.MANAGER,
                "capabilities": ["planning", "coordination", "delegation"],
                "instructions": "You are a project manager responsible for coordinating team efforts."
            }
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = templates[template_name].copy()
        
        # Apply customizations
        if customizations:
            template.update(customizations)
        
        # Create agent config
        agent_config = AgentConfig(
            agent_id=f"{template_name}_{uuid_lib.uuid4().hex[:8]}",
            model=self._default_models.get(framework, "gemini-2.0-flash"),
            **template
        )
        
        return agent_config
    
    def validate_agent_compatibility(
        self, 
        agent_config: AgentConfig, 
        framework: str
    ) -> bool:
        """Validate that an agent is compatible with a framework.
        
        Args:
            agent_config: Agent configuration to validate
            framework: Target framework
            
        Returns:
            True if compatible, False otherwise
        """
        adapter = self._registry.get_adapter(framework)
        if not adapter:
            return False
        
        # Check if framework supports required capabilities
        required_capabilities = []
        if agent_config.tools:
            required_capabilities.append(FrameworkCapability.TOOL_CALLING)
        if agent_config.knowledge_bases:
            required_capabilities.append(FrameworkCapability.KNOWLEDGE_BASE)
        
        for capability in required_capabilities:
            if not adapter.supports_capability(capability):
                logger.warning(f"Framework {framework} does not support {capability}")
                return False
        
        return True
