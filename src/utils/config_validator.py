"""
Configuration validator for multi-agent system.

This module provides validation utilities for agent configurations,
workflow configurations, and system settings.
"""

import logging
from typing import Dict, Any, List, Optional, Set
import re

from ..core.models import AgentConfig, MultiAgentConfig, WorkflowConfig, Task
from ..core.enums import AgentType, WorkflowType, ExecutionStrategy, FrameworkCapability
from ..core.exceptions import (
    MultiAgentError, AgentError, WorkflowError, TaskError
)
from ..registry import get_registry

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validator for multi-agent system configurations."""
    
    def __init__(self):
        self._registry = get_registry()
        
        # Validation rules
        self._agent_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        self._max_agent_name_length = 100
        self._max_description_length = 500
        self._max_instructions_length = 2000
        self._min_temperature = 0.0
        self._max_temperature = 2.0
        self._max_iterations = 100
        self._max_timeout_seconds = 3600  # 1 hour
    
    def validate_agent_config(self, config: AgentConfig) -> List[str]:
        """Validate agent configuration.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate agent ID
        if not config.agent_id:
            errors.append("Agent ID is required")
        elif not self._agent_id_pattern.match(config.agent_id):
            errors.append("Agent ID must contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate name
        if not config.name or not config.name.strip():
            errors.append("Agent name is required")
        elif len(config.name) > self._max_agent_name_length:
            errors.append(f"Agent name must be {self._max_agent_name_length} characters or less")
        
        # Validate description
        if config.description and len(config.description) > self._max_description_length:
            errors.append(f"Agent description must be {self._max_description_length} characters or less")
        
        # Validate instructions
        if config.instructions and len(config.instructions) > self._max_instructions_length:
            errors.append(f"Agent instructions must be {self._max_instructions_length} characters or less")
        
        # Validate model
        if not config.model or not config.model.strip():
            errors.append("Agent model is required")
        
        # Validate temperature
        if not (self._min_temperature <= config.temperature <= self._max_temperature):
            errors.append(f"Temperature must be between {self._min_temperature} and {self._max_temperature}")
        
        # Validate max iterations
        if config.max_iterations <= 0 or config.max_iterations > self._max_iterations:
            errors.append(f"Max iterations must be between 1 and {self._max_iterations}")
        
        # Validate timeout
        if config.timeout_seconds is not None:
            if config.timeout_seconds <= 0 or config.timeout_seconds > self._max_timeout_seconds:
                errors.append(f"Timeout must be between 1 and {self._max_timeout_seconds} seconds")
        
        # Validate capabilities
        if config.capabilities:
            invalid_capabilities = self._validate_capabilities(config.capabilities)
            if invalid_capabilities:
                errors.append(f"Invalid capabilities: {', '.join(invalid_capabilities)}")
        
        return errors
    
    def validate_workflow_config(self, config: WorkflowConfig) -> List[str]:
        """Validate workflow configuration.
        
        Args:
            config: Workflow configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate workflow type
        if config.workflow_type not in WorkflowType:
            errors.append(f"Invalid workflow type: {config.workflow_type}")
        
        # Validate execution strategy
        if config.execution_strategy not in ExecutionStrategy:
            errors.append(f"Invalid execution strategy: {config.execution_strategy}")
        
        # Validate max concurrent agents
        if config.max_concurrent_agents <= 0:
            errors.append("Max concurrent agents must be positive")
        
        # Validate timeout
        if config.timeout_seconds is not None:
            if config.timeout_seconds <= 0 or config.timeout_seconds > self._max_timeout_seconds:
                errors.append(f"Workflow timeout must be between 1 and {self._max_timeout_seconds} seconds")
        
        # Validate hierarchical workflow specific settings
        if config.workflow_type == WorkflowType.HIERARCHICAL:
            if config.manager_agent_id and not config.manager_agent_id.strip():
                errors.append("Manager agent ID cannot be empty if specified")
            
            if config.expert_agent_ids:
                for agent_id in config.expert_agent_ids:
                    if not agent_id or not agent_id.strip():
                        errors.append("Expert agent IDs cannot be empty")
        
        # Validate sequential workflow specific settings
        if config.workflow_type == WorkflowType.SEQUENTIAL:
            if config.pipeline_stages:
                stage_ids = set()
                for i, stage in enumerate(config.pipeline_stages):
                    if not isinstance(stage, dict):
                        errors.append(f"Pipeline stage {i} must be a dictionary")
                        continue
                    
                    if "stage" not in stage:
                        errors.append(f"Pipeline stage {i} missing 'stage' field")
                    elif stage["stage"] in stage_ids:
                        errors.append(f"Duplicate stage ID: {stage['stage']}")
                    else:
                        stage_ids.add(stage["stage"])
                    
                    if "agent_id" not in stage:
                        errors.append(f"Pipeline stage {i} missing 'agent_id' field")
        
        return errors
    
    def validate_multi_agent_config(self, config: MultiAgentConfig) -> List[str]:
        """Validate multi-agent configuration.
        
        Args:
            config: Multi-agent configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate framework
        if not config.framework:
            errors.append("Framework is required")
        elif not self._registry.is_registered(config.framework):
            errors.append(f"Framework not registered: {config.framework}")
        
        # Validate fallback frameworks
        if config.fallback_frameworks:
            for framework in config.fallback_frameworks:
                if not self._registry.is_registered(framework):
                    errors.append(f"Fallback framework not registered: {framework}")
        
        # Validate agents
        if not config.agents:
            errors.append("At least one agent is required")
        else:
            agent_ids = set()
            for i, agent_config in enumerate(config.agents):
                # Validate individual agent config
                agent_errors = self.validate_agent_config(agent_config)
                for error in agent_errors:
                    errors.append(f"Agent {i}: {error}")
                
                # Check for duplicate agent IDs
                if agent_config.agent_id in agent_ids:
                    errors.append(f"Duplicate agent ID: {agent_config.agent_id}")
                else:
                    agent_ids.add(agent_config.agent_id)
        
        # Validate workflow config
        if config.workflow:
            workflow_errors = self.validate_workflow_config(config.workflow)
            for error in workflow_errors:
                errors.append(f"Workflow: {error}")
            
            # Cross-validate workflow and agents
            workflow_agent_errors = self._validate_workflow_agent_compatibility(
                config.workflow, config.agents
            )
            errors.extend(workflow_agent_errors)
        else:
            errors.append("Workflow configuration is required")
        
        return errors
    
    def validate_task(self, task: Task) -> List[str]:
        """Validate task configuration.
        
        Args:
            task: Task to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate task ID
        if not task.task_id:
            errors.append("Task ID is required")
        
        # Validate title
        if not task.title or not task.title.strip():
            errors.append("Task title is required")
        elif len(task.title) > 200:
            errors.append("Task title must be 200 characters or less")
        
        # Validate description
        if task.description and len(task.description) > 1000:
            errors.append("Task description must be 1000 characters or less")
        
        # Validate timeout
        if task.timeout_seconds is not None:
            if task.timeout_seconds <= 0 or task.timeout_seconds > self._max_timeout_seconds:
                errors.append(f"Task timeout must be between 1 and {self._max_timeout_seconds} seconds")
        
        # Validate retry settings
        if task.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        if task.retry_count < 0:
            errors.append("Retry count cannot be negative")
        
        if task.retry_count > task.max_retries:
            errors.append("Retry count cannot exceed max retries")
        
        return errors
    
    def validate_framework_compatibility(
        self, 
        config: MultiAgentConfig
    ) -> List[str]:
        """Validate framework compatibility with agent requirements.
        
        Args:
            config: Multi-agent configuration to validate
            
        Returns:
            List of compatibility errors (empty if compatible)
        """
        errors = []
        
        adapter = self._registry.get_adapter(config.framework)
        if not adapter:
            errors.append(f"Framework adapter not found: {config.framework}")
            return errors
        
        # Check framework capabilities against agent requirements
        for agent_config in config.agents:
            # Check tool calling capability
            if agent_config.tools and not adapter.supports_capability(FrameworkCapability.TOOL_CALLING):
                errors.append(
                    f"Agent {agent_config.agent_id} requires tool calling, "
                    f"but framework {config.framework} does not support it"
                )
            
            # Check knowledge base capability
            if agent_config.knowledge_bases and not adapter.supports_capability(FrameworkCapability.KNOWLEDGE_BASE):
                errors.append(
                    f"Agent {agent_config.agent_id} requires knowledge base access, "
                    f"but framework {config.framework} does not support it"
                )
            
            # Check multi-agent capability for hierarchical workflows
            if (config.workflow.workflow_type == WorkflowType.HIERARCHICAL and 
                not adapter.supports_capability(FrameworkCapability.MULTI_AGENT)):
                errors.append(
                    f"Hierarchical workflow requires multi-agent capability, "
                    f"but framework {config.framework} does not support it"
                )
        
        return errors
    
    def _validate_capabilities(self, capabilities: List[str]) -> List[str]:
        """Validate agent capabilities.
        
        Args:
            capabilities: List of capabilities to validate
            
        Returns:
            List of invalid capabilities
        """
        valid_capabilities = {
            "reasoning", "planning", "coordination", "delegation", "management",
            "research", "writing", "analysis", "translation", "coding", "design",
            "finance", "marketing", "tool_calling", "knowledge_retrieval",
            "data_processing", "visualization", "communication", "problem_solving"
        }
        
        invalid_capabilities = []
        for capability in capabilities:
            if capability not in valid_capabilities:
                invalid_capabilities.append(capability)
        
        return invalid_capabilities
    
    def _validate_workflow_agent_compatibility(
        self, 
        workflow_config: WorkflowConfig, 
        agent_configs: List[AgentConfig]
    ) -> List[str]:
        """Validate compatibility between workflow and agents.
        
        Args:
            workflow_config: Workflow configuration
            agent_configs: List of agent configurations
            
        Returns:
            List of compatibility errors
        """
        errors = []
        
        agent_ids = {config.agent_id for config in agent_configs}
        agent_types = {config.agent_id: config.agent_type for config in agent_configs}
        
        # Validate hierarchical workflow requirements
        if workflow_config.workflow_type == WorkflowType.HIERARCHICAL:
            # Check for manager agent
            managers = [config for config in agent_configs if config.agent_type == AgentType.MANAGER]
            if not managers:
                errors.append("Hierarchical workflow requires at least one manager agent")
            
            # Check for expert agents
            experts = [config for config in agent_configs if config.agent_type == AgentType.EXPERT]
            if not experts:
                errors.append("Hierarchical workflow requires at least one expert agent")
            
            # Validate manager agent ID if specified
            if workflow_config.manager_agent_id:
                if workflow_config.manager_agent_id not in agent_ids:
                    errors.append(f"Specified manager agent not found: {workflow_config.manager_agent_id}")
                elif agent_types[workflow_config.manager_agent_id] != AgentType.MANAGER:
                    errors.append(f"Specified manager agent is not of type MANAGER: {workflow_config.manager_agent_id}")
            
            # Validate expert agent IDs if specified
            if workflow_config.expert_agent_ids:
                for expert_id in workflow_config.expert_agent_ids:
                    if expert_id not in agent_ids:
                        errors.append(f"Specified expert agent not found: {expert_id}")
                    elif agent_types[expert_id] != AgentType.EXPERT:
                        errors.append(f"Specified expert agent is not of type EXPERT: {expert_id}")
        
        # Validate sequential workflow requirements
        if workflow_config.workflow_type == WorkflowType.SEQUENTIAL:
            if workflow_config.pipeline_stages:
                for stage in workflow_config.pipeline_stages:
                    if isinstance(stage, dict) and "agent_id" in stage:
                        agent_id = stage["agent_id"]
                        if agent_id not in agent_ids:
                            errors.append(f"Pipeline stage references unknown agent: {agent_id}")
        
        return errors
    
    def validate_and_raise(self, config: MultiAgentConfig) -> None:
        """Validate configuration and raise exception if invalid.
        
        Args:
            config: Configuration to validate
            
        Raises:
            MultiAgentError: If configuration is invalid
        """
        errors = self.validate_multi_agent_config(config)
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise MultiAgentError(error_message)
