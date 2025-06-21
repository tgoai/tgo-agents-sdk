"""
Test configuration and utilities.

This module provides test configuration, utilities, and helper functions
for the test suite.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class TestConfig:
    """Test configuration class."""
    
    def __init__(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.temp_dir = Path(__file__).parent / "temp"
        self.log_level = logging.DEBUG if self.is_debug_mode() else logging.INFO
        
        # Create directories if they don't exist
        self.test_data_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def is_debug_mode() -> bool:
        """Check if running in debug mode."""
        return os.getenv("TEST_DEBUG", "false").lower() == "true"
    
    @staticmethod
    def is_integration_mode() -> bool:
        """Check if running integration tests."""
        return os.getenv("TEST_INTEGRATION", "false").lower() == "true"
    
    @staticmethod
    def get_test_timeout() -> int:
        """Get test timeout in seconds."""
        return int(os.getenv("TEST_TIMEOUT", "30"))
    
    @staticmethod
    def get_mock_responses() -> bool:
        """Check if should use mock responses."""
        return os.getenv("TEST_MOCK_RESPONSES", "true").lower() == "true"


class TestDataManager:
    """Manager for test data."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self._test_data: Dict[str, Any] = {}
    
    def load_test_data(self, filename: str) -> Dict[str, Any]:
        """Load test data from file."""
        file_path = self.config.test_data_dir / filename
        
        if not file_path.exists():
            return {}
        
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_test_data(self, filename: str, data: Dict[str, Any]) -> None:
        """Save test data to file."""
        file_path = self.config.test_data_dir / filename
        
        import json
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_sample_task_data(self) -> Dict[str, Any]:
        """Get sample task data."""
        return {
            "title": "Sample Test Task",
            "description": "A sample task for testing purposes",
            "input_data": {
                "test_parameter": "test_value",
                "numeric_parameter": 42,
                "list_parameter": ["item1", "item2", "item3"]
            }
        }
    
    def get_sample_agent_data(self) -> Dict[str, Any]:
        """Get sample agent data."""
        return {
            "name": "Sample Test Agent",
            "agent_type": "expert",
            "model": "gemini-2.0-flash",
            "capabilities": ["reasoning", "analysis", "tool_calling"],
            "tools": ["search", "calculator", "file_reader"],
            "knowledge_bases": ["general_kb", "technical_kb"]
        }
    
    def get_sample_workflow_data(self) -> Dict[str, Any]:
        """Get sample workflow data."""
        return {
            "workflow_type": "hierarchical",
            "execution_strategy": "fail_fast",
            "max_concurrent_agents": 3,
            "timeout_seconds": 300
        }


class TestLogger:
    """Test logging utilities."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup test logging."""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.temp_dir / "test.log")
            ]
        )
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger for testing."""
        return logging.getLogger(f"test.{name}")


class MockResponseManager:
    """Manager for mock responses."""
    
    def __init__(self):
        self._responses: Dict[str, Any] = {}
        self._load_default_responses()
    
    def _load_default_responses(self):
        """Load default mock responses."""
        self._responses.update({
            "agent_creation_success": {
                "success": True,
                "agent_id": "mock_agent_001",
                "status": "created"
            },
            "task_execution_success": {
                "success": True,
                "result": {
                    "response": "Mock task execution completed successfully",
                    "execution_time": 1000,
                    "tokens_used": 150
                }
            },
            "tool_call_success": {
                "success": True,
                "result": {
                    "tool_output": "Mock tool execution result",
                    "execution_time": 500
                }
            },
            "knowledge_base_query_success": {
                "success": True,
                "results": [
                    {
                        "content": "Mock knowledge base result 1",
                        "score": 0.95,
                        "source": "mock_document_1"
                    },
                    {
                        "content": "Mock knowledge base result 2", 
                        "score": 0.87,
                        "source": "mock_document_2"
                    }
                ],
                "total_results": 2
            },
            "workflow_execution_success": {
                "success": True,
                "workflow_type": "hierarchical",
                "agents_used": ["manager_001", "expert_001", "expert_002"],
                "execution_time": 5000,
                "result": {
                    "final_response": "Mock workflow execution completed",
                    "manager_analysis": "Task decomposed successfully",
                    "expert_contributions": [
                        "Expert 1 analysis complete",
                        "Expert 2 research complete"
                    ]
                }
            }
        })
    
    def get_response(self, response_type: str) -> Optional[Dict[str, Any]]:
        """Get a mock response by type."""
        return self._responses.get(response_type)
    
    def add_response(self, response_type: str, response: Dict[str, Any]):
        """Add a custom mock response."""
        self._responses[response_type] = response
    
    def get_error_response(self, error_message: str = "Mock error") -> Dict[str, Any]:
        """Get a mock error response."""
        return {
            "success": False,
            "error_message": error_message,
            "error_code": "MOCK_ERROR",
            "timestamp": "2024-01-01T00:00:00Z"
        }


# Global test configuration instances
test_config = TestConfig()
test_data_manager = TestDataManager(test_config)
test_logger = TestLogger(test_config)
mock_response_manager = MockResponseManager()


def setup_test_environment():
    """Setup test environment."""
    # Create necessary directories
    test_config.temp_dir.mkdir(exist_ok=True)
    test_config.test_data_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = test_logger.get_logger("setup")
    logger.info("Test environment setup complete")


def cleanup_test_environment():
    """Cleanup test environment."""
    import shutil
    
    # Clean up temporary files
    if test_config.temp_dir.exists():
        shutil.rmtree(test_config.temp_dir)
    
    logger = test_logger.get_logger("cleanup")
    logger.info("Test environment cleanup complete")


# Test markers for pytest
UNIT_TEST_MARKER = "unit"
INTEGRATION_TEST_MARKER = "integration"
SLOW_TEST_MARKER = "slow"
MOCK_TEST_MARKER = "mock"
