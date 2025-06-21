"""
Unit tests for adapter registry.

This module tests the AdapterRegistry class and its functionality
for managing AI framework adapters.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.registry.adapter_registry import AdapterRegistry, get_registry
from src.core.interfaces import BaseFrameworkAdapter
from src.core.enums import FrameworkCapability
from src.core.exceptions import FrameworkNotFoundError, FrameworkInitializationError


class MockAdapter(BaseFrameworkAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self._capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.TOOL_CALLING
        ]
    
    async def initialize(self):
        self._initialized = True
    
    async def cleanup(self):
        self._initialized = False
    
    async def create_agent(self, config):
        return Mock()
    
    async def delete_agent(self, agent_id):
        return True
    
    async def execute_task(self, agent_id, task, context):
        return Mock()
    
    async def call_tool(self, agent_id, tool_id, tool_name, parameters, context):
        return Mock()
    
    async def query_knowledge_base(self, agent_id, kb_id, kb_name, query, parameters, context):
        return Mock()


class TestAdapterRegistry:
    """Test cases for AdapterRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AdapterRegistry()
        
        assert len(registry.list_adapters()) == 0
        assert registry.get_default_adapter() is None
        assert not registry.is_registered("nonexistent")

    def test_adapter_registration(self):
        """Test registering adapters."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        
        # Register adapter
        registry.register("test-framework", adapter)
        
        assert registry.is_registered("test-framework")
        assert len(registry.list_adapters()) == 1
        assert registry.get_adapter("test-framework") == adapter
        assert registry.get_default_adapter() == adapter

    def test_adapter_registration_with_metadata(self):
        """Test registering adapters with metadata."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        metadata = {"description": "Test adapter", "version": "1.0.0"}
        
        registry.register("test-framework", adapter, metadata=metadata)
        
        info = registry.get_adapter_info("test-framework")
        assert info["metadata"] == metadata

    def test_multiple_adapter_registration(self):
        """Test registering multiple adapters."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2, is_default=True)
        
        assert len(registry.list_adapters()) == 2
        assert registry.get_default_adapter() == adapter2
        assert registry.is_registered("framework1")
        assert registry.is_registered("framework2")

    def test_adapter_override(self):
        """Test overriding existing adapter."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("test-framework", "1.0.0")
        adapter2 = MockAdapter("test-framework", "2.0.0")
        
        registry.register("test-framework", adapter1)
        registry.register("test-framework", adapter2)  # Override
        
        assert registry.get_adapter("test-framework") == adapter2
        assert len(registry.list_adapters()) == 1

    def test_adapter_unregistration(self):
        """Test unregistering adapters."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        
        registry.register("test-framework", adapter)
        assert registry.is_registered("test-framework")
        
        # Unregister
        result = registry.unregister("test-framework")
        assert result is True
        assert not registry.is_registered("test-framework")
        assert len(registry.list_adapters()) == 0
        
        # Unregister nonexistent
        result = registry.unregister("nonexistent")
        assert result is False

    def test_default_adapter_management(self):
        """Test default adapter management."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        # First registered becomes default
        registry.register("framework1", adapter1)
        assert registry.get_default_adapter() == adapter1
        
        # Explicit default setting
        registry.register("framework2", adapter2)
        registry.set_default_adapter("framework2")
        assert registry.get_default_adapter() == adapter2
        
        # Setting nonexistent as default fails
        result = registry.set_default_adapter("nonexistent")
        assert result is False
        assert registry.get_default_adapter() == adapter2

    def test_capability_based_search(self):
        """Test finding adapters by capability."""
        registry = AdapterRegistry()
        
        # Create adapters with different capabilities
        adapter1 = MockAdapter("framework1")
        adapter1._capabilities = [FrameworkCapability.SINGLE_AGENT]
        
        adapter2 = MockAdapter("framework2")
        adapter2._capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.MULTI_AGENT
        ]
        
        adapter3 = MockAdapter("framework3")
        adapter3._capabilities = [FrameworkCapability.STREAMING]
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2)
        registry.register("framework3", adapter3)
        
        # Search by capability
        single_agent_adapters = registry.find_adapters_by_capability(
            FrameworkCapability.SINGLE_AGENT
        )
        assert set(single_agent_adapters) == {"framework1", "framework2"}
        
        multi_agent_adapters = registry.find_adapters_by_capability(
            FrameworkCapability.MULTI_AGENT
        )
        assert multi_agent_adapters == ["framework2"]
        
        streaming_adapters = registry.find_adapters_by_capability(
            FrameworkCapability.STREAMING
        )
        assert streaming_adapters == ["framework3"]

    def test_adapter_info_retrieval(self):
        """Test retrieving adapter information."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework", "1.2.3")
        metadata = {"author": "Test Author"}
        
        registry.register("test-framework", adapter, metadata=metadata)
        
        info = registry.get_adapter_info("test-framework")
        assert info["name"] == "test-framework"
        assert info["framework_name"] == "test-framework"
        assert info["version"] == "1.2.3"
        assert info["capabilities"] == adapter.capabilities
        assert info["is_initialized"] is False
        assert info["is_default"] is True
        assert info["metadata"] == metadata
        
        # Nonexistent adapter
        assert registry.get_adapter_info("nonexistent") is None

    def test_all_adapter_info_retrieval(self):
        """Test retrieving all adapter information."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2)
        
        all_info = registry.get_all_adapter_info()
        assert len(all_info) == 2
        
        names = [info["name"] for info in all_info]
        assert set(names) == {"framework1", "framework2"}

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test adapter initialization."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        
        registry.register("test-framework", adapter)
        assert not adapter.is_initialized
        
        # Initialize adapter
        result = await registry.initialize_adapter("test-framework")
        assert result is True
        assert adapter.is_initialized
        
        # Initialize already initialized adapter
        result = await registry.initialize_adapter("test-framework")
        assert result is True

    @pytest.mark.asyncio
    async def test_adapter_initialization_failure(self):
        """Test adapter initialization failure."""
        registry = AdapterRegistry()
        adapter = Mock(spec=BaseFrameworkAdapter)
        adapter.framework_name = "test-framework"
        adapter.version = "1.0.0"
        adapter.is_initialized = False
        adapter.initialize = AsyncMock(side_effect=Exception("Init failed"))
        
        registry.register("test-framework", adapter)
        
        with pytest.raises(FrameworkInitializationError):
            await registry.initialize_adapter("test-framework")

    @pytest.mark.asyncio
    async def test_nonexistent_adapter_initialization(self):
        """Test initializing nonexistent adapter."""
        registry = AdapterRegistry()
        
        with pytest.raises(FrameworkNotFoundError):
            await registry.initialize_adapter("nonexistent")

    @pytest.mark.asyncio
    async def test_initialize_all_adapters(self):
        """Test initializing all adapters."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2)
        
        results = await registry.initialize_all()
        
        assert len(results) == 2
        assert results["framework1"] is True
        assert results["framework2"] is True
        assert adapter1.is_initialized
        assert adapter2.is_initialized

    @pytest.mark.asyncio
    async def test_adapter_cleanup(self):
        """Test adapter cleanup."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        
        registry.register("test-framework", adapter)
        await registry.initialize_adapter("test-framework")
        assert adapter.is_initialized
        
        # Cleanup adapter
        result = await registry.cleanup_adapter("test-framework")
        assert result is True
        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_cleanup_all_adapters(self):
        """Test cleaning up all adapters."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2)
        await registry.initialize_all()
        
        results = await registry.cleanup_all()
        
        assert len(results) == 2
        assert results["framework1"] is True
        assert results["framework2"] is True
        assert not adapter1.is_initialized
        assert not adapter2.is_initialized

    def test_health_status(self):
        """Test getting health status."""
        registry = AdapterRegistry()
        adapter1 = MockAdapter("framework1")
        adapter2 = MockAdapter("framework2")
        
        registry.register("framework1", adapter1)
        registry.register("framework2", adapter2)
        
        status = registry.get_health_status()
        
        assert status["total_adapters"] == 2
        assert status["initialized_adapters"] == 0
        assert status["default_adapter"] == "framework1"
        assert "adapters" in status
        assert len(status["adapters"]) == 2

    def test_case_insensitive_names(self):
        """Test case insensitive adapter names."""
        registry = AdapterRegistry()
        adapter = MockAdapter("Test-Framework")
        
        registry.register("Test-Framework", adapter)
        
        # Should work with different cases
        assert registry.is_registered("test-framework")
        assert registry.is_registered("TEST-FRAMEWORK")
        assert registry.get_adapter("test-framework") == adapter
        assert registry.get_adapter("TEST-FRAMEWORK") == adapter

    def test_validation_errors(self):
        """Test validation errors."""
        registry = AdapterRegistry()
        
        # Empty name
        with pytest.raises(ValueError):
            registry.register("", Mock())
        
        # Invalid adapter type
        with pytest.raises(ValueError):
            registry.register("test", "not_an_adapter")

    @pytest.mark.asyncio
    async def test_managed_lifecycle(self):
        """Test managed lifecycle context manager."""
        registry = AdapterRegistry()
        adapter = MockAdapter("test-framework")
        registry.register("test-framework", adapter)
        
        async with registry.managed_lifecycle() as managed_registry:
            assert managed_registry == registry
            assert adapter.is_initialized
        
        # Should be cleaned up after context
        assert not adapter.is_initialized


class TestGlobalRegistry:
    """Test cases for global registry singleton."""

    def test_global_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        assert registry1 is registry2
        assert isinstance(registry1, AdapterRegistry)
