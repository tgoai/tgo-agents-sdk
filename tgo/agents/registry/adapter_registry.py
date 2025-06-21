"""
Adapter registry for managing AI framework adapters.

This module provides a centralized registry for discovering, registering,
and managing different AI framework adapters.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from contextlib import asynccontextmanager
import asyncio
from threading import Lock

from ..core.interfaces import BaseFrameworkAdapter
from ..core.exceptions import (
     FrameworkNotFoundError, FrameworkInitializationError
)
from ..core.enums import FrameworkCapability

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for managing AI framework adapters.
    
    This class provides a centralized way to register, discover, and manage
    different AI framework adapters. It supports:
    - Dynamic adapter registration and unregistration
    - Capability-based adapter discovery
    - Automatic initialization and cleanup
    - Thread-safe operations
    - Health monitoring
    """
    
    def __init__(self):
        self._adapters: Dict[str, BaseFrameworkAdapter] = {}
        self._adapter_metadata: Dict[str, Dict[str, Any]] = {}
        self._default_adapter: Optional[str] = None
        self._lock = Lock()
        self._initialized_adapters: Set[str] = set()
        
    def register(
        self, 
        name: str, 
        adapter: BaseFrameworkAdapter,
        is_default: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a framework adapter.
        
        Args:
            name: Unique name for the adapter
            adapter: The adapter instance
            is_default: Whether this should be the default adapter
            metadata: Additional metadata about the adapter
            
        Raises:
            ValueError: If name is empty or adapter is invalid
            FrameworkError: If registration fails
        """
        if not name or not name.strip():
            raise ValueError("Adapter name cannot be empty")
        
        # Note: isinstance check removed as it's always true for BaseFrameworkAdapter
        
        name = name.strip().lower()

        with self._lock:
            
            if name in self._adapters:
                logger.warning(f"Overriding existing adapter: {name}")
            
            self._adapters[name] = adapter
            self._adapter_metadata[name] = metadata or {}
            
            if is_default or self._default_adapter is None:
                self._default_adapter = name
            logger.info(f"Registered adapter: {name} (default: {is_default})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a framework adapter.
        
        Args:
            name: Name of the adapter to unregister
            
        Returns:
            True if adapter was unregistered, False if not found
        """
        if not name:
            return False
            
        name = name.strip().lower()
        
        with self._lock:
            if name not in self._adapters:
                return False
            
            # Clean up if it was initialized
            if name in self._initialized_adapters:
                try:
                    asyncio.create_task(self._adapters[name].cleanup())
                    self._initialized_adapters.discard(name)
                except Exception as e:
                    logger.error(f"Error cleaning up adapter {name}: {e}")
            
            del self._adapters[name]
            self._adapter_metadata.pop(name, None)
            
            # Update default if necessary
            if self._default_adapter == name:
                self._default_adapter = next(iter(self._adapters.keys()), None)
            
            logger.info(f"Unregistered adapter: {name}")
            return True
    
    def get_adapter(self, name: str) -> Optional[BaseFrameworkAdapter]:
        """Get a registered adapter by name.
        
        Args:
            name: Name of the adapter
            
        Returns:
            The adapter instance or None if not found
        """
        if not name:
            return None
            
        name = name.strip().lower()
        return self._adapters.get(name)
    
    def get_default_adapter(self) -> Optional[BaseFrameworkAdapter]:
        """Get the default adapter.
        
        Returns:
            The default adapter instance or None if no adapters registered
        """
        if not self._default_adapter:
            return None
        return self._adapters.get(self._default_adapter)
    
    def set_default_adapter(self, name: str) -> bool:
        """Set the default adapter.
        
        Args:
            name: Name of the adapter to set as default
            
        Returns:
            True if successful, False if adapter not found
        """
        if not name:
            return False
            
        name = name.strip().lower()
        
        with self._lock:
            if name not in self._adapters:
                return False
            
            self._default_adapter = name
            logger.info(f"Set default adapter: {name}")
            return True
    
    def list_adapters(self) -> List[str]:
        """List all registered adapter names.
        
        Returns:
            List of adapter names
        """
        return list(self._adapters.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if an adapter is registered.
        
        Args:
            name: Name of the adapter
            
        Returns:
            True if registered, False otherwise
        """
        if not name:
            return False
        return name.strip().lower() in self._adapters
    
    def find_adapters_by_capability(
        self, 
        capability: FrameworkCapability
    ) -> List[str]:
        """Find adapters that support a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of adapter names that support the capability
        """
        matching_adapters: List[str] = []
        
        for name, adapter in self._adapters.items():
            if adapter.supports_capability(capability):
                matching_adapters.append(name)
        
        return matching_adapters
    
    def get_adapter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an adapter.
        
        Args:
            name: Name of the adapter
            
        Returns:
            Dictionary with adapter information or None if not found
        """
        if not name:
            return None
            
        name = name.strip().lower()
        adapter = self._adapters.get(name)
        
        if not adapter:
            return None
        
        return {
            "name": name,
            "framework_name": adapter.framework_name,
            "version": adapter.version,
            "capabilities": adapter.capabilities,
            "is_initialized": adapter.is_initialized,
            "is_default": name == self._default_adapter,
            "metadata": self._adapter_metadata.get(name, {})
        }
    
    def get_all_adapter_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered adapters.
        
        Returns:
            List of dictionaries with adapter information
        """
        result: List[Dict[str, Any]] = []
        for name in self._adapters.keys():
            info = self.get_adapter_info(name)
            if info is not None:
                result.append(info)
        return result
    
    async def initialize_adapter(self, name: str) -> bool:
        """Initialize a specific adapter.
        
        Args:
            name: Name of the adapter to initialize
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            FrameworkNotFoundError: If adapter not found
            FrameworkInitializationError: If initialization fails
        """
        if not name:
            raise ValueError("Adapter name cannot be empty")
            
        name = name.strip().lower()
        adapter = self._adapters.get(name)
        
        if not adapter:
            raise FrameworkNotFoundError(f"Adapter not found: {name}")
        
        if adapter.is_initialized:
            logger.debug(f"Adapter already initialized: {name}")
            return True
        
        try:
            await adapter.initialize()
            self._initialized_adapters.add(name)
            logger.info(f"Initialized adapter: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize adapter {name}: {e}")
            raise FrameworkInitializationError(
                f"Failed to initialize adapter {name}: {e}",
                framework_name=name
            )
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered adapters.
        
        Returns:
            Dictionary mapping adapter names to initialization success status
        """
        results: Dict[str, bool] = {}
        
        for name in self._adapters.keys():
            try:
                success = await self.initialize_adapter(name)
                results[name] = success
            except Exception as e:
                logger.error(f"Failed to initialize adapter {name}: {e}")
                results[name] = False
        
        return results
    
    async def cleanup_adapter(self, name: str) -> bool:
        """Clean up a specific adapter.
        
        Args:
            name: Name of the adapter to clean up
            
        Returns:
            True if successful, False otherwise
        """
        if not name:
            return False
            
        name = name.strip().lower()
        adapter = self._adapters.get(name)
        
        if not adapter:
            return False
        
        try:
            await adapter.cleanup()
            self._initialized_adapters.discard(name)
            logger.info(f"Cleaned up adapter: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clean up adapter {name}: {e}")
            return False
    
    async def cleanup_all(self) -> Dict[str, bool]:
        """Clean up all initialized adapters.
        
        Returns:
            Dictionary mapping adapter names to cleanup success status
        """
        results:Dict[str, bool] = {}
        
        for name in list(self._initialized_adapters):
            success = await self.cleanup_adapter(name)
            results[name] = success
        
        return results
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for automatic adapter lifecycle management.
        
        Automatically initializes all adapters on entry and cleans them up on exit.
        """
        try:
            await self.initialize_all()
            yield self
        finally:
            await self.cleanup_all()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all adapters.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "total_adapters": len(self._adapters),
            "initialized_adapters": len(self._initialized_adapters),
            "default_adapter": self._default_adapter,
            "adapters": {
                name: {
                    "initialized": name in self._initialized_adapters,
                    "capabilities": len(adapter.capabilities)
                }
                for name, adapter in self._adapters.items()
            }
        }


# Global registry instance
_global_registry: Optional[AdapterRegistry] = None
_registry_lock = Lock()


def get_registry() -> AdapterRegistry:
    """Get the global adapter registry instance.
    
    Returns:
        The global AdapterRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AdapterRegistry()
    
    return _global_registry
