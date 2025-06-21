"""
In-memory memory manager implementation.

This module provides an in-memory implementation of the MemoryManager
interface for development and testing purposes.
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from threading import Lock
import re

from ..core.interfaces import MemoryManager
from ..core.models import MemoryEntry, MemoryConfig
from ..core.enums import SessionType

logger = logging.getLogger(__name__)


class InMemoryMemoryManager(MemoryManager):
    """In-memory implementation of MemoryManager.
    
    This implementation stores memories in memory and provides basic
    memory management capabilities. It's suitable for development and
    testing but not for production use.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self._config = config or MemoryConfig()
        self._memories: Dict[str, MemoryEntry] = {}  # memory_id -> MemoryEntry
        self._session_memories: Dict[str, List[str]] = {}  # session_id -> [memory_ids]
        self._lock = Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info("InMemoryMemoryManager initialized")
    
    async def store_memory(
        self, 
        session_id: str, 
        content: str, 
        memory_type: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> MemoryEntry:
        """Store a memory entry."""
        with self._lock:
            # Check session memory limit
            session_memory_count = len(self._session_memories.get(session_id, []))
            if session_memory_count >= self._config.max_memories_per_session:
                # Remove oldest low-importance memory
                await self._cleanup_session_memories(session_id)
            
            # Create memory entry
            memory = MemoryEntry(
                session_id=session_id,
                session_type=session_type,
                agent_id=agent_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store memory
            self._memories[memory.memory_id] = memory
            
            # Update session memories index
            if session_id not in self._session_memories:
                self._session_memories[session_id] = []
            self._session_memories[session_id].append(memory.memory_id)
            
            logger.debug(f"Stored memory: {memory.memory_id} for session: {session_id}")
            return memory
    
    async def retrieve_memories(
        self, 
        session_id: str, 
        session_type: SessionType = SessionType.SINGLE_CHAT,
        memory_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """Retrieve memories for a session."""
        with self._lock:
            memory_ids = self._session_memories.get(session_id, [])
            memories = []
            
            for memory_id in memory_ids:
                memory = self._memories.get(memory_id)
                if not memory:
                    continue
                
                # Apply filters
                if session_type != memory.session_type:
                    continue
                
                if memory_type and memory.memory_type != memory_type:
                    continue
                
                if agent_id and memory.agent_id != agent_id:
                    continue
                
                if memory.importance < min_importance:
                    continue
                
                memories.append(memory)
            
            # Sort by importance and recency
            memories.sort(
                key=lambda m: (m.importance, m.created_at), 
                reverse=True
            )
            
            # Update access information
            for memory in memories[:limit]:
                memory.update_access()
            
            return memories[:limit]
    
    async def search_memories(
        self, 
        session_id: str, 
        query: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        memory_type: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[MemoryEntry]:
        """Search memories by content similarity."""
        with self._lock:
            memory_ids = self._session_memories.get(session_id, [])
            matching_memories = []
            
            # Simple text-based search (in production, use semantic search)
            query_lower = query.lower()
            query_words = set(re.findall(r'\w+', query_lower))
            
            for memory_id in memory_ids:
                memory = self._memories.get(memory_id)
                if not memory:
                    continue
                
                # Apply filters
                if session_type != memory.session_type:
                    continue
                
                if memory_type and memory.memory_type != memory_type:
                    continue
                
                # Calculate simple similarity score
                content_lower = memory.content.lower()
                content_words = set(re.findall(r'\w+', content_lower))
                
                if not content_words:
                    continue
                
                # Jaccard similarity
                intersection = query_words.intersection(content_words)
                union = query_words.union(content_words)
                similarity = len(intersection) / len(union) if union else 0
                
                # Also check for substring matches
                if query_lower in content_lower:
                    similarity = max(similarity, 0.8)
                
                if similarity >= similarity_threshold:
                    matching_memories.append((memory, similarity))
            
            # Sort by similarity and importance
            matching_memories.sort(
                key=lambda x: (x[1], x[0].importance), 
                reverse=True
            )
            
            # Update access information and return memories
            result_memories = []
            for memory, _ in matching_memories[:limit]:
                memory.update_access()
                result_memories.append(memory)
            
            return result_memories
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        with self._lock:
            memory = self._memories.get(memory_id)
            if not memory:
                return False
            
            # Remove from memories
            del self._memories[memory_id]
            
            # Remove from session memories index
            session_id = memory.session_id
            if session_id in self._session_memories:
                try:
                    self._session_memories[session_id].remove(memory_id)
                    if not self._session_memories[session_id]:
                        del self._session_memories[session_id]
                except ValueError:
                    pass  # Memory ID not in list
            
            logger.debug(f"Deleted memory: {memory_id}")
            return True
    
    async def cleanup_old_memories(
        self, 
        retention_days: int = 30,
        min_importance: float = 0.1
    ) -> int:
        """Clean up old or low-importance memories."""
        with self._lock:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            memories_to_delete = []
            
            for memory_id, memory in self._memories.items():
                # Delete if too old and low importance
                if (memory.created_at < cutoff_date and 
                    memory.importance < min_importance):
                    memories_to_delete.append(memory_id)
            
            # Delete memories
            count = 0
            for memory_id in memories_to_delete:
                if await self.delete_memory(memory_id):
                    count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} old memories")
            
            return count
    
    async def _cleanup_session_memories(self, session_id: str):
        """Clean up memories for a session when limit is reached."""
        memory_ids = self._session_memories.get(session_id, [])
        if not memory_ids:
            return
        
        # Get all memories for the session
        memories = []
        for memory_id in memory_ids:
            memory = self._memories.get(memory_id)
            if memory:
                memories.append(memory)
        
        # Sort by importance and age (keep important and recent ones)
        memories.sort(key=lambda m: (m.importance, m.created_at))
        
        # Remove the least important/oldest memories
        memories_to_remove = len(memories) - self._config.max_memories_per_session + 1
        for i in range(min(memories_to_remove, len(memories) // 2)):  # Remove at most half
            await self.delete_memory(memories[i].memory_id)
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._config.cleanup_interval_hours > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval_hours * 3600)
                await self.cleanup_old_memories(
                    retention_days=self._config.memory_retention_days,
                    min_importance=0.1
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        with self._lock:
            return {
                "total_memories": len(self._memories),
                "total_sessions": len(self._session_memories),
                "avg_memories_per_session": (
                    len(self._memories) / len(self._session_memories) 
                    if self._session_memories else 0
                )
            }
    
    async def shutdown(self):
        """Shutdown the memory manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("InMemoryMemoryManager shutdown")
