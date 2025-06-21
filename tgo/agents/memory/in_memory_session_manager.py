"""
In-memory session manager implementation.

This module provides an in-memory implementation of the SessionManager
interface for development and testing purposes.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from threading import Lock

from ..core.interfaces import SessionManager
from ..core.models import Session, SessionConfig
from ..core.enums import SessionType

logger = logging.getLogger(__name__)


class InMemorySessionManager(SessionManager):
    """In-memory implementation of SessionManager.
    
    This implementation stores sessions in memory and provides basic
    session lifecycle management. It's suitable for development and
    testing but not for production use.
    """
    
    def __init__(self, config: Optional[SessionConfig] = None):
        self._config = config or SessionConfig()
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self._lock = Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info("InMemorySessionManager initialized")
    
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        **kwargs: Any
    ) -> Session:
        """Create a new session."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            
            # Check user session limit
            user_session_count = len(self._user_sessions.get(user_id, []))
            if user_session_count >= self._config.max_sessions_per_user:
                raise ValueError(f"User {user_id} has reached maximum session limit")
            
            # Calculate expiration time
            expires_at = None
            if self._config.session_timeout_minutes > 0:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    minutes=self._config.session_timeout_minutes
                )
            
            # Create session
            session = Session(
                session_id=session_id,
                user_id=user_id,
                session_type=session_type,
                expires_at=expires_at,
                **kwargs
            )
            
            # Store session
            self._sessions[session_id] = session
            
            # Update user sessions index
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            logger.info(f"Created session: {session_id} for user: {user_id}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired():
                session.update_activity()
                return session
            elif session and session.is_expired():
                # Remove expired session
                await self._remove_session_internal(session_id)
                return None
            return None
    
    async def update_session(self, session_id: str, **updates: Any) -> bool:
        """Update session properties."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            # Update allowed fields
            allowed_fields = {'status', 'context', 'metadata', 'expires_at'}
            for key, value in updates.items():
                if key in allowed_fields:
                    setattr(session, key, value)
            
            session.update_activity()
            logger.debug(f"Updated session: {session_id}")
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            return await self._remove_session_internal(session_id)
    
    async def list_user_sessions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[Session]:
        """List sessions for a user."""
        with self._lock:
            user_session_ids = self._user_sessions.get(user_id, [])
            sessions: List[Session] = []
            
            for session_id in user_session_ids:
                session = self._sessions.get(session_id)
                if not session:
                    continue
                

                
                # Filter by active status if requested
                if active_only and not session.is_active():
                    continue
                
                # Check if expired
                if session.is_expired():
                    await self._remove_session_internal(session_id)
                    continue
                
                sessions.append(session)
            
            return sessions
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        with self._lock:
            expired_sessions: List[str] = []
            for session_id, session in self._sessions.items():
                if session.is_expired():
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            count = 0
            for session_id in expired_sessions:
                if await self._remove_session_internal(session_id):
                    count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
            
            return count
    
    async def _remove_session_internal(self, session_id: str) -> bool:
        """Internal method to remove a session (assumes lock is held)."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Remove from sessions
        del self._sessions[session_id]
        
        # Remove from user sessions index
        user_id = session.user_id
        if user_id in self._user_sessions:
            try:
                self._user_sessions[user_id].remove(session_id)
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]
            except ValueError:
                pass  # Session ID not in list
        
        logger.debug(f"Removed session: {session_id}")
        return True
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._config.cleanup_interval_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval_minutes * 60)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get session statistics."""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": len([s for s in self._sessions.values() if s.is_active()]),
                "total_users": len(self._user_sessions)
            }
    
    async def shutdown(self):
        """Shutdown the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("InMemorySessionManager shutdown")
