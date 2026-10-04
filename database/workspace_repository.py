"""
Dedicated Workspace Persistence Repository for ARX SaaS Foundation (Phase 1G).

Provides the domain-level storage interface for workspace entities,
memberships, and workspace-owned persistent data.
Enforces INV-SAAS-03 (record resolves to exactly one workspace) and
INV-SAAS-06 (single workspace identity authority).
"""

import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime

from api.context.workspace_identity import (
    derive_compatibility_workspace_id,
    is_valid_workspace_id,
)
from database.workspace_migration import (
    apply_workspace_tenancy_migration,
    backfill_workspace_tenancy,
)


class WorkspaceRepository:
    """Repository managing workspace tenancy entities and scoped data access."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_initialized()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialized(self) -> None:
        conn = self._get_connection()
        try:
            apply_workspace_tenancy_migration(conn)
            backfill_workspace_tenancy(conn)
        finally:
            conn.close()

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve workspace record by workspace_id."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT workspace_id, name, created_at, updated_at FROM workspaces WHERE workspace_id = ?;",
                (workspace_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "workspace_id": row["workspace_id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            return None
        finally:
            conn.close()

    def create_workspace(self, workspace_id: str, name: str) -> Dict[str, Any]:
        """Create a new workspace entity if it does not already exist."""
        if workspace_id == "ws_default":
            raise ValueError("Cannot create persistent workspace for 'ws_default' (INV-SAAS-07).")
        if not is_valid_workspace_id(workspace_id):
            raise ValueError(f"Invalid workspace_id: '{workspace_id}'")

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workspaces (workspace_id, name)
                    VALUES (?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        name = excluded.name,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (workspace_id, name),
                )
            return self.get_workspace(workspace_id) or {"workspace_id": workspace_id, "name": name}
        finally:
            conn.close()

    def add_membership(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "owner",
    ) -> Dict[str, Any]:
        """Add a compatibility membership association between user and workspace."""
        if workspace_id == "ws_default":
            raise ValueError("Cannot provision persistent membership for 'ws_default' (INV-SAAS-07).")
        if not is_valid_workspace_id(workspace_id):
            raise ValueError(f"Invalid workspace_id: '{workspace_id}'")
        if not user_id or not str(user_id).strip() or str(user_id).strip() in ("default_user", "default"):
            raise ValueError("user_id cannot be empty or default for membership provisioning (INV-SAAS-07).")

        # Ensure workspace exists
        self.create_workspace(workspace_id, f"Workspace {user_id}")

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workspace_memberships (workspace_id, user_id, role)
                    VALUES (?, ?, ?)
                    ON CONFLICT(workspace_id, user_id) DO UPDATE SET
                        role = excluded.role,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (workspace_id, user_id.strip(), role),
                )
            return {
                "workspace_id": workspace_id,
                "user_id": user_id.strip(),
                "role": role,
            }
        finally:
            conn.close()

    def get_memberships(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List all memberships in a workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, workspace_id, user_id, role, created_at, updated_at
                FROM workspace_memberships
                WHERE workspace_id = ?
                ORDER BY id ASC;
                """,
                (workspace_id,),
            )
            return [
                {
                    "id": row["id"],
                    "workspace_id": row["workspace_id"],
                    "user_id": row["user_id"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_user_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """List all workspaces associated with a user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT w.workspace_id, w.name, m.role, w.created_at, w.updated_at
                FROM workspaces w
                JOIN workspace_memberships m ON w.workspace_id = m.workspace_id
                WHERE m.user_id = ?
                ORDER BY w.created_at ASC;
                """,
                (user_id.strip(),),
            )
            return [
                {
                    "workspace_id": row["workspace_id"],
                    "name": row["name"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
