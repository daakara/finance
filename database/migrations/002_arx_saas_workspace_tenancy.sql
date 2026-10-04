-- Migration 002: ARX SaaS Foundation Phase 1G Workspace Tenancy (Expand Phase)
-- Target Engines: SQLite (Local/Test) & PostgreSQL (Railway Production)
-- Invariants:
-- 1. Additive expand-contract strategy (contract phase NOT authorized).
-- 2. workspaces and workspace_memberships tables established.
-- 3. workspace_id column added additively (nullable) to WORKSPACE_OWNED tables:
--    - portfolio_holdings
--    - user_trade_journal
--    - user_cockpit_actions
-- 4. user_profiles preserved as ACTOR_PROFILE without workspace_id.
-- 5. Evidence tables (gem_screening_history, forecast_history, trade_recommendation_history)
--    preserved as EVIDENCE_IMMUTABLE / SYSTEM_GLOBAL.
-- 6. Zero commercial plans, pricing tiers, or billing columns introduced.

-- 1. Core Workspaces Table
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Workspace Memberships Table
CREATE TABLE IF NOT EXISTS workspace_memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'owner',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workspace_id, user_id),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
);

-- Indices for membership lookups
CREATE INDEX IF NOT EXISTS idx_workspace_memberships_user ON workspace_memberships (user_id);
CREATE INDEX IF NOT EXISTS idx_workspace_memberships_ws ON workspace_memberships (workspace_id);

-- 3. Additive Columns & Indices for WORKSPACE_OWNED Tables
-- Note: In SQLite, column additions are executed conditionally if not already present.
-- ALTER TABLE portfolio_holdings ADD COLUMN workspace_id TEXT;
-- ALTER TABLE user_trade_journal ADD COLUMN workspace_id TEXT;
-- ALTER TABLE user_cockpit_actions ADD COLUMN workspace_id TEXT;

-- Indices for workspace-scoped query performance
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_ws ON portfolio_holdings (workspace_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_ws_sym ON portfolio_holdings (workspace_id, symbol);
CREATE INDEX IF NOT EXISTS idx_user_trade_journal_ws ON user_trade_journal (workspace_id);
CREATE INDEX IF NOT EXISTS idx_user_cockpit_actions_ws ON user_cockpit_actions (workspace_id);
