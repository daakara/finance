-- ARX Smart Money PostgreSQL 16 Cluster Bootstrapping Script
-- Out-of-band administrative script for initializing roles on fresh clusters.
-- Run strictly by cluster administrator (e.g. postgres user).

CREATE ROLE smart_money_ingestion WITH
  LOGIN
  NOSUPERUSER
  NOCREATEROLE
  NOREPLICATION
  CONNECTION LIMIT -1;

CREATE ROLE smart_money_web WITH
  LOGIN
  NOSUPERUSER
  NOCREATEROLE
  NOREPLICATION
  CONNECTION LIMIT -1;

CREATE ROLE smart_money_migration WITH
  LOGIN
  NOSUPERUSER
  NOCREATEROLE
  NOREPLICATION
  CONNECTION LIMIT -1;

CREATE DATABASE smart_money OWNER postgres;

\c smart_money

GRANT CONNECT ON DATABASE smart_money TO smart_money_ingestion;
GRANT CONNECT ON DATABASE smart_money TO smart_money_web;
GRANT CONNECT ON DATABASE smart_money TO smart_money_migration;

GRANT USAGE ON SCHEMA public TO smart_money_ingestion;
GRANT USAGE ON SCHEMA public TO smart_money_web;
GRANT USAGE ON SCHEMA public TO smart_money_migration;

-- Ingestion role rights
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO smart_money_ingestion;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO smart_money_ingestion;

-- Web reader role rights
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO smart_money_web;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO smart_money_web;

-- Migration role rights
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO smart_money_migration;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO smart_money_migration;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO smart_money_migration;
