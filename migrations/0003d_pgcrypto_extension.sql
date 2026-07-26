-- Ensure pgcrypto exists in a separate migration before 0004 references digest().
-- The migration runner submits one SQL file at a time, so an extension created
-- later in the same file is not available while PostgreSQL parses digest calls.
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
