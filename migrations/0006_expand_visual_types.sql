-- Phase 3F follow-up: allow the visual categories used by textbook corpora.
-- This is forward-only because 0002_rag_schema.sql is already applied.

ALTER TABLE rag_visuals
    DROP CONSTRAINT IF EXISTS rag_visuals_visual_type_check;

ALTER TABLE rag_visuals
    ADD CONSTRAINT rag_visuals_visual_type_check
    CHECK (visual_type IN (
        'diagram',
        'figure',
        'table',
        'graph',
        'chemical-structure',
        'equation'
    ));
