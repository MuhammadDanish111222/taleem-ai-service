-- New imported visuals still require explicit review, but their eventual display
-- policy defaults to relevance-based selection instead of permanent hiding.
ALTER TABLE rag_visuals
    ALTER COLUMN display_policy SET DEFAULT 'llm_decide';

-- Rows that are still unreviewed retain no final admin decision, so move the old
-- import default forward. Approved and rejected rows are not changed.
UPDATE rag_visuals
SET display_policy = 'llm_decide'
WHERE review_status = 'pending'
  AND display_policy = 'never';
