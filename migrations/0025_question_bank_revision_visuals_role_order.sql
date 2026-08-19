-- Migration 0025: Scope display_order by role on question_bank_revision_visuals

ALTER TABLE question_bank_revision_visuals
    DROP CONSTRAINT IF EXISTS question_bank_revision_visuals_revision_id_display_order_key;

ALTER TABLE question_bank_revision_visuals
    DROP CONSTRAINT IF EXISTS question_bank_revision_visuals_role_order_key;

ALTER TABLE question_bank_revision_visuals
    ADD CONSTRAINT question_bank_revision_visuals_role_order_key
        UNIQUE (revision_id, role, display_order);
