-- Module 4 Run 1: public Ask foundation, atomic usage fallback, prompts,
-- generated candidates, and the one unified approved Question-Answer Bank.
-- Forward-only: migrations 0001-0008 are intentionally left unchanged.

SET search_path = public, pg_catalog;

-- Existing generated-answer tables remain the only candidate/audit pool.
ALTER TABLE ai_requests DROP CONSTRAINT IF EXISTS ai_requests_answer_mode_check;
ALTER TABLE ai_requests
    ADD CONSTRAINT ai_requests_answer_mode_check
    CHECK (answer_mode IN (
        'concise', 'detailed', 'step_by_step', 'exam_style',
        'short', 'long', 'mcq'
    ));
ALTER TABLE ai_requests DROP CONSTRAINT IF EXISTS ai_requests_status_check;
ALTER TABLE ai_requests
    ADD CONSTRAINT ai_requests_status_check
    CHECK (status IN (
        'pending', 'processing', 'completed', 'failed', 'cached', 'no_answer'
    ));
ALTER TABLE ai_requests
    ADD COLUMN IF NOT EXISTS client_request_id UUID NULL,
    ADD COLUMN IF NOT EXISTS uid_hash CHAR(64) NULL,
    ADD COLUMN IF NOT EXISTS chapter_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS answer_style TEXT NOT NULL DEFAULT 'exam_style',
    ADD COLUMN IF NOT EXISTS answer_source TEXT NULL,
    ADD COLUMN IF NOT EXISTS source_feature TEXT NOT NULL DEFAULT 'single_question',
    ADD COLUMN IF NOT EXISTS normalization_version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS usage_business_date DATE NULL,
    ADD COLUMN IF NOT EXISTS retention_expires_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS terminal_error_code TEXT NULL,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE ai_requests
    ADD CONSTRAINT ai_requests_uid_hash_sha256
        CHECK (uid_hash IS NULL OR char_length(uid_hash) = 64),
    ADD CONSTRAINT ai_requests_answer_style_valid
        CHECK (answer_style = 'exam_style'),
    ADD CONSTRAINT ai_requests_answer_source_valid
        CHECK (answer_source IS NULL OR answer_source IN (
            'approved_bank', 'syllabus_grounded', 'general_knowledge'
        )),
    ADD CONSTRAINT ai_requests_source_feature_valid
        CHECK (source_feature IN ('single_question', 'multiple_question')),
    ADD CONSTRAINT ai_requests_normalization_version_positive
        CHECK (normalization_version > 0);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_requests_uid_client_request
    ON ai_requests (uid_hash, client_request_id)
    WHERE uid_hash IS NOT NULL AND client_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ai_requests_candidate_retention
    ON ai_requests (retention_expires_at)
    WHERE retention_expires_at IS NOT NULL AND status IN ('pending', 'failed');

ALTER TABLE ai_answers
    ADD COLUMN IF NOT EXISTS answer_blocks JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS answer_source TEXT NULL,
    ADD COLUMN IF NOT EXISTS answer_mode TEXT NULL,
    ADD COLUMN IF NOT EXISTS answer_style TEXT NOT NULL DEFAULT 'exam_style',
    ADD COLUMN IF NOT EXISTS citation_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS visual_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS provider TEXT NULL,
    ADD COLUMN IF NOT EXISTS model TEXT NULL,
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS reviewed_by TEXT NULL,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS rejection_reason TEXT NULL,
    ADD COLUMN IF NOT EXISTS retention_expires_at TIMESTAMPTZ NULL;
ALTER TABLE ai_answers
    ADD CONSTRAINT ai_answers_blocks_array
        CHECK (jsonb_typeof(answer_blocks) = 'array'),
    ADD CONSTRAINT ai_answers_citation_ids_array
        CHECK (jsonb_typeof(citation_ids) = 'array'),
    ADD CONSTRAINT ai_answers_visual_ids_array
        CHECK (jsonb_typeof(visual_ids) = 'array'),
    ADD CONSTRAINT ai_answers_source_valid
        CHECK (answer_source IS NULL OR answer_source IN (
            'approved_bank', 'syllabus_grounded', 'general_knowledge'
        )),
    ADD CONSTRAINT ai_answers_mode_valid
        CHECK (answer_mode IS NULL OR answer_mode IN ('short', 'long', 'mcq')),
    ADD CONSTRAINT ai_answers_style_valid CHECK (answer_style = 'exam_style'),
    ADD CONSTRAINT ai_answers_review_status_valid
        CHECK (review_status IN ('pending', 'approved', 'rejected'));
CREATE INDEX IF NOT EXISTS idx_ai_answers_pending_review
    ON ai_answers (created_at, id) WHERE review_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_ai_answers_retention
    ON ai_answers (retention_expires_at)
    WHERE retention_expires_at IS NOT NULL AND review_status IN ('pending', 'rejected');

-- Convert the lightweight Phase 3 table into the unified immutable revision bank.
ALTER TABLE approved_question_bank RENAME TO approved_question_bank_legacy;

CREATE TABLE question_bank_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE question_bank_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id UUID NOT NULL REFERENCES question_bank_questions(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    chapter_id TEXT NULL,
    answer_mode TEXT NOT NULL CHECK (answer_mode IN ('short', 'long', 'mcq')),
    answer_style TEXT NOT NULL DEFAULT 'exam_style' CHECK (answer_style = 'exam_style'),
    difficulty TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    marks NUMERIC(8,2) NOT NULL CHECK (marks > 0),
    question_text TEXT NOT NULL CHECK (btrim(question_text) <> ''),
    normalized_question TEXT NOT NULL CHECK (btrim(normalized_question) <> ''),
    question_hash CHAR(64) NOT NULL,
    normalization_version INTEGER NOT NULL DEFAULT 1 CHECK (normalization_version > 0),
    embedding vector(768) NULL,
    embedding_model TEXT NULL,
    embedding_revision TEXT NULL,
    embedding_config_fingerprint TEXT NULL,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'embedded', 'failed')),
    answer_blocks JSONB NOT NULL CHECK (
        jsonb_typeof(answer_blocks) = 'array' AND jsonb_array_length(answer_blocks) > 0
    ),
    review_status TEXT NOT NULL CHECK (
        review_status IN ('pending', 'approved', 'rejected', 'archived')
    ),
    source TEXT NOT NULL,
    approved_by TEXT NULL,
    approved_at TIMESTAMPTZ NULL,
    rejected_by TEXT NULL,
    rejected_at TIMESTAMPTZ NULL,
    rejection_reason TEXT NULL,
    superseded_at TIMESTAMPTZ NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (question_id, version_no),
    CHECK (char_length(question_hash) = 64),
    CHECK (
        (review_status = 'approved' AND approved_by IS NOT NULL AND approved_at IS NOT NULL)
        OR review_status <> 'approved'
    )
);
CREATE UNIQUE INDEX idx_question_bank_one_active_revision
    ON question_bank_revisions (question_id)
    WHERE review_status = 'approved' AND superseded_at IS NULL;
CREATE INDEX idx_question_bank_exact_lookup
    ON question_bank_revisions (
        board_id, class_id, subject_id, answer_mode, question_hash, chapter_id
    )
    WHERE review_status = 'approved' AND superseded_at IS NULL;
CREATE INDEX idx_question_bank_normalized_lookup
    ON question_bank_revisions (
        board_id, class_id, subject_id, answer_mode, normalized_question, chapter_id
    )
    WHERE review_status = 'approved' AND superseded_at IS NULL;
CREATE INDEX idx_question_bank_revision_embedding_work
    ON question_bank_revisions (embedding_status, created_at)
    WHERE review_status = 'approved' AND superseded_at IS NULL;

CREATE TABLE question_bank_mcq_options (
    revision_id UUID NOT NULL REFERENCES question_bank_revisions(id) ON DELETE RESTRICT,
    option_key TEXT NOT NULL CHECK (btrim(option_key) <> ''),
    option_text TEXT NOT NULL CHECK (btrim(option_text) <> ''),
    display_order INTEGER NOT NULL CHECK (display_order >= 0),
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (revision_id, option_key),
    UNIQUE (revision_id, display_order)
);
CREATE UNIQUE INDEX idx_question_bank_one_correct_mcq
    ON question_bank_mcq_options (revision_id) WHERE is_correct;

CREATE TABLE question_bank_variations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revision_id UUID NOT NULL REFERENCES question_bank_revisions(id) ON DELETE RESTRICT,
    variation_text TEXT NOT NULL CHECK (btrim(variation_text) <> ''),
    normalized_variation TEXT NOT NULL CHECK (btrim(normalized_variation) <> ''),
    variation_hash CHAR(64) NOT NULL,
    normalization_version INTEGER NOT NULL DEFAULT 1 CHECK (normalization_version > 0),
    embedding vector(768) NULL,
    embedding_model TEXT NULL,
    embedding_revision TEXT NULL,
    embedding_config_fingerprint TEXT NULL,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'embedded', 'failed')),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (char_length(variation_hash) = 64),
    UNIQUE (revision_id, variation_hash)
);
CREATE INDEX idx_question_bank_variation_exact
    ON question_bank_variations (variation_hash, revision_id)
    WHERE active;
CREATE INDEX idx_question_bank_variation_embedding_work
    ON question_bank_variations (embedding_status, created_at)
    WHERE active;

CREATE TABLE question_bank_revision_visuals (
    revision_id UUID NOT NULL REFERENCES question_bank_revisions(id) ON DELETE RESTRICT,
    visual_id UUID NOT NULL REFERENCES rag_visuals(id) ON DELETE RESTRICT,
    display_order INTEGER NOT NULL CHECK (display_order >= 0),
    PRIMARY KEY (revision_id, visual_id),
    UNIQUE (revision_id, display_order)
);

CREATE TABLE question_bank_revision_citations (
    revision_id UUID NOT NULL REFERENCES question_bank_revisions(id) ON DELETE RESTRICT,
    chunk_id UUID NOT NULL REFERENCES rag_chunks(id) ON DELETE RESTRICT,
    display_order INTEGER NOT NULL CHECK (display_order >= 0),
    PRIMARY KEY (revision_id, chunk_id),
    UNIQUE (revision_id, display_order)
);

CREATE TABLE question_bank_imports (
    import_key TEXT PRIMARY KEY CHECK (btrim(import_key) <> ''),
    payload_hash CHAR(64) NOT NULL,
    revision_ids UUID[] NOT NULL DEFAULT '{}',
    actor_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (char_length(payload_hash) = 64)
);

INSERT INTO question_bank_questions (id, source, created_by, created_at)
SELECT id, source, 'legacy_migration', created_at
FROM approved_question_bank_legacy;

INSERT INTO question_bank_revisions (
    question_id, version_no, board_id, class_id, subject_id, chapter_id,
    answer_mode, answer_style, difficulty, marks, question_text,
    normalized_question, question_hash, normalization_version, answer_blocks,
    review_status, source, approved_by, approved_at, created_by, created_at
)
SELECT
    id, 1, board_id, class_id, subject_id, NULL,
    'short', 'exam_style', 'medium', 1,
    normalized_question, normalized_question, question_hash, 1,
    jsonb_build_array(jsonb_build_object('type', 'paragraph', 'text', answer_text)),
    CASE status
        WHEN 'approved' THEN 'approved'
        WHEN 'archived' THEN 'archived'
        ELSE 'pending'
    END,
    source,
    CASE WHEN status = 'approved' THEN 'legacy_migration' ELSE NULL END,
    CASE WHEN status = 'approved' THEN created_at ELSE NULL END,
    'legacy_migration', created_at
FROM approved_question_bank_legacy;

DROP TABLE approved_question_bank_legacy;

ALTER TABLE ai_answers
    ADD COLUMN IF NOT EXISTS approved_revision_id UUID NULL
        REFERENCES question_bank_revisions(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_ai_answers_approved_revision
    ON ai_answers (approved_revision_id)
    WHERE approved_revision_id IS NOT NULL;

-- Versioned editable teaching prompts. Hard safety instructions remain in code.
CREATE TABLE prompt_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_key TEXT NOT NULL CHECK (prompt_key IN ('ask_grounded', 'ask_general')),
    answer_mode TEXT NOT NULL CHECK (answer_mode IN ('short', 'long', 'mcq')),
    board_id TEXT NULL,
    class_id TEXT NULL,
    subject_id TEXT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    content TEXT NOT NULL CHECK (btrim(content) <> '' AND char_length(content) <= 20000),
    status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'retired')),
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_by TEXT NULL,
    activated_at TIMESTAMPTZ NULL,
    retired_at TIMESTAMPTZ NULL,
    CHECK (board_id IS NULL OR (class_id IS NOT NULL AND subject_id IS NOT NULL)),
    CHECK (class_id IS NULL OR subject_id IS NOT NULL),
    CHECK (
        (status = 'active' AND activated_by IS NOT NULL AND activated_at IS NOT NULL)
        OR status <> 'active'
    )
);
CREATE UNIQUE INDEX idx_prompt_scope_version
    ON prompt_versions (
        prompt_key, answer_mode,
        COALESCE(board_id, ''), COALESCE(class_id, ''), COALESCE(subject_id, ''),
        version
    );
CREATE UNIQUE INDEX idx_prompt_one_active_scope
    ON prompt_versions (
        prompt_key, answer_mode,
        COALESCE(board_id, ''), COALESCE(class_id, ''), COALESCE(subject_id, '')
    ) WHERE status = 'active';
CREATE INDEX idx_prompt_resolution
    ON prompt_versions (prompt_key, answer_mode, status, subject_id, class_id, board_id);

INSERT INTO prompt_versions (
    prompt_key, answer_mode, board_id, class_id, subject_id, version, content,
    status, created_by, activated_by, activated_at
)
SELECT
    seed.prompt_key, seed.answer_mode, NULL, NULL, NULL, 1, seed.content,
    'active', 'migration', 'migration', NOW()
FROM (VALUES
    ('ask_grounded', 'short',
     'Write a concise, exam-ready answer using only the supplied textbook evidence.'),
    ('ask_grounded', 'long',
     'Write a clear, well-structured, exam-ready answer using only the supplied textbook evidence.'),
    ('ask_grounded', 'mcq',
     'Write one exam-ready multiple-choice answer using only the supplied textbook evidence.'),
    ('ask_general', 'short',
     'Write a concise, age-appropriate general-knowledge answer and do not claim textbook verification.'),
    ('ask_general', 'long',
     'Write a clear, well-structured, age-appropriate general-knowledge answer and do not claim textbook verification.'),
    ('ask_general', 'mcq',
     'Write one age-appropriate general-knowledge multiple-choice answer and do not claim textbook verification.')
) AS seed(prompt_key, answer_mode, content);

CREATE TABLE cache_generations (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    generation BIGINT NOT NULL DEFAULT 1 CHECK (generation > 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace, cache_key)
);

CREATE TABLE ask_source_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    class_id TEXT NULL,
    subject_id TEXT NULL,
    allow_general BOOLEAN NOT NULL DEFAULT FALSE,
    semantic_reuse_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    semantic_distance_threshold DOUBLE PRECISION NULL,
    updated_by TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (class_id IS NULL OR subject_id IS NOT NULL),
    CHECK (
        (semantic_reuse_enabled AND semantic_distance_threshold IS NOT NULL
         AND semantic_distance_threshold >= 0
         AND semantic_distance_threshold <= 2)
        OR (NOT semantic_reuse_enabled AND semantic_distance_threshold IS NULL)
    )
);
CREATE UNIQUE INDEX idx_ask_source_policy_scope
    ON ask_source_policies (COALESCE(class_id, ''), COALESCE(subject_id, ''));
INSERT INTO ask_source_policies (class_id, subject_id, allow_general, updated_by)
VALUES (NULL, NULL, FALSE, 'migration');

-- Daily quota policy and authoritative PostgreSQL mirror/fallback.
CREATE TABLE usage_policies (
    feature TEXT NOT NULL CHECK (
        feature IN ('single_question', 'multiple_question_batch')
    ),
    account_tier TEXT NOT NULL CHECK (
        account_tier IN ('anonymous', 'google', 'premium')
    ),
    daily_limit INTEGER NOT NULL CHECK (daily_limit >= 0),
    student_visible BOOLEAN NOT NULL DEFAULT TRUE,
    updated_by TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (feature, account_tier)
);
INSERT INTO usage_policies (
    feature, account_tier, daily_limit, student_visible, updated_by
) VALUES
    ('single_question', 'anonymous', 5, TRUE, 'migration'),
    ('single_question', 'google', 5, TRUE, 'migration'),
    ('single_question', 'premium', 10000, FALSE, 'migration'),
    ('multiple_question_batch', 'anonymous', 0, TRUE, 'migration'),
    ('multiple_question_batch', 'google', 1, TRUE, 'migration'),
    ('multiple_question_batch', 'premium', 1000, FALSE, 'migration')
ON CONFLICT (feature, account_tier) DO NOTHING;

CREATE TABLE daily_usage (
    business_date DATE NOT NULL,
    feature TEXT NOT NULL,
    uid_hash CHAR(64) NOT NULL,
    used INTEGER NOT NULL DEFAULT 0 CHECK (used >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (business_date, feature, uid_hash),
    CHECK (char_length(uid_hash) = 64)
);

CREATE TABLE usage_reservations (
    request_id UUID NOT NULL,
    business_date DATE NOT NULL,
    feature TEXT NOT NULL,
    uid_hash CHAR(64) NOT NULL,
    account_tier TEXT NOT NULL CHECK (
        account_tier IN ('anonymous', 'google', 'premium')
    ),
    backend TEXT NOT NULL CHECK (backend IN ('redis', 'postgresql')),
    status TEXT NOT NULL CHECK (status IN ('reserved', 'committed', 'refunded')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (request_id, uid_hash),
    CHECK (char_length(uid_hash) = 64)
);
CREATE INDEX idx_usage_reservations_lookup
    ON usage_reservations (business_date, feature, uid_hash, status);

-- Redis-outage replay protection. Only an HMAC digest of the JTI is stored.
CREATE TABLE internal_jti_replay (
    jti_hash CHAR(64) PRIMARY KEY,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (char_length(jti_hash) = 64)
);
CREATE INDEX idx_internal_jti_replay_expiry ON internal_jti_replay (expires_at);

-- Deny direct public/client access to every new Module 4 object.
ALTER TABLE question_bank_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_mcq_options ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_variations ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_revision_visuals ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_revision_citations ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_bank_imports ENABLE ROW LEVEL SECURITY;
ALTER TABLE prompt_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cache_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ask_source_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_reservations ENABLE ROW LEVEL SECURITY;
ALTER TABLE internal_jti_replay ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON question_bank_questions, question_bank_revisions,
    question_bank_mcq_options, question_bank_variations,
    question_bank_revision_visuals, question_bank_revision_citations,
    question_bank_imports, prompt_versions, cache_generations,
    ask_source_policies, usage_policies, daily_usage, usage_reservations,
    internal_jti_replay
FROM PUBLIC, anon, authenticated;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC, anon, authenticated;
