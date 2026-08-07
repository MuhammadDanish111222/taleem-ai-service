# Module 5 Runs 1–2 — Multiple Ask foundation

Status: Runs 1–2 are complete and feature-gated. `MULTIPLE_ASK_RUN1_ENABLED=false` remains mandatory until the later answer-flow/UI runs. Runs 1–2 add no answer-generation or DeepSeek path. Migrations `0010`–`0013` are applied; no real student upload, paid LLM call, deployment, commit, or push was performed.

## Secure path and scope

Every session and durable parent job persists an immutable `board_id`, `class_id`, `subject_id`, and optional `chapter_id`. A retry with the same identity-scoped request UUID must match its input metadata, text when applicable, and complete scope or it is rejected as an idempotency conflict. Jobs copy the session scope so later workers never trust mutable browser data.

The browser calls only the same-origin BFF. The BFF verifies Firebase identity and uses a new short-lived internal JWT to call the private service. The service role creates a one-object Supabase signed upload capability for an unguessable temporary path. The browser sends the raw body with the supported `PUT` signed-upload flow and only the returned `Content-Type` header. It never receives a service role, storage credential, OCR/provider key, or Railway URL.

The configured bucket is an operational prerequisite: it must already exist, remain private (Supabase buckets are private by default), allow the service role to create signed uploads, and restrict object access so no browser credential can list/read temporary sources. The Supabase signed-upload API has its own provider token lifetime; Taleem accepts finalization only within its 15-minute upload-capability deadline and removes abandoned objects through bounded cleanup.

## Validation and quota

Finalization only creates the durable `multiple_ask_validate` Railway job and starts 24-hour raw-source retention. It does not reserve quota. The Railway handler reads only bounded bytes from the private object: it verifies JPEG/PNG/WebP magic, declared-type consistency, decodability and a 20-million-pixel cap; it checks PDF magic, bounded size, strict parseability, one to ten pages, and at most 30,000 embedded characters. Pasted text is read from the private database row and is rechecked for non-empty, NUL-free content of at most 30,000 characters. A scanned PDF may have no embedded text and remains eligible for Run 2 OCR. No validation path imports or calls OCR, DeepSeek, or any other text LLM.

After successful canonical validation, the handler uses the existing `UsageService` against the stored HMAC UID and commits exactly one `multiple_question_batch` reservation in the transaction that marks the parent `validated`. Invalid inputs become `invalid` without a reservation. A quota rejection becomes terminal `limit_reached`; it creates no OCR/extraction work. Retry/concurrency paths lock the durable parent and observe `quota_status`, preventing a double charge. If later OCR/extraction fails after this point, or deterministic extraction detects more than 60 questions, the committed reservation is refunded exactly once and the job records a safe terminal error code.

## Run 2 OCR, extraction and correction

Railway uses the local Tesseract provider only for temporary student images and scanned PDF pages. Embedded PDF text is preferred; each rendered PDF page is capped at 12 million pixels before OCR. Extracted paper questions keep their visible label and section context, including objective labels such as `1.1`, grouped short-answer parts such as `2(ii)`, and long-answer parts such as `5(a)`. Parser output is deterministic: MCQ is accepted only with ordered A–D options; incomplete options and unreadable text become `not_clear` rather than guessed. The 60-question ceiling is an explicit terminal result, not truncation.

Students correct only `not_clear` items within the original job. They submit the corrected text and explicitly choose `short`, `long`, or `mcq`; an MCQ correction must carry exactly A, B, C, D options. No corrected item causes a new batch quota reservation. Polling exposes only scope, safe workflow/queue state, item metadata, paper labels/context, counts, retention expiry, and a safe terminal error code—never raw source text, uploaded bytes, storage keys, signed URLs, or provider credentials.

## Lifecycle and retention

Business workflow states are separate from `job_queue.status`: `queued`, `validating`, `validated`, `extracting`, `needs_correction`, `answering`, `partially_completed`, `completed`, `failed`, `cancelled`, `invalid`, and `limit_reached`. Run 1 stops at `validated`; it adds no extraction or answer work. Per-item modes remain only `short`, `long`, `mcq`, and `not_clear`; ordered MCQ options, unclear reason, correction/extraction metadata, and links to the existing `ai_requests`/`ai_answers` pool are pre-modelled without an answer cache.

1. An unfinalized file capability is accepted for 15 minutes. Cleanup claims abandoned sessions in bounded batches, deletes any one temporary object, and deletes the session.
2. A finalized raw file or pasted source is retained for 24 hours. Cleanup deletes the private object and text at that boundary, then redacts source metadata. A finalized source is never selected by the 15-minute capability cleanup.
3. Terminal/cancelled/expired parent jobs and items retain for seven days. Cleanup deletes items and the parent/session afterward. Audit rows use nullable `ON DELETE SET NULL` references, so they never preserve source data or block deletion. Audit records contain only action/kind/error metadata—not text, bytes, signed URLs, bucket keys, or storage credentials.

Railway-public owns `multiple_ask_validate` and future user-input OCR/extraction/answer job types only; it never owns local-admin ingestion or bulk embeddings.

The Railway-owned worker runs bounded Multiple Ask cleanup every five minutes. Cleanup failures are logged and retried on the next interval without stopping durable job processing.
