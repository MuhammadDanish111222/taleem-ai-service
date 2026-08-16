# Module 6 — Stage 3 deployment

Apply `migrations/0019_module6_ephemeral_test_generation.sql`,
`migrations/0020_question_answer_visual_roles.sql`, and
`migrations/0021_module6_test_paper_visuals.sql`, then deploy
`supabase/functions/generate-test-paper`. Configure Edge secrets only:
`SUPABASE_SERVICE_ROLE_KEY`, `TALEEM_INTERNAL_JWT_PUBLIC_KEY`, and
`TALEEM_INTERNAL_JWT_KEY_ID` (`SUPABASE_URL` is provided by Supabase).

`taleem-web` needs only `SUPABASE_URL`, `INTERNAL_JWT_PRIVATE_KEY`, and
`INTERNAL_JWT_KEY_ID`; it must never receive a Supabase service-role key.
The per-function `verify_jwt = false` setting is intentional because the Edge
function verifies Taleem's RS256 `taleem-test-generator` token itself.

Browser → `POST /api/tests/generate` → Edge Function → one
`taleem_generate_test_paper` RPC → ephemeral paper-safe JSON. Railway, AI
providers, queues, persistence, PDF generation, and answer keys are not part
of this path. Run `npm run typecheck` and the focused Vitest test below; Edge
JWT helper checks run with `deno test supabase/functions/generate-test-paper`.

Question images use the same Edge function only through the authenticated
Next.js visual BFF. It verifies that the visual belongs to the current approved
question revision and returns the private Drive reference to that server BFF
only; the browser receives image bytes, never the storage key.
