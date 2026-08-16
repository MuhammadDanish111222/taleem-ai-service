import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";
import { verifyTestGeneratorJwt } from "./auth.ts";

const MAX_BODY_BYTES = 32_000;
const scope = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const headers = { "content-type": "application/json", "cache-control": "no-store" };
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers });

function fail(code: string, status: number) { return response({ error: { code } }, status); }
function isObject(value: unknown): value is Record<string, unknown> { return !!value && typeof value === "object" && !Array.isArray(value); }

Deno.serve(async (request) => {
  if (request.method !== "POST") return fail("INVALID_REQUEST", 405);
  let claims: Awaited<ReturnType<typeof verifyTestGeneratorJwt>>;
  try { claims = await verifyTestGeneratorJwt(request.headers.get("authorization")); }
  catch { return fail("UNAUTHENTICATED", 401); }
  const length = Number(request.headers.get("content-length") ?? "0");
  if (!Number.isFinite(length) || length > MAX_BODY_BYTES) return fail("INVALID_REQUEST", 400);
  let body: unknown;
  try { body = await request.json(); } catch { return fail("INVALID_REQUEST", 400); }
  if (JSON.stringify(body).length > MAX_BODY_BYTES) return fail("INVALID_REQUEST", 400);
  if (!isObject(body) || "uid" in body || typeof body.seed !== "string" || body.seed !== claims.request_id) return fail("INVALID_REQUEST", 400);

  const serviceRole = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  if (!serviceRole || !supabaseUrl) return fail("TEST_GENERATOR_UNAVAILABLE", 503);
  const client = createClient(supabaseUrl, serviceRole, { auth: { persistSession: false, autoRefreshToken: false } });
  if (body.operation === "visual_reference") {
    if (!uuid.test(String(body.question_id)) || typeof body.visual_id !== "string" || !body.visual_id.trim() || body.visual_id.length > 160
        || ![body.board_id, body.class_id, body.subject_id].every((item) => typeof item === "string" && scope.test(item))) return fail("INVALID_REQUEST", 400);
    const { data, error } = await client.rpc("taleem_test_paper_visual_reference", {
      p_question_id: body.question_id,
      p_visual_id: body.visual_id,
      p_board_id: body.board_id,
      p_class_id: body.class_id,
      p_subject_id: body.subject_id,
    });
    if (error) return fail("TEST_GENERATOR_UNAVAILABLE", 503);
    return data ? response(data) : fail("VISUAL_NOT_FOUND", 404);
  }
  if ((body.mode !== "board" && body.mode !== "custom")
      || ![body.board_id, body.class_id, body.subject_id].every((item) => typeof item === "string" && scope.test(item))) return fail("INVALID_REQUEST", 400);
  if (body.mode === "board" ? ("spec" in body) : !isObject(body.spec)) return fail("INVALID_REQUEST", 400);
  const { data, error } = await client.rpc("taleem_generate_test_paper", {
    p_mode: body.mode, p_board_id: body.board_id, p_class_id: body.class_id,
    p_subject_id: body.subject_id, p_spec: body.mode === "custom" ? body.spec : null, p_seed: body.seed,
  });
  if (error) {
    const message = error.message || "";
    const code = ["NO_ACTIVE_BLUEPRINT", "INVALID_CUSTOM_SPEC", "INSUFFICIENT_QUESTION_BANK", "INVALID_REQUEST"].find((item) => message.includes(item));
    return fail(code ?? "TEST_GENERATOR_UNAVAILABLE", code ? 400 : 503);
  }
  return response(data);
});
