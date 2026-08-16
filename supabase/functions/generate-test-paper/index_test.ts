import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { exportSPKI, generateKeyPair, SignJWT } from "https://deno.land/x/jose@v5.9.6/index.ts";
import { handler } from "./index.ts";

const { privateKey, publicKey } = await generateKeyPair("RS256");
const publicPem = await exportSPKI(publicKey);
Deno.env.set("TALEEM_INTERNAL_JWT_PUBLIC_KEY", publicPem);
Deno.env.set("TALEEM_INTERNAL_JWT_KEY_ID", "test-kid");

async function createAuthToken(requestId = "req-123") {
  return new SignJWT({ uid: "student-1", feature: "test_generator", request_id: requestId })
    .setProtectedHeader({ alg: "RS256", kid: "test-kid" })
    .setIssuer("taleem-web")
    .setAudience("taleem-test-generator")
    .setIssuedAt()
    .setJti("jti-123")
    .setExpirationTime("60s")
    .sign(privateKey);
}

function createMockClient(options: {
  featureState?: string | null;
  featureStateError?: unknown;
  paperResult?: unknown;
  paperError?: unknown;
  visualResult?: unknown;
  visualError?: unknown;
  onPaperRpc?: () => void;
  onVisualRpc?: () => void;
}) {
  return {
    from: () => ({
      insert: () => Promise.resolve({ data: null, error: null }),
    }),
    rpc: (name: string, args: unknown) => {
      if (name === "taleem_runtime_feature_state") {
        if (options.featureStateError) return Promise.resolve({ data: null, error: options.featureStateError });
        return Promise.resolve({ data: options.featureState ?? "enabled", error: null });
      }
      if (name === "taleem_generate_test_paper") {
        options.onPaperRpc?.();
        if (options.paperError) return Promise.resolve({ data: null, error: options.paperError });
        return Promise.resolve({ data: options.paperResult ?? { mode: "board", sections: [] }, error: null });
      }
      if (name === "taleem_test_paper_visual_reference") {
        options.onVisualRpc?.();
        if (options.visualError) return Promise.resolve({ data: null, error: options.visualError });
        return Promise.resolve({ data: options.visualResult ?? { storage_key: "key1" }, error: null });
      }
      return Promise.resolve({ data: null, error: new Error(`Unknown RPC ${name}`) });
    },
  };
}

Deno.test("feature_state operation returns feature state enum", async () => {
  const token = await createAuthToken("req-1");
  const client = createMockClient({ featureState: "coming_soon" });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ operation: "feature_state", seed: "req-1" }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 200);
  const json = await res.json();
  assertEquals(json, { feature: "test_generation", state: "coming_soon" });
});

Deno.test("feature_state operation returns 503 on RPC failure", async () => {
  const token = await createAuthToken("req-1");
  const client = createMockClient({ featureStateError: new Error("DB down") });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ operation: "feature_state", seed: "req-1" }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 503);
  const json = await res.json();
  assertEquals(json.error.code, "TEST_GENERATOR_UNAVAILABLE");
});

Deno.test("paper generation returns 404 when feature is disabled without calling paper RPC", async () => {
  let paperRpcCalled = false;
  const token = await createAuthToken("req-1");
  const client = createMockClient({
    featureState: "disabled",
    onPaperRpc: () => { paperRpcCalled = true; },
  });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ mode: "board", board_id: "punjab", class_id: "class-9", subject_id: "physics", seed: "req-1" }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 404);
  assertEquals(paperRpcCalled, false);
  const json = await res.json();
  assertEquals(json.error.code, "NOT_FOUND");
});

Deno.test("paper generation returns 409 when feature is coming_soon without calling paper RPC", async () => {
  let paperRpcCalled = false;
  const token = await createAuthToken("req-1");
  const client = createMockClient({
    featureState: "coming_soon",
    onPaperRpc: () => { paperRpcCalled = true; },
  });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ mode: "board", board_id: "punjab", class_id: "class-9", subject_id: "physics", seed: "req-1" }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 409);
  assertEquals(paperRpcCalled, false);
  const json = await res.json();
  assertEquals(json.error.code, "FEATURE_COMING_SOON");
});

Deno.test("paper generation proceeds to paper RPC when feature is enabled", async () => {
  let paperRpcCalled = false;
  const token = await createAuthToken("req-1");
  const client = createMockClient({
    featureState: "enabled",
    paperResult: { mode: "board", sections: [{ title: "Section A" }] },
    onPaperRpc: () => { paperRpcCalled = true; },
  });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ mode: "board", board_id: "punjab", class_id: "class-9", subject_id: "physics", seed: "req-1" }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 200);
  assertEquals(paperRpcCalled, true);
});

Deno.test("visual reference returns 404 when feature is disabled without calling visual RPC", async () => {
  let visualRpcCalled = false;
  const token = await createAuthToken("req-1");
  const client = createMockClient({
    featureState: "disabled",
    onVisualRpc: () => { visualRpcCalled = true; },
  });
  const req = new Request("http://localhost/generate-test-paper", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({
      operation: "visual_reference",
      question_id: "123e4567-e89b-42d3-a456-426614174000",
      visual_id: "benzene",
      board_id: "punjab",
      class_id: "class-9",
      subject_id: "chemistry",
      seed: "req-1",
    }),
  });

  const res = await handler(req, client);
  assertEquals(res.status, 404);
  assertEquals(visualRpcCalled, false);
});
