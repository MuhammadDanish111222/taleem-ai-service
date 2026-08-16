import { assertEquals, assertRejects } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { exportSPKI, generateKeyPair, SignJWT } from "https://deno.land/x/jose@v5.9.6/index.ts";
import { verifyTestGeneratorJwt } from "./auth.ts";

const { privateKey, publicKey } = await generateKeyPair("RS256");
const publicPem = await exportSPKI(publicKey);
Deno.env.set("TALEEM_INTERNAL_JWT_PUBLIC_KEY", publicPem);
Deno.env.set("TALEEM_INTERNAL_JWT_KEY_ID", "test-kid");
async function token(overrides: { uid?: string; feature?: string; audience?: string; issuer?: string } = {}) {
  return new SignJWT({ uid: overrides.uid ?? "student", feature: overrides.feature ?? "test_generator", request_id: "request" })
    .setProtectedHeader({ alg: "RS256", kid: "test-kid" }).setIssuer(overrides.issuer ?? "taleem-web").setAudience(overrides.audience ?? "taleem-test-generator")
    .setIssuedAt().setJti("jti").setExpirationTime("60s").sign(privateKey);
}
Deno.test("accepts only a valid dedicated test-generator JWT", async () => {
  assertEquals((await verifyTestGeneratorJwt(`Bearer ${await token()}`)).uid, "student");
});
for (const [name, overrides] of [["wrong audience", { audience: "taleem-ai-service" }], ["wrong issuer", { issuer: "bad" }], ["wrong feature", { feature: "ask" }], ["missing uid", { uid: "" }]] as const) {
  Deno.test(`rejects ${name}`, async () => assertRejects(() => verifyTestGeneratorJwt(`Bearer ${await token(overrides)}`)));
}
Deno.test("rejects malformed and expired tokens", async () => {
  await assertRejects(() => verifyTestGeneratorJwt("Bearer malformed"));
  const expired = new SignJWT({ uid: "student", feature: "test_generator", request_id: "request" })
    .setProtectedHeader({ alg: "RS256", kid: "test-kid" }).setIssuer("taleem-web").setAudience("taleem-test-generator")
    .setIssuedAt(1).setJti("jti").setExpirationTime(2).sign(privateKey);
  await assertRejects(() => verifyTestGeneratorJwt(`Bearer ${await expired}`));
});
Deno.test("rejects a token signed by another private key", async () => {
  const forged = await generateKeyPair("RS256");
  const forgedToken = await new SignJWT({ uid: "student", feature: "test_generator", request_id: "request" })
    .setProtectedHeader({ alg: "RS256", kid: "test-kid" }).setIssuer("taleem-web").setAudience("taleem-test-generator")
    .setIssuedAt().setJti("jti").setExpirationTime("60s").sign(forged.privateKey);
  await assertRejects(() => verifyTestGeneratorJwt(`Bearer ${forgedToken}`));
});
