import { importSPKI, jwtVerify } from "https://deno.land/x/jose@v5.9.6/index.ts";

export type TestGeneratorClaims = { uid: string; request_id: string };

export async function verifyTestGeneratorJwt(authorization: string | null): Promise<TestGeneratorClaims> {
  if (!authorization?.startsWith("Bearer ")) throw new Error("AUTH_INVALID_TOKEN");
  const pem = Deno.env.get("TALEEM_INTERNAL_JWT_PUBLIC_KEY")?.replace(/\\n/g, "\n");
  const kid = Deno.env.get("TALEEM_INTERNAL_JWT_KEY_ID");
  if (!pem || !kid) throw new Error("AUTH_INVALID_TOKEN");
  const token = authorization.slice(7);
  const key = await importSPKI(pem, "RS256");
  const { payload, protectedHeader } = await jwtVerify(token, key, {
    algorithms: ["RS256"], issuer: "taleem-web", audience: "taleem-test-generator", maxTokenAge: "60s",
  });
  if (protectedHeader.alg !== "RS256" || protectedHeader.kid !== kid
      || typeof payload.uid !== "string" || !payload.uid.trim()
      || payload.feature !== "test_generator"
      || typeof payload.request_id !== "string" || !payload.request_id.trim()
      || typeof payload.jti !== "string" || !payload.jti.trim()
      || typeof payload.iat !== "number" || typeof payload.exp !== "number"
      || payload.exp <= payload.iat || payload.exp - payload.iat > 60) {
    throw new Error("AUTH_INVALID_TOKEN");
  }
  return { uid: payload.uid, request_id: payload.request_id };
}
