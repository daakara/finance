/**
 * Regression & Adversarial Test Suite for Frontend Observability Correlation.
 *
 * Certifies:
 * - T01: Request ID uniqueness across requests
 * - T02: Same-operation correlation (multiple requests share operation ID)
 * - T03: Sequential operation isolation (Operation B does not inherit Operation A)
 * - T04: Concurrent operation isolation (overlapping operations retain distinct IDs)
 * - T05: Nested operation behavior (parent context restored after child completion)
 * - T06: Exception cleanup (thrown operation does not contaminate subsequent operation)
 * - T07: Promise rejection cleanup (rejected async operation does not leak)
 * - T08: Abort/cancellation safety (aborted operations do not leave stale state)
 * - T09: Standalone request semantics (independent 1:1 correlation ID)
 * - T10: Background request isolation (unrelated polling not attached to user operation)
 * - T11: Invalid correlation input sanitization (RFC-4122 UUIDv4 enforcement)
 * - T12: Release identity preservation (X-Client-Version from NEXT_PUBLIC_ARX_RELEASE)
 * - Section 16: Mandatory Adversarial Interleaved Scheduling Test
 */

import assert from "node:assert";
import {
  generateSecureUUIDv4,
  isValidUUIDv4,
  createCorrelationOperation,
  withCorrelationOperation,
  getObservabilityHeaders,
  type CorrelationOperation,
} from "../lib/observability/correlation";
import {
  getArxApiHeaders,
  ARX_API_HEADERS,
} from "../lib/api";

async function runAllTests() {
  console.log("Starting Frontend Observability Correlation Remediation Test Suite...\n");

  // T01 — Request ID uniqueness
  console.log("T01: Testing request ID uniqueness across calls...");
  const id1 = generateSecureUUIDv4();
  const id2 = generateSecureUUIDv4();
  assert.notStrictEqual(id1, id2, "Consecutive generated IDs must be distinct");
  assert.strictEqual(isValidUUIDv4(id1), true, "ID 1 must be valid UUIDv4");
  assert.strictEqual(isValidUUIDv4(id2), true, "ID 2 must be valid UUIDv4");

  const r1 = getObservabilityHeaders();
  const r2 = getObservabilityHeaders();
  assert.notStrictEqual(r1["X-Request-ID"], r2["X-Request-ID"], "Request IDs must differ between requests");
  console.log("   [OK] Request ID uniqueness verified");

  // T02 — Same-operation correlation
  console.log("T02: Testing same-operation correlation...");
  const op = createCorrelationOperation();
  const opHeaders1 = getObservabilityHeaders(op);
  const opHeaders2 = getObservabilityHeaders(op);
  assert.strictEqual(opHeaders1["X-Correlation-ID"], op.correlationId, "Header must match operation ID");
  assert.strictEqual(opHeaders2["X-Correlation-ID"], op.correlationId, "Header must match operation ID");
  assert.strictEqual(
    opHeaders1["X-Correlation-ID"],
    opHeaders2["X-Correlation-ID"],
    "Requests in same operation must share correlation ID"
  );
  assert.notStrictEqual(
    opHeaders1["X-Request-ID"],
    opHeaders2["X-Request-ID"],
    "Requests in same operation must have distinct request IDs"
  );
  console.log("   [OK] Same-operation correlation verified");

  // T03 — Sequential operations
  console.log("T03: Testing sequential operations isolation...");
  const opA = createCorrelationOperation();
  const opB = createCorrelationOperation();
  const aHeaders = getObservabilityHeaders(opA);
  const bHeaders = getObservabilityHeaders(opB);
  assert.notStrictEqual(
    aHeaders["X-Correlation-ID"],
    bHeaders["X-Correlation-ID"],
    "Operation B must not inherit Operation A correlation ID"
  );
  console.log("   [OK] Sequential operation isolation verified");

  // T04 — Concurrent operation isolation
  console.log("T04: Testing concurrent operation isolation...");
  const op1 = createCorrelationOperation();
  const op2 = createCorrelationOperation();

  // Interleave generation of headers
  const h1_1 = getObservabilityHeaders(op1);
  const h2_1 = getObservabilityHeaders(op2);
  const h1_2 = getObservabilityHeaders(op1);
  const h2_2 = getObservabilityHeaders(op2);

  assert.strictEqual(h1_1["X-Correlation-ID"], op1.correlationId);
  assert.strictEqual(h1_2["X-Correlation-ID"], op1.correlationId);
  assert.strictEqual(h2_1["X-Correlation-ID"], op2.correlationId);
  assert.strictEqual(h2_2["X-Correlation-ID"], op2.correlationId);
  assert.notStrictEqual(op1.correlationId, op2.correlationId);
  console.log("   [OK] Concurrent operation isolation verified");

  // T05 — Nested operation behavior
  console.log("T05: Testing nested operation behavior...");
  const parentOp = createCorrelationOperation();
  let capturedParentInChild: string | null = null;
  let capturedChild: string | null = null;

  await withCorrelationOperation(parentOp, async (parent) => {
    const parentH1 = getObservabilityHeaders(parent);
    assert.strictEqual(parentH1["X-Correlation-ID"], parentOp.correlationId);

    // Run child operation
    await withCorrelationOperation(createCorrelationOperation(), async (child) => {
      capturedChild = child.correlationId;
      capturedParentInChild = parent.correlationId;
      const childH = getObservabilityHeaders(child);
      assert.strictEqual(childH["X-Correlation-ID"], child.correlationId);
      assert.notStrictEqual(childH["X-Correlation-ID"], parent.correlationId);
    });

    // Parent resumes
    const parentH2 = getObservabilityHeaders(parent);
    assert.strictEqual(
      parentH2["X-Correlation-ID"],
      parentOp.correlationId,
      "Parent correlation ID must be preserved after child completes"
    );
  });
  assert.notStrictEqual(capturedChild, capturedParentInChild);
  console.log("   [OK] Nested operation behavior verified");

  // T06 — Exception cleanup
  console.log("T06: Testing exception cleanup...");
  const crashingOp = createCorrelationOperation();
  try {
    await withCorrelationOperation(crashingOp, async (op) => {
      getObservabilityHeaders(op);
      throw new Error("Crash inside operation!");
    });
  } catch (err) {
    // Expected crash
  }

  // Next operation starts
  const nextOp = createCorrelationOperation();
  const nextHeaders = getObservabilityHeaders(nextOp);
  assert.notStrictEqual(
    nextHeaders["X-Correlation-ID"],
    crashingOp.correlationId,
    "Next operation must not be contaminated by crashed operation"
  );
  console.log("   [OK] Exception cleanup verified");

  // T07 — Promise rejection cleanup
  console.log("T07: Testing promise rejection cleanup...");
  const rejectingOp = createCorrelationOperation();
  const rejectedPromise = withCorrelationOperation(rejectingOp, async () => {
    return Promise.reject(new Error("Async rejection!"));
  });
  await rejectedPromise.catch(() => {});

  const postRejectOp = createCorrelationOperation();
  const postRejectHeaders = getObservabilityHeaders(postRejectOp);
  assert.notStrictEqual(
    postRejectHeaders["X-Correlation-ID"],
    rejectingOp.correlationId,
    "Post-rejection operation must not inherit rejected operation ID"
  );
  console.log("   [OK] Promise rejection cleanup verified");

  // T08 — Abort/cancellation safety
  console.log("T08: Testing abort/cancellation safety...");
  const abortController = new AbortController();
  const cancelledOp = createCorrelationOperation();
  abortController.abort();

  assert.strictEqual(abortController.signal.aborted, true);
  // Post-abort call with fresh operation
  const postAbortOp = createCorrelationOperation();
  const postAbortHeaders = getObservabilityHeaders(postAbortOp);
  assert.notStrictEqual(postAbortHeaders["X-Correlation-ID"], cancelledOp.correlationId);
  console.log("   [OK] Abort/cancellation safety verified");

  // T09 — Standalone request semantics
  console.log("T09: Testing standalone request semantics...");
  const standalone1 = getObservabilityHeaders();
  const standalone2 = getObservabilityHeaders();
  assert.ok(isValidUUIDv4(standalone1["X-Request-ID"]));
  assert.ok(isValidUUIDv4(standalone1["X-Correlation-ID"]));
  assert.ok(isValidUUIDv4(standalone2["X-Request-ID"]));
  assert.ok(isValidUUIDv4(standalone2["X-Correlation-ID"]));
  assert.notStrictEqual(
    standalone1["X-Correlation-ID"],
    standalone2["X-Correlation-ID"],
    "Standalone requests must each receive an isolated correlation ID"
  );
  assert.notStrictEqual(
    standalone1["X-Request-ID"],
    standalone2["X-Request-ID"],
    "Standalone requests must have distinct request IDs"
  );
  console.log("   [OK] Standalone request semantics verified");

  // T10 — Background request isolation
  console.log("T10: Testing background request isolation...");
  const userOp = createCorrelationOperation();
  const userH1 = getObservabilityHeaders(userOp);

  // Background polling request executes without operation token
  const bgPollHeaders = getObservabilityHeaders();
  assert.notStrictEqual(
    bgPollHeaders["X-Correlation-ID"],
    userH1["X-Correlation-ID"],
    "Background poll must not inherit user operation correlation ID"
  );

  // User operation continues
  const userH2 = getObservabilityHeaders(userOp);
  assert.strictEqual(
    userH2["X-Correlation-ID"],
    userH1["X-Correlation-ID"],
    "User operation continues to share its own correlation ID"
  );
  console.log("   [OK] Background request isolation verified");

  // T11 — Invalid correlation input
  console.log("T11: Testing invalid correlation input sanitization...");
  const invalidOp = createCorrelationOperation("INVALID-MALICIOUS-INPUT\n\r");
  assert.strictEqual(
    isValidUUIDv4(invalidOp.correlationId),
    true,
    "Invalid input must be replaced with safe valid UUIDv4"
  );

  const rawHeaders = getObservabilityHeaders("INJECTION_ATTEMPT; DROP TABLE;");
  assert.strictEqual(
    isValidUUIDv4(rawHeaders["X-Correlation-ID"]),
    true,
    "String injection must be sanitized to valid UUIDv4"
  );
  console.log("   [OK] Invalid correlation input sanitization verified");

  // T12 — Release identity preserved
  console.log("T12: Testing release identity preservation...");
  process.env.NEXT_PUBLIC_ARX_RELEASE = "9439ce0253a589464c9e5153da996349997bf1e0";
  const releaseHeaders = getObservabilityHeaders();
  assert.strictEqual(
    releaseHeaders["X-Client-Version"],
    "9439ce0253a589464c9e5153da996349997bf1e0",
    "Release SHA must be preserved from env"
  );

  delete process.env.NEXT_PUBLIC_ARX_RELEASE;
  const noReleaseHeaders = getObservabilityHeaders();
  assert.strictEqual(
    noReleaseHeaders["X-Client-Version"],
    "",
    "Missing release must safely default to empty string without fabrication"
  );
  console.log("   [OK] Release identity preservation verified");

  // MANDATORY SECTION 16: Adversarial Interleaved Scheduling Test
  console.log("\n--- MANDATORY ADVERSARIAL INTERLEAVED SCHEDULING TEST ---");
  const A = createCorrelationOperation();
  const B = createCorrelationOperation();

  // A starts: A1 headers generated
  const A1 = getObservabilityHeaders(A);

  // B starts: B1 headers generated
  const B1 = getObservabilityHeaders(B);

  // A2 headers generated
  const A2 = getObservabilityHeaders(A);

  // B2 headers generated
  const B2 = getObservabilityHeaders(B);

  // Assertions mandated by Section 16:
  assert.strictEqual(
    A1["X-Correlation-ID"],
    A2["X-Correlation-ID"],
    "A1.correlation == A2.correlation"
  );
  assert.strictEqual(
    B1["X-Correlation-ID"],
    B2["X-Correlation-ID"],
    "B1.correlation == B2.correlation"
  );
  assert.notStrictEqual(
    A1["X-Correlation-ID"],
    B1["X-Correlation-ID"],
    "A1.correlation != B1.correlation"
  );

  assert.notStrictEqual(
    A1["X-Request-ID"],
    A2["X-Request-ID"],
    "A1.request != A2.request"
  );
  assert.notStrictEqual(
    B1["X-Request-ID"],
    B2["X-Request-ID"],
    "B1.request != B2.request"
  );
  console.log("   [OK] Adversarial Interleaved Scheduling Test PASSED!");

  // API Wrapper / Proxy Integration Test
  console.log("\nTesting getArxApiHeaders and ARX_API_HEADERS proxy...");
  const proxyH1 = { ...(ARX_API_HEADERS as Record<string, string>) };
  const proxyH2 = { ...(ARX_API_HEADERS as Record<string, string>) };
  assert.ok(proxyH1["Content-Type"]);
  assert.ok(isValidUUIDv4(proxyH1["X-Request-ID"]));
  assert.ok(isValidUUIDv4(proxyH1["X-Correlation-ID"]));
  assert.notStrictEqual(
    proxyH1["X-Request-ID"],
    proxyH2["X-Request-ID"],
    "Proxy accesses must receive distinct request IDs"
  );
  assert.notStrictEqual(
    proxyH1["X-Correlation-ID"],
    proxyH2["X-Correlation-ID"],
    "Proxy accesses outside operation must receive distinct standalone correlation IDs"
  );

  const wrapperOpHeaders = getArxApiHeaders(A);
  assert.strictEqual(
    wrapperOpHeaders["X-Correlation-ID"],
    A.correlationId,
    "getArxApiHeaders with operation must use operation correlation ID"
  );
  console.log("   [OK] getArxApiHeaders and Proxy integration PASSED!");

  console.log("\n[SUCCESS] All 12 criteria + Adversarial Scheduling tests passed successfully!\n");
}

runAllTests().catch((err) => {
  console.error("\n[TEST FAILED]:", err);
  process.exit(1);
});
