import { test } from "node:test";
import assert from "node:assert/strict";
import { resolveUxPreviewDeployment } from "../next.config.mjs";
test("Pages preview locks without the optional review flag", () => {
  for (const flag of [undefined, "", "false", "invalid", "true"]) {
    assert.equal(resolveUxPreviewDeployment({ CF_PAGES: "1", CF_PAGES_BRANCH: "codex/ux-v3-final-completion-20261029", NEXT_PUBLIC_UX_REVIEW_CANDIDATE: flag }), "true");
  }
});
test("unknown Pages branch fails closed", () => {
  assert.equal(resolveUxPreviewDeployment({ CF_PAGES: "1" }), "true");
});
test("production build setting remains distinct", () => {
  assert.equal(resolveUxPreviewDeployment({ CF_PAGES: "1", CF_PAGES_BRANCH: "release/production" }), "false");
  assert.equal(resolveUxPreviewDeployment({}), "false");
});
