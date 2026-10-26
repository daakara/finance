/**
 * ARX Terminal Prototype Sandbox — Session Exporter Utility
 *
 * Packages session JSON Lines logs and SHA-256 digests into browser downloads.
 */

export function downloadSessionFiles(participantId: string, jsonlContent: string, sha256Digest: string) {
  if (typeof window === "undefined") return;

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

  // 1. Download JSONL file
  const jsonlBlob = new Blob([jsonlContent], { type: "application/x-jsonlines;charset=utf-8" });
  const jsonlUrl = URL.createObjectURL(jsonlBlob);
  const jsonlLink = document.createElement("a");
  jsonlLink.href = jsonlUrl;
  jsonlLink.download = `session_${participantId}_${timestamp}.jsonl`;
  document.body.appendChild(jsonlLink);
  jsonlLink.click();
  document.body.removeChild(jsonlLink);
  URL.revokeObjectURL(jsonlUrl);

  // 2. Download SHA-256 checksum file
  const sha256Text = `${sha256Digest}  session_${participantId}_${timestamp}.jsonl\n`;
  const shaBlob = new Blob([sha256Text], { type: "text/plain;charset=utf-8" });
  const shaUrl = URL.createObjectURL(shaBlob);
  const shaLink = document.createElement("a");
  shaLink.href = shaUrl;
  shaLink.download = `session_${participantId}_${timestamp}.sha256`;
  document.body.appendChild(shaLink);
  shaLink.click();
  document.body.removeChild(shaLink);
  URL.revokeObjectURL(shaUrl);
}
