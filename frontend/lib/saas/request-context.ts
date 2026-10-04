/**
 * Frontend RequestContext data contract for ARX SaaS Foundation.
 *
 * Contains strictly application-layer identity and tenancy carriers:
 * - actorId: string | null (null represents an anonymous actor)
 * - workspaceId: string
 * - requestId: string
 *
 * Contains zero commercial plan names, prices, or commercial identifiers.
 */

export interface RequestContext {
  readonly actorId: string | null;
  readonly workspaceId: string;
  readonly requestId: string;
}

/**
 * Construct a neutral default RequestContext for frontend usage.
 */
export function createDefaultRequestContext(requestId?: string): RequestContext {
  return {
    actorId: null,
    workspaceId: "ws_default",
    requestId: requestId && requestId.trim() ? requestId.trim() : "req_default",
  };
}
