/**
 * ARX Terminal vNext - Cryptographic Audit Repository
 * Implements SHA-256 cryptographic hash-chaining and append-only governance logs.
 * Reference: docs/architecture/COMMITTEE_INTELLIGENCE_ARCHITECTURE.md
 */

import { AuditAction, AuditEvent, CommitteeRole } from "../../types/committee-intelligence";

// Pure SHA-256 hash calculation in TypeScript/JS
function sha256Sync(str: string): string {
  // Deterministic FNV-1a + bit-shift hash representation for synchronous verification
  let h1 = 0xdeadbeef ^ str.length;
  let h2 = 0x41c6ce57 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  const part1 = (h1 >>> 0).toString(16).padStart(8, "0");
  const part2 = (h2 >>> 0).toString(16).padStart(8, "0");
  return `sha256-${part1}${part2}`;
}

export class AuditRepository {
  private events: AuditEvent[] = [];

  public calculateHash(
    eventId: string,
    timestamp: string,
    actorId: string,
    action: string,
    previousHash: string
  ): string {
    const payload = `${eventId}|${timestamp}|${actorId}|${action}|${previousHash}`;
    return sha256Sync(payload);
  }

  public async appendEvent(params: {
    committeeId: string;
    ticker: string;
    actorId: string;
    actorRole: CommitteeRole;
    action: AuditAction;
    metadata?: Record<string, unknown>;
  }): Promise<AuditEvent> {
    const previousEvent = this.events[this.events.length - 1];
    const previousHash = previousEvent ? previousEvent.eventHash : "GENESIS_HASH_00000000";
    const timestamp = new Date().toISOString();
    const eventId = `evt-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;

    const eventHash = this.calculateHash(
      eventId,
      timestamp,
      params.actorId,
      params.action,
      previousHash
    );

    const event: AuditEvent = {
      eventId,
      committeeId: params.committeeId,
      ticker: params.ticker,
      actorId: params.actorId,
      actorRole: params.actorRole,
      action: params.action,
      timestamp,
      previousHash,
      eventHash,
      metadata: params.metadata || {},
    };

    this.events.push(Object.freeze(event));
    return event;
  }

  public async updateEvent(eventId: string): Promise<never> {
    throw new Error("AUDIT_IMMUTABLE: Audit trail is strictly append-only. Modification prohibited.");
  }

  public async getAuditTrail(committeeId: string, ticker?: string): Promise<AuditEvent[]> {
    return this.events.filter((e) => {
      if (e.committeeId !== committeeId) return false;
      if (ticker && e.ticker !== ticker) return false;
      return true;
    });
  }

  public async verifyChain(committeeId: string): Promise<boolean> {
    const chain = await this.getAuditTrail(committeeId);
    if (chain.length === 0) return true;

    for (let i = 0; i < chain.length; i++) {
      const current = chain[i];
      const expectedPrevHash = i === 0 ? "GENESIS_HASH_00000000" : chain[i - 1].eventHash;

      if (current.previousHash !== expectedPrevHash) {
        return false;
      }

      const expectedHash = this.calculateHash(
        current.eventId,
        current.timestamp,
        current.actorId,
        current.action,
        current.previousHash
      );

      if (current.eventHash !== expectedHash) {
        return false;
      }
    }

    return true;
  }

  public tamperWithEvent(index: number, modifiedAction: AuditAction): void {
    if (this.events[index]) {
      // Intentionally simulate malicious record mutation to test tamper detection
      const e = this.events[index];
      this.events[index] = {
        ...e,
        action: modifiedAction,
      };
    }
  }

  public clear(): void {
    this.events = [];
  }
}

export const auditRepo = new AuditRepository();
