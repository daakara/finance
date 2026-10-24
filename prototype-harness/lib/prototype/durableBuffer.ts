/**
 * ARX Terminal Prototype Sandbox — Durable Browser Journal & Ephemeral Staging
 * Defect: ARX-UX-P00-INSTR-001
 * Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
 * Governing Authorization: RO-DEC-2026-10-23-012
 *
 * Cryptographic & Durability Invariants:
 * 1. Synchronous IndexedDB commit per event (survives page reload, crash, and restart).
 * 2. Web Crypto API AES-GCM-256 non-extractable staging key stored in IndexedDB.
 * 3. Bounded 4-hour staging TTL with automated purge.
 * 4. Zero-fill buffer wiping on participant withdrawal or verified sealed acceptance.
 * 5. Full journal recovery across page reload and browser restart.
 */

import { InteractionEvent, TaskSessionRecord } from "./HarnessLogger";

const DB_NAME = "ARX_RESEARCH_JOURNAL_V1";
const DB_VERSION = 1;
const STAGING_TTL_MS = 4 * 60 * 60 * 1000; // 4 hours

interface EncryptedJournalEntry {
  id: string;
  entry_type: "EVENT" | "TRIAL";
  seq: number;
  timestamp: number;
  iv: Uint8Array;
  ciphertext: ArrayBuffer;
}

interface StoredKeyEntry {
  id: string;
  key: CryptoKey;
  issued_at: number;
}

interface StoredSessionState {
  key: string;
  session_id: string;
  participant_id: string;
  session_token: string | null;
  status: string;
  started_at: number;
}

export class DurableBuffer {
  private db: IDBDatabase | null = null;
  private stagingKey: CryptoKey | null = null;
  private activeSessionId: string = "";
  private activeParticipantId: string = "";
  private sessionToken: string | null = null;

  async initBuffer(
    sessionId: string,
    participantId: string,
    bootstrapToken?: string
  ): Promise<{
    sessionToken: string | null;
    recovered: boolean;
    sessionId?: string;
    participantId?: string;
  }> {
    this.activeSessionId = sessionId;
    this.activeParticipantId = participantId;

    if (typeof window === "undefined" || !window.indexedDB) {
      return { sessionToken: null, recovered: false };
    }

    this.db = await this.openDatabase();
    await this.initStagingKey();

    // Check for existing session token or execute handshake if bootstrap token provided
    const existingState = await this.loadSessionState();
    if (existingState && existingState.session_token) {
      this.sessionToken = existingState.session_token;
      this.activeSessionId = existingState.session_id;
      this.activeParticipantId = existingState.participant_id;
      return {
        sessionToken: this.sessionToken,
        recovered: true,
        sessionId: existingState.session_id,
        participantId: existingState.participant_id,
      };
    }

    if (bootstrapToken) {
      const token = await this.performHandshake(bootstrapToken, sessionId, participantId);
      this.sessionToken = token;
      await this.saveSessionState({
        key: "active_session",
        session_id: sessionId,
        participant_id: participantId,
        session_token: token,
        status: "ACTIVE",
        started_at: Date.now(),
      });
      return { sessionToken: this.sessionToken, recovered: false };
    }

    return { sessionToken: null, recovered: false };
  }

  getSessionToken(): string | null {
    return this.sessionToken;
  }

  async setSessionToken(token: string) {
    this.sessionToken = token;
    await this.saveSessionState({
      key: "active_session",
      session_id: this.activeSessionId,
      participant_id: this.activeParticipantId,
      session_token: token,
      status: "ACTIVE",
      started_at: Date.now(),
    });
  }

  private openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains("crypto_keys")) {
          db.createObjectStore("crypto_keys", { keyPath: "id" });
        }
        if (!db.objectStoreNames.contains("events_journal")) {
          db.createObjectStore("events_journal", { keyPath: "id" });
        }
        if (!db.objectStoreNames.contains("session_state")) {
          db.createObjectStore("session_state", { keyPath: "key" });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private async initStagingKey(): Promise<void> {
    if (!this.db || !window.crypto || !window.crypto.subtle) return;

    const existing = await this.getKeyEntry();
    const now = Date.now();

    if (existing) {
      if (now - existing.issued_at > STAGING_TTL_MS) {
        // Expired staging key: purge and throw recovery failure
        await this.purgeActiveJournal(true);
        throw new Error("[SECURITY_VIOLATION] Staging buffer TTL expired (>4h). Journal safely purged.");
      }
      this.stagingKey = existing.key;
      return;
    }

    // Generate fresh non-extractable 256-bit AES-GCM key
    const newKey = await window.crypto.subtle.generateKey(
      { name: "AES-GCM", length: 256 },
      false, // non-extractable: raw key bytes cannot be read by JS
      ["encrypt", "decrypt"]
    );

    this.stagingKey = newKey;
    await this.saveKeyEntry({
      id: "staging_key",
      key: newKey,
      issued_at: now,
    });
  }

  private getKeyEntry(): Promise<StoredKeyEntry | null> {
    return new Promise((resolve, reject) => {
      if (!this.db) return resolve(null);
      const tx = this.db.transaction("crypto_keys", "readonly");
      const store = tx.objectStore("crypto_keys");
      const req = store.get("staging_key");
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  private saveKeyEntry(entry: StoredKeyEntry): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) return resolve();
      const tx = this.db.transaction("crypto_keys", "readwrite");
      const store = tx.objectStore("crypto_keys");
      const req = store.put(entry);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  private loadSessionState(): Promise<StoredSessionState | null> {
    return new Promise((resolve, reject) => {
      if (!this.db) return resolve(null);
      const tx = this.db.transaction("session_state", "readonly");
      const store = tx.objectStore("session_state");
      const req = store.get("active_session");
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  private saveSessionState(state: StoredSessionState): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) return resolve();
      const tx = this.db.transaction("session_state", "readwrite");
      const store = tx.objectStore("session_state");
      const req = store.put(state);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async performHandshake(
    bootstrapToken: string,
    sessionId: string,
    participantId: string
  ): Promise<string> {
    const res = await fetch("/api/research/custody/handshake", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        bootstrap_token: bootstrapToken,
        session_id: sessionId,
        participant_id: participantId,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Handshake failed with HTTP ${res.status}`);
    }

    const data = await res.json();
    this.sessionToken = data.session_token;
    await this.saveSessionState({
      key: "active_session",
      session_id: sessionId,
      participant_id: participantId,
      session_token: data.session_token,
      status: "ACTIVE",
      started_at: Date.now(),
    });
    return data.session_token;
  }

  async storeEvent(event: InteractionEvent, trialIdx: number): Promise<void> {
    if (!this.db || !this.stagingKey || !window.crypto || !window.crypto.subtle) return;

    const iv = window.crypto.getRandomValues(new Uint8Array(12));
    const enc = new TextEncoder();
    const plaintext = enc.encode(JSON.stringify(event));

    const ciphertext = await window.crypto.subtle.encrypt(
      { name: "AES-GCM", iv: iv as any },
      this.stagingKey,
      plaintext
    );

    const entry: EncryptedJournalEntry = {
      id: `EVT-${trialIdx}-${event.event_seq}-${Date.now()}`,
      entry_type: "EVENT",
      seq: event.event_seq,
      timestamp: Date.now(),
      iv,
      ciphertext,
    };

    return new Promise((resolve, reject) => {
      if (!this.db) return resolve();
      const tx = this.db.transaction("events_journal", "readwrite");
      const store = tx.objectStore("events_journal");
      store.put(entry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async storeTrialRecord(record: TaskSessionRecord): Promise<void> {
    if (!this.db || !this.stagingKey || !window.crypto || !window.crypto.subtle) return;

    const iv = window.crypto.getRandomValues(new Uint8Array(12));
    const enc = new TextEncoder();
    const plaintext = enc.encode(JSON.stringify(record));

    const ciphertext = await window.crypto.subtle.encrypt(
      { name: "AES-GCM", iv: iv as any },
      this.stagingKey,
      plaintext
    );

    const entry: EncryptedJournalEntry = {
      id: `TRIAL-${record.sequence_position}-${Date.now()}`,
      entry_type: "TRIAL",
      seq: record.sequence_position,
      timestamp: Date.now(),
      iv,
      ciphertext,
    };

    return new Promise((resolve, reject) => {
      if (!this.db) return resolve();
      const tx = this.db.transaction("events_journal", "readwrite");
      const store = tx.objectStore("events_journal");
      store.put(entry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async recoverActiveJournal(): Promise<{
    events: InteractionEvent[];
    trialRecords: TaskSessionRecord[];
  }> {
    if (!this.db || !this.stagingKey || !window.crypto || !window.crypto.subtle) {
      return { events: [], trialRecords: [] };
    }

    const entries: EncryptedJournalEntry[] = await new Promise((resolve, reject) => {
      if (!this.db) return resolve([]);
      const tx = this.db.transaction("events_journal", "readonly");
      const store = tx.objectStore("events_journal");
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });

    const dec = new TextDecoder();
    const events: InteractionEvent[] = [];
    const trialRecords: TaskSessionRecord[] = [];

    for (const entry of entries) {
      try {
        const decrypted = await window.crypto.subtle.decrypt(
          { name: "AES-GCM", iv: entry.iv as any },
          this.stagingKey,
          entry.ciphertext
        );
        const parsed = JSON.parse(dec.decode(decrypted));
        if (entry.entry_type === "TRIAL") {
          trialRecords.push(parsed as TaskSessionRecord);
        } else {
          events.push(parsed as InteractionEvent);
        }
      } catch (err) {
        console.error("Journal decryption failure on entry:", entry.id, err);
      }
    }

    // Sort by sequence number
    events.sort((a, b) => a.event_seq - b.event_seq);
    trialRecords.sort((a, b) => a.sequence_position - b.sequence_position);

    return { events, trialRecords };
  }

  async purgeActiveJournal(zeroFill = true): Promise<void> {
    if (!this.db) return;

    if (zeroFill) {
      // Overwrite journal records with zeroed buffers before dropping
      await new Promise<void>((resolve, reject) => {
        if (!this.db) return resolve();
        const tx = this.db.transaction("events_journal", "readwrite");
        const store = tx.objectStore("events_journal");
        const req = store.getAll();
        req.onsuccess = () => {
          const allEntries: EncryptedJournalEntry[] = req.result || [];
          for (const item of allEntries) {
            const zeroIv = new Uint8Array(item.iv.length).fill(0);
            const zeroCipher = new Uint8Array(item.ciphertext.byteLength).fill(0).buffer;
            store.put({
              ...item,
              iv: zeroIv,
              ciphertext: zeroCipher,
            });
          }
        };
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    }

    // Clear all stores
    await new Promise<void>((resolve, reject) => {
      if (!this.db) return resolve();
      const tx = this.db.transaction(["events_journal", "crypto_keys", "session_state"], "readwrite");
      tx.objectStore("events_journal").clear();
      tx.objectStore("crypto_keys").clear();
      tx.objectStore("session_state").clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });

    this.stagingKey = null;
    this.sessionToken = null;
  }
}
