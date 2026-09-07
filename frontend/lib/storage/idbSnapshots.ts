/**
 * ARX Terminal vNext - Client-Owned IndexedDB Snapshot Storage
 * Source: ADR-002, ADR-004 & Sprint 3 Architecture
 */

import { TickerSnapshotRecord, TickerSnapshot } from "../../types/change-intelligence";

const DB_NAME = "arx_change_intelligence_db";
const DB_VERSION = 1;
const STORE_NAME = "ticker_snapshots";

// In-memory fallback for SSR and non-browser environments
const memoryStore = new Map<string, TickerSnapshotRecord>();

function isIndexedDBAvailable(): boolean {
  return typeof window !== "undefined" && typeof window.indexedDB !== "undefined";
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!isIndexedDBAvailable()) {
      reject(new Error("IndexedDB is unavailable in this environment."));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "ticker" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Retrieve a ticker's snapshot record
 */
export async function getTickerRecord(ticker: string): Promise<TickerSnapshotRecord | null> {
  const cleanTicker = ticker.toUpperCase().trim();

  if (!isIndexedDBAvailable()) {
    return memoryStore.get(cleanTicker) || null;
  }

  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(cleanTicker);

      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("[ChangeIntelligence] Falling back to memory store:", err);
    return memoryStore.get(cleanTicker) || null;
  }
}

/**
 * Save or update a ticker's snapshot record
 */
export async function saveTickerRecord(record: TickerSnapshotRecord): Promise<void> {
  const cleanTicker = record.ticker.toUpperCase().trim();
  const normalizedRecord = { ...record, ticker: cleanTicker };

  // Always update in-memory cache
  memoryStore.set(cleanTicker, normalizedRecord);

  if (!isIndexedDBAvailable()) {
    return;
  }

  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(normalizedRecord);

      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("[ChangeIntelligence] Failed to write to IndexedDB:", err);
  }
}

/**
 * 1-Click Thesis Acknowledgement:
 * Resets baselineSnapshot to current latestSnapshot, updates acknowledgedAt,
 * and clears the active deltaSummary.
 */
export async function acknowledgeBaseline(ticker: string): Promise<TickerSnapshotRecord | null> {
  const existing = await getTickerRecord(ticker);
  if (!existing) return null;

  const now = new Date().toISOString();
  const updatedRecord: TickerSnapshotRecord = {
    ...existing,
    baselineSnapshot: {
      ...existing.latestSnapshot,
      timestamp: now,
    },
    acknowledgedAt: now,
    deltaSummary: undefined, // Clear unacknowledged delta
  };

  await saveTickerRecord(updatedRecord);
  return updatedRecord;
}

/**
 * Retrieve all ticker records (used for Portfolio Attention Feed)
 */
export async function getAllTickerRecords(): Promise<TickerSnapshotRecord[]> {
  if (!isIndexedDBAvailable()) {
    return Array.from(memoryStore.values());
  }

  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();

      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("[ChangeIntelligence] Falling back to memory store:", err);
    return Array.from(memoryStore.values());
  }
}

/**
 * Delete a ticker's snapshot record
 */
export async function deleteTickerRecord(ticker: string): Promise<void> {
  const cleanTicker = ticker.toUpperCase().trim();
  memoryStore.delete(cleanTicker);

  if (!isIndexedDBAvailable()) return;

  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(cleanTicker);

      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("[ChangeIntelligence] Failed to delete record:", err);
  }
}
