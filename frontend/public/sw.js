// Service Worker v3 - Force cache invalidation on deployment
const CACHE_NAME = "finance-platform-v3-" + Date.now();

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          return caches.delete(key);
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  // Explicitly bypass service-worker interception for all API requests (/api/v1, /api/backend, etc.)
  try {
    const url = new URL(event.request.url);
    if (url.pathname.startsWith("/api/")) {
      return;
    }
  } catch (_err) {
    // If URL parsing fails, pass through to native network
    return;
  }

  // Always fetch live network first for static assets
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
