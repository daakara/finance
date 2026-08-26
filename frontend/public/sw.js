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
  // Always fetch live network first
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});