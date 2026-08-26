"use client";

import { useEffect } from "react";

export default function ServiceWorkerRegister() {
  useEffect(() => {
    if (typeof window !== "undefined" && "serviceWorker" in navigator) {
      window.addEventListener("load", () => {
        navigator.serviceWorker
          .register("/sw.js")
          .then((reg) => {
            console.log("Finance Platform PWA Service Worker Registered:", reg.scope);
          })
          .catch((err) => {
            console.warn("Service Worker registration warning:", err);
          });
      });
    }
  }, []);

  return null;
}