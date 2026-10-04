// Offline core: the shell is cached on install; API responses are served
// network-first and fall back to the last cached copy, so the card, the fan
// chart and the sell-or-wait sliders all work with no signal.
const CACHE = "sanjha-v3";
const SHELL = ["/", "/index.html", "/style.css", "/app.js", "/manifest.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        if (res.ok) caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(async () => {
        const hit = await caches.match(e.request);
        if (!hit) return Response.error();
        // flag it, so the page can say honestly that this is a saved copy
        const headers = new Headers(hit.headers);
        headers.set("X-Sanjha-Saved", "1");
        return new Response(await hit.blob(), { status: hit.status, headers });
      })
  );
});
