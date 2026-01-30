const CACHE_NAME = 'cra-web-v14';

// Resources to cache immediately on install
const PRECACHE_RESOURCES = [
    './',
    './index.html',
    './app.js',
    './worker.js',
    './wasm/dither.js',
    './wasm/dither_bg.wasm',
    './assets/forest_plain.png',
    './assets/flowers_golden.png',
    './scripts/color_correction_basic.py',
    './scripts/color_correction_basic_rgb.py',
    './scripts/color_correction_cra.py',
    './scripts/color_correction_cra_rgb.py',
    './scripts/color_correction_tiled.py'
];

// Install: precache essential resources
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(PRECACHE_RESOURCES))
            .then(() => self.skipWaiting())
    );
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch: cache-first for static assets, network-first for CDN resources
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // For Pyodide CDN resources, use cache-first with network fallback
    if (url.hostname === 'cdn.jsdelivr.net') {
        event.respondWith(
            caches.open(CACHE_NAME).then((cache) => {
                return cache.match(event.request).then((cachedResponse) => {
                    if (cachedResponse) {
                        return cachedResponse;
                    }
                    return fetch(event.request).then((networkResponse) => {
                        // Cache the response for future use
                        if (networkResponse.ok) {
                            cache.put(event.request, networkResponse.clone());
                        }
                        return networkResponse;
                    });
                });
            })
        );
        return;
    }

    // For local resources, use cache-first
    if (url.origin === self.location.origin) {
        event.respondWith(
            caches.match(event.request).then((cachedResponse) => {
                if (cachedResponse) {
                    return cachedResponse;
                }
                return fetch(event.request).then((networkResponse) => {
                    // Cache successful responses
                    if (networkResponse.ok) {
                        const responseClone = networkResponse.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return networkResponse;
                });
            })
        );
    }
});
