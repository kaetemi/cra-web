/**
 * <linear-img> - A custom element that displays images using linear RGB resizing
 *
 * Unlike browser-native image scaling (which operates in sRGB space and causes
 * darkening/color shifts), this element uses proper linear RGB interpolation.
 *
 * Usage:
 *   <linear-img src="photo.jpg" width="200" height="150"></linear-img>
 *   <linear-img src="photo.jpg" style="width: 200px;"></linear-img>
 *
 * Attributes:
 *   src        - Image source URL (required)
 *   width      - Display width in pixels
 *   height     - Display height in pixels
 *   method     - Interpolation: "lanczos" (default) or "bilinear"
 *   fit        - Object-fit style: "contain" (default), "cover", "fill"
 *
 * The element shows a browser-scaled preview immediately, then swaps in
 * the correctly resized version when processing completes.
 */

// Shared worker instance for all <linear-img> elements
let sharedWorker = null;
let workerReady = false;
let workerReadyPromise = null;
let pendingRequests = new Map();
let requestId = 0;

// Image cache: caches resized results by "url:width:height:method"
const resultCache = new Map();
const MAX_CACHE_SIZE = 50;

// Initialize the shared worker
function initSharedWorker() {
    if (workerReadyPromise) return workerReadyPromise;

    workerReadyPromise = new Promise((resolve, reject) => {
        try {
            // Determine base path from this script's location
            const scriptUrl = new URL(import.meta.url);
            const basePath = scriptUrl.href.substring(0, scriptUrl.href.lastIndexOf('/') + 1);

            sharedWorker = new Worker(basePath + 'resize-worker.js', { type: 'module' });

            sharedWorker.onmessage = function(e) {
                const { type, ...data } = e.data;

                if (type === 'ready') {
                    workerReady = true;
                    resolve();
                } else if (type === 'complete' || type === 'error') {
                    // Find the pending request and resolve/reject it
                    const reqId = data.requestId;
                    const pending = pendingRequests.get(reqId);
                    if (pending) {
                        pendingRequests.delete(reqId);
                        if (type === 'complete') {
                            pending.resolve(data);
                        } else {
                            pending.reject(new Error(data.message));
                        }
                    }
                } else if (type === 'progress') {
                    const reqId = data.requestId;
                    const pending = pendingRequests.get(reqId);
                    if (pending && pending.onProgress) {
                        pending.onProgress(data.percent);
                    }
                }
            };

            sharedWorker.onerror = function(e) {
                reject(new Error('Worker failed to load: ' + e.message));
            };

            sharedWorker.postMessage({ type: 'init' });
        } catch (err) {
            reject(err);
        }
    });

    return workerReadyPromise;
}

// Request a resize operation from the worker
async function requestResize(fileBytes, srcWidth, srcHeight, dstWidth, dstHeight, method, onProgress) {
    await initSharedWorker();

    const id = ++requestId;
    const methodCode = method === 'bilinear' ? 0 : 1; // 0=bilinear, 1=lanczos

    return new Promise((resolve, reject) => {
        pendingRequests.set(id, { resolve, reject, onProgress });

        sharedWorker.postMessage({
            type: 'resize',
            requestId: id,
            fileBytes,
            dstWidth,
            dstHeight,
            interpolation: methodCode,
            scaleMode: 0, // Independent - we calculate exact dimensions ourselves
            ditherMode: 4, // Mixed
            ditherTechnique: 2, // ColorAware
            perceptualSpace: 1 // OKLab
        });
    });
}

// Fetch image as ArrayBuffer
async function fetchImageBytes(src) {
    const response = await fetch(src);
    if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status}`);
    }
    return await response.arrayBuffer();
}

// Get cache key
function getCacheKey(src, width, height, method) {
    return `${src}:${width}:${height}:${method}`;
}

// Add to cache with LRU eviction
function cacheResult(key, imageData) {
    if (resultCache.size >= MAX_CACHE_SIZE) {
        // Remove oldest entry
        const firstKey = resultCache.keys().next().value;
        resultCache.delete(firstKey);
    }
    resultCache.set(key, imageData);
}

// Custom element definition
class LinearImg extends HTMLElement {
    static get observedAttributes() {
        return ['src', 'width', 'height', 'method', 'fit'];
    }

    constructor() {
        super();

        // Create shadow DOM
        this.attachShadow({ mode: 'open' });

        // Internal state
        this._src = null;
        this._naturalWidth = 0;
        this._naturalHeight = 0;
        this._displayWidth = 0;
        this._displayHeight = 0;
        this._method = 'lanczos';
        this._fit = 'contain';
        this._imageBytes = null;
        this._processing = false;
        this._pendingResize = false;

        // Create internal elements
        this._container = document.createElement('div');
        this._container.style.cssText = 'position: relative; display: inline-block; overflow: hidden;';

        // Preview image (browser-scaled, shown while processing)
        this._preview = document.createElement('img');
        this._preview.style.cssText = 'display: block; width: 100%; height: 100%; object-fit: contain;';

        // Canvas for final output
        this._canvas = document.createElement('canvas');
        this._canvas.style.cssText = 'display: none; position: absolute; top: 0; left: 0;';

        this._container.appendChild(this._preview);
        this._container.appendChild(this._canvas);

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            :host {
                display: inline-block;
                overflow: hidden;
            }
            :host([hidden]) {
                display: none;
            }
        `;

        this.shadowRoot.appendChild(style);
        this.shadowRoot.appendChild(this._container);

        // ResizeObserver to detect size changes
        this._resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this._handleResize(Math.round(width), Math.round(height));
                }
            }
        });
    }

    connectedCallback() {
        this._resizeObserver.observe(this);
        this._updateFromAttributes();
    }

    disconnectedCallback() {
        this._resizeObserver.disconnect();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;

        switch (name) {
            case 'src':
                this._src = newValue;
                this._loadImage();
                break;
            case 'width':
                this._container.style.width = newValue ? newValue + 'px' : '';
                break;
            case 'height':
                this._container.style.height = newValue ? newValue + 'px' : '';
                break;
            case 'method':
                this._method = newValue === 'bilinear' ? 'bilinear' : 'lanczos';
                this._scheduleResize();
                break;
            case 'fit':
                this._fit = newValue || 'contain';
                this._preview.style.objectFit = this._fit;
                this._scheduleResize();
                break;
        }
    }

    _updateFromAttributes() {
        this._src = this.getAttribute('src');
        this._method = this.getAttribute('method') === 'bilinear' ? 'bilinear' : 'lanczos';
        this._fit = this.getAttribute('fit') || 'contain';
        this._preview.style.objectFit = this._fit;

        const width = this.getAttribute('width');
        const height = this.getAttribute('height');
        if (width) this._container.style.width = width + 'px';
        if (height) this._container.style.height = height + 'px';

        if (this._src) {
            this._loadImage();
        }
    }

    async _loadImage() {
        if (!this._src) return;

        // Show preview immediately using browser scaling
        this._preview.src = this._src;
        this._preview.style.display = 'block';
        this._canvas.style.display = 'none';

        try {
            // Wait for preview to load to get natural dimensions
            await new Promise((resolve, reject) => {
                if (this._preview.complete && this._preview.naturalWidth > 0) {
                    resolve();
                } else {
                    this._preview.onload = resolve;
                    this._preview.onerror = () => reject(new Error('Failed to load image'));
                }
            });

            this._naturalWidth = this._preview.naturalWidth;
            this._naturalHeight = this._preview.naturalHeight;

            // Fetch raw bytes for WASM processing
            this._imageBytes = await fetchImageBytes(this._src);

            // Trigger resize with current display dimensions
            const rect = this.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                this._handleResize(Math.round(rect.width), Math.round(rect.height));
            }

        } catch (err) {
            console.error('LinearImg: Failed to load image:', err);
            this.dispatchEvent(new CustomEvent('error', { detail: err }));
        }
    }

    _handleResize(width, height) {
        if (width === this._displayWidth && height === this._displayHeight) {
            return;
        }

        this._displayWidth = width;
        this._displayHeight = height;
        this._scheduleResize();
    }

    _scheduleResize() {
        if (!this._imageBytes || !this._displayWidth || !this._displayHeight) {
            return;
        }

        if (this._processing) {
            this._pendingResize = true;
            return;
        }

        this._doResize();
    }

    async _doResize() {
        if (!this._imageBytes || this._displayWidth <= 0 || this._displayHeight <= 0) {
            return;
        }

        // Calculate actual output dimensions based on fit mode
        const { outputWidth, outputHeight } = this._calculateOutputDimensions();

        // Skip if output would be same as natural size (no resize needed)
        if (outputWidth === this._naturalWidth && outputHeight === this._naturalHeight) {
            // Just show the preview at full res
            this._preview.style.display = 'block';
            this._canvas.style.display = 'none';
            return;
        }

        // Check cache
        const cacheKey = getCacheKey(this._src, outputWidth, outputHeight, this._method);
        const cached = resultCache.get(cacheKey);
        if (cached) {
            this._displayResult(cached, outputWidth, outputHeight);
            return;
        }

        this._processing = true;

        try {
            const result = await requestResize(
                this._imageBytes,
                this._naturalWidth,
                this._naturalHeight,
                outputWidth,
                outputHeight,
                this._method
            );

            const imageData = new ImageData(
                new Uint8ClampedArray(result.outputData),
                result.width,
                result.height
            );

            // Cache the result
            cacheResult(cacheKey, imageData);

            // Display if dimensions still match (user might have resized during processing)
            const { outputWidth: currentW, outputHeight: currentH } = this._calculateOutputDimensions();
            if (currentW === outputWidth && currentH === outputHeight) {
                this._displayResult(imageData, outputWidth, outputHeight);
            }

            this.dispatchEvent(new CustomEvent('load', {
                detail: { width: outputWidth, height: outputHeight }
            }));

        } catch (err) {
            console.error('LinearImg: Resize failed:', err);
            // Keep showing browser-scaled preview on error
        } finally {
            this._processing = false;

            // Process any pending resize
            if (this._pendingResize) {
                this._pendingResize = false;
                this._scheduleResize();
            }
        }
    }

    _calculateOutputDimensions() {
        const displayW = this._displayWidth;
        const displayH = this._displayHeight;
        const naturalW = this._naturalWidth;
        const naturalH = this._naturalHeight;

        if (!naturalW || !naturalH) {
            return { outputWidth: displayW, outputHeight: displayH };
        }

        const displayAspect = displayW / displayH;
        const naturalAspect = naturalW / naturalH;

        let outputWidth, outputHeight;

        switch (this._fit) {
            case 'fill':
                // Stretch to fill (ignores aspect ratio)
                outputWidth = displayW;
                outputHeight = displayH;
                break;

            case 'cover':
                // Scale to cover container (may crop)
                if (displayAspect > naturalAspect) {
                    outputWidth = displayW;
                    outputHeight = Math.round(displayW / naturalAspect);
                } else {
                    outputHeight = displayH;
                    outputWidth = Math.round(displayH * naturalAspect);
                }
                break;

            case 'contain':
            default:
                // Scale to fit within container (may letterbox)
                if (displayAspect > naturalAspect) {
                    outputHeight = displayH;
                    outputWidth = Math.round(displayH * naturalAspect);
                } else {
                    outputWidth = displayW;
                    outputHeight = Math.round(displayW / naturalAspect);
                }
                break;
        }

        // Ensure minimum size of 1
        outputWidth = Math.max(1, outputWidth);
        outputHeight = Math.max(1, outputHeight);

        return { outputWidth, outputHeight };
    }

    _displayResult(imageData, width, height) {
        this._canvas.width = width;
        this._canvas.height = height;

        const ctx = this._canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        // Position canvas based on fit mode
        const displayW = this._displayWidth;
        const displayH = this._displayHeight;

        if (this._fit === 'cover') {
            // Center the oversized canvas
            const left = Math.round((displayW - width) / 2);
            const top = Math.round((displayH - height) / 2);
            this._canvas.style.left = left + 'px';
            this._canvas.style.top = top + 'px';
        } else if (this._fit === 'contain') {
            // Center the undersized canvas
            const left = Math.round((displayW - width) / 2);
            const top = Math.round((displayH - height) / 2);
            this._canvas.style.left = left + 'px';
            this._canvas.style.top = top + 'px';
        } else {
            this._canvas.style.left = '0';
            this._canvas.style.top = '0';
        }

        this._canvas.style.width = width + 'px';
        this._canvas.style.height = height + 'px';

        // Show canvas, hide preview
        this._canvas.style.display = 'block';
        this._preview.style.display = 'none';
    }

    // Public API

    get naturalWidth() {
        return this._naturalWidth;
    }

    get naturalHeight() {
        return this._naturalHeight;
    }

    get complete() {
        return !this._processing && this._canvas.style.display === 'block';
    }

    // Force re-render
    refresh() {
        this._pendingResize = false;
        this._scheduleResize();
    }
}

// Register the custom element
customElements.define('linear-img', LinearImg);

// Export for module usage
export { LinearImg };
