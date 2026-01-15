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
 *   src        - Image source URL or data URL (required)
 *   width      - Display width in pixels
 *   height     - Display height in pixels
 *   method     - Interpolation: "lanczos" (default) or "bilinear"
 *   fit        - Object-fit style: "contain" (default), "cover", "fill"
 */

// Shared worker instance for all <linear-img> elements
let sharedWorker = null;
let workerReady = false;
let workerReadyPromise = null;
let pendingRequests = new Map();
let requestId = 0;

// Image cache: caches resized results by "src:width:height:method"
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

// Request a resize operation from the worker using raw pixel data
async function requestResizePixels(pixelData, srcWidth, srcHeight, dstWidth, dstHeight, method) {
    await initSharedWorker();

    const id = ++requestId;
    const methodCode = method === 'bilinear' ? 0 : 1; // 0=bilinear, 1=lanczos

    return new Promise((resolve, reject) => {
        pendingRequests.set(id, { resolve, reject });

        sharedWorker.postMessage({
            type: 'resize-pixels',
            requestId: id,
            pixelData,
            srcWidth,
            srcHeight,
            dstWidth,
            dstHeight,
            interpolation: methodCode,
            scaleMode: 0 // Independent - we calculate exact dimensions ourselves
        });
    });
}

// Get cache key (includes DPR for pixel-perfect caching)
function getCacheKey(src, width, height, method, dpr) {
    return `${src}:${width}:${height}:${method}:${dpr}`;
}

// Add to cache with LRU eviction
function cacheResult(key, imageData) {
    if (resultCache.size >= MAX_CACHE_SIZE) {
        const firstKey = resultCache.keys().next().value;
        resultCache.delete(firstKey);
    }
    resultCache.set(key, imageData);
}

// Extract pixels from an image using canvas
function getImagePixels(img) {
    console.log('[linear-img] Extracting pixels from image:', img.naturalWidth, 'x', img.naturalHeight);
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    console.log('[linear-img] Got pixel data, length:', imageData.data.length);
    return imageData;
}

// Custom element definition
class LinearImg extends HTMLElement {
    static get observedAttributes() {
        return ['src', 'width', 'height', 'method', 'fit'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        // Internal state
        this._src = null;
        this._loadedSrc = null;  // Track which src we've loaded to prevent duplicates
        this._naturalWidth = 0;
        this._naturalHeight = 0;
        this._displayWidth = 0;
        this._displayHeight = 0;
        this._method = 'lanczos';
        this._fit = 'contain';
        this._pixelData = null;  // Raw RGBA pixels from canvas
        this._processing = false;
        this._pendingResize = false;
        this._loading = false;  // Guard against concurrent loads

        // Create container
        this._container = document.createElement('div');
        this._container.style.cssText = 'position: relative; display: inline-block; overflow: hidden; width: 100%; height: 100%;';

        // Preview image (browser-scaled, shown while processing)
        this._preview = document.createElement('img');
        this._preview.style.cssText = 'display: block; width: 100%; height: 100%; object-fit: contain;';

        // Canvas for final output (pointer-events: none lets right-clicks pass to preview)
        this._canvas = document.createElement('canvas');
        this._canvas.style.cssText = 'display: none; position: absolute; top: 0; left: 0; pointer-events: none;';

        this._container.appendChild(this._preview);
        this._container.appendChild(this._canvas);

        // Styles
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
        // Note: we also need to watch devicePixelRatio for zoom changes
        this._resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this._handleResize(width, height);
                }
            }
        });

        // Watch for devicePixelRatio changes (browser zoom)
        this._dprMediaQuery = null;
        this._setupDprWatcher();
    }

    _setupDprWatcher() {
        // Watch for devicePixelRatio changes using matchMedia
        // This fires when browser zoom changes
        const updateDpr = () => {
            // Remove old listener
            if (this._dprMediaQuery) {
                this._dprMediaQuery.removeEventListener('change', this._dprChangeHandler);
            }
            // Create new query for current DPR
            this._dprMediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
            this._dprChangeHandler = () => {
                console.log('[linear-img] devicePixelRatio changed to:', window.devicePixelRatio);
                // Force recalculation with new DPR
                if (this._displayWidth && this._displayHeight) {
                    this._lastDpr = null;  // Force re-render
                    this._scheduleResize();
                }
                updateDpr();  // Re-setup for next change
            };
            this._dprMediaQuery.addEventListener('change', this._dprChangeHandler);
        };
        updateDpr();
        this._lastDpr = window.devicePixelRatio;
    }

    connectedCallback() {
        this._resizeObserver.observe(this);
        this._updateFromAttributes();
    }

    disconnectedCallback() {
        this._resizeObserver.disconnect();
        if (this._dprMediaQuery) {
            this._dprMediaQuery.removeEventListener('change', this._dprChangeHandler);
        }
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;

        switch (name) {
            case 'src':
                this._src = newValue;
                this._loadedSrc = null;  // Clear to allow fresh load
                this._pixelData = null;  // Clear old pixel data
                this._loadImage();
                break;
            case 'width':
                this._container.style.width = newValue ? newValue + 'px' : '100%';
                break;
            case 'height':
                this._container.style.height = newValue ? newValue + 'px' : '100%';
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
        // Note: attributeChangedCallback handles src changes and calls _loadImage
        // This method just syncs internal state for attributes that may have been
        // set before connectedCallback
        this._src = this.getAttribute('src');
        this._method = this.getAttribute('method') === 'bilinear' ? 'bilinear' : 'lanczos';
        this._fit = this.getAttribute('fit') || 'contain';
        this._preview.style.objectFit = this._fit;

        const width = this.getAttribute('width');
        const height = this.getAttribute('height');
        this._container.style.width = width ? width + 'px' : '100%';
        this._container.style.height = height ? height + 'px' : '100%';

        // If we have a src but haven't loaded it yet, load it
        // This handles the case where attributeChangedCallback fired before connectedCallback
        if (this._src && !this._loadedSrc && !this._loading) {
            this._loadImage();
        }
    }

    async _loadImage() {
        if (!this._src) return;

        // Skip if already loaded this src or currently loading
        if (this._src === this._loadedSrc) {
            console.log('[linear-img] Already loaded this src, skipping');
            return;
        }
        if (this._loading) {
            console.log('[linear-img] Already loading, skipping');
            return;
        }

        console.log('[linear-img] _loadImage called, src length:', this._src.length);
        this._loading = true;

        // Show preview immediately using browser scaling
        this._preview.src = this._src;
        this._preview.style.display = 'block';
        this._preview.style.opacity = '1';  // Reset opacity
        this._canvas.style.display = 'none';

        try {
            // Wait for image to load
            await new Promise((resolve, reject) => {
                if (this._preview.complete && this._preview.naturalWidth > 0) {
                    console.log('[linear-img] Image already complete');
                    resolve();
                } else {
                    this._preview.onload = () => {
                        console.log('[linear-img] Image onload fired');
                        resolve();
                    };
                    this._preview.onerror = (e) => {
                        console.error('[linear-img] Image onerror fired', e);
                        reject(new Error('Failed to load image'));
                    };
                }
            });

            this._naturalWidth = this._preview.naturalWidth;
            this._naturalHeight = this._preview.naturalHeight;
            this._loadedSrc = this._src;  // Mark as loaded
            console.log('[linear-img] Natural size:', this._naturalWidth, 'x', this._naturalHeight);

            // Extract pixels via canvas
            const imageData = getImagePixels(this._preview);
            this._pixelData = imageData.data;

            // Trigger resize with current display dimensions
            const rect = this.getBoundingClientRect();
            console.log('[linear-img] Bounding rect:', rect.width, 'x', rect.height);
            if (rect.width > 0 && rect.height > 0) {
                this._displayWidth = Math.round(rect.width);
                this._displayHeight = Math.round(rect.height);
                // Call _scheduleResize directly since we now have pixel data
                this._scheduleResize();
            }

        } catch (err) {
            console.error('[linear-img] Failed to load image:', err);
            this.dispatchEvent(new CustomEvent('error', { detail: err }));
        } finally {
            this._loading = false;
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
        if (!this._pixelData || !this._displayWidth || !this._displayHeight) {
            return;
        }

        if (this._processing) {
            this._pendingResize = true;
            return;
        }

        this._doResize();
    }

    async _doResize() {
        console.log('[linear-img] _doResize called');
        if (!this._pixelData || this._displayWidth <= 0 || this._displayHeight <= 0) {
            console.log('[linear-img] _doResize early return - no pixel data or zero dimensions');
            return;
        }

        const { outputWidth, outputHeight, dpr } = this._calculateOutputDimensions();
        console.log('[linear-img] Output dimensions:', outputWidth, 'x', outputHeight, 'at DPR:', dpr);

        // Skip if output would be same as natural size
        if (outputWidth === this._naturalWidth && outputHeight === this._naturalHeight) {
            console.log('[linear-img] Same as natural size, showing preview');
            this._preview.style.display = 'block';
            this._preview.style.opacity = '1';
            this._canvas.style.display = 'none';
            return;
        }

        // Check cache
        const cacheKey = getCacheKey(this._src, outputWidth, outputHeight, this._method, dpr);
        const cached = resultCache.get(cacheKey);
        if (cached) {
            console.log('[linear-img] Using cached result');
            this._displayResult(cached, outputWidth, outputHeight, dpr);
            return;
        }

        this._processing = true;
        console.log('[linear-img] Starting resize:', this._naturalWidth, 'x', this._naturalHeight, '->', outputWidth, 'x', outputHeight);

        try {
            const result = await requestResizePixels(
                this._pixelData,
                this._naturalWidth,
                this._naturalHeight,
                outputWidth,
                outputHeight,
                this._method
            );

            console.log('[linear-img] Resize complete, result:', result.width, 'x', result.height, 'data length:', result.outputData.length);

            const imageData = new ImageData(
                new Uint8ClampedArray(result.outputData),
                result.width,
                result.height
            );

            cacheResult(cacheKey, imageData);

            // Display if dimensions and DPR still match
            const { outputWidth: currentW, outputHeight: currentH, dpr: currentDpr } = this._calculateOutputDimensions();
            if (currentW === outputWidth && currentH === outputHeight && currentDpr === dpr) {
                console.log('[linear-img] Displaying result');
                this._displayResult(imageData, outputWidth, outputHeight, dpr);
            } else {
                console.log('[linear-img] Dimensions changed during processing, not displaying');
            }

            this.dispatchEvent(new CustomEvent('load', {
                detail: { width: outputWidth, height: outputHeight, dpr }
            }));

        } catch (err) {
            console.error('[linear-img] Resize failed:', err);
            // Keep showing browser-scaled preview on error
        } finally {
            this._processing = false;

            if (this._pendingResize) {
                this._pendingResize = false;
                this._scheduleResize();
            }
        }
    }

    _calculateOutputDimensions() {
        const dpr = window.devicePixelRatio || 1;
        // Convert CSS pixels to physical pixels for pixel-perfect rendering
        const displayW = Math.round(this._displayWidth * dpr);
        const displayH = Math.round(this._displayHeight * dpr);
        const naturalW = this._naturalWidth;
        const naturalH = this._naturalHeight;

        if (!naturalW || !naturalH) {
            return { outputWidth: displayW, outputHeight: displayH, dpr };
        }

        const displayAspect = displayW / displayH;
        const naturalAspect = naturalW / naturalH;

        let outputWidth, outputHeight;

        switch (this._fit) {
            case 'fill':
                outputWidth = displayW;
                outputHeight = displayH;
                break;

            case 'cover':
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
                if (displayAspect > naturalAspect) {
                    outputHeight = displayH;
                    outputWidth = Math.round(displayH * naturalAspect);
                } else {
                    outputWidth = displayW;
                    outputHeight = Math.round(displayW / naturalAspect);
                }
                break;
        }

        return {
            outputWidth: Math.max(1, outputWidth),
            outputHeight: Math.max(1, outputHeight),
            dpr
        };
    }

    _displayResult(imageData, width, height, dpr) {
        // Canvas dimensions are in physical pixels
        this._canvas.width = width;
        this._canvas.height = height;

        const ctx = this._canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        // CSS dimensions are physical pixels / DPR
        const cssWidth = width / dpr;
        const cssHeight = height / dpr;

        // Position canvas (in CSS pixels)
        const displayW = this._displayWidth;
        const displayH = this._displayHeight;

        const left = Math.round((displayW - cssWidth) / 2);
        const top = Math.round((displayH - cssHeight) / 2);
        this._canvas.style.left = left + 'px';
        this._canvas.style.top = top + 'px';
        this._canvas.style.width = cssWidth + 'px';
        this._canvas.style.height = cssHeight + 'px';

        // Show canvas, keep preview invisible but in layout (prevents collapse)
        // Use opacity: 0 (not visibility: hidden) so it still receives right-clicks
        this._canvas.style.display = 'block';
        this._preview.style.opacity = '0';
        this._preview.style.display = 'block';
    }

    // Public API - property accessors that reflect to attributes
    get src() { return this.getAttribute('src'); }
    set src(value) {
        if (value) {
            this.setAttribute('src', value);
        } else {
            this.removeAttribute('src');
        }
    }

    get width() { return this.getAttribute('width'); }
    set width(value) { this.setAttribute('width', value); }

    get height() { return this.getAttribute('height'); }
    set height(value) { this.setAttribute('height', value); }

    get naturalWidth() { return this._naturalWidth; }
    get naturalHeight() { return this._naturalHeight; }
    get complete() { return !this._processing && this._canvas.style.display === 'block'; }

    refresh() {
        this._pendingResize = false;
        this._scheduleResize();
    }
}

customElements.define('linear-img', LinearImg);
export { LinearImg };
