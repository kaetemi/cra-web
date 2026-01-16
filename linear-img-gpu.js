/**
 * <linear-img-gpu> - WebGPU-accelerated linear RGBA image resizing
 *
 * A standalone custom element that displays images using proper linear RGB
 * interpolation with Lanczos3 resampling, accelerated by WebGPU compute shaders.
 *
 * Unlike browser-native image scaling (which operates in sRGB space and causes
 * darkening/color shifts), this element converts to linear RGB, resamples with
 * Lanczos3, and converts back to sRGB. Alpha channel is preserved and
 * interpolated linearly (no gamma conversion needed for alpha).
 *
 * Usage:
 *   <linear-img-gpu src="photo.jpg" width="200" height="150"></linear-img-gpu>
 *   <linear-img-gpu src="photo.jpg" style="width: 200px;"></linear-img-gpu>
 *
 * Attributes:
 *   src    - Image source URL (required)
 *   width  - Display width in pixels
 *   height - Display height in pixels
 *   fit    - Object-fit style: "contain" (default), "cover", "fill"
 *
 * Constants and algorithm match COLORSPACES.md and rescale.rs exactly.
 */

// ============================================================================
// WebGPU Shader Code (WGSL)
// ============================================================================

// sRGB to Linear RGB conversion shader
// Constants from COLORSPACES.md: threshold 0.04045, slope 12.92, gamma 2.4
// Alpha channel is passed through unchanged (already linear)
const SHADER_SRGB_TO_LINEAR = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(inputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pixel = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
    let idx = (gid.y * dims.x + gid.x) * 4u;

    // sRGB decode (COLORSPACES.md): if srgb <= 0.04045: linear = srgb / 12.92
    // else: linear = ((srgb + 0.055) / 1.055)^2.4
    output[idx + 0u] = srgbToLinear(pixel.r);
    output[idx + 1u] = srgbToLinear(pixel.g);
    output[idx + 2u] = srgbToLinear(pixel.b);
    output[idx + 3u] = pixel.a; // Alpha is already linear
}

fn srgbToLinear(s: f32) -> f32 {
    if (s <= 0.04045) {
        return s / 12.92;
    } else {
        return pow((s + 0.055) / 1.055, 2.4);
    }
}
`;

// Horizontal Lanczos3 resample shader (RGBA - 4 channels)
const SHADER_LANCZOS_HORIZONTAL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;

struct Params {
    srcWidth: u32,
    srcHeight: u32,
    dstWidth: u32,
    dstHeight: u32,
    scale: f32,
    filterScale: f32,
    radius: i32,
    offset: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dstX = gid.x % params.dstWidth;
    let y = gid.x / params.dstWidth;
    if (y >= params.srcHeight) { return; }

    // Coordinate mapping (rescale.rs:81)
    // src_pos = (dst_i + 0.5) * scale - 0.5 + offset
    let srcPos = (f32(dstX) + 0.5) * params.scale - 0.5 + params.offset;
    let center = i32(floor(srcPos));

    // Compute kernel weights and apply (rescale.rs:84-97)
    var sumR: f32 = 0.0;
    var sumG: f32 = 0.0;
    var sumB: f32 = 0.0;
    var sumA: f32 = 0.0;
    var weightSum: f32 = 0.0;

    let startIdx = max(center - params.radius, 0);
    let endIdx = min(center + params.radius, i32(params.srcWidth) - 1);

    for (var si: i32 = startIdx; si <= endIdx; si = si + 1) {
        let d = (srcPos - f32(si)) / params.filterScale;
        let weight = lanczos3(d);
        weightSum += weight;

        let srcIdx = (y * params.srcWidth + u32(si)) * 4u;
        sumR += input[srcIdx + 0u] * weight;
        sumG += input[srcIdx + 1u] * weight;
        sumB += input[srcIdx + 2u] * weight;
        sumA += input[srcIdx + 3u] * weight;
    }

    // Normalize weights (rescale.rs:100-103)
    if (abs(weightSum) > 1e-8) {
        sumR /= weightSum;
        sumG /= weightSum;
        sumB /= weightSum;
        sumA /= weightSum;
    }

    let dstIdx = (y * params.dstWidth + dstX) * 4u;
    output[dstIdx + 0u] = sumR;
    output[dstIdx + 1u] = sumG;
    output[dstIdx + 2u] = sumB;
    output[dstIdx + 3u] = sumA;
}

// Lanczos3 kernel (rescale.rs:40-50)
fn lanczos3(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-8) {
        return 1.0;
    } else if (ax >= 3.0) {
        return 0.0;
    } else {
        let piX = PI * x;
        let piX3 = piX / 3.0;
        return (sin(piX) / piX) * (sin(piX3) / piX3);
    }
}
`;

// Vertical Lanczos3 resample shader (RGBA - 4 channels)
const SHADER_LANCZOS_VERTICAL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;

struct Params {
    srcWidth: u32,
    srcHeight: u32,
    dstWidth: u32,
    dstHeight: u32,
    scale: f32,
    filterScale: f32,
    radius: i32,
    offset: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x % params.dstWidth;
    let dstY = gid.x / params.dstWidth;
    if (dstY >= params.dstHeight) { return; }

    // Coordinate mapping (rescale.rs:81)
    let srcPos = (f32(dstY) + 0.5) * params.scale - 0.5 + params.offset;
    let center = i32(floor(srcPos));

    var sumR: f32 = 0.0;
    var sumG: f32 = 0.0;
    var sumB: f32 = 0.0;
    var sumA: f32 = 0.0;
    var weightSum: f32 = 0.0;

    let startIdx = max(center - params.radius, 0);
    let endIdx = min(center + params.radius, i32(params.srcHeight) - 1);

    for (var si: i32 = startIdx; si <= endIdx; si = si + 1) {
        let d = (srcPos - f32(si)) / params.filterScale;
        let weight = lanczos3(d);
        weightSum += weight;

        // Input is (srcHeight x dstWidth) from horizontal pass
        let srcIdx = (u32(si) * params.dstWidth + x) * 4u;
        sumR += input[srcIdx + 0u] * weight;
        sumG += input[srcIdx + 1u] * weight;
        sumB += input[srcIdx + 2u] * weight;
        sumA += input[srcIdx + 3u] * weight;
    }

    if (abs(weightSum) > 1e-8) {
        sumR /= weightSum;
        sumG /= weightSum;
        sumB /= weightSum;
        sumA /= weightSum;
    }

    let dstIdx = (dstY * params.dstWidth + x) * 4u;
    output[dstIdx + 0u] = sumR;
    output[dstIdx + 1u] = sumG;
    output[dstIdx + 2u] = sumB;
    output[dstIdx + 3u] = sumA;
}

fn lanczos3(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-8) {
        return 1.0;
    } else if (ax >= 3.0) {
        return 0.0;
    } else {
        let piX = PI * x;
        let piX3 = piX / 3.0;
        return (sin(piX) / piX) * (sin(piX3) / piX3);
    }
}
`;

// Linear RGB to sRGB conversion shader (RGBA - 4 channels)
// Constants from COLORSPACES.md: threshold 0.0031308, gamma 1/2.4
// Alpha is passed through unchanged (already linear, no gamma)
const SHADER_LINEAR_TO_SRGB = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Dims {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixelCount = dims.width * dims.height;
    if (gid.x >= pixelCount) { return; }

    let idx = gid.x * 4u;
    let r = linearToSrgb(input[idx + 0u]);
    let g = linearToSrgb(input[idx + 1u]);
    let b = linearToSrgb(input[idx + 2u]);
    let a = input[idx + 3u]; // Alpha is already linear, no conversion

    // Pack as RGBA u32 (for easy copy to canvas)
    let ri = u32(clamp(r * 255.0 + 0.5, 0.0, 255.0));
    let gi = u32(clamp(g * 255.0 + 0.5, 0.0, 255.0));
    let bi = u32(clamp(b * 255.0 + 0.5, 0.0, 255.0));
    let ai = u32(clamp(a * 255.0 + 0.5, 0.0, 255.0));
    output[gid.x] = ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

fn linearToSrgb(l: f32) -> f32 {
    // Clamp to valid range (Lanczos can produce slight overshoot)
    let lc = clamp(l, 0.0, 1.0);
    if (lc <= 0.0031308) {
        return 12.92 * lc;
    } else {
        return 1.055 * pow(lc, 1.0 / 2.4) - 0.055;
    }
}
`;

// ============================================================================
// WebGPU Pipeline Manager
// ============================================================================

class LanczosGPU {
    constructor() {
        this.device = null;
        this.pipelines = null;
        this.initPromise = null;
    }

    async init() {
        if (this.device) return true;
        if (this.initPromise) return this.initPromise;

        this.initPromise = this._doInit();
        return this.initPromise;
    }

    async _doInit() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        // Request higher storage buffer size limit for large images (up to 512MB)
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: Math.min(512 * 1024 * 1024, adapter.limits.maxStorageBufferBindingSize)
            }
        });

        // Create shader modules
        const srgbToLinearModule = this.device.createShaderModule({
            code: SHADER_SRGB_TO_LINEAR
        });
        const lanczosHModule = this.device.createShaderModule({
            code: SHADER_LANCZOS_HORIZONTAL
        });
        const lanczosVModule = this.device.createShaderModule({
            code: SHADER_LANCZOS_VERTICAL
        });
        const linearToSrgbModule = this.device.createShaderModule({
            code: SHADER_LINEAR_TO_SRGB
        });

        // Create pipelines
        this.pipelines = {
            srgbToLinear: this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: srgbToLinearModule, entryPoint: 'main' }
            }),
            lanczosH: this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: lanczosHModule, entryPoint: 'main' }
            }),
            lanczosV: this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: lanczosVModule, entryPoint: 'main' }
            }),
            linearToSrgb: this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: linearToSrgbModule, entryPoint: 'main' }
            })
        };

        return true;
    }

    /**
     * Resize an image using Lanczos3 in linear RGB space
     * @param {ImageData} imageData - Source image
     * @param {number} dstWidth - Target width
     * @param {number} dstHeight - Target height
     * @returns {Promise<ImageData>} - Resized image
     */
    async resize(imageData, dstWidth, dstHeight) {
        await this.init();

        const srcWidth = imageData.width;
        const srcHeight = imageData.height;
        const srcPixels = srcWidth * srcHeight;
        const dstPixels = dstWidth * dstHeight;

        // Calculate scale factors (rescale.rs:128-129 - Independent mode)
        const scaleX = srcWidth / dstWidth;
        const scaleY = srcHeight / dstHeight;

        // Filter scales and radii (rescale.rs:231-234)
        const filterScaleX = Math.max(scaleX, 1.0);
        const filterScaleY = Math.max(scaleY, 1.0);
        const radiusX = Math.ceil(3.0 * filterScaleX);
        const radiusY = Math.ceil(3.0 * filterScaleY);

        // Offsets for centering (rescale.rs:77-78)
        const mappedSrcW = dstWidth * scaleX;
        const mappedSrcH = dstHeight * scaleY;
        const offsetX = (srcWidth - mappedSrcW) / 2.0;
        const offsetY = (srcHeight - mappedSrcH) / 2.0;

        // Create buffers
        const srcTexture = this.device.createTexture({
            size: [srcWidth, srcHeight],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        this.device.queue.writeTexture(
            { texture: srcTexture },
            imageData.data,
            { bytesPerRow: srcWidth * 4 },
            [srcWidth, srcHeight]
        );

        // Linear RGBA buffers (4 floats per pixel)
        const linearSrcBuffer = this.device.createBuffer({
            size: srcPixels * 4 * 4,
            usage: GPUBufferUsage.STORAGE
        });

        // Intermediate buffer after horizontal pass (dstWidth x srcHeight)
        const intermediatePixels = dstWidth * srcHeight;
        const intermediateBuffer = this.device.createBuffer({
            size: intermediatePixels * 4 * 4,
            usage: GPUBufferUsage.STORAGE
        });

        // Final linear buffer (dstWidth x dstHeight)
        const linearDstBuffer = this.device.createBuffer({
            size: dstPixels * 4 * 4,
            usage: GPUBufferUsage.STORAGE
        });

        // Output RGBA buffer
        const outputBuffer = this.device.createBuffer({
            size: dstPixels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Staging buffer for readback
        const stagingBuffer = this.device.createBuffer({
            size: dstPixels * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Uniform buffers for parameters
        const paramsHBuffer = this.device.createBuffer({
            size: 32, // 8 x 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const paramsVBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const dimsBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Write horizontal params
        const paramsH = new ArrayBuffer(32);
        const paramsHView = new DataView(paramsH);
        paramsHView.setUint32(0, srcWidth, true);
        paramsHView.setUint32(4, srcHeight, true);
        paramsHView.setUint32(8, dstWidth, true);
        paramsHView.setUint32(12, dstHeight, true);
        paramsHView.setFloat32(16, scaleX, true);
        paramsHView.setFloat32(20, filterScaleX, true);
        paramsHView.setInt32(24, radiusX, true);
        paramsHView.setFloat32(28, offsetX, true);
        this.device.queue.writeBuffer(paramsHBuffer, 0, paramsH);

        // Write vertical params
        const paramsV = new ArrayBuffer(32);
        const paramsVView = new DataView(paramsV);
        paramsVView.setUint32(0, dstWidth, true);  // srcWidth for vertical = dstWidth from horizontal
        paramsVView.setUint32(4, srcHeight, true); // srcHeight for vertical = srcHeight (intermediate)
        paramsVView.setUint32(8, dstWidth, true);
        paramsVView.setUint32(12, dstHeight, true);
        paramsVView.setFloat32(16, scaleY, true);
        paramsVView.setFloat32(20, filterScaleY, true);
        paramsVView.setInt32(24, radiusY, true);
        paramsVView.setFloat32(28, offsetY, true);
        this.device.queue.writeBuffer(paramsVBuffer, 0, paramsV);

        // Write dims
        const dims = new Uint32Array([dstWidth, dstHeight]);
        this.device.queue.writeBuffer(dimsBuffer, 0, dims);

        // Create bind groups
        const srgbToLinearBG = this.device.createBindGroup({
            layout: this.pipelines.srgbToLinear.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: srcTexture.createView() },
                { binding: 1, resource: { buffer: linearSrcBuffer } }
            ]
        });

        const lanczosHBG = this.device.createBindGroup({
            layout: this.pipelines.lanczosH.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: linearSrcBuffer } },
                { binding: 1, resource: { buffer: intermediateBuffer } },
                { binding: 2, resource: { buffer: paramsHBuffer } }
            ]
        });

        const lanczosVBG = this.device.createBindGroup({
            layout: this.pipelines.lanczosV.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: intermediateBuffer } },
                { binding: 1, resource: { buffer: linearDstBuffer } },
                { binding: 2, resource: { buffer: paramsVBuffer } }
            ]
        });

        const linearToSrgbBG = this.device.createBindGroup({
            layout: this.pipelines.linearToSrgb.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: linearDstBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: dimsBuffer } }
            ]
        });

        // Execute compute passes
        const commandEncoder = this.device.createCommandEncoder();

        // Pass 1: sRGB to Linear
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pipelines.srgbToLinear);
            pass.setBindGroup(0, srgbToLinearBG);
            pass.dispatchWorkgroups(
                Math.ceil(srcWidth / 16),
                Math.ceil(srcHeight / 16)
            );
            pass.end();
        }

        // Pass 2: Horizontal Lanczos
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pipelines.lanczosH);
            pass.setBindGroup(0, lanczosHBG);
            // Dispatch for each output pixel in horizontal pass
            pass.dispatchWorkgroups(Math.ceil(intermediatePixels / 256));
            pass.end();
        }

        // Pass 3: Vertical Lanczos
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pipelines.lanczosV);
            pass.setBindGroup(0, lanczosVBG);
            pass.dispatchWorkgroups(Math.ceil(dstPixels / 256));
            pass.end();
        }

        // Pass 4: Linear to sRGB
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pipelines.linearToSrgb);
            pass.setBindGroup(0, linearToSrgbBG);
            pass.dispatchWorkgroups(Math.ceil(dstPixels / 256));
            pass.end();
        }

        // Copy output to staging buffer
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, dstPixels * 4);

        this.device.queue.submit([commandEncoder.finish()]);

        // Read back result
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Uint8ClampedArray(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();

        // Cleanup
        srcTexture.destroy();
        linearSrcBuffer.destroy();
        intermediateBuffer.destroy();
        linearDstBuffer.destroy();
        outputBuffer.destroy();
        stagingBuffer.destroy();
        paramsHBuffer.destroy();
        paramsVBuffer.destroy();
        dimsBuffer.destroy();

        return new ImageData(resultData, dstWidth, dstHeight);
    }
}

// Shared GPU instance
let sharedGPU = null;

async function getGPU() {
    if (!sharedGPU) {
        sharedGPU = new LanczosGPU();
    }
    await sharedGPU.init();
    return sharedGPU;
}

// Check WebGPU support
function isWebGPUSupported() {
    return !!navigator.gpu;
}

// ============================================================================
// Result Cache
// ============================================================================

const resultCache = new Map();
const MAX_CACHE_SIZE = 50;

function getCacheKey(src, width, height, dpr) {
    return `${src}:${width}:${height}:${dpr}`;
}

function cacheResult(key, imageData) {
    if (resultCache.size >= MAX_CACHE_SIZE) {
        const firstKey = resultCache.keys().next().value;
        resultCache.delete(firstKey);
    }
    resultCache.set(key, imageData);
}

// ============================================================================
// Custom Element
// ============================================================================

class LinearImgGPU extends HTMLElement {
    static get observedAttributes() {
        return ['src', 'width', 'height', 'fit'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        this._src = null;
        this._loadedSrc = null;
        this._naturalWidth = 0;
        this._naturalHeight = 0;
        this._displayWidth = 0;
        this._displayHeight = 0;
        this._fit = 'contain';
        this._sourceImageData = null;
        this._processing = false;
        this._pendingResize = false;
        this._loading = false;

        // Create container
        this._container = document.createElement('div');
        this._container.style.cssText = 'position: relative; display: inline-block; overflow: hidden; width: 100%; height: 100%;';

        // Preview image (browser-scaled, shown while processing)
        this._preview = document.createElement('img');
        this._preview.style.cssText = 'display: block; width: 100%; height: 100%; object-fit: contain;';

        // Canvas for final output
        this._canvas = document.createElement('canvas');
        this._canvas.style.cssText = 'display: none; position: absolute; top: 0; left: 0; pointer-events: none;';

        this._container.appendChild(this._preview);
        this._container.appendChild(this._canvas);

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

        // ResizeObserver
        this._resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this._handleResize(width, height);
                }
            }
        });

        // DPR watcher
        this._dprMediaQuery = null;
        this._setupDprWatcher();
    }

    _setupDprWatcher() {
        const updateDpr = () => {
            if (this._dprMediaQuery) {
                this._dprMediaQuery.removeEventListener('change', this._dprChangeHandler);
            }
            this._dprMediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
            this._dprChangeHandler = () => {
                if (this._displayWidth && this._displayHeight) {
                    this._scheduleResize();
                }
                updateDpr();
            };
            this._dprMediaQuery.addEventListener('change', this._dprChangeHandler);
        };
        updateDpr();
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
                this._loadedSrc = null;
                this._sourceImageData = null;
                this._loadImage();
                break;
            case 'width':
                this._container.style.width = newValue ? newValue + 'px' : '100%';
                break;
            case 'height':
                this._container.style.height = newValue ? newValue + 'px' : '100%';
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
        this._fit = this.getAttribute('fit') || 'contain';
        this._preview.style.objectFit = this._fit;

        const width = this.getAttribute('width');
        const height = this.getAttribute('height');
        this._container.style.width = width ? width + 'px' : '100%';
        this._container.style.height = height ? height + 'px' : '100%';

        if (this._src && !this._loadedSrc && !this._loading) {
            this._loadImage();
        }
    }

    async _loadImage() {
        if (!this._src) return;
        if (this._src === this._loadedSrc || this._loading) return;

        this._loading = true;

        this._preview.src = this._src;
        this._preview.style.display = 'block';
        this._preview.style.opacity = '1';
        this._canvas.style.display = 'none';

        try {
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
            this._loadedSrc = this._src;

            // Extract pixels via canvas
            const canvas = document.createElement('canvas');
            canvas.width = this._naturalWidth;
            canvas.height = this._naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this._preview, 0, 0);
            this._sourceImageData = ctx.getImageData(0, 0, this._naturalWidth, this._naturalHeight);

            const rect = this.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                this._displayWidth = Math.round(rect.width);
                this._displayHeight = Math.round(rect.height);
                this._scheduleResize();
            }

        } catch (err) {
            console.error('[linear-img-gpu] Load failed:', err);
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
        if (!this._sourceImageData || !this._displayWidth || !this._displayHeight) {
            return;
        }

        if (this._processing) {
            this._pendingResize = true;
            return;
        }

        this._doResize();
    }

    async _doResize() {
        if (!this._sourceImageData || this._displayWidth <= 0 || this._displayHeight <= 0) {
            return;
        }

        const { outputWidth, outputHeight, dpr } = this._calculateOutputDimensions();

        // Skip if output would be same as natural size
        if (outputWidth === this._naturalWidth && outputHeight === this._naturalHeight) {
            this._preview.style.display = 'block';
            this._preview.style.opacity = '1';
            this._canvas.style.display = 'none';
            return;
        }

        // Check cache
        const cacheKey = getCacheKey(this._src, outputWidth, outputHeight, dpr);
        const cached = resultCache.get(cacheKey);
        if (cached) {
            this._displayResult(cached, outputWidth, outputHeight, dpr);
            return;
        }

        this._processing = true;

        try {
            // Check WebGPU support
            if (!isWebGPUSupported()) {
                console.warn('[linear-img-gpu] WebGPU not supported, using preview');
                return;
            }

            console.log('[linear-img-gpu] Resizing:', this._naturalWidth, 'x', this._naturalHeight, '->',
                outputWidth, 'x', outputHeight, '@' + dpr + 'x');

            const gpu = await getGPU();
            const result = await gpu.resize(this._sourceImageData, outputWidth, outputHeight);

            cacheResult(cacheKey, result);

            // Display if dimensions still match
            const { outputWidth: currentW, outputHeight: currentH, dpr: currentDpr } = this._calculateOutputDimensions();
            if (currentW === outputWidth && currentH === outputHeight && currentDpr === dpr) {
                console.log('[linear-img-gpu] Done:', outputWidth, 'x', outputHeight);
                this._displayResult(result, outputWidth, outputHeight, dpr);
            }

            this.dispatchEvent(new CustomEvent('load', {
                detail: { width: outputWidth, height: outputHeight, dpr }
            }));

        } catch (err) {
            console.error('[linear-img-gpu] Resize failed:', err);
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
        this._canvas.width = width;
        this._canvas.height = height;

        const ctx = this._canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        const cssWidth = width / dpr;
        const cssHeight = height / dpr;

        const displayW = this._displayWidth;
        const displayH = this._displayHeight;

        const left = Math.round((displayW - cssWidth) / 2);
        const top = Math.round((displayH - cssHeight) / 2);
        this._canvas.style.left = left + 'px';
        this._canvas.style.top = top + 'px';
        this._canvas.style.width = cssWidth + 'px';
        this._canvas.style.height = cssHeight + 'px';

        this._canvas.style.display = 'block';
        this._preview.style.opacity = '0';
        this._preview.style.display = 'block';
    }

    // Public API
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

customElements.define('linear-img-gpu', LinearImgGPU);

// Export for module usage
export { LinearImgGPU, LanczosGPU, isWebGPUSupported };
