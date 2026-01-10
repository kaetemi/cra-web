// Web Worker for CRA image processing
// Runs Pyodide in a separate thread to keep UI responsive

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js');

let pyodide = null;
let wasmReady = false;
let scriptsLoaded = {};

// Send progress update to main thread
function sendProgress(stage, message, percent) {
    self.postMessage({ type: 'progress', stage, message, percent });
}

// Send console output to main thread
function sendConsole(text) {
    self.postMessage({ type: 'console', text });
}

// Send error to main thread
function sendError(error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
}

// Send completion to main thread
function sendComplete(outputData) {
    self.postMessage({ type: 'complete', outputData });
}

// Initialize Pyodide and WASM
async function initialize() {
    try {
        // Load WASM dither module
        sendProgress('init', 'Loading WASM dither module...', 5);
        const wasmModule = await import('./wasm/dither.js');
        await wasmModule.default();
        self.floyd_steinberg_dither_wasm = wasmModule.floyd_steinberg_dither;
        wasmReady = true;
        sendProgress('init', 'WASM module loaded', 10);

        // Load Pyodide
        sendProgress('init', 'Loading Python runtime...', 15);
        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/'
        });
        sendProgress('init', 'Python runtime loaded', 50);

        // Setup stdout redirect
        pyodide.setStdout({
            batched: (text) => sendConsole(text)
        });

        // Load packages
        sendProgress('init', 'Loading image processing packages...', 55);
        await pyodide.loadPackage(['numpy', 'opencv-python', 'pillow', 'scikit-image', 'micropip']);
        sendProgress('init', 'Packages loaded', 90);

        sendProgress('init', 'Ready!', 100);
        self.postMessage({ type: 'ready' });

    } catch (error) {
        sendError(error);
    }
}

// Load a Python script
async function loadScript(scriptName) {
    if (scriptsLoaded[scriptName]) {
        return;
    }

    const response = await fetch('./scripts/' + scriptName);
    let code = await response.text();

    // Remove the if __name__ == "__main__": block
    code = code.replace(/if\s+__name__\s*==\s*["']__main__["']:\s*[\s\S]*$/, '');

    await pyodide.runPythonAsync(code);
    scriptsLoaded[scriptName] = true;
}

// Process images
async function processImages(inputData, refData, method, config) {
    try {
        sendProgress('process', 'Loading script...', 0);
        await loadScript(config.script);

        sendProgress('process', 'Preparing images...', 10);
        pyodide.FS.writeFile('/input.png', inputData);
        pyodide.FS.writeFile('/ref.png', refData);

        sendProgress('process', 'Processing...', 20);

        let pythonCode;
        if (config.script === 'color_correction_basic.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', keep_luminosity=False, verbose=True)`;
        } else if (config.script === 'color_correction_basic_rgb.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', verbose=True)`;
        } else if (config.script === 'color_correction_cra.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', keep_luminosity=False, verbose=True)`;
        } else if (config.script === 'color_correction_tiled.py') {
            const tiledLum = config.options.tiled_luminosity ? 'True' : 'False';
            pythonCode = `main('/input.png', '/ref.png', '/output.png', tiled_luminosity=${tiledLum}, verbose=True)`;
        } else if (config.script === 'color_correction_cra_rgb.py') {
            const perceptual = config.options.perceptual ? 'True' : 'False';
            pythonCode = `main('/input.png', '/ref.png', '/output.png', verbose=True, use_perceptual=${perceptual})`;
        }

        await pyodide.runPythonAsync(pythonCode);

        sendProgress('process', 'Reading result...', 90);
        const outputData = pyodide.FS.readFile('/output.png');

        sendComplete(outputData);

    } catch (error) {
        sendError(error);
    }
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, ...data } = e.data;

    switch (type) {
        case 'init':
            await initialize();
            break;
        case 'process':
            await processImages(data.inputData, data.refData, data.method, data.config);
            break;
    }
};
