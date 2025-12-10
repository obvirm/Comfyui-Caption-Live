/**
 * C++ Caption Engine WASM Integration
 * 
 * This module will replace the Rust WASM module when C++ WASM is ready.
 * For now, it's a stub that can be tested when the WASM build completes.
 */

let cppCaptionEngine = null;
let cppEngineReady = false;

/**
 * Load the C++ Caption Engine WASM module
 */
export async function loadCppCaptionEngine() {
    if (cppCaptionEngine) {
        return cppCaptionEngine;
    }

    try {
        // Try to load the C++ WASM module
        // Assumes /extensions/caption-live/wasm/caption_engine_wasm.js
        const module = await import('./wasm/caption_engine_wasm.js');
        cppCaptionEngine = await module.default({
            locateFile: (path, prefix) => {
                if (path.endsWith('.data')) {
                    return '/extensions/caption-live/wasm/' + path;
                }
                if (path.endsWith('.wasm')) {
                    return '/extensions/caption-live/wasm/' + path;
                }
                return prefix + path;
            }
        });
        cppEngineReady = true;
        console.log('✅ C++ Caption Engine WASM loaded');
        return cppCaptionEngine;
    } catch (e) {
        console.warn('⚠️ C++ Caption Engine WASM not available:', e);
        return null;
    }
}

/**
 * Check if C++ engine is available
 */
export function isCppEngineReady() {
    return cppEngineReady;
}

/**
 * Render a frame using the C++ engine
 * @param {string} templateJson - JSON template string
 * @param {number} time - Current time in seconds
 * @returns {ImageData|null} - ImageData or null if engine not ready
 */
export function renderFrameCpp(templateJson, time) {
    if (!cppCaptionEngine) {
        return null;
    }

    try {
        const result = cppCaptionEngine.render_frame_rgba(templateJson, time);

        // Create ImageData from raw pixels
        const pixels = new Uint8ClampedArray(result.data);
        return new ImageData(pixels, result.width, result.height);
    } catch (e) {
        console.error('C++ render error:', e);
        return null;
    }
}

/**
 * Integration with existing renderer.js
 * Call this from startRenderLoop to try C++ engine first
 */
export async function renderWithCppEngine(canvas, ctx, templateJson, time) {
    await loadCppCaptionEngine();

    if (!cppEngineReady) {
        return false; // Fallback to existing Rust/JS renderer
    }

    const imageData = renderFrameCpp(templateJson, time);
    if (imageData) {
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        ctx.putImageData(imageData, 0, 0);
        return true;
    }

    return false;
}

/**
 * Test WebGPU Compute Shader
 */
export function testCompute() {
    if (!cppCaptionEngine) return false;
    try {
        console.log("🧪 Triggering C++ Engine GPU Test...");
        const result = cppCaptionEngine.test_compute();
        console.log("C++ GPU Test Result:", result ? "PASS" : "FAIL");
        return result;
    } catch (e) {
        console.error("GPU Test Error:", e);
        return false;
    }
}

export default {
    loadCppCaptionEngine,
    isCppEngineReady,
    renderFrameCpp,
    renderWithCppEngine,
    testCompute
};
