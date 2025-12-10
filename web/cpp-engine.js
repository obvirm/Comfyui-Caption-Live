/**
 * C++ Caption Engine WASM Integration
 * 
 * Unified API: process_frame() matches Python backend exactly.
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
        const module = await import('./wasm/caption_engine_wasm.js');
        cppCaptionEngine = await module.default({
            locateFile: (path, prefix) => {
                if (path.endsWith('.data') || path.endsWith('.wasm')) {
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
 * UNIFIED API: Process frame with input image
 * Matches Python: process_frame(json, time, input_array) -> output_array
 * 
 * @param {string} templateJson - JSON scene description
 * @param {number} time - Current time in seconds
 * @param {Uint8ClampedArray} inputData - Input image data (RGBA)
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {ImageData|null} - Composited ImageData or null if failed
 */
export function processFrame(templateJson, time, inputData, width, height) {
    if (!cppCaptionEngine || !cppCaptionEngine.process_frame) {
        return null;
    }

    try {
        const result = cppCaptionEngine.process_frame(
            templateJson,
            time,
            inputData,
            width,
            height
        );

        const pixels = new Uint8ClampedArray(result.data);
        return new ImageData(pixels, result.width, result.height);
    } catch (e) {
        console.error('process_frame error:', e);
        return null;
    }
}

/**
 * Legacy: Render frame without input (caption only on transparent)
 */
export function renderFrameCpp(templateJson, time) {
    if (!cppCaptionEngine) {
        return null;
    }

    try {
        const result = cppCaptionEngine.render_frame_rgba(templateJson, time);
        const pixels = new Uint8ClampedArray(result.data);
        return new ImageData(pixels, result.width, result.height);
    } catch (e) {
        console.error('C++ render error:', e);
        return null;
    }
}

/**
 * UNIFIED: Render with C++ engine (process_frame API)
 * @param {HTMLCanvasElement} canvas
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} templateJson
 * @param {number} time
 * @param {ImageData} inputImageData - Optional input image to composite on
 */
export async function renderWithCppEngine(canvas, ctx, templateJson, time, inputImageData = null) {
    await loadCppCaptionEngine();

    if (!cppEngineReady) {
        return false;
    }

    try {
        let imageData;

        if (inputImageData && cppCaptionEngine.process_frame) {
            // Use unified process_frame API (same as Python backend)
            imageData = processFrame(
                templateJson,
                time,
                inputImageData.data,
                inputImageData.width,
                inputImageData.height
            );
        } else if (cppCaptionEngine.render_frame_rgba) {
            // Fallback: render caption only (for preview without input)
            imageData = renderFrameCpp(templateJson, time);
        }

        if (imageData) {
            // Handle dimension changes
            if (canvas.width !== imageData.width || canvas.height !== imageData.height) {
                canvas.width = imageData.width;
                canvas.height = imageData.height;
            }
            ctx.putImageData(imageData, 0, 0);
            return true;
        }
    } catch (e) {
        console.error('renderWithCppEngine error:', e);
    }

    return false;
}

/**
 * Test GPU Compute
 */
export function testCompute() {
    if (!cppCaptionEngine) return false;
    try {
        const result = cppCaptionEngine.test_compute();
        console.log("GPU Test:", result ? "PASS" : "FAIL");
        return result;
    } catch (e) {
        console.error("GPU Test Error:", e);
        return false;
    }
}

export default {
    loadCppCaptionEngine,
    isCppEngineReady,
    processFrame,
    renderFrameCpp,
    renderWithCppEngine,
    testCompute
};
