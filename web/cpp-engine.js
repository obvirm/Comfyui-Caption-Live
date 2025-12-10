/**
 * C++ Caption Engine WASM Integration (Unified GPU Pipeline)
 * 
 * Uses the same C++ engine as the backend:
 * - WebGPU rendering in browser
 * - SPIR-V shaders compiled to WGSL
 * - SDF text rendering for crisp text at any resolution
 * 
 * API matches Python's caption_engine_unified module.
 */

let wasmModule = null;
let renderPipeline = null;
let engineReady = false;
let initPromise = null;

// WASM module paths
const WASM_PATHS = [
    '/extensions/caption-live/wasm/caption_engine_unified.js',
    '/extensions/caption-live/lib/caption_engine_unified.js',
    './wasm/caption_engine_unified.js'
];

/**
 * Initialize the Unified GPU Render Pipeline (WASM)
 */
export async function initUnifiedPipeline() {
    if (renderPipeline) return renderPipeline;
    if (initPromise) return initPromise;

    initPromise = (async () => {
        console.log('🚀 Loading Unified GPU Pipeline (WASM)...');

        // Try each possible path
        for (const path of WASM_PATHS) {
            try {
                const module = await import(path);
                wasmModule = await module.default({
                    locateFile: (file, prefix) => {
                        if (file.endsWith('.wasm')) {
                            return path.replace('.js', '.wasm');
                        }
                        return prefix + file;
                    }
                });

                // Get the global pipeline (same as Python's get_pipeline())
                renderPipeline = wasmModule.get_pipeline();

                // Initialize with common resolution
                const target = new wasmModule.RenderTarget();
                target.width = 1920;
                target.height = 1080;
                target.fps = 60.0;

                if (renderPipeline.initialize(target)) {
                    engineReady = true;
                    console.log('✅ Unified GPU Pipeline Ready!');
                    console.log(`   Backend: ${renderPipeline.backend_name()}`);
                    return renderPipeline;
                }
            } catch (e) {
                console.log(`   Trying ${path}... not found`);
            }
        }

        console.warn('⚠️ Unified Pipeline WASM not available, using fallback');
        return null;
    })();

    return initPromise;
}

/**
 * UNIFIED API: Process frame with scene template
 * Identical to Python: process_frame(template_json, time, input_image)
 * 
 * @param {string} templateJson - Scene description JSON
 * @param {number} time - Current time in seconds
 * @param {ImageData} inputImage - Input image to composite on
 * @returns {ImageData|null} - Composited result
 */
export async function processFrame(templateJson, time, inputImage) {
    const pipeline = await initUnifiedPipeline();

    if (!pipeline || !wasmModule) {
        return null;
    }

    try {
        // Parse scene template
        const scene = wasmModule.SceneTemplate.from_json(templateJson);

        // Create timing info
        const timing = new wasmModule.FrameTiming();
        timing.current_time = time;
        timing.duration = scene.duration || 5.0;
        timing.delta_time = 1.0 / 60.0;

        // Call unified render (composites input + captions)
        const output = pipeline.render_frame_composite(
            scene,
            timing,
            inputImage.data,
            inputImage.width,
            inputImage.height
        );

        // Convert to ImageData
        const pixels = output.to_numpy();
        return new ImageData(
            new Uint8ClampedArray(pixels),
            output.width,
            output.height
        );
    } catch (e) {
        console.error('processFrame error:', e);
        return null;
    }
}

/**
 * Render frame without input (transparent background)
 */
export async function renderFrame(templateJson, time, width, height) {
    const pipeline = await initUnifiedPipeline();

    if (!pipeline || !wasmModule) {
        return null;
    }

    try {
        const scene = wasmModule.SceneTemplate.from_json(templateJson);

        // Re-initialize if size changed
        if (scene.target.width !== width || scene.target.height !== height) {
            scene.target.width = width;
            scene.target.height = height;
            const target = new wasmModule.RenderTarget();
            target.width = width;
            target.height = height;
            pipeline.initialize(target);
        }

        const timing = new wasmModule.FrameTiming();
        timing.current_time = time;
        timing.duration = scene.duration || 5.0;

        const output = pipeline.render_frame(scene, timing);

        return new ImageData(
            new Uint8ClampedArray(output.to_numpy()),
            output.width,
            output.height
        );
    } catch (e) {
        console.error('renderFrame error:', e);
        return null;
    }
}

/**
 * Render to canvas (convenience function)
 */
export async function renderToCanvas(canvas, ctx, templateJson, time, inputImageData = null) {
    await initUnifiedPipeline();

    if (!engineReady) {
        return false;
    }

    try {
        let imageData;

        if (inputImageData) {
            imageData = await processFrame(templateJson, time, inputImageData);
        } else {
            imageData = await renderFrame(templateJson, time, canvas.width, canvas.height);
        }

        if (imageData) {
            if (canvas.width !== imageData.width || canvas.height !== imageData.height) {
                canvas.width = imageData.width;
                canvas.height = imageData.height;
            }
            ctx.putImageData(imageData, 0, 0);
            return true;
        }
    } catch (e) {
        console.error('renderToCanvas error:', e);
    }

    return false;
}

/**
 * Check if pipeline is ready
 */
export function isReady() {
    return engineReady;
}

/**
 * Get backend info
 */
export async function getBackendInfo() {
    const pipeline = await initUnifiedPipeline();
    if (!pipeline) {
        return { name: 'None', type: 'fallback', hasCUDA: false };
    }

    return {
        name: pipeline.backend_name(),
        type: wasmModule.BackendType[pipeline.active_backend()],
        hasCUDA: pipeline.has_cuda()
    };
}

// Legacy exports for backward compatibility
export const loadCppCaptionEngine = initUnifiedPipeline;
export const isCppEngineReady = isReady;
export const renderWithCppEngine = renderToCanvas;

export default {
    initUnifiedPipeline,
    processFrame,
    renderFrame,
    renderToCanvas,
    isReady,
    getBackendInfo,
    // Legacy
    loadCppCaptionEngine,
    isCppEngineReady,
    renderWithCppEngine
};
