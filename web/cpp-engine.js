// C++ Caption Engine Adapter (WASM)
// Adapts the existing caption_engine_wasm binary to the renderer interface

let wasmModule = null;
let engineReady = false;
let initPromise = null;

// WASM module paths
const WASM_PATHS = [
    '/extensions/caption-live/wasm/caption_engine_wasm.js',
    '/extensions/caption-live/lib/caption_engine_wasm/caption_engine.js',
    './wasm/caption_engine_wasm.js'
];

export async function initUnifiedPipeline() {
    if (engineReady && wasmModule) return wasmModule;
    if (initPromise) return initPromise;

    initPromise = (async () => {
        console.log('🚀 Loading C++ Caption Engine (WASM)...');

        for (const path of WASM_PATHS) {
            try {
                const module = await import(path);

                // Initialize module
                wasmModule = await module.default({
                    locateFile: (file, prefix) => {
                        if (file.endsWith('.wasm')) return path.replace('.js', '.wasm');
                        if (file.endsWith('.data')) return path.replace('.js', '.data');
                        return prefix + file;
                    },
                    print: (text) => console.log(`[WASM] ${text}`),
                    printErr: (text) => console.error(`[WASM Error] ${text}`)
                });

                engineReady = true;
                console.log('✅ C++ Caption Engine Ready!');
                return wasmModule;

            } catch (e) {
                console.log(`   Trying ${path}... failed:`, e);
            }
        }

        console.error('❌ Failed to load C++ WASM Engine from any path.');
        return null;
    })();

    return initPromise;
}

export async function processFrame(templateJson, time, inputImage) {
    if (!wasmModule) await initUnifiedPipeline();
    if (!wasmModule) return null;

    try {
        // Call WASM process_frame
        // Signature: (template_json, time, input_data, width, height)
        const result = wasmModule.process_frame(
            templateJson,
            time,
            inputImage.data,
            inputImage.width,
            inputImage.height
        );

        return new ImageData(
            new Uint8ClampedArray(result.data),
            result.width,
            result.height
        );
    } catch (e) {
        console.error("WASM processFrame error:", e);
        return null;
    }
}

export async function renderFrame(templateJson, time, width, height) {
    if (!wasmModule) await initUnifiedPipeline();
    if (!wasmModule) return null;

    try {
        // Call WASM render_frame_rgba
        const result = wasmModule.render_frame_rgba(templateJson, time);

        return new ImageData(
            new Uint8ClampedArray(result.data),
            result.width,
            result.height
        );
    } catch (e) {
        console.error("WASM renderFrame error:", e);
        return null;
    }
}

export async function renderToCanvas(canvas, ctx, templateJson, time, inputImageData = null) {
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
        console.error("renderToCanvas error:", e);
    }
    return false;
}

export function isReady() {
    if (engineReady && wasmModule && wasmModule.is_engine_ready) {
        return wasmModule.is_engine_ready();
    }
    return engineReady;
}

export async function getBackendInfo() {
    return {
        name: "C++ WASM (WebGPU)",
        type: "GPU",
        hasCUDA: false
    };
}

// Aliases for compatibility
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
    loadCppCaptionEngine,
    isCppEngineReady,
    renderWithCppEngine
};
