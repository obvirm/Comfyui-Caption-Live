/**
 * Caption Live Renderer - Unified WASM Version
 * 
 * Uses caption_engine WASM for both preview and final render.
 * This ensures pixel-perfect parity between browser preview and backend output.
 */
import { app } from "../../scripts/app.js";
import { AspectRatioManager } from "./aspect-ratio.js";
import { loadCppCaptionEngine, renderWithCppEngine, isCppEngineReady } from "./cpp-engine.js";

// --- GLOBAL VARIABLES ---
let captionEngine = null;
let gpuManager = null;
let engineLoaded = false;
let engineFailed = false;

// Load Caption Engine WASM (Prioritize C++ Engine)
async function loadCaptionEngine() {
    // Try C++ Engine first
    const cppEngine = await loadCppCaptionEngine();
    if (cppEngine) {
        captionEngine = cppEngine; // Fix: Assign to global
        engineLoaded = true;
        console.log("✅ Using C++ Caption Engine (WASM)");
        return cppEngine;
    }

    // Fallback logic for legacy Rust engine (if needed)
    if (engineLoaded) return captionEngine;
    if (engineFailed) return null;


    const modulePath = "/extensions/caption-live/lib/caption_engine_wasm/caption_engine.js";

    return new Promise((resolve) => {
        const timeout = setTimeout(() => {
            console.warn("Caption Engine load timeout, using Canvas 2D fallback");
            engineFailed = true;
            resolve(null);
        }, 10000);

        import(modulePath)
            .then(async (module) => {
                clearTimeout(timeout);
                await module.default(); // Initialize WASM
                captionEngine = module;

                try {
                    // Initialize GPU Compute Manager
                    if (module.WasmComputeManager) {
                        console.log("⚡ Initializing WebGPU Compute Manager...");
                        gpuManager = await module.WasmComputeManager.new();
                        console.log("⚡ WebGPU Compute Manager Initialized!");

                        // Load Test Effect (Liquid)
                        const testEffect = `
effect: "test_liquid"
parameters:
  intensity: float
kernel:
  type: "compute"
  workgroup_size: [64, 1, 1]
  precision: "fp32"
  code: |
    // Simple verification: double the input
    output_data[index] = input_data[index] * 2.0;
                        `;

                        gpuManager.load_effect(testEffect);

                        // Run a tiny test
                        const input = new Float32Array([1.0, 2.0, 3.0, 4.0]);
                        const params = new Float32Array([1.0]); // dummy param
                        const result = await gpuManager.execute_effect(input, params);
                        console.log("🌊 GPU Test Result (Should be doubled):", result);
                    }
                } catch (gpuError) {
                    console.warn("⚠️ WebGPU init failed (using CPU fallback):", gpuError);
                    // Optional: Display this error on the canvas for user visibility
                    const ctx = document.createElement('canvas').getContext('2d');
                    // We can't easily draw to the main canvas here without passing it, 
                    // but the console warning is key.
                }

                engineLoaded = true;
                console.log("🎬 Caption Engine WASM Loaded!");
                resolve(captionEngine);
            })
            .catch((e) => {
                clearTimeout(timeout);
                console.warn("Caption Engine failed to load, using Canvas 2D fallback", e);
                engineFailed = true;
                resolve(null);
            });
    });
}

// Build template JSON from node widgets
function buildTemplate(node, canvas, currentTime, duration) {
    const segmentsWidget = node.widgets?.find(w => w.name === "segments");
    const segmentsStr = segmentsWidget ? String(segmentsWidget.value) : "[]";

    // Check for 'text' widget (standard input)
    const textWidget = node.widgets?.find(w => w.name === "text");
    const rawText = textWidget ? String(textWidget.value) : "Hello World";

    const fontSize = node.widgets?.find(w => w.name === "font_size")?.value || 50;
    const style = node.widgets?.find(w => w.name === "style")?.value || "box";
    const highlight = node.widgets?.find(w => w.name === "highlight_color")?.value || "#39E55F";
    const textCol = node.widgets?.find(w => w.name === "text_color")?.value || "#FFFFFF";
    const strokeColor = node.widgets?.find(w => w.name === "stroke_color")?.value || "#000000";
    const strokeWidth = node.widgets?.find(w => w.name === "stroke_width")?.value || 4;
    const posX = node.widgets?.find(w => w.name === "pos_x")?.value || 0.5;
    const posY = node.widgets?.find(w => w.name === "pos_y")?.value || 0.8;

    // Parse segments
    let segments = [];
    try {
        segments = JSON.parse(segmentsStr.replace(/'/g, '"'));
    } catch (e) {
        segments = [];
    }

    // If segments are empty, build from rawText
    if (!segments || segments.length === 0) {
        // Split by words for simple simulation
        const words = rawText.split(/\s+/);
        const timePerWord = duration / Math.max(1, words.length);
        segments = words.map((word, i) => ({
            text: word,
            start: i * timePerWord,
            end: (i + 1) * timePerWord
        }));
    }

    // Build animation based on style
    let animation = null;
    if (style === "box" || style === "box_highlight") {
        animation = {
            type: "box_highlight",
            segments: segments,
            box_color: highlight,
            box_radius: 8.0,
            box_padding: 8.0
        };
    } else if (style === "typewriter") {
        animation = {
            type: "typewriter",
            segments: segments
        };
    } else if (style === "bounce") {
        animation = {
            type: "bounce",
            segments: segments,
            intensity: 1.2
        };
    } else if (style === "colored") {
        animation = {
            type: "colored",
            segments: segments,
            active_color: highlight
        };
    }

    // Combine all segment texts for content
    const content = segments.map(s => s.text).join(" ");

    // Smart Font Scaling
    // Scale relative to 1080p height.
    // 16:9 (1080h) -> 1.0x
    // 9:16 (1920h) -> 1.77x (Text grows for vertical video, typical for TikTok)
    const scaleFactor = canvas.height / 1080.0;

    const template = {
        canvas: { width: canvas.width, height: canvas.height },
        duration: duration,
        fps: 60.0,
        layers: [
            {
                type: "text",
                content: content,
                style: {
                    font_size: fontSize * scaleFactor,
                    color: textCol,
                    stroke_color: strokeColor,
                    stroke_width: strokeWidth
                },
                position: { x: posX, y: posY },
                animation: animation
            }
        ]
    };

    return JSON.stringify(template);
}

// Canvas 2D Fallback Renderer
function renderWithCanvas2D(ctx, canvas, node, currentTime, duration) {
    const segmentsWidget = node.widgets?.find(w => w.name === "segments");
    const segmentsStr = segmentsWidget ? String(segmentsWidget.value) : "[]";
    const fontSize = node.widgets?.find(w => w.name === "font_size")?.value || 50;
    const textCol = node.widgets?.find(w => w.name === "text_color")?.value || "#FFFFFF";

    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    try {
        const segments = JSON.parse(segmentsStr.replace(/'/g, '"'));
        const segment = segments.find(s => currentTime >= s.start && currentTime < s.end);
        if (segment) {
            ctx.font = `${fontSize}px Arial`;
            ctx.fillStyle = textCol;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(segment.text, canvas.width / 2, canvas.height / 2);
        }
    } catch (e) {
        ctx.font = "20px Arial";
        ctx.fillStyle = "#888";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Preview Ready (Canvas 2D)", canvas.width / 2, canvas.height / 2);
    }
}

// Main Render Loop
export function startRenderLoop(node, canvas, _OldEngine, app, getIsPlaying, isGpu = false, effectsCache = {}) {
    // Initialize Caption Engine (and GPU Manager)
    loadCaptionEngine();

    let lastTemplateJson = "";
    let cachedImageData = null;
    let lastEffectName = "";
    let lastParamsHash = "";

    // Test pattern texture cache
    let inputTexture = null;
    let inputTextureWidth = 0;
    let inputTextureHeight = 0;

    // Resize throttling state
    let lastResizeTime = 0;

    // Performance optimization
    let frameCount = 0;
    let lastRenderTime = 0;
    const TARGET_FPS = 30; // Throttle to 30fps for preview (saves CPU/GPU)
    const FRAME_TIME = 1000 / TARGET_FPS;
    let cachedTemplate = null;
    let lastTemplateHash = "";

    // Preview resolution scale (1.0 = full resolution for accurate preview)
    const PREVIEW_SCALE = 1.0;
    const render = async () => {
        frameCount++;

        // DEBUG: Check loop life
        // if (frameCount % 60 === 0) {
        // console.log(`[Renderer] Frame ${frameCount} | Connected: ${canvas.isConnected} | W: ${canvas.width} H: ${canvas.height} | GPU: ${isGpu} | Loaded: ${engineLoaded} | Manager: ${!!gpuManager}`);
        // }

        if (!canvas.isConnected) {
            // Keep requesting frame even if not connected, to resume when reconnected
            requestAnimationFrame(render);
            return;
        }
        // Check Play/Pause State
        if (!getIsPlaying()) {
            requestAnimationFrame(render);
            return;
        }

        const duration = node.widgets?.find(w => w.name === "duration")?.value || 5.0;
        // For GPU effects, we might use "param2" as time sometimes
        const time = (Date.now() / 1000) % duration;

        // Frame throttling - skip frames to maintain target FPS
        const now = performance.now();
        if (now - lastRenderTime < FRAME_TIME) {
            requestAnimationFrame(render);
            return;
        }
        lastRenderTime = now;

        // --- RESIZE LOGIC (SCALED FOR PREVIEW PERFORMANCE) ---
        const ratioWidget = node.widgets?.find(w => w.name === "aspect_ratio");
        const ratioStr = ratioWidget ? ratioWidget.value : "16:9";

        // Use AspectRatioManager for resolution lookup, but SCALE DOWN for preview
        const resolution = AspectRatioManager.getResolution(ratioStr);
        const targetW = Math.round(resolution.width * PREVIEW_SCALE);
        const targetH = Math.round(resolution.height * PREVIEW_SCALE);
        const cssRatio = AspectRatioManager.toCssRatio(ratioStr);

        // Apply CSS Aspect Ratio to container (Responsive resizing)
        if (canvas.parentNode && canvas.parentNode.style.aspectRatio !== cssRatio) {
            canvas.parentNode.style.aspectRatio = cssRatio;
        }

        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
            cachedImageData = null;
            inputTexture = null;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) {
            requestAnimationFrame(render);
            return;
        }

        // DEBUG: Force visible background if nothing else draws
        ctx.fillStyle = "#FF00FF"; // Magenta
        ctx.fillRect(0, 0, 10, 10); // Tiny dot in corner

        // Use Caption Engine WASM if available
        if (engineLoaded && captionEngine) {
            if (isGpu) {
                if (gpuManager) {
                    // --- GPU RENDERER ---
                    await renderGPUPreview(ctx, canvas, node, effectsCache, time);
                } else {
                    // --- GPU NODE FALLBACK (No WebGPU) ---
                    if (frameCount % 120 === 0) console.log("[Renderer] Fallback Active - Drawing warning");

                    // Directly draw warning message using Canvas 2D
                    // We don't use captionEngine here because the GPU node lacks text widgets for the template.

                    // Background: Dark Blue to distinguish from "broken" black
                    ctx.fillStyle = "#000033";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    // Error Text
                    ctx.fillStyle = "#ff5555";
                    ctx.font = "bold 14px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("⚠️ WebGPU Not Available", canvas.width / 2, canvas.height / 2 - 20);

                    // Info Text
                    ctx.fillStyle = "#cccccc";
                    ctx.font = "12px Arial";
                    ctx.fillText("Preview disabled in browser.", canvas.width / 2, canvas.height / 2 + 10);
                    ctx.fillText("Rendering happens on server.", canvas.width / 2, canvas.height / 2 + 30);
                }
            } else {
                // --- CAPTION RENDERER (CPU/Standard) ---
                try {
                    const templateJson = buildTemplate(node, canvas, time, duration);

                    // Try rendering with C++ Engine first
                    const success = await renderWithCppEngine(canvas, ctx, templateJson, time);

                    if (!success) {
                        // Fallback to legacy Rust/JS rendering if C++ failed or not ready
                        // (Existing legacy code logic would be here if we kept it, 
                        //  but for now assuming C++ takes over if loaded)

                        if (captionEngine && captionEngine.render_frame_rgba && !isCppEngineReady()) {
                            // Legacy Rust path (only if C++ not ready)
                            const rgba = captionEngine.render_frame_rgba(templateJson, time);
                            const imageData = new ImageData(
                                new Uint8ClampedArray(rgba),
                                canvas.width,
                                canvas.height
                            );
                            ctx.putImageData(imageData, 0, 0);
                        }
                    }
                } catch (e) {
                    console.error("Caption Engine render error:", e);
                    renderWithCanvas2D(ctx, canvas, node, time, duration);
                }
            }
        } else if (engineFailed) {
            // Fallback to Canvas 2D
            renderWithCanvas2D(ctx, canvas, node, time, duration);
        } else {
            // Loading state
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#aaa";
            ctx.font = "16px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Loading Caption Engine...", canvas.width / 2, canvas.height / 2);
        }

        requestAnimationFrame(render);
    };

    requestAnimationFrame(render);

    // Cleanup on node removal
    node.onRemoved = () => {
        // Nothing to clean up for WASM renderer
    };
}
