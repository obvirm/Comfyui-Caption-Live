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

// Load Caption Engine WASM (C++ Engine ONLY - no fallbacks)
async function loadCaptionEngine() {
    if (engineLoaded) return captionEngine;
    if (engineFailed) return null;

    // Load C++ WASM Engine
    const cppEngine = await loadCppCaptionEngine();
    if (cppEngine) {
        captionEngine = cppEngine;
        engineLoaded = true;
        console.log("✅ C++ Caption Engine (WASM) loaded successfully!");
        return cppEngine;
    }

    // C++ Engine failed - NO fallback available
    console.error("❌ C++ WASM Engine failed to load! No fallback available.");
    console.error("   Build WASM: cd core && ./build_wasm.ps1");
    engineFailed = true;
    return null;
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
                    // Stroke scaling is handled by C++ Engine (Backend) automatically
                    stroke_width: strokeWidth,
                    outline_width: strokeWidth,
                    outlineWidth: strokeWidth,
                    stroke: strokeWidth,
                    outline: strokeWidth,
                    thickness: strokeWidth
                },
                position: { x: posX, y: posY },
                animation: animation
            }
        ]
    };

    return JSON.stringify(template);
}

// GPU Engine Status Display (NO Canvas 2D rendering - status only)
function displayEngineError(ctx, canvas, errorMessage) {
    // Clear with dark background
    ctx.fillStyle = "#0d0d1a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Error icon and message (minimal status display, not rendering)
    ctx.fillStyle = "#ff4444";
    ctx.font = "bold 14px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("⚠️ C++ Engine Required", canvas.width / 2, canvas.height / 2 - 20);

    ctx.fillStyle = "#888888";
    ctx.font = "12px monospace";
    ctx.fillText(errorMessage, canvas.width / 2, canvas.height / 2 + 10);
    ctx.fillText("Build WASM: cd core && ./build_wasm.ps1", canvas.width / 2, canvas.height / 2 + 30);
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

                    // Display WebGPU not available error (status display only, no content rendering)\n                    // GPU effects require WebGPU - when unavailable, rendering happens on server only
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
                // --- CAPTION RENDERER (C++ WASM Only) ---
                try {
                    const templateJson = buildTemplate(node, canvas, time, duration);

                    // Render with C++ Engine (NO fallback)
                    const success = await renderWithCppEngine(canvas, ctx, templateJson, time);

                    if (!success) {
                        displayEngineError(ctx, canvas, "C++ Engine render failed");
                    }
                } catch (e) {
                    console.error("Caption Engine render error:", e);
                    displayEngineError(ctx, canvas, `Render Error: ${e.message}`);
                }
            }
        } else if (engineFailed) {
            // NO Canvas 2D fallback - display error
            displayEngineError(ctx, canvas, "C++ WASM Engine failed to load");
        } else {
            // Loading state - minimal status display
            ctx.fillStyle = "#0d0d1a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#4488ff";
            ctx.font = "14px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("⏳ Loading C++ Engine (WASM)...", canvas.width / 2, canvas.height / 2);
        }

        requestAnimationFrame(render);
    };

    requestAnimationFrame(render);

    // Cleanup on node removal
    node.onRemoved = () => {
        // Nothing to clean up for WASM renderer
    };
}
