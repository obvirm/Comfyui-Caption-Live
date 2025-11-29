export function startRenderLoop(node, canvas, CaptionEngine, app, getIsPlaying) {
    // --- Render Loop State ---
    let engine = null;
    let lastSegments = "";
    let animationFrameId;
    let isGpuInitialized = false;
    let isGpuInitializing = false;

    const ctx = canvas.getContext("2d");

    const render = () => {
        // Keep loop alive even if not connected yet (crucial for page reloads)
        if (!canvas.isConnected) {
            animationFrameId = requestAnimationFrame(render);
            return;
        }

        const segmentsWidget = node.widgets?.find(w => w.name === "segments");
        const styleWidget = node.widgets?.find(w => w.name === "style");
        const fontSizeWidget = node.widgets?.find(w => w.name === "font_size");
        const posXWidget = node.widgets?.find(w => w.name === "pos_x");
        const posYWidget = node.widgets?.find(w => w.name === "pos_y");
        const highlightWidget = node.widgets?.find(w => w.name === "highlight_color");
        const textWidget = node.widgets?.find(w => w.name === "text_color");

        // Check if segments widget exists (it might be converted to input)
        if (!segmentsWidget) {
            // Draw "Preview Unavailable" message
            ctx.fillStyle = "#202020";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#888";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Preview Unavailable", canvas.width / 2, canvas.height / 2 - 10);
            ctx.fillText("(Input Connected)", canvas.width / 2, canvas.height / 2 + 10);

            animationFrameId = requestAnimationFrame(render);
            return;
        }

        let segmentsStr = (segmentsWidget && segmentsWidget.value != null) ? String(segmentsWidget.value) : "[]";
        // Ensure empty string also falls back to default for critical UI elements if needed, 
        // but for text fields, empty string might be valid. 
        // For style/colors, we want defaults if empty.
        const style = (styleWidget && styleWidget.value) ? String(styleWidget.value) : "box";
        const fontSizeRaw = (fontSizeWidget && fontSizeWidget.value != null) ? Number(fontSizeWidget.value) : 56.0;
        const posXRaw = (posXWidget && posXWidget.value != null) ? Number(posXWidget.value) : 0;
        const posYRaw = (posYWidget && posYWidget.value != null) ? Number(posYWidget.value) : 0;
        const highlightColor = (highlightWidget && highlightWidget.value) ? String(highlightWidget.value) : "#39E55F";
        const textColor = (textWidget && textWidget.value) ? String(textWidget.value) : "#FFFFFF";

        // Smart JSON Sanitization
        try {
            // 1. Try parsing as-is first (Best for valid JSON with apostrophes like "It's")
            JSON.parse(segmentsStr);
        } catch (e) {
            // 2. If failed, try fixing common Python/LLM errors (single quotes)
            // Only do this if initial parse fails, to avoid breaking valid text containing '
            try {
                let fixedStr = segmentsStr.replace(/'/g, '"');
                // Extract JSON array if embedded in text
                const firstOpen = fixedStr.indexOf("[");
                const lastClose = fixedStr.lastIndexOf("]");
                if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
                    fixedStr = fixedStr.substring(firstOpen, lastClose + 1);
                }
                segmentsStr = fixedStr;
            } catch (e2) {
                // Give up, pass original
            }
        }

        if (segmentsStr !== lastSegments && CaptionEngine) {
            try {
                engine = new CaptionEngine(segmentsStr);
                lastSegments = segmentsStr;
                // Reset GPU init status when engine is recreated
                isGpuInitialized = false;
                isGpuInitializing = false;
            } catch (e) {
                // Silent
            }
        }

        // --- STABLE RENDERING LOGIC (Smart Zoom + Supersampling) ---
        // "Ubah size jadi koordinat z" interpretation:
        // We use the Node's Logical Size (stable) but multiply by the Zoom Level (Z-coord/scale)
        // to determine the resolution.
        // ADDED: * 2.0 for Supersampling to make it ultra-sharp ("tajam").
        const zoom = app.canvas.ds.scale || 1;
        const dpr = (window.devicePixelRatio || 1) * zoom * 2.0;

        // Width: Node Width - Margins (approx 6px total)
        const logicalWidth = (node.size ? node.size[0] : 200) - 6;

        // Height: Parse from the container style set by onResize/onDrawForeground
        let logicalHeight = 200;
        if (canvas.parentElement && canvas.parentElement.style.height) {
            logicalHeight = parseFloat(canvas.parentElement.style.height);
        }

        // Lock internal resolution to Logical Size * DPR
        const targetWidth = Math.round(logicalWidth * dpr);
        const targetHeight = Math.round(logicalHeight * dpr);

        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }

        const isPlaying = getIsPlaying();

        if (isPlaying && engine) {
            // Skip render if canvas is too small
            if (canvas.width > 50 && canvas.height > 50) {
                const fontPathWidget = node.widgets?.find(w => w.name === "font_path");
                let fontFamily = "Arial";
                if (fontPathWidget && fontPathWidget.value) {
                    const path = String(fontPathWidget.value);
                    const filename = path.split(/[\\/]/).pop();
                    if (filename) {
                        fontFamily = filename.split('.')[0];
                    }
                }

                try {
                    const time = (Date.now() / 1000) % 10.0;
                    const scaleFactor = canvas.width / 210.0;
                    const targetFontSizePx = fontSizeRaw * scaleFactor;
                    const targetPosX = posXRaw * scaleFactor;
                    const targetPosY = posYRaw * scaleFactor;

                    const gpuWidget = node.widgets?.find(w => w.name === "gpu_acceleration");
                    // Force disable GPU for preview to prevent context conflicts and spam
                    const useGpu = false; // gpuWidget && gpuWidget.value === true;

                    if (useGpu) {
                        if (!isGpuInitialized && !isGpuInitializing) {
                            isGpuInitializing = true;
                            // Note: init_gpu is async. We start it and wait.
                            engine.init_gpu(canvas).then(() => {
                                console.log("WebGPU Initialized!");
                                isGpuInitialized = true;
                                isGpuInitializing = false;
                            }).catch(e => {
                                console.error("WebGPU Init Failed:", e);
                                isGpuInitializing = false;
                            });
                        }

                        if (isGpuInitialized) {
                            engine.draw_frame_gpu(canvas.width, canvas.height, time, style, targetFontSizePx, targetPosX, targetPosY, highlightColor, textColor, fontFamily);
                        }
                    } else {
                        // Fallback to CPU/Canvas2D if GPU is disabled or not ready
                        engine.draw_frame(ctx, canvas.width, canvas.height, time, style, targetFontSizePx, targetPosX, targetPosY, highlightColor, textColor, fontFamily);
                    }
                } catch (e) {
                    console.warn("CaptionLive Render Error:", e);
                    console.warn("Debug Args:", { style, highlightColor, textColor, segmentsStr, fontSizeRaw, posXRaw, posYRaw, fontFamily });
                }
            }
        }

        animationFrameId = requestAnimationFrame(render);
    };

    requestAnimationFrame(render);

    // Cleanup on node removal
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (onRemoved) onRemoved.apply(this, arguments);
        cancelAnimationFrame(animationFrameId);
    };
}
