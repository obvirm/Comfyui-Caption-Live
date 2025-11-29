import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CaptionLive.PreviewWASM_v2",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CaptionLiveNode") {

            let CaptionEngine;
            try {
                // Cache busting to ensure latest WASM is loaded
                const modulePath = "/extensions/caption-live/js/pkg/caption_live_wasm.js?v=" + Date.now();
                const module = await import(modulePath);

                const init = module.default;
                CaptionEngine = module.CaptionEngine;

                await init();
                console.log("🦀 CaptionLive WASM Initialized Successfully!");
            } catch (e) {
                console.error("❌ Failed to load CaptionLive WASM (Absolute Path):", e);

                try {
                    const module = await import("./pkg/caption_live_wasm.js?v=" + Date.now());
                    const init = module.default;
                    CaptionEngine = module.CaptionEngine;
                    await init();
                    console.log("🦀 CaptionLive WASM Initialized (Relative Path)!");
                } catch (e2) {
                    console.error("❌ Failed to load CaptionLive WASM (Relative Path):", e2);
                }
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Minimal padding - almost flush with node border
                const nodePadding = 3;

                // --- HTML Widget Setup ---
                const container = document.createElement("div");
                container.style.width = `calc(100% - ${nodePadding * 2}px)`;
                container.style.height = "200px";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#202020";
                container.style.marginTop = "0px";
                container.style.marginBottom = `${nodePadding}px`;
                container.style.marginLeft = `${nodePadding}px`;
                container.style.marginRight = `${nodePadding}px`;
                container.style.borderRadius = "8px";
                container.style.overflow = "hidden";
                container.style.border = "none";
                container.style.boxSizing = "border-box";

                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "100%";
                canvas.style.display = "block";
                container.appendChild(canvas);

                // --- Play/Pause Button ---
                const playBtn = document.createElement("div");
                playBtn.innerText = "⏸";
                playBtn.style.position = "absolute";
                playBtn.style.top = "5px";
                playBtn.style.right = "5px";
                playBtn.style.width = "24px";
                playBtn.style.height = "24px";
                playBtn.style.backgroundColor = "rgba(0,0,0,0.5)";
                playBtn.style.color = "white";
                playBtn.style.borderRadius = "50%";
                playBtn.style.textAlign = "center";
                playBtn.style.lineHeight = "24px";
                playBtn.style.cursor = "pointer";
                playBtn.style.fontSize = "12px";
                playBtn.style.zIndex = "10";
                playBtn.style.userSelect = "none";

                // Make container relative so absolute button works
                container.style.position = "relative";
                container.appendChild(playBtn);

                let isPlaying = false;
                playBtn.innerText = isPlaying ? "⏸" : "▶";
                playBtn.onclick = (e) => {
                    e.stopPropagation(); // Prevent selecting node
                    isPlaying = !isPlaying;
                    playBtn.innerText = isPlaying ? "⏸" : "▶";
                };


                this.addDOMWidget("preview", "canvas", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                // --- Auto-Resize on Ratio Change ---
                // --- Auto-Resize on Ratio Change ---
                const ratioWidget = this.widgets?.find(w => w.name === "aspect_ratio");
                if (ratioWidget) {
                    ratioWidget.callback = (value) => {
                        // Trigger resize logic to snap to new ratio immediately
                        if (this.onResize && this.size) {
                            this.onResize(this.size);
                        }
                        app.graph.setDirtyCanvas(true, true);
                    };
                }

                // --- Render Loop ---
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

                    const segmentsWidget = this.widgets?.find(w => w.name === "segments");
                    const styleWidget = this.widgets?.find(w => w.name === "style");
                    const fontSizeWidget = this.widgets?.find(w => w.name === "font_size");
                    const posXWidget = this.widgets?.find(w => w.name === "pos_x");
                    const posYWidget = this.widgets?.find(w => w.name === "pos_y");
                    const highlightWidget = this.widgets?.find(w => w.name === "highlight_color");
                    const textWidget = this.widgets?.find(w => w.name === "text_color");

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
                    const logicalWidth = (this.size ? this.size[0] : 200) - 6;

                    // Height: Parse from the container style set by onResize/onDrawForeground
                    let logicalHeight = 200;
                    if (container.style.height) {
                        logicalHeight = parseFloat(container.style.height);
                    }

                    // Lock internal resolution to Logical Size * DPR
                    const targetWidth = Math.round(logicalWidth * dpr);
                    const targetHeight = Math.round(logicalHeight * dpr);

                    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
                        canvas.width = targetWidth;
                        canvas.height = targetHeight;
                    }

                    if (isPlaying && engine) {
                        // Skip render if canvas is too small
                        if (canvas.width > 50 && canvas.height > 50) {
                            const fontPathWidget = this.widgets?.find(w => w.name === "font_path");
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

                                const gpuWidget = this.widgets?.find(w => w.name === "gpu_acceleration");
                                const useGpu = gpuWidget && gpuWidget.value === true;

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
                                    // Note: Switching contexts on the same canvas (2D <-> WebGPU) can be tricky.
                                    // Ideally, we should stick to one, but for now we try to coexist or just use what's requested.
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

                const onRemoved = this.onRemoved;
                this.onRemoved = function () {
                    if (onRemoved) onRemoved.apply(this, arguments);
                    cancelAnimationFrame(animationFrameId);
                };

                return r;
            };

            // --- Helper: Parse Aspect Ratio ---
            const getAspectRatio = (str) => {
                if (!str || str === "Custom") return null;
                const parts = str.split(":");
                if (parts.length === 2) {
                    return parseFloat(parts[0]) / parseFloat(parts[1]);
                }
                return null;
            };

            // --- Responsive Resizing Logic ---
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }

                if (!this.widgets) return;

                const previewWidget = this.widgets.find(w => w.name === "preview");
                const ratioWidget = this.widgets.find(w => w.name === "aspect_ratio");

                if (!previewWidget) return;

                // 1. Calculate height needed for inputs (excluding preview)
                const prevIndex = this.widgets.indexOf(previewWidget);
                let inputsHeight = 0;

                if (prevIndex > -1) {
                    this.widgets.splice(prevIndex, 1);
                    inputsHeight = this.computeSize([size[0], 0])[1];
                    this.widgets.splice(prevIndex, 0, previewWidget);
                }

                // 2. Determine Target Height
                const padding = 25;
                let targetPreviewHeight = 200; // Default min
                const targetRatio = ratioWidget ? getAspectRatio(ratioWidget.value) : null;

                if (targetRatio) {
                    // Fixed Aspect Ratio Mode
                    // Width of the widget is roughly Node Width - (Margin * 2)
                    // We assume margin is around 6px total (3px each side)
                    const widgetWidth = size[0] - 6;
                    targetPreviewHeight = widgetWidth / targetRatio;
                } else {
                    // Custom/Fill Mode
                    const availableSpace = size[1] - inputsHeight - padding;
                    targetPreviewHeight = Math.max(200, availableSpace);
                }

                const targetTotalHeight = inputsHeight + targetPreviewHeight + padding;

                // 3. Auto-grow or Auto-shrink (if fixed ratio)
                // Only resize node if the difference is significant to avoid loop
                if (Math.abs(size[1] - targetTotalHeight) > 5) {
                    this.setSize([size[0], targetTotalHeight]);
                }
            };

            // --- Layout Sync ---
            // Ensures the preview widget fits perfectly in the remaining space
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                if (this.flags.collapsed) return;

                const previewWidget = this.widgets?.find(w => w.name === "preview");
                const ratioWidget = this.widgets?.find(w => w.name === "aspect_ratio");

                if (previewWidget && previewWidget.element) {
                    const widgetY = previewWidget.last_y;

                    if (widgetY !== undefined) {
                        let height = 0;
                        const targetRatio = ratioWidget ? getAspectRatio(ratioWidget.value) : null;

                        if (targetRatio) {
                            // Force height based on width and ratio
                            const widgetWidth = this.size[0] - 6; // Approximate width
                            height = widgetWidth / targetRatio;
                        } else {
                            // Fill remaining space
                            const padding = 25;
                            height = this.size[1] - widgetY - padding;
                        }

                        // Prevent negative height
                        if (height < 0) height = 0;

                        previewWidget.element.style.height = height + "px";
                    }
                }
            };
        }
    }
});
