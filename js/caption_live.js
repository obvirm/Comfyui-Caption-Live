import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CaptionLive.PreviewWASM_v2",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CaptionLiveNode") {

            let CaptionEngine;
            try {
                const modulePath = "/extensions/caption-live/js/pkg/caption_live_wasm.js";
                const module = await import(modulePath);

                const init = module.default;
                CaptionEngine = module.CaptionEngine;

                await init();
                console.log("🦀 CaptionLive WASM Initialized Successfully!");
            } catch (e) {
                console.error("❌ Failed to load CaptionLive WASM (Absolute Path):", e);

                try {
                    const module = await import("./pkg/caption_live_wasm.js");
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

                this.addDOMWidget("preview", "canvas", container, {
                    serialize: false,
                    hideOnZoom: false
                });

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

                    let segmentsStr = segmentsWidget ? segmentsWidget.value : "[]";
                    const style = styleWidget ? styleWidget.value : "box";
                    const fontSizeRaw = fontSizeWidget ? fontSizeWidget.value : 56.0;
                    const posXRaw = posXWidget ? posXWidget.value : 0;
                    const posYRaw = posYWidget ? posYWidget.value : 0;

                    // Sanitize JSON input for WASM
                    try {
                        // 1. Replace single quotes with double quotes (common user/LLM error)
                        segmentsStr = segmentsStr.replace(/'/g, '"');
                        
                        // 2. Extract the JSON array part if there's surrounding text
                        const firstOpen = segmentsStr.indexOf("[");
                        const lastClose = segmentsStr.lastIndexOf("]");
                        
                        if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
                            segmentsStr = segmentsStr.substring(firstOpen, lastClose + 1);
                        }
                    } catch (e) {
                        // If sanitization fails, pass original string and let engine handle/fail
                    }

                    if (segmentsStr !== lastSegments && CaptionEngine) {
                        try {
                            engine = new CaptionEngine(segmentsStr);
                            lastSegments = segmentsStr;
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

                    if (engine) {
                        // Skip render if canvas is too small
                        if (canvas.width > 50 && canvas.height > 50) {
                            try {
                                const time = (Date.now() / 1000) % 10.0;
                                
                                // Calculate pixel font size based on zoom/DPI
                                // We multiply by app.canvas.ds.scale (zoom level) to keep text proportional to the node
                                const zoomLevel = app.canvas.ds.scale || 1;
                                const targetFontSizePx = fontSizeRaw * dpr;
                                
                                // Position offsets (also scaled by DPR to match canvas coordinates)
                                const targetPosX = posXRaw * dpr;
                                const targetPosY = posYRaw * dpr;

                                engine.draw_frame(ctx, canvas.width, canvas.height, time, style, targetFontSizePx, targetPosX, targetPosY);
                            } catch (e) {
                                console.warn("CaptionLive Render Error:", e);
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
