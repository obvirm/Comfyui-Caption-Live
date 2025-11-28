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

                // --- Render Loop ---
                let engine = null;
                let lastSegments = "";
                let animationFrameId;

                const ctx = canvas.getContext("2d");

                const render = () => {
                    if (!canvas.isConnected) return;

                    const segmentsWidget = this.widgets?.find(w => w.name === "segments");
                    const styleWidget = this.widgets?.find(w => w.name === "style");

                    const segmentsStr = segmentsWidget ? segmentsWidget.value : "[]";
                    const style = styleWidget ? styleWidget.value : "box";

                    if (segmentsStr !== lastSegments && CaptionEngine) {
                        try {
                            engine = new CaptionEngine(segmentsStr);
                            lastSegments = segmentsStr;
                        } catch (e) {
                            // Silent
                        }
                    }

                    const rect = container.getBoundingClientRect();
                    const dpr = window.devicePixelRatio || 1;

                    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
                        canvas.width = rect.width * dpr;
                        canvas.height = rect.height * dpr;
                    }

                    if (engine) {
                        const time = (Date.now() / 1000) % 10.0;
                        engine.draw_frame(ctx, canvas.width, canvas.height, time, style);
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

            // --- Responsive Resizing Logic ---
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }

                if (!this.widgets) return;

                const previewWidget = this.widgets.find(w => w.name === "preview");
                if (!previewWidget) return;

                // 1. Calculate height needed for inputs (excluding preview)
                const prevIndex = this.widgets.indexOf(previewWidget);
                let inputsHeight = 0;
                
                if (prevIndex > -1) {
                    this.widgets.splice(prevIndex, 1);
                    inputsHeight = this.computeSize([size[0], 0])[1];
                    this.widgets.splice(prevIndex, 0, previewWidget);
                }

                // 2. Sizing Constants
                const minPreviewHeight = 200; 
                const padding = 20;
                const targetTotalHeight = inputsHeight + minPreviewHeight + padding;

                // 3. Auto-grow if too small
                if (size[1] < targetTotalHeight) {
                    this.setSize([size[0], targetTotalHeight]);
                }
                // Note: Actual style height is now handled in onDrawForeground for accuracy
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
                if (previewWidget && previewWidget.element) {
                    // last_y is updated by LiteGraph during the draw phase
                    const widgetY = previewWidget.last_y;
                    
                    if (widgetY !== undefined) {
                        const padding = 25; // Increased space for bottom border
                        let height = this.size[1] - widgetY - padding;
                        
                        // Prevent negative height
                        if (height < 0) height = 0;
                        
                        // Update the DOM element height to match the visual layout
                        previewWidget.element.style.height = height + "px";
                    }
                }
            };
        }
    }
});
