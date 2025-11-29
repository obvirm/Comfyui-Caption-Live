import { app } from "../../scripts/app.js";
import { createPreviewUI } from "./ui.js";
import { setupLayout, setupWidgetCallbacks } from "./layout.js";
import { startRenderLoop } from "./renderer.js";

app.registerExtension({
    name: "CaptionLive.PreviewWASM_v2",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CaptionLiveNode") {

            let CaptionEngine;
            try {
                // Cache busting to ensure latest WASM is loaded
                const modulePath = "/extensions/caption-live/pkg/caption_live_wasm.js?v=" + Date.now();
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

            // Setup Layout Prototype Overrides (Resize & DrawForeground)
            setupLayout(nodeType);

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // UI Setup
                const { container, canvas, getIsPlaying } = createPreviewUI(3);

                this.addDOMWidget("preview", "canvas", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                // Widget Callbacks (e.g. Aspect Ratio change)
                setupWidgetCallbacks(this, app);

                // Start Render Loop
                startRenderLoop(this, canvas, CaptionEngine, app, getIsPlaying);

                return r;
            };
        }
    }
});
