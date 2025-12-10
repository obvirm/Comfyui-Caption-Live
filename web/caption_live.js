// caption_live.js - Main ComfyUI Extension Entry Point
// Uses Professional Frontend Preview Architecture v2.0

import { app } from "../../scripts/app.js";
import { createPreviewUI } from "./ui.js";
import { setupLayout, setupWidgetCallbacks } from "./layout.js";
import { startRenderLoop } from "./renderer.js";
import { SafeAreaOverlay } from "./safe-area-overlay.js";
import { PreviewOptimizer } from "./preview-optimizer.js";
import { VERSION } from "./index.js";

// Load Color Picker Widget (auto-registers)
import "./color-widget.js";

// Cache for effects definitions
let effectsCache = null;

async function fetchEffects() {
    if (effectsCache) return effectsCache;
    try {
        const response = await fetch("/caption-live/effects");
        effectsCache = await response.json();
        console.log("🎨 Loaded effects:", Object.keys(effectsCache));
        return effectsCache;
    } catch (e) {
        console.error("Failed to load effects:", e);
        return {};
    }
}

app.registerExtension({
    name: "CaptionLive.PreviewWASM_v2",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CaptionLiveNode" || nodeData.name === "CaptionLiveGPUNode") {

            // Pre-fetch effects if GPU node
            if (nodeData.name === "CaptionLiveGPUNode") {
                await fetchEffects();
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

                // Initialize Preview Optimizer for adaptive quality
                const optimizer = new PreviewOptimizer();
                optimizer.adaptiveEnabled = true;
                this.previewOptimizer = optimizer;

                // Add quality indicator to container
                const qualityIndicator = optimizer.createIndicatorElement();
                container.appendChild(qualityIndicator);

                // Initialize Safe Area Overlay (hidden by default)
                const safeArea = new SafeAreaOverlay(canvas);
                safeArea.init();
                this.safeAreaOverlay = safeArea;

                // Add safe area toggle button
                const safeAreaBtn = document.createElement("div");
                safeAreaBtn.innerText = "📐";
                safeAreaBtn.title = "Toggle Safe Area Guides";
                safeAreaBtn.style.cssText = `
                    position: absolute;
                    top: 5px;
                    left: 5px;
                    width: 24px;
                    height: 24px;
                    background: rgba(0,0,0,0.5);
                    color: white;
                    border-radius: 50%;
                    text-align: center;
                    line-height: 24px;
                    cursor: pointer;
                    font-size: 12px;
                    z-index: 10;
                    user-select: none;
                `;
                safeAreaBtn.onclick = (e) => {
                    e.stopPropagation();
                    safeArea.toggle();
                };
                container.appendChild(safeAreaBtn);

                // Widget Callbacks (e.g. Aspect Ratio change)
                setupWidgetCallbacks(this, app);

                // Update safe area when aspect ratio changes
                const ratioWidget = this.widgets?.find(w => w.name === "aspect_ratio");
                if (ratioWidget) {
                    const originalCallback = ratioWidget.callback;
                    ratioWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback(value);
                        }
                        safeArea.setAspectRatio(value);
                        safeArea.resize();
                    };
                }

                // Start Render Loop
                const isGpu = this.widgets?.some(w => w.name === "effect_name") || nodeData.name === "CaptionLiveGPUNode";
                console.log(`[CaptionLive v${VERSION}] Node Created: ${nodeData.name} | isGpu: ${isGpu}`);

                startRenderLoop(this, canvas, null, app, getIsPlaying, isGpu, effectsCache);

                return r;
            };

            // Cleanup on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }

                // Cleanup safe area overlay
                if (this.safeAreaOverlay) {
                    this.safeAreaOverlay.destroy();
                }
            };
        }
    }
});

console.log(`🎬 CaptionLive Preview System v${VERSION} loaded`);
