// layout.js - Responsive Layout System using AspectRatioManager
// Based on professional frontend architecture

import { AspectRatioManager } from './aspect-ratio.js';

export function setupLayout(nodeType) {
    // --- Responsive Resizing Logic ---
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
        if (onResize) {
            onResize.apply(this, arguments);
        }

        if (!this.widgets) return;
        if (this.flags.collapsed) return;

        const previewWidget = this.widgets.find(w => w.name === "preview");
        const ratioWidget = this.widgets.find(w => w.name === "aspect_ratio");

        if (!previewWidget) return;

        // Enforce Aspect Ratio Lock if widget exists
        if (ratioWidget && ratioWidget.value) {
            const ratio = AspectRatioManager.parseRatio(ratioWidget.value);
            if (ratio) {
                // Initialize state if missing
                if (!this.lastSize) this.lastSize = [...size];

                // 1. Calculate non-preview height (Inputs)
                const originalWidgets = this.widgets;
                this.widgets = this.widgets.filter(w => w.name !== "preview");
                const nonPreviewDims = this.computeSize([size[0], 0]);
                this.widgets = originalWidgets;
                const nonPreviewHeight = nonPreviewDims[1];

                // 2. Enforce Min Width to prevent layout breaking
                const MIN_WIDTH = 250;
                if (size[0] < MIN_WIDTH) {
                    size[0] = MIN_WIDTH;
                    this.size[0] = MIN_WIDTH;
                }

                // 3. Use AspectRatioManager for EXACT viewport calculation
                const viewport = AspectRatioManager.calculateViewport(
                    size[0], // containerWidth (node width)
                    0,       // containerHeight (auto)
                    ratioWidget.value,
                    'exact'  // Mode: exact aspect ratio
                );

                // 4. Set total node height
                const targetTotalHeight = nonPreviewHeight + viewport.height;
                this.size[1] = Math.round(targetTotalHeight);

                // Update state
                this.lastSize = [...this.size];
            }
        }
    };

    // --- Layout Sync ---
    // Ensures the preview widget matches exact aspect ratio
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        if (onDrawForeground) {
            onDrawForeground.apply(this, arguments);
        }
        if (this.flags.collapsed) return;

        const previewWidget = this.widgets?.find(w => w.name === "preview");
        const ratioWidget = this.widgets?.find(w => w.name === "aspect_ratio");

        if (previewWidget && previewWidget.element && ratioWidget?.value) {
            // Get ACTUAL rendered width of the container
            const actualWidth = previewWidget.element.offsetWidth;

            if (actualWidth > 0) {
                // Use AspectRatioManager for precise calculation
                const viewport = AspectRatioManager.calculateViewport(
                    actualWidth,
                    0,
                    ratioWidget.value,
                    'exact'
                );

                // Force EXACT height (width, min, max for no flexibility)
                const exactHeight = viewport.height;
                previewWidget.element.style.height = exactHeight + "px";
                previewWidget.element.style.minHeight = exactHeight + "px";
                previewWidget.element.style.maxHeight = exactHeight + "px";
            }
        }
    };
}

export function setupWidgetCallbacks(node, app) {
    // --- Auto-Resize on Ratio Change ---
    const ratioWidget = node.widgets?.find(w => w.name === "aspect_ratio");
    if (ratioWidget) {
        ratioWidget.callback = (value) => {
            const ratio = AspectRatioManager.parseRatio(value);
            if (ratio && node.size) {
                const previewWidget = node.widgets?.find(w => w.name === "preview");

                // Get the actual preview width
                let previewWidth = previewWidget?.element?.offsetWidth || 0;
                if (previewWidth === 0) {
                    previewWidth = node.size[0] - 20; // Fallback with margin
                }

                // Use AspectRatioManager for precise calculation
                const viewport = AspectRatioManager.calculateViewport(
                    previewWidth,
                    0,
                    value,
                    'exact'
                );

                // Calculate height of other widgets
                let otherWidgetsHeight = 0;
                if (node.widgets) {
                    for (const w of node.widgets) {
                        if (w.name !== "preview") {
                            otherWidgetsHeight += 24;
                        }
                    }
                }
                otherWidgetsHeight += 40; // Header + footer

                // Resize node
                node.setSize([node.size[0], otherWidgetsHeight + viewport.height]);

                // Also update container height immediately
                if (previewWidget?.element) {
                    previewWidget.element.style.height = viewport.height + "px";
                    previewWidget.element.style.minHeight = viewport.height + "px";
                    previewWidget.element.style.maxHeight = viewport.height + "px";
                }
            }
            app.graph.setDirtyCanvas(true, true);
        };
    }
}
