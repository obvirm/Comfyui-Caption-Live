// --- Helper: Parse Aspect Ratio ---
const getAspectRatio = (str) => {
    if (!str || str === "Custom") return null;
    const parts = str.split(":");
    if (parts.length === 2) {
        return parseFloat(parts[0]) / parseFloat(parts[1]);
    }
    return null;
};

export function setupLayout(nodeType) {
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

export function setupWidgetCallbacks(node, app) {
    // --- Auto-Resize on Ratio Change ---
    const ratioWidget = node.widgets?.find(w => w.name === "aspect_ratio");
    if (ratioWidget) {
        ratioWidget.callback = (value) => {
            // Trigger resize logic to snap to new ratio immediately
            if (node.onResize && node.size) {
                node.onResize(node.size);
            }
            app.graph.setDirtyCanvas(true, true);
        };
    }
}
